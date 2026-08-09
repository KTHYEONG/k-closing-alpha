"""Persisted single-stock policy schema and BUY/ABSTAIN selection for the live path.

This module retains only persisted-policy deserialization, BUY/ABSTAIN decision
selection, and runtime input validation. OOF policy calibration, candidate grid
construction, and ``evaluate_single_stock_policy_oof`` live under
``legacy/ml_research/evaluation/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pydantic

logger = logging.getLogger(__name__)

_POLICY_VERSION = "ml-single-stock-v1"

_ALWAYS_BUY_CANDIDATE = "always_buy_top1"
_MARGIN_CANDIDATE_PREFIX = "margin_quantile."

# 결정 사유: 실행 결과 불변식(BUY 1개 또는 ABSTAIN)을 설명하는 유일한 사유 집합.
REASON_TOP1_BUY = "top1_buy"
REASON_MARGIN_BUY = "top1_buy_margin"
REASON_NO_CANDIDATE = "no_executable_candidate"
REASON_INSUFFICIENT_HISTORY = "insufficient_policy_history"
REASON_INSUFFICIENT_CROSS_SECTION = "insufficient_cross_section"
REASON_BELOW_MARGIN = "below_margin_threshold"
REASON_MISSING_POLICY = "missing_validated_policy"


def _candidate_quantile(candidate: str) -> float:
    """``margin_quantile.<q>`` 후보에서 q 를 파싱합니다."""
    if candidate == _ALWAYS_BUY_CANDIDATE:
        raise ValueError("always_buy_top1 has no quantile")
    if not candidate.startswith(_MARGIN_CANDIDATE_PREFIX):
        raise ValueError(f"unknown policy candidate {candidate!r}")
    raw = candidate[len(_MARGIN_CANDIDATE_PREFIX) :]
    try:
        q = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid margin candidate {candidate!r}") from exc
    if not 0.0 < q < 1.0:
        raise ValueError(f"margin quantile must be in (0, 1), got {q}")
    return q


class SingleStockPolicy(pydantic.BaseModel):
    """불변 단일 종목 선택 정책 상태.

    - ``always_buy_top1``: 매일 최고 ``rank_score`` 종목을 매수.
    - ``margin_quantile.<q>``: 최고 종목을 정규화 스코어 마진이 q 분위수 이상일
      때만 매수하고, 그 외에는 관망(ABSTAIN).

    ``margin_threshold`` 와 ``reference_margin`` 은 OOF 보정(calibration)이
    채워 넣는 값으로, 새 거래일의 마진을 평가하는 데 필요한 참조 분포입니다.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    policy_id: str
    candidate: str
    version: str = _POLICY_VERSION
    calibration_cutoff: str
    candidate_grid: tuple[float, ...] = ()
    history_length: int = 0
    score_col: str = "rank_score"
    margin_threshold: float | None = None
    reference_margin: tuple[float, ...] = ()

    @pydantic.field_validator("candidate")
    @classmethod
    def _validate_candidate(cls, value: str) -> str:
        if value != _ALWAYS_BUY_CANDIDATE:
            _candidate_quantile(value)
        return value

    @pydantic.field_validator("margin_threshold")
    @classmethod
    def _validate_threshold(cls, value: float | None) -> float | None:
        if value is not None and not np.isfinite(value):
            raise ValueError("margin_threshold must be finite")
        return value


def always_buy_policy(
    calibration_cutoff: str,
    *,
    version: str = _POLICY_VERSION,
    score_col: str = "rank_score",
) -> SingleStockPolicy:
    """``always_buy_top1`` 정책 상태를 생성합니다."""
    return SingleStockPolicy(
        policy_id=_ALWAYS_BUY_CANDIDATE,
        candidate=_ALWAYS_BUY_CANDIDATE,
        version=version,
        calibration_cutoff=calibration_cutoff,
        score_col=score_col,
    )


def margin_quantile_policy(
    q: float,
    calibration_cutoff: str,
    *,
    version: str = _POLICY_VERSION,
    score_col: str = "rank_score",
    margin_threshold: float | None = None,
    reference_margin: tuple[float, ...] = (),
    history_length: int = 0,
) -> SingleStockPolicy:
    """``margin_quantile.<q>`` 정책 상태를 생성합니다."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"margin quantile must be in (0, 1), got {q}")
    candidate = f"{_MARGIN_CANDIDATE_PREFIX}{q:.2f}"
    return SingleStockPolicy(
        policy_id=candidate,
        candidate=candidate,
        version=version,
        calibration_cutoff=calibration_cutoff,
        candidate_grid=(q,),
        history_length=history_length,
        score_col=score_col,
        margin_threshold=margin_threshold,
        reference_margin=reference_margin,
    )


def load_single_stock_policy(models_bundle: dict[str, Any]) -> SingleStockPolicy | None:
    """번들에 영속화된 ``SingleStockPolicy`` 상태를 복원합니다.

    유효한 정책 상태가 없으면 ``None`` 을 반환하며, 호출부가 조용한 Top-N 폴백
    대신 명시적 ``ABSTAIN``(``missing_validated_policy``)을 산출하게 합니다.
    """
    raw = models_bundle.get("single_stock_policy")
    if raw is None:
        return None
    if isinstance(raw, SingleStockPolicy):
        return raw
    if isinstance(raw, dict):
        return SingleStockPolicy(**raw)
    logger.info(
        "인식할 수 없는 single_stock_policy 상태입니다. "
        "ABSTAIN(missing_validated_policy) 으로 결정합니다."
    )
    return None


def _group_starts(group_vals: np.ndarray) -> np.ndarray:
    """그룹화된(정렬된) 배열에서 각 그룹 시작 인덱스를 반환합니다."""
    change = np.concatenate(([True], group_vals[1:] != group_vals[:-1]))
    return np.flatnonzero(change)


_ACTION_RESOLUTION_MODES: tuple[str, ...] = (
    "exclude_multi_scenario",
    "score_best_action",
    "require_final_action",
)


def resolve_stock_actions(
    oof_df: pd.DataFrame,
    group_col: str,
    stock_col: str = "stock_code",
    scenario_col: str = "chart_analysis",
    score_col: str = "pred",
    mode: str = "exclude_multi_scenario",
    executable_col: str = "is_executable_action",
) -> pd.DataFrame:
    """스코어링된 행동 패널을 유일한 ``(group, stock)`` 종목 패널로 해소합니다.

    시나리오 행동 패널에서 날짜-종목당 실행 가능한 행동 하나를 선택합니다.
    실현 수익률(``target_col``)이나 원천 행 순서로 행동을 선택하지 않습니다.

    모드:
    - ``exclude_multi_scenario``: 날짜-종목에 행동이 둘 이상이면 해당 종목을
      패널에서 제외합니다.
    - ``score_best_action``: 행동별 예측 점수(``score_col``)가 가장 높은 하나를
      선택하며, 동점은 ``scenario_col`` 오름차순으로 결정합니다.
    - ``require_final_action``: ``executable_col`` 이 True 인 행동이 정확히 하나인
      날짜-종목만 선택하며, 없거나 둘 이상이면 ``ValueError`` 입니다.
    """
    if mode not in _ACTION_RESOLUTION_MODES:
        raise ValueError(f"mode must be one of {list(_ACTION_RESOLUTION_MODES)}, got {mode!r}")
    required = [group_col, stock_col, scenario_col, score_col]
    missing = [col for col in required if col not in oof_df.columns]
    if missing:
        raise ValueError(f"missing required columns in oof_df for action resolution: {missing}")
    null_cols = [col for col in required if oof_df[col].isna().any()]
    if null_cols:
        raise ValueError(f"required columns contain nulls: {null_cols}")

    work = oof_df.reset_index(drop=True).copy()
    group_keys = [group_col, stock_col]

    if mode == "exclude_multi_scenario":
        counts = work.groupby(group_keys, sort=False)[scenario_col].transform("size")
        resolved = work.loc[counts == 1]
    elif mode == "score_best_action":
        resolved = (
            work.sort_values(
                [*group_keys, score_col, scenario_col],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .groupby(group_keys, sort=False)
            .head(1)
        )
    else:  # require_final_action
        if executable_col not in work.columns:
            raise ValueError(
                f"executable_col {executable_col!r} is required for require_final_action mode"
            )
        executable = work[executable_col].fillna(False).astype(bool)
        group_sizes = work.groupby(group_keys, sort=False).size()
        executable_counts = (
            work.assign(_executable=executable)
            .groupby(group_keys, sort=False)["_executable"]
            .sum()
            .reindex(group_sizes.index, fill_value=0)
        )
        invalid = executable_counts[executable_counts != 1]
        if not invalid.empty:
            raise ValueError(
                "require_final_action expects exactly one executable action per "
                f"{group_col}-{stock_col} group; got {len(invalid)} invalid groups"
            )
        resolved = work.loc[executable]

    return resolved.reset_index(drop=True)


def _validate_scored_df(
    scored_df: pd.DataFrame,
    group_col: str,
    stock_col: str,
    scenario_col: str,
    score_col: str,
) -> None:
    required = [group_col, stock_col, scenario_col, score_col]
    missing = [col for col in required if col not in scored_df.columns]
    if missing:
        raise ValueError(f"missing required identity/score columns in scored_df: {missing}")
    null_cols = [col for col in required if scored_df[col].isna().any()]
    if null_cols:
        raise ValueError(f"required columns contain nulls: {null_cols}")
    scores = scored_df[score_col].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite values in score column {score_col!r}")


def _build_panel(
    resolved: pd.DataFrame,
    group_col: str,
    stock_col: str,
    scenario_col: str,
    score_col: str,
) -> pd.DataFrame:
    """정규화 종목코드로 타이브레이크된 결정적 패널을 생성합니다."""
    work = resolved.copy()
    work["_stock_sort_key"] = work[stock_col].astype(str)
    return work.sort_values(
        [group_col, score_col, "_stock_sort_key"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


@dataclass(frozen=True)
class _PanelMargins:
    """일자별 정렬 패널의 벡터화된 순위/마진 진단 정보."""

    group_values: np.ndarray
    sizes: np.ndarray
    best: np.ndarray
    second: np.ndarray
    margin: np.ndarray
    winner_stock: np.ndarray
    winner_scenario: np.ndarray
    winner_target: np.ndarray | None
    winner_market_type: np.ndarray | None


def _compute_margins(
    panel: pd.DataFrame,
    group_col: str,
    stock_col: str,
    scenario_col: str,
    score_col: str,
    target_col: str | None = None,
    market_type_col: str | None = None,
) -> _PanelMargins:
    """일자별 best/second/std/margin 과 Top-1 종목 정보를 NumPy 벡터로 계산합니다.

    마진 = ``(best - second) / cross_sectional_std(score)``. 일자 내 종목이 1개이거나
    횡단면 분산이 0이면 마진은 NaN(교차섹션 부재)입니다.
    """
    scores = panel[score_col].to_numpy(dtype=np.float64)
    group_vals = panel[group_col].to_numpy()
    starts = _group_starts(group_vals)
    sizes = np.diff(np.concatenate((starts, np.array([group_vals.size])))).astype(np.int64)

    best = scores[starts]
    second = np.full(starts.size, np.nan)
    second[sizes >= 2] = scores[starts[sizes >= 2] + 1]

    sums = np.add.reduceat(scores, starts)
    sumsq = np.add.reduceat(scores * scores, starts)
    means = sums / sizes
    variance = np.maximum(sumsq / sizes - means * means, 0.0)
    std = np.sqrt(variance)

    margin = np.full(starts.size, np.nan)
    valid = (sizes >= 2) & (std > 0.0)
    margin[valid] = (best[valid] - second[valid]) / std[valid]

    winner_target: np.ndarray | None = None
    if target_col is not None and target_col in panel.columns:
        winner_target = panel[target_col].to_numpy(dtype=np.float64)[starts]
    winner_market_type: np.ndarray | None = None
    if market_type_col is not None and market_type_col in panel.columns:
        winner_market_type = panel[market_type_col].to_numpy()[starts]

    return _PanelMargins(
        group_values=group_vals[starts],
        sizes=sizes,
        best=best,
        second=second,
        margin=margin,
        winner_stock=panel[stock_col].to_numpy()[starts],
        winner_scenario=panel[scenario_col].to_numpy()[starts],
        winner_target=winner_target,
        winner_market_type=winner_market_type,
    )


def _build_decision_rows(
    margins: _PanelMargins,
    policy: SingleStockPolicy,
    *,
    group_col: str,
    stock_col: str,
    scenario_col: str,
    score_col: str,
    thresholds: np.ndarray | None,
    warm_up: np.ndarray,
) -> pd.DataFrame:
    """일자별 결정 레코드(BUY/ABSTAIN + 진단)를 한 번에 구성합니다."""
    n = margins.sizes.size
    if thresholds is None:
        thresholds = np.full(n, np.nan)
    is_margin = policy.candidate != _ALWAYS_BUY_CANDIDATE

    rows: dict[str, list[Any]] = {
        group_col: [],
        stock_col: [],
        scenario_col: [],
        score_col: [],
        "second_score": [],
        "margin": [],
        "decision": [],
        "decision_reason": [],
        "policy_id": [],
        "candidate": [],
        "version": [],
        "calibration_cutoff": [],
        "n_candidates": [],
        "n_unique_stocks": [],
    }
    if margins.winner_market_type is not None:
        rows["market_type"] = []

    for i in range(n):
        has_candidate = margins.sizes[i] > 0
        if has_candidate:
            rows[group_col].append(margins.group_values[i])
            rows[stock_col].append(margins.winner_stock[i])
            rows[scenario_col].append(margins.winner_scenario[i])
            rows[score_col].append(float(margins.best[i]))
            rows["second_score"].append(float(margins.second[i]))
            rows["margin"].append(float(margins.margin[i]))
        else:
            rows[group_col].append(margins.group_values[i])
            rows[stock_col].append(None)
            rows[scenario_col].append(None)
            rows[score_col].append(None)
            rows["second_score"].append(None)
            rows["margin"].append(None)

        if not has_candidate:
            decision, reason = "ABSTAIN", REASON_NO_CANDIDATE
        elif warm_up[i]:
            decision, reason = "ABSTAIN", REASON_INSUFFICIENT_HISTORY
        elif not is_margin:
            decision, reason = "BUY", REASON_TOP1_BUY
        elif margins.sizes[i] < 2:
            decision, reason = "ABSTAIN", REASON_INSUFFICIENT_CROSS_SECTION
        elif not np.isfinite(thresholds[i]):
            decision, reason = "ABSTAIN", REASON_INSUFFICIENT_HISTORY
        elif margins.margin[i] >= thresholds[i]:
            decision, reason = "BUY", REASON_MARGIN_BUY
        else:
            decision, reason = "ABSTAIN", REASON_BELOW_MARGIN

        rows["decision"].append(decision)
        rows["decision_reason"].append(reason)
        rows["policy_id"].append(policy.policy_id)
        rows["candidate"].append(policy.candidate)
        rows["version"].append(policy.version)
        rows["calibration_cutoff"].append(policy.calibration_cutoff)
        rows["n_candidates"].append(int(margins.sizes[i]))
        rows["n_unique_stocks"].append(int(margins.sizes[i]))
        if margins.winner_market_type is not None:
            rows["market_type"].append(margins.winner_market_type[i] if has_candidate else None)

    return pd.DataFrame(rows)


def _abstain_only_record(
    reason: str,
    *,
    group_col: str,
    stock_col: str,
    scenario_col: str,
    score_col: str,
    policy: SingleStockPolicy | None,
    group_value: Any = None,
) -> pd.DataFrame:
    """후보가 전혀 없는 패널에 대한 단일 ABSTAIN 레코드를 반환합니다."""
    return pd.DataFrame(
        {
            group_col: [group_value],
            stock_col: [None],
            scenario_col: [None],
            score_col: [None],
            "second_score": [None],
            "margin": [None],
            "decision": ["ABSTAIN"],
            "decision_reason": [reason],
            "policy_id": [policy.policy_id if policy is not None else reason],
            "candidate": [policy.candidate if policy is not None else reason],
            "version": [policy.version if policy is not None else ""],
            "calibration_cutoff": [policy.calibration_cutoff if policy is not None else ""],
            "n_candidates": [0],
            "n_unique_stocks": [0],
        }
    )


def abstain_decision(
    reason: str = REASON_MISSING_POLICY,
    policy: SingleStockPolicy | None = None,
    *,
    group_col: str = "date",
    stock_col: str = "stock_code",
    scenario_col: str = "chart_analysis",
    score_col: str = "rank_score",
    group_value: Any = None,
) -> pd.DataFrame:
    """호출부에서 명시적 ABSTAIN 레코드를 구성할 때 사용합니다.

    실행 경로가 정책 상태 없이 결정해야 하는 경우(missing_validated_policy)나
    후보가 없는 경우(no_executable_candidate)에 사용됩니다.
    """
    return _abstain_only_record(
        reason,
        group_col=group_col,
        stock_col=stock_col,
        scenario_col=scenario_col,
        score_col=score_col,
        policy=policy,
        group_value=group_value,
    )


def select_single_daily_trade(
    scored_df: pd.DataFrame,
    policy: SingleStockPolicy,
    group_col: str,
    stock_col: str = "stock_code",
    scenario_col: str = "chart_analysis",
    score_col: str | None = None,
    *,
    resolve_mode: str = "score_best_action",
    executable_col: str = "is_executable_action",
) -> pd.DataFrame:
    """스코어링된 일일 패널에서 단일 BUY/ABSTAIN 결정 레코드를 반환합니다.

    - 같은 날짜-종목의 시나리오 행동을 ``resolve_stock_actions`` 로 해소한 뒤
      종목을 ``score_col`` 내림차순으로 정렬합니다(동점은 정규화 종목코드 오름차순).
    - 후보가 전혀 없으면 ``no_executable_candidate`` ABSTAIN, 데이터 계약 위반은
      ``ValueError`` 입니다(잘못된 정책 상태 포함).
    """
    if not isinstance(policy, SingleStockPolicy):
        raise ValueError("policy must be a SingleStockPolicy (invalid policy state)")
    if policy.candidate != _ALWAYS_BUY_CANDIDATE:
        _candidate_quantile(policy.candidate)
        if policy.margin_threshold is None:
            raise ValueError(
                "invalid policy state: margin_quantile policy requires margin_threshold"
            )

    if score_col is None:
        score_col = policy.score_col

    if scored_df is None or len(scored_df) == 0:
        return _abstain_only_record(
            REASON_NO_CANDIDATE,
            group_col=group_col,
            stock_col=stock_col,
            scenario_col=scenario_col,
            score_col=score_col,
            policy=policy,
        )

    _validate_scored_df(scored_df, group_col, stock_col, scenario_col, score_col)
    resolved = resolve_stock_actions(
        scored_df,
        group_col,
        stock_col=stock_col,
        scenario_col=scenario_col,
        score_col=score_col,
        mode=resolve_mode,
        executable_col=executable_col,
    )
    duplicates = resolved.duplicated(subset=[group_col, stock_col], keep=False)
    if duplicates.any():
        raise ValueError("resolved date-stock keys are not unique (data-contract violation)")

    panel = _build_panel(resolved, group_col, stock_col, scenario_col, score_col)
    margins = _compute_margins(
        panel, group_col, stock_col, scenario_col, score_col
    )
    n = margins.sizes.size
    if policy.candidate == _ALWAYS_BUY_CANDIDATE:
        thresholds: np.ndarray | None = None
    else:
        assert policy.margin_threshold is not None
        thresholds = np.full(n, policy.margin_threshold)
    warm_up = np.zeros(n, dtype=bool)
    return _build_decision_rows(
        margins,
        policy,
        group_col=group_col,
        stock_col=stock_col,
        scenario_col=scenario_col,
        score_col=score_col,
        thresholds=thresholds,
        warm_up=warm_up,
    )
