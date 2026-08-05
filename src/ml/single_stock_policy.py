"""단일 종목 일일 매수 + 인과적 관망(abstention) 정책 모듈.

`docs/specs/ml_single_stock_abstention.md` 계약 구현:

- 매 거래일마다 정확히 하나의 결정(BUY 종목 1개 또는 ABSTAIN)을 내립니다.
- ``rank_score`` / OOF ``pred`` 가 유일한 선택 신호이며, 유틸리티 등급·분위수·
  ``p_good``/``p_bad`` 는 진단 필드일 뿐 실행 게이트가 아닙니다.
- 마진 문턱과 정책 선택은 반드시 날짜 ``D`` 이전 OOF 날짜만 사용하는 인과적
  (causal) 계산입니다. 워밍업 기간은 명시적 ABSTAIN 으로 유지됩니다.
- exit ledger 가 없으므로 드로다운은 ``entry_sequence_drawdown`` 로만 명명하며
  포트폴리오 NAV/MDD 로 보고하지 않습니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pydantic

from src.ml.backtest_evaluator import (
    _aggregate_metrics,
    _extract_year,
    _group_starts,
    _max_drawdown,
    _yearly_breakdown,
    resolve_stock_actions,
)

logger = logging.getLogger(__name__)

_POLICY_VERSION = "ml-single-stock-v1"
_DEFAULT_QUANTILE_GRID: tuple[float, ...] = (0.70, 0.90)
_DEFAULT_MIN_HISTORY_DATES = 252
_MIN_YEAR_SAMPLES = 5

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


def default_policy_candidates(
    calibration_cutoff: str,
    *,
    grid: tuple[float, ...] = _DEFAULT_QUANTILE_GRID,
    score_col: str = "rank_score",
    version: str = _POLICY_VERSION,
) -> tuple[SingleStockPolicy, ...]:
    """버전화된 후보 정책 집합(always_buy + margin quantile grid)을 반환합니다."""
    always = always_buy_policy(calibration_cutoff, version=version, score_col=score_col)
    margins = tuple(
        margin_quantile_policy(q, calibration_cutoff, version=version, score_col=score_col)
        for q in grid
    )
    return (always, *margins)


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
    score_col: str = "rank_score",
    *,
    resolve_mode: str = "score_best_action",
    executable_col: str = "is_executable_action",
) -> pd.DataFrame:
    """스코어링된 일일 패널에서 단일 BUY/ABSTAIN 결정 레코드를 반환합니다.

    - 같은 날짜-종목의 시나리오 행동을 ``resolve_stock_actions`` 로 해소한 뒤
      종목을 ``rank_score`` 내림차순으로 정렬합니다(동점은 정규화 종목코드 오름차순).
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


def _validate_oof(
    oof_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    stock_col: str,
    scenario_col: str,
    score_col: str,
) -> None:
    required = [group_col, target_col, stock_col, scenario_col, score_col]
    missing = [col for col in required if col not in oof_df.columns]
    if missing:
        raise ValueError(f"missing required columns in oof_df: {missing}")
    null_cols = [col for col in required if oof_df[col].isna().any()]
    if null_cols:
        raise ValueError(f"required columns contain nulls: {null_cols}")
    parsed = pd.to_datetime(oof_df[group_col], errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError("group_col contains unparseable dates (chronology contract violation)")
    for col, label in ((score_col, "score"), (target_col, "target")):
        values = oof_df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {label} column {col!r}")


def _validate_candidates(
    policy_candidates: tuple[SingleStockPolicy, ...],
) -> tuple[SingleStockPolicy, ...]:
    if not policy_candidates:
        raise ValueError("policy_candidates must not be empty")
    seen: set[str] = set()
    for cand in policy_candidates:
        if not isinstance(cand, SingleStockPolicy):
            raise ValueError("policy_candidates must be SingleStockPolicy instances")
        if cand.candidate in seen:
            raise ValueError(f"duplicate policy candidate {cand.candidate!r}")
        seen.add(cand.candidate)
    return policy_candidates


def _causal_thresholds(
    margins: np.ndarray,
    q: float,
    min_history_dates: int,
) -> np.ndarray:
    """날짜별 마진 문턱을 이전 날짜(< D)의 마진만으로 계산합니다."""
    n = margins.size
    thresholds = np.full(n, np.nan)
    for i in range(min_history_dates, n):
        prior = margins[:i]
        prior = prior[np.isfinite(prior)]
        if prior.size == 0:
            continue
        thresholds[i] = float(np.quantile(prior, q))
    return thresholds


def _candidate_stats(scheduled: np.ndarray, buy: np.ndarray) -> dict[str, float]:
    agg = _aggregate_metrics(scheduled)
    active = scheduled[buy]
    return {
        "scheduled_mean_return": float(agg["top_1_return"]),
        "scheduled_sharpe": float(agg["sharpe"]),
        "scheduled_win_rate": float(agg["win_rate"]),
        "profit_factor": float(agg["profit_factor"]),
        "entry_sequence_drawdown": _max_drawdown(scheduled),
        "buy_rate": float(np.mean(buy)) if buy.size else float("nan"),
        "active_trade_mean_return": float(np.mean(active)) if active.size else float("nan"),
        "active_trade_win_rate": float(np.mean(active > 0.0)) if active.size else float("nan"),
    }


def _select_best_candidate(outcomes: dict[str, dict[str, Any]]) -> str:
    """결정적 목적함수 순서로 최적 후보를 선택합니다.

    스케줄 일자당 평균 수익(관망=0) 최대화 → 샤프 → 낮은 entry-sequence 드로다운 →
    높은 매수율 → 안정적인 정책 식별자.
    """
    rows = [
        {
            "policy_id": pid,
            "mean": item["stats"]["scheduled_mean_return"],
            "sharpe": item["stats"]["scheduled_sharpe"],
            "mdd": item["stats"]["entry_sequence_drawdown"],
            "buy_rate": item["stats"]["buy_rate"],
        }
        for pid, item in outcomes.items()
    ]
    frame = pd.DataFrame(rows)
    frame["mdd"] = frame["mdd"].fillna(0.0)
    frame["sharpe"] = frame["sharpe"].fillna(-np.inf)
    frame = frame.sort_values(
        ["mean", "sharpe", "mdd", "buy_rate", "policy_id"],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    )
    return str(frame.iloc[0]["policy_id"])


def _turnover_selected_codes(codes: np.ndarray) -> float:
    """매수일 종목코드 시퀀스의 전일 대비 변경률 평균(턴오버)."""
    if codes.size <= 1:
        return float("nan")
    changed = codes[1:] != codes[:-1]
    return float(np.mean(changed))


def _finalize_selected_policy(
    candidate: SingleStockPolicy,
    margins: _PanelMargins,
    *,
    n_dates: int,
    calibration_cutoff: str,
) -> SingleStockPolicy:
    """선택된 후보에 OOF 참조 분포/문턱을 채워 영속 가능한 정책을 만듭니다."""
    if candidate.candidate == _ALWAYS_BUY_CANDIDATE:
        return always_buy_policy(
            calibration_cutoff,
            version=candidate.version,
            score_col=candidate.score_col,
        )
    finite = margins.margin[np.isfinite(margins.margin)]
    q = _candidate_quantile(candidate.candidate)
    threshold: float | None = float(np.quantile(finite, q)) if finite.size else None
    return margin_quantile_policy(
        q,
        calibration_cutoff,
        version=candidate.version,
        score_col=candidate.score_col,
        margin_threshold=threshold,
        reference_margin=tuple(float(v) for v in finite),
        history_length=n_dates,
    )


def _build_evaluation_metrics(
    rows: pd.DataFrame,
    scheduled: np.ndarray,
    *,
    group_col: str,
    stock_col: str,
    market_type_col: str | None,
) -> dict[str, Any]:
    buy = rows["decision"].to_numpy() == "BUY"
    n = int(scheduled.size)
    agg = _aggregate_metrics(scheduled)
    active = scheduled[buy]
    active_mean = float(np.mean(active)) if active.size else float("nan")
    active_win = float(np.mean(active > 0.0)) if active.size else float("nan")

    reasons = rows["decision_reason"].to_numpy()
    reason_counts: dict[str, int] = {}
    for value in np.unique(reasons):
        reason_counts[str(value)] = int(np.sum(reasons == value))

    selected = rows[stock_col].to_numpy(dtype=object)[buy]
    return {
        "n_scheduled_dates": n,
        "n_buy": int(buy.sum()),
        "n_abstain": n - int(buy.sum()),
        "buy_rate": float(np.mean(buy)),
        "abstain_rate": float(np.mean(~buy)),
        "reason_counts": reason_counts,
        "scheduled_mean_return": float(agg["top_1_return"]),
        "scheduled_win_rate": float(agg["win_rate"]),
        "profit_factor": float(agg["profit_factor"]),
        "scheduled_sharpe": float(agg["sharpe"]),
        "active_trade_mean_return": active_mean,
        "active_trade_win_rate": active_win,
        "turnover": _turnover_selected_codes(selected),
        "entry_sequence_drawdown": _max_drawdown(scheduled),
    }


@dataclass(frozen=True)
class SingleStockPolicyEvaluation:
    """단일 종목 정책의 인과적 OOF 평가 결과."""

    selected_policy: SingleStockPolicy
    decisions: pd.DataFrame
    scheduled_returns: np.ndarray
    metrics: dict[str, Any]
    yearly_breakdown: dict[int, dict[str, float] | None]
    market_type_breakdown: dict[str, dict[str, float]]
    candidate_results: dict[str, dict[str, float]]


def evaluate_single_stock_policy_oof(
    oof_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    stock_col: str,
    policy_candidates: tuple[SingleStockPolicy, ...],
    min_history_dates: int,
    *,
    scenario_col: str = "chart_analysis",
    score_col: str = "rank_score",
) -> SingleStockPolicyEvaluation:
    """OOF 패널에서 단일 종목 정책을 인과적으로 보정·평가합니다.

    날짜 ``D`` 의 마진 문턱과 정책 선택은 오직 ``D`` 이전 날짜의 결과만
    사용합니다. 워밍업 기간은 ``insufficient_policy_history`` ABSTAIN 으로
    유지되며 삭제되지 않습니다. 모든 일자별 지표는 관망(ABSTAIN)을 0 으로
    계산하며, 드로다운은 ``entry_sequence_drawdown`` 로 명명합니다.
    """
    if min_history_dates < 1:
        raise ValueError(f"min_history_dates must be >= 1, got {min_history_dates}")
    _validate_candidates(policy_candidates)
    _validate_oof(oof_df, target_col, group_col, stock_col, scenario_col, score_col)

    resolved = resolve_stock_actions(
        oof_df,
        group_col,
        stock_col=stock_col,
        scenario_col=scenario_col,
        score_col=score_col,
        mode="score_best_action",
    )
    duplicates = resolved.duplicated(subset=[group_col, stock_col], keep=False)
    if duplicates.any():
        raise ValueError("resolved date-stock keys are not unique (data-contract violation)")

    panel = _build_panel(resolved, group_col, stock_col, scenario_col, score_col)
    margins = _compute_margins(
        panel,
        group_col,
        stock_col,
        scenario_col,
        score_col,
        target_col=target_col,
        market_type_col="market_type",
    )
    n = margins.sizes.size
    warm_up = np.arange(n) < min_history_dates

    candidate_outcomes: dict[str, dict[str, Any]] = {}
    for cand in policy_candidates:
        if cand.candidate == _ALWAYS_BUY_CANDIDATE:
            thresholds: np.ndarray | None = None
        else:
            thresholds = _causal_thresholds(
                margins.margin, _candidate_quantile(cand.candidate), min_history_dates
            )
        rows = _build_decision_rows(
            margins,
            cand,
            group_col=group_col,
            stock_col=stock_col,
            scenario_col=scenario_col,
            score_col=score_col,
            thresholds=thresholds,
            warm_up=warm_up,
        )
        buy = rows["decision"].to_numpy() == "BUY"
        assert margins.winner_target is not None
        scheduled = np.where(buy, margins.winner_target, 0.0).astype(np.float64)
        candidate_outcomes[cand.policy_id] = {
            "candidate": cand,
            "rows": rows,
            "scheduled": scheduled,
            "stats": _candidate_stats(scheduled, buy),
        }

    selected_id = _select_best_candidate(candidate_outcomes)
    selected = candidate_outcomes[selected_id]
    calibration_cutoff = str(panel[group_col].max())
    selected_policy = _finalize_selected_policy(
        selected["candidate"], margins, n_dates=n, calibration_cutoff=calibration_cutoff
    )

    rows_df = selected["rows"].copy()
    scheduled = selected["scheduled"].astype(np.float64)
    rows_df["scheduled_return"] = scheduled
    metrics = _build_evaluation_metrics(
        rows_df,
        scheduled,
        group_col=group_col,
        stock_col=stock_col,
        market_type_col="market_type" if "market_type" in rows_df.columns else None,
    )

    years = _extract_year(rows_df[group_col])
    yearly = _yearly_breakdown(scheduled, scheduled, years)

    market_breakdown: dict[str, dict[str, float]] = {}
    if "market_type" in rows_df.columns:
        buy = rows_df["decision"].to_numpy() == "BUY"
        market_values = rows_df["market_type"].to_numpy(dtype=object)
        for value in np.unique(market_values[buy]):
            mask = buy & (market_values == value)
            market_breakdown[str(value)] = _aggregate_metrics(scheduled[mask])

    return SingleStockPolicyEvaluation(
        selected_policy=selected_policy,
        decisions=rows_df,
        scheduled_returns=scheduled,
        metrics=metrics,
        yearly_breakdown=yearly,
        market_type_breakdown=market_breakdown,
        candidate_results={pid: item["stats"] for pid, item in candidate_outcomes.items()},
    )
