"""Fold-local feature screening/selection using train-only statistics.

`docs/specs/ml_feature_selection_pipeline.md` 의 fold-local 선택 계약을
구현합니다. 각 outer fold 의 train 분할에서만 계산하므로 validation/이후 날짜
라벨은 선택에 절대 노출되지 않습니다.

``selection_rule`` 에 따라 두 가지 흐름이 있습니다 (train 구간 전용, 결정적):

``fixed_cap`` (legacy, API 기본):
1. 품질 거부: 전부 비유한(all-nonfinite) 컬럼, train 결측률 > 0.35, 영분산 컬럼을
   이유와 함께 거부합니다.
2. 결정적 CPU LightGBM(Huber) 스크리닝 모델로 gain 중요도를 계산합니다 (CPU fit 은
   ``deterministic=True`` + ``force_col_wise=True`` 이므로 스레드 수와 무관하게
   재현됩니다).
3. ``gain <= min_gain`` 인 영-gain 컬럼을 ``zero_gain`` 으로 거부합니다.
4. train 행만 사용한 exact pairwise-complete Spearman 상관이 ``correlation_threshold``
   를 넘는 쌍을 (gain desc, 이름 asc) 결정적 tie-break 으로 가지치기합니다.
5. 양의 gain 컬럼을 (gain desc, 이름 asc) 로 정렬해 최대 ``max_retained`` 개를
   유지하며, 초과분은 ``beyond_max_retained`` 로 거부합니다.
6. 유지 수가 ``min_retained`` 미만(또는 ``hard_max_retained`` 초과)이면
   ``ValueError`` 로 fail-closed 하며, zero-importance 컬럼으로 패딩하지 않습니다.

``permutation_fwer`` (신규 리서치 기본): 3 번 대신 아래의 null 임계값을 사용합니다.
1. 품질 거부 -> 2. 실 target 으로 실제 gain 계산.
3. ``null_permutations`` 회, 같은 ``group_col`` 값(거래일) 안에서만 target 을 섞어
   동일 스크리닝을 재적합하고 각 순열의 최대 gain 을 수집합니다. 날짜-로컬 순열은
   일별 target 분포를 보존하면서 피처-수익 연관만 끊으므로 결정 시점 간 정보 이동이
   없습니다.
4. 임계값 ``T`` = 순열 최대 gain 의 empirical ``(1 - null_alpha)`` 분위수. 실제
   gain 이 ``T`` 를 엄격히 초과하는 컬럼만 유지하며, 이하 컬럼은
   ``not_significant_vs_null`` 로 거부합니다 (FWER ``alpha`` 제어).
5. 유지 목록에 exact pairwise-complete Spearman 가지치기를 적용합니다. ``max_retained``
   / ``min_retained`` 는 적용하지 않으며, 유지 수가
   ``[min_significant_features, hard_max_retained]`` 를 벗어나면 ``ValueError`` 로
   fail-closed 합니다 (조용한 truncation 없음).

Vectorized 전용이며 ``pd.apply`` 를 사용하지 않습니다. SHAP 은 선택기가 아니라
옵션 사후 설명 전용입니다.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pydantic
from lightgbm import LGBMRegressor

from src.ml.purged_cv import PurgedGroupTimeSeriesSplit

FEATURE_SELECTION_VERSION = "fold_local_gain_v1"

_REASON_ALL_NONFINITE = "all_nonfinite"
_REASON_MISSING_COLUMN = "missing_column"
_REASON_TRAIN_MISSING_RATE = "train_missing_rate"
_REASON_ZERO_VARIANCE = "zero_variance"
_REASON_ZERO_GAIN = "zero_gain"
_REASON_CORRELATED = "correlated_pair_pruned"
_REASON_BEYOND_CAP = "beyond_max_retained"
_REASON_NOT_SIGNIFICANT = "not_significant_vs_null"

_KNOWN_REJECTION_REASONS: frozenset[str] = frozenset(
    {
        _REASON_ALL_NONFINITE,
        _REASON_MISSING_COLUMN,
        _REASON_TRAIN_MISSING_RATE,
        _REASON_ZERO_VARIANCE,
        _REASON_ZERO_GAIN,
        _REASON_CORRELATED,
        _REASON_BEYOND_CAP,
        _REASON_NOT_SIGNIFICANT,
    }
)

FEATURE_QUALITY_VERSION = "feature_quality_v2"
_BASELINE_FAMILY = "baseline"
_QUALITY_ACTION_NAMES = (
    "source_incomplete",
    "capacity_limited",
    "redundant",
    "unstable",
    "screening_weak",
)


class FeatureSelectionConfig(pydantic.BaseModel):
    """Fold-local 선택 설정 (불변).

    기본값은 리서치 계약(300--400--500, 결측률 0.35, 상관 0.98)이며, 테스트는
    작은 후보 집합으로 검증할 수 있도록 축소 값을 허용합니다.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    min_retained: int = 300
    max_retained: int = 400
    hard_max_retained: int = 500
    missing_rate_threshold: float = 0.35
    correlation_threshold: float = 0.98
    random_seed: int = 42
    min_gain: float = 0.0
    catalogue_version: str = "unversioned"
    catalogue: Mapping[str, Mapping[str, str]] = {}
    # A feature must be selected in every fold to be called stable. With two
    # folds, a 0.5 threshold labels a one-fold feature as stable by definition.
    min_fold_selection_rate: float = 1.0
    screening_device: Literal["cpu", "gpu", "auto"] = "cpu"
    n_jobs: int = -1
    selection_rule: Literal["fixed_cap", "permutation_fwer"] = "fixed_cap"
    null_alpha: float = 0.05
    null_permutations: int = 19
    min_significant_features: int = 1
    min_pairwise_observations: int = 3

    @pydantic.model_validator(mode="after")
    def _validate_ranges(self) -> FeatureSelectionConfig:
        if not (1 <= self.min_retained <= self.max_retained <= self.hard_max_retained):
            raise ValueError(
                f"selection bounds must satisfy 1 <= min_retained <= max_retained "
                f"<= hard_max_retained, got ({self.min_retained}, {self.max_retained}, "
                f"{self.hard_max_retained})"
            )
        if not 0.0 <= self.missing_rate_threshold <= 1.0:
            raise ValueError(
                f"missing_rate_threshold must be in [0, 1], got {self.missing_rate_threshold}"
            )
        if not 0.0 <= self.correlation_threshold <= 1.0:
            raise ValueError(
                f"correlation_threshold must be in [0, 1], got {self.correlation_threshold}"
            )
        if self.min_gain < 0.0:
            raise ValueError(f"min_gain must be >= 0, got {self.min_gain}")
        if not 0.0 < self.min_fold_selection_rate <= 1.0:
            raise ValueError(
                f"min_fold_selection_rate must be in (0, 1], "
                f"got {self.min_fold_selection_rate}"
            )
        if self.n_jobs != -1 and self.n_jobs < 1:
            raise ValueError(
                f"n_jobs must be -1 (all cores) or >= 1, got {self.n_jobs}"
            )
        if not 0.0 < self.null_alpha < 1.0:
            raise ValueError(f"null_alpha must be in (0, 1), got {self.null_alpha}")
        min_null_permutations = math.ceil(1.0 / self.null_alpha) - 1
        if self.null_permutations < min_null_permutations:
            raise ValueError(
                f"null_permutations must be >= ceil(1/null_alpha) - 1 = "
                f"{min_null_permutations}, got {self.null_permutations}"
            )
        if self.min_significant_features < 1:
            raise ValueError(
                f"min_significant_features must be >= 1, got {self.min_significant_features}"
            )
        if self.min_pairwise_observations < 3:
            raise ValueError(
                f"min_pairwise_observations must be >= 3, got {self.min_pairwise_observations}"
            )
        if self.selection_rule == "permutation_fwer" and self.screening_device != "cpu":
            raise ValueError(
                "selection_rule='permutation_fwer' requires screening_device='cpu'"
            )
        return self


class FeatureSelectionResult(pydantic.BaseModel):
    """선택 결과 요약 (불변).

    - ``selected_features``: 최종 유지 피처 (gain desc, 이름 asc) 순서.
    - ``gains``: 양의 gain 을 가진 후보의 (feature, gain) 쌍, (gain desc, 이름 asc).
    - ``rejected``: (feature, reason) 쌍 (품질/영-gain/상관/초과 사유 포함).
    - ``counts``: n_candidates, n_rejected, n_positive_gain, n_pre_correlation,
      n_post_correlation, n_retained 등.
    - ``metadata``: data_cutoff, n_train_rows, n_groups, random_seed 등.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    selected_features: tuple[str, ...]
    gains: tuple[tuple[str, float], ...]
    rejected: tuple[tuple[str, str], ...]
    counts: dict[str, int]
    metadata: dict[str, Any]
    quality_evidence: Mapping[str, Mapping[str, float | bool]] = {}


def config_fingerprint(config: FeatureSelectionConfig) -> str:
    """버전 + 설정 JSON 의 결정적 지문 (sha256) 을 반환합니다.

    캐시 키·순열 null 시드·진단 재현에 사용됩니다. ``catalogue`` 매핑을 포함한
    설정 전부를 정렬된 JSON 으로 직렬화해, 설정이 조금이라도 바뀌면 지문이
    바뀌도록 합니다.
    """
    payload = json.dumps(
        {"version": FEATURE_SELECTION_VERSION, **config.model_dump()},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FoldFeaturePlan:
    """외부 fold 하나의 불변 피처 계획.

    같은 fold 의 expanding/recent return 전문가가 동일한 ``selected_features``
    를 공유해야 하므로, 선택 결과와 그 근거(거부 사유·gain·카운트·설정 지문·
    시드)를 불변 값으로 캡슐화합니다. ``config`` 는 OOF 평가 후 research 번들
    용 full-data ``full_training_selection`` 을 재현할 때만 사용됩니다.
    """

    fold: int
    data_cutoff: str
    selected_features: tuple[str, ...]
    gains: tuple[tuple[str, float], ...]
    rejected: tuple[tuple[str, str], ...]
    counts: Mapping[str, int]
    metadata: Mapping[str, Any]
    config_fingerprint: str
    seed: int
    config: FeatureSelectionConfig
    selection: FeatureSelectionResult | None = None


def build_fold_feature_plans(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    config: FeatureSelectionConfig,
    n_splits: int = 5,
    purge_gap: int = 1,
) -> list[FoldFeaturePlan]:
    """외부 walk-forward fold 별 train 전용 피처 계획을 1회 구성합니다.

    각 계획은 그 fold 의 train 분할에서만 ``select_features`` 로 선택되므로
    validation/이후 라벨에 불변입니다. 두 return 전문가(expanding, recent) 가
    같은 계획을 공유하며, ``_prepare_history_frame`` 계약의 성립 대신 이 함수
    자체가 fold 단위 train-only 불변량을 보장합니다.
    """
    work = df.sort_values(group_col).copy()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    fingerprint = config_fingerprint(config)
    plans: list[FoldFeaturePlan] = []
    for fold, (train_idx, _val_idx) in enumerate(
        splitter.split(work, y=work[target_col], groups=work[group_col])
    ):
        train = work.iloc[train_idx]
        result = select_features(
            train,
            feature_cols,
            target_col,
            config,
            group_col=group_col,
            provenance="fold_train_selection",
        )
        plans.append(
            FoldFeaturePlan(
                fold=fold,
                data_cutoff=str(train[group_col].max()),
                selected_features=result.selected_features,
                gains=result.gains,
                rejected=result.rejected,
                counts=result.counts,
                metadata=result.metadata,
                config_fingerprint=fingerprint,
                seed=config.random_seed,
                config=config,
                selection=result,
            )
        )
    return plans


def permutation_null_stability(
    feature_lists: Sequence[Sequence[str]],
    *,
    random_seed: int,
    n_permutations: int = 199,
    null_gate_quantile: float = 0.95,
    universe: Sequence[str] | None = None,
) -> dict[str, Any]:
    """관측 fold 안정성을 날짜 블록 순열 null 분포와 비교합니다.

    fold 는 서로 다른 날짜 블록에서 나왔으므로, 각 fold 의 선택 크기를 보존한 채
    후보 유니온 풀에서 무작위 재추출한 순열 목록의 median pairwise Jaccard 로
    null 을 구성합니다 (결정적 시드). ``universe`` 는 후보 전체 풀 (선택기가
    선택했을 수 있는 모든 컬럼) 이며 기본은 선택 feature 의 합집합입니다.
    ``null_gate_quantile`` 분위를 관측값이 초과하지 못하면 게이트 미통과(불안정)
    로 간주해 약한 피처로 패딩하는 대신 후보를 거부합니다.
    """
    if not 0.0 <= null_gate_quantile <= 1.0:
        raise ValueError(f"null_gate_quantile must be in [0, 1], got {null_gate_quantile}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    lists = [list(features) for features in feature_lists]
    observed = median_pairwise_jaccard(feature_lists)
    if len(lists) < 2:
        return {
            "observed_median_jaccard": float(observed),
            "null_mean": None,
            "null_std": None,
            "null_gate_value": None,
            "null_gate_quantile": null_gate_quantile,
            "n_permutations": 0,
            "gate_passed": True,
            "reason": "fewer_than_two_folds_abstain",
        }
    if universe is not None:
        pool = sorted(set(universe))
    else:
        pool = sorted({feature for features in lists for feature in features})
    sizes = [len(features) for features in lists]
    if any(size > len(pool) for size in sizes):
        raise ValueError("fold feature list exceeds the candidate universe size")
    rng = np.random.default_rng(random_seed)
    null_scores: list[float] = []
    for _ in range(n_permutations):
        permuted = [
            rng.choice(pool, size=size, replace=False).tolist() for size in sizes
        ]
        null_scores.append(median_pairwise_jaccard(permuted))
    null_arr = np.asarray(null_scores, dtype=np.float64)
    gate_value = float(np.quantile(null_arr, null_gate_quantile))
    return {
        "observed_median_jaccard": float(observed),
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std()),
        "null_gate_value": gate_value,
        "null_gate_quantile": null_gate_quantile,
        "n_permutations": n_permutations,
        "gate_passed": bool(observed > gate_value),
        "reason": None,
    }


def _reject_quality(
    train: pd.DataFrame,
    feature_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], dict[str, str], dict[str, dict[str, float | bool]]]:
    """품질 기준으로 후보를 거부하고 생존자 목록을 반환합니다 (행렬 단위 검사).

    컬럼별 Python 판다스 연산 대신 결측/비유한/분산 판정을 2D 행렬로 벡터화해
    단일 패스로 계산합니다. 거부 우선순위는 기존 계약과 동일합니다:
    missing column -> all nonfinite -> missing rate -> zero variance.

    추가로 후보별 품질 증거(``finite_ratio``, ``missing_rate``, ``zero_variance``)
    를 함께 반환해 ``FeatureSelectionResult`` 에 보관하고, 품질 리포트가 두 번째
    데이터 스캔 없이 폴드별 가용성을 집계하도록 합니다.
    """
    rejected: dict[str, str] = {}
    present = [col for col in feature_cols if col in train.columns]
    for col in feature_cols:
        if col not in train.columns:
            rejected[col] = _REASON_MISSING_COLUMN
    if not present:
        return [], rejected, {}

    mat = np.column_stack([train[col].to_numpy(dtype=np.float64) for col in present])
    finite = np.isfinite(mat)
    finite_any = finite.any(axis=0)
    finite_ratio = finite.mean(axis=0)
    is_nan = np.isnan(mat)
    missing_rate = is_nan.mean(axis=0)
    min_val = np.where(finite, mat, np.inf).min(axis=0)
    max_val = np.where(finite, mat, -np.inf).max(axis=0)
    zero_variance = ~is_nan.any(axis=0) & (min_val == max_val)

    evidence: dict[str, dict[str, float | bool]] = {}
    for idx, col in enumerate(present):
        evidence[col] = {
            "finite_ratio": float(finite_ratio[idx]),
            "missing_rate": float(missing_rate[idx]),
            "zero_variance": bool(zero_variance[idx]),
        }
        if not finite_any[idx]:
            rejected[col] = _REASON_ALL_NONFINITE
        elif missing_rate[idx] > config.missing_rate_threshold:
            rejected[col] = _REASON_TRAIN_MISSING_RATE
        elif zero_variance[idx]:
            rejected[col] = _REASON_ZERO_VARIANCE
    survivors = [col for col in present if col not in rejected]
    return survivors, rejected, evidence


def _prune_correlated(
    train: pd.DataFrame,
    ordered_names: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], dict[str, str]]:
    """train 행만 사용한 exact pairwise-complete Spearman 가지치기 (BLAS 행렬곱).

    각 컬럼을 유한 관측값으로 rank 하고, 쌍 ``(i, j)`` 의 상관은 두 컬럼이 모두
    존재하는 행만 사용해 계산합니다. 쌍별 개수·랭크합·랭크제곱합·교차곱 5 개
    행렬을 ``mask``/``rank`` 행렬의 행렬곱으로 한 번에 유도하므로 ``pd.apply`` 나
    컬럼 순회가 없습니다. 공통 관측 수가 ``min_pairwise_observations`` 미만이거나
    쌍 분산이 0 이면 상관은 0.0 으로 정의해 해당 쌍은 가지치기에서 생존합니다
    (결측을 0 으로 채운 표준화 랭크로 전체 행 수로 나누는 기존 방식의 확장 오류
    제거). 유지/거부 판정은 결정적 greedy 스캔((gain desc, 이름 asc) 순서)으로
    절대 상관이 ``correlation_threshold`` 를 엄격히 초과할 때만 거부합니다.
    """
    if len(ordered_names) < 2:
        return ordered_names, {}
    ranked = train[ordered_names].rank(method="average")
    r = ranked.to_numpy(dtype=np.float64)
    present = np.isfinite(r)
    mask = present.astype(np.float64)
    r0 = np.where(present, r, 0.0)

    pair_count = mask.T @ mask
    rank_sum_i = r0.T @ mask
    rank_sum_j = mask.T @ r0
    rank_sq_i = (r0 * r0).T @ mask
    rank_sq_j = mask.T @ (r0 * r0)
    cross = r0.T @ r0

    n = pair_count
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_i = rank_sum_i / n
        mean_j = rank_sum_j / n
        var_i = rank_sq_i / n - mean_i * mean_i
        var_j = rank_sq_j / n - mean_j * mean_j
        cov = cross / n - mean_i * mean_j
        denom = np.sqrt(var_i * var_j)
        corr = np.where(denom > 0.0, cov / np.where(denom > 0.0, denom, np.nan), 0.0)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    corr[n < config.min_pairwise_observations] = 0.0
    corr[var_i <= 0.0] = 0.0
    corr[var_j <= 0.0] = 0.0
    np.fill_diagonal(corr, 1.0)

    kept_indices: list[int] = []
    rejected: dict[str, str] = {}
    for idx, name in enumerate(ordered_names):
        if (
            kept_indices
            and np.abs(corr[idx, kept_indices]).max() > config.correlation_threshold
        ):
            rejected[name] = _REASON_CORRELATED
        else:
            kept_indices.append(idx)
    kept = [ordered_names[i] for i in kept_indices]
    return kept, rejected


_SCREENING_DEVICE_RESULTS: dict[str, tuple[str, str | None]] = {}


def _probe_gpu_device() -> tuple[str, str | None]:
    """아주 작은 LightGBM GPU fit 을 1회 수행해 GPU 사용 가능 여부를 확인합니다."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    target = frame["a"] + rng.normal(scale=0.1, size=40)
    try:
        model = LGBMRegressor(
            objective="huber",
            random_state=0,
            n_estimators=2,
            num_leaves=4,
            device="gpu",
            verbosity=-1,
        )
        model.fit(frame, target)
        return "gpu", None
    except Exception as exc:  # noqa: BLE001 - probe must fall back on any failure
        return "cpu", f"gpu_probe_failed: {type(exc).__name__}: {str(exc)[:120]}"


def _resolve_screening_device(device: str) -> tuple[str, str | None]:
    """요청된 스크리닝 장치를 결정합니다.

    ``auto`` 는 GPU probe 를 1회 수행하고 실패(드라이버/NVML/LightGBM 빌드/VRAM)
    시 단일 스레드 CPU 로 폴백하며 실패 사유를 반환합니다. probe 결과는 프로세스
    내에서 캐시되어 컨트롤/후보가 항상 같은 resolved device 를 사용합니다.
    """
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError(
            f"screening_device must be one of 'cpu', 'gpu', 'auto', got {device!r}"
        )
    if device != "auto":
        return device, None
    if device not in _SCREENING_DEVICE_RESULTS:
        _SCREENING_DEVICE_RESULTS[device] = _probe_gpu_device()
    return _SCREENING_DEVICE_RESULTS[device]


def _screen_gains(
    x: pd.DataFrame, target: np.ndarray, config: FeatureSelectionConfig
) -> np.ndarray:
    """결정적 CPU Huber LightGBM 스크리닝을 적합하고 컬럼별 gain 을 반환합니다.

    CPU 적합에는 ``deterministic=True`` + ``force_col_wise=True`` 를 적용해 스레드
    수(config ``n_jobs``)와 무관하게 같은 결과가 재현되며, GPU 는 기존 설정을
    유지합니다. ``x`` 의 컬럼 순서와 반환 배열이 1:1 대응합니다.
    """
    resolved_device, _ = _resolve_screening_device(config.screening_device)
    kwargs: dict[str, Any] = {
        "objective": "huber",
        "random_state": config.random_seed,
        "n_jobs": config.n_jobs,
        "verbosity": -1,
        "device": resolved_device,
    }
    if resolved_device == "cpu":
        kwargs["deterministic"] = True
        kwargs["force_col_wise"] = True
    model = LGBMRegressor(**kwargs)
    model.fit(x, target)
    return np.asarray(
        model.booster_.feature_importance(importance_type="gain"), dtype=np.float64
    )


def _permutation_null_maxima(
    frame: pd.DataFrame,
    feature_names: list[str],
    target: np.ndarray,
    config: FeatureSelectionConfig,
    group_col: str,
) -> np.ndarray:
    """같은 ``group_col``(거래일) 안에서만 target 을 섞은 null 순열 최대 gain 분포.

    각 순열은 ``config.random_seed`` 로 구동되는 단일 결정적 RNG 를 공유해 항상
    같은 순열 시퀀스를 산출합니다. target 은 어떤 거래일에서도 다른 거래일로
    이동하지 않으므로 일별 target 분포는 보존되고 피처-수익 연관만 끊어집니다.
    반환 배열은 순열별 생존 피처 최대 gain 입니다.
    """
    x = frame[feature_names]
    groups = frame.groupby(group_col, sort=True).indices
    group_positions = [np.asarray(positions, dtype=np.intp) for positions in groups.values()]
    rng = np.random.default_rng(config.random_seed)
    maxima = np.empty(config.null_permutations, dtype=np.float64)
    for i in range(config.null_permutations):
        permuted = target.copy()
        for positions in group_positions:
            chunk = permuted[positions]
            permuted[positions] = chunk[rng.permutation(chunk.size)]
        maxima[i] = _screen_gains(x, permuted, config).max()
    return maxima

def select_features(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: FeatureSelectionConfig,
    *,
    group_col: str | None = None,
    provenance: Literal["fold_train_selection", "full_training_selection"] = "full_training_selection",
) -> FeatureSelectionResult:
    """``train`` 구간 데이터만 사용해 fold-local 피처 선택을 수행합니다.

    ``selection_rule`` 이 ``fixed_cap`` 이면 기존 300--400--500 계약을, ``permutation_fwer``
    이면 날짜-로컬 순열 null 임계값 계약을 따릅니다 (모듈 docstring 참조).

    Args:
        train: 선택에 사용할 train 분할 DataFrame (validation/이후 행 금지).
        feature_cols: 후보 피처 컬럼 (순서는 결과에 영향을 주지 않습니다).
        target_col: 학습 라벨 컬럼.
        config: 선택 설정.
        group_col: ``permutation_fwer`` 에 필수인 거래일 그룹 컬럼. target 은 이
            컬럼 값이 같은 행 안에서만 순열됩니다.
        provenance: 결과 메타데이터에 기록할 선택 출처. fold train 전용이면
            ``fold_train_selection``, full-data 최종이면 ``full_training_selection``.

    Returns:
        ``FeatureSelectionResult``.

    Raises:
        ValueError: 후보가 없거나, ``permutation_fwer`` 에서 ``group_col`` 누락/
            유지 수가 ``[min_significant_features, hard_max_retained]`` 를 벗어나거나,
            ``fixed_cap`` 에서 유지 수가 ``[min_retained, hard_max_retained]`` 를
            벗어나면 발생합니다. zero-importance 컬럼으로 패딩하거나 조용히
            truncation 하지 않습니다.
    """
    if target_col not in train.columns:
        raise ValueError(f"train must contain target_col {target_col!r}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")
    target_arr = train[target_col].to_numpy(dtype=np.float64)
    if not np.isfinite(target_arr).all():
        raise ValueError("target_col contains non-finite values; selection failed closed")

    # 방어적 정제: 후보의 모든 비유한 값은 결측으로 취급합니다 (LightGBM/상관에
    # 무한대가 도달하지 않도록 합니다).
    sanitized = train.copy()
    for col in feature_cols:
        if col not in sanitized.columns:
            continue
        arr = sanitized[col].to_numpy(dtype=np.float64)
        sanitized[col] = np.where(np.isfinite(arr), arr, np.nan)

    survivors, rejected, quality_evidence = _reject_quality(
        sanitized, list(feature_cols), config
    )
    if not survivors:
        raise ValueError(
            f"no candidate features survived quality screening "
            f"(n_candidates={len(feature_cols)})"
        )

    resolved_device, fallback_reason = _resolve_screening_device(config.screening_device)
    feature_matrix = sanitized[survivors]

    if config.selection_rule == "fixed_cap":
        raw_gains = _screen_gains(feature_matrix, target_arr, config)
        gain_by_name = dict(zip(survivors, raw_gains.tolist(), strict=True))

        positive = [name for name in survivors if gain_by_name[name] > config.min_gain]
        positive.sort(key=lambda name: (-gain_by_name[name], name))
        for name in survivors:
            if name not in set(positive):
                rejected[name] = _REASON_ZERO_GAIN

        pruned, pruned_rejected = _prune_correlated(sanitized, positive, config)
        rejected.update(pruned_rejected)

        selected = pruned[: config.max_retained]
        for name in pruned[config.max_retained :]:
            rejected[name] = _REASON_BEYOND_CAP

        n_retained = len(selected)
        if not (config.min_retained <= n_retained <= config.hard_max_retained):
            raise ValueError(
                f"retained feature count {n_retained} is outside "
                f"[{config.min_retained}, {config.hard_max_retained}]; "
                "selection failed closed without padding from zero-gain columns"
            )
        gains = tuple((name, float(gain_by_name[name])) for name in positive)
        pre_prune = positive
        threshold: float | None = None
        null_maxima = np.empty(0, dtype=np.float64)
    else:
        if group_col is None:
            raise ValueError(
                "selection_rule='permutation_fwer' requires a non-None group_col"
            )
        if group_col not in sanitized.columns:
            raise ValueError(f"train must contain group_col {group_col!r}")
        real_gains = _screen_gains(feature_matrix, target_arr, config)
        gain_by_name = dict(zip(survivors, real_gains.tolist(), strict=True))
        null_maxima = _permutation_null_maxima(
            sanitized, survivors, target_arr, config, group_col
        )
        threshold = float(np.quantile(null_maxima, 1.0 - config.null_alpha))

        significant = [name for name in survivors if gain_by_name[name] > threshold]
        significant.sort(key=lambda name: (-gain_by_name[name], name))
        for name in survivors:
            if name not in set(significant):
                rejected[name] = _REASON_NOT_SIGNIFICANT

        pruned, pruned_rejected = _prune_correlated(sanitized, significant, config)
        rejected.update(pruned_rejected)

        selected = pruned
        n_retained = len(selected)
        if not (
            config.min_significant_features <= n_retained <= config.hard_max_retained
        ):
            raise ValueError(
                f"retained feature count {n_retained} is outside "
                f"[{config.min_significant_features}, {config.hard_max_retained}] for "
                "selection_rule='permutation_fwer'; selection failed closed without "
                "padding or silent truncation"
            )
        gains = tuple((name, float(gain_by_name[name])) for name in significant)
        pre_prune = significant

    counts = {
        "n_candidates": len(feature_cols),
        "n_survived_quality": len(survivors),
        "n_positive_gain": len(pre_prune),
        "n_pre_correlation": len(pre_prune),
        "n_post_correlation": len(pruned),
        "n_rejected": len(rejected),
        "n_retained": n_retained,
    }
    metadata = {
        "version": FEATURE_SELECTION_VERSION,
        "random_seed": config.random_seed,
        "n_train_rows": len(sanitized),
        "requested_screening_device": config.screening_device,
        "resolved_screening_device": resolved_device,
        "gpu_fallback_reason": fallback_reason,
        "config_fingerprint": config_fingerprint(config),
        "selection_rule": config.selection_rule,
        "selection_threshold": threshold,
        "null_alpha": (
            config.null_alpha if config.selection_rule == "permutation_fwer" else None
        ),
        "null_permutations": (
            config.null_permutations if config.selection_rule == "permutation_fwer" else 0
        ),
        "null_max_gain_min": (
            float(null_maxima.min()) if null_maxima.size else None
        ),
        "null_max_gain_median": (
            float(np.median(null_maxima)) if null_maxima.size else None
        ),
        "null_max_gain_max": (
            float(null_maxima.max()) if null_maxima.size else None
        ),
        "permutation_seed": (
            config.random_seed if config.selection_rule == "permutation_fwer" else None
        ),
        "provenance": provenance,
        "n_survived_quality": len(survivors),
        "n_pre_correlation": len(pre_prune),
        "n_post_correlation": len(pruned),
        "n_retained": n_retained,
        "at_hard_max_retained": n_retained == config.hard_max_retained,
    }
    return FeatureSelectionResult(
        selected_features=tuple(selected),
        gains=gains,
        rejected=tuple(sorted(rejected.items(), key=lambda item: item[0])),
        counts=counts,
        metadata=metadata,
        quality_evidence=quality_evidence,
    )


def median_pairwise_jaccard(feature_lists: Sequence[Sequence[str]]) -> float:
    """outer fold 피처 목록 간 median pairwise Jaccard 안정성을 반환합니다."""
    lists = [set(features) for features in feature_lists]
    if len(lists) < 2:
        return 1.0
    scores: list[float] = []
    for a, b in itertools.combinations(lists, 2):
        union = len(a | b)
        scores.append(len(a & b) / union if union else 1.0)
    return float(np.median(scores))


class FeatureQualityRecord(pydantic.BaseModel):
    """폴드 선택 결과에서 집계된 피처 품질 프로필 (불변).

    ``finite_ratio`` / ``missing_rate`` / ``zero_variance`` 는 각 fold 의 train
    행렬에서 한 번 계산되어 ``FeatureSelectionResult.quality_evidence`` 에 보관된
    값을 폴드 간 평균/논리합으로 집계합니다 (두 번째 데이터 스캔 없음).
    ``selection_count`` / ``selection_rate`` 는 폴드별 선택 빈도, ``rejection_counts``
    는 거부 사유별 폴드 수입니다.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    feature_name: str
    family: str
    finite_ratio: float | None
    missing_rate: float | None
    zero_variance: bool | None
    selection_count: int
    selection_rate: float
    positive_gain_count: int
    rejection_counts: dict[str, int]


def build_feature_quality_report(
    selections: Sequence[FeatureSelectionResult],
    candidate_feature_cols: Sequence[str],
    catalogue: Mapping[str, Mapping[str, str]],
    min_fold_selection_rate: float,
) -> dict[str, Any]:
    """폴드별 ``FeatureSelectionResult`` 를 결정적 피처/가족 품질 리포트로 집계합니다.

    Args:
        selections: 외부 fold 의 train 분할에서만 파생된 선택 결과 (검증/이후 행
            라벨은 절대 폴드 리포트에 영향을 주지 않습니다).
        candidate_feature_cols: 품질을 측정할 후보 컬럼 (선택 빈도 0 인 후보 포함).
        catalogue: 카탈로그 피처의 품질 메타데이터 (``family`` 필수). 카탈로그에
            없는 후보는 ``baseline`` 가족으로 분류합니다.
        min_fold_selection_rate: ``selection_rate`` 가 이 값 미만이면 ``unstable``
            로 분류합니다.

    Returns:
        ``version``(feature_quality_v2), ``n_folds``, ``feature_records``,
        ``family_records``, ``actions`` 로 구성된 결정적 리포트.

    Raises:
        ValueError: ``min_fold_selection_rate`` 가 (0, 1] 밖이거나, 카탈로그에
            등재된 히스토리 피처의 ``family`` 메타데이터가 누락됐거나, 선택 증거에
            알려지지 않은 거부 사유가 있으면 발생합니다.
    """
    if not 0.0 < min_fold_selection_rate <= 1.0:
        raise ValueError(
            f"min_fold_selection_rate must be in (0, 1], got {min_fold_selection_rate}"
        )
    n_folds = len(selections)
    features = sorted(set(candidate_feature_cols))

    def _family_of(feature: str) -> str:
        entry = catalogue.get(feature)
        if entry is None:
            return _BASELINE_FAMILY
        family = str(entry.get("family", "") or "")
        if not family:
            raise ValueError(
                f"catalogue metadata missing family for historical feature {feature!r}"
            )
        return family

    selection_counts: dict[str, int] = {}
    positive_gain_counts: dict[str, int] = {}
    rejection_counts: dict[str, dict[str, int]] = {}
    finite_ratios: dict[str, list[float]] = {}
    missing_rates: dict[str, list[float]] = {}
    zero_variances: dict[str, list[bool]] = {}

    for selection in selections:
        for _feature, reason in selection.rejected:
            if reason not in _KNOWN_REJECTION_REASONS:
                raise ValueError(
                    f"unknown rejection reason {reason!r} for feature {_feature!r}"
                )
        selected = set(selection.selected_features)
        rejected_by_feature = dict(selection.rejected)
        for feature in features:
            if feature in selected:
                selection_counts[feature] = selection_counts.get(feature, 0) + 1
            rejected_reason = rejected_by_feature.get(feature)
            if rejected_reason is not None:
                per_feature = rejection_counts.setdefault(feature, {})
                per_feature[rejected_reason] = per_feature.get(rejected_reason, 0) + 1
        gain_names = {name for name, _gain in selection.gains}
        for feature in features:
            if feature in gain_names:
                positive_gain_counts[feature] = positive_gain_counts.get(feature, 0) + 1
        for feature, evidence in selection.quality_evidence.items():
            if feature not in features:
                continue
            finite_ratios.setdefault(feature, []).append(
                float(evidence.get("finite_ratio", float("nan")))
            )
            missing_rates.setdefault(feature, []).append(
                float(evidence.get("missing_rate", float("nan")))
            )
            zero_variances.setdefault(feature, []).append(
                bool(evidence.get("zero_variance", False))
            )

    feature_records: list[FeatureQualityRecord] = []
    actions: dict[str, list[str]] = {}
    for feature in features:
        rej = rejection_counts.get(feature, {})
        feature_actions: list[str] = []
        if rej.get(_REASON_ALL_NONFINITE, 0) or rej.get(_REASON_TRAIN_MISSING_RATE, 0):
            feature_actions.append("source_incomplete")
        if rej.get(_REASON_BEYOND_CAP, 0):
            feature_actions.append("capacity_limited")
        if rej.get(_REASON_CORRELATED, 0):
            feature_actions.append("redundant")
        if rej.get(_REASON_NOT_SIGNIFICANT, 0):
            feature_actions.append("screening_weak")
        selection_count = selection_counts.get(feature, 0)
        selection_rate = selection_count / n_folds if n_folds else 0.0
        if selection_rate < min_fold_selection_rate:
            feature_actions.append("unstable")
        positive_gain_count = positive_gain_counts.get(feature, 0)
        if positive_gain_count == 0 and "source_incomplete" not in feature_actions:
            feature_actions.append("screening_weak")
        actions[feature] = feature_actions
        feature_records.append(
            FeatureQualityRecord(
                feature_name=feature,
                family=_family_of(feature),
                finite_ratio=(
                    float(np.nanmean(finite_ratios[feature])) if finite_ratios.get(feature) else None
                ),
                missing_rate=(
                    float(np.nanmean(missing_rates[feature])) if missing_rates.get(feature) else None
                ),
                zero_variance=(
                    bool(any(zero_variances[feature])) if zero_variances.get(feature) else None
                ),
                selection_count=selection_count,
                selection_rate=selection_rate,
                positive_gain_count=positive_gain_count,
                rejection_counts=dict(sorted(rej.items())),
            )
        )

    family_records: list[dict[str, Any]] = []
    for family in sorted({record.family for record in feature_records}):
        members = [record for record in feature_records if record.family == family]
        action_counts = dict.fromkeys(_QUALITY_ACTION_NAMES, 0)
        for record in members:
            for action in actions[record.feature_name]:
                action_counts[action] += 1
        family_records.append(
            {
                "family": family,
                "n_features": len(members),
                "n_selected": sum(1 for record in members if record.selection_count > 0),
                "selection_rate": (
                    sum(record.selection_rate for record in members) / len(members)
                ),
                **action_counts,
                "features": [record.feature_name for record in members],
            }
        )

    return {
        "version": FEATURE_QUALITY_VERSION,
        "n_folds": n_folds,
        "feature_records": [record.model_dump() for record in feature_records],
        "family_records": family_records,
        "actions": actions,
    }
