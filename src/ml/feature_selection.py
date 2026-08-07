"""Fold-local feature screening/selection using train-only statistics.

`docs/specs/ml_feature_selection_pipeline.md` 의 fold-local 선택 계약을
구현합니다. 각 outer fold 의 train 분할에서만 계산하므로 validation/이후 날짜
라벨은 선택에 절대 노출되지 않습니다.

선택 흐름 (train 구간 전용, 결정적):
1. 품질 거부: 전부 비유한(all-nonfinite) 컬럼, train 결측률 > 0.35, 영분산 컬럼을
   이유와 함께 거부합니다.
2. 고정 시드·단일 스레드 LightGBM(Huber) 스크리닝 모델로 gain 중요도를 계산합니다.
3. ``gain <= min_gain`` 인 영-gain 컬럼을 ``zero_gain`` 으로 거부합니다.
4. train 행만 사용한 절대 Spearman 상관이 0.98 을 넘는 쌍을 (gain desc, 이름 asc)
   결정적 tie-break 으로 가지치기합니다 (``correlated_pair_pruned``).
5. 양의 gain 컬럼을 (gain desc, 이름 asc) 로 정렬해 최대 ``max_retained`` 개를
   유지하며, 초과분은 ``beyond_max_retained`` 로 거부합니다.
6. 유지 수가 ``min_retained`` 미만(또는 ``hard_max_retained`` 초과)이면
   ``ValueError`` 로 fail-closed 하며, zero-importance 컬럼으로 패딩하지 않습니다.

Vectorized 전용이며 ``pd.apply`` 를 사용하지 않습니다. SHAP 은 선택기가 아니라
옵션 사후 설명 전용입니다.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import pydantic
from lightgbm import LGBMRegressor

FEATURE_SELECTION_VERSION = "fold_local_gain_v1"

_REASON_ALL_NONFINITE = "all_nonfinite"
_REASON_MISSING_COLUMN = "missing_column"
_REASON_TRAIN_MISSING_RATE = "train_missing_rate"
_REASON_ZERO_VARIANCE = "zero_variance"
_REASON_ZERO_GAIN = "zero_gain"
_REASON_CORRELATED = "correlated_pair_pruned"
_REASON_BEYOND_CAP = "beyond_max_retained"


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
    screening_device: Literal["cpu", "gpu", "auto"] = "cpu"

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
        return self


class FeatureSelectionResult(pydantic.BaseModel):
    """선택 결과 요약 (불변).

    - ``selected_features``: 최종 유지 피처 (gain desc, 이름 asc) 순서.
    - ``gains``: 양의 gain 을 가진 후보의 (feature, gain) 쌍, (gain desc, 이름 asc).
    - ``rejected``: (feature, reason) 쌍 (품질/영-gain/상관/초과 사유 포함).
    - ``counts``: n_candidates, n_rejected, n_positive_gain, n_retained 등.
    - ``metadata``: data_cutoff, n_train_rows, n_groups, random_seed 등.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    selected_features: tuple[str, ...]
    gains: tuple[tuple[str, float], ...]
    rejected: tuple[tuple[str, str], ...]
    counts: dict[str, int]
    metadata: dict[str, Any]


def _reject_quality(
    train: pd.DataFrame,
    feature_cols: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], dict[str, str]]:
    """품질 기준으로 후보를 거부하고 생존자 목록을 반환합니다."""
    rejected: dict[str, str] = {}
    for col in feature_cols:
        if col not in train.columns:
            rejected[col] = _REASON_MISSING_COLUMN
            continue
        arr = train[col]
        if not np.isfinite(arr.to_numpy(dtype=np.float64)).any():
            rejected[col] = _REASON_ALL_NONFINITE
            continue
        if float(arr.isna().mean()) > config.missing_rate_threshold:
            rejected[col] = _REASON_TRAIN_MISSING_RATE
            continue
        if arr.nunique(dropna=False) <= 1:
            rejected[col] = _REASON_ZERO_VARIANCE
            continue
    survivors = [col for col in feature_cols if col not in rejected]
    return survivors, rejected


def _prune_correlated(
    train: pd.DataFrame,
    ordered_names: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], dict[str, str]]:
    """train 행만 사용한 Spearman 상관 가지치기 (결정적 tie-break)."""
    if len(ordered_names) < 2:
        return ordered_names, {}
    corr = train[ordered_names].corr(method="spearman")
    kept: list[str] = []
    rejected: dict[str, str] = {}
    for name in ordered_names:
        if any(abs(corr.at[name, k]) > config.correlation_threshold for k in kept):
            rejected[name] = _REASON_CORRELATED
        else:
            kept.append(name)
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


def select_features(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: FeatureSelectionConfig,
) -> FeatureSelectionResult:
    """``train`` 구간 데이터만 사용해 fold-local 피처 선택을 수행합니다.

    Args:
        train: 선택에 사용할 train 분할 DataFrame (validation/이후 행 금지).
        feature_cols: 후보 피처 컬럼 (순서는 결과에 영향을 주지 않습니다).
        target_col: 학습 라벨 컬럼.
        config: 선택 설정.

    Returns:
        ``FeatureSelectionResult``.

    Raises:
        ValueError: 후보가 없거나 유지 수가 ``[min_retained, hard_max_retained]``
            범위를 벗어나면 발생합니다. zero-importance 패딩은 없습니다.
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

    survivors, rejected = _reject_quality(sanitized, list(feature_cols), config)
    if not survivors:
        raise ValueError(
            f"no candidate features survived quality screening "
            f"(n_candidates={len(feature_cols)})"
        )

    resolved_device, fallback_reason = _resolve_screening_device(config.screening_device)
    model = LGBMRegressor(
        objective="huber",
        random_state=config.random_seed,
        n_jobs=1,
        verbosity=-1,
        device=resolved_device,
    )
    model.fit(sanitized[survivors], sanitized[target_col])
    raw_gains = np.asarray(
        model.booster_.feature_importance(importance_type="gain"), dtype=np.float64
    )
    gain_by_name = dict(zip(survivors, raw_gains.tolist(), strict=True))

    positive = [
        name for name in survivors if gain_by_name[name] > config.min_gain
    ]
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
    counts = {
        "n_candidates": len(feature_cols),
        "n_survived_quality": len(survivors),
        "n_positive_gain": len(positive),
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
    }
    return FeatureSelectionResult(
        selected_features=tuple(selected),
        gains=gains,
        rejected=tuple(sorted(rejected.items(), key=lambda item: item[0])),
        counts=counts,
        metadata=metadata,
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
