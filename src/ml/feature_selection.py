"""Fold-local feature selection contract (``fold_local_v1``).

`docs/specs/ml_feature_engineering_evolution.md` 계약을 구현합니다. 각 outer
walk-forward fold 의 train 구간에서만 데이터를 사용해 LightGBM(Huber) gain
중요도 기반으로 후보를 선별합니다.

- inner purged walk-forward fold 만 사용하며 inner validation 라벨은 중요도에
  사용하지 않습니다.
- outer validation 라벨과 이후 날짜는 selector 에 전혀 노출되지 않습니다.
- positive gain 을 가진 inner fold 의 strict majority 지원을 가진 피처만
  유지하고, (지원 수 desc, 중앙 정규화 gain desc, 이름 asc) 로 정렬해
  설정된 ``retain_count`` 를 유지합니다.
- 후보/적격/유지 수 불변량 위반은 ``ValueError`` 로 fail-closed 합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.ml.purged_cv import PurgedGroupTimeSeriesSplit
from src.processing.feature_catalog import MAX_CANDIDATES, MIN_CANDIDATES

logger = logging.getLogger(__name__)

FEATURE_SELECTION_VERSION = "fold_local_v1"


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """fold-local 선택 설정 (불변)."""

    n_inner_splits: int = 5
    purge_gap: int = 1
    retain_count: int = 400
    min_retain: int = 300
    max_retain: int = 500
    min_candidates: int = MIN_CANDIDATES
    max_candidates: int = MAX_CANDIDATES
    random_state: int = 42


@dataclass(frozen=True)
class FeatureSelectionResult:
    """선택 결과와 진단 요약."""

    selected_feature_cols: list[str]
    eligible_feature_cols: list[str]
    candidate_feature_cols: list[str]
    n_inner_folds: int
    support_summary: pd.DataFrame
    config: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)


def _eligible_columns(
    df: pd.DataFrame,
    candidate_cols: list[str],
) -> tuple[list[str], dict[str, str]]:
    """train 구간 데이터만 사용해 적격 컬럼을 결정합니다.

    선언되지 않은 카탈로그 컬럼, 전부 결측인 컬럼, 유한 값이 영분산인 컬럼,
    선언된 이력 룩백 요건을 평가할 수 없는 컬럼을 거부합니다.
    """
    manifest = df.attrs.get("feature_manifest")
    manifest_names: set[str] = set()
    availability: dict[str, str] = {}
    if manifest is not None and "feature_name" in manifest.columns:
        manifest_names = set(manifest["feature_name"].astype(str))
        for _, row in manifest[["feature_name", "availability_rule"]].iterrows():
            availability[str(row["feature_name"])] = str(row["availability_rule"])

    rejected: dict[str, str] = {}
    for col in candidate_cols:
        if col not in df.columns:
            rejected[col] = "missing_column"
            continue
        if manifest_names and col not in manifest_names:
            rejected[col] = "undeclared_catalog_column"
            continue
        arr = df[col].to_numpy(dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            if availability.get(col) == "prior_date_history_only":
                rejected[col] = "history_lookback_unavailable"
            else:
                rejected[col] = "all_missing"
            continue
        if np.unique(finite).size <= 1:
            rejected[col] = "zero_variance"
            continue
    eligible = [col for col in candidate_cols if col not in rejected]
    return eligible, rejected


def _fold_normalized_gains(
    work: pd.DataFrame,
    eligible: list[str],
    target_col: str,
    group_col: str,
    config: FeatureSelectionConfig,
) -> list[np.ndarray]:
    """inner purged fold 의 train 파티션에서 LightGBM gain 중요도를 수집합니다."""
    splitter = PurgedGroupTimeSeriesSplit(n_splits=config.n_inner_splits, purge_gap=config.purge_gap)
    gains: list[np.ndarray] = []
    for train_idx, _val_idx in splitter.split(
        work, y=work[target_col], groups=work[group_col]
    ):
        inner_train = work.iloc[train_idx]
        model = LGBMRegressor(
            objective="huber",
            random_state=config.random_state,
            verbosity=-1,
            num_threads=1,
        )
        model.fit(inner_train[eligible], inner_train[target_col])
        raw = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=np.float64)
        positive = np.where(raw > 0, raw, 0.0)
        total = positive.sum()
        if total > 0:
            gains.append(positive / total)
        else:
            gains.append(np.zeros(len(eligible), dtype=np.float64))
    return gains


class FoldLocalFeatureSelector:
    """outer fold 마다 분리된 결정적 fold-local 피처 선택기."""

    def __init__(self, config: FeatureSelectionConfig | None = None) -> None:
        self.config = config or FeatureSelectionConfig()
        if self.config.min_candidates > self.config.max_candidates:
            raise ValueError("min_candidates must be <= max_candidates")
        if self.config.min_retain > self.config.max_retain:
            raise ValueError("min_retain must be <= max_retain")
        if not (self.config.min_retain <= self.config.retain_count <= self.config.max_retain):
            raise ValueError(
                f"retain_count {self.config.retain_count} must be within "
                f"[{self.config.min_retain}, {self.config.max_retain}]"
            )

    def select(
        self,
        df: pd.DataFrame,
        candidate_cols: list[str],
        target_col: str,
        group_col: str,
    ) -> FeatureSelectionResult:
        """outer train 구간에서만 fold-local 선택을 수행합니다.

        Raises:
            ValueError: 후보/적격/유지 수 불변량 위반 또는 inner fold 가
                데이터를 지지하지 못하는 경우.
        """
        if target_col not in df.columns or group_col not in df.columns:
            raise ValueError(
                f"df must contain target_col {target_col!r} and group_col {group_col!r}"
            )
        work = df.sort_values(group_col).copy()
        work.attrs = dict(df.attrs)

        eligible, _rejected = _eligible_columns(work, list(candidate_cols))
        n_eligible = len(eligible)
        if not (self.config.min_candidates <= n_eligible <= self.config.max_candidates):
            raise ValueError(
                f"eligible candidate count {n_eligible} is outside the research target "
                f"[{self.config.min_candidates}, {self.config.max_candidates}]; "
                "data cannot support fold-local selection"
            )

        gains = _fold_normalized_gains(work, eligible, target_col, group_col, self.config)
        n_inner_folds = len(gains)
        if n_inner_folds == 0:
            raise ValueError("no usable inner folds; too few train groups for fold-local selection")

        gain_matrix = np.vstack(gains)
        support_counts = (gain_matrix > 0).sum(axis=0)
        median_gains = np.median(gain_matrix, axis=0)
        majority_threshold = n_inner_folds / 2.0

        rows = [
            {
                "feature_name": name,
                "support_count": int(support_counts[i]),
                "n_inner_folds": n_inner_folds,
                "median_normalized_gain": float(median_gains[i]),
            }
            for i, name in enumerate(eligible)
        ]
        support_summary = pd.DataFrame(rows, columns=[
            "feature_name",
            "support_count",
            "n_inner_folds",
            "median_normalized_gain",
        ])

        order = (
            support_summary.assign(rank=0)
            .sort_values(
                by=["support_count", "median_normalized_gain", "feature_name"],
                ascending=[False, False, True],
            )
        )
        majority = order[order["support_count"] > majority_threshold].copy()
        retained = majority.head(self.config.retain_count)
        retained_feature_cols = retained["feature_name"].tolist()

        if not (self.config.min_retain <= len(retained_feature_cols) <= self.config.max_retain):
            raise ValueError(
                f"retained feature count {len(retained_feature_cols)} is outside "
                f"[{self.config.min_retain}, {self.config.max_retain}]; selection failed closed"
            )

        retained_names = set(retained_feature_cols)
        support_summary["retained"] = support_summary["feature_name"].isin(retained_names)

        return FeatureSelectionResult(
            selected_feature_cols=retained_feature_cols,
            eligible_feature_cols=list(eligible),
            candidate_feature_cols=list(candidate_cols),
            n_inner_folds=n_inner_folds,
            support_summary=support_summary,
            config=self.config,
        )


def serialize_selection_result(result: FeatureSelectionResult) -> dict[str, Any]:
    """선택 결과를 번들 영속화용 직렬화 가능 dict 로 변환합니다."""
    summary = result.support_summary.copy()
    summary["median_normalized_gain"] = summary["median_normalized_gain"].fillna(0.0)
    return {
        "selected_feature_cols": list(result.selected_feature_cols),
        "eligible_feature_cols": list(result.eligible_feature_cols),
        "candidate_feature_cols": list(result.candidate_feature_cols),
        "candidate_count": len(result.candidate_feature_cols),
        "eligible_count": len(result.eligible_feature_cols),
        "retained_count": len(result.selected_feature_cols),
        "n_inner_folds": result.n_inner_folds,
        "support_summary": summary,
    }
