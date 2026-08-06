"""Fold-local feature selection 단위 테스트.

SCENARIO_FOLD_LOCAL_SELECTION_01: outer validation 라벨 또는 이후 라벨 변경이
outer-fold 선별 목록을 바꾸지 않습니다.
SCENARIO_FOLD_LOCAL_SELECTION_02: 동일 입력/시드 → 동일 목록; 적격 후보 부족 또는
유지 수가 300--500 을 벗어나면 ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_selection import FeatureSelectionConfig, FoldLocalFeatureSelector
from src.ml.sizing_engine import _train_inline_bundle
from src.ml.training.pipelines import run_model_pipeline

N_CANDIDATES = 650


def _predictive_df(n_groups: int = 150, per_group: int = 13, noise: float = 0.28) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = n_groups * per_group
    signal = rng.normal(0, 1, n)
    noise_matrix = rng.normal(0, noise, (n, N_CANDIDATES))
    cols: dict[str, np.ndarray] = {
        f"f{j:04d}": signal * (1 + j % 5 * 0.1) + noise_matrix[:, j]
        for j in range(N_CANDIDATES)
    }
    df = pd.DataFrame(cols)
    df.insert(0, "date", (np.arange(n) // per_group).astype(int))
    df["target_return"] = signal + rng.normal(0, 0.08, n)
    df["stock_code"] = [f"{i % 13:06d}" for i in range(n)]
    df["chart_analysis"] = "scenario_a"
    df["selection_rank"] = np.tile(np.arange(per_group), n_groups)
    return df


def _candidate_cols() -> list[str]:
    return [f"f{j:04d}" for j in range(N_CANDIDATES)]


def test_selector_deterministic_same_input_same_list() -> None:
    """SCENARIO_FOLD_LOCAL_SELECTION_02: 동일 입력/시드는 동일 목록을 산출합니다."""
    df = _predictive_df()
    cfg = FeatureSelectionConfig()
    first = FoldLocalFeatureSelector(cfg).select(df, _candidate_cols(), "target_return", "date")
    second = FoldLocalFeatureSelector(cfg).select(df, _candidate_cols(), "target_return", "date")
    assert first.selected_feature_cols == second.selected_feature_cols
    assert first.eligible_feature_cols == second.eligible_feature_cols
    pd.testing.assert_frame_equal(first.support_summary, second.support_summary)
    assert 300 <= len(first.selected_feature_cols) <= 500


def test_selector_raises_on_too_few_eligible_candidates() -> None:
    """SCENARIO_FOLD_LOCAL_SELECTION_02: 적격 후보가 600 미만이면 ValueError."""
    df = _predictive_df()
    with pytest.raises(ValueError, match="eligible candidate count"):
        FoldLocalFeatureSelector().select(df, _candidate_cols()[:100], "target_return", "date")


def test_selector_raises_on_retained_outside_range() -> None:
    """SCENARIO_FOLD_LOCAL_SELECTION_02: 유지 수가 300--500 을 벗어나면 ValueError."""
    df = _predictive_df()
    small = df[df["date"] < 30].copy()
    with pytest.raises(ValueError, match="retained feature count"):
        FoldLocalFeatureSelector().select(small, _candidate_cols(), "target_return", "date")


def test_selector_config_rejects_retain_count_outside_range() -> None:
    """유지 설정이 [300, 500] 을 벗어나면 선택기 초기화가 실패합니다."""
    with pytest.raises(ValueError, match="retain_count"):
        FoldLocalFeatureSelector(FeatureSelectionConfig(retain_count=250))


def test_pipeline_wiring_reports_selected_features_by_fold() -> None:
    """run_model_pipeline 은 fold-local 선택 결과를 OOF 리포트에 기록합니다."""
    df = _predictive_df()
    cfg = FeatureSelectionConfig(n_inner_splits=3)
    result = run_model_pipeline(
        df,
        _candidate_cols(),
        "target_return",
        "date",
        n_splits=2,
        purge_gap=1,
        model_type="lgb_regressor",
        feature_selection_config=cfg,
    )
    assert "selected_features_by_fold" in result
    assert result["feature_selection_version"] == "fold_local_v1"
    assert set(result["selected_features_by_fold"]) == {"0", "1"}
    for fold_payload in result["selected_features_by_fold"].values():
        assert 300 <= len(fold_payload["selected_feature_cols"]) <= 500
        assert fold_payload["eligible_count"] >= 600


def test_pipeline_selection_invariant_to_validation_and_later_labels() -> None:
    """SCENARIO_FOLD_LOCAL_SELECTION_01: outer validation/이후 라벨 변경은 선별 불변."""
    df = _predictive_df()
    cfg = FeatureSelectionConfig(n_inner_splits=3)
    baseline = run_model_pipeline(
        df,
        _candidate_cols(),
        "target_return",
        "date",
        n_splits=2,
        purge_gap=1,
        model_type="lgb_regressor",
        feature_selection_config=cfg,
    )

    n_groups = df["date"].max() + 1
    test_size = n_groups // 3
    # 마지막 outer validation 블록의 라벨만 변경합니다 (어떤 fold 의 train 에도 포함되지
    # 않으므로, fold-local 선택은 어느 fold 의 선별 목록도 바꾸면 안 됩니다).
    perturb_mask = df["date"] >= n_groups - test_size
    perturbed = df.copy()
    rng = np.random.default_rng(7)
    perturbed.loc[perturb_mask, "target_return"] = (
        perturbed.loc[perturb_mask, "target_return"] + rng.normal(0, 0.5, int(perturb_mask.sum()))
    )

    changed = run_model_pipeline(
        perturbed,
        _candidate_cols(),
        "target_return",
        "date",
        n_splits=2,
        purge_gap=1,
        model_type="lgb_regressor",
        feature_selection_config=cfg,
    )

    base = baseline["selected_features_by_fold"]
    mod = changed["selected_features_by_fold"]
    assert set(base) == set(mod)
    for fold in base:
        assert base[fold]["selected_feature_cols"] == mod[fold]["selected_feature_cols"]


def test_run_model_pipeline_rejects_invalid_selection_config() -> None:
    """선택 설정이 FeatureSelectionConfig 가 아니면 ValueError 로 거부합니다."""
    df = _predictive_df()
    with pytest.raises(ValueError, match="FeatureSelectionConfig"):
        run_model_pipeline(
            df,
            _candidate_cols(),
            "target_return",
            "date",
            n_splits=2,
            feature_selection_config="not-a-config",  # type: ignore[arg-type]
        )


def test_train_inline_bundle_persists_catalog_metadata() -> None:
    """카탈로그 버전/해시 attrs 가 번들에 영속화됩니다."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "date": (np.arange(n) // 15).astype(int),
            "f_a": rng.normal(size=n),
            "f_b": rng.normal(size=n),
            "target_return": rng.normal(size=n),
        }
    )
    df.attrs["catalog_version"] = "causal_expanded_v1"
    df.attrs["catalog_hash"] = "abc123"
    bundle = _train_inline_bundle(df, ["f_a", "f_b"], "target_return", "date")
    assert bundle["catalog_version"] == "causal_expanded_v1"
    assert bundle["catalog_hash"] == "abc123"


def test_train_inline_bundle_persists_selection_artifact() -> None:
    """전체 이력 번들은 최종 선별 목록과 지원/게인 요약을 영속화합니다."""
    df = _predictive_df()
    cfg = FeatureSelectionConfig()
    bundle = _train_inline_bundle(
        df,
        _candidate_cols(),
        "target_return",
        "date",
        feature_selection_config=cfg,
    )
    assert bundle["feature_selection_version"] == "fold_local_v1"
    assert bundle["selected_feature_cols"] == bundle["feature_cols"]
    assert 300 <= bundle["retained_count"] <= 500
    assert bundle["candidate_count"] == N_CANDIDATES
    assert bundle["eligible_count"] >= 600
    assert len(bundle["feature_support_summary"]) == bundle["eligible_count"]
    assert {"feature_name", "support_count", "median_normalized_gain"}.issubset(
        set(bundle["feature_support_summary"].columns)
    )
