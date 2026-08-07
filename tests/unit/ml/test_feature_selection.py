"""Fold-local feature selection 단위 테스트.

`docs/specs/ml_feature_selection_pipeline.md` 시나리오:
- FS-04 fold isolation: validation/이후 행·라벨 변경이 train-fold 선택 이름/게인을
  바꾸지 않습니다.
- FS-05 quality rejection: Null/상수/희소/영-gain/상관 컬럼이 명시적 거부 사유를
  받습니다.
- FS-06 deterministic ordering: 고정 시드/동률에서 항상 같은 정렬 목록을 산출합니다.
- FS-07 fail-closed count: 300 개 미만 양의 gain 컬럼이면 패딩 없이 ValueError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_selection import (
    FeatureSelectionConfig,
    median_pairwise_jaccard,
    select_features,
)
from src.ml.model_pipeline import run_model_pipeline


def _predictive_df(
    n_candidates: int = 40,
    n_groups: int = 40,
    rows_per_group: int = 8,
    seed: int = 7,
) -> pd.DataFrame:
    """대체로 예측 가능한 신호 + 노이즈 후보를 가진 학습 DataFrame 을 생성합니다."""
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    dates = pd.to_datetime(
        [f"2024-03-{1 + d % 28:02d}" for d in range(n_groups)]
    )
    signal = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"trade_date": [d for d in dates for _ in range(rows_per_group)]})
    df["target_return"] = 0.02 * signal + rng.normal(0.0, 0.02, n)
    df["selection_rank"] = df.groupby("trade_date", sort=False).cumcount() + 1
    for j in range(n_candidates):
        df[f"f{j:03d}"] = rng.normal(0.0, 1.0, n) + 0.3 * signal * (j % 3)
    return df


def _candidate_cols(n_candidates: int) -> list[str]:
    return [f"f{j:03d}" for j in range(n_candidates)]


def _small_config(min_retained: int = 5, max_retained: int = 20) -> FeatureSelectionConfig:
    return FeatureSelectionConfig(
        min_retained=min_retained,
        max_retained=max_retained,
        hard_max_retained=40,
    )


def test_fs04_fold_selection_invariant_to_validation_and_later_labels() -> None:
    """validation/이후 라벨 변경은 어느 fold 의 선택 목록도 바꾸지 않습니다."""
    df = _predictive_df(n_groups=40, rows_per_group=8)
    cfg = _small_config()
    baseline = run_model_pipeline(
        df,
        _candidate_cols(40),
        "target_return",
        "trade_date",
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
        feature_selection_config=cfg,
    )

    n_groups = df["trade_date"].nunique()
    test_size = n_groups // 4
    perturb_mask = df["trade_date"] >= sorted(df["trade_date"].unique())[-test_size]
    perturbed = df.copy()
    rng = np.random.default_rng(11)
    perturbed.loc[perturb_mask, "target_return"] = (
        perturbed.loc[perturb_mask, "target_return"] + rng.normal(0.0, 0.5, int(perturb_mask.sum()))
    )
    changed = run_model_pipeline(
        perturbed,
        _candidate_cols(40),
        "target_return",
        "trade_date",
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
        feature_selection_config=cfg,
    )
    base_folds = baseline["feature_selection_diagnostics"]["fold_selections"]
    mod_folds = changed["feature_selection_diagnostics"]["fold_selections"]
    for base, mod in zip(base_folds, mod_folds, strict=True):
        assert base["selected_features"] == mod["selected_features"]
        assert base["gains"] == mod["gains"]
        assert base["counts"] == mod["counts"]


def test_fs05_quality_rejection_reasons() -> None:
    """Null/상수/희소/영-gain/상관 컬럼이 명시적 거부 사유를 받습니다."""
    rng = np.random.default_rng(42)
    n = 300
    signal = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"target_return": signal + rng.normal(0.0, 0.05, n)})
    df["f_signal"] = signal
    df["f_noise"] = rng.normal(0.0, 1.0, n)
    df["null_col"] = np.nan
    df["const_col"] = 0.0
    sparse = rng.normal(0.0, 1.0, n)
    sparse[:130] = np.nan  # 43% 결측 > 0.35
    df["sparse_col"] = sparse
    df["dup_col"] = signal  # f_signal 과 완전 동일 -> LightGBM 0 gain
    df["corr_col"] = signal + rng.normal(0.0, 0.001, n)  # corr ~0.999 > 0.98

    cols = [
        "f_signal",
        "f_noise",
        "null_col",
        "const_col",
        "sparse_col",
        "dup_col",
        "corr_col",
    ]
    cfg = FeatureSelectionConfig(min_retained=1, max_retained=10, hard_max_retained=20)
    result = select_features(df, cols, "target_return", cfg)
    reasons = dict(result.rejected)
    assert reasons["null_col"] == "all_nonfinite"
    assert reasons["const_col"] == "zero_variance"
    assert reasons["sparse_col"] == "train_missing_rate"
    assert reasons["dup_col"] == "zero_gain"
    assert reasons["corr_col"] == "correlated_pair_pruned"
    assert set(result.selected_features) <= {"f_signal", "f_noise"}
    assert result.counts["n_rejected"] + result.counts["n_retained"] == result.counts["n_candidates"]


def test_fs06_deterministic_ordering_same_input_same_list() -> None:
    """동일 입력/시드는 항상 같은 정렬 목록과 gain 을 산출합니다."""
    df = _predictive_df(n_groups=50, rows_per_group=10, seed=3)
    cfg = _small_config()
    first = select_features(df, _candidate_cols(40), "target_return", cfg)
    second = select_features(df, _candidate_cols(40), "target_return", cfg)
    assert first.selected_features == second.selected_features
    assert first.gains == second.gains
    assert first.rejected == second.rejected
    # gains 는 (gain desc, 이름 asc) 로 정렬됩니다.
    gains = list(first.gains)
    assert all(gains[i][1] >= gains[i + 1][1] for i in range(len(gains) - 1))


def test_fs06_tie_break_deterministic_duplicate_features() -> None:
    """gain 동률(완전 동일 컬럼)에서도 이름 오름차순 tie-break 로 결정적입니다."""
    rng = np.random.default_rng(5)
    n = 200
    signal = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "target_return": signal + rng.normal(0.0, 0.02, n),
            "b_strong": signal,
            "a_strong": signal,
        }
    )
    cfg = FeatureSelectionConfig(min_retained=1, max_retained=10, hard_max_retained=20)
    first = select_features(df, ["b_strong", "a_strong"], "target_return", cfg)
    second = select_features(df, ["b_strong", "a_strong"], "target_return", cfg)
    assert first.selected_features == second.selected_features
    assert set(first.selected_features) <= {"b_strong", "a_strong"}


def test_fs07_fails_closed_when_fewer_than_min_retained() -> None:
    """양의 gain 컬럼이 min_retained 미만이면 패딩 없이 ValueError 를 발생합니다."""
    df = _predictive_df(n_candidates=6, n_groups=40, rows_per_group=8)
    cfg = FeatureSelectionConfig(min_retained=10, max_retained=20, hard_max_retained=40)
    with pytest.raises(ValueError, match="retained feature count"):
        select_features(df, _candidate_cols(6), "target_return", cfg)


def test_fs07_rejects_invalid_config_bounds() -> None:
    """max_retained > hard_max_retained 설정은 초기화 시 거부됩니다."""
    with pytest.raises(ValueError, match="selection bounds"):
        FeatureSelectionConfig(min_retained=300, max_retained=600, hard_max_retained=500)


def test_fs04_rejects_missing_target_or_empty_candidates() -> None:
    """타깃 누락/빈 후보는 ValueError 로 fail-closed 됩니다."""
    df = _predictive_df(n_candidates=8, n_groups=30, rows_per_group=8)
    cfg = _small_config()
    with pytest.raises(ValueError, match="target_col"):
        select_features(df, _candidate_cols(8), "missing_target", cfg)
    with pytest.raises(ValueError, match="must not be empty"):
        select_features(df, [], "target_return", cfg)


def test_fs06_median_pairwise_jaccard() -> None:
    """median pairwise Jaccard 안정성 지표가 결정적으로 계산됩니다."""
    assert median_pairwise_jaccard([["a", "b"]]) == 1.0
    assert median_pairwise_jaccard([["a", "b", "c"], ["a", "b", "d"]]) == 0.5
    assert median_pairwise_jaccard([["a"], ["a"], ["a"]]) == 1.0
    assert median_pairwise_jaccard([]) == 1.0


def test_hfs05_selector_treats_inf_as_missing_and_never_passes_inf() -> None:
    """``inf`` 는 결측으로 간주되어 LightGBM/Ridge 에 도달하지 않습니다."""
    rng = np.random.default_rng(21)
    n = 300
    signal = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"target_return": signal + rng.normal(0.0, 0.05, n)})
    df["f_good"] = signal
    df["f_noise"] = rng.normal(0.0, 1.0, n)
    inf_col = rng.normal(0.0, 1.0, n)
    inf_col[::2] = np.inf  # 50% inf -> 결측으로 취급 -> 결측률 0.5 > 0.35 거부
    df["f_inf"] = inf_col

    cfg = FeatureSelectionConfig(min_retained=1, max_retained=10, hard_max_retained=20)
    result = select_features(df, ["f_good", "f_noise", "f_inf"], "target_return", cfg)
    # inf 컬럼은 결측률 > 0.35 로 거부되어야 합니다 (inf 가 NaN 으로 대체된 뒤).
    assert dict(result.rejected)["f_inf"] == "train_missing_rate"
    assert "f_inf" not in result.selected_features

    # Ridge baseline 은 inf 를 NaN 으로 대체하고 정상 동작합니다.
    from src.ml.model_pipeline import _fit_predict_linear_baseline

    train = df.iloc[:200].reset_index(drop=True)
    val = df.iloc[200:].reset_index(drop=True)
    pred = _fit_predict_linear_baseline(train, val, ["f_good", "f_inf"], "target_return")
    assert np.isfinite(pred).all()


def test_hfs05_target_nonfinite_fails_closed() -> None:
    """타깃의 비유한 값은 선택을 fail-closed 로 거부합니다."""
    df = _predictive_df(n_candidates=8, n_groups=30, rows_per_group=8)
    df.loc[df.index[0], "target_return"] = np.inf
    cfg = _small_config()
    with pytest.raises(ValueError, match="target_col contains non-finite"):
        select_features(df, _candidate_cols(8), "target_return", cfg)


def test_hfs06_gpu_auto_fallback_records_reason_and_preserves_cpu_output(monkeypatch) -> None:
    """GPU probe 실패 시 CPU 로 폴백하고 사유를 기록하며 CPU 출력과 동일합니다."""
    from src.ml.feature_selection import _SCREENING_DEVICE_RESULTS

    _SCREENING_DEVICE_RESULTS.pop("auto", None)

    def fake_probe() -> tuple[str, str | None]:
        return "cpu", "gpu_probe_failed: RuntimeError: NVML blocked"

    monkeypatch.setattr("src.ml.feature_selection._probe_gpu_device", fake_probe)
    df = _predictive_df(n_groups=50, rows_per_group=10, seed=8)
    auto_cfg = FeatureSelectionConfig(
        min_retained=5, max_retained=20, hard_max_retained=40, screening_device="auto"
    )
    auto_result = select_features(df, _candidate_cols(40), "target_return", auto_cfg)
    metadata = auto_result.metadata
    assert metadata["requested_screening_device"] == "auto"
    assert metadata["resolved_screening_device"] == "cpu"
    assert "gpu_probe_failed" in metadata["gpu_fallback_reason"]

    cpu_result = select_features(
        df,
        _candidate_cols(40),
        "target_return",
        FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40),
    )
    assert auto_result.selected_features == cpu_result.selected_features
    assert auto_result.gains == cpu_result.gains


def test_hfs06_gpu_probe_failure_returns_cpu_reason(monkeypatch) -> None:
    """실패한 GPU probe 는 CPU + 사유를 결정적으로 반환합니다."""
    from src.ml.feature_selection import _probe_gpu_device

    monkeypatch.setattr(
        "src.ml.feature_selection._probe_gpu_device",
        lambda: ("cpu", "gpu_probe_failed: RuntimeError: NVML blocked"),
    )
    resolved, reason = _probe_gpu_device()
    assert resolved == "cpu"
    assert "gpu_probe_failed" in reason
