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
    FEATURE_QUALITY_VERSION,
    FeatureSelectionConfig,
    build_feature_quality_report,
    build_fold_feature_plans,
    median_pairwise_jaccard,
    permutation_null_stability,
    select_features,
)
from src.ml.history_feature_research import _resolve_research_selection_config
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


def _fwer_config() -> FeatureSelectionConfig:
    return FeatureSelectionConfig(
        selection_rule="permutation_fwer",
        min_retained=1,
        max_retained=20,
        hard_max_retained=40,
        null_alpha=0.05,
        null_permutations=19,
        min_significant_features=1,
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


def test_mto02_fold_plan_invariant_to_outer_labels_and_reused() -> None:
    """MTO-02: 외부 validation/이후 라벨 변경이 fold plan 을 바꾸지 않고, plan 은
    파이프라인에서 두 return 전문가가 재사용할 수 있도록 진단에 그대로 반영됩니다."""
    df = _predictive_df(n_groups=40, rows_per_group=8)
    cfg = _small_config()
    plans = build_fold_feature_plans(
        df, _candidate_cols(40), "target_return", "trade_date", cfg, n_splits=3, purge_gap=1
    )
    assert len(plans) == 3
    assert all(p.selected_features for p in plans)
    assert all(p.config_fingerprint == plans[0].config_fingerprint for p in plans)
    assert all(p.seed == cfg.random_seed for p in plans)
    assert all(p.data_cutoff for p in plans)

    n_groups = df["trade_date"].nunique()
    future_dates = sorted(df["trade_date"].unique())[-(n_groups // (3 + 1)):]
    mutated = df.copy()
    mutated.loc[mutated["trade_date"].isin(future_dates), "target_return"] += 0.5
    plans_mutated = build_fold_feature_plans(
        mutated, _candidate_cols(40), "target_return", "trade_date", cfg, n_splits=3, purge_gap=1
    )
    assert [p.selected_features for p in plans] == [p.selected_features for p in plans_mutated]
    assert [p.gains for p in plans] == [p.gains for p in plans_mutated]
    assert [p.counts for p in plans] == [p.counts for p in plans_mutated]

    # plan 이 파이프라인에서 재계산 없이 재사용됩니다.
    result = run_model_pipeline(
        df,
        _candidate_cols(40),
        "target_return",
        "trade_date",
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
        fold_feature_plans=plans,
    )
    diagnostics = result["feature_selection_diagnostics"]
    assert diagnostics is not None
    assert [list(f["selected_features"]) for f in diagnostics["fold_selections"]] == [
        list(p.selected_features) for p in plans
    ]
    assert diagnostics["config_fingerprint"] == plans[0].config_fingerprint
    assert diagnostics["final_selection_provenance"] == "full_training_selection"
    assert diagnostics["stability"]["n_permutations"] >= 1


def test_mto02_permutation_null_stability_deterministic_gate() -> None:
    """MTO-02: 날짜 블록 순열 null 게이트가 결정적이고, 동일 fold 선택은 통과하며
    서로소 fold 선택은 거부합니다."""
    universe = [chr(ord("a") + i) for i in range(13)]
    identical = permutation_null_stability(
        [["a", "b", "c", "d"], ["a", "b", "c", "d"], ["a", "b", "c", "d"]],
        random_seed=42,
        universe=universe,
    )
    again = permutation_null_stability(
        [["a", "b", "c", "d"], ["a", "b", "c", "d"], ["a", "b", "c", "d"]],
        random_seed=42,
        universe=universe,
    )
    assert identical == again
    assert identical["observed_median_jaccard"] == 1.0
    assert identical["gate_passed"] is True

    disjoint = permutation_null_stability(
        [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]],
        random_seed=42,
        universe=universe,
    )
    assert disjoint["observed_median_jaccard"] == 0.0
    assert disjoint["gate_passed"] is False

    abstain = permutation_null_stability([["a", "b"]], random_seed=42)
    assert abstain["reason"] == "fewer_than_two_folds_abstain"
    assert abstain["gate_passed"] is True

    with pytest.raises(ValueError, match="null_gate_quantile"):
        permutation_null_stability([["a"], ["b"]], random_seed=1, null_gate_quantile=1.5)


def test_mto02_feq01_fold_quality_report_invariant_to_outer_labels() -> None:
    """MTO-02-FEQ-01: 외부 validation/이후 라벨 변경은 fold 품질 기록, 가족 선택
    빈도, 가족 액션을 바꾸지 않습니다."""
    df = _predictive_df(n_groups=40, rows_per_group=8)
    cfg = _small_config()
    plans = build_fold_feature_plans(
        df, _candidate_cols(40), "target_return", "trade_date", cfg, n_splits=3, purge_gap=1
    )

    n_groups = df["trade_date"].nunique()
    future_dates = sorted(df["trade_date"].unique())[-(n_groups // (3 + 1)):]
    mutated = df.copy()
    mutated.loc[mutated["trade_date"].isin(future_dates), "target_return"] += 0.5
    plans_mutated = build_fold_feature_plans(
        mutated, _candidate_cols(40), "target_return", "trade_date", cfg, n_splits=3, purge_gap=1
    )

    def _report(ps: list) -> dict:
        return build_feature_quality_report(
            [p.selection for p in ps],
            _candidate_cols(40),
            {},
            cfg.min_fold_selection_rate,
        )

    report = _report(plans)
    report_mutated = _report(plans_mutated)
    assert report == report_mutated
    assert report["version"] == FEATURE_QUALITY_VERSION
    assert report["n_folds"] == 3
    # 결정적 정렬: 피처는 이름, 가족은 이름 오름차순.
    assert [r["feature_name"] for r in report["feature_records"]] == sorted(
        r["feature_name"] for r in report["feature_records"]
    )
    assert [f["family"] for f in report["family_records"]] == sorted(
        f["family"] for f in report["family_records"]
    )
    for record in report["feature_records"]:
        assert record["selection_rate"] == record["selection_count"] / 3
    # 각 fold 계획은 train 전용 품질 증거(유한 비율/결측률/영분산)를 보관합니다.
    for plan in plans:
        assert set(plan.selection.quality_evidence) <= set(_candidate_cols(40))
        for evidence in plan.selection.quality_evidence.values():
            assert "finite_ratio" in evidence
            assert "missing_rate" in evidence
            assert "zero_variance" in evidence

    # 모델 파이프라인 진단에도 결정적 품질 리포트가 반영됩니다.
    result = run_model_pipeline(
        df,
        _candidate_cols(40),
        "target_return",
        "trade_date",
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
        feature_selection_config=cfg,
    )
    diagnostics = result["feature_selection_diagnostics"]
    assert diagnostics["quality_report"]["version"] == FEATURE_QUALITY_VERSION
    assert diagnostics["quality_report"]["n_folds"] == 3


def test_mto02_feq03_source_incomplete_and_capacity_limited() -> None:
    """MTO-02-FEQ-03: 전부 비유한 히스토리 피처는 source_incomplete 로,
    beyond_max_retained 피처는 low_quality 가 아닌 capacity_limited 로 분류됩니다."""
    rng = np.random.default_rng(31)
    n = 300
    s1 = rng.normal(0.0, 1.0, n)
    s2 = rng.normal(0.0, 1.0, n)
    s3 = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "target_return": 0.05 * (s1 + s2 + s3) + rng.normal(0.0, 0.01, n),
            "f_signal_a": s1,
            "f_signal_b": s2,
            "f_signal_c": s3,
            "f_unavailable": np.nan,
        }
    )
    cols = ["f_signal_a", "f_signal_b", "f_signal_c", "f_unavailable"]
    cfg = FeatureSelectionConfig(min_retained=1, max_retained=1, hard_max_retained=20)
    result = select_features(df, cols, "target_return", cfg)
    reasons = dict(result.rejected)
    assert reasons["f_unavailable"] == "all_nonfinite"
    beyond = {feature for feature, reason in reasons.items() if reason == "beyond_max_retained"}
    assert len(beyond) == 2
    assert beyond <= {"f_signal_a", "f_signal_b", "f_signal_c"}

    catalogue = {
        "f_signal_a": {"family": "return_trend_mean_reversion"},
        "f_signal_b": {"family": "ohlc_range_gap"},
        "f_signal_c": {"family": "liquidity_size_turnover"},
        "f_unavailable": {"family": "market_regime_context"},
    }
    report = build_feature_quality_report([result], cols, catalogue, 1.0)
    actions = report["actions"]
    assert "source_incomplete" in actions["f_unavailable"]
    for feature in beyond:
        assert "capacity_limited" in actions[feature]
        assert "source_incomplete" not in actions[feature]
    assert "screening_weak" not in actions["f_unavailable"]

    by_name = {r["feature_name"]: r for r in report["feature_records"]}
    assert by_name["f_unavailable"]["family"] == "market_regime_context"
    for feature in beyond:
        assert by_name[feature]["family"] != "baseline"
    assert by_name["f_signal_a"]["finite_ratio"] == 1.0
    assert by_name["f_unavailable"]["finite_ratio"] == 0.0

    # 카탈로그 누락 family 는 fail-closed 로 거부됩니다.
    with pytest.raises(ValueError, match="family"):
        build_feature_quality_report(
            [result], cols, {"f_unavailable": {}}, 1.0
        )
    with pytest.raises(ValueError, match="min_fold_selection_rate"):
        build_feature_quality_report([result], cols, catalogue, 0.0)

def test_fs04_permutation_fwer_fold_plans_invariant_to_validation_and_later_labels() -> None:
    """FS-04: permutation-FWER fold 계획은 validation/이후 라벨과 무관하게 불변입니다."""
    df = _predictive_df(n_groups=40, rows_per_group=8)
    cfg = _fwer_config()
    cols = _candidate_cols(40)
    plans = build_fold_feature_plans(
        df, cols, "target_return", "trade_date", cfg, n_splits=3, purge_gap=1
    )
    assert all(p.selected_features for p in plans)
    assert all(p.metadata["provenance"] == "fold_train_selection" for p in plans)

    n_groups = df["trade_date"].nunique()
    future_dates = sorted(df["trade_date"].unique())[-(n_groups // (3 + 1)):]
    mutated = df.copy()
    mutated.loc[mutated["trade_date"].isin(future_dates), "target_return"] += 0.5
    plans_mutated = build_fold_feature_plans(
        mutated, cols, "target_return", "trade_date", cfg, n_splits=3, purge_gap=1
    )
    assert [p.selected_features for p in plans] == [p.selected_features for p in plans_mutated]
    assert [p.metadata["selection_threshold"] for p in plans] == [
        p.metadata["selection_threshold"] for p in plans_mutated
    ]
    assert [p.metadata["null_max_gain_max"] for p in plans] == [
        p.metadata["null_max_gain_max"] for p in plans_mutated
    ]
    assert [p.counts for p in plans] == [p.counts for p in plans_mutated]


def test_fs05_permutation_fwer_threshold_rejection_and_no_cap_truncation() -> None:
    """FS-05: 임계값 위 피처는 유지되고, 이하 피처는 not_significant_vs_null 로
    거부되며, max_retained 는 결과를 조용히 잘라내지 않습니다."""
    rng = np.random.default_rng(7)
    n_groups, rows_per_group = 40, 8
    n = n_groups * rows_per_group
    dates = pd.to_datetime([f"2024-03-{1 + d % 28:02d}" for d in range(n_groups)])
    df = pd.DataFrame({"trade_date": [d for d in dates for _ in range(rows_per_group)]})
    signals = [rng.normal(0.0, 1.0, n) for _ in range(4)]
    for j, signal in enumerate(signals):
        df[f"f{j:03d}"] = signal
    for j in range(4, 10):
        df[f"f{j:03d}"] = rng.normal(0.0, 1.0, n)
    df["target_return"] = 0.08 * sum(signals) + rng.normal(0.0, 0.01, n)

    cfg = FeatureSelectionConfig(
        selection_rule="permutation_fwer",
        min_retained=1,
        max_retained=2,
        hard_max_retained=20,
        null_alpha=0.05,
        null_permutations=19,
        min_significant_features=1,
    )
    result = select_features(df, [f"f{j:03d}" for j in range(10)], "target_return", cfg, group_col="trade_date")
    threshold = result.metadata["selection_threshold"]
    reasons = dict(result.rejected)
    # 임계값을 엄격히 초과하는 피처만 유지됩니다.
    assert all(gain > threshold for _name, gain in result.gains)
    # 이하 피처는 명시적 screening-weak 거부 사유를 받습니다.
    not_significant = {
        feature for feature, reason in reasons.items() if reason == "not_significant_vs_null"
    }
    assert len(not_significant) == 6
    # max_retained 는 유지 수를 잘라내지 않습니다.
    assert result.counts["n_retained"] > cfg.max_retained
    assert result.counts["n_retained"] == result.counts["n_post_correlation"]


def test_fs05_permutation_fwer_rejects_zero_significant_features() -> None:
    """FS-05: 유의 피처가 없으면 permutation_fwer 는 fail-closed 로 거부합니다."""
    rng = np.random.default_rng(9)
    n_groups, rows_per_group = 20, 6
    n = n_groups * rows_per_group
    dates = pd.to_datetime([f"2024-03-{1 + d % 28:02d}" for d in range(n_groups)])
    df = pd.DataFrame({"trade_date": [d for d in dates for _ in range(rows_per_group)]})
    for j in range(6):
        df[f"f{j:03d}"] = rng.normal(0.0, 1.0, n)
    df["target_return"] = 0.0
    with pytest.raises(ValueError, match="retained feature count"):
        select_features(df, _candidate_cols(6), "target_return", _fwer_config(), group_col="trade_date")


def test_fs05_permutation_fwer_fails_closed_without_group_col() -> None:
    """FS-05: permutation_fwer 는 group_col 없이 호출되면 fail-closed 로 거부합니다."""
    df = _predictive_df(n_candidates=12, n_groups=30, rows_per_group=8)
    with pytest.raises(ValueError, match="group_col"):
        select_features(df, _candidate_cols(12), "target_return", _fwer_config())


def test_hfs05_staggered_missingness_pairwise_spearman_prunes_not_false_positive() -> None:
    """FS-05/HFS-05: 결측이 엇갈린(도약) 쌍이 공통 관측에서 0.98 을 넘으면
    가지치기되고, 겹침이 부족한 쌍은 절대 거짓 양성으로 가지치기되지 않습니다."""
    rng = np.random.default_rng(21)
    n = 300
    signal = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"target_return": signal + rng.normal(0.0, 0.05, n)})
    df["f_signal"] = signal
    dup = signal.copy()
    dup_mask = rng.random(n) < 0.4
    signal_mask = rng.random(n) < 0.4
    dup[dup_mask & ~signal_mask] = np.nan
    df["f_dup"] = dup
    low_overlap = rng.normal(0.0, 1.0, n)
    low_overlap[df["f_dup"].isna()] = np.nan
    df["f_low_overlap"] = low_overlap

    cfg = FeatureSelectionConfig(min_retained=1, max_retained=10, hard_max_retained=20)
    result = select_features(
        df, ["f_signal", "f_dup", "f_low_overlap"], "target_return", cfg
    )
    reasons = dict(result.rejected)
    # 공통 관측에서 동일한 f_dup 은 상관 > 0.98 로 가지치기됩니다.
    assert reasons["f_dup"] == "correlated_pair_pruned"
    # 겹침이 부족한 피처는 correlated 로 거부되지 않습니다 (부족하면 상관 0).
    assert reasons.get("f_low_overlap") != "correlated_pair_pruned"


def test_mto02_permutation_fwer_diagnostics_expose_rule_threshold_null_and_provenance() -> None:
    """MTO-02: 진단이 rule, 임계값, null 요약, 선택 카운트와 fold-local/full-training
    선택 출처를 노출합니다."""
    df = _predictive_df(n_groups=40, rows_per_group=8)
    cfg = _fwer_config()
    result = run_model_pipeline(
        df,
        _candidate_cols(40),
        "target_return",
        "trade_date",
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
        feature_selection_config=cfg,
    )
    diagnostics = result["feature_selection_diagnostics"]
    assert diagnostics["final_selection_provenance"] == "full_training_selection"
    assert diagnostics["final_selection"]["metadata"]["provenance"] == "full_training_selection"
    final_meta = diagnostics["final_selection"]["metadata"]
    assert final_meta["selection_rule"] == "permutation_fwer"
    assert final_meta["selection_threshold"] is not None
    assert final_meta["null_alpha"] == 0.05
    assert final_meta["null_permutations"] == 19
    assert final_meta["null_max_gain_min"] is not None
    assert final_meta["null_max_gain_median"] is not None
    assert final_meta["null_max_gain_max"] is not None
    assert (
        final_meta["null_max_gain_min"]
        <= final_meta["null_max_gain_median"]
        <= final_meta["null_max_gain_max"]
    )
    for fold in diagnostics["fold_selections"]:
        assert fold["metadata"]["provenance"] == "fold_train_selection"
        assert fold["metadata"]["selection_rule"] == "permutation_fwer"
        assert fold["metadata"]["selection_threshold"] is not None
        assert fold["counts"]["n_pre_correlation"] >= fold["counts"]["n_post_correlation"]
        assert fold["counts"]["n_post_correlation"] == fold["counts"]["n_retained"]


def test_fs07_permutation_fwer_config_validation() -> None:
    """FS-07: permutation_fwer 설정 범위 위반은 초기화 시 fail-closed 로 거부됩니다."""
    with pytest.raises(ValueError, match="null_alpha"):
        FeatureSelectionConfig(null_alpha=0.0)
    with pytest.raises(ValueError, match="null_permutations"):
        FeatureSelectionConfig(null_alpha=0.05, null_permutations=10)
    with pytest.raises(ValueError, match="min_significant_features"):
        FeatureSelectionConfig(min_significant_features=0)
    with pytest.raises(ValueError, match="min_pairwise_observations"):
        FeatureSelectionConfig(min_pairwise_observations=2)
    with pytest.raises(ValueError, match="requires screening_device"):
        FeatureSelectionConfig(selection_rule="permutation_fwer", screening_device="gpu")


def test_fs10_research_default_selection_rule_is_permutation_fwer() -> None:
    """FS-10: 리서치 설정을 생략하면 permutation_fwer 기본값, 명시 설정은 유지됩니다."""
    default = _resolve_research_selection_config(None)
    assert default.selection_rule == "permutation_fwer"
    legacy = FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40)
    assert _resolve_research_selection_config(legacy) is legacy
    assert legacy.selection_rule == "fixed_cap"

def test_mto02_not_significant_vs_null_maps_to_screening_weak_in_quality_report() -> None:
    """MTO-02: not_significant_vs_null 거부는 capacity_limited 가 아닌 screening_weak
    액션으로 품질 리포트에 매핑됩니다."""
    rng = np.random.default_rng(7)
    n_groups, rows_per_group = 40, 8
    n = n_groups * rows_per_group
    dates = pd.to_datetime([f"2024-03-{1 + d % 28:02d}" for d in range(n_groups)])
    df = pd.DataFrame({"trade_date": [d for d in dates for _ in range(rows_per_group)]})
    df["f_signal"] = rng.normal(0.0, 1.0, n)
    for j in range(4, 10):
        df[f"f{j:03d}"] = rng.normal(0.0, 1.0, n)
    df["target_return"] = 0.08 * df["f_signal"] + rng.normal(0.0, 0.01, n)

    cfg = _fwer_config()
    cols = ["f_signal", *[f"f{j:03d}" for j in range(4, 10)]]
    result = select_features(df, cols, "target_return", cfg, group_col="trade_date")
    reasons = dict(result.rejected)
    rejected_weak = {
        feature for feature, reason in reasons.items() if reason == "not_significant_vs_null"
    }
    assert rejected_weak

    report = build_feature_quality_report([result], cols, {}, cfg.min_fold_selection_rate)
    actions = report["actions"]
    for feature in rejected_weak:
        assert "screening_weak" in actions[feature]
        assert "capacity_limited" not in actions[feature]
