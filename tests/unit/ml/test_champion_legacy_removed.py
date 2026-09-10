"""Champion legacy removal contract tests."""
from __future__ import annotations

def test_champion_and_legacy_only_modules_are_fully_removed() -> None:
    import importlib

    import pytest

    for module_name in (
        "src.ml.champion",
        "src.ml.tuning",
        "src.serving.realtime.policy",
        "src.ml.policy_eval",
        "src.ml.feature_selection",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_inference_module_keeps_only_shared_bundle_constants() -> None:
    import src.serving.realtime.inference as inference

    # Then: champion-only prediction/sizing/grading surface is gone
    for gone in (
        "predict_daily_sizing",
        "calculate_utility_score",
        "assign_sizing_grades",
        "apply_risk_limits",
        "_predict_from_bundle",
        "add_close_morning_decision_score",
        "_convex_rank_blend",
        "_validate_algorithm_ensemble_config",
    ):
        assert not hasattr(inference, gone), gone

    # Then: the shared constants bundle.py imports at module load time survive
    for kept in (
        "ROUND_TRIP_COST_RATIO",
        "_QUANTILE_COLS",
        "_QUANTILE_ALPHAS",
        "_STRONG_PCT",
        "_GOOD_PCT",
        "_WEAK_PCT",
        "_GRADE_MULTIPLIERS",
    ):
        assert hasattr(inference, kept), kept


def test_validation_module_keeps_only_cpcv_path_evidence_chain() -> None:
    import src.ml.validation as validation

    for gone in (
        "ValidationConfig",
        "GateOutcome",
        "PromotionDecision",
        "evaluate_locked_oos",
        "temporal_sign_consistency",
        "evaluate_screen_grid",
        "run_promotion_gate",
        "publish_bundle",
    ):
        assert not hasattr(validation, gone), gone

    for kept in (
        "cpcv_path_evidence",
        "paired_t_p_value",
        "minimum_detectable_effect",
    ):
        assert hasattr(validation, kept), kept


def test_universe_module_keeps_only_screen_config_and_panel_builder() -> None:
    import src.ml.universe as universe

    for gone in (
        "SCREEN_REGISTRY",
        "OPERATOR_LEGACY_SCREEN",
        "BAND_2_15_SCREEN",
        "BAND_5_15_HIGHVALUE_SCREEN",
        "apply_screen_mask",
    ):
        assert not hasattr(universe, gone), gone

    for kept in (
        "ScreenConfig",
        "COST_AWARE_SCREEN",
        "build_universe_panel",
        "screen_baseline_stats",
    ):
        assert hasattr(universe, kept), kept


def test_buyability_module_keeps_ceiling_and_liquidity_helpers_only() -> None:
    import src.ml.buyability as buyability

    for gone in (
        "evaluate_buyability_sleeves",
        "summarize_buyability_sleeves",
        "BuyabilitySleeveResult",
        "_sleeve_stats",
    ):
        assert not hasattr(buyability, gone), gone

    for kept in (
        "classify_ceiling_entry",
        "attach_entry_auction_liquidity",
        "estimate_fill_ratio",
        "apply_buyability_gate",
    ):
        assert hasattr(buyability, kept), kept


def test_features_module_keeps_engineer_features_and_ranker_features_only() -> None:
    import src.serving.realtime.features as features

    for gone in ("build_snapshot_features", "add_scenario_features"):
        assert not hasattr(features, gone), gone

    for kept in (
        "engineer_features",
        "_apply_robust_z",
        "build_topk_ranker_features",
        "SCENARIO_ONE_HOT_FEATURES",
        "SCENARIO_CONTEXT_FEATURES",
    ):
        assert hasattr(features, kept), kept


def test_bundle_module_untouched_by_champion_removal() -> None:
    import src.ml.bundle as bundle

    # Then: bundle.py is shared infra (build_inline_bundle backs the reranker's
    # own train_production_bundle) -- nothing here is removed by this contract
    for kept in (
        "build_inline_bundle",
        "CHAMPION_DEFAULT_MODEL_PARAMS",
        "SeedEnsembleModel",
        "fit_seed_ensemble",
        "ChronoCalibratedClassifier",
    ):
        assert hasattr(bundle, kept), kept
