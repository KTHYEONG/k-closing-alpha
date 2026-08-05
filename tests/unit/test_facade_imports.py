"""4개 호환성 퍼사드 경로의 공개 심볼 import 유지 검증.

리팩토링 기간 동안 ``src.api.kis_client``, ``src.backfill.backfill_price``,
``src.daily.predict``, ``src.ml.model_pipeline`` 는 공개 심볼을 그대로
노출해야 합니다.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backfill.backfill_price import fetch_one_symbol as facade_fetch_one_symbol
import src.api.kis.client as kis_client_impl
import src.api.kis.indicators as kis_indicators_impl
import src.api.kis.rate_limit as kis_rate_limit_impl
import src.backfill.price.config as backfill_price_config_impl
import src.backfill.price.factors as backfill_price_factors_impl
import src.backfill.price.normalize as backfill_price_normalize_impl
import src.backfill.price.runner as backfill_price_runner_impl
import src.backfill.price.sources as backfill_price_sources_impl
import src.backfill.price.universe as backfill_price_universe_impl
import src.daily.model_bundle_service as model_bundle_service_impl
import src.daily.prediction_service as prediction_service_impl
import src.ml.training.experiments as training_experiments_impl
import src.ml.training.fitting as training_fitting_impl
import src.ml.training.pipelines as training_pipelines_impl
import src.ml.training.policy_calibration as training_policy_impl
import src.ml.training.validation as training_validation_impl

_CANONICAL_MODULES = (
    kis_client_impl,
    kis_indicators_impl,
    kis_rate_limit_impl,
    backfill_price_config_impl,
    backfill_price_factors_impl,
    backfill_price_normalize_impl,
    backfill_price_runner_impl,
    backfill_price_sources_impl,
    backfill_price_universe_impl,
    model_bundle_service_impl,
    prediction_service_impl,
    training_experiments_impl,
    training_fitting_impl,
    training_pipelines_impl,
    training_policy_impl,
    training_validation_impl,
)


def test_kis_client_facade_exposes_public_symbols() -> None:
    mod = importlib.import_module("src.api.kis_client")
    assert callable(mod.KisApiClient)
    assert callable(mod.AsyncRateLimiter)
    assert callable(mod.calculate_all_moving_averages)
    assert callable(mod.fetch_index_and_calculate_volatility)
    assert callable(mod.prefetch_ohlcv_for_sma120)
    assert callable(mod.calculate_stock_sma)
    assert callable(mod.calculate_stock_ema)
    assert callable(mod.calculate_multiple_emas)
    assert callable(mod.fetch_kospi200_and_calculate_vkospi)


def test_backfill_price_facade_exposes_public_symbols() -> None:
    mod = importlib.import_module("src.backfill.backfill_price")
    assert callable(facade_fetch_one_symbol)
    assert mod.fetch_one_symbol is facade_fetch_one_symbol
    assert callable(mod.run_backfill)
    assert callable(mod.preview_windows)
    assert callable(mod.main)
    assert mod.FetchConfig is not None
    assert mod.DEFAULT_CONFIG is not None


def test_backfill_price_facade_dry_run_logging(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    mod = importlib.import_module("src.backfill.backfill_price")
    args = SimpleNamespace(
        dry_run=True,
        parquet_out=str(tmp_path / "history.parquet"),
        symbols="",
        lookback_days=40,
        limit_symbols=None,
        workers=1,
        kis_rest_rps=20.0,
        kis_safety_ratio=0.6,
        kis_max_parallel=1,
    )
    monkeypatch.setattr(mod, "_parse_args", lambda: args)
    monkeypatch.setattr(mod, "preview_windows", lambda **_: pd.DataFrame())
    mod.main()

    monkeypatch.setattr(mod, "preview_windows", lambda **_: pd.DataFrame({"symbol": ["005930"]}))
    mod.main()


def test_daily_predict_facade_exposes_public_symbols() -> None:
    mod = importlib.import_module("src.daily.predict")
    assert callable(mod.run_daily_sizing_inference)
    assert callable(mod.apply_standard_feature_engineering)
    assert callable(mod.ensure_valid_model_bundle)
    assert callable(mod.train_and_save_real_model_bundle)
    assert callable(mod.build_result_rows)
    assert callable(mod.select_top_actionable)
    assert callable(mod.main)


def test_model_pipeline_facade_exposes_public_symbols() -> None:
    mod = importlib.import_module("src.ml.model_pipeline")
    assert callable(mod.run_model_pipeline)
    assert callable(mod.run_sizing_pipeline)
    assert callable(mod.calculate_recency_sample_weight)
    assert callable(mod.evaluate_close_morning_quality)
    assert callable(mod.run_close_morning_reranker_v2_experiment)
    assert callable(mod.run_close_morning_recency_ensemble_experiment)
    assert callable(mod._calibrate_oof_policy)


def test_new_submodules_are_importable() -> None:
    """리팩토링으로 도입된 새 하위 모듈이 정상 import 되는지 검증합니다."""
    new_modules = [
        "src.api.kis.client",
        "src.api.kis.rate_limit",
        "src.api.kis.indicators",
        "src.daily.prediction_service",
        "src.daily.model_bundle_service",
        "src.ml.training.validation",
        "src.ml.training.fitting",
        "src.ml.training.policy_calibration",
        "src.ml.training.experiments",
        "src.ml.training.pipelines",
        "src.backfill.price.config",
        "src.backfill.price.universe",
        "src.backfill.price.sources",
        "src.backfill.price.normalize",
        "src.backfill.price.factors",
        "src.backfill.price.runner",
    ]
    for module_name in new_modules:
        assert importlib.import_module(module_name) is not None


def test_ml_and_processing_do_not_import_outer_facades() -> None:
    """ml/processing 은 daily/backfill/sync/CLI 퍼사드로 import 하지 않습니다.

    의존성은 CLI/facade -> 서비스 -> 도메인/처리 -> 어댑터 방향으로만
    허용됩니다 (인바운드 의존성 규칙).
    """
    forbidden = ("src.daily", "src.backfill", "src.sync", "src.api.kis_client")
    for module_name in ("src.ml.training.pipelines", "src.ml.training.fitting", "src.processing.preprocessor"):
        module = importlib.import_module(module_name)
        source = __import__("inspect").getsourcelines(module)[0]
        for line in source:
            stripped = line.strip()
            if stripped.startswith("from") or stripped.startswith("import"):
                assert not any(stripped.startswith(f"{token} ") or stripped.startswith(f"from {token}") for token in forbidden), (
                    f"{module_name} violates inward dependency rule: {stripped}"
                )
