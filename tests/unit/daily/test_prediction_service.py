"""당일 예측 서비스의 선택 인지 번들 서빙 단위 테스트.

SCENARIO_SELECTION_BUNDLE_SERVING_01: 선택 인지 번들은 매니페스트와 선별 컬럼을
라운드트립하고, 선별된 available 피처가 없으면 서빙이 ``ValueError`` 로
fail-closed 합니다. 레거시 번들은 기존 0-fill 호환 동작을 유지합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.daily.prediction_service import (
    apply_standard_feature_engineering,
    run_daily_sizing_inference,
)
from src.ml.feature_manifest import build_feature_manifest
from src.ml.sizing_engine import _train_inline_bundle


def _daily_raw_snapshot(n_stocks: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows: list[dict[str, object]] = []
    for j in range(n_stocks):
        base = 50000.0 + j * 500
        rows.append(
            {
                "시나리오": "거래량 폭증",
                "종목명": f"종목{j}",
                "종목코드": f"{100000 + j:06d}",
                "시가": base,
                "고가": base * 1.02,
                "저가": base * 0.98,
                "종가": base * 1.01,
                "전일종가": base,
                "시가총액": 1000.0 + j * 100,
                "거래대금": 100.0 + j * 20,
                "등락률": float(rng.normal(0.5, 2.0)),
                "선정순위": float(j + 1),
                "기관_순매수": float((j - 2) * 5e7),
                "외국인_순매수": float(j * 3e7),
                "프로그램_순매수": float((j - 1) * 1e7),
                "체결강도": 110.0 + j,
                "시장구분": "KOSPI" if j % 2 == 0 else "KOSDAQ",
                "총_종목수": float(n_stocks),
                "평균_거래대금": 90.0 + j,
                "kospi": 0.3,
                "kosdaq": 0.1,
                "v_kospi": 15.0,
                "v_kosdaq": 18.0,
                "거래량": 1_000_000 + j * 1000,
                "테마_섹터": f"theme{j % 3}",
                "차트분석": "신고가 근접",
            }
        )
    return pd.DataFrame(rows)


def _daily_price_history(n_symbols: int = 10, n_dates: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="D")
    rows: list[dict[str, object]] = []
    for s in range(n_symbols):
        symbol = f"{100000 + s:06d}"
        close = 50000.0 + s * 1000
        for date in dates:
            close = close * (1 + rng.normal(0, 0.02))
            open_ = close * (1 + rng.normal(0, 0.005))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            volume = abs(rng.normal(1_000_000, 200_000))
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "prev_close": close / (1 + rng.normal(0, 0.02)),
                    "volume": volume,
                    "trade_value_100m": volume * close / 100_000_000,
                    "market_cap_100m": close * 1_000_000 / 100_000_000,
                    "daily_change_pct": rng.normal(0.0, 2.0),
                    "market": "KOSPI",
                }
            )
    return pd.DataFrame(rows)


def _selection_aware_bundle() -> dict:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "date": (np.arange(n) // 10).astype(int),
            "f_a": rng.normal(size=n),
            "f_b": rng.normal(size=n),
            "target_return": rng.normal(0, 0.02, size=n),
        }
    )
    bundle = _train_inline_bundle(df, ["f_a", "f_b"], "target_return", "date")
    bundle["feature_selection_version"] = "fold_local_v1"
    bundle["feature_manifest"] = build_feature_manifest(["f_a", "f_b"])
    return bundle


def test_apply_standard_feature_engineering_reproduces_catalog_with_history() -> None:
    """당일 스냅샷 + prior 이력으로 카탈로그가 재현되고 매니페스트가 기록됩니다."""
    work = apply_standard_feature_engineering(_daily_raw_snapshot(), _daily_price_history())
    assert "snap_log_market_cap_100m" in work.columns
    assert "hist_ret_5d" in work.columns
    assert work.attrs["catalog_version"] == "causal_expanded_v1"
    assert work.attrs["catalog_hash"]
    manifest = work.attrs["feature_manifest"]
    assert {"family", "source_columns", "lookback_groups", "availability_rule"}.issubset(
        set(manifest.columns)
    )


def test_selection_aware_bundle_serves_when_all_selected_features_present() -> None:
    """SCENARIO_SELECTION_BUNDLE_SERVING_01: 선별 피처가 모두 있으면 서빙이 동작합니다."""
    bundle = _selection_aware_bundle()
    rng = np.random.default_rng(1)
    serving = pd.DataFrame(
        {
            "date": [str(d) for d in range(20)],
            "f_a": rng.normal(size=20),
            "f_b": rng.normal(size=20),
        }
    )
    out = run_daily_sizing_inference(serving, bundle)
    assert "utility_score" in out.columns
    assert "grade" in out.columns


def test_selection_aware_serving_raises_on_missing_selected_feature() -> None:
    """SCENARIO_SELECTION_BUNDLE_SERVING_01: 선별 피처 누락 시 ValueError."""
    bundle = _selection_aware_bundle()
    rng = np.random.default_rng(1)
    serving = pd.DataFrame({"date": [str(d) for d in range(20)], "f_a": rng.normal(size=20)})
    with pytest.raises(ValueError, match="missing selected features"):
        run_daily_sizing_inference(serving, bundle)


def test_selection_aware_serving_raises_on_non_finite_decision_time_feature() -> None:
    """선택 인지 번들은 결정 시점 available 피처의 비유한 값을 거부합니다."""
    bundle = _selection_aware_bundle()
    rng = np.random.default_rng(1)
    serving = pd.DataFrame(
        {
            "date": [str(d) for d in range(20)],
            "f_a": rng.normal(size=20),
            "f_b": rng.normal(size=20),
        }
    )
    serving.loc[0, "f_a"] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        run_daily_sizing_inference(serving, bundle)


def test_legacy_bundle_keeps_zero_fill_behavior() -> None:
    """레거시 번들(feature_selection_version 없음)은 기존 0-fill 동작을 유지합니다."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "date": (np.arange(n) // 10).astype(int),
            "f_a": rng.normal(size=n),
            "f_b": rng.normal(size=n),
            "target_return": rng.normal(0, 0.02, size=n),
        }
    )
    bundle = _train_inline_bundle(df, ["f_a", "f_b"], "target_return", "date")
    assert "feature_selection_version" not in bundle
    serving = pd.DataFrame({"date": [str(d) for d in range(10)], "f_a": rng.normal(size=10)})
    out = run_daily_sizing_inference(serving, bundle)
    assert out["f_b"].eq(0.0).all()
