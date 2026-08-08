"""Causal history feature catalogue/panel 단위 테스트.

`docs/specs/ml_feature_selection_pipeline.md` 시나리오:
- FS-01 causal join: same-date EOD 는 제외되고, 공휴일을 넘어 결정일 엄격히 이전의
  마지막 이력 행이 사용됩니다.
- FS-02 no future fill: 첫/누락 종목 key 는 NaN 을 유지하고 절대 이후 값을 사용하지
  않습니다.
- FS-03 catalogue determinism: 합성 판넬에서 정확히 720 개 후보와 반복 가능한
  매니페스트를 산출합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.history_features import (
    HISTORICAL_CATALOGUE,
    HISTORICAL_CATALOGUE_COUNT,
    HISTORICAL_CATALOGUE_VERSION,
    REQUIRED_HISTORY_COLUMNS,
    HistoricalFeatureConfig,
    HistoryFeatureExecutionConfig,
    _sanitize_finite,
    build_catalogue_manifest,
    build_causal_history_feature_panel,
    build_causal_history_feature_panel_from_parquet,
    catalogue_quality_metadata,
)


def make_price_history(
    symbols: list[str] | None = None,
    dates: list[pd.Timestamp] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """완전한 EOD 판넬 컬럼을 갖는 합성 가격 이력을 생성합니다."""
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["000001", "000002", "000003"]
    if dates is None:
        dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 12)]))
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        price = 10_000.0 + rng.uniform(0, 5_000)
        for date in dates:
            o = price
            c = price * (1 + rng.normal(0, 0.01))
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": o,
                    "high": hi,
                    "low": lo,
                    "close": c,
                    "prev_close": price,
                    "market_cap_100m": rng.uniform(100, 5_000),
                    "trade_value_100m": rng.uniform(10, 500),
                    "daily_change_pct": c / price - 1.0,
                    "market": "KOSPI" if int(symbol) % 2 else "KOSDAQ",
                    "volume": rng.uniform(1e4, 1e7),
                    "foreign_netbuy": rng.normal(0, 1e5),
                    "inst_netbuy": rng.normal(0, 1e5),
                    "program_netbuy": rng.normal(0, 5e4),
                    "kospi_pct": rng.normal(0, 0.004),
                    "kosdaq_pct": rng.normal(0, 0.004),
                    "v_kospi": rng.uniform(15, 30),
                    "v_kosdaq": rng.uniform(12, 25),
                }
            )
            price = c
    return pd.DataFrame(rows)


def make_decision_keys(symbols: list[str], dates: list[pd.Timestamp]) -> pd.DataFrame:
    date_list = list(dates)
    return pd.DataFrame(
        {
            "stock_code": [sym for sym in symbols for _ in date_list],
            "trade_date": date_list * len(symbols),
        }
    )


def test_fs01_causal_join_excludes_same_date_and_uses_prior_across_holidays() -> None:
    """결정일과 같은 날짜 EOD 는 제외하고, 공휴일을 건너뛰어 직전 이력 행을 사용합니다."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09"])
    hist = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["000001"] * 4,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "prev_close": [99.0, 100.5, 101.5, 102.5],
            "market_cap_100m": [100.0] * 4,
            "trade_value_100m": [10.0, 20.0, 30.0, 40.0],
            "daily_change_pct": [0.01, 0.01, 0.01, 0.01],
            "market": ["KOSPI"] * 4,
            "volume": [1000.0, 2000.0, 3000.0, 4000.0],
            "foreign_netbuy": [1.0, 2.0, 3.0, 4.0],
            "inst_netbuy": [0.5, 1.0, 1.5, 2.0],
            "program_netbuy": [0.2, 0.4, 0.6, 0.8],
            "kospi_pct": [0.001] * 4,
            "kosdaq_pct": [0.001] * 4,
            "v_kospi": [20.0] * 4,
            "v_kosdaq": [15.0] * 4,
        }
    )
    # 1/3: 같은 날짜 EOD 가 있음 -> 엄격히 이전 1/2 행 사용 (volume 1000).
    # 1/7: 공휴일(무거래) -> 1/3 행 사용 (volume 2000).
    keys = make_decision_keys(["000001"], pd.to_datetime(["2024-01-03", "2024-01-07"]))
    panel = build_causal_history_feature_panel(hist, keys)

    by_date = panel.set_index("trade_date")
    assert abs(float(by_date.loc[pd.Timestamp("2024-01-03"), "log_volume_0"]) - np.log1p(1000.0)) < 1e-5
    assert abs(float(by_date.loc[pd.Timestamp("2024-01-07"), "log_volume_0"]) - np.log1p(2000.0)) < 1e-5
    # 같은 날짜 행을 사용했다면 2000/3000 값이 보였어야 하므로 미래 노출이 없음을 확인합니다.
    assert not np.isclose(by_date.loc[pd.Timestamp("2024-01-03"), "log_volume_0"], np.log1p(2000.0))


def test_fs02_first_or_missing_symbol_keys_remain_nan_never_future() -> None:
    """이력이 없는/첫 이력 이후 충분하지 않은 key 는 NaN 을 유지하며 미래 값을 쓰지 않습니다."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"])
    hist = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["000001"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "prev_close": [99.0, 100.5, 101.5],
            "market_cap_100m": [100.0] * 3,
            "trade_value_100m": [10.0] * 3,
            "daily_change_pct": [0.01] * 3,
            "market": ["KOSPI"] * 3,
            "volume": [1000.0, 2000.0, 3000.0],
            "foreign_netbuy": [1.0] * 3,
            "inst_netbuy": [1.0] * 3,
            "program_netbuy": [1.0] * 3,
            "kospi_pct": [0.001] * 3,
            "kosdaq_pct": [0.001] * 3,
            "v_kospi": [20.0] * 3,
            "v_kosdaq": [15.0] * 3,
        }
    )
    # 1/1: 이력 시작 이전 key -> 전부 NaN. 1/4: 종목 '999999' 는 이력 자체가 없음 -> 전부 NaN.
    keys = pd.DataFrame(
        {
            "stock_code": ["000001", "999999"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
        }
    )
    panel = build_causal_history_feature_panel(hist, keys)
    feature_cols = [c for c in panel.columns if c not in ("stock_code", "trade_date")]
    assert panel[feature_cols].isna().all().all()
    # 1/4('000001'): 1/3 행만 이전 -> rolling(window>=3) 피처는 NaN (미래 채움 없음).
    keys2 = make_decision_keys(["000001"], pd.to_datetime(["2024-01-04"]))
    panel2 = build_causal_history_feature_panel(hist, keys2)
    assert np.isnan(panel2["ma_dist_3"].iloc[0])
    # 미래 값이 새지 않았는지: 1/8 행(volume 3000)이 1/4 결정에 사용되면 안 되고
    # 엄격히 이전 1/3 행(volume 2000)의 값이어야 합니다.
    assert abs(float(panel2["log_volume_0"].iloc[0]) - np.log1p(2000.0)) < 1e-5


def test_fs03_catalogue_exactly_720_and_deterministic() -> None:
    """합성 판넬은 정확히 720 개의 결정적 후보와 반복 가능한 매니페스트를 산출합니다."""
    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 20)]))
    hist = make_price_history(symbols=["000001", "000002", "000003"], dates=dates, seed=3)
    keys = make_decision_keys(["000001", "000002", "000003"], dates[:10])

    panel = build_causal_history_feature_panel(hist, keys)
    expected_features = [str(entry["feature_name"]) for entry in HISTORICAL_CATALOGUE]
    assert len(expected_features) == HISTORICAL_CATALOGUE_COUNT == 720
    assert len(set(expected_features)) == 720
    assert panel.columns.tolist() == ["stock_code", "trade_date", *expected_features]

    panel2 = build_causal_history_feature_panel(hist, keys)
    pd.testing.assert_frame_equal(panel, panel2)

    manifest = build_catalogue_manifest()
    manifest2 = build_catalogue_manifest()
    pd.testing.assert_frame_equal(manifest, manifest2)
    assert len(manifest) == 720
    assert (manifest["availability_rule"] == "prior_eod_available_at_decision_time").all()
    assert (manifest["catalogue_version"] == HISTORICAL_CATALOGUE_VERSION).all()
    assert manifest["feature_name"].tolist() == expected_features


def test_fs03_duplicate_decision_keys_are_deduplicated() -> None:
    """결정 key 중복은 첫 행으로 축소되고 그 뒤 시나리오 행동으로 재조인될 수 있습니다."""
    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 8)]))
    hist = make_price_history(symbols=["000001"], dates=dates, seed=1)
    keys = make_decision_keys(["000001"], [dates[2], dates[2], dates[3]])
    panel = build_causal_history_feature_panel(hist, keys)
    assert len(panel) == 2
    assert panel.duplicated(subset=["stock_code", "trade_date"]).sum() == 0


def test_mto02_feq04_catalogue_quality_metadata_complete() -> None:
    """MTO-02-FEQ-04: 품질 리포트 메타데이터는 720 피처 전체를 카탈로그 순서와
    동일하게, 필수 키를 모두 포함해 반환합니다."""
    metadata = catalogue_quality_metadata()
    assert len(metadata) == HISTORICAL_CATALOGUE_COUNT == 720
    expected_order = [str(entry["feature_name"]) for entry in HISTORICAL_CATALOGUE]
    assert list(metadata) == expected_order
    required_keys = {
        "family",
        "source_column",
        "transform",
        "lookback",
        "availability_rule",
        "panel_scope",
    }
    for entry in metadata.values():
        assert set(entry) == required_keys
        assert entry["family"]
        assert entry["panel_scope"] in {"history_temporal_panel", "decision_candidate_panel"}
        assert entry["availability_rule"] == "prior_eod_available_at_decision_time"


def test_fs01_rejects_missing_required_source_columns() -> None:
    """필수 원천 컬럼이 없으면 fail-closed 로 거부합니다."""
    dates = list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    hist = make_price_history(symbols=["000001"], dates=dates)
    hist = hist.drop(columns=["foreign_netbuy"])
    keys = make_decision_keys(["000001"], [dates[1]])
    with pytest.raises(ValueError, match="required columns"):
        build_causal_history_feature_panel(hist, keys)


def test_fs02_symbols_normalized_to_six_digits() -> None:
    """종목 코드는 6 자리로 정규화되어 이력/결정 key 가 일치합니다."""
    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 8)]))
    hist = make_price_history(symbols=["000001"], dates=dates, seed=2)
    keys = pd.DataFrame(
        {"stock_code": ["1", "000001"], "trade_date": [dates[3], dates[4]]}
    )
    panel = build_causal_history_feature_panel(hist, keys)
    assert panel["log_volume_0"].notna().all()


def test_fs03_build_config_default_contract() -> None:
    """기본 HistoricalFeatureConfig 는 6 자리 종목/결정 key 계약을 고정합니다."""
    config = HistoricalFeatureConfig()
    assert config.history_symbol_col == "symbol"
    assert config.history_date_col == "date"
    assert config.decision_symbol_col == "stock_code"
    assert config.decision_date_col == "trade_date"
    assert config.catalogue_version == HISTORICAL_CATALOGUE_VERSION


def test_hfs01_batch_equivalence_single_vs_many_batches() -> None:
    """한 배치와 여러 배치가 동일한 key/720 값을 산출합니다 (공휴일 merge_asof 포함)."""
    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 20)]))
    # 공휴일(비거래일)을 포함한 결정 날짜.
    decision_dates = pd.to_datetime(["2024-01-03", "2024-01-07", "2024-01-15", "2024-01-19"])
    hist = make_price_history(symbols=["000001", "000002", "000003", "000004", "000005"], dates=dates, seed=9)
    keys = make_decision_keys(["000001", "000002", "000003", "000004", "000005"], decision_dates)

    single = build_causal_history_feature_panel(hist, keys)
    many = build_causal_history_feature_panel(
        hist, keys, execution_config=HistoryFeatureExecutionConfig(symbols_per_batch=2)
    )
    pd.testing.assert_frame_equal(single, many)
    assert len(single.columns) == HISTORICAL_CATALOGUE_COUNT + 2
    # 배치 분할이 지표에만 반영되고 값에는 영향을 주지 않습니다.
    assert many.attrs["history_feature_build_metrics"]["batch_count"] > 1


def test_hfs02_zero_denominator_and_missing_sources_are_nan_not_inf() -> None:
    """분모 0, 고가/저가 0, 누락 원천은 NaN 을 산출하며 무한대가 없습니다."""
    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 12)]))
    hist = make_price_history(symbols=["000001", "000002"], dates=dates, seed=4)
    rng = np.random.default_rng(1)
    hist = hist.copy()
    # 분모/분자 0, 누락 원천을 인위적으로 주입합니다.
    hist.loc[0, "market_cap_100m"] = 0.0  # turnover 분모 0
    hist.loc[1, "prev_close"] = 0.0  # gap_ratio/overnight_return 분모 0
    hist.loc[2, "close"] = 0.0  # 다수 ratio 분자/분모 0
    hist.loc[3, "high"] = 0.0
    hist.loc[4, "low"] = 0.0
    hist.loc[5, "foreign_netbuy"] = np.nan
    hist.loc[6, "trade_value_100m"] = np.nan
    hist.loc[7, "volume"] = np.nan
    keys = make_decision_keys(["000001", "000002"], dates[5:9])
    panel = build_causal_history_feature_panel(hist, keys)
    values = panel.select_dtypes(include="number").to_numpy()
    assert not np.isinf(values).any()
    assert np.isnan(values).any()
    metrics = panel.attrs["history_feature_build_metrics"]
    assert metrics["nonfinite_to_nan_count"] >= 0


def test_hfs03_memory_budget_fails_closed_and_reports_metrics() -> None:
    """낮은 예산은 전체 할당 전에 실패하고, 충분한 예산은 지표를 보고합니다."""
    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 12)]))
    hist = make_price_history(symbols=["000001", "000002", "000003"], dates=dates, seed=5)
    keys = make_decision_keys(["000001", "000002", "000003"], dates[4:7])

    with pytest.raises(ValueError, match="memory budget too small"):
        build_causal_history_feature_panel(
            hist, keys, execution_config=HistoryFeatureExecutionConfig(memory_budget_bytes=1)
        )

    generous = HistoryFeatureExecutionConfig(memory_budget_bytes=10**10)
    panel = build_causal_history_feature_panel(hist, keys, execution_config=generous)
    metrics = panel.attrs["history_feature_build_metrics"]
    for key in (
        "input_history_rows",
        "decision_key_rows",
        "output_rows",
        "batch_count",
        "estimated_bytes_per_source_row",
        "peak_rss_bytes",
        "elapsed_seconds",
        "nonfinite_to_nan_count",
    ):
        assert key in metrics
    assert metrics["peak_rss_bytes"] < 10**10
    assert metrics["output_rows"] == len(panel)


def test_hfs02_sanitize_finite_preserves_alphanumeric_identifiers() -> None:
    """수치 피처만 정제하며, 우선주 등 영문자 포함 종목코드는 유지합니다."""
    frame = pd.DataFrame({"stock_code": ["00499K"], "feature": [float("inf")]})

    sanitized, count = _sanitize_finite(frame)

    assert count == 1
    assert sanitized.loc[0, "stock_code"] == "00499K"
    assert pd.isna(sanitized.loc[0, "feature"])


def test_scenario_history_feature_oom_hardening_01_parquet_batch_rows_is_bounded() -> None:
    """SCENARIO_HISTORY_FEATURE_OOM_HARDENING_01: Arrow scan batch size is validated."""
    assert HistoryFeatureExecutionConfig(parquet_batch_rows=1_000).parquet_batch_rows == 1_000
    with pytest.raises(ValueError, match="parquet_batch_rows"):
        HistoryFeatureExecutionConfig(parquet_batch_rows=0)


def test_scenario_history_feature_oom_hardening_02_budget_preflight_includes_baseline() -> None:
    """SCENARIO_HISTORY_FEATURE_OOM_HARDENING_02: baseline plus batch is fail-closed."""
    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 12)]))
    hist = make_price_history(symbols=["000001", "000002"], dates=dates, seed=8)
    keys = make_decision_keys(["000001", "000002"], dates[4:7])
    with pytest.raises(ValueError, match="memory budget too small"):
        build_causal_history_feature_panel(
            hist,
            keys,
            execution_config=HistoryFeatureExecutionConfig(memory_budget_bytes=1),
        )


def test_hfs03_invalid_execution_config_rejected() -> None:
    """유효하지 않은 실행 설정은 초기화 시 fail-closed 됩니다."""
    with pytest.raises(ValueError, match="symbols_per_batch"):
        HistoryFeatureExecutionConfig(symbols_per_batch=0)
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        HistoryFeatureExecutionConfig(memory_budget_bytes=0)
    with pytest.raises(ValueError, match="required history columns"):
        HistoryFeatureExecutionConfig(parquet_columns=("open", "high"))


def test_hfs04_parquet_reader_matches_dataframe_input_and_projects_columns(
    tmp_path, monkeypatch,
) -> None:
    """경로 reader 는 필수 컬럼만 요청하고 DataFrame 입력과 동일한 출력을 반환합니다."""
    import pyarrow.parquet as pq

    dates = list(pd.to_datetime([f"2024-01-{d:02d}" for d in range(2, 12)]))
    hist = make_price_history(symbols=["000001", "000002", "000003"], dates=dates, seed=6)
    hist = hist.copy()
    hist["extra_secret_column"] = np.arange(len(hist), dtype=np.float64)  # 투영에서 제외되어야 함.
    keys = make_decision_keys(["000001", "000002", "000003"], dates[4:8])
    path = tmp_path / "price_history.parquet"
    hist.to_parquet(path)

    requested_columns: list[list[str]] = []
    original_read = pq.read_table

    def tracking_read_table(*args, **kwargs):  # type: ignore[no-untyped-def]
        if "columns" in kwargs:
            requested_columns.append(list(kwargs["columns"]))
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", tracking_read_table)
    from_parquet = build_causal_history_feature_panel_from_parquet(str(path), keys)
    from_frame = build_causal_history_feature_panel(hist, keys)
    pd.testing.assert_frame_equal(from_parquet, from_frame)
    allowed = {"date", "symbol", *REQUIRED_HISTORY_COLUMNS}
    for columns in requested_columns:
        assert set(columns) <= allowed
        assert "extra_secret_column" not in columns
