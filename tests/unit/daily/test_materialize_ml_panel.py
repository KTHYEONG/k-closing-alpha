from __future__ import annotations


def test_materialize_ml_panel_writes_new_artifact_and_logs_provenance(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.daily import materialize_ml_panel as mod

    # Arrange
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame(
        {
            "매수날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "(종가)": [100.0],
            "(수익률, %)": [1.0],
            "(매수 가격)": [100.0],
            "(매도 가격)": [101.0],
        }
    ).to_parquet(trade_path)
    price_path = tmp_path / "price_history.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-03-03"]),
            "symbol": ["005930", "005930"],
            "open": [100.0, 102.0],
            "close": [100.0, 102.0],
            "high": [101.0, 103.0],
            "low": [99.0, 101.0],
            "prev_close": [99.0, 100.0],
            "volume": [1000.0, 1000.0],
            "market_cap_100m": [900.0, 900.0],
            "trade_value_100m": [300.0, 300.0],
        }
    ).to_parquet(price_path)
    out_path = tmp_path / "ml_training_panel.parquet"

    def _fake_restore(df, price_history_df, *, condition_history_path=None, theme_df=None, offset_min_rows=500, **kw):
        out = df.copy()
        out.attrs["panel_restoration"] = {
            "execution_offset_pct": -0.3,
            "restored_rows": 0,
            "restored_dates": 0,
            "restored_date_min": "",
            "restored_date_max": "",
        }
        return out

    monkeypatch.setattr(mod, "build_restored_trade_log", _fake_restore)

    with caplog.at_level(logging.INFO, logger="src.daily.materialize_ml_panel"):
        mod.main(
            [
                "--trade-log", str(trade_path),
                "--price-history", str(price_path),
                "--out", str(out_path),
            ]
        )

    # Assert: new artifact written, trade-log path untouched, provenance logged.
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert len(written) == 1
    assert "[DATA] stage=ml_panel_materialize" in caplog.text
    assert "restored_rows=0" in caplog.text


def test_materialize_ml_panel_warns_on_stale_price_history_without_failing(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.daily import materialize_ml_panel as mod

    # Arrange: price_history frozen months behind today.
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame(
        {
            "매수날짜": ["2025-12-01"],
            "종목코드": ["005930"],
            "(종가)": [100.0],
            "(수익률, %)": [1.0],
            "(매수 가격)": [100.0],
            "(매도 가격)": [101.0],
        }
    ).to_parquet(trade_path)
    price_path = tmp_path / "price_history.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-01"]),
            "symbol": ["005930"],
            "open": [100.0],
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
            "prev_close": [99.0],
            "volume": [1000.0],
            "market_cap_100m": [900.0],
            "trade_value_100m": [300.0],
        }
    ).to_parquet(price_path)
    out_path = tmp_path / "ml_training_panel.parquet"

    def _fake_restore(df, price_history_df, *, condition_history_path=None, theme_df=None, offset_min_rows=500, **kw):
        out = df.copy()
        out.attrs["panel_restoration"] = {
            "execution_offset_pct": 0.0,
            "restored_rows": 0,
            "restored_dates": 0,
            "restored_date_min": "",
            "restored_date_max": "",
        }
        return out

    monkeypatch.setattr(mod, "build_restored_trade_log", _fake_restore)

    with caplog.at_level(logging.WARNING, logger="src.daily.materialize_ml_panel"):
        mod.main(["--trade-log", str(trade_path), "--price-history", str(price_path), "--out", str(out_path)])

    # Assert: warned, but did not raise, and still wrote the artifact.
    assert "stage=ml_panel_freshness" in caplog.text
    assert "status=stale" in caplog.text
    assert out_path.exists()


def test_materialize_ml_panel_preserves_percent_formatted_executed_returns(tmp_path, monkeypatch) -> None:
    """A legacy sheet return like '5.95%' must survive materialization, not become NaN."""
    import pandas as pd

    from src.daily import materialize_ml_panel as mod

    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame(
        {
            "매수날짜": ["2025-12-01", "2025-12-02"],
            "종목코드": ["005930", "000660"],
            "(종가)": [100.0, 200.0],
            "(수익률, %)": ["5.95%", "-1.96%"],
            "(매수 가격)": [100.0, 200.0],
            "(매도 가격)": [105.95, 196.08],
        }
    ).to_parquet(trade_path)
    price_path = tmp_path / "price_history.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-01", "2025-12-02"]),
            "symbol": ["005930", "000660"],
            "open": [100.0, 200.0],
            "close": [100.0, 200.0],
            "high": [101.0, 202.0],
            "low": [99.0, 198.0],
            "prev_close": [99.0, 198.0],
            "volume": [1000.0, 1000.0],
            "market_cap_100m": [900.0, 900.0],
            "trade_value_100m": [300.0, 300.0],
        }
    ).to_parquet(price_path)
    out_path = tmp_path / "ml_training_panel.parquet"

    def _fake_restore(df, price_history_df, *, condition_history_path=None, theme_df=None, offset_min_rows=500, **kw):
        out = df.copy()
        out.attrs["panel_restoration"] = {
            "execution_offset_pct": 0.0,
            "restored_rows": 0,
            "restored_dates": 0,
            "restored_date_min": "",
            "restored_date_max": "",
        }
        return out

    monkeypatch.setattr(mod, "build_restored_trade_log", _fake_restore)

    mod.main(["--trade-log", str(trade_path), "--price-history", str(price_path), "--out", str(out_path)])

    written = pd.read_parquet(out_path)
    assert written["(수익률, %)"].isna().sum() == 0
    assert sorted(written["(수익률, %)"].tolist()) == [-1.96, 5.95]


def test_materialize_ml_panel_prepares_price_panel(tmp_path, monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.daily.materialize_ml_panel as mod

    # Given: a percent-encoded price_history parquet and a minimal trade log
    dates = pd.bdate_range("2023-02-01", periods=4)
    rows = []
    px = 10000.0
    for d in dates:
        nxt = px * 1.08
        rows.append({
            "date": d, "symbol": "000001", "open": px, "high": nxt * 1.01, "low": px * 0.99,
            "close": nxt, "prev_close": px, "market_cap_100m": 900.0,
            "trade_value_100m": 300.0, "daily_change_pct": 8.0, "market": "KOSPI",
            "volume": 1e5, "foreign_netbuy": 0.0, "inst_netbuy": 0.0, "program_netbuy": 0.0,
            "kospi_pct": 0.001, "kosdaq_pct": 0.001, "v_kospi": 18.0, "v_kosdaq": 22.0,
        })
        px = nxt
    ph_path = tmp_path / "price_history.parquet"
    pd.DataFrame(rows).to_parquet(ph_path)

    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-02-02"], "종목코드": ["000001"], "(종가)": [10800.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [10800.0], "(매도 가격)": [10900.0]}).to_parquet(trade_path)

    seen: dict[str, object] = {}

    def _fake_restore(trade_log_df, price_history_df, **kwargs):
        seen["max_abs_chg"] = float(np.nanmax(np.abs(price_history_df["daily_change_pct"].to_numpy(dtype=float))))
        seen["has_tick_cost"] = "tick_cost_bp" in price_history_df.columns
        out = trade_log_df.copy()
        out.attrs["panel_restoration"] = {}
        return out

    monkeypatch.setattr(mod, "build_restored_trade_log", _fake_restore)
    monkeypatch.setattr(mod, "check_price_history_freshness", lambda df: {"is_stale": False})

    # When
    mod.main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
              "--price-history", str(ph_path),
              "--condition-history", str(tmp_path / "missing_cond.parquet"),
              "--out", str(tmp_path / "ml_training_panel.parquet")])

    # Then
    assert seen["has_tick_cost"] is True
    assert np.isclose(float(seen["max_abs_chg"]), 0.08)


