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
