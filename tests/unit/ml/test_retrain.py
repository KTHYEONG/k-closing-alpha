"""retrain CLI 의 --feature-set 인자 배선 계약."""
from __future__ import annotations

import logging

import pytest

from src.ml.retrain import main


def test_retrain_rejects_unknown_feature_set(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--feature-set", "bogus"])
    assert exc.value.code != 0
    assert "feature-set" in capsys.readouterr().err


def _write_trade_log(path) -> None:
    import pandas as pd

    pd.DataFrame(
        {
            "매수날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "(종가)": [100.0],
            "(수익률, %)": [1.0],
            "(매수 가격)": [100.0],
            "(매도 가격)": [101.0],
        }
    ).to_parquet(path)


def test_main_restores_panel_and_logs_provenance(tmp_path, monkeypatch, caplog) -> None:
    import pandas as pd

    from src.ml import retrain as mod

    trade_path = tmp_path / "trade_log.parquet"
    _write_trade_log(trade_path)
    price_path = tmp_path / "price_history.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02"]),
            "symbol": ["005930"],
            "open": [100.0],
            "close": [100.0],
        }
    ).to_parquet(price_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", price_path)

    calls: dict = {}

    def _fake_restore(df, price_history_df, *, theme_df=None):
        calls["theme_df"] = theme_df
        out = df.copy()
        out.attrs["panel_restoration"] = {
            "execution_offset_pct": -0.30,
            "restored_rows": 5,
            "restored_dates": 2,
            "restored_date_min": "2026-03-02",
            "restored_date_max": "2026-03-03",
        }
        return out

    def _fake_train(trade_log_df, theme_df, **kwargs):
        calls["trained_rows"] = len(trade_log_df)
        return {"training_cutoff": "2026-03-03"}

    monkeypatch.setattr(mod, "build_restored_trade_log", _fake_restore)
    monkeypatch.setattr(mod, "train_champion_bundle", _fake_train)

    with caplog.at_level(logging.INFO, logger="src.ml.retrain"):
        mod.main(
            [
                "--trade-log",
                str(trade_path),
                "--theme",
                str(tmp_path / "missing_theme.parquet"),
                "--export-dir",
                str(tmp_path / "models"),
            ]
        )

    assert calls["trained_rows"] == 1
    assert calls["theme_df"] is None
    assert "[DATA] stage=panel_restore" in caplog.text
    assert "restored_rows=5" in caplog.text


def test_main_no_restore_panel_skips_synthesis(tmp_path, monkeypatch) -> None:
    from src.ml import retrain as mod

    trade_path = tmp_path / "trade_log.parquet"
    _write_trade_log(trade_path)

    def _boom(df, price_history_df, **kwargs):
        raise AssertionError("build_restored_trade_log must not run with --no-restore-panel")

    monkeypatch.setattr(mod, "build_restored_trade_log", _boom)
    monkeypatch.setattr(mod, "train_champion_bundle", lambda *a, **k: {"training_cutoff": "x"})

    mod.main(
        [
            "--trade-log",
            str(trade_path),
            "--theme",
            str(tmp_path / "missing_theme.parquet"),
            "--export-dir",
            str(tmp_path / "models"),
            "--no-restore-panel",
        ]
    )


def test_main_skips_restoration_when_price_history_missing(tmp_path, monkeypatch, caplog) -> None:
    """Default restore-on path must not crash when price_history.parquet is absent."""
    from src.ml import retrain as mod

    trade_path = tmp_path / "trade_log.parquet"
    _write_trade_log(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "missing_price_history.parquet")

    def _boom(df, price_history_df, **kwargs):
        raise AssertionError("build_restored_trade_log must not run when price_history_df is None")

    calls: dict = {}

    def _fake_train(trade_log_df, theme_df, **kwargs):
        calls["trained_rows"] = len(trade_log_df)
        return {"training_cutoff": "x"}

    monkeypatch.setattr(mod, "build_restored_trade_log", _boom)
    monkeypatch.setattr(mod, "train_champion_bundle", _fake_train)

    with caplog.at_level(logging.WARNING, logger="src.ml.retrain"):
        mod.main(
            [
                "--trade-log",
                str(trade_path),
                "--theme",
                str(tmp_path / "missing_theme.parquet"),
                "--export-dir",
                str(tmp_path / "models"),
            ]
        )

    assert calls["trained_rows"] == 1
    assert "status=skipped" in caplog.text
    assert "reason=price_history_missing" in caplog.text


def test_retrain_warns_on_stale_price_history_without_changing_behavior(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.ml import retrain as mod

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
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", price_path)

    calls: dict = {}

    def _fake_restore(df, price_history_df, *, theme_df=None):
        out = df.copy()
        out.attrs["panel_restoration"] = {
            "execution_offset_pct": 0.0, "restored_rows": 0, "restored_dates": 0,
            "restored_date_min": "", "restored_date_max": "",
        }
        return out

    def _fake_train(trade_log_df, theme_df, **kwargs):
        calls["trained_rows"] = len(trade_log_df)
        return {"training_cutoff": "x"}

    monkeypatch.setattr(mod, "build_restored_trade_log", _fake_restore)
    monkeypatch.setattr(mod, "train_champion_bundle", _fake_train)

    with caplog.at_level(logging.WARNING, logger="src.ml.retrain"):
        mod.main(
            [
                "--trade-log", str(trade_path),
                "--theme", str(tmp_path / "missing_theme.parquet"),
                "--export-dir", str(tmp_path / "models"),
            ]
        )

    # Assert: warned about staleness, but training still ran on the restored frame.
    assert calls["trained_rows"] == 1
    assert "stage=ml_panel_freshness" in caplog.text
    assert "status=stale" in caplog.text
