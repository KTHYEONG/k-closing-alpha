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
            "high": [101.0],
            "low": [99.0],
            "prev_close": [99.0],
            "volume": [1000.0],
            "market_cap_100m": [900.0],
            "trade_value_100m": [300.0],
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
            "high": [101.0],
            "low": [99.0],
            "prev_close": [99.0],
            "volume": [1000.0],
            "market_cap_100m": [900.0],
            "trade_value_100m": [300.0],
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


def test_retrain_publish_requires_locked_oos_window() -> None:
    import pytest

    from src.ml.retrain import build_arg_parser

    parser = build_arg_parser()

    args = parser.parse_args(["--tuned", "--oos-reserve-start", "2025-09-01"])
    assert args.label_mode == "mechanical"
    assert args.cost_mode == "per_row"
    assert args.publish is False
    assert args.target_notional_100m == 0.5
    assert args.production_dir == "artifacts/models"

    published = parser.parse_args(
        ["--tuned", "--oos-reserve-start", "2025-09-01", "--publish"]
    )
    assert published.publish is True

    # Fail-closed: publishing without a locked window is rejected by the parser
    with pytest.raises(SystemExit):
        parser.parse_args(["--tuned", "--publish"])


def test_main_tuned_wires_screen_and_production_dir(tmp_path, monkeypatch) -> None:
    """Regression: --screen was parsed but never read, and --production-dir was
    parsed but never passed through to train_tuned_champion_bundle."""
    import src.ml.retrain as mod
    from src.ml.universe import SCREEN_REGISTRY

    trade_path = tmp_path / "trade_log.parquet"
    _write_trade_log(trade_path)
    theme_path = tmp_path / "theme_missing.parquet"

    captured: dict[str, object] = {}

    def _fake_train_tuned(trade_log_df, theme_df, cfg, **kwargs):
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs
        return {"training_cutoff": "2026-03-02", "tuning_provenance": {"control_vs_candidate": {}}}

    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _fake_train_tuned)

    main([
        "--trade-log", str(trade_path),
        "--theme", str(theme_path),
        "--tuned",
        "--no-restore-panel",
        "--screen", "band_2_15",
        "--production-dir", str(tmp_path / "prod"),
    ])

    assert captured["cfg"].screen is SCREEN_REGISTRY["band_2_15"]
    assert captured["kwargs"]["production_dir"] == str(tmp_path / "prod")


def test_retrain_promotion_alpha_flows_to_validation_config(tmp_path, monkeypatch) -> None:
    import src.ml.retrain as mod

    trade_path = tmp_path / "trade_log.parquet"
    _write_trade_log(trade_path)
    theme_path = tmp_path / "theme_missing.parquet"

    captured: dict[str, object] = {}

    def _fake_train_tuned(trade_log_df, theme_df, cfg, **kwargs):
        captured["cfg"] = cfg
        return {"training_cutoff": "2026-03-02", "tuning_provenance": {"control_vs_candidate": {}}}

    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _fake_train_tuned)

    main([
        "--trade-log", str(trade_path), "--theme", str(theme_path),
        "--tuned", "--no-restore-panel",
        "--oos-reserve-start", "2025-09-01", "--promotion-alpha", "0.20",
    ])

    cfg = captured["cfg"]
    assert cfg.promotion_alpha == 0.20
    assert cfg.validation is not None
    assert cfg.validation.promotion_alpha == 0.20


def test_retrain_scenario_source_flows_to_config_and_default(tmp_path, monkeypatch) -> None:
    import src.ml.retrain as mod
    from src.ml.retrain import build_arg_parser

    assert build_arg_parser().parse_args(["--tuned"]).scenario_source == "manual"

    trade_path = tmp_path / "trade_log.parquet"
    _write_trade_log(trade_path)
    theme_path = tmp_path / "theme_missing.parquet"

    captured: dict[str, object] = {}

    def _fake_train_tuned(trade_log_df, theme_df, cfg, **kwargs):
        captured["cfg"] = cfg
        return {"training_cutoff": "2026-03-02", "tuning_provenance": {"control_vs_candidate": {}}}

    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _fake_train_tuned)

    main([
        "--trade-log", str(trade_path), "--theme", str(theme_path),
        "--tuned", "--no-restore-panel", "--scenario-source", "none",
    ])
    assert captured["cfg"].scenario_source == "none"


def test_retrain_scenario_source_auto_requires_price_history(capsys) -> None:
    from src.ml.retrain import build_arg_parser

    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tuned", "--no-restore-panel", "--scenario-source", "auto"])
    assert "requires price_history" in capsys.readouterr().err


import numpy as np
import pandas as pd
import pytest

import src.ml.retrain as mod
from src.ml.retrain import main


def _price_history_file(path) -> None:
    dates = pd.bdate_range("2023-01-02", periods=20)
    pd.DataFrame({
        "date": list(dates) * 2,
        "symbol": ["000001"] * 20 + ["000002"] * 20,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "prev_close": 100.0,
        "market_cap_100m": 900.0, "trade_value_100m": 200.0, "daily_change_pct": 0.005,
        "market": "KOSPI", "volume": 1e5, "foreign_netbuy": 0.0, "inst_netbuy": 0.0,
        "program_netbuy": 0.0, "kospi_pct": 0.001, "kosdaq_pct": 0.001,
        "v_kospi": 18.0, "v_kosdaq": 22.0,
    }).to_parquet(path)


def test_retrain_universe_research_mode_dispatches_and_skips_champion(tmp_path, monkeypatch) -> None:
    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    def _fake_grid(price_history_df, screens, **kwargs):
        seen["n_screens"] = len(screens)
        seen["rows"] = len(price_history_df)
        return pd.DataFrame([{ "screen_name": "operator_legacy", "ranked_top1_net_bp": -10.0 }])

    def _boom_champion(*a, **k):
        raise AssertionError("champion training must not run in --universe-research mode")

    monkeypatch.setattr(mod, "run_universe_screen_grid", _fake_grid)
    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _boom_champion)
    monkeypatch.setattr(mod, "train_champion_bundle", _boom_champion)

    main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
          "--no-restore-panel", "--universe-research", "--export-dir", str(tmp_path)])

    assert seen["n_screens"] >= 5 and seen["rows"] == 40
    assert (tmp_path / "universe_grid.parquet").exists()


import pandas as pd
import pytest

import src.ml.retrain as mod
from src.ml.retrain import main


def test_retrain_universe_research_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
              "--no-restore-panel", "--universe-research", "--export-dir", str(tmp_path)])


def test_retrain_universe_research_prepares_price_panel(tmp_path, monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    # Given: a price_history whose vendor change column is percent-encoded
    dates = pd.bdate_range("2023-02-01", periods=6)
    prev = 10000.0
    rows = []
    for sym in ("000001", "000002"):
        px = prev
        for d in dates:
            nxt = px * 1.08
            rows.append({
                "date": d, "symbol": sym, "open": px, "high": nxt * 1.01, "low": px * 0.99,
                "close": nxt, "prev_close": px, "market_cap_100m": 900.0,
                "trade_value_100m": 300.0, "daily_change_pct": 8.0, "market": "KOSPI",
                "volume": 1e5, "foreign_netbuy": 0.0, "inst_netbuy": 0.0, "program_netbuy": 0.0,
                "kospi_pct": 0.001, "kosdaq_pct": 0.001, "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
            px = nxt
    ph_path = tmp_path / "price_history.parquet"
    pd.DataFrame(rows).to_parquet(ph_path)

    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-02-02"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    def _fake_grid(price_history_df, screens, **kwargs):
        seen["max_abs_chg"] = float(np.nanmax(np.abs(price_history_df["daily_change_pct"].to_numpy(dtype=float))))
        seen["has_tick_cost"] = "tick_cost_bp" in price_history_df.columns
        seen["has_chg_ratio"] = "chg_ratio" in price_history_df.columns
        seen["rows"] = len(price_history_df)
        return pd.DataFrame([{"screen_name": "operator_legacy", "ranked_top1_net_bp": -10.0}])

    def _boom(*a, **k):
        raise AssertionError("champion training must not run in --universe-research mode")

    monkeypatch.setattr(mod, "run_universe_screen_grid", _fake_grid)
    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _boom)
    monkeypatch.setattr(mod, "train_champion_bundle", _boom)

    # When
    main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
          "--no-restore-panel", "--universe-research", "--export-dir", str(tmp_path)])

    # Then: the harness never sees the percent-encoded column again
    assert seen["rows"] == 12
    assert seen["has_tick_cost"] is True
    assert seen["has_chg_ratio"] is True
    assert float(seen["max_abs_chg"]) < 1.0
    assert np.isclose(float(seen["max_abs_chg"]), 0.08)


def test_retrain_cost_aware_backtest_mode_dispatches_and_skips_champion(tmp_path, monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.ml.retrain as mod

    ph_path = tmp_path / "price_history.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"]), "symbol": ["000001"], "open": [100.0], "high": [101.0],
        "low": [99.0], "close": [100.0], "prev_close": [99.0], "volume": [1000.0],
        "market_cap_100m": [900.0], "trade_value_100m": [300.0], "market": ["KOSPI"],
    }).to_parquet(ph_path)
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-02-02"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)

    seen: dict[str, object] = {}

    def _fake_run(ph, market_dates, d_to_idx, **kwargs):
        seen["called"] = True
        seen["rows"] = len(ph)
        from src.ml.costaware_topk import CostAwareTopKReport
        return CostAwareTopKReport(
            strategy_id="KCA-TOPK-COSTAWARE-001", top_k=3, universe={}, cost={}, date_min="2023-02-01",
            date_max="2023-02-01", regimes={}, cost_stress=[], verdict="INSUFFICIENT_COVERAGE", verdict_reasons=["synthetic"],
        )

    def _boom(*a, **k):
        raise AssertionError("champion training must not run in --cost-aware-backtest mode")

    monkeypatch.setattr(mod, "run_cost_aware_topk_backtest", _fake_run)
    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _boom)
    monkeypatch.setattr(mod, "train_champion_bundle", _boom)

    mod.main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
              "--no-restore-panel", "--cost-aware-backtest", "--export-dir", str(tmp_path)])

    assert seen.get("called") is True
    assert seen["rows"] == 1
    assert (tmp_path / "costaware_topk_report.parquet").exists()


def test_retrain_cost_aware_backtest_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    import src.ml.retrain as mod

    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        mod.main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
                  "--no-restore-panel", "--cost-aware-backtest", "--export-dir", str(tmp_path)])


def test_retrain_ranker_topk_research_dispatches_and_skips_champion(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    class _Ev:
        top_k = 3
        n_paths = 28
        path_win_rate = 0.893
        mean_path_delta_bp = 8.04
        pooled_delta_bp = 10.97
        p_paired_t = 0.054

    class _Report:
        verdict = "PASS_POST_REFORM"
        verdict_reasons = ["ok"]
        top_k = 3
        path_evidence = _Ev()

    def _fake_run(ph, market_dates, d_to_idx, **kwargs):
        seen["rows"] = len(ph)
        return _Report()

    def _fake_frame(report):
        seen["framed"] = report.verdict
        return pd.DataFrame([{"row_type": "path_evidence", "verdict": report.verdict}])

    def _boom_champion(*a, **k):
        raise AssertionError("champion training must not run in --ranker-topk-research mode")

    monkeypatch.setattr(mod, "run_topk_ranker_backtest", _fake_run)
    monkeypatch.setattr(mod, "topk_ranker_report_to_frame", _fake_frame)
    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _boom_champion)
    monkeypatch.setattr(mod, "train_champion_bundle", _boom_champion)

    main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
          "--no-restore-panel", "--ranker-topk-research", "--export-dir", str(tmp_path)])

    assert seen["rows"] == 40
    assert seen["framed"] == "PASS_POST_REFORM"
    assert (tmp_path / "topk_ranker_report.parquet").exists()


def test_retrain_ranker_topk_research_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
              "--no-restore-panel", "--ranker-topk-research", "--export-dir", str(tmp_path)])


def test_retrain_ranker_topk_research_passes_explicit_train_start(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    class _Ev:
        top_k = 3
        n_paths = 28
        n_folds_total = 28
        n_folds_scored = 28
        path_win_rate = 1.0
        mean_path_delta_bp = 9.94
        pooled_delta_bp = 9.94
        p_paired_t = 0.068

    class _Report:
        verdict = "PASS_POST_REFORM"
        verdict_reasons = ["ok"]
        top_k = 3
        train_start = "2021-01-01"
        path_evidence = _Ev()

    def _fake_run(ph, market_dates, d_to_idx, **kwargs):
        seen["train_start"] = kwargs.get("train_start")
        return _Report()

    def _fake_frame(report):
        seen["framed"] = report.train_start
        return pd.DataFrame([{"row_type": "path_evidence", "verdict": report.verdict}])

    def _boom(*a, **k):
        raise AssertionError("champion training must not run in --ranker-topk-research mode")

    monkeypatch.setattr(mod, "run_topk_ranker_backtest", _fake_run)
    monkeypatch.setattr(mod, "topk_ranker_report_to_frame", _fake_frame)
    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _boom)
    monkeypatch.setattr(mod, "train_champion_bundle", _boom)

    # When: the operator widens the training window from the CLI
    main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
          "--no-restore-panel", "--ranker-topk-research", "--ranker-train-start", "2021-01-01",
          "--export-dir", str(tmp_path)])

    # Then: the flag reaches the harness as a Timestamp, not a string
    assert seen["train_start"] == pd.Timestamp("2021-01-01")
    assert (tmp_path / "topk_ranker_report.parquet").exists()

    # Then: omitting the flag leaves the harness on its certification-regime default
    seen.clear()
    main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
          "--no-restore-panel", "--ranker-topk-research", "--export-dir", str(tmp_path)])
    assert seen["train_start"] is None


def test_retrain_train_ranker_bundle_dispatches_and_skips_champion(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        seen["rows"] = len(ph)
        return {"feature_cols": ["f1"], "rank_model": object(), "quantile_models": {},
                "calibrators": {}, "top_k": 3}

    def _fake_save(bundle, export_dir):
        seen["export_dir"] = export_dir
        seen["bundle_top_k"] = bundle["top_k"]
        return str(tmp_path / "topk_ranker" / "sizing_pipeline_bundle.joblib")

    def _boom(*a, **k):
        raise AssertionError("champion training must not run in --train-ranker-bundle mode")

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.setattr(mod, "train_tuned_champion_bundle", _boom)
    monkeypatch.setattr(mod, "train_champion_bundle", _boom)

    main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
          "--no-restore-panel", "--train-ranker-bundle", "--export-dir", str(tmp_path)])

    assert seen["rows"] == 40
    assert seen["bundle_top_k"] == 3


def test_retrain_train_ranker_bundle_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    trade_path = tmp_path / "trade_log.parquet"
    pd.DataFrame({"매수날짜": ["2023-01-03"], "종목코드": ["000001"], "(종가)": [100.0],
                  "(수익률, %)": [1.0], "(매수 가격)": [100.0], "(매도 가격)": [101.0]}).to_parquet(trade_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--trade-log", str(trade_path), "--theme", str(tmp_path / "missing.parquet"),
              "--no-restore-panel", "--train-ranker-bundle", "--export-dir", str(tmp_path)])
