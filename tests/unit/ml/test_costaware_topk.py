"""Cost-aware top-k backtest contract tests."""
from __future__ import annotations


def test_select_topk_by_tick_cost_ranks_ascending_and_caps_at_k() -> None:
    import pandas as pd

    from src.ml.costaware_topk import select_topk_by_tick_cost

    # Given: 4 same-day candidates with distinct tick costs
    cands = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"] * 4),
        "symbol": ["S1", "S2", "S3", "S4"],
        "tick_cost_bp": [6.803, 5.952, 5.291, 5.013],
    })

    # When
    picks = select_topk_by_tick_cost(cands, 3)

    # Then: lowest 3 kept, ascending, S1 (highest cost) dropped
    assert picks["symbol"].tolist() == ["S4", "S3", "S2"]
    assert len(picks) == 3

def test_select_topk_by_tick_cost_raises_on_missing_cost_column() -> None:
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import select_topk_by_tick_cost

    cands = pd.DataFrame({"date": pd.to_datetime(["2023-02-01"]), "symbol": ["S1"]})

    with pytest.raises(ValueError, match="tick_cost_bp"):
        select_topk_by_tick_cost(cands, 3)

    with pytest.raises(ValueError, match="k must be"):
        select_topk_by_tick_cost(pd.DataFrame({"date": [], "symbol": [], "tick_cost_bp": []}), 0)

def test_compute_net_return_propagates_nan_never_zero() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.costaware_topk import compute_net_return

    # Given: one clean row, one NaN gross, one NaN tick
    picks = pd.DataFrame({
        "gross_return": [0.02, np.nan, 0.02],
        "tick_cost_bp": [10.0, 10.0, np.nan],
    })

    # When
    net = compute_net_return(picks, round_trip_ticks=2.0, statutory_bp=20.0)

    # Then: row 0 = 0.02 - (20 + 2*10)/1e4 = 0.02 - 0.004 = 0.016
    assert np.isclose(net[0], 0.016)
    assert np.isnan(net[1])
    assert np.isnan(net[2])

def test_day_level_feasibility_flags_zero_and_below_k_candidate_days() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.costaware_topk import day_level_feasibility

    # Given: day1 has 2 candidates, day2 has 4, day3 has none (absent from cands)
    cands = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-01", "2023-02-02", "2023-02-02", "2023-02-02", "2023-02-02"]),
        "symbol": ["A", "B", "C", "D", "E", "F"],
    })
    market_dates = np.array(pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-03"]))

    # When
    feasible = day_level_feasibility(cands, 3, market_dates)

    # Then
    assert bool(feasible.loc[pd.Timestamp("2023-02-01")]) is False
    assert bool(feasible.loc[pd.Timestamp("2023-02-02")]) is True
    assert bool(feasible.loc[pd.Timestamp("2023-02-03")]) is False
    assert len(feasible) == 3

def test_compute_regime_metrics_splits_at_tick_reform_date() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.costaware_topk import compute_regime_metrics

    # Given: 2 pre-reform days (both feasible) and 2 post-reform days (both feasible)
    dates = pd.to_datetime(["2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26"])
    daily_net = pd.Series([0.01, 0.02, 0.03, 0.04], index=pd.DatetimeIndex(dates))
    feasibility = pd.Series([True, True, True, True], index=pd.DatetimeIndex(dates))

    # When
    regimes = compute_regime_metrics(daily_net, feasibility)

    # Then
    assert set(regimes.keys()) == {"pre_reform", "post_reform", "full_history"}
    assert regimes["pre_reform"].n_calendar_days == 2
    assert regimes["pre_reform"].n_days_with_signal == 2
    assert np.isclose(regimes["pre_reform"].mean_net_bp, 150.0)  # mean(0.01,0.02)*1e4
    assert regimes["post_reform"].n_calendar_days == 2
    assert np.isclose(regimes["post_reform"].mean_net_bp, 350.0)  # mean(0.03,0.04)*1e4
    assert regimes["full_history"].n_calendar_days == 4
    assert np.isclose(regimes["full_history"].mean_net_bp, 250.0)  # mean of all 4 *1e4
    assert regimes["pre_reform"].coverage_status == "OK"
    assert regimes["post_reform"].coverage_status == "OK"

def test_compute_regime_metrics_flags_insufficient_coverage_below_threshold() -> None:
    import pandas as pd

    from src.ml.costaware_topk import compute_regime_metrics

    # Given: post-reform has 10 signal days but only 5 are feasible (50% < 90% threshold)
    dates = pd.to_datetime([f"2023-02-{d:02d}" for d in range(1, 11)])
    daily_net = pd.Series([0.01] * 10, index=pd.DatetimeIndex(dates))
    feasibility = pd.Series([True] * 5 + [False] * 5, index=pd.DatetimeIndex(dates))

    # When
    regimes = compute_regime_metrics(daily_net, feasibility)

    # Then: coverage fails even though every day has a finite, identical net return
    assert regimes["post_reform"].feasible_day_fraction == 0.5
    assert regimes["post_reform"].coverage_status == "INSUFFICIENT_COVERAGE"
    assert regimes["post_reform"].n_days_with_signal == 10

def test_compute_cost_stress_grid_reports_pass_flag_per_tick_level() -> None:
    import pandas as pd

    from src.ml.costaware_topk import compute_cost_stress

    # Given: 3 post-reform days, gross return 1%, tick_cost_bp=30 -> cost = (20+ticks*30)/1e4
    # ticks=2: cost=80bp -> net=20bp (pass); ticks=3: cost=110bp -> net=-10bp (fail)
    dates = pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-03"])
    picks = pd.DataFrame({
        "date": dates,
        "gross_return": [0.01, 0.01, 0.01],
        "tick_cost_bp": [30.0, 30.0, 30.0],
    })

    # When
    stress = compute_cost_stress(picks, ticks_grid=(2.0, 3.0), regime="post_reform")

    # Then
    by_ticks = {p.round_trip_ticks: p for p in stress}
    assert by_ticks[2.0].passes is True
    assert by_ticks[2.0].mean_net_bp > 0
    assert by_ticks[3.0].passes is False
    assert by_ticks[3.0].mean_net_bp < 0

def test_evaluate_verdict_passes_when_all_gates_met() -> None:
    from src.ml.costaware_topk import CostStressPoint, RegimeMetrics, evaluate_verdict

    post = RegimeMetrics(
        regime="post_reform", n_calendar_days=878, n_days_with_signal=878, n_days_feasible=866,
        feasible_day_fraction=0.986, coverage_status="OK", mean_net_bp=18.75, median_net_bp=5.79,
        std_net_bp=180.0, win_rate=0.518, t_stat=3.08, sharpe=1.65, ci_low_bp=6.74, ci_high_bp=30.15, dsr=0.6,
    )
    pre = RegimeMetrics(
        regime="pre_reform", n_calendar_days=1903, n_days_with_signal=0, n_days_feasible=1087,
        feasible_day_fraction=0.571, coverage_status="INSUFFICIENT_COVERAGE", mean_net_bp=float("nan"),
        median_net_bp=float("nan"), std_net_bp=float("nan"), win_rate=float("nan"), t_stat=float("nan"),
        sharpe=float("nan"), ci_low_bp=float("nan"), ci_high_bp=float("nan"), dsr=float("nan"),
    )
    full = RegimeMetrics(
        regime="full_history", n_calendar_days=2781, n_days_with_signal=878, n_days_feasible=1953,
        feasible_day_fraction=0.702, coverage_status="INSUFFICIENT_COVERAGE", mean_net_bp=18.75,
        median_net_bp=5.79, std_net_bp=180.0, win_rate=0.518, t_stat=3.08, sharpe=1.65,
        ci_low_bp=6.74, ci_high_bp=30.15, dsr=0.6,
    )
    stress = [
        CostStressPoint(regime="post_reform", round_trip_ticks=2.0, n_days=878, mean_net_bp=18.75, median_net_bp=5.79, t_stat=3.08, passes=True),
        CostStressPoint(regime="post_reform", round_trip_ticks=3.0, n_days=878, mean_net_bp=13.41, median_net_bp=0.30, t_stat=2.20, passes=True),
        CostStressPoint(regime="post_reform", round_trip_ticks=4.0, n_days=878, mean_net_bp=8.07, median_net_bp=-5.18, t_stat=1.33, passes=False),
    ]

    verdict, reasons = evaluate_verdict({"pre_reform": pre, "post_reform": post, "full_history": full}, stress)

    assert verdict == "PASS_POST_REFORM"
    assert any("pre_reform" in r and "not gate-eligible" in r for r in reasons)
    assert not any("failed gate" in r for r in reasons)

def test_evaluate_verdict_fails_on_negative_median_despite_positive_mean() -> None:
    from src.ml.costaware_topk import CostStressPoint, RegimeMetrics, evaluate_verdict

    post = RegimeMetrics(
        regime="post_reform", n_calendar_days=878, n_days_with_signal=878, n_days_feasible=866,
        feasible_day_fraction=0.986, coverage_status="OK", mean_net_bp=34.16, median_net_bp=-14.69,
        std_net_bp=280.0, win_rate=0.469, t_stat=3.60, sharpe=1.93, ci_low_bp=15.34, ci_high_bp=52.95, dsr=0.6,
    )
    full = RegimeMetrics(
        regime="full_history", n_calendar_days=2781, n_days_with_signal=878, n_days_feasible=1953,
        feasible_day_fraction=0.702, coverage_status="INSUFFICIENT_COVERAGE", mean_net_bp=34.16,
        median_net_bp=-14.69, std_net_bp=280.0, win_rate=0.469, t_stat=3.60, sharpe=1.93,
        ci_low_bp=15.34, ci_high_bp=52.95, dsr=0.6,
    )
    stress = [
        CostStressPoint(regime="post_reform", round_trip_ticks=2.0, n_days=878, mean_net_bp=34.16, median_net_bp=-14.69, t_stat=3.60, passes=False),
        CostStressPoint(regime="post_reform", round_trip_ticks=3.0, n_days=878, mean_net_bp=29.06, median_net_bp=-20.00, t_stat=3.06, passes=False),
    ]

    verdict, reasons = evaluate_verdict({"post_reform": post, "full_history": full}, stress)

    assert verdict == "FAIL"
    assert any("failed gate" in r and "median" in r for r in reasons)

def test_evaluate_verdict_insufficient_coverage_short_circuits_statistics() -> None:
    from src.ml.costaware_topk import RegimeMetrics, evaluate_verdict

    post = RegimeMetrics(
        regime="post_reform", n_calendar_days=100, n_days_with_signal=40, n_days_feasible=40,
        feasible_day_fraction=0.40, coverage_status="INSUFFICIENT_COVERAGE", mean_net_bp=500.0,
        median_net_bp=500.0, std_net_bp=10.0, win_rate=1.0, t_stat=99.0, sharpe=5.0,
        ci_low_bp=490.0, ci_high_bp=510.0, dsr=0.9,
    )
    full = RegimeMetrics(
        regime="full_history", n_calendar_days=200, n_days_with_signal=40, n_days_feasible=40,
        feasible_day_fraction=0.20, coverage_status="INSUFFICIENT_COVERAGE", mean_net_bp=500.0,
        median_net_bp=500.0, std_net_bp=10.0, win_rate=1.0, t_stat=99.0, sharpe=5.0,
        ci_low_bp=490.0, ci_high_bp=510.0, dsr=0.9,
    )

    verdict, reasons = evaluate_verdict({"post_reform": post, "full_history": full}, [])

    assert verdict == "INSUFFICIENT_COVERAGE"
    assert any("0.4" in r or "feasible" in r.lower() for r in reasons)

def test_run_cost_aware_topk_backtest_rejects_top_k_below_minimum() -> None:
    import dataclasses

    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import run_cost_aware_topk_backtest
    from src.strategy.contract import KCA_TOPK_COSTAWARE_001

    bad_spec = dataclasses.replace(KCA_TOPK_COSTAWARE_001, top_k=1)

    with pytest.raises(ValueError, match="below the minimum investable K"):
        run_cost_aware_topk_backtest(pd.DataFrame(), np.array([]), {}, spec=bad_spec)

def test_run_cost_aware_topk_backtest_end_to_end_produces_report() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.data.panel_integrity import prepare_price_panel
    from src.ml.costaware_topk import run_cost_aware_topk_backtest
    from src.strategy.contract import KCA_TOPK_COSTAWARE_001

    # Given: 4 symbols at fixed post-reform tick=10 prices (14700/16800/18900/19950),
    # each row independently set to a +5% chg (prev_close = close/1.05) so every
    # pre-reform AND post-reform day offers the identical 4-name pool; only the
    # PIT tick table differs by regime (pre-reform tick=50 pushes all 4 names
    # above the 7.5bp COST_AWARE cap, so pre-reform has zero candidates/day).
    # Numeric symbol codes avoid str.zfill(6) altering the id (prepare_price_panel
    # zero-pads the symbol column). Verified against the real pipeline in
    # scratch/verify_costaware_fixture.py before this contract was finalized.
    prices = {
        "000001": (14000.0, 14700.0), "000002": (16000.0, 16800.0),
        "000003": (18000.0, 18900.0), "000004": (19000.0, 19950.0),
    }
    # D+1 open for day1/day2/day3 exits: entry_close * (1+gain), gain=3%,1%,2%
    gains = {"2023-01-25": 0.03, "2023-01-26": 0.01, "2023-01-27": 0.02}
    dates = ["2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27", "2023-01-30"]
    exit_map = {"2023-01-25": "2023-01-26", "2023-01-26": "2023-01-27", "2023-01-27": "2023-01-30"}

    rows = []
    for d in dates:
        is_exit_only_day = d == "2023-01-30"
        for sym, (prev, close) in prices.items():
            row_close, row_prev = (prev, prev) if is_exit_only_day else (close, prev)  # flat on the exit-only day
            rows.append({
                "date": pd.Timestamp(d), "symbol": sym, "open": row_close, "high": row_close * 1.01,
                "low": row_close * 0.99, "close": row_close, "prev_close": row_prev, "volume": 1000.0,
                "market_cap_100m": 900.0, "trade_value_100m": 300.0, "market": "KOSPI",
                "kospi_pct": 0.001, "kosdaq_pct": 0.001, "inst_netbuy": 0.0,
            })
    ph = pd.DataFrame(rows)

    # Wire the D+1 'open' of each exit date to entry_close * (1 + gain) for the 3 evaluated entry dates
    for entry_d, exit_d in exit_map.items():
        gain = gains[entry_d]
        for sym, (prev, close) in prices.items():  # noqa: B007
            target_open = close * (1.0 + gain)
            ph.loc[(ph["date"] == pd.Timestamp(exit_d)) & (ph["symbol"] == sym), "open"] = target_open

    prepared, _ = prepare_price_panel(ph)
    market_dates = np.array(sorted(prepared["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}

    # When
    report = run_cost_aware_topk_backtest(prepared, market_dates, d_to_idx, spec=KCA_TOPK_COSTAWARE_001)

    # Then: pre-reform is infeasible (coarser pre-reform tick bands push every
    # name above the 7.5bp cap at these prices), post-reform is fully feasible
    post = report.regimes["post_reform"]
    pre = report.regimes["pre_reform"]
    assert pre.coverage_status == "INSUFFICIENT_COVERAGE"
    assert pre.feasible_day_fraction == 0.0
    assert post.coverage_status == "OK"
    assert post.n_calendar_days == 3
    assert post.feasible_day_fraction == 1.0
    # top-3 of {000001..4} by ascending tick_cost_bp excludes 000001 (highest tick cost).
    # Exact values confirmed against the real pipeline: mean_net_bp=169.16, median_net_bp=169.16,
    # t_stat=2.93, win_rate=1.0; 3-tick stress mean_net_bp=163.74, median_net_bp=163.74.
    assert post.mean_net_bp == pytest.approx(169.16, abs=0.01)
    assert post.median_net_bp == pytest.approx(169.16, abs=0.01)
    assert post.win_rate == 1.0
    assert post.t_stat == pytest.approx(2.93, abs=0.01)
    stress_at_3 = next(p for p in report.cost_stress if p.round_trip_ticks == 3.0)
    assert stress_at_3.mean_net_bp == pytest.approx(163.74, abs=0.01)
    assert stress_at_3.median_net_bp == pytest.approx(163.74, abs=0.01)
    assert stress_at_3.passes is True
    assert report.verdict == "PASS_POST_REFORM"
    assert any("pre_reform" in r and "not gate-eligible" in r for r in report.verdict_reasons)

def test_report_to_frame_flattens_regimes_and_cost_stress() -> None:
    from src.ml.costaware_topk import CostAwareTopKReport, CostStressPoint, RegimeMetrics, report_to_frame

    rm = RegimeMetrics(
        regime="post_reform", n_calendar_days=3, n_days_with_signal=3, n_days_feasible=3,
        feasible_day_fraction=1.0, coverage_status="OK", mean_net_bp=169.16, median_net_bp=169.16,
        std_net_bp=100.0, win_rate=1.0, t_stat=2.93, sharpe=26.85, ci_low_bp=100.0, ci_high_bp=240.0, dsr=0.5,
    )
    cs = CostStressPoint(regime="post_reform", round_trip_ticks=3.0, n_days=3, mean_net_bp=163.74, median_net_bp=163.74, t_stat=2.5, passes=True)
    report = CostAwareTopKReport(
        strategy_id="KCA-TOPK-COSTAWARE-001", top_k=3, universe={}, cost={}, date_min="2023-01-23",
        date_max="2023-01-30", regimes={"post_reform": rm}, cost_stress=[cs],
        verdict="PASS_POST_REFORM", verdict_reasons=["ok"],
    )

    frame = report_to_frame(report)

    assert len(frame) == 2
    assert set(frame["row_type"]) == {"regime", "cost_stress"}
    assert (frame["strategy_id"] == "KCA-TOPK-COSTAWARE-001").all()
    assert (frame["verdict"] == "PASS_POST_REFORM").all()
    regime_row = frame[frame["row_type"] == "regime"].iloc[0]
    assert regime_row["mean_net_bp"] == 169.16
    stress_row = frame[frame["row_type"] == "cost_stress"].iloc[0]
    assert stress_row["round_trip_ticks"] == 3.0


# Supplemental guard coverage for fail-closed branches without a skeleton.


def test_select_topk_by_tick_cost_raises_on_missing_date_column() -> None:
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import select_topk_by_tick_cost

    cands = pd.DataFrame({"tick_cost_bp": [1.0], "symbol": ["S1"]})

    with pytest.raises(ValueError, match="date"):
        select_topk_by_tick_cost(cands, 3)


def test_compute_net_return_raises_on_missing_columns() -> None:
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import compute_net_return

    with pytest.raises(ValueError, match="gross_return"):
        compute_net_return(pd.DataFrame({"tick_cost_bp": [1.0]}), round_trip_ticks=2.0)

    with pytest.raises(ValueError, match="tick_cost_bp"):
        compute_net_return(pd.DataFrame({"gross_return": [0.01]}), round_trip_ticks=2.0)


def test_daily_mean_series_raises_on_bad_inputs() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import daily_mean_series

    picks = pd.DataFrame({"date": pd.to_datetime(["2023-02-01"]), "symbol": ["S1"]})

    with pytest.raises(ValueError, match="date_col"):
        daily_mean_series(picks.drop(columns=["date"]), np.array([0.01]))

    with pytest.raises(ValueError, match="length"):
        daily_mean_series(picks, np.array([0.01, 0.02]))


def test_day_level_feasibility_raises_on_bad_inputs() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import day_level_feasibility

    cands = pd.DataFrame({"date": pd.to_datetime(["2023-02-01"]), "symbol": ["A"]})
    market_dates = np.array(pd.to_datetime(["2023-02-01"]))

    with pytest.raises(ValueError, match="k must be"):
        day_level_feasibility(cands, 0, market_dates)

    with pytest.raises(ValueError, match="date_col"):
        day_level_feasibility(cands.drop(columns=["date"]), 3, market_dates)


def test_compute_regime_metrics_raises_on_non_datetime_index() -> None:
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import compute_regime_metrics

    good_daily = pd.Series([0.01], index=pd.DatetimeIndex(pd.to_datetime(["2023-02-01"])))
    good_feas = pd.Series([True], index=pd.DatetimeIndex(pd.to_datetime(["2023-02-01"])))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_regime_metrics(pd.Series([0.01], index=[0]), good_feas)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_regime_metrics(good_daily, pd.Series([True], index=[0]))


def test_compute_cost_stress_raises_on_unknown_regime() -> None:
    import pandas as pd
    import pytest

    from src.ml.costaware_topk import compute_cost_stress

    picks = pd.DataFrame({"date": pd.to_datetime(["2023-02-01"])})

    with pytest.raises(ValueError, match="regime"):
        compute_cost_stress(picks, regime="bogus")


def test_evaluate_verdict_documents_regime_break_without_pooling() -> None:
    from src.ml.costaware_topk import CostStressPoint, RegimeMetrics, evaluate_verdict

    def _rm(regime: str, mean: float, median: float) -> RegimeMetrics:
        return RegimeMetrics(
            regime=regime, n_calendar_days=10, n_days_with_signal=10, n_days_feasible=10,
            feasible_day_fraction=1.0, coverage_status="OK", mean_net_bp=mean,
            median_net_bp=median, std_net_bp=100.0, win_rate=0.6, t_stat=3.0,
            sharpe=1.0, ci_low_bp=1.0, ci_high_bp=2.0, dsr=0.5,
        )

    post = _rm("post_reform", 10.0, 5.0)
    full = _rm("full_history", -10.0, -5.0)
    stress = [
        CostStressPoint(
            regime="post_reform", round_trip_ticks=3.0, n_days=10,
            mean_net_bp=8.0, median_net_bp=4.0, t_stat=2.5, passes=True,
        ),
    ]

    verdict, reasons = evaluate_verdict({"post_reform": post, "full_history": full}, stress)

    assert verdict == "PASS_POST_REFORM"
    assert any("regime break" in r and "do not pool" in r for r in reasons)


def test_main_writes_parquet_report_and_rejects_missing_price_history(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import numpy as np
    import pandas as pd
    import pytest

    import src.ml.costaware_topk as mod
    from src.ml.costaware_topk import CostAwareTopKReport, RegimeMetrics

    ph_path = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2023-02-01"])}).to_parquet(ph_path)

    rm = RegimeMetrics(
        regime="post_reform", n_calendar_days=1, n_days_with_signal=1, n_days_feasible=1,
        feasible_day_fraction=1.0, coverage_status="OK", mean_net_bp=10.0, median_net_bp=5.0,
        std_net_bp=1.0, win_rate=1.0, t_stat=3.0, sharpe=1.0, ci_low_bp=1.0, ci_high_bp=2.0, dsr=0.5,
    )
    fake_report = CostAwareTopKReport(
        strategy_id="KCA-TOPK-COSTAWARE-001", top_k=5, universe={}, cost={},
        date_min="2023-02-01", date_max="2023-02-01", regimes={"post_reform": rm},
        cost_stress=[], verdict="PASS_POST_REFORM", verdict_reasons=["ok"],
    )
    seen: dict[str, object] = {}

    def _fake_load(path):
        seen["path"] = str(path)
        return (
            pd.DataFrame({"date": pd.to_datetime(["2023-02-01"])}),
            np.array(pd.to_datetime(["2023-02-01"])),
            {},
        )

    def _fake_run(ph, market_dates, d_to_idx, **kwargs):
        seen["top_k"] = kwargs["spec"].top_k
        return fake_report

    monkeypatch.setattr(mod, "load_and_prepare_price_history", _fake_load)
    monkeypatch.setattr(mod, "run_cost_aware_topk_backtest", _fake_run)

    out = tmp_path / "report.parquet"
    with caplog.at_level(logging.INFO, logger="src.ml.costaware_topk"):
        mod.main(["--price-history", str(ph_path), "--top-k", "5", "--out", str(out)])
    assert seen["top_k"] == 5
    assert out.exists()
    assert "PASS_POST_REFORM" in caplog.text

    out2 = tmp_path / "report2.parquet"
    mod.main(["--price-history", str(ph_path), "--out", str(out2)])
    assert out2.exists()

    with pytest.raises(ValueError, match="price_history not found"):
        mod.main(["--price-history", str(tmp_path / "nope.parquet"), "--out", str(tmp_path / "x.parquet")])
