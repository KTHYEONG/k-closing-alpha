"""Top-K cost-aware ranker research harness tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

def _synthetic_prepared_panel() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Build a small PIT-prepared panel: 12 symbols x 14 post-reform business days.

    Eight symbols priced 18,000 KRW carry a 10 KRW tick (5.56bp) and clear the
    7.5bp cap; four priced 30,000 KRW carry a 50 KRW tick (16.67bp) and do not.
    Every row has chg_ratio 0.05, so the wide pool is 168 rows over 14 days and
    the cost-capped select pool is 112 rows, 8 per day on all 14 days.
    """
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel

    dates = pd.bdate_range("2023-02-01", periods=14)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(12):
        base = 18000.0 if i < 8 else 30000.0
        for d in dates:
            prev = base / 1.05
            rows.append({
                "date": d, "symbol": f"{i:06d}", "open": prev, "high": base * 1.01,
                "low": prev * 0.99, "close": base, "prev_close": prev, "volume": 1e6,
                "market_cap_100m": 3000.0, "trade_value_100m": 500.0,
                "market": "KOSPI" if i % 2 == 0 else "KOSDAQ", "daily_change_pct": 0.05,
                "inst_netbuy": float(rng.integers(-10**8, 10**8)),
                "foreign_netbuy": float(rng.integers(-10**8, 10**8)),
                "program_netbuy": 0.0, "kospi_pct": 0.001, "kosdaq_pct": 0.002,
                "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
    ph, _prov = prepare_price_panel(pd.DataFrame(rows))
    market_dates = np.array(sorted(ph["date"].unique()))
    return ph, market_dates, {d: i for i, d in enumerate(market_dates)}

def test_assert_nested_universe_specs_accepts_only_cost_cap_difference() -> None:
    import dataclasses

    import pytest

    from src.ml.topk_ranker_research import assert_nested_universe_specs
    from src.strategy.contract import COST_AWARE_UNIVERSE, DEFAULT_UNIVERSE

    # Given/When: the shipped pair differs only in max_tick_cost_bp -> accepted
    assert assert_nested_universe_specs(DEFAULT_UNIVERSE, COST_AWARE_UNIVERSE) is None

    # Then: a train spec that already carries a cap is rejected
    with pytest.raises(ValueError, match="max_tick_cost_bp"):
        assert_nested_universe_specs(COST_AWARE_UNIVERSE, COST_AWARE_UNIVERSE)

    # Then: a select spec without a cap is cap-free and trivially nested
    assert assert_nested_universe_specs(DEFAULT_UNIVERSE, DEFAULT_UNIVERSE) is None

    # Then: any other differing field is named in the error
    skewed = dataclasses.replace(COST_AWARE_UNIVERSE, chg_min=0.05)
    with pytest.raises(ValueError, match="chg_min"):
        assert_nested_universe_specs(DEFAULT_UNIVERSE, skewed)

def test_select_topk_by_score_ranks_descending_and_validates_inputs() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import select_topk_by_score

    # Given: 4 same-day candidates with distinct scores
    cands = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"] * 4),
        "symbol": ["S1", "S2", "S3", "S4"],
        "pred": [0.001, 0.004, 0.003, -0.002],
    })

    # When
    picks = select_topk_by_score(cands, 3)

    # Then: highest 3 kept in descending order, S4 dropped
    assert picks["symbol"].tolist() == ["S2", "S3", "S1"]
    assert len(picks) == 3

    with pytest.raises(ValueError, match="k must be"):
        select_topk_by_score(cands, 0)
    with pytest.raises(ValueError, match="pred"):
        select_topk_by_score(cands.drop(columns=["pred"]), 3)
    with pytest.raises(ValueError, match="date"):
        select_topk_by_score(cands.drop(columns=["date"]), 3)

def test_assert_unique_date_symbol_raises_on_cpcv_fold_duplicates() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import assert_unique_date_symbol

    # Given: a clean frame passes
    clean = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-01"]),
        "symbol": ["S1", "S2"],
    })
    assert assert_unique_date_symbol(clean) is None

    # Given: the measured CPCV shape -- the same pair repeated once per fold
    dup = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"] * 4),
        "symbol": ["S1", "S1", "S2", "S2"],
    })

    # Then: fail closed, reporting duplicate pair count and multiplicity
    with pytest.raises(ValueError) as exc:  # noqa: PT011 - spec skeleton asserts message content below
        assert_unique_date_symbol(dup)
    msg = str(exc.value)
    assert "2" in msg

def test_dedupe_cpcv_oof_collapses_folds_to_one_row_per_pair() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.topk_ranker_research import assert_unique_date_symbol, dedupe_cpcv_oof

    # Given: one date, two symbols, each scored by 2 CPCV folds
    oof = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"] * 4),
        "symbol": ["S1", "S1", "S2", "S2"],
        "cpcv_fold": [0, 1, 0, 1],
        "pred": [0.002, 0.004, 0.010, 0.010],
        "net_pit": [0.01, 0.01, -0.02, -0.02],
        "tick_cost_bp": [5.0, 5.0, 7.0, 7.0],
    })

    # When
    out = dedupe_cpcv_oof(oof, value_cols=("net_pit", "tick_cost_bp"))

    # Then: one row per pair, pred averaged, value columns preserved
    assert len(out) == 2
    assert assert_unique_date_symbol(out) is None
    s1 = out[out["symbol"] == "S1"].iloc[0]
    assert np.isclose(float(s1["pred"]), 0.003)
    assert np.isclose(float(s1["net_pit"]), 0.01)
    assert np.isclose(float(s1["tick_cost_bp"]), 5.0)

def test_attach_pit_net_label_uses_pit_tick_cost_and_clips_label() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import attach_pit_net_label
    from src.strategy.contract import AA_COST

    # Given: one ordinary row, one huge-gross row, one NaN tick cost
    cands = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"] * 3),
        "symbol": ["S1", "S2", "S3"],
        "gross_return": [0.02, 0.50, 0.02],
        "tick_cost_bp": [10.0, 10.0, np.nan],
    })

    # When
    out = attach_pit_net_label(cands, cost=AA_COST, label_clip=0.10)

    # Then: net_pit = gross - (20 + 2*tick)/1e4; NaN tick propagates, never 0
    assert np.isclose(float(out.loc[0, "net_pit"]), 0.016)
    assert np.isnan(float(out.loc[2, "net_pit"]))
    # Then: train_label is the clipped net_pit
    assert np.isclose(float(out.loc[1, "train_label"]), 0.10)
    assert np.isclose(float(out.loc[0, "train_label"]), 0.016)
    assert np.isnan(float(out.loc[2, "train_label"]))

    with pytest.raises(ValueError, match="tick_cost_bp"):
        attach_pit_net_label(cands.drop(columns=["tick_cost_bp"]), cost=AA_COST)
    with pytest.raises(ValueError, match="gross_return"):
        attach_pit_net_label(cands.drop(columns=["gross_return"]), cost=AA_COST)

def test_compute_yearly_stability_masks_nan_and_reports_each_year() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.topk_ranker_research import compute_yearly_stability

    # Given: two years of daily net returns, one NaN planted in each
    idx_a = pd.bdate_range("2024-01-01", periods=30)
    idx_b = pd.bdate_range("2025-01-01", periods=30)
    vals_a = np.full(30, 0.001, dtype=np.float64)
    vals_a[0] = np.nan
    vals_b = np.full(30, 0.004, dtype=np.float64)
    vals_b[0] = np.nan
    daily = pd.Series(np.concatenate([vals_a, vals_b]), index=idx_a.append(idx_b))

    # When
    out = compute_yearly_stability(daily)

    # Then: one record per year, ascending, NaN excluded from n and mean
    assert [r.year for r in out] == [2024, 2025]
    assert out[0].n_days == 29
    assert out[1].n_days == 29
    assert np.isclose(out[0].mean_net_bp, 10.0)
    assert np.isclose(out[1].mean_net_bp, 40.0)
    assert np.isfinite(out[0].median_net_bp)

def test_compute_path_evidence_selects_within_fold_not_pooled() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.topk_ranker_research import compute_path_evidence

    # Given: 1 date x 3 symbols x 2 folds. Pooled top-3 would return 3 copies of
    # the single cheapest symbol (the measured collapse); per-fold top-3 must
    # average all three distinct symbols instead.
    rows = []
    for fold in (0, 1):
        for sym, tick, net, pred in (
            ("S1", 4.0, -0.030, 0.005),
            ("S2", 5.0, 0.030, 0.004),
            ("S3", 6.0, 0.030, 0.003),
        ):
            rows.append({"date": pd.Timestamp("2023-02-01"), "symbol": sym, "cpcv_fold": fold,
                         "tick_cost_bp": tick, "net_pit": net, "pred": pred})
    sel = pd.DataFrame(rows)

    # When
    ev = compute_path_evidence(sel, top_k=3)

    # Then: both folds evaluated; ranker top-3 mean is +0.01 (all three names),
    # control top-3 mean is identical, so the delta is 0 and no fold is a win.
    assert ev.n_paths == 2
    assert ev.top_k == 3
    assert np.isclose(ev.mean_path_delta_bp, 0.0)
    assert np.isclose(ev.path_win_rate, 0.0)

def test_compute_path_evidence_counts_ranker_wins_and_requires_fold_column() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import compute_path_evidence

    # Given: 2 folds x 1 date x 3 symbols; the ranker's top-1 is the profitable
    # name while the cheapest name is the loser, so the ranker wins both folds.
    rows = []
    for fold in (0, 1):
        for sym, tick, net, pred in (
            ("S1", 4.0, -0.010, 0.001),
            ("S2", 9.0, 0.020, 0.009),
            ("S3", 5.0, -0.010, 0.002),
        ):
            rows.append({"date": pd.Timestamp("2023-02-01"), "symbol": sym, "cpcv_fold": fold,
                         "tick_cost_bp": tick, "net_pit": net, "pred": pred})
    sel = pd.DataFrame(rows)

    # When
    ev = compute_path_evidence(sel, top_k=1)

    # Then: ranker picks S2 (+200bp), control picks S1 (-100bp)
    assert ev.n_paths == 2
    assert np.isclose(ev.path_win_rate, 1.0)
    assert np.isclose(ev.mean_path_delta_bp, 300.0)
    assert np.isclose(ev.pooled_delta_bp, 300.0)

    # Then: a frame without the fold column fails closed
    with pytest.raises(ValueError, match="cpcv_fold"):
        compute_path_evidence(sel.drop(columns=["cpcv_fold"]), top_k=1)

def test_evaluate_ranker_verdict_gates_coverage_paths_and_control() -> None:
    from src.ml.costaware_topk import CostStressPoint, RegimeMetrics
    from src.ml.topk_ranker_research import (
        ArmMetrics,
        PathEvidence,
        evaluate_ranker_verdict,
    )

    def _rm(mean: float, median: float, t: float, coverage: str = "OK") -> RegimeMetrics:
        return RegimeMetrics(
            regime="post_reform", n_calendar_days=879, n_days_with_signal=878,
            n_days_feasible=867, feasible_day_fraction=0.9863, coverage_status=coverage,
            mean_net_bp=mean, median_net_bp=median, std_net_bp=180.0, win_rate=0.56,
            t_stat=t, sharpe=2.5, ci_low_bp=10.0, ci_high_bp=45.0, dsr=0.9,
        )

    def _arm(name: str, mean: float, median: float, t: float, coverage: str = "OK",
             stress_pass: bool = True) -> ArmMetrics:
        return ArmMetrics(
            arm=name, top_k=3, regimes={"post_reform": _rm(mean, median, t, coverage)},
            by_year=[],
            cost_stress=[CostStressPoint(regime="post_reform", round_trip_ticks=3.0, n_days=878,
                                         mean_net_bp=24.9, median_net_bp=17.4, t_stat=3.3,
                                         passes=stress_pass)],
        )

    good_ev = PathEvidence(top_k=3, n_paths=28, path_win_rate=0.893,
                           mean_path_delta_bp=8.04, pooled_delta_bp=10.97, p_paired_t=0.054)
    control = _arm("costsort", 18.75, 5.79, 3.08)

    # Then: all gates satisfied
    verdict, reasons = evaluate_ranker_verdict(_arm("ranker", 29.72, 18.6, 4.81), control, good_ev)
    assert verdict == "PASS_POST_REFORM"

    # Then: coverage is checked ahead of any statistic
    verdict, reasons = evaluate_ranker_verdict(
        _arm("ranker", 29.72, 18.6, 4.81, coverage="INSUFFICIENT_COVERAGE"), control, good_ev)
    assert verdict == "INSUFFICIENT_COVERAGE"

    # Then: a path win rate below the gate fails
    weak_ev = PathEvidence(top_k=3, n_paths=28, path_win_rate=0.464,
                           mean_path_delta_bp=-0.78, pooled_delta_bp=-0.55, p_paired_t=0.96)
    verdict, reasons = evaluate_ranker_verdict(_arm("ranker", 29.72, 18.6, 4.81), control, weak_ev)
    assert verdict == "FAIL"
    assert any("path_win_rate" in r for r in reasons)

    # Then: not beating the model-free control fails even with strong paths
    verdict, reasons = evaluate_ranker_verdict(_arm("ranker", 12.0, 4.0, 4.81), control, good_ev)
    assert verdict == "FAIL"
    assert any("control" in r for r in reasons)

    # Then: a failing 3-tick stress point fails
    verdict, reasons = evaluate_ranker_verdict(
        _arm("ranker", 29.72, 18.6, 4.81, stress_pass=False), control, good_ev)
    assert verdict == "FAIL"

def test_build_dual_pool_returns_wide_pool_with_cost_capped_subset_mask() -> None:
    import numpy as np
    import pytest

    from src.ml.research.v3_engine import FEATURE_COLS
    from src.ml.topk_ranker_research import build_dual_pool
    from src.strategy.contract import COST_AWARE_UNIVERSE, DEFAULT_UNIVERSE

    ph, market_dates, d_to_idx = _synthetic_prepared_panel()

    # When
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx,
        train_spec=DEFAULT_UNIVERSE, select_spec=COST_AWARE_UNIVERSE,
    )

    # Then: the wide pool carries forward-exit and every derived feature column
    assert len(pool) == 168
    assert {"gross_return", "tick_cost_bp"}.issubset(pool.columns)
    assert not [c for c in FEATURE_COLS if c not in pool.columns]
    # Then: the mask is a strict, non-empty subset of the same frame
    assert sel_mask.dtype == np.bool_
    assert len(sel_mask) == len(pool)
    assert int(sel_mask.sum()) == 112
    # Then: every selected row respects the PIT tick-cost cap
    capped = pool.loc[sel_mask, "tick_cost_bp"].to_numpy(dtype=np.float64)
    assert np.all(capped <= float(COST_AWARE_UNIVERSE.max_tick_cost_bp))
    assert np.all(pool.loc[~sel_mask, "tick_cost_bp"].to_numpy(dtype=np.float64) > 7.5)

    # Then: a cap-free select spec identical to the train spec is trivially nested
    full_pool, full_mask = build_dual_pool(ph, market_dates, d_to_idx,
                    train_spec=DEFAULT_UNIVERSE, select_spec=DEFAULT_UNIVERSE)
    assert int(full_mask.sum()) == len(full_pool)

def test_score_pool_cpcv_fails_closed_below_min_rows_and_drops_nan_target() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import score_pool_cpcv

    # Given: 10 dates x 4 symbols with one NaN label planted
    dates = pd.bdate_range("2023-02-01", periods=10)
    rows = []
    rng = np.random.default_rng(3)
    for d in dates:
        for s in range(4):
            rows.append({"date": d, "symbol": f"{s:06d}",  # noqa: PERF401 - spec skeleton
                         "f1": float(rng.normal()), "f2": float(rng.normal()),
                         "train_label": float(rng.normal()) * 0.01})
    train_df = pd.DataFrame(rows)
    train_df.loc[0, "train_label"] = np.nan
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=0)

    # When
    oof = score_pool_cpcv(train_df, ["f1", "f2"], cv=cv, min_train_rows=10)

    # Then: predictions carry the fold id and the NaN-label row never trains
    assert "pred" in oof.columns
    assert "cpcv_fold" in oof.columns
    assert len(oof) > 0
    assert np.isfinite(oof["pred"].to_numpy(dtype=np.float64)).all()

    # Then: an undersized pool fails closed rather than training on noise
    with pytest.raises(ValueError, match="min_train_rows|TRAIN_POOL_MIN_ROWS|rows"):  # noqa: RUF043 - spec skeleton alternation
        score_pool_cpcv(train_df, ["f1", "f2"], cv=cv, min_train_rows=10_000)

def test_compute_arm_metrics_masks_nan_and_carries_regimes_years_and_stress() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.topk_ranker_research import compute_arm_metrics
    from src.strategy.contract import AA_COST

    # Given: 3 picks/day over 2 pre-reform and 4 post-reform days, one NaN net
    pre = pd.bdate_range("2022-12-01", periods=2)
    post = pd.bdate_range("2023-02-01", periods=4)
    rows = []
    for d in list(pre) + list(post):
        for s in range(3):
            rows.append({"date": d, "symbol": f"{s:06d}", "gross_return": 0.004,  # noqa: PERF401 - spec skeleton
                         "tick_cost_bp": 6.0, "net_pit": 0.004 - (20.0 + 12.0) / 1e4})
    picks = pd.DataFrame(rows)
    picks.loc[0, "net_pit"] = np.nan
    calendar = pd.DatetimeIndex(list(pre) + list(post))
    feasibility_full = pd.Series(np.ones(len(calendar), dtype=bool), index=calendar)

    # When
    arm = compute_arm_metrics(picks, feasibility_full, arm="ranker", top_k=3, cost=AA_COST)

    # Then: all three regimes are present and NaN never propagates into a stat
    assert arm.arm == "ranker"
    assert arm.top_k == 3
    assert set(arm.regimes) == {"pre_reform", "post_reform", "full_history"}
    assert arm.regimes["post_reform"].n_days_with_signal == 4
    assert np.isfinite(arm.regimes["post_reform"].mean_net_bp)
    assert np.isclose(arm.regimes["post_reform"].mean_net_bp, 8.0)
    assert arm.regimes["post_reform"].coverage_status == "OK"
    # Then: per-year stability is reported, never aggregated away
    assert [y.year for y in arm.by_year] == [2022, 2023]
    # Then: the round-trip-tick stress grid is carried for the post-reform slice
    assert [p.round_trip_ticks for p in arm.cost_stress] == [2.0, 3.0, 4.0]
    assert all(p.regime == "post_reform" for p in arm.cost_stress)

def test_run_topk_ranker_backtest_reports_both_arms_and_rejects_k_below_min() -> None:
    import dataclasses

    import pytest

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import CERT_REGIME_START, TopKRankerReport, run_topk_ranker_backtest
    from src.strategy.contract import KCA_TOPK_COSTAWARE_001

    ph, market_dates, d_to_idx = _synthetic_prepared_panel()
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=0)

    # When
    report = run_topk_ranker_backtest(
        ph, market_dates, d_to_idx, spec=KCA_TOPK_COSTAWARE_001,
        cv=cv, min_train_rows=10, train_start=CERT_REGIME_START,
    )

    # Then: both arms scored at the same k, over the post-reform regime
    assert isinstance(report, TopKRankerReport)
    assert report.top_k == 3
    assert report.ranker.arm == "ranker"
    assert report.control.arm == "costsort"
    assert report.control.top_k == report.ranker.top_k
    assert set(report.ranker.regimes) == {"pre_reform", "post_reform", "full_history"}
    assert report.path_evidence.top_k == 3
    assert report.path_evidence.n_paths > 0
    assert report.verdict in {"PASS_POST_REFORM", "FAIL", "INSUFFICIENT_COVERAGE"}
    assert report.n_select_rows < report.n_train_rows
    # Then: the cost cap is recorded on the select universe only
    assert report.select_universe["max_tick_cost_bp"] is not None
    assert report.train_universe["max_tick_cost_bp"] is None

    # Then: k below the investable minimum is rejected by measurement
    thin = dataclasses.replace(KCA_TOPK_COSTAWARE_001, top_k=MIN_TOP_K - 1)
    with pytest.raises(ValueError, match="top_k"):
        run_topk_ranker_backtest(ph, market_dates, d_to_idx, spec=thin, cv=cv, min_train_rows=10)

def test_topk_ranker_report_to_frame_flattens_every_row_type() -> None:
    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import (
        CERT_REGIME_START,
        run_topk_ranker_backtest,
        topk_ranker_report_to_frame,
    )

    ph, market_dates, d_to_idx = _synthetic_prepared_panel()
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=0)
    report = run_topk_ranker_backtest(ph, market_dates, d_to_idx, cv=cv, min_train_rows=10, train_start=CERT_REGIME_START)

    # When
    frame = topk_ranker_report_to_frame(report)

    # Then: every row type is present and self-describing
    assert set(frame["row_type"].unique()) >= {"regime", "cost_stress", "path_evidence"}
    assert set(frame.loc[frame["row_type"] == "regime", "arm"].unique()) == {"ranker", "costsort"}
    assert frame["strategy_id"].nunique() == 1
    assert (frame["top_k"] == report.top_k).all()
    assert (frame["verdict"] == report.verdict).all()

def test_main_writes_report_and_fails_closed_on_missing_price_history(
    tmp_path, monkeypatch, caplog
) -> None:
    import logging

    import pytest

    import src.ml.topk_ranker_research as mod

    ph, market_dates, d_to_idx = _synthetic_prepared_panel()
    ph_path = tmp_path / "price_history.parquet"
    ph_path.write_text("stub")
    seen: dict[str, object] = {}

    def _fake_load(path):
        seen["path"] = str(path)
        return ph, market_dates, d_to_idx

    def _fake_run(_ph, _md, _d2i, **kwargs):
        seen["top_k"] = kwargs["spec"].top_k
        return mod.TopKRankerReport(
            strategy_id="KCA-TOPK-COSTAWARE-001", top_k=int(kwargs["spec"].top_k),
            train_universe={}, select_universe={}, cost={}, date_min="2023-02-01",
            date_max="2023-02-20", n_train_rows=168, n_select_rows=32,
            ranker=mod.ArmMetrics(arm="ranker", top_k=int(kwargs["spec"].top_k), regimes={},
                                  by_year=[], cost_stress=[]),
            control=mod.ArmMetrics(arm="costsort", top_k=int(kwargs["spec"].top_k), regimes={},
                                   by_year=[], cost_stress=[]),
            path_evidence=mod.PathEvidence(top_k=int(kwargs["spec"].top_k), n_paths=28,
                                           path_win_rate=0.893, mean_path_delta_bp=8.04,
                                           pooled_delta_bp=10.97, p_paired_t=0.054),
            verdict="PASS_POST_REFORM", verdict_reasons=["ok"],
        )

    monkeypatch.setattr(mod, "load_and_prepare_price_history", _fake_load)
    monkeypatch.setattr(mod, "run_topk_ranker_backtest", _fake_run)

    out = tmp_path / "report.parquet"
    with caplog.at_level(logging.INFO, logger="src.ml.topk_ranker_research"):
        mod.main(["--price-history", str(ph_path), "--top-k", "5", "--out", str(out)])

    assert seen["top_k"] == 5
    assert out.exists()
    assert "PASS_POST_REFORM" in caplog.text

    with pytest.raises(ValueError, match="price_history not found"):
        mod.main(["--price-history", str(tmp_path / "nope.parquet"),
                  "--out", str(tmp_path / "x.parquet")])

def _two_regime_prepared_panel() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Build a PIT-prepared panel straddling the 2023-01-25 tick reform.

    Eight KOSDAQ symbols priced 190,000 KRW carry a 100 KRW tick (5.26bp) and
    clear the 7.5bp cap in BOTH regimes, so pre-reform rows enter the select
    pool on cost grounds alone -- only the training window can exclude them.
    Four KOSPI symbols priced 30,000 KRW (16.67bp) never clear the cap.
    Yields 312 pool rows over 26 days: 96 pre-reform and 112 post-reform
    select rows, 8 per day in each regime.
    """
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel

    pre_d = pd.bdate_range("2022-12-01", periods=12)
    post_d = pd.bdate_range("2023-02-01", periods=14)
    rng = np.random.default_rng(11)
    rows = []
    for i in range(12):
        base, mkt = (190000.0, "KOSDAQ") if i < 8 else (30000.0, "KOSPI")
        for d in list(pre_d) + list(post_d):
            prev = base / 1.05
            rows.append({
                "date": d, "symbol": f"{i:06d}", "open": prev, "high": base * 1.01,
                "low": prev * 0.99, "close": base, "prev_close": prev, "volume": 1e6,
                "market_cap_100m": 3000.0, "trade_value_100m": 500.0, "market": mkt,
                "daily_change_pct": 0.05,
                "inst_netbuy": float(rng.integers(-10**8, 10**8)),
                "foreign_netbuy": float(rng.integers(-10**8, 10**8)),
                "program_netbuy": 0.0, "kospi_pct": 0.001, "kosdaq_pct": 0.002,
                "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
    ph, _prov = prepare_price_panel(pd.DataFrame(rows))
    market_dates = np.array(sorted(ph["date"].unique()))
    return ph, market_dates, {d: i for i, d in enumerate(market_dates)}

def test_split_regime_frames_separates_certification_rows_from_history() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import CERT_REGIME_START, split_regime_frames

    # Given: rows straddling the 2023-01-25 tick reform
    frame = pd.DataFrame({
        "date": pd.to_datetime(
            ["2021-06-01", "2022-06-01", "2023-01-24", "2023-01-25", "2023-02-01"]
        ),
        "symbol": ["S1", "S2", "S3", "S4", "S5"],
        "train_label": [0.01, 0.01, 0.01, 0.01, 0.01],
    })

    # When: the default window keeps no pre-reform history
    cert, hist = split_regime_frames(frame, train_start=CERT_REGIME_START)

    # Then: the reform date itself certifies; history is empty
    assert cert["symbol"].tolist() == ["S4", "S5"]
    assert len(hist) == 0

    # When: the window is widened to 2022 (the knob is augmentation only)
    cert2, hist2 = split_regime_frames(frame, train_start=pd.Timestamp("2022-01-01"))

    # Then: the certification frame is UNCHANGED -- widening never moves the boundary
    assert cert2["symbol"].tolist() == ["S4", "S5"]
    assert hist2["symbol"].tolist() == ["S2", "S3"]

    # Then: a train_start after the certification start is rejected, since it would
    # silently shrink the certification regime instead of augmenting training
    with pytest.raises(ValueError, match="train_start"):
        split_regime_frames(frame, train_start=pd.Timestamp("2023-06-01"))

    # Then: a missing date column fails closed
    with pytest.raises(ValueError, match="date"):
        split_regime_frames(frame.drop(columns=["date"]), train_start=CERT_REGIME_START)

def test_cpcv_score_with_history_bins_on_certification_rows_only() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import cpcv_score_with_history

    rng = np.random.default_rng(5)

    def _rows(dates: pd.DatetimeIndex) -> pd.DataFrame:
        out = []
        for d in dates:
            for s in range(4):
                out.append({"date": d, "symbol": f"{s:06d}",  # noqa: PERF401 - spec skeleton
                            "f1": float(rng.normal()), "f2": float(rng.normal()),
                            "train_label": float(rng.normal()) * 0.01})
        return pd.DataFrame(out)

    cert = _rows(pd.bdate_range("2023-02-01", periods=10))
    hist = _rows(pd.bdate_range("2022-06-01", periods=20))
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=0)

    # When
    oof = cpcv_score_with_history(cert, hist, ["f1", "f2"], cv=cv, min_train_rows=10)

    # Then: every scored row is a certification row -- history trains, never scores
    assert len(oof) > 0
    assert (pd.to_datetime(oof["date"]) >= pd.Timestamp("2023-02-01")).all()
    assert oof["cpcv_fold"].nunique() == 28
    assert np.isfinite(oof["pred"].to_numpy(dtype=np.float64)).all()

    # Then: an empty history frame is the ordinary default path, not an error
    oof_no_hist = cpcv_score_with_history(cert, hist.iloc[0:0], ["f1", "f2"], cv=cv, min_train_rows=10)
    assert oof_no_hist["cpcv_fold"].nunique() == 28

    # Then: a starved training pool fails closed rather than training on noise
    with pytest.raises(ValueError, match="min_train_rows"):
        cpcv_score_with_history(cert, hist, ["f1", "f2"], cv=cv, min_train_rows=100_000)

def test_compute_path_evidence_rejects_pre_certification_regime_rows() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import CERT_REGIME_START, compute_path_evidence

    # Given: the exact defect that produced the shipped FAIL -- pre-reform rows,
    # where the select pool is ~1.8 names/day, pooled into the fold deltas.
    rows = []
    for fold in (0, 1):
        for day in ("2022-12-01", "2023-02-01"):
            for sym, tick, net, pred in (
                ("S1", 4.0, -0.010, 0.001),
                ("S2", 9.0, 0.020, 0.009),
                ("S3", 5.0, -0.010, 0.002),
            ):
                rows.append({"date": pd.Timestamp(day), "symbol": sym, "cpcv_fold": fold,
                             "tick_cost_bp": tick, "net_pit": net, "pred": pred})
    polluted = pd.DataFrame(rows)

    # Then: fail closed naming the regime boundary, never silently pool
    with pytest.raises(ValueError, match="2023-01-25|certification regime"):  # noqa: RUF043 - spec skeleton alternation
        compute_path_evidence(polluted, top_k=3)

    # When: only certification-regime rows are supplied
    clean = polluted[pd.to_datetime(polluted["date"]) >= CERT_REGIME_START]
    ev = compute_path_evidence(clean, top_k=1)

    # Then: it scores normally -- ranker takes S2 (+200bp), control takes S1 (-100bp)
    assert ev.n_paths == 2
    assert ev.path_win_rate == 1.0
    assert round(ev.mean_path_delta_bp, 2) == 300.0

def test_compute_path_evidence_reports_fold_accounting_and_fails_on_thin_coverage() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import compute_path_evidence

    def _frame(n_folds: int, n_blank: int) -> pd.DataFrame:
        rows = []
        for fold in range(n_folds):
            blank = fold < n_blank
            for sym, tick, net, pred in (
                ("S1", 4.0, -0.010, 0.001),
                ("S2", 9.0, 0.020, 0.009),
                ("S3", 5.0, -0.010, 0.002),
            ):
                rows.append({
                    "date": pd.Timestamp("2023-02-01"), "symbol": sym, "cpcv_fold": fold,
                    "tick_cost_bp": tick, "pred": pred,
                    # a blank fold carries no usable label, mirroring folds whose
                    # test bins hold no scorable certification-regime day
                    "net_pit": np.nan if blank else net,
                })
        return pd.DataFrame(rows)

    # When: 10 folds, 1 unscorable -> exactly at the 0.90 floor
    ev = compute_path_evidence(_frame(10, 1), top_k=1)

    # Then: the silent drop is now visible in the payload
    assert ev.n_folds_total == 10
    assert ev.n_folds_scored == 9
    assert ev.n_paths == 9
    assert ev.path_win_rate == 1.0

    # Then: below the floor it fails closed instead of certifying on a thinned path set
    with pytest.raises(ValueError, match="scored"):
        compute_path_evidence(_frame(10, 4), top_k=1)

def test_run_topk_ranker_backtest_widened_window_never_moves_the_boundary() -> None:
    import pandas as pd

    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import run_topk_ranker_backtest

    ph, market_dates, d_to_idx = _two_regime_prepared_panel()
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=0)

    # When: an operator widens the window past the reform (probe: every window passes)
    report = run_topk_ranker_backtest(
        ph, market_dates, d_to_idx, cv=cv, min_train_rows=10,
        train_start=pd.Timestamp("2000-01-01"),
    )

    # Then: the requested window is recorded verbatim for audit
    assert report.train_start == "2000-01-01"
    # Then: the certification boundary and the fold count are UNCHANGED --
    # widening augments training only, it never re-bins or re-scopes the verdict
    assert report.certification_regime_start == "2023-01-25"
    assert report.path_evidence.n_folds_total == 28
    assert report.path_evidence.n_folds_scored == 28
    assert report.ranker.regimes["pre_reform"].n_days_with_signal == 0

def _synthetic_bundle_and_fixture() -> tuple[dict, "pd.DataFrame"]:  # noqa: UP037 - spec skeleton
    """A minimal 4-feature production bundle plus a matching 2-symbol snapshot.

    Trains via the real build_inline_bundle recipe on synthetic data so the
    bundle satisfies load_model_bundle's schema (rank_model/quantile_models/
    calibrators) exactly like a real production bundle would.
    """
    import numpy as np
    import pandas as pd

    from src.ml.bundle import build_inline_bundle

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-02-01", periods=30)
    rows = []
    for d in dates:
        for s in range(5):
            rows.append({  # noqa: PERF401 - spec skeleton
                "date": d, "symbol": f"{s:06d}",
                "chg_ratio": float(rng.uniform(0.02, 0.10)),
                "log_tv": float(rng.normal(6, 1)),
                "train_label": float(rng.normal(0, 0.01)),
            })
    df = pd.DataFrame(rows)
    bundle = build_inline_bundle(df, ["chg_ratio", "log_tv"], "train_label", "date")
    bundle["feature_cols"] = ["chg_ratio", "log_tv"]

    snapshot = pd.DataFrame({
        "date": pd.Timestamp("2023-03-15"),
        "symbol": ["000001", "000002", "000003"],
        "chg_ratio": [0.05, 0.03, 0.08],
        "log_tv": [6.1, 5.9, 6.5],
    })
    return bundle, snapshot

def test_train_production_bundle_trains_on_certification_regime_wide_pool() -> None:
    import pandas as pd

    from src.ml.topk_ranker_research import CERT_REGIME_START, train_production_bundle

    ph, market_dates, d_to_idx = _two_regime_prepared_panel()

    # When
    bundle = train_production_bundle(ph, market_dates, d_to_idx, min_train_rows=10)

    # Then: satisfies load_model_bundle's schema and carries audit provenance
    assert bundle["feature_cols"]
    for key in ("rank_model", "return_model", "quantile_models", "calibrators"):
        assert key in bundle
    assert bundle["train_start"] == str(pd.to_datetime(ph["date"]).min().date())
    assert bundle["certification_regime_start"] == str(CERT_REGIME_START.date())
    assert bundle["top_k"] == 3
    assert bundle["select_universe"]["max_tick_cost_bp"] is not None

def test_train_production_bundle_fails_closed_below_min_train_rows() -> None:
    import pytest

    from src.ml.topk_ranker_research import CERT_REGIME_START, train_production_bundle

    ph, market_dates, d_to_idx = _synthetic_prepared_panel()

    with pytest.raises(ValueError, match="min_train_rows"):
        train_production_bundle(ph, market_dates, d_to_idx, min_train_rows=1_000_000, train_start=CERT_REGIME_START)

def test_save_production_bundle_writes_joblib_loadable_by_load_model_bundle(tmp_path) -> None:
    from src.serving.realtime.artifacts import load_model_bundle
    from src.ml.topk_ranker_research import CERT_REGIME_START, save_production_bundle, train_production_bundle

    ph, market_dates, d_to_idx = _synthetic_prepared_panel()
    bundle = train_production_bundle(ph, market_dates, d_to_idx, min_train_rows=10, train_start=CERT_REGIME_START)
    export_dir = str(tmp_path / "topk_ranker")

    # When
    path = save_production_bundle(bundle, export_dir=export_dir)

    # Then: the artifact round-trips through the EXISTING, unmodified loader
    assert path.startswith(export_dir)
    reloaded = load_model_bundle(import_dir=export_dir)
    assert reloaded["feature_cols"] == bundle["feature_cols"]
    assert reloaded["top_k"] == 3

def test_select_topk_equal_weight_selects_by_rank_score_and_allocates_equally() -> None:
    import numpy as np

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import select_topk_equal_weight

    bundle, snapshot = _synthetic_bundle_and_fixture()

    # When: top_k matches the certified minimum (3 == the fixture's 3 rows)
    out = select_topk_equal_weight(snapshot, bundle, top_k=MIN_TOP_K)

    # Then: every row is kept (only 3 candidates for k=3), scored, and
    # equally weighted
    assert len(out) == 3
    assert "pred" in out.columns
    assert np.allclose(out["allocation"].to_numpy(dtype=np.float64), 1.0 / MIN_TOP_K)
    # Then: diagnostic-only quantile/probability columns are attached but
    # never drive selection or allocation
    for col in ("pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad"):
        assert col in out.columns
    # Then: rows are ordered by descending pred (highest score first)
    preds = out["pred"].to_numpy(dtype=np.float64)
    assert (preds[:-1] >= preds[1:]).all()

def test_select_topk_equal_weight_rejects_non_certified_top_k() -> None:
    import pytest

    from src.ml.topk_ranker_research import select_topk_equal_weight

    bundle, snapshot = _synthetic_bundle_and_fixture()

    with pytest.raises(ValueError, match="top_k"):
        select_topk_equal_weight(snapshot, bundle, top_k=1)

def test_select_topk_equal_weight_rejects_missing_feature_columns() -> None:
    import pytest

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import select_topk_equal_weight

    # Given: the live snapshot is missing one of the bundle's declared features
    bundle, snapshot = _synthetic_bundle_and_fixture()
    thin = snapshot.drop(columns=["log_tv"])

    # When / Then: fail closed naming the missing column instead of zero-filling
    with pytest.raises(ValueError, match="log_tv"):
        select_topk_equal_weight(thin, bundle, top_k=MIN_TOP_K)

def test_select_topk_equal_weight_selects_only_admitted_rows() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import select_topk_equal_weight

    # Given: a wide 5-row cross-section where only 3 rows cleared admission
    bundle, _ = _synthetic_bundle_and_fixture()
    wide = pd.DataFrame({
        "date": [pd.Timestamp("2023-03-15")] * 5,
        "symbol": ["000001", "000002", "000003", "000004", "000005"],
        "chg_ratio": [0.05, 0.03, 0.08, 0.04, 0.09],
        "log_tv": [6.1, 5.9, 6.5, 6.3, 6.7],
        "admitted": [True, False, True, True, False],
    })

    # When
    out = select_topk_equal_weight(wide, bundle, top_k=MIN_TOP_K)

    # Then: exactly the admitted names, equally weighted
    assert sorted(out["symbol"].tolist()) == ["000001", "000003", "000004"]
    assert np.allclose(out["allocation"].to_numpy(dtype=np.float64), 1.0 / MIN_TOP_K)

def test_select_topk_equal_weight_excludes_dates_below_min_admitted() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import select_topk_equal_weight

    # Given: D1 has 3 admitted names, D2 has only 2 (below the certified K)
    bundle, _ = _synthetic_bundle_and_fixture()
    d1, d2 = pd.Timestamp("2023-03-15"), pd.Timestamp("2023-03-16")
    wide = pd.DataFrame({
        "date": [d1, d1, d1, d2, d2, d2],
        "symbol": ["000001", "000002", "000003", "000004", "000005", "000006"],
        "chg_ratio": [0.05, 0.03, 0.08, 0.04, 0.09, 0.06],
        "log_tv": [6.1, 5.9, 6.5, 6.3, 6.7, 6.2],
        "admitted": [True, True, True, True, True, False],
    })

    # When
    out = select_topk_equal_weight(wide, bundle, top_k=MIN_TOP_K)

    # Then: D2 is dropped entirely rather than entered with a shrunk denominator
    assert out["date"].nunique() == 1
    assert out["date"].iloc[0] == d1
    assert len(out) == MIN_TOP_K
    assert np.allclose(out["allocation"].to_numpy(dtype=np.float64), 1.0 / MIN_TOP_K)

def test_train_production_bundle_rejects_top_k_below_min() -> None:
    import dataclasses

    import pytest

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import train_production_bundle
    from src.strategy.contract import KCA_TOPK_COSTAWARE_001

    ph, market_dates, d_to_idx = _synthetic_prepared_panel()
    thin_spec = dataclasses.replace(KCA_TOPK_COSTAWARE_001, top_k=MIN_TOP_K - 1)

    with pytest.raises(ValueError, match="top_k"):
        train_production_bundle(ph, market_dates, d_to_idx, spec=thin_spec, min_train_rows=10)

def test_select_topk_equal_weight_rejects_empty_feature_cols() -> None:
    import pytest

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import select_topk_equal_weight

    bundle, snapshot = _synthetic_bundle_and_fixture()
    bundle["feature_cols"] = []

    with pytest.raises(ValueError, match="feature_cols"):
        select_topk_equal_weight(snapshot, bundle, top_k=MIN_TOP_K)

def test_select_topk_equal_weight_uses_float_fallback_for_degenerate_calibrator() -> None:
    import pandas as pd

    from src.ml.bundle import build_inline_bundle
    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import select_topk_equal_weight

    # Given: every training label sits below the p_good threshold (0.01) and
    # above the p_bad threshold (-0.02) -- both calibrators degenerate to a
    # constant float fallback (single-class labels, build_inline_bundle's
    # _fit_calibrator_cv contract)
    dates = pd.bdate_range("2023-02-01", periods=10)
    rows = [
        {"date": d, "symbol": f"{s:06d}", "chg_ratio": 0.05, "log_tv": 6.0, "train_label": -0.005}
        for d in dates for s in range(5)
    ]
    df = pd.DataFrame(rows)
    bundle = build_inline_bundle(df, ["chg_ratio", "log_tv"], "train_label", "date")
    bundle["feature_cols"] = ["chg_ratio", "log_tv"]
    assert isinstance(bundle["calibrators"]["p_good"], float)
    assert isinstance(bundle["calibrators"]["p_bad"], float)

    snapshot = pd.DataFrame({
        "date": pd.Timestamp("2023-03-15"),
        "symbol": ["000001", "000002", "000003"],
        "chg_ratio": [0.05, 0.03, 0.08],
        "log_tv": [6.1, 5.9, 6.5],
    })

    # When
    out = select_topk_equal_weight(snapshot, bundle, top_k=MIN_TOP_K)

    # Then: the constant float fallback is broadcast to every row, never a
    # classifier predict_proba call
    assert (out["p_good"] == bundle["calibrators"]["p_good"]).all()
    assert (out["p_bad"] == bundle["calibrators"]["p_bad"]).all()


def test_assert_nested_universe_specs_accepts_a_capfree_select_spec() -> None:
    import dataclasses

    import pytest

    from src.ml.topk_ranker_research import assert_nested_universe_specs
    from src.strategy.contract import (
        CAPFREE_UNIVERSE,
        COST_AWARE_UNIVERSE,
        DEFAULT_UNIVERSE,
    )

    # Given / When: a cap-free select spec equal to the wide train spec
    # Then: it is trivially nested and accepted
    assert assert_nested_universe_specs(DEFAULT_UNIVERSE, CAPFREE_UNIVERSE) is None
    # And: the capped spec stays nested too
    assert assert_nested_universe_specs(DEFAULT_UNIVERSE, COST_AWARE_UNIVERSE) is None

    # And: a non-finite cap is still refused
    with pytest.raises(ValueError, match="max_tick_cost_bp"):
        assert_nested_universe_specs(
            DEFAULT_UNIVERSE,
            dataclasses.replace(COST_AWARE_UNIVERSE, max_tick_cost_bp=float("inf")),
        )

    # And: a screen field that differs is still refused by name
    with pytest.raises(ValueError, match="chg_min"):
        assert_nested_universe_specs(
            DEFAULT_UNIVERSE, dataclasses.replace(CAPFREE_UNIVERSE, chg_min=0.03)
        )


def test_attach_pit_net_label_nets_with_the_point_in_time_statutory_rate() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.topk_ranker_research import attach_pit_net_label
    from src.strategy.contract import AA_COST

    # Given: identical gross and tick cost in the 2018 and 2025 tax regimes
    cands = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-01", "2025-06-02"]),
            "symbol": ["000001", "000002"],
            "gross_return": [0.20, 0.01],
            "tick_cost_bp": [5.0, 5.0],
        }
    )

    # When: attaching the PIT net label
    out = attach_pit_net_label(cands, cost=AA_COST, label_clip=0.10)

    # Then: the statutory leg is 30bp in 2018 and 15bp in 2025
    np.testing.assert_allclose(
        out["net_pit"].to_numpy(), [0.20 - 40.0 / 1e4, 0.01 - 25.0 / 1e4]
    )
    # And: the training label is clipped symmetrically
    assert out["train_label"].to_numpy()[0] == pytest.approx(0.10)

    # And: every required column is guarded by name
    for missing in ("tick_cost_bp", "gross_return", "date"):
        with pytest.raises(ValueError, match=missing):
            attach_pit_net_label(cands.drop(columns=[missing]), cost=AA_COST)


def test_evaluate_falsification_refutes_a_profitable_pre_reform_regime() -> None:
    from src.ml.costaware_topk import RegimeMetrics
    from src.ml.topk_ranker_research import ArmMetrics, evaluate_falsification

    def _regime(regime: str, *, n_signal: int, mean_bp: float, t_stat: float) -> RegimeMetrics:
        return RegimeMetrics(
            regime=regime,
            n_calendar_days=n_signal,
            n_days_with_signal=n_signal,
            n_days_feasible=n_signal,
            feasible_day_fraction=1.0,
            coverage_status="OK",
            mean_net_bp=mean_bp,
            median_net_bp=mean_bp,
            std_net_bp=100.0,
            win_rate=0.55,
            t_stat=t_stat,
            sharpe=1.0,
            ci_low_bp=mean_bp - 5.0,
            ci_high_bp=mean_bp + 5.0,
            dsr=0.5,
        )

    def _arm(pre: RegimeMetrics) -> ArmMetrics:
        return ArmMetrics(
            arm="costsort_full",
            top_k=3,
            regimes={
                "pre_reform": pre,
                "post_reform": _regime("post_reform", n_signal=880, mean_bp=20.0, t_stat=3.3),
                "full_history": _regime("full_history", n_signal=2600, mean_bp=5.0, t_stat=1.0),
            },
            by_year=[],
            cost_stress=[],
        )

    # Given: a 1,735-day pre-reform sample that certifies profit under PIT cost
    status, reasons = evaluate_falsification(
        _arm(_regime("pre_reform", n_signal=1735, mean_bp=12.0, t_stat=3.1))
    )

    # Then: the 46bp-cost regime cannot be profitable, so the run is refuted
    assert status == "REFUTED"
    assert any("pre_reform" in r for r in reasons)

    # When: the measured reality (-10bp/day) is reported instead
    status_ok, reasons_ok = evaluate_falsification(
        _arm(_regime("pre_reform", n_signal=1735, mean_bp=-10.2, t_stat=-2.7))
    )

    # Then: the cost model and the label are consistent
    assert status_ok == "CONSISTENT"
    assert reasons_ok == []

    # When: the pre-reform sample is too short to judge
    status_thin, reasons_thin = evaluate_falsification(
        _arm(_regime("pre_reform", n_signal=100, mean_bp=12.0, t_stat=3.1))
    )

    # Then: it reports insufficiency rather than passing or refuting silently
    assert status_thin == "INSUFFICIENT_PRE_REFORM_SAMPLE"
    assert any("min_days" in r for r in reasons_thin)


def test_evaluate_ranker_verdict_short_circuits_on_a_refuted_falsification() -> None:
    from src.ml.costaware_topk import CostStressPoint, RegimeMetrics
    from src.ml.topk_ranker_research import ArmMetrics, PathEvidence, evaluate_ranker_verdict

    def _regime(mean_bp: float, t_stat: float) -> RegimeMetrics:
        return RegimeMetrics(
            regime="post_reform",
            n_calendar_days=880,
            n_days_with_signal=880,
            n_days_feasible=880,
            feasible_day_fraction=1.0,
            coverage_status="OK",
            mean_net_bp=mean_bp,
            median_net_bp=mean_bp,
            std_net_bp=100.0,
            win_rate=0.55,
            t_stat=t_stat,
            sharpe=1.8,
            ci_low_bp=mean_bp - 5.0,
            ci_high_bp=mean_bp + 5.0,
            dsr=0.6,
        )

    def _arm(mean_bp: float, t_stat: float, arm: str) -> ArmMetrics:
        return ArmMetrics(
            arm=arm,
            top_k=3,
            regimes={"post_reform": _regime(mean_bp, t_stat)},
            by_year=[],
            cost_stress=[
                CostStressPoint(
                    regime="post_reform",
                    round_trip_ticks=3.0,
                    n_days=880,
                    mean_net_bp=mean_bp - 5.0,
                    median_net_bp=mean_bp - 5.0,
                    t_stat=2.4,
                    passes=True,
                )
            ],
        )

    ranker = _arm(24.0, 3.3, "ranker")
    control = _arm(19.5, 3.0, "costsort")
    evidence = PathEvidence(
        top_k=3,
        n_paths=28,
        path_win_rate=0.75,
        mean_path_delta_bp=4.0,
        pooled_delta_bp=4.5,
        p_paired_t=0.01,
        n_folds_total=28,
        n_folds_scored=28,
    )

    # Given: every post-reform gate passes
    verdict_ok, _ = evaluate_ranker_verdict(ranker, control, evidence)
    assert verdict_ok == "PASS_POST_REFORM"

    # When: the out-of-regime falsification tier refutes the cost model
    verdict, reasons = evaluate_ranker_verdict(
        ranker, control, evidence, falsification_status="REFUTED"
    )

    # Then: it short-circuits before the statistics, so nothing can ship
    assert verdict == "REFUTED"
    assert reasons == [
        "falsification tier REFUTED: the selection rule certifies positive net "
        "return in the pre-reform cost regime"
    ]


def test_run_topk_ranker_backtest_widens_training_and_reports_falsification() -> None:
    import pandas as pd

    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import (
        CERT_REGIME_START,
        run_topk_ranker_backtest,
        topk_ranker_report_to_frame,
    )

    # Given: a PIT-prepared panel straddling the 2023-01-25 tick reform
    ph, market_dates, d_to_idx = _two_regime_prepared_panel()

    # When: running with the default training window
    report = run_topk_ranker_backtest(
        ph,
        market_dates,
        d_to_idx,
        cv=CombinatorialPurgedCV(n_groups=4, k_test=2, purge_gap=0, embargo_gap=0),
        min_train_rows=1,
    )

    # Then: training starts at the panel minimum, not at the certification start
    assert report.train_start == str(pd.to_datetime(ph["date"]).min().date())
    assert report.train_start < str(CERT_REGIME_START.date())
    # And: the certification boundary is unmoved by the widening
    assert report.certification_regime_start == str(CERT_REGIME_START.date())

    # And: the falsification tier is evaluated and recorded on the artifact
    assert report.falsification_status in {
        "CONSISTENT",
        "REFUTED",
        "INSUFFICIENT_PRE_REFORM_SAMPLE",
    }
    assert isinstance(report.falsification_reasons, list)
    # And: the capped select screen declares its post-reform constructibility
    assert set(report.screen_day_fractions) == {"post_reform"}
    assert report.screen_day_fractions["post_reform"] >= 0.95

    # And: both land in the persisted artifact so the gate is auditable
    frame = topk_ranker_report_to_frame(report)
    ev_row = frame[frame["row_type"] == "path_evidence"].iloc[0]
    assert ev_row["falsification_status"] == report.falsification_status
    screen_rows = frame[frame["row_type"] == "screen_constructibility"]
    assert screen_rows["regime"].tolist() == ["post_reform"]
    assert screen_rows["constructible_day_fraction"].iloc[0] >= 0.95


def test_run_topk_ranker_backtest_accepts_the_capfree_arm() -> None:
    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import run_topk_ranker_backtest
    from src.strategy.contract import KCA_TOPK_CAPFREE_001

    # Given: the same two-regime panel
    ph, market_dates, d_to_idx = _two_regime_prepared_panel()

    # When: running the cap-free arm
    report = run_topk_ranker_backtest(
        ph,
        market_dates,
        d_to_idx,
        spec=KCA_TOPK_CAPFREE_001,
        cv=CombinatorialPurgedCV(n_groups=4, k_test=2, purge_gap=0, embargo_gap=0),
        min_train_rows=1,
    )

    # Then: no cap means the select pool is the whole train pool
    assert report.n_select_rows == report.n_train_rows
    assert report.select_universe["max_tick_cost_bp"] is None
    # And: constructibility is asserted in both regimes, unlike the capped arm
    assert set(report.screen_day_fractions) == {"pre_reform", "post_reform"}
    assert min(report.screen_day_fractions.values()) >= 0.95


def test_topk_ranker_main_selects_the_capfree_arm_and_logs_falsification(
    tmp_path, monkeypatch, caplog
) -> None:
    import logging

    import pandas as pd

    import src.ml.topk_ranker_research as mod
    from src.strategy.contract import KCA_TOPK_CAPFREE_001, KCA_TOPK_COSTAWARE_001

    # Given: a stubbed backtest that records the spec the CLI chose
    seen: dict[str, object] = {}

    def _fake_backtest(ph, market_dates, d_to_idx, *, spec, **kwargs):
        seen["strategy_id"] = spec.strategy_id
        seen["train_start"] = kwargs.get("train_start")
        return mod.TopKRankerReport(
            strategy_id=spec.strategy_id,
            top_k=spec.top_k,
            train_universe={},
            select_universe={},
            cost={},
            date_min="2016-01-04",
            date_max="2026-09-04",
            n_train_rows=1,
            n_select_rows=1,
            ranker=mod.ArmMetrics(arm="ranker", top_k=spec.top_k, regimes={}, by_year=[], cost_stress=[]),
            control=mod.ArmMetrics(arm="costsort", top_k=spec.top_k, regimes={}, by_year=[], cost_stress=[]),
            path_evidence=mod.PathEvidence(
                top_k=spec.top_k,
                n_paths=1,
                path_win_rate=1.0,
                mean_path_delta_bp=1.0,
                pooled_delta_bp=1.0,
                p_paired_t=0.01,
            ),
            verdict="PASS_POST_REFORM",
            verdict_reasons=[],
            train_start="2016-01-04",
            certification_regime_start="2023-01-25",
            falsification_status="CONSISTENT",
            falsification_reasons=[],
            screen_day_fractions={"pre_reform": 1.0, "post_reform": 1.0},
        )

    monkeypatch.setattr(mod, "run_topk_ranker_backtest", _fake_backtest)
    monkeypatch.setattr(
        mod,
        "load_and_prepare_price_history",
        lambda path: (pd.DataFrame({"date": pd.to_datetime(["2016-01-04"])}), [], {}),
    )
    ph_path = tmp_path / "ph.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2016-01-04"])}).to_parquet(ph_path)
    out = tmp_path / "report.parquet"

    # When: the CLI is invoked with the cap-free flag
    with caplog.at_level(logging.INFO, logger="src.ml.topk_ranker_research"):
        mod.main(["--price-history", str(ph_path), "--capfree", "--out", str(out)])

    # Then: the cap-free arm was selected and the falsification status is logged
    assert seen["strategy_id"] == KCA_TOPK_CAPFREE_001.strategy_id
    assert "falsification=CONSISTENT" in caplog.text
    assert out.exists()

    # And: without the flag the shipped capped arm is still the default
    caplog.clear()
    mod.main(["--price-history", str(ph_path), "--out", str(tmp_path / "r2.parquet")])
    assert seen["strategy_id"] == KCA_TOPK_COSTAWARE_001.strategy_id


def test_train_production_bundle_matches_research_training_window() -> None:
    import pandas as pd

    from src.ml.topk_ranker_research import CERT_REGIME_START, train_production_bundle

    # Given: the two-regime prepared panel
    ph, market_dates, d_to_idx = _two_regime_prepared_panel()

    # When: training the production bundle with the default window
    bundle = train_production_bundle(ph, market_dates, d_to_idx, min_train_rows=1)

    # Then: it trains from the panel minimum, exactly like the research harness
    assert bundle["train_start"] == str(pd.to_datetime(ph["date"]).min().date())
    # And: the certification boundary is still recorded and unmoved
    assert bundle["certification_regime_start"] == str(CERT_REGIME_START.date())
