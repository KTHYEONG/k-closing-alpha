"""Decision-time (15:20) re-certification tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_TINY_PARAMS = {
    "n_estimators": 10,
    "num_leaves": 7,
    "min_child_samples": 5,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "num_threads": 1,
}


def _synth_inputs(n_days=120, n_symbols=40, seed=11, auction=False, start="2023-02-01"):
    from src.data.panel_integrity import prepare_price_panel

    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)
    raw_rows: list[dict] = []
    panel_rows: list[dict] = []
    prev_gap: dict = {}
    for i in range(n_symbols):
        symbol = f"{i:06d}"
        base = 15000.0 + i * 50.0
        prev = base / 1.05
        lane = float(i) / max(1, n_symbols - 1) - 0.5
        for day in dates:
            if auction:
                s = float(rng.uniform(-0.005, 0.005))
                am = float(rng.choice([-0.01, 0.01]) + rng.normal(0.0, 0.002))
                share = float(rng.uniform(0.05, 0.25))
                gap = float(prev_gap.get(symbol, 0.0))
            else:
                s = lane * 0.004
                am, share = 0.0, 0.0
                gap = lane * 0.008
            close_1520 = prev * 1.05 * (1.0 + s)
            close_eod = close_1520 * (1.0 + am)
            open_t = prev * (1.0 + gap)
            high = max(open_t, close_eod) * 1.01
            low = min(open_t, close_eod) * 0.99
            tv_1520 = 500.0
            tv_eod = tv_1520 / (1.0 - share) if share else tv_1520
            inst = float(rng.integers(-10**8, 10**8))
            foreign = float(rng.integers(-10**8, 10**8))
            raw_rows.append({
                "date": day, "symbol": symbol, "open": open_t, "high": high, "low": low,
                "close": close_eod, "close_raw": close_eod, "prev_close": prev, "volume": 1e6,
                "market_cap_100m": 3000.0, "trade_value_100m": tv_eod, "market": "KOSPI",
                "daily_change_pct": 5.0, "inst_netbuy": inst, "foreign_netbuy": foreign,
                "program_netbuy": 0.0, "kospi_pct": 0.001, "kosdaq_pct": 0.002,
                "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
            panel_rows.append({
                "date": day, "symbol": symbol, "market": "KOSPI", "open": open_t,
                "high": max(open_t, close_1520) * 1.01, "low": min(open_t, close_1520) * 0.99,
                "close": close_1520, "close_raw": close_1520, "prev_close": prev, "volume": 1e6,
                "trade_value_100m": tv_1520, "market_cap_100m": 3000.0,
                "inst_netbuy": inst, "foreign_netbuy": foreign,
                "inst_netbuy_prev": 0.0, "foreign_netbuy_prev": 0.0,
                "kospi_pct": 0.001, "kosdaq_pct": 0.002, "v_kospi": 18.0, "v_kosdaq": 22.0,
                "index_basis": "live_1520", "source": "live_decision", "bars_vendor": "",
                "n_bars": 0, "first_bar_hms": "", "last_bar_hms": "", "capture_run_id": "test",
            })
            prev_gap[symbol] = 2.0 * am + float(rng.normal(0.0, 0.002)) if auction else 0.0
    ph, _prov = prepare_price_panel(pd.DataFrame(raw_rows))
    ph["is_screenable"] = True
    ph["screenable_source"] = "real"
    market_dates = np.array(sorted(ph["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}
    from src.data.pit1520_panel import PIT1520_PANEL_COLUMNS

    panel = pd.DataFrame(panel_rows)
    panel = panel[list(PIT1520_PANEL_COLUMNS)]
    panel_days = pd.DataFrame({
        "date": list(dates),
        "source": ["live_decision"] * len(dates),
        "n_rows": [n_symbols] * len(dates),
        "n_excluded": [0] * len(dates),
        "n_superset": [n_symbols] * len(dates),
        "n_superset_present": [n_symbols] * len(dates),
        "superset_coverage": [1.0] * len(dates),
        "index_basis": ["live_1520"] * len(dates),
    })
    from src.ml.topk_history_features import HISTORY_REQUIRED_COLUMNS

    serving_history = ph[list(HISTORY_REQUIRED_COLUMNS)].copy()
    return ph, market_dates, d_to_idx, panel, panel_days, serving_history


def _cert_config(**overrides):
    from src.ml.research.pit_certification import PitCertificationConfig

    kwargs = {"bootstrap_n_boot": 500}
    kwargs.update(overrides)
    return PitCertificationConfig(**kwargs)


def _run(synth, cv=None, **overrides):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.ml.research.pit_certification import run_pit_certification
    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import CERT_REGIME_START

    ph, market_dates, d_to_idx, panel, panel_days, serving_history = synth
    kwargs = {
        "cv": cv or CombinatorialPurgedCV(n_groups=4, k_test=2, purge_gap=0, embargo_gap=0),
        "model_params": dict(_TINY_PARAMS),
        "seeds": (1,),
        "train_start": CERT_REGIME_START,
        "config": _cert_config(),
        "now": datetime(2026, 10, 4, tzinfo=ZoneInfo("Asia/Seoul")),
    }
    kwargs.update(overrides)
    return run_pit_certification(
        ph, market_dates, d_to_idx, panel=panel, panel_days=panel_days,
        serving_history=serving_history, **kwargs,
    )


@pytest.mark.slow
def test_run_pit_certification_identical_states_zero_haircut() -> None:
    report, daily = _run(_synth_inputs())

    assert report.status.value == "OK"
    assert report.haircut.delta == 0.0
    assert report.pick_overlap_mean["pit_native"] == 1.0
    assert report.pick_overlap_mean["pit_feature"] == 1.0
    eod = daily[daily["arm"] == "eod_full"].set_index("date")["topk_net_bp"]
    nat = daily[daily["arm"] == "pit_native"].set_index("date")["topk_net_bp"]
    pd.testing.assert_series_equal(eod, nat, check_names=False)


@pytest.mark.slow
def test_run_pit_certification_detects_eod_optimism() -> None:
    report, _daily = _run(_synth_inputs(auction=True))

    assert report.status.value == "OK"
    assert report.haircut.delta > 0.0
    assert report.rank_ic_mean["eod_full"] > report.rank_ic_mean["pit_native"]


@pytest.mark.slow
def test_run_pit_certification_components_sum_to_haircut() -> None:
    report, _daily = _run(_synth_inputs())

    total = (
        report.coverage_component.delta
        + report.feature_component.delta
        + report.selection_component.delta
    )
    assert report.haircut.delta == total or abs(total - report.haircut.delta) <= 1e-9


@pytest.mark.slow
def test_run_pit_certification_scores_pit_rows_out_of_sample(monkeypatch) -> None:
    import src.ml.research.pit_certification as pit_mod
    from src.ml.robust_eval import CombinatorialPurgedCV

    real_cpcv = pit_mod.cpcv_score_with_history
    captured: dict = {}

    def _spy_cpcv(*args, **kwargs):
        captured["observer"] = kwargs["fold_observer"]
        return real_cpcv(*args, **kwargs)

    monkeypatch.setattr(pit_mod, "cpcv_score_with_history", _spy_cpcv)
    scored: list = []
    real_finite = pit_mod._finite_nan

    def _spy_finite(frame, cols):
        scored.append(set(pd.to_datetime(frame["date"]).dt.normalize().tolist()))
        return real_finite(frame, cols)

    monkeypatch.setattr(pit_mod, "_finite_nan", _spy_finite)
    synth = _synth_inputs(n_days=60, n_symbols=80)
    report, _daily = _run(synth)
    assert report.status.value == "OK"

    from src.ml.topk_ranker_research import (
        CERT_REGIME_START,
        attach_pit_net_label,
        build_dual_pool,
        demean_label_by_date,
        split_regime_frames,
    )
    from src.strategy.contract import DEFAULT_UNIVERSE, KCA_TOPK_COSTAWARE_001

    ph, _md, _d2i, _panel, _days, _hist = synth
    pool, _mask = build_dual_pool(
        ph, _md, _d2i, train_spec=DEFAULT_UNIVERSE, select_spec=KCA_TOPK_COSTAWARE_001.universe,
    )
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=KCA_TOPK_COSTAWARE_001.cost))
    cert_df, _hist_df = split_regime_frames(labeled, train_start=CERT_REGIME_START)
    cv = CombinatorialPurgedCV(n_groups=4, k_test=2, purge_gap=0, embargo_gap=0)
    _work, test_dates = pit_mod._fold_test_dates(cert_df, group_col="date", cv=cv)

    class _Zero:
        def predict(self, X):
            return np.zeros(len(X))

    for fold_id, fold_test in enumerate(test_dates):
        before = len(scored)
        captured["observer"](fold_id, pd.DataFrame({"date": sorted(fold_test)}), _Zero())
        for seen in scored[before:]:
            assert seen <= set(fold_test)


def test_build_pit_feature_frames_ignore_future_rows() -> None:
    from src.ml.research.pit_certification import build_pit_feature_frames
    from src.ml.topk_ranker_research import (
        attach_pit_net_label,
        build_dual_pool,
        demean_label_by_date,
    )
    from src.strategy.contract import DEFAULT_UNIVERSE, KCA_TOPK_COSTAWARE_001

    synth = _synth_inputs(n_days=20, n_symbols=12)
    ph, market_dates, d_to_idx, panel, panel_days, serving_history = synth
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=DEFAULT_UNIVERSE,
        select_spec=KCA_TOPK_COSTAWARE_001.universe,
    )
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=KCA_TOPK_COSTAWARE_001.cost))
    usable = sorted(pd.to_datetime(panel_days["date"]).dt.normalize().unique().tolist())[:15]

    feat_a, nat_a = build_pit_feature_frames(
        panel=panel, usable_days=usable, eod_pool=labeled, sel_mask=sel_mask,
        serving_history=serving_history, spec=KCA_TOPK_COSTAWARE_001,
    )
    rng = np.random.default_rng(0)
    future = pd.to_datetime(panel["date"]) > pd.Timestamp(usable[-1])
    panel_b = panel.copy()
    panel_b.loc[future, "close"] = panel_b.loc[future, "close"] * rng.uniform(0.9, 1.1, int(future.sum()))
    hist_b = serving_history.copy()
    hfuture = pd.to_datetime(hist_b["date"]) > pd.Timestamp(usable[-1])
    hist_b.loc[hfuture, "close"] = hist_b.loc[hfuture, "close"] * rng.uniform(0.9, 1.1, int(hfuture.sum()))
    feat_b, nat_b = build_pit_feature_frames(
        panel=panel_b, usable_days=usable, eod_pool=labeled, sel_mask=sel_mask,
        serving_history=hist_b, spec=KCA_TOPK_COSTAWARE_001,
    )
    key = ["date", "symbol"]
    pd.testing.assert_frame_equal(
        feat_a.sort_values(key).reset_index(drop=True), feat_b.sort_values(key).reset_index(drop=True))
    pd.testing.assert_frame_equal(
        nat_a.sort_values(key).reset_index(drop=True), nat_b.sort_values(key).reset_index(drop=True))


def test_build_pit_feature_frames_native_ranks_ignore_out_of_pool_rows() -> None:
    from src.ml.research.pit_certification import build_pit_feature_frames
    from src.ml.topk_ranker_research import attach_pit_net_label, build_dual_pool, demean_label_by_date
    from src.strategy.contract import KCA_TOPK_COSTAWARE_001, training_universe

    ph, market_dates, d_to_idx, panel, panel_days, serving_history = _synth_inputs(n_days=20, n_symbols=12)
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=training_universe(KCA_TOPK_COSTAWARE_001.universe),
        select_spec=KCA_TOPK_COSTAWARE_001.universe,
    )
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=KCA_TOPK_COSTAWARE_001.cost))
    usable = sorted(pd.to_datetime(panel_days["date"]).dt.normalize().unique().tolist())[10:15]
    outsiders = panel[pd.to_datetime(panel["date"]).isin(usable)].copy()
    outsiders["symbol"] = "9" + outsiders["symbol"].str[1:]
    outsiders["close"] = outsiders["prev_close"] * 1.15
    outsiders["close_raw"] = outsiders["close"]
    outsiders["high"] = outsiders["close"]
    widened = pd.concat([panel, outsiders], ignore_index=True)

    _f_a, nat_a = build_pit_feature_frames(
        panel=panel, usable_days=usable, eod_pool=labeled, sel_mask=sel_mask,
        serving_history=serving_history, spec=KCA_TOPK_COSTAWARE_001,
    )
    _f_b, nat_b = build_pit_feature_frames(
        panel=widened, usable_days=usable, eod_pool=labeled, sel_mask=sel_mask,
        serving_history=serving_history, spec=KCA_TOPK_COSTAWARE_001,
    )

    key = ["date", "symbol"]
    assert set(nat_b["symbol"]) == set(nat_a["symbol"])
    pd.testing.assert_frame_equal(
        nat_a.sort_values(key).reset_index(drop=True)[[*key, "chg_rank", "tv_rank"]],
        nat_b.sort_values(key).reset_index(drop=True)[[*key, "chg_rank", "tv_rank"]],
    )


def test_run_pit_certification_defaults_to_production_strategy() -> None:
    import inspect

    from src.ml.research.pit_certification import run_pit_certification
    from src.strategy.contract import PRODUCTION_STRATEGY

    params = inspect.signature(run_pit_certification).parameters
    assert params["spec"].default is PRODUCTION_STRATEGY
    assert params["train_spec"].default is None


def test_run_pit_certification_attaches_class_verdict_from_training_panel() -> None:
    from src.ml.research.pit_certification import _with_screenable_class

    ph, _md, _idx, panel, _days, _hist = _synth_inputs(n_days=3, n_symbols=3)
    ph.loc[ph["symbol"] == "000001", "is_screenable"] = False
    extra = panel.iloc[[0]].copy()
    extra["symbol"] = "999999"
    attached = _with_screenable_class(pd.concat([panel, extra], ignore_index=True), ph)

    by_symbol = attached.groupby("symbol")["is_screenable"].all()
    assert bool(by_symbol["000000"]) is True
    assert bool(by_symbol["000001"]) is False
    assert bool(by_symbol["999999"]) is False


def test_attach_eod_labels_enter_at_final_close() -> None:
    from src.ml.research.pit_certification import attach_eod_labels
    from src.strategy.contract import AA_COST

    synth = _synth_inputs(n_days=6, n_symbols=3, auction=True)
    ph, market_dates, d_to_idx, _panel, _days, _hist = synth
    all_dates = np.array(sorted(pd.to_datetime(ph["date"]).unique()))
    day = pd.Timestamp(all_dates[1])
    keys = pd.DataFrame({"date": [day] * 3, "symbol": ["000000", "000001", "000002"]})

    out = attach_eod_labels(keys, ph, market_dates, d_to_idx, cost=AA_COST)

    assert len(out) == 3
    eod_close = ph[(ph["date"] == day)].set_index("symbol")["close"]
    nxt = pd.Timestamp(all_dates[2])
    next_open = ph[(ph["date"] == nxt)].set_index("symbol")["open"]
    for sym in ("000000", "000001", "000002"):
        row = out[out["symbol"] == sym].iloc[0]
        assert row["gross_return"] == next_open[sym] / eod_close[sym] - 1.0


def test_run_pit_certification_insufficient_days_status() -> None:
    synth = _synth_inputs(n_days=120, n_symbols=40)
    ph, market_dates, d_to_idx, panel, panel_days, serving_history = synth
    keep = pd.to_datetime(panel_days["date"]) < pd.Timestamp("2023-02-15")
    panel = panel[pd.to_datetime(panel["date"]).isin(
        pd.to_datetime(panel_days.loc[keep, "date"]))]
    panel_days = panel_days[keep].reset_index(drop=True)

    report, _daily = _run((ph, market_dates, d_to_idx, panel, panel_days, serving_history))

    assert report.status.value == "INSUFFICIENT_DAYS"
    assert np.isnan(report.haircut.ci_low) and np.isnan(report.haircut.ci_high)


def test_select_usable_panel_days_coverage_rules() -> None:
    from src.ml.research.pit_certification import PitCertificationConfig, select_usable_panel_days

    days = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-02"]),
        "source": ["bars", "live_decision"],
        "superset_coverage": [0.5, 0.5],
        "index_basis": ["eod_fallback", "live_1520"],
    })
    usable = select_usable_panel_days(
        days,
        cert_dates=[pd.Timestamp("2023-02-01"), pd.Timestamp("2023-02-02")],
        config=PitCertificationConfig(),
    )

    assert usable == [pd.Timestamp("2023-02-02")]


def test_build_pit_feature_frames_native_arm_screens_on_1520_values() -> None:
    from src.data.pit1520_panel import PIT1520_PANEL_COLUMNS
    from src.ml.research.pit_certification import build_pit_feature_frames
    from src.ml.topk_history_features import HISTORY_REQUIRED_COLUMNS
    from src.strategy.contract import KCA_TOPK_COSTAWARE_001

    day = pd.Timestamp("2023-02-01")
    prev_days = pd.bdate_range(end=day - pd.Timedelta(days=1), periods=30)
    hist_rows = [
        {"date": d, "symbol": sym, "open": 100.0, "close": 105.0, "prev_close": 100.0, "volume": 1e6,
         "inst_netbuy": 0.0, "foreign_netbuy": 0.0}
        for sym in ("000001", "000002", "000003")
        for d in prev_days
    ]
    serving_history = pd.DataFrame(hist_rows)

    def _panel_row(sym, chg):
        prev = 10000.0
        close = prev * (1.0 + chg)
        return {"date": day, "symbol": sym, "market": "KOSPI", "open": prev,
                "high": close * 1.01, "low": prev * 0.99, "close": close, "close_raw": close,
                "prev_close": prev, "volume": 1e6, "trade_value_100m": 500.0,
                "market_cap_100m": 3000.0, "inst_netbuy": 0.0, "foreign_netbuy": 0.0,
                "inst_netbuy_prev": 0.0, "foreign_netbuy_prev": 0.0,
                "kospi_pct": 0.001, "kosdaq_pct": 0.002, "v_kospi": 18.0, "v_kosdaq": 22.0,
                "index_basis": "live_1520", "source": "live_decision", "bars_vendor": "",
                "n_bars": 0, "first_bar_hms": "", "last_bar_hms": "", "capture_run_id": "t"}

    panel = pd.DataFrame([_panel_row("000001", 0.015), _panel_row("000002", 0.05), _panel_row("000003", 0.06)])
    panel = panel[list(PIT1520_PANEL_COLUMNS)]
    eod_pool = pd.DataFrame({"date": [day] * 3, "symbol": ["000001", "000002", "000003"]})
    sel_mask = np.array([True, True, True])

    feat, nat = build_pit_feature_frames(
        panel=panel, usable_days=[day], eod_pool=eod_pool, sel_mask=sel_mask,
        serving_history=serving_history[list(HISTORY_REQUIRED_COLUMNS)], spec=KCA_TOPK_COSTAWARE_001,
    )

    assert ((feat["symbol"] == "000001") & feat["selectable"]).any()
    assert not ((nat["symbol"] == "000001") & nat["selectable"]).any()


def test_perturb_auction_noise_contract() -> None:
    from src.ml.research.pit_certification import perturb_auction_noise
    from src.ml.topk_contract import RANKER_FEATURE_COLS
    from src.strategy.contract import derive_chg_ratio, tick_cost_bp

    n = 6
    base = {
        "date": pd.bdate_range("2023-02-01", periods=2).repeat(n // 2),
        "symbol": [f"{i:06d}" for i in range(n)],
        "market": ["KOSPI"] * n,
        "open": np.full(n, 10000.0),
        "high": np.full(n, 10600.0),
        "low": np.full(n, 9900.0),
        "close": np.full(n, 10500.0),
        "close_raw": np.full(n, 10500.0),
        "prev_close": np.full(n, 10000.0),
        "tv_clean": np.full(n, 500.0),
        "mc_clean": np.full(n, 3000.0),
        "train_label": np.linspace(-0.01, 0.01, n),
        "net_pit": np.linspace(-0.01, 0.01, n),
        "f_dist_high60": np.full(n, 0.02),
        "tick_cost_bp": np.full(n, 5.0),
    }
    for col in RANKER_FEATURE_COLS:
        if col not in base:
            base[col] = np.full(n, 0.5)
    base["log_mc"] = np.log1p(3000.0)
    train = pd.DataFrame(base)
    draws = pd.DataFrame({"auction_move": np.full(n, 0.01), "auction_tv_share": np.full(n, 0.1)})

    out = perturb_auction_noise(train, draws, perturb_trade_value=True)

    assert out.index.equals(train.index) and len(out) == len(train)
    expected_close = 10500.0 / 1.01
    assert np.allclose(out["close"].to_numpy(), expected_close)
    assert np.allclose(out["chg_ratio"].to_numpy(), derive_chg_ratio(out["close"].to_numpy(), train["prev_close"].to_numpy()))
    assert np.allclose(out["tv_clean"].to_numpy(), 450.0)
    assert np.allclose(out["train_label"].to_numpy(), train["train_label"].to_numpy())
    assert np.allclose(out["net_pit"].to_numpy(), train["net_pit"].to_numpy())
    assert np.allclose(out["f_dist_high60"].to_numpy(), 0.02)
    assert np.allclose(out["log_mc"].to_numpy(), train["log_mc"].to_numpy())
    level = out["close_raw"].to_numpy(dtype=np.float64)
    assert np.allclose(out["f_tick_cost"].to_numpy(), tick_cost_bp(
        level, pd.to_datetime(out["date"]).to_numpy(), out["market"].astype(str).to_numpy(dtype=object)))
    assert np.allclose(out["f_log_close"].to_numpy(), np.log(level))


def test_augmentation_source_is_fold_causal() -> None:
    import src.ml.research.pit_certification as pit_mod
    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.topk_ranker_research import (
        CERT_REGIME_START,
        HISTORY_SEAM_EMBARGO_DAYS,
        attach_pit_net_label,
        build_dual_pool,
        demean_label_by_date,
        split_regime_frames,
    )
    from src.strategy.contract import DEFAULT_UNIVERSE, KCA_TOPK_COSTAWARE_001

    synth = _synth_inputs(n_days=60, n_symbols=40, auction=True)
    ph, market_dates, d_to_idx, panel, panel_days, _hist = synth
    pool, _mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=DEFAULT_UNIVERSE,
        select_spec=KCA_TOPK_COSTAWARE_001.universe)
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=KCA_TOPK_COSTAWARE_001.cost))
    cert_df, _h = split_regime_frames(labeled, train_start=CERT_REGIME_START)
    cv = CombinatorialPurgedCV(n_groups=4, k_test=2, purge_gap=0, embargo_gap=0)
    cert_work, test_dates = pit_mod._fold_test_dates(cert_df, group_col="date", cv=cv)
    pool_dates = sorted({pd.Timestamp(d).normalize() for d in pd.to_datetime(cert_work["date"]).tolist()})
    usable = sorted({pd.Timestamp(d).normalize() for d in pd.to_datetime(panel["date"]).tolist()})
    source = pit_mod.measure_auction_moves(panel, ph, usable_days=usable)
    assert len(source) > 0

    pos = {d: i for i, d in enumerate(pool_dates)}
    for fold_test in test_dates:
        src = pit_mod._causal_source_days(
            source, pool_dates, fold_test, embargo_days=HISTORY_SEAM_EMBARGO_DAYS)
        test_pos = sorted(pos[d] for d in fold_test if d in pos)
        for day in pd.to_datetime(src["date"]).dt.normalize().unique().tolist():
            i = pos[pd.Timestamp(day).normalize()]
            assert all(abs(i - t) > HISTORY_SEAM_EMBARGO_DAYS for t in test_pos)


@pytest.mark.slow
def test_augmentation_insufficient_source_is_not_evaluated() -> None:
    from src.ml.research.pit_certification import AuctionNoiseConfig, PitCertificationConfig

    plain_report, plain_daily = _run(_synth_inputs(auction=True))
    aug_config = _cert_config(augmentation=AuctionNoiseConfig(min_source_days=10**6))
    aug_report, _aug_daily = _run(_synth_inputs(auction=True), config=aug_config)

    assert aug_report.augmentation is not None
    assert aug_report.augmentation.status == "INSUFFICIENT_SOURCE"
    assert aug_report.augmentation.verdict == "NOT_EVALUATED"
    assert aug_report.mean_net_bp == plain_report.mean_net_bp
    assert aug_report.haircut.delta == plain_report.haircut.delta


def test_augmentation_adoption_requires_bonferroni() -> None:
    from src.ml.pit_report import PairedDelta
    from src.ml.research.pit_certification import _augmentation_verdict

    improvement = PairedDelta(delta=1.0, ci_low=0.5, ci_high=1.5, p_value=0.04, n_days=60)
    assert _augmentation_verdict(improvement, declared_trials=3, alpha=0.05) == "REJECT"
    assert _augmentation_verdict(improvement, declared_trials=1, alpha=0.05) == "ADOPT_CANDIDATE"


@pytest.mark.slow
def test_run_pit_certification_is_deterministic() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    synth = _synth_inputs()
    now = datetime(2026, 10, 4, tzinfo=ZoneInfo("Asia/Seoul"))
    report_a, daily_a = _run(synth, now=now)
    report_b, daily_b = _run(synth, now=now)

    assert report_a.mean_net_bp == report_b.mean_net_bp
    assert report_a.haircut.delta == report_b.haircut.delta
    assert report_a.rank_ic_mean == report_b.rank_ic_mean
    pd.testing.assert_frame_equal(daily_a, daily_b)
