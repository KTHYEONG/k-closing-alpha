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


def _recon_gate_config(**overrides):
    from src.ml.research.pit_certification import ReconstructionGateConfig

    base = {"bootstrap_n_boot": 200, "bootstrap_seed": 7}
    base.update(overrides)
    return ReconstructionGateConfig(**base)


def _recon_daily(days, *, full, matched):
    rows = []
    for day, net_full, net_matched in zip(days, full, matched, strict=True):
        rows.append({"date": day, "arm": "eod_full", "topk_net_bp": net_full, "rank_ic": 0.02,
                     "n_pool": 10, "overlap_vs_eod": 1.0, "index_basis": "live_1520"})
        rows.append({"date": day, "arm": "eod_matched", "topk_net_bp": net_matched, "rank_ic": 0.02,
                     "n_pool": 9, "overlap_vs_eod": 0.9, "index_basis": "live_1520"})
    return pd.DataFrame(rows)


def _adoptable_dailies(n_days=35, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-05", periods=n_days + 2).strftime("%Y-%m-%d").tolist()
    exact_days, recon_days = days[:n_days], days[1:n_days + 1]
    exact_full = 20.0 + rng.normal(0, 1.0, n_days)
    exact_matched = 15.0 + rng.normal(0, 1.0, n_days)
    recon_full = 20.0 + rng.normal(0, 1.0, n_days)
    recon_matched = 19.0 + rng.normal(0, 1.0, n_days)
    return (
        _recon_daily(exact_days, full=exact_full, matched=exact_matched),
        _recon_daily(recon_days, full=recon_full, matched=recon_matched),
    )


def _adoptable_loop(n_days=35, seed=5, top_k=2):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-03-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    symbols = [f"{i:06d}" for i in range(4)]
    exact_rows, recon_rows, label_rows = [], [], []
    for day in days:
        base = 10000.0 + rng.normal(0, 50.0)
        for i, symbol in enumerate(symbols):
            tv = 500.0 + i * 100.0 + rng.normal(0, 1.0)
            chg = 0.01 * (i + 1) + rng.normal(0, 0.0005)
            exact_rows.append({"date": day, "symbol": symbol, "trade_value": tv, "chg": chg})
            recon_rows.append({"date": day, "symbol": symbol, "trade_value": tv * 0.97,
                               "chg": chg + rng.normal(0, 0.0002)})
            label_rows.append({"date": day, "symbol": symbol,
                               "net_return": 0.001 * (i + 1) + rng.normal(0, 0.0002)})
    return pd.DataFrame(exact_rows), pd.DataFrame(recon_rows), pd.DataFrame(label_rows)


def test_pair_certification_days_uses_intersection() -> None:
    from src.ml.research.pit_certification import pair_certification_days

    exact, recon = _adoptable_dailies(n_days=6)
    paired, dropped = pair_certification_days(exact, recon)
    assert len(paired) == 5 and len(dropped) == 2
    assert dropped == sorted(dropped)
    with pytest.raises(ValueError, match="date"):
        pair_certification_days(exact.drop(columns=["date"]), recon)
    with pytest.raises(ValueError, match="date"):
        pair_certification_days(exact, recon.drop(columns=["date"]))


def test_paired_coverage_improvement_sign_and_scope() -> None:
    from src.ml.research.pit_certification import paired_coverage_improvement

    exact, recon = _adoptable_dailies()
    improvement = paired_coverage_improvement(exact, recon, config=_recon_gate_config())
    assert improvement.delta == pytest.approx(4.0, abs=0.6)
    assert improvement.ci_low > 0.0
    assert improvement.n_days == 34
    thin_exact, thin_recon = _adoptable_dailies(n_days=6)
    thin = paired_coverage_improvement(thin_exact, thin_recon, config=_recon_gate_config())
    assert np.isnan(thin.ci_low) and thin.n_days == 5
    armless = thin_exact[thin_exact["arm"] == "eod_full"].reset_index(drop=True)
    empty = paired_coverage_improvement(armless, thin_recon, config=_recon_gate_config())
    assert empty.n_days == 0 and np.isnan(empty.delta)
    with pytest.raises(ValueError, match="topk_net_bp"):
        paired_coverage_improvement(exact.drop(columns=["topk_net_bp"]), recon, config=_recon_gate_config())


def test_coverage_by_year_and_basis_reports_means() -> None:
    from src.ml.research.pit_certification import coverage_by_year_and_basis

    frame = pd.DataFrame([
        {"date": "2025-06-02", "index_basis": "live_1520", "superset_coverage": 0.9},
        {"date": "2025-06-03", "index_basis": "live_1520", "superset_coverage": 0.7},
        {"date": "2026-09-18", "index_basis": "eod_fallback", "superset_coverage": 1.0},
        {"date": "2026-09-21", "index_basis": "eod_fallback", "superset_coverage": float("nan")},
    ])
    got = coverage_by_year_and_basis(frame)
    assert got["2025"]["live_1520"] == pytest.approx(0.8)
    assert got["2026"]["eod_fallback"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="superset_coverage"):
        coverage_by_year_and_basis(frame.drop(columns=["superset_coverage"]))


def test_calibration_to_loop_frames_derive_both_arms() -> None:
    from src.data.nxt_decomposition import DecompositionConfig
    from src.ml.research.pit_certification import calibration_to_loop_frames

    config = DecompositionConfig(
        ewma_alpha=0.5, min_prior_days=1, max_gap_days=30, auction_fraction_mean=0.0365,
        bias_correction=1.0, volume_rel_err_p90=0.144, close_bp_err_p90=11.9,
        calibrated_through="2026-09-30",
    )
    days = pd.bdate_range("2026-03-02", periods=5).strftime("%Y-%m-%d").tolist()
    rows = [
        {
            "date": day, "symbol": symbol, "v_krx_1520": 6800.0, "v_cons_1520": 10000.0,
            "eod_volume": 7165.0, "auction_volume": 365.0,
            "close_krx_1519": 50000.0, "close_cons_1519": 50050.0,
        }
        for day in days
        for symbol in ("000001", "000002")
    ]
    calibration = pd.DataFrame(rows)
    prev_closes = {(day, symbol): 49000.0 for day in days for symbol in ("000001", "000002")}
    frames = calibration_to_loop_frames(calibration, config=config, prev_closes=prev_closes)
    assert len(frames.exact) == 10
    assert len(frames.recon) == 8
    assert frames.n_no_share == 2
    assert set(frames.recon["date"]) == set(days[1:])
    with pytest.raises(ValueError, match="eod_volume"):
        calibration_to_loop_frames(calibration.drop(columns=["eod_volume"]), config=config, prev_closes=prev_closes)
    bad_config = DecompositionConfig(
        ewma_alpha=0.5, min_prior_days=1, max_gap_days=30, auction_fraction_mean=float("nan"),
        bias_correction=1.0, volume_rel_err_p90=0.144, close_bp_err_p90=11.9,
        calibrated_through="2026-09-30",
    )
    with pytest.raises(ValueError, match="auction fraction"):
        calibration_to_loop_frames(calibration, config=bad_config, prev_closes=prev_closes)
    broken_dates = calibration.copy()
    broken_dates.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="unparseable"):
        calibration_to_loop_frames(broken_dates, config=config, prev_closes=prev_closes)
    odd_prev = dict(prev_closes)
    odd_prev[(days[0], "000001")] = None
    odd = calibration_to_loop_frames(calibration, config=config, prev_closes=odd_prev)
    assert len(odd.exact) == 9


def test_reconstruction_in_the_loop_uses_hidden_truth_only() -> None:
    from src.ml.research.pit_certification import reconstruction_in_the_loop

    exact, recon, labels = _adoptable_loop(n_days=8)
    result = reconstruction_in_the_loop(exact=exact, recon=recon, labels=labels, top_k=2,
                                        config=_recon_gate_config())
    assert result.n_symbol_days == 32
    assert result.n_paired_days == 8
    assert result.chg_rank_correlation > 0.99
    assert result.tv_rank_correlation > 0.99
    assert result.top_k_overlap == pytest.approx(1.0)
    assert abs(result.feature_component.delta) < 1.0

    day0 = exact["date"].iloc[0]
    trimmed = labels[labels["date"] != day0].reset_index(drop=True)
    result = reconstruction_in_the_loop(exact=exact, recon=recon, labels=trimmed, top_k=2,
                                        config=_recon_gate_config())
    assert result.n_paired_days == 7

    with pytest.raises(ValueError, match="trade_value"):
        reconstruction_in_the_loop(exact=exact.drop(columns=["trade_value"]), recon=recon,
                                   labels=labels, top_k=2, config=_recon_gate_config())
    with pytest.raises(ValueError, match="net_return"):
        reconstruction_in_the_loop(exact=exact, recon=recon,
                                   labels=labels.drop(columns=["net_return"]), top_k=2,
                                   config=_recon_gate_config())
    with pytest.raises(ValueError, match="top_k"):
        reconstruction_in_the_loop(exact=exact, recon=recon, labels=labels, top_k=0,
                                   config=_recon_gate_config())
    dup = pd.concat([exact, exact.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        reconstruction_in_the_loop(exact=dup, recon=recon, labels=labels, top_k=2,
                                   config=_recon_gate_config())


def test_reconstruction_in_the_loop_skips_degenerate_days() -> None:
    from src.ml.research.pit_certification import reconstruction_in_the_loop

    exact, recon, labels = _adoptable_loop(n_days=4)
    ordered_days = recon["date"].drop_duplicates().tolist()
    exact = exact[exact["date"] != ordered_days[0]].reset_index(drop=True)
    thin = (exact["date"] != ordered_days[1]) | (exact["symbol"] == "000000")
    exact = exact[thin].reset_index(drop=True)
    nan_frame = recon.copy()
    nan_frame.loc[nan_frame["date"] == ordered_days[2], "trade_value"] = float("nan")
    result = reconstruction_in_the_loop(exact=exact, recon=nan_frame, labels=labels, top_k=2,
                                        config=_recon_gate_config())
    assert result.n_symbol_days == 9
    assert result.n_paired_days == 1
    assert np.isnan(result.feature_component.ci_low)


def test_check_calibration_stability_first_half_fit_second_half_score() -> None:
    from src.ml.research.pit_certification import check_calibration_stability

    def _calibration(shares_early, shares_late):
        early = pd.bdate_range("2026-01-05", periods=len(shares_early)).strftime("%Y-%m-%d").tolist()
        late = pd.bdate_range("2026-03-02", periods=len(shares_late)).strftime("%Y-%m-%d").tolist()
        rows = []
        for i, day in enumerate(early):
            for symbol in ("000001", "000002"):
                share = shares_early[i]
                rows.append(_calibration_row(day, symbol, share))
        for i, day in enumerate(late):
            for symbol in ("000001", "000002"):
                share = shares_late[i]
                rows.append(_calibration_row(day, symbol, share))
        return pd.DataFrame(rows)

    fit_kwargs = {
        "alphas": [0.3, 0.5],
        "min_prior_days": 1,
        "max_gap_days": 90,
        "identity_tolerance": 0.05,
        "max_median_rel_err": 0.5,
    }
    stable = check_calibration_stability(_calibration([0.68] * 4, [0.68] * 4), **fit_kwargs)
    assert stable.passed is True
    assert stable.median_rel_err == pytest.approx(0.0)
    assert stable.n_scored == 8
    dirty = _calibration([0.68] * 4, [0.68] * 4)
    dirty.loc[len(dirty)] = {**_calibration_row("2026-01-06", "000001", 0.68), "eod_volume": "bogus"}
    dirty.loc[len(dirty)] = {**_calibration_row("2026-03-03", "000001", 0.68), "v_krx_1520": "bogus"}
    dirty_stable = check_calibration_stability(dirty, **fit_kwargs)
    assert dirty_stable.passed is True
    assert dirty_stable.n_scored == 8
    guarded = _calibration([0.68] * 4, [0.68] * 4)
    guarded.loc[len(guarded)] = {**_calibration_row("2026-01-06", "000001", 0.68), "eod_volume": 0.0}
    guarded.loc[len(guarded)] = {**_calibration_row("2026-03-03", "000001", 0.68), "eod_volume": 999999.0}
    guarded_stable = check_calibration_stability(guarded, **fit_kwargs)
    assert guarded_stable.passed is True
    assert guarded_stable.n_scored == 8
    stale = check_calibration_stability(
        _calibration([0.68] * 4, [0.68] * 4), **{**fit_kwargs, "max_gap_days": 1}
    )
    assert stale.passed is False and "no second-half row predictable" in stale.detail
    broken = check_calibration_stability(
        _calibration([0.5, 0.9, 0.5, 0.9], [0.30] * 4), **{**fit_kwargs, "max_median_rel_err": 0.05}
    )
    assert broken.passed is False
    assert "above ceiling" in broken.detail
    empty = check_calibration_stability(pd.DataFrame(), **fit_kwargs)
    assert empty.passed is False and empty.n_scored == 0
    single = check_calibration_stability(_calibration([0.68], []), **fit_kwargs)
    assert single.passed is False and "two calibration dates" in single.detail
    thin_first = check_calibration_stability(
        pd.DataFrame([_calibration_row("2026-01-05", "000001", 0.68),
                      _calibration_row("2026-03-02", "000001", 0.68)]),
        **fit_kwargs,
    )
    assert thin_first.passed is False and "first-half fit refused" in thin_first.detail
    early_days = pd.bdate_range("2026-01-05", periods=2).strftime("%Y-%m-%d").tolist()
    late_days = pd.bdate_range("2026-03-02", periods=2).strftime("%Y-%m-%d").tolist()
    disjoint_rows = [_calibration_row(day, symbol, 0.68) for day in early_days for symbol in ("000001", "000002")]
    disjoint_rows += [_calibration_row(day, symbol, 0.68) for day in late_days for symbol in ("000003", "000004")]
    unscored = check_calibration_stability(pd.DataFrame(disjoint_rows), **fit_kwargs)
    assert unscored.passed is False and "no second-half row predictable" in unscored.detail


def _calibration_row(day, symbol, share, v_cons=10000.0):
    v_krx = share * v_cons
    return {
        "date": day, "symbol": symbol, "v_krx_1520": v_krx, "auction_volume": 365.0,
        "v_cons_1520": v_cons, "eod_volume": v_krx + 365.0,
        "close_krx_1519": 50000.0, "close_cons_1519": 50000.0,
    }


def _gate_inputs(**overrides):
    from src.ml.pit_report import PairedDelta

    values = {
        "coverage_improvement": PairedDelta(delta=4.0, ci_low=2.0, ci_high=6.0, p_value=0.001, n_days=60),
        "reconstruction_feature": PairedDelta(delta=0.2, ci_low=-1.0, ci_high=1.4, p_value=0.6, n_days=60),
        "stability": None,
    }
    values.update(overrides)
    if values["stability"] is None:
        from src.ml.pit_report import CalibrationStability

        values["stability"] = CalibrationStability(passed=True, median_rel_err=0.03, n_scored=40, detail="")
    return values


def test_reconstruction_gate_adopts_only_on_all_criteria() -> None:
    from src.ml.pit_report import CalibrationStability, PairedDelta
    from src.ml.research.pit_certification import evaluate_reconstruction_gate

    config = _recon_gate_config()
    verdict = evaluate_reconstruction_gate(**_gate_inputs(), config=config)
    assert verdict.verdict == "ADOPT"
    assert verdict.reasons == ()
    assert verdict.stability_passed is True

    cases = [
        ({"coverage_improvement": PairedDelta(delta=0.5, ci_low=-1.0, ci_high=2.0, p_value=0.4, n_days=60)}, "includes zero", {}),
        ({"coverage_improvement": PairedDelta(delta=float("nan"), ci_low=float("nan"), ci_high=float("nan"), p_value=float("nan"), n_days=60)}, "not estimable", {}),
        ({"coverage_improvement": PairedDelta(delta=0.5, ci_low=0.2, ci_high=0.8, p_value=0.01, n_days=60)}, "at or below the minimum", {"min_coverage_improvement_bp": 1.0}),
        ({"coverage_improvement": PairedDelta(delta=4.0, ci_low=2.0, ci_high=6.0, p_value=0.001, n_days=10)}, "only 10 paired days", {}),
        ({"reconstruction_feature": PairedDelta(delta=-3.0, ci_low=-5.0, ci_high=-1.0, p_value=0.01, n_days=60)}, "significant", {}),
        ({"reconstruction_feature": PairedDelta(delta=0.2, ci_low=-1.0, ci_high=1.4, p_value=float("nan"), n_days=60)}, "not established", {}),
        ({"reconstruction_feature": PairedDelta(delta=0.2, ci_low=-1.0, ci_high=1.4, p_value=0.6, n_days=5)}, "only 5 paired days", {}),
        ({"stability": CalibrationStability(passed=False, median_rel_err=0.4, n_scored=10, detail="ceiling")}, "stability failed", {}),
    ]
    for overrides, needle, cfg_overrides in cases:
        verdict = evaluate_reconstruction_gate(
            **_gate_inputs(**overrides), config=_recon_gate_config(**cfg_overrides)
        )
        assert verdict.verdict == "REJECT"
        assert any(needle in reason for reason in verdict.reasons), (needle, verdict.reasons)


def test_run_reconstruction_certification_combines_three_arms() -> None:
    from src.ml.pit_report import CalibrationStability
    from src.ml.research.pit_certification import run_reconstruction_certification

    exact_daily, recon_daily = _adoptable_dailies()
    loop_exact, loop_recon, loop_labels = _adoptable_loop()
    stability = CalibrationStability(passed=True, median_rel_err=0.03, n_scored=40, detail="")
    coverage = {"exact": {"2026": {"live_1520": 0.95}}, "with_reconstruction": {"2026": {"live_1520": 0.99}}}
    cert = run_reconstruction_certification(
        daily_exact=exact_daily, daily_recon=recon_daily, loop_exact=loop_exact,
        loop_recon=loop_recon, labels=loop_labels, stability=stability, coverage=coverage,
        top_k=2, config=_recon_gate_config(), exact_dir="exact", recon_dir="recon",
        generated_at="2026-10-05T00:00:00+09:00",
    )
    assert cert.gate_verdict == "ADOPT"
    assert cert.gate_reasons == ()
    assert len(cert.paired_days) == 34
    assert len(cert.dropped_days) == 2
    assert cert.coverage_by_year_and_basis == coverage
    again = run_reconstruction_certification(
        daily_exact=exact_daily, daily_recon=recon_daily, loop_exact=loop_exact,
        loop_recon=loop_recon, labels=loop_labels, stability=stability, coverage=coverage,
        top_k=2, config=_recon_gate_config(),
    )
    assert again.gate_verdict == "ADOPT"
    assert again.generated_at != ""


def test_main_reconstruction_certification_reads_report_dirs(tmp_path, monkeypatch) -> None:
    import json

    from src.data.nxt_decomposition import DecompositionConfig, save_decomposition_config
    from src.ml.pit_report import save_pit_haircut_report
    from src.ml.research.pit_certification import (
        ReconstructionGateConfig,
        _prev_close_map,
        main_reconstruction_certification,
    )

    exact_daily, recon_daily = _adoptable_dailies(n_days=8)
    for name, daily in (("exact", exact_daily), ("recon", recon_daily)):
        arm_dir = tmp_path / name
        arm_dir.mkdir()
        report = _cert_report_stub()
        save_pit_haircut_report(report, daily, out_dir=arm_dir)
    days = pd.bdate_range("2026-03-02", periods=5).strftime("%Y-%m-%d").tolist()
    cal_rows = [
        {**_calibration_row(day, symbol, 0.68), "close_cons_1519": 50050.0}
        for day in days
        for symbol in ("000001", "000002")
    ]
    calibration_path = tmp_path / "calibration.parquet"
    pd.DataFrame(cal_rows).to_parquet(calibration_path, index=False)
    decomp = DecompositionConfig(
        ewma_alpha=0.5, min_prior_days=1, max_gap_days=30, auction_fraction_mean=0.0365,
        bias_correction=1.0, volume_rel_err_p90=0.144, close_bp_err_p90=11.9,
        calibrated_through="2026-03-31",
    )
    config_path = tmp_path / "nxt_decomposition_config.json"
    save_decomposition_config(decomp, config_path)

    raw_rows = []
    for day in pd.bdate_range("2026-03-02", periods=6):
        for i, symbol in enumerate(("000001", "000002")):
            base = 50000.0 + i * 100.0
            raw_rows.append({
                "date": day, "symbol": symbol, "open": base, "high": base * 1.01,
                "low": base * 0.99, "close": base, "close_raw": base, "prev_close": base / 1.02,
                "volume": 1e6, "market_cap_100m": 3000.0, "trade_value_100m": 500.0,
                "market": "KOSPI", "daily_change_pct": 0.02, "inst_netbuy": 0.0,
                "foreign_netbuy": 0.0, "program_netbuy": 0.0, "kospi_pct": 0.001,
                "kosdaq_pct": 0.002, "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
    from src.data.panel_integrity import prepare_price_panel

    ph, _prov = prepare_price_panel(pd.DataFrame(raw_rows))
    market_dates = np.array(sorted(ph["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}

    def _fake_loader(path):
        assert str(path).endswith(".parquet")
        return ph, market_dates, d_to_idx

    monkeypatch.setattr("src.ml.research.v3_engine.load_and_prepare_price_history", _fake_loader)
    days_exact = tmp_path / "exact_days.parquet"
    pd.DataFrame([
        {"date": "2026-03-02", "index_basis": "live_1520", "superset_coverage": 0.9},
    ]).to_parquet(days_exact, index=False)
    days_recon = tmp_path / "recon_days.parquet"
    pd.DataFrame([
        {"date": "2026-03-02", "index_basis": "live_1520", "superset_coverage": 0.95},
    ]).to_parquet(days_recon, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cert = main_reconstruction_certification(
        exact_dir=tmp_path / "exact", recon_dir=tmp_path / "recon", out_dir=out_dir,
        calibration_path=calibration_path, config_path=config_path,
        price_history_path=tmp_path / "price_history.parquet",
        top_k=2, exact_days_path=days_exact, recon_days_path=days_recon,
        gate_config=ReconstructionGateConfig(bootstrap_n_boot=50),
    )
    assert cert.paired_days != []
    assert cert.gate_verdict == "REJECT"
    payload = json.loads((out_dir / "reconstruction_certification.json").read_text(encoding="utf-8"))
    assert payload["gate_verdict"] == cert.gate_verdict
    assert payload["coverage_by_year_and_basis"]["exact"]["2026"]["live_1520"] == pytest.approx(0.9)
    assert payload["coverage_by_year_and_basis"]["with_reconstruction"]["2026"]["live_1520"] == pytest.approx(0.95)
    with pytest.raises(FileNotFoundError, match="daily evidence"):
        main_reconstruction_certification(
            exact_dir=tmp_path / "exact", recon_dir=tmp_path / "missing", out_dir=out_dir,
            calibration_path=calibration_path, config_path=config_path,
            price_history_path=tmp_path / "price_history.parquet", top_k=2,
        )
    with pytest.raises(ValueError, match="close"):
        _prev_close_map(pd.DataFrame([{"date": "2026-03-02", "symbol": "000001"}]))


def _cert_report_stub():
    import dataclasses

    from src.ml.pit_report import PairedDelta, PitHaircutReport, PitReportStatus
    from src.strategy.contract import PRODUCTION_STRATEGY

    return PitHaircutReport(
        generated_at="2026-10-05T00:00:00+09:00",
        strategy_id="KCA-TOPK-COSTAWARE-002",
        strategy_fingerprint="fp",
        top_k=2,
        select_universe=dataclasses.asdict(PRODUCTION_STRATEGY.universe),
        feature_contract_version="1",
        model_params={"n_estimators": 10},
        seeds=(1,),
        status=PitReportStatus.OK,
        panel_date_min="2026-03-02",
        panel_date_max="2026-03-13",
        n_usable_days=8,
        n_paired_days=8,
        n_live_days=8,
        n_eod_index_days=0,
        mean_net_bp={"eod_full": 8.0, "eod_matched": 8.0, "pit_feature": 6.0, "pit_native": 6.0},
        haircut=PairedDelta(delta=2.0, ci_low=0.0, ci_high=4.0, p_value=0.1, n_days=8),
        coverage_component=PairedDelta(delta=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0, n_days=8),
        feature_component=PairedDelta(delta=1.0, ci_low=0.0, ci_high=2.0, p_value=0.2, n_days=8),
        selection_component=PairedDelta(delta=1.0, ci_low=0.0, ci_high=2.0, p_value=0.2, n_days=8),
        pit_native_vs_zero=PairedDelta(delta=6.0, ci_low=0.0, ci_high=9.0, p_value=0.01, n_days=8),
        rank_ic_mean={"eod_full": 0.03, "eod_matched": 0.03, "pit_feature": 0.02, "pit_native": 0.02},
        rank_ic_haircut=PairedDelta(delta=0.01, ci_low=0.0, ci_high=0.02, p_value=0.2, n_days=8),
        pick_overlap_mean={"pit_feature": 0.9, "pit_native": 0.8},
        haircut_by_index_basis={"live_1520": 2.0, "eod_fallback": float("nan")},
        augmentation=None,
    )
