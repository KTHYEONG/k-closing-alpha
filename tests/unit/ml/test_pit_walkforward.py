import numpy as np
import pandas as pd
import pytest

from src.ml.research import pit_walkforward as wf


def _calendar(n: int) -> np.ndarray:
    return np.array(pd.bdate_range("2024-01-01", periods=n))


def _dataset(days: np.ndarray, n_names: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, day in enumerate(days):
        exit_day = days[i + 1] if i + 1 < len(days) else pd.NaT
        rows.extend(
            {
                "date": pd.Timestamp(day),
                "symbol": f"{j:06d}",
                "tier": wf.TIER_EXACT,
                "pre_panel": False,
                "f_tick_cost": 5.0 + j,
                "x": float(j),
                "net_pit": 0.001 * j + rng.normal(0, 1e-4),
                "train_label": 0.001 * (j - (n_names - 1) / 2),
                "exit_date": exit_day,
            }
            for j in range(n_names)
        )
    return pd.DataFrame(rows)


class _LinearFit:
    def __init__(self, train: pd.DataFrame) -> None:
        self.n_rows = len(train)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features.iloc[:, 0].to_numpy(dtype=np.float64)


def test_training_rows_exclude_labels_resolving_at_or_after_embargo_boundary() -> None:
    days = _calendar(10)
    ds = _dataset(days)
    cfg = wf.ModelConfig(train_source=wf.TRAIN_PIT_ONLY, embargo_days=1, min_train_days=1)
    train = wf.training_rows(ds, calendar=days, block_start=pd.Timestamp(days[5]), config=cfg)
    assert pd.to_datetime(train["exit_date"]).max() < pd.Timestamp(days[4])
    assert pd.to_datetime(train["date"]).max() == pd.Timestamp(days[2])


def test_training_rows_respect_train_source_tiers() -> None:
    days = _calendar(6)
    ds = _dataset(days)
    ds.loc[ds["symbol"] == "000000", "tier"] = wf.TIER_EOD
    start = pd.Timestamp(days[5])
    pit = wf.training_rows(
        ds, calendar=days, block_start=start, config=wf.ModelConfig(train_source=wf.TRAIN_PIT_ONLY, min_train_days=1)
    )
    eod = wf.training_rows(
        ds, calendar=days, block_start=start, config=wf.ModelConfig(train_source=wf.TRAIN_EOD_ALL, min_train_days=1)
    )
    assert (pit["tier"] != wf.TIER_EOD).all()
    assert (eod["tier"] == wf.TIER_EOD).all()


def test_training_rows_rejects_non_calendar_block_start() -> None:
    days = _calendar(5)
    with pytest.raises(ValueError, match="not a calendar date"):
        wf.training_rows(_dataset(days), calendar=days, block_start=pd.Timestamp("2023-12-31"), config=wf.ModelConfig())


def test_future_label_perturbation_leaves_earlier_predictions_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _calendar(30)
    ds = _dataset(days)
    monkeypatch.setattr(wf, "feature_columns", lambda _fs: ["x"])
    seen: list[pd.Timestamp] = []

    def fit(train: pd.DataFrame, feats: list[str]) -> _LinearFit:
        seen.append(pd.to_datetime(train["exit_date"]).max())
        return _LinearFit(train)

    cfg = wf.ModelConfig(train_source=wf.TRAIN_PIT_ONLY, retrain_every_days=5, min_train_days=3)
    period = wf.Period("t", str(pd.Timestamp(days[10]).date()), str(pd.Timestamp(days[29]).date()))
    base = wf.run_model(ds, period=period, config=cfg, calendar=days, fit_fn=fit)
    perturbed = ds.copy()
    late = pd.to_datetime(perturbed["date"]) >= pd.Timestamp(days[20])
    perturbed.loc[late, ["train_label", "net_pit"]] = 9.0
    again = wf.run_model(perturbed, period=period, config=cfg, calendar=days, fit_fn=fit)
    early = pd.to_datetime(base["date"]) < pd.Timestamp(days[20])
    pd.testing.assert_series_equal(
        base.loc[early, "pred"].reset_index(drop=True), again.loc[early, "pred"].reset_index(drop=True)
    )
    assert all(cut < pd.Timestamp(days[10]) for cut in seen[:1])


def test_evaluate_portfolio_equal_weight_and_abstain_rule() -> None:
    day = pd.Timestamp("2024-01-02")
    scored = pd.DataFrame(
        {
            "date": [day] * 4,
            "symbol": ["a", "b", "c", "d"],
            "pred": [0.4, 0.3, 0.2, 0.1],
            "net_pit": [0.01, 0.02, 0.03, 0.04],
            "f_tick_cost": [1.0, 2.0, 20.0, 4.0],
        }
    )
    out = wf.evaluate_portfolio(scored, wf.PortfolioConfig(top_k=3, max_tick_cost_bp=12.0))
    row = out.iloc[0]
    assert row["n_selectable"] == 3
    assert row["model_net"] == pytest.approx((0.01 + 0.02 + 0.04) / 3)
    assert row["pool_mean"] == pytest.approx((0.01 + 0.02 + 0.04) / 3)
    abstain = wf.evaluate_portfolio(scored, wf.PortfolioConfig(top_k=4, max_tick_cost_bp=12.0)).iloc[0]
    assert not abstain["active"] and abstain["model_net"] == 0.0


def test_inverse_tick_weights_sum_to_one_and_favor_cheap_ticks() -> None:
    picks = pd.DataFrame({"f_tick_cost": [2.0, 4.0, 8.0]})
    w = wf.portfolio_weights(picks, wf.WEIGHT_INV_TICK)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
    bad = wf.portfolio_weights(pd.DataFrame({"f_tick_cost": [2.0, np.nan]}), wf.WEIGHT_INV_TICK)
    assert bad == pytest.approx([0.5, 0.5])


def test_attach_lagged_index_uses_previous_trading_date() -> None:
    table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "kospi_pct": [0.01, 0.02, 0.03],
            "kosdaq_pct": [0.1, 0.2, 0.3],
            "v_kospi": [10.0, 20.0, 30.0],
        }
    )
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-02"]), "symbol": ["x", "y", "z"]}
    )
    out = wf.attach_lagged_index(frame, table).set_index("symbol")
    assert out.loc["x", "kospi_pct_lag1"] == pytest.approx(0.01)
    assert out.loc["y", "v_kospi_lag1"] == pytest.approx(20.0)
    assert np.isnan(out.loc["z", "kosdaq_pct_lag1"])


def test_feature_sets_keep_certified_order_and_drop_or_lag_index() -> None:
    base = wf.feature_columns(wf.FEATURE_SET_BASE)
    lag = wf.feature_columns(wf.FEATURE_SET_INDEX_LAG1)
    drop = wf.feature_columns(wf.FEATURE_SET_INDEX_DROP)
    assert len(lag) == len(base) and [c for c in lag if c.endswith(wf.LAG1_SUFFIX)] == [
        f"{c}_lag1" for c in wf.INDEX_FEATURE_COLS
    ]
    assert not set(wf.INDEX_FEATURE_COLS) & set(drop)
    assert drop == [c for c in base if c not in wf.INDEX_FEATURE_COLS]


def test_series_metrics_drawdown_and_newey_west_on_constant_sign() -> None:
    r = np.array([0.01, -0.02, 0.01, 0.01])
    m = wf.series_metrics(r)
    assert m["mdd"] == pytest.approx(0.02, rel=1e-9)
    assert m["mean_bp"] == pytest.approx(25.0)
    assert np.isnan(wf.newey_west_t(np.array([0.01, 0.02])))


def test_model_view_drops_panel_era_eod_rows_unless_eod_all() -> None:
    ds = pd.DataFrame({"tier": [wf.TIER_EXACT, wf.TIER_EOD, wf.TIER_EOD], "pre_panel": [False, True, False]})
    assert len(wf.model_view(ds, wf.ModelConfig(train_source=wf.TRAIN_PIT_EOD_AUG))) == 2
    assert len(wf.model_view(ds, wf.ModelConfig(train_source=wf.TRAIN_EOD_ALL))) == 3


def test_trial_registry_counts_distinct_trials_per_period(tmp_path) -> None:
    reg = wf.TrialRegistry(tmp_path / "t.jsonl")
    reg.append({"trial_id": "a", "period": "discovery"})
    reg.append({"trial_id": "a", "period": "discovery"})
    reg.append({"trial_id": "b", "period": "discovery"})
    reg.append({"trial_id": "a", "period": "holdout"})
    assert reg.n_trials("discovery") == 2
    assert reg.n_trials("holdout") == 1


def test_regime_gate_uses_only_prior_pool_returns_and_keeps_ungated_ids_stable() -> None:
    days = pd.bdate_range("2024-01-01", periods=6)
    nets = [-0.01, -0.01, 0.02, 0.02, 0.02, -0.05]
    scored = pd.DataFrame(
        {
            "date": np.repeat(days, 3),
            "symbol": ["a", "b", "c"] * 6,
            "pred": [0.3, 0.2, 0.1] * 6,
            "net_pit": np.repeat(nets, 3),
            "f_tick_cost": [5.0] * 18,
        }
    )
    gated = wf.evaluate_portfolio(scored, wf.PortfolioConfig(top_k=3, regime_lookback=2)).set_index("date")
    assert bool(gated.loc[days[0], "active"])
    assert not bool(gated.loc[days[1], "active"]) and not bool(gated.loc[days[2], "active"])
    assert bool(gated.loc[days[3], "active"])
    assert bool(gated.loc[days[4], "active"]) and bool(gated.loc[days[5], "active"])
    assert gated.loc[days[5], "model_net"] == pytest.approx(-0.05)
    plain = wf.PortfolioConfig(top_k=3)
    assert "regime_lookback" not in plain.identity()
    assert wf.trial_id(wf.ModelConfig(), plain) != wf.trial_id(
        wf.ModelConfig(), wf.PortfolioConfig(top_k=3, regime_lookback=20)
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"feature_set": "nope"}, "feature_set"),
        ({"train_source": "nope"}, "train_source"),
        ({"retrain_every_days": 0}, "retrain_every_days"),
    ],
)
def test_model_config_rejects_invalid_values(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        wf.ModelConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"top_k": 0}, "top_k"),
        ({"weighting": "nope"}, "weighting"),
        ({"max_tick_cost_bp": 0.0}, "max_tick_cost_bp"),
        ({"regime_lookback": 0}, "regime_lookback"),
    ],
)
def test_portfolio_config_rejects_invalid_values(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        wf.PortfolioConfig(**kwargs)


def test_feature_columns_rejects_unknown_set() -> None:
    with pytest.raises(ValueError, match="unknown feature_set"):
        wf.feature_columns("nope")


def test_attach_lagged_index_rejects_missing_or_duplicate_dates() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"])})
    with pytest.raises(ValueError, match="missing columns"):
        wf.attach_lagged_index(frame, pd.DataFrame({"date": frame["date"]}))
    dup = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02"] * 2), "kospi_pct": 0.0, "kosdaq_pct": 0.0, "v_kospi": 1.0}
    )
    with pytest.raises(ValueError, match="duplicate"):
        wf.attach_lagged_index(frame, dup)


def test_index_table_takes_one_row_per_date() -> None:
    ph = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "kospi_pct": [0.01, 0.01, 0.02],
            "kosdaq_pct": [0.0, 0.0, 0.1],
            "v_kospi": [15.0, 15.0, 16.0],
        }
    )
    table = wf.index_table_from_price_history(ph)
    assert len(table) == 2 and table["v_kospi"].tolist() == [15.0, 16.0]
    with pytest.raises(ValueError, match="missing index columns"):
        wf.index_table_from_price_history(ph[["date"]])


def test_empty_inputs_yield_empty_or_neutral_outputs() -> None:
    assert wf.portfolio_weights(pd.DataFrame({"f_tick_cost": []}), wf.WEIGHT_EQUAL).size == 0
    assert wf.series_metrics(np.array([np.nan]))["n_days"] == 0
    day = pd.Timestamp("2024-01-02")
    none_sel = pd.DataFrame({"date": [day], "symbol": ["a"], "pred": [0.1], "net_pit": [0.01], "f_tick_cost": [50.0]})
    row = wf.evaluate_portfolio(none_sel, wf.PortfolioConfig(top_k=1)).iloc[0]
    assert row["n_selectable"] == 0 and not row["active"] and row["pool_mean"] == 0.0
    assert np.isnan(wf.deflated_sharpe(np.zeros(5), 10))


def test_newey_west_t_matches_plain_t_without_autocorrelation_lags() -> None:
    x = np.array([0.01, 0.03, -0.01, 0.02, 0.00, 0.04])
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(x.size))
    assert wf.newey_west_t(x, lags=0) == pytest.approx(plain, rel=1e-12)


def test_recovered_bars_loader_reads_dated_files_and_tolerates_absence(tmp_path) -> None:
    pd.DataFrame({"symbol": ["000001"], "ts_hms": [90100]}).to_parquet(tmp_path / "2024-01-02.parquet", index=False)
    pd.DataFrame().to_parquet(tmp_path / "2024-01-03.parquet", index=False)
    load = wf.recovered_bars_loader(tmp_path)
    assert len(load("2024-01-02")) == 1
    assert load("2024-01-03").empty and load("2024-01-04").empty
    assert wf.recovered_bars_loader(None)("2024-01-02").empty


def test_default_grid_is_the_preregistered_size() -> None:
    models, portfolios = wf.default_grid()
    assert len(models) == 7 and len(portfolios) == 8
    assert {m.train_source for m in models} == set(wf.TRAIN_SOURCES)


def test_run_grid_registers_every_trial_and_leaderboard_deflates_by_trial_count(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = _calendar(60)
    ds = _dataset(days, n_names=6, seed=3)
    ds.loc[ds.index % 7 == 0, "tier"] = wf.TIER_RECONSTRUCTED
    monkeypatch.setattr(wf, "feature_columns", lambda _fs: ["x"])
    period = wf.Period("t", str(pd.Timestamp(days[30]).date()), str(pd.Timestamp(days[58]).date()))
    models = [wf.ModelConfig(train_source=wf.TRAIN_PIT_ONLY, retrain_every_days=10, min_train_days=5)]
    portfolios = [wf.PortfolioConfig(top_k=2), wf.PortfolioConfig(top_k=3, weighting=wf.WEIGHT_INV_TICK)]
    records = wf.run_grid(
        ds, period=period, models=models, portfolios=portfolios, out_dir=tmp_path, fit_fn=lambda tr, _f: _LinearFit(tr)
    )
    assert len(records) == 2
    again = wf.run_grid(
        ds, period=period, models=models, portfolios=portfolios, out_dir=tmp_path, fit_fn=lambda tr, _f: _LinearFit(tr)
    )
    assert [r["summary"] for r in again] == [r["summary"] for r in records]
    board = wf.leaderboard(tmp_path, "t")
    assert len(board) == 2 and (board["n_trials"] == 2).all()
    assert (board["alpha_bp"] > 0).all()
    assert {"year_2024", "recon_day_True"} <= set(records[0]["breakdown"])
    assert wf.TrialRegistry(tmp_path / "absent.jsonl").load() == []


def test_run_model_skips_blocks_without_enough_history_and_rejects_missing_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = _calendar(12)
    ds = _dataset(days)
    period = wf.Period("t", str(pd.Timestamp(days[2]).date()), str(pd.Timestamp(days[11]).date()))
    cfg = wf.ModelConfig(train_source=wf.TRAIN_PIT_EOD_AUG, min_train_days=500)
    monkeypatch.setattr(wf, "feature_columns", lambda _fs: ["x"])
    empty = wf.run_model(ds, period=period, config=cfg, fit_fn=lambda tr, _f: _LinearFit(tr))
    assert empty.empty and "pred" in empty.columns
    monkeypatch.setattr(wf, "feature_columns", lambda _fs: ["absent"])
    with pytest.raises(ValueError, match="missing feature columns"):
        wf.run_model(ds, period=period, config=cfg, fit_fn=lambda tr, _f: _LinearFit(tr))


def test_default_fit_fn_trains_a_seed_ensemble_that_ranks_the_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    train = pd.DataFrame({"x": x, "train_label": 0.01 * x + rng.normal(0, 1e-3, size=400)})
    cfg = wf.ModelConfig(
        model_params=(("n_estimators", 30), ("learning_rate", 0.1), ("num_leaves", 4), ("min_child_samples", 10)),
        seeds=(1, 2),
    )
    model = wf.default_fit_fn(cfg)(train, ["x"])
    pred = model.predict(pd.DataFrame({"x": [-2.0, 0.0, 2.0]}))
    assert pred[0] < pred[1] < pred[2]


def test_newey_west_t_is_nan_for_zero_variance() -> None:
    assert np.isnan(wf.newey_west_t(np.full(10, 0.01)))


def test_run_cpcv_never_trains_on_test_or_embargoed_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _calendar(40)
    ds = _dataset(days, n_names=4)
    monkeypatch.setattr(wf, "feature_columns", lambda _fs: ["x"])
    seen: list[set] = []

    def fit(train: pd.DataFrame, _f: list[str]) -> _LinearFit:
        seen.append(set(pd.to_datetime(train["date"])))
        return _LinearFit(train)

    cfg = wf.ModelConfig(train_source=wf.TRAIN_PIT_ONLY)
    out = wf.run_cpcv(
        ds,
        start=str(pd.Timestamp(days[0]).date()),
        end=str(pd.Timestamp(days[38]).date()),
        config=cfg,
        n_groups=4,
        k_test=1,
        embargo_groups=1,
        fit_fn=fit,
    )
    groups = np.array_split(np.array(days[:39]), 4)
    assert len(seen) == 4
    for i, train_days in enumerate(seen):
        banned = set(np.concatenate([groups[j] for j in range(max(0, i - 1), min(4, i + 2))]))
        assert not train_days & {pd.Timestamp(d) for d in banned}
    assert (out["n_folds"] == 1).all() and out["date"].nunique() == 39
    with pytest.raises(ValueError, match="at least n_groups"):
        wf.run_cpcv(ds, start="2024-01-01", end="2024-01-02", config=cfg, n_groups=4, k_test=1, fit_fn=fit)
    with pytest.raises(ValueError, match="k_test"):
        wf.run_cpcv(ds, start="2024-01-01", end="2024-03-01", config=cfg, n_groups=4, k_test=4, fit_fn=fit)


def test_rolling_window_and_decay_weights_restrict_and_downweight_old_rows() -> None:
    days = _calendar(30)
    ds = _dataset(days)
    cfg = wf.ModelConfig(train_source=wf.TRAIN_PIT_ONLY, train_window_days=10, embargo_days=1)
    train = wf.training_rows(ds, calendar=days, block_start=pd.Timestamp(days[25]), config=cfg)
    assert pd.to_datetime(train["date"]).min() >= pd.Timestamp(days[15])
    w = wf.sample_age_weights(
        pd.Series([days[20], days[15]]), calendar=days, block_start=pd.Timestamp(days[25]), half_life_days=5
    )
    assert w == pytest.approx([0.5, 0.25])
    grades = wf.lambdarank_grades(pd.DataFrame({"date": [days[0]] * 5, "train_label": [0.1, 0.2, 0.3, 0.4, 0.5]}))
    assert grades.tolist() == [0, 1, 2, 3, 4]
    assert "objective" not in wf.ModelConfig().identity()
    assert wf.ModelConfig(extra_features=("x_rv_late",)).features()[-1] == "x_rv_late"
    for bad in ({"objective": "nope"}, {"train_window_days": 0}, {"decay_half_life_days": 0.0}):
        with pytest.raises(ValueError, match="must"):
            wf.ModelConfig(**bad)


def test_decay_config_passes_age_weights_to_the_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _calendar(30)
    ds = _dataset(days)
    monkeypatch.setattr(wf, "feature_columns", lambda _fs: ["x"])
    captured: list[np.ndarray] = []

    def fit(train: pd.DataFrame, _f: list[str]) -> _LinearFit:
        captured.append(train[wf.SAMPLE_WEIGHT_COL].to_numpy())
        return _LinearFit(train)

    cfg = wf.ModelConfig(
        train_source=wf.TRAIN_PIT_ONLY, retrain_every_days=10, min_train_days=5, decay_half_life_days=5.0
    )
    period = wf.Period("t", str(pd.Timestamp(days[20]).date()), str(pd.Timestamp(days[29]).date()))
    wf.run_model(ds, period=period, config=cfg, calendar=days, fit_fn=fit)
    assert captured and captured[0].max() <= 1.0 and captured[0].min() < captured[0].max()


def test_default_fit_fn_lambdarank_orders_by_the_signal() -> None:
    rng = np.random.default_rng(1)
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=40), 10)
    x = rng.normal(size=len(dates))
    train = pd.DataFrame({"date": dates, "x": x, "train_label": 0.01 * x + rng.normal(0, 1e-3, size=len(dates))})
    train[wf.SAMPLE_WEIGHT_COL] = 1.0
    cfg = wf.ModelConfig(
        objective=wf.OBJECTIVE_LAMBDARANK,
        model_params=(("n_estimators", 30), ("learning_rate", 0.1), ("num_leaves", 4), ("min_child_samples", 5)),
        seeds=(1,),
    )
    model = wf.default_fit_fn(cfg)(train, ["x"])
    pred = model.predict(pd.DataFrame({"x": [-2.0, 0.0, 2.0]}))
    assert pred[0] < pred[1] < pred[2]
