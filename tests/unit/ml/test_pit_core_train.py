"""PIT-CORE training store, training frame and CORE bundle trainer invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import pit_core_train as core
from src.ml.topk_contract import (
    PIT_CORE_CONTRACT_VERSION_KEY,
    PIT_CORE_FEATURE_COLS,
    RANKER_FEATURE_COLS,
    feature_contract_issue,
)

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
_INDEX_STEP = 1e-4


def _synth_ph(n_days: int = 90, n_symbols: int = 30, seed: int = 7):
    from src.data.panel_integrity import prepare_price_panel

    dates = pd.bdate_range("2023-02-01", periods=n_days)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_symbols):
        symbol = f"{i:06d}"
        prev = (15000.0 + i * 50.0) / 1.05
        lane = float(i) / max(1, n_symbols - 1) - 0.5
        for k, day in enumerate(dates):
            close = prev * 1.05 * (1.0 + lane * 0.004 + float(rng.normal(0.0, 0.002)))
            rows.append({
                "date": day, "symbol": symbol, "open": prev, "high": max(prev, close) * 1.01,
                "low": min(prev, close) * 0.99, "close": close, "close_raw": close, "prev_close": prev,
                "volume": 1e6, "market_cap_100m": 3000.0, "trade_value_100m": 500.0, "market": "KOSPI",
                "daily_change_pct": 5.0, "inst_netbuy": 0.0, "foreign_netbuy": 0.0, "program_netbuy": 0.0,
                "kospi_pct": _INDEX_STEP * (k + 1), "kosdaq_pct": 0.002, "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
    ph, _prov = prepare_price_panel(pd.DataFrame(rows))
    ph["is_screenable"] = True
    ph["screenable_source"] = "real"
    market_dates = np.array(sorted(ph["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}
    return ph, market_dates, d_to_idx


def _store_slice(ph, day_slice: slice, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.DatetimeIndex(sorted(ph["date"].unique()))[day_slice]
    rows = ph.loc[ph["date"].isin(days), ["date", "symbol"]].copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows["symbol"] = rows["symbol"].astype(str)
    rows["tier"] = "exact"
    for col in (*RANKER_FEATURE_COLS, *core.INTRADAY_FEATURES):
        rows[col] = rng.normal(size=len(rows))
    return rows[list(core.PIT_STORE_COLUMNS)].reset_index(drop=True)


def _build(ph, market_dates, d_to_idx, store, raw=None):
    raw = ph[["date", "symbol", "close", "prev_close", "trade_value_100m", "market_cap_100m",
              "kospi_pct", "kosdaq_pct", "v_kospi"]].copy() if raw is None else raw
    return core.build_core_training_frame(
        ph, market_dates, d_to_idx, store=store, raw_history=raw, label_clip=0.2,
    )


def _store_day_rows(day: str, symbols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(symbols)
    frame = pd.DataFrame({"date": [pd.Timestamp(day)] * n, "symbol": symbols, "tier": ["exact"] * n})
    for col in (*RANKER_FEATURE_COLS, *core.INTRADAY_FEATURES):
        frame[col] = rng.normal(size=n)
    return frame[list(core.PIT_STORE_COLUMNS)]


def test_store_day_roundtrip_and_immutable(tmp_path) -> None:
    rows = _store_day_rows("2026-03-02", ["000001", "000002", "000003"])

    core.write_store_day(tmp_path, pd.Timestamp("2026-03-02"), rows)
    back = core.read_pit_store(tmp_path)

    pd.testing.assert_frame_equal(back, rows.reset_index(drop=True), check_like=False)
    with pytest.raises(FileExistsError):
        core.write_store_day(tmp_path, pd.Timestamp("2026-03-02"), rows)


def test_store_rejects_foreign_tier_or_mixed_dates(tmp_path) -> None:
    eod = _store_day_rows("2026-03-02", ["000001"])
    eod["tier"] = "eod"
    with pytest.raises(ValueError, match="tiers"):
        core.write_store_day(tmp_path, pd.Timestamp("2026-03-02"), eod)

    mixed = pd.concat([_store_day_rows("2026-03-02", ["000001"]), _store_day_rows("2026-03-03", ["000002"])])
    with pytest.raises(ValueError, match="dates"):
        core.write_store_day(tmp_path, pd.Timestamp("2026-03-02"), mixed)


def test_seed_writes_only_pit_tiers_with_path_features(tmp_path) -> None:
    rng = np.random.default_rng(1)
    keys = [("2026-03-02", "000001"), ("2026-03-02", "000002"), ("2026-03-03", "000001")]
    dataset = pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d, _ in keys],
            "symbol": [s for _, s in keys],
            "tier": ["exact", "eod", "exact"],
        }
    )
    for col in RANKER_FEATURE_COLS:
        dataset[col] = rng.normal(size=len(dataset))
    ext = dataset[["date", "symbol"]].copy()
    for col in core.INTRADAY_FEATURES:
        ext[col] = rng.normal(size=len(ext))

    written = core.seed_pit_store(dataset, ext, tmp_path)
    stored = core.read_pit_store(tmp_path)

    assert written == 2
    assert len(stored) == 2
    assert set(stored["tier"]) == {"exact"}
    expected = ext[dataset["tier"].eq("exact").to_numpy()].reset_index(drop=True)
    for col in core.INTRADAY_FEATURES:
        np.testing.assert_allclose(stored[col].to_numpy(), expected[col].to_numpy(), rtol=1e-12, atol=0.0)
    assert core.seed_pit_store(dataset, ext, tmp_path) == 0


def test_training_frame_eod_rows_precede_store_and_lack_core_extras() -> None:
    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-25, None))

    frame = _build(ph, market_dates, d_to_idx, store)

    store_start = store["date"].min()
    eod = frame[frame["tier"] == "eod"]
    stored = frame[frame["tier"] != "eod"]
    assert len(eod) > 0 and len(stored) > 0
    assert (eod["date"] < store_start).all()
    for col in ("x_rv_late", "x_rv1m", "x_r30", "x_r_pm", "h_amihud20", "h_turnover20"):
        assert eod[col].isna().all(), col
    expected_h = core.core_history_features(ph[["date", "symbol", "close", "prev_close", "trade_value_100m",
                                              "market_cap_100m", "kospi_pct", "kosdaq_pct", "v_kospi"]])
    merged = stored.merge(expected_h, on=["date", "symbol"], suffixes=("", "_exp"), validate="one_to_one")
    for col in ("h_amihud20", "h_turnover20"):
        np.testing.assert_allclose(merged[col].to_numpy(), merged[f"{col}_exp"].to_numpy(), rtol=1e-12, equal_nan=True)


def test_training_frame_labels_ignore_future_perturbation() -> None:
    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-25, -12))
    base = _build(ph, market_dates, d_to_idx, store)
    resolved = base[base["exit_date"].notna()]
    cutoff = pd.Timestamp(resolved["exit_date"].max()).normalize()
    assert cutoff < pd.Timestamp(ph["date"].max()).normalize()

    perturbed = ph.copy()
    future = pd.to_datetime(perturbed["date"]).dt.normalize() > cutoff
    for col in ("open", "high", "low", "close", "close_raw", "prev_close"):
        perturbed.loc[future, col] = perturbed.loc[future, col] * 1.3
    raw = ph[["date", "symbol", "close", "prev_close", "trade_value_100m", "market_cap_100m",
              "kospi_pct", "kosdaq_pct", "v_kospi"]].copy()
    raw_future = pd.to_datetime(raw["date"]).dt.normalize() > cutoff
    raw.loc[raw_future, "close"] = raw.loc[raw_future, "close"] * 1.3
    shifted = _build(perturbed, market_dates, d_to_idx, store, raw=raw)

    key = ["date", "symbol", "tier"]
    left = base.loc[base["exit_date"].notna(), key + list(PIT_CORE_FEATURE_COLS) + ["train_label"]]
    right = shifted.loc[shifted["exit_date"].notna(), key + list(PIT_CORE_FEATURE_COLS) + ["train_label"]]
    left = left.sort_values(key).reset_index(drop=True)
    right = right.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, rtol=1e-9, atol=1e-12)


def test_training_frame_has_lagged_index_on_all_rows() -> None:
    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-25, None))

    frame = _build(ph, market_dates, d_to_idx, store)

    idx = pd.to_datetime(frame["date"]).dt.normalize().map(lambda d: d_to_idx[np.datetime64(d)])
    expected = _INDEX_STEP * idx.to_numpy(dtype=np.float64)
    first = idx.to_numpy() == 0
    assert first.any() and (~first).any()
    assert np.isnan(frame["kospi_pct_lag1"].to_numpy(dtype=np.float64)[first]).all()
    np.testing.assert_allclose(frame["kospi_pct_lag1"].to_numpy(dtype=np.float64)[~first], expected[~first], rtol=1e-9)


def test_train_core_bundle_stamps_contract_and_columns() -> None:
    from src.ml.topk_contract import score_topk_candidates

    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-25, None))
    frame = _build(ph, market_dates, d_to_idx, store)

    bundle = core.train_core_bundle(frame, model_params=_TINY_PARAMS, seeds=(1,), min_train_rows=50)

    assert bundle["feature_cols"] == PIT_CORE_FEATURE_COLS
    assert bundle[PIT_CORE_CONTRACT_VERSION_KEY] == "1"
    assert bundle["model_recipe"] == "pit_core_v1"
    assert feature_contract_issue(bundle) is None
    scored = score_topk_candidates(frame.loc[frame["tier"] != "eod"].reset_index(drop=True), bundle)
    assert len(scored) == int((frame["tier"] != "eod").sum())


def test_train_core_bundle_excludes_unresolved_rows(monkeypatch) -> None:
    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-12, None))
    frame = _build(ph, market_dates, d_to_idx, store)
    unresolved = frame["exit_date"].isna()
    assert unresolved.any() and (~unresolved).any()
    captured: dict[str, pd.DataFrame] = {}

    def _fake_inline(fit, feature_cols, target_col, group_col, **kwargs):
        captured["fit"] = fit
        return {"feature_cols": list(feature_cols)}

    monkeypatch.setattr(core, "build_inline_bundle", _fake_inline)
    core.train_core_bundle(frame, model_params=_TINY_PARAMS, seeds=(1,), min_train_rows=1)

    fit = captured["fit"]
    assert fit["exit_date"].notna().all()
    assert np.isfinite(fit["train_label"].to_numpy(dtype=np.float64)).all()
    assert len(fit) == int((~unresolved & np.isfinite(frame["train_label"].astype(float))).sum())


def test_gate_eval_frame_is_recent_pit_rows() -> None:
    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-25, None))
    frame = _build(ph, market_dates, d_to_idx, store)

    gate = core.build_core_gate_eval_frame(frame, eval_days=3)

    assert (gate["tier"] != "eod").all()
    dates = pd.to_datetime(gate["date"]).dt.normalize()
    last_three = sorted(pd.to_datetime(store["date"]).dt.normalize().unique())[-3:]
    assert set(dates.unique()) == set(last_three)
    assert set(PIT_CORE_FEATURE_COLS) <= set(gate.columns)


def test_store_rejects_missing_columns_and_duplicate_keys(tmp_path) -> None:
    rows = _store_day_rows("2026-03-02", ["000001", "000002"])
    with pytest.raises(ValueError, match="missing columns"):
        core.write_store_day(tmp_path, pd.Timestamp("2026-03-02"), rows.drop(columns=["x_r30"]))

    duplicated = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        core.write_store_day(tmp_path, pd.Timestamp("2026-03-02"), duplicated)


def test_read_pit_store_rejects_empty_store(tmp_path) -> None:
    with pytest.raises(ValueError, match="no days"):
        core.read_pit_store(tmp_path)


def test_train_core_bundle_rejects_short_top_k_and_thin_or_unresolved_frames() -> None:
    import dataclasses

    from src.strategy.contract import PRODUCTION_STRATEGY

    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-25, None))
    frame = _build(ph, market_dates, d_to_idx, store)
    short_k = dataclasses.replace(PRODUCTION_STRATEGY, top_k=2)
    with pytest.raises(ValueError, match="minimum investable"):
        core.train_core_bundle(frame, spec=short_k, model_params=_TINY_PARAMS, seeds=(1,), min_train_rows=1)

    with pytest.raises(ValueError, match="below min_train_rows"):
        core.train_core_bundle(frame, model_params=_TINY_PARAMS, seeds=(1,), min_train_rows=10**9)

    eod_only = frame[frame["tier"] == "eod"]
    with pytest.raises(ValueError, match="no PIT store rows"):
        core.train_core_bundle(eod_only, model_params=_TINY_PARAMS, seeds=(1,), min_train_rows=1)


def test_train_core_bundle_rejects_feature_set_drift(monkeypatch) -> None:
    ph, market_dates, d_to_idx = _synth_ph()
    store = _store_slice(ph, slice(-25, None))
    frame = _build(ph, market_dates, d_to_idx, store)

    def _drifted(fit, feature_cols, target_col, group_col, **kwargs):
        return {"feature_cols": list(feature_cols)[:-1]}

    monkeypatch.setattr(core, "build_inline_bundle", _drifted)
    with pytest.raises(RuntimeError, match="differ from PIT_CORE_FEATURE_COLS"):
        core.train_core_bundle(frame, model_params=_TINY_PARAMS, seeds=(1,), min_train_rows=1)


def test_main_fails_closed_without_private_store(monkeypatch, capsys) -> None:
    from src.strategy.store import StrategyStoreError

    def _no_store():
        raise StrategyStoreError("private strategy store required")

    monkeypatch.setattr(core, "require_production_store", _no_store)
    with pytest.raises(SystemExit) as exc:
        core.main(["extend", "--through", "2026-03-02"])
    assert exc.value.code == 1


def test_main_dispatches_seed_and_extend(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(2)
    dataset = pd.DataFrame({
        "date": [pd.Timestamp("2026-03-02")], "symbol": ["000001"], "tier": ["exact"],
        **{c: rng.normal(size=1) for c in RANKER_FEATURE_COLS},
    })
    ext = pd.DataFrame({
        "date": [pd.Timestamp("2026-03-02")], "symbol": ["000001"],
        **{c: rng.normal(size=1) for c in core.INTRADAY_FEATURES},
    })
    dataset_path = tmp_path / "dataset.parquet"
    ext_path = tmp_path / "ext.parquet"
    dataset.to_parquet(dataset_path, index=False)
    ext.to_parquet(ext_path, index=False)
    store_dir = tmp_path / "store"
    extended: list[pd.Timestamp] = []
    monkeypatch.setattr(core, "require_production_store", lambda: None)
    monkeypatch.setattr(core, "pit_store_dir", lambda: store_dir)
    monkeypatch.setattr(core, "extend_pit_store", lambda d, *, through: extended.append(through) or [])

    core.main(["seed", "--dataset", str(dataset_path), "--ext", str(ext_path)])
    core.main(["extend", "--through", "2026-03-03"])

    assert len(core.read_pit_store(store_dir)) == 1
    assert extended == [pd.Timestamp("2026-03-03")]
