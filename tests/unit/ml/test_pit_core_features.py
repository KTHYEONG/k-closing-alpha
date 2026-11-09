from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import pit_core_features as core
from src.ml.research import pit_ext_features as xf
from src.ml.research import pit_walkforward as wf
from src.ml.topk_contract import PIT_CORE_FEATURE_COLS

_SYMBOLS = ("000001", "000002", "000003")
_MISSING = (("000002", 20),)


def _panel(n_dates: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    index = pd.DataFrame(
        {
            "date": dates,
            "kospi_pct": rng.normal(0, 0.01, n_dates),
            "kosdaq_pct": rng.normal(0, 0.012, n_dates),
            "v_kospi": 20 + rng.normal(0, 1, n_dates),
        }
    )
    rows = []
    for symbol in _SYMBOLS:
        close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.02, n_dates)))
        prev = np.concatenate([[close[0]], close[:-1]])
        frame = index.assign(
            symbol=symbol,
            close=close,
            prev_close=prev,
            trade_value_100m=rng.uniform(50, 500, n_dates),
            market_cap_100m=rng.uniform(1000, 5000, n_dates),
            volume=rng.uniform(1e5, 1e6, n_dates),
            program_netbuy=rng.normal(0, 10, n_dates),
        )
        rows.append(frame)
    ph = pd.concat(rows, ignore_index=True)
    drop = {(s, dates[i]) for s, i in _MISSING}
    keep = [(s, d) not in drop for s, d in zip(ph["symbol"], ph["date"], strict=True)]
    return ph[np.asarray(keep)].reset_index(drop=True)


def test_core_feature_cols_match_research_core() -> None:
    config = wf.ModelConfig(
        feature_set="idx_lag1",
        extra_features=("x_rv_late", "x_rv1m", "x_r30", "x_r_pm", "h_amihud20", "h_turnover20"),
    )
    assert config.features() == PIT_CORE_FEATURE_COLS


def test_core_history_matches_research_stock_history() -> None:
    ph = _panel()
    mine = core.core_history_features(ph)
    research = xf.stock_history(ph)
    merged = mine.merge(research, on=["date", "symbol"], suffixes=("_mine", ""), validate="one_to_one")
    assert len(merged) == len(ph)
    for name in core.PIT_CORE_HISTORY_FEATURES:
        np.testing.assert_allclose(
            merged[f"{name}_mine"].to_numpy(dtype=np.float64),
            merged[name].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=0.0,
            equal_nan=True,
        )


def test_latest_history_equals_training_next_row() -> None:
    ph = _panel()
    cut = pd.Timestamp(sorted(ph["date"].unique())[30])
    latest = core.latest_core_history(ph[ph["date"] < cut]).set_index("symbol")
    at_cut = core.core_history_features(ph)
    at_cut = at_cut[at_cut["date"] == cut].set_index("symbol")
    assert set(at_cut.index) <= set(latest.index)
    for symbol in at_cut.index:
        for name in core.PIT_CORE_HISTORY_FEATURES:
            assert latest.loc[symbol, name] == pytest.approx(at_cut.loc[symbol, name], rel=1e-12, abs=0.0)


def test_latest_index_lag_equals_attach_lagged_index() -> None:
    ph = _panel()
    cut = pd.Timestamp(sorted(ph["date"].unique())[30])
    attached = wf.attach_lagged_index(pd.DataFrame({"date": [cut]}), wf.index_table_from_price_history(ph))
    got = core.latest_index_lag(ph[ph["date"] < cut])
    assert list(got) == list(core.PIT_CORE_INDEX_LAG_FEATURES)
    for name in core.PIT_CORE_INDEX_LAG_FEATURES:
        assert got[name] == pytest.approx(float(attached[name].iloc[0]), rel=1e-12, abs=0.0)


def test_latest_index_lag_rejects_nonfinite() -> None:
    ph = _panel()
    last = ph["date"].max()
    ph.loc[ph["date"] == last, "v_kospi"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        core.latest_index_lag(ph)


def test_latest_index_lag_rejects_empty_history() -> None:
    ph = _panel().iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        core.latest_index_lag(ph)


def test_core_history_ignores_same_and_later_rows() -> None:
    ph = _panel()
    cut = pd.Timestamp(sorted(ph["date"].unique())[30])
    perturbed = ph.copy()
    later = perturbed["date"] >= cut
    perturbed.loc[later, ["close", "prev_close", "trade_value_100m", "market_cap_100m"]] = 1e6
    base = core.core_history_features(ph)
    shifted = core.core_history_features(perturbed)
    before = base["date"] <= cut
    np.testing.assert_array_equal(base.loc[before, "symbol"].to_numpy(), shifted.loc[before, "symbol"].to_numpy())
    for name in core.PIT_CORE_HISTORY_FEATURES:
        np.testing.assert_allclose(
            base.loc[before, name].to_numpy(dtype=np.float64),
            shifted.loc[before, name].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_serving_config_rejects_bad_coverage(bad: float) -> None:
    with pytest.raises(ValueError, match="min_path_coverage"):
        core.PitCoreServingConfig(min_path_coverage=bad)


def test_serving_config_accepts_full_coverage() -> None:
    assert core.PitCoreServingConfig(min_path_coverage=1.0).min_path_coverage == 1.0


def test_latest_index_lag_rejects_missing_index_column() -> None:
    ph = _panel().drop(columns=["v_kospi"])
    with pytest.raises(ValueError, match="missing index columns"):
        core.latest_index_lag(ph)
