"""Invariant guards for the EOD fetch superset (fetch filter only, never a screen)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.collection import CollectionSettings
from src.data.eod_superset import EodSupersetScreen, eod_superset_mask
from src.strategy.contract import DEFAULT_UNIVERSE, select_universe


def _profile(**overrides) -> CollectionSettings:
    return CollectionSettings(_env_file=None, **overrides)


def _screen(**overrides) -> EodSupersetScreen:
    return EodSupersetScreen.from_profile(_profile(**overrides))


def _rows(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def test_eod_superset_mask_bounds_are_inclusive_exclusive() -> None:
    screen = _screen()
    panel = _rows([
        {"symbol": "005930", "chg_ratio": 0.01, "tv_clean": 100.0, "mc_clean": 495.0, "volume": 10.0},
        {"symbol": "000660", "chg_ratio": 0.12, "tv_clean": 500.0, "mc_clean": 900.0, "volume": 10.0},
        {"symbol": "005380", "chg_ratio": 0.05, "tv_clean": 100.0, "mc_clean": 495.0, "volume": 10.0},
    ])
    assert eod_superset_mask(panel, screen).tolist() == [True, False, True]


def test_eod_superset_mask_nan_fails_closed() -> None:
    screen = _screen()
    panel = _rows([
        {"symbol": "005930", "chg_ratio": float("nan"), "tv_clean": 500.0, "mc_clean": 900.0, "volume": 10.0},
        {"symbol": "000660", "chg_ratio": 0.05, "tv_clean": float("nan"), "mc_clean": 900.0, "volume": 10.0},
        {"symbol": "005380", "chg_ratio": 0.05, "tv_clean": 500.0, "mc_clean": float("nan"), "volume": 10.0},
    ])
    assert eod_superset_mask(panel, screen).tolist() == [False, False, False]


def test_eod_superset_mask_excludes_zero_volume() -> None:
    screen = _screen()
    panel = _rows([
        {"symbol": "005930", "chg_ratio": 0.05, "tv_clean": 500.0, "mc_clean": 900.0, "volume": 0.0},
    ])
    assert eod_superset_mask(panel, screen).tolist() == [False]


def test_eod_superset_mask_common_stock_rule() -> None:
    base = [
        {"symbol": code, "chg_ratio": 0.05, "tv_clean": 500.0, "mc_clean": 900.0, "volume": 10.0}
        for code in ("005930", "005935", "900110", "0008Z0")
    ]
    panel = _rows(base)
    assert eod_superset_mask(panel, _screen()).tolist() == [True, True, True, True]
    assert eod_superset_mask(panel, _screen(COLLECTION_PIT_BACKFILL_COMMON_STOCK_ONLY=True)).tolist() == [
        True, False, False, True,
    ]


def test_eod_superset_mask_contains_default_universe() -> None:
    rng = np.random.default_rng(1520)
    n = 2000
    panel = pd.DataFrame({
        "symbol": [f"{100000 + i:06d}" for i in range(n)],
        "chg_ratio": rng.uniform(-0.05, 0.15, n),
        "tv_clean": rng.uniform(50.0, 500.0, n),
        "mc_clean": rng.uniform(300.0, 1000.0, n),
        "close": np.full(n, 50000.0),
        "volume": rng.uniform(1.0, 1000.0, n),
        "is_ceiling": np.zeros(n, dtype=bool),
    })
    mask = eod_superset_mask(panel, _screen())
    universe = select_universe(panel, DEFAULT_UNIVERSE)
    assert bool(np.all(mask[universe]))


def test_eod_superset_from_profile_rejects_non_superset() -> None:
    with pytest.raises(ValueError, match="min_change_ratio"):
        _screen(COLLECTION_PIT_BACKFILL_MIN_CHANGE_RATIO=0.03)
    with pytest.raises(ValueError, match="max_change_ratio"):
        _screen(COLLECTION_PIT_BACKFILL_MAX_CHANGE_RATIO=0.09)
    with pytest.raises(ValueError, match="min_trade_value_100m"):
        _screen(COLLECTION_PIT_BACKFILL_MIN_TRADE_VALUE_100M=100.01)
    with pytest.raises(ValueError, match="min_market_cap_100m"):
        _screen(COLLECTION_PIT_BACKFILL_MIN_MARKET_CAP_100M=500.01)


def test_eod_superset_mask_names_missing_columns() -> None:
    panel = _rows([
        {"symbol": "005930", "chg_ratio": 0.05, "volume": 10.0},
    ])
    with pytest.raises(ValueError, match="tv_clean") as exc:
        eod_superset_mask(panel, _screen())
    assert "mc_clean" in str(exc.value)
