from __future__ import annotations

import pandas as pd

from src.backfill.price.config import FetchConfig
from src.backfill.price.universe import _build_symbol_windows


def test_new_symbol_uses_requested_lookback_window() -> None:
    universe = pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]})
    cfg = FetchConfig(
        lookback_trading_days=10,
        fixed_start_date=pd.Timestamp("2016-01-01"),
        fixed_end_date=pd.Timestamp("2025-01-31"),
    )

    windows = _build_symbol_windows(universe, fetch_cfg=cfg)

    assert len(windows) == 1
    symbol, start, end, market = windows[0]
    assert symbol == "005930"
    assert start == pd.Timestamp("2025-01-17")
    assert end == pd.Timestamp("2025-01-31")
    assert market == "KOSPI"


def test_existing_symbol_keeps_incremental_overlap_window() -> None:
    universe = pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]})
    cfg = FetchConfig(
        lookback_trading_days=10,
        calendar_buffer_days=30,
        fixed_start_date=pd.Timestamp("2016-01-01"),
        fixed_end_date=pd.Timestamp("2025-01-31"),
    )

    windows = _build_symbol_windows(
        universe,
        fetch_cfg=cfg,
        existing_last_dates={"005930": pd.Timestamp("2025-01-20")},
    )

    assert windows[0][1] == pd.Timestamp("2024-12-21")

def test_default_backfill_end_date_tracks_today() -> None:
    import pandas as pd

    from src.backfill.price.config import FetchConfig, default_backfill_end_date

    # Arrange / Act
    today = pd.Timestamp.today().normalize()
    cfg = FetchConfig()

    # Assert
    assert default_backfill_end_date() == today
    assert cfg.fixed_end_date == today
    assert cfg.fixed_end_date > pd.Timestamp("2025-12-31")


def test_load_candidate_universe_includes_condition_history_symbols(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import universe as mod

    # Arrange
    monkeypatch.setattr(
        mod,
        "load_or_build_snapshot",
        lambda **kwargs: pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]}),
    )
    monkeypatch.setattr(
        mod,
        "load_candidate_universe_symbols",
        lambda: pd.DataFrame({"symbol": ["000660", "035720"], "market": ["KOSPI", "KOSDAQ"]}),
    )

    # Act
    out = mod._load_candidate_universe()

    # Assert
    assert set(out["symbol"]) == {"005930", "000660", "035720"}
    assert out["symbol"].is_unique
    assert list(out.columns) == ["symbol", "market"]

def test_load_candidate_universe_falls_back_to_trade_log_on_candidate_failure(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import universe as mod

    # Arrange: candidate source raises; trade-log universe must still load.
    monkeypatch.setattr(
        mod,
        "load_or_build_snapshot",
        lambda **kwargs: pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]}),
    )

    def _boom() -> pd.DataFrame:
        raise RuntimeError("candidate store unavailable")

    monkeypatch.setattr(mod, "load_candidate_universe_symbols", _boom)

    # Act
    out = mod._load_candidate_universe()

    # Assert
    assert set(out["symbol"]) == {"005930"}
    assert list(out.columns) == ["symbol", "market"]
