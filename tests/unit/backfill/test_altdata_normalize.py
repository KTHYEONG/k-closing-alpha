from pathlib import Path

import pandas as pd

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.normalize import normalize_panel


def test_normalize_panel_dedups_and_filters_universe() -> None:
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
        "symbol": ["5930", "5930", "660"],
        "per": [1.0, 2.0, 3.0],
    })
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-02-01"),
        out_dir=Path("x"), universe_symbols=frozenset({"005930"}),
    )
    out = normalize_panel(raw, "fundamental", cfg)
    assert list(out["symbol"]) == ["005930"]
    assert out["per"].iloc[0] == 2.0
    assert str(out["date"].dtype).startswith("datetime64")
