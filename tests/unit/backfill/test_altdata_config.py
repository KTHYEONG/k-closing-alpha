from pathlib import Path

import pandas as pd
import pytest

from src.backfill.altdata.config import AltDataFetchConfig


def test_altdata_config_rejects_invalid_domains() -> None:
    ok = AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"))
    assert ok.sources[0] == "shorting"
    with pytest.raises(ValueError, match="start"):
        AltDataFetchConfig(start=pd.Timestamp("2020-02-01"), end=pd.Timestamp("2020-01-01"), out_dir=Path("x"))
    with pytest.raises(ValueError, match="source"):
        AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"), sources=("orderbook",))
    with pytest.raises(ValueError, match="market"):
        AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"), markets=("NASDAQ",))
    with pytest.raises(ValueError, match="pykrx_requests_per_sec"):
        AltDataFetchConfig(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-02-01"), out_dir=Path("x"), pykrx_requests_per_sec=0.0)


import pathlib


def test_altdata_package_has_no_ml_or_realtime_imports() -> None:
    bad = []
    for p in pathlib.Path("src/backfill/altdata").rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        if "src.ml" in t or "src.serving" in t:
            bad.append((str(p), "ml/serving import"))
        if "inquire-asking-price" in t or "websocket" in t.lower():
            bad.append((str(p), "realtime endpoint"))
    assert bad == [], bad
