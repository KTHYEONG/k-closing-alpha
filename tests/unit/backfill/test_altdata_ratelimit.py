from pathlib import Path

import pandas as pd

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.ratelimit import retry_call


def _cfg() -> AltDataFetchConfig:
    return AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-01-05"),
        out_dir=Path("x"), retries=3, retry_sleep_sec=0.0,
    )


def test_retry_call_returns_none_after_exhaustion() -> None:
    calls = {"n": 0}

    def _boom() -> int:
        calls["n"] += 1
        raise RuntimeError("krx down")

    assert retry_call(_boom, _cfg(), label="boom") is None
    assert calls["n"] == 3
    assert retry_call(lambda: 42, _cfg(), label="ok") == 42
