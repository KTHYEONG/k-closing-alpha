from pathlib import Path

import pandas as pd

import pytest

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.ratelimit import DartNonRetryableError, retry_call


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


def test_retry_call_fails_fast_on_dart_nonretryable_error() -> None:
    """DART 계정 한도초과처럼 재시도해도 회복되지 않는 오류는 즉시 실패해야 한다.

    k-stock-engine 과 DART_API_KEY 를 공유하므로 한도초과가 하루 중 언제든 발생할
    수 있는데, 재시도로 낭비되는 수 분(cfg.retry_sleep_sec 누적)을 없애고 원래
    예외를 그대로 호출자에게 전달한다.
    """
    calls = {"n": 0}

    def _quota_exceeded() -> int:
        calls["n"] += 1
        raise DartNonRetryableError("DART error status=020 msg=사용한도를 초과하였습니다.")

    with pytest.raises(DartNonRetryableError):
        retry_call(_quota_exceeded, _cfg(), label="dart list p1")
    assert calls["n"] == 1
