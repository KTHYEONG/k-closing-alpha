"""백필 설정(FetchConfig) 및 KIS 레이트-리밋 정책."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class PipelineConfig:
    """레거시 `src.pipeline.config.PipelineConfig`의 하위 호환 최소 설정."""

    def __init__(self, snapshot_path: Path | None = None) -> None:
        self.snapshot_path: Path | None = snapshot_path


DEFAULT_CONFIG = PipelineConfig()


KRW_100M = 100_000_000.0


@dataclass(frozen=True)
class FetchConfig:
    lookback_trading_days: int = 40
    calendar_buffer_days: int = 120
    max_workers: int = 4
    retries: int = 3
    retry_sleep_sec: float = 0.8
    request_sleep_sec: float = 0.03
    fixed_start_date: pd.Timestamp = pd.Timestamp("2016-01-01")
    fixed_end_date: pd.Timestamp = pd.Timestamp("2025-12-31")
    # KIS official samples commonly state REST guidance around 20 req/s.
    # Keep a conservative default to protect stability under parallel backfill.
    kis_rest_limit_per_sec: float = 20.0
    kis_rest_safety_ratio: float = 0.6
    kis_max_parallel_calls: int = 1
    pykrx_requests_per_sec: float = 8.0
    include_flows: bool = True
    force_full_history: bool = False


_KIS_SEMAPHORE: threading.Semaphore | None = None
_KIS_SEMAPHORE_SIZE = 0
_PYKRX_LOCK = threading.Lock()
_PYKRX_NEXT_REQUEST = 0.0


def _effective_kis_sleep_sec(fetch_cfg: FetchConfig) -> float:
    safe_rps = max(1e-6, float(fetch_cfg.kis_rest_limit_per_sec) * float(fetch_cfg.kis_rest_safety_ratio))
    return max(float(fetch_cfg.request_sleep_sec), 1.0 / safe_rps)


def _ensure_kis_semaphore(fetch_cfg: FetchConfig) -> threading.Semaphore:
    global _KIS_SEMAPHORE, _KIS_SEMAPHORE_SIZE
    size = max(1, int(fetch_cfg.kis_max_parallel_calls))
    if _KIS_SEMAPHORE is None or size != _KIS_SEMAPHORE_SIZE:
        _KIS_SEMAPHORE = threading.Semaphore(size)
        _KIS_SEMAPHORE_SIZE = size
    return _KIS_SEMAPHORE


@contextmanager
def _kis_slot(fetch_cfg: FetchConfig):
    sem = _ensure_kis_semaphore(fetch_cfg)
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def _wait_for_pykrx_slot(fetch_cfg: FetchConfig) -> None:
    """프로세스 전체에서 pykrx 호출률을 제한합니다."""
    global _PYKRX_NEXT_REQUEST
    interval = 1.0 / max(0.1, float(fetch_cfg.pykrx_requests_per_sec))
    with _PYKRX_LOCK:
        now = time.monotonic()
        wait = max(0.0, _PYKRX_NEXT_REQUEST - now)
        _PYKRX_NEXT_REQUEST = max(now, _PYKRX_NEXT_REQUEST) + interval
    if wait > 0:
        time.sleep(wait)
