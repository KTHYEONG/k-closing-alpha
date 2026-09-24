"""레이트 리밋 및 재시도 유틸리티."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from src.backfill.altdata.config import AltDataFetchConfig
from src.data.capture_contracts import RawCaptureError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_PYKRX_LOCK = threading.Lock()
_PYKRX_NEXT = 0.0

_DART_LOCK = threading.Lock()
_DART_NEXT = 0.0

_KRX_LOCK = threading.Lock()
_KRX_NEXT = 0.0


class DartNonRetryableError(RuntimeError):
    """DART account-level error unresolvable by retrying (e.g. daily quota exhausted).

    다른 프로젝트(k-stock-engine)와 DART_API_KEY 를 공유하므로 하루 중 언제든
    계정 한도가 소진될 수 있다. 재시도해도 당일 내 회복되지 않으므로 즉시 실패시킨다.
    """


class DartQuotaExhaustedError(DartNonRetryableError):
    """Every configured DART key returned the daily-quota status in this process; the message lists the key labels only."""


class DartKeysUnusableError(DartNonRetryableError):
    """No DART key is usable and at least one was rejected for a non-quota reason (unregistered, unusable, or IP-blocked key); the message lists label=reason pairs only."""


def wait_for_pykrx_slot(cfg: AltDataFetchConfig) -> None:
    """pykrx 호출 간격을 제한합니다.

    Args:
        cfg: Alt-data 설정.
    """
    global _PYKRX_NEXT
    interval = 1.0 / max(0.1, float(cfg.pykrx_requests_per_sec))
    with _PYKRX_LOCK:
        now = time.monotonic()
        wait = max(0.0, _PYKRX_NEXT - now)
        _PYKRX_NEXT = max(now, _PYKRX_NEXT) + interval
    if wait > 0:
        time.sleep(wait)


def wait_for_dart_slot(cfg: AltDataFetchConfig) -> None:
    """DART 호출 간격을 제한합니다.

    Args:
        cfg: Alt-data 설정.
    """
    global _DART_NEXT
    interval = 1.0 / max(0.1, float(cfg.dart_requests_per_sec))
    with _DART_LOCK:
        now = time.monotonic()
        wait = max(0.0, _DART_NEXT - now)
        _DART_NEXT = max(now, _DART_NEXT) + interval
    if wait > 0:
        time.sleep(wait)


def wait_for_krx_slot(cfg: AltDataFetchConfig) -> None:
    """KRX Open API 호출 간격을 제한합니다.

    Args:
        cfg: Alt-data 설정.
    """
    global _KRX_NEXT
    interval = 1.0 / max(0.1, float(cfg.krx_requests_per_sec))
    with _KRX_LOCK:
        now = time.monotonic()
        wait = max(0.0, _KRX_NEXT - now)
        _KRX_NEXT = max(now, _KRX_NEXT) + interval
    if wait > 0:
        time.sleep(wait)


def retry_call(fn: Callable[[], _T], cfg: AltDataFetchConfig, *, label: str) -> _T | None:
    """Retry source operations while allowing durable-capture failures to escape.

    Args:
        fn: Existing source operation.
        cfg: Existing retry configuration.
        label: Credential-free diagnostic label.
    Returns:
        Successful operation result or None after existing source retries.
    Raises:
        RawCaptureError: Durable observation failed; it is not a recoverable source error.
    """
    last_exc: Exception | None = None
    for attempt in range(max(1, int(cfg.retries))):
        try:
            return fn()
        except RawCaptureError:
            raise
        except DartNonRetryableError as exc:
            logger.warning("[DATA] stage=altdata_retry label=%s status=FAIL_FATAL error=%s", label, exc)
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < int(cfg.retries) - 1:
                time.sleep(float(cfg.retry_sleep_sec) * (attempt + 1))
    if last_exc is not None:
        logger.warning("[DATA] stage=altdata_retry label=%s status=FAIL error=%s", label, last_exc)
    return None
