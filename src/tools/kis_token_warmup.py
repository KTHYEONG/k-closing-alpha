"""이 호스트가 하루 1회 발급 책임을 지는 KIS 토큰 전부를 예열한다.

발급 대상은 key_pool의 선언(호스트 배정 데이터 슬롯 + HOST_ISSUED_KEY_SPECS)이
단일 원천이며, 다른 잡과 타 프로젝트 컨테이너는 캐시를 읽기만 한다. 발급 호출
성공과 캐시 반영은 별개이므로 기록까지 확인해야 소비자가 실제로 토큰을 읽는다.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

from src import settings
from src.api.kis.client import KisApiClient
from src.api.kis.key_pool import (
    kis_key_id,
    load_kis_env,
    read_token_issued_date,
    resolve_host_issued_credentials,
    token_cache_path,
)
from src.data.session_calendar import SessionDay, SessionKind, resolve_session_day
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

_MSG_CD_RE = re.compile(r"EGW\d+")


def _warmup_failure_reason(exc: BaseException) -> str:
    """Summarize a slot failure without credentials (exception type + vendor msg_cd)."""
    name = type(exc).__name__
    match = _MSG_CD_RE.search(str(exc))
    return f"{name}:{match.group(0)}" if match is not None else name


def should_skip_warmup(today: date, *, session_day_fn: Callable[[date], SessionDay] | None = None) -> bool:
    """Return True only when the verified calendar declares the date CLOSED.

    Token issuance is cheap and consumers break without it, so only a verified
    closure skips; UNKNOWN (calendar not extended) and SHIFTED dates still warm up.
    """
    resolver = session_day_fn if session_day_fn is not None else resolve_session_day
    return resolver(today).kind is SessionKind.CLOSED


async def warmup_host_tokens(
    session: aiohttp.ClientSession, env: Mapping[str, str], *, today: str | None = None
) -> dict[str, bool]:
    """Issue every declared host-issued KIS token once per day, isolating slot failures.

    Consumers on this host (kca jobs and the krx-alpha collector) only read the
    shared cache, so one slot's vendor error must not deprive the remaining
    slots of their daily token. Every slot is attempted; failures are raised
    together after the loop so systemd retries the unit and only missing slots
    are re-issued (the same-day guard skips slots already issued today).

    Args:
        session: HTTP session.
        env: KIS credential mapping.
        today: KST date (YYYY-MM-DD); None uses the current KST date.

    Returns:
        Slot name -> True when issued in this run, False on a same-day cache hit.

    Raises:
        RuntimeError: At least one slot failed to issue or its cache was not
            refreshed to today; the message lists every failed slot with its
            key_id and exception type, never credentials.
    """
    day = today or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    results: dict[str, bool] = {}
    failures: list[str] = []
    for cred in resolve_host_issued_credentials(env):
        key_id = kis_key_id(cred.app_key)
        try:
            token_file = token_cache_path(cred.app_key, settings.KIS_TOKEN_CACHE_DIR)
            client = KisApiClient(  # type: ignore[no-untyped-call]
                app_key=cred.app_key,
                app_secret=cred.app_secret,
                hts_id=cred.hts_id,
                token_file=str(token_file),
            )
            issued = await client.issue_daily_token(session)
            results[cred.slot] = issued
            logger.info(
                "[SYS] stage=kis_token_warmup slot=%s key_id=%s issued=%s",
                cred.slot,
                key_id,
                issued,
            )
            cached = read_token_issued_date(token_file)
            if cached != day:
                raise RuntimeError(
                    f"token cache not refreshed slot={cred.slot} key_id={key_id} cached={cached} expected={day}"
                )
        except (RuntimeError, aiohttp.ClientError, TimeoutError, OSError) as exc:
            reason = _warmup_failure_reason(exc)
            logger.error(
                "[SYS] stage=kis_token_warmup slot=%s key_id=%s status=FAILED reason=%s",
                cred.slot,
                key_id,
                reason,
            )
            results.pop(cred.slot, None)
            failures.append(f"{cred.slot}(key_id={key_id}, reason={reason})")
    if failures:
        raise RuntimeError(f"kis_token_warmup failed slots: {'; '.join(failures)}")
    return results


def main() -> None:
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    if should_skip_warmup(today):
        logger.info("[SYS] stage=kis_token_warmup status=SKIP reason=non_trading_day date=%s", today.isoformat())
        return
    env = load_kis_env(Path(settings.BASE_DIR) / ".env")

    async def _run() -> dict[str, bool]:
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await warmup_host_tokens(session, env)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
