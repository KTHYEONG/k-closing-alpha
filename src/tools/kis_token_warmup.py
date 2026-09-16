"""이 호스트가 하루 1회 발급 책임을 지는 KIS 토큰 전부를 예열한다.

발급 대상은 key_pool의 선언(호스트 배정 데이터 슬롯 + HOST_ISSUED_KEY_SPECS)이
단일 원천이며, 다른 잡과 타 프로젝트 컨테이너는 캐시를 읽기만 한다. 발급 호출
성공과 캐시 반영은 별개이므로 기록까지 확인해야 소비자가 실제로 토큰을 읽는다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime
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

logger = logging.getLogger(__name__)


async def warmup_host_tokens(
    session: aiohttp.ClientSession, env: Mapping[str, str], *, today: str | None = None
) -> dict[str, bool]:
    """선언된 호스트 발급 키 전부를 당일 1회 발급하고 캐시 반영까지 검증한다.

    Args:
        session: HTTP 세션.
        env: KIS 자격증명 매핑.
        today: 기준일(YYYY-MM-DD, KST). None이면 현재 KST 날짜.

    Returns:
        슬롯명 -> 이번 실행에서 실제로 발급했는지 여부(당일 캐시 적중이면 False).

    Raises:
        RuntimeError: 발급 호출은 성공했으나 토큰 캐시가 당일자로 갱신되지 않은 경우.
            읽기전용 마운트나 권한 오류로 소비자가 토큰을 읽지 못하는 상태를 침묵시키지 않는다.
    """
    day = today or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    results: dict[str, bool] = {}
    for cred in resolve_host_issued_credentials(env):
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
            kis_key_id(cred.app_key),
            issued,
        )
        cached = read_token_issued_date(token_file)
        if cached != day:
            raise RuntimeError(
                f"token cache not refreshed slot={cred.slot} key_id={kis_key_id(cred.app_key)} cached={cached} expected={day}"
            )
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    env = load_kis_env(Path(settings.BASE_DIR) / ".env")

    async def _run() -> dict[str, bool]:
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await warmup_host_tokens(session, env)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
