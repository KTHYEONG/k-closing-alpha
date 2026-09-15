"""KIS data key token warmup (host data slots, read-only)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path

import aiohttp

from src import settings
from src.api.kis.client import KisApiClient
from src.api.kis.key_pool import (
    kis_key_id,
    load_kis_env,
    resolve_host_data_credentials,
    token_cache_path,
)

logger = logging.getLogger(__name__)


async def warmup_host_tokens(session: aiohttp.ClientSession, env: Mapping[str, str]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for cred in resolve_host_data_credentials(env):
        if not cred.app_key.strip() or not cred.app_secret.strip():
            raise ValueError(f"missing credentials for slot {cred.slot}")
        client = KisApiClient(  # type: ignore[no-untyped-call]
            app_key=cred.app_key,
            app_secret=cred.app_secret,
            hts_id=cred.hts_id,
            token_file=str(token_cache_path(cred.app_key, settings.KIS_TOKEN_CACHE_DIR)),
        )
        issued = await client.issue_daily_token(session)
        results[cred.slot] = issued
        logger.info(
            "[SYS] stage=kis_token_warmup slot=%s key_id=%s issued=%s",
            cred.slot,
            kis_key_id(cred.app_key),
            issued,
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
