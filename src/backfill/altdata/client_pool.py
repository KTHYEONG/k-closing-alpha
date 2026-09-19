"""종목별 패널 다중 키 팬아웃 헬퍼."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import aiohttp

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.api.kis.key_pool import token_cache_path
from src.backfill.altdata.config import AltDataFetchConfig


async def fan_out_symbol_calls(
    cfg: AltDataFetchConfig,
    symbols: Sequence[str],
    call: Callable[[KisApiClient, aiohttp.ClientSession, str], Awaitable[dict[str, Any]]],
    on_error: Callable[[str, Exception], dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """종목 목록을 사용 가능한 KIS 키들에 라운드로빈으로 분산해 호출한다.

    Args:
        cfg: 추가 키 목록을 포함한 수집 설정.
        symbols: 조회할 종목 코드 목록.
        call: 클라이언트/세션/종목코드로 원시 응답을 반환하는 콜백.
        on_error: 종목코드와 예외로 폴백 응답을 반환하는 콜백.

    Returns:
        종목코드와 응답 dict의 튜플 리스트.
    """
    if len(symbols) == 0:
        return []
    clients = [KisApiClient(**kis_data_client_kwargs())]
    for app_key, app_secret, hts_id in cfg.extra_client_kwargs:
        clients.append(
            KisApiClient(
                app_key,
                app_secret,
                "",
                hts_id,
                token_file=str(token_cache_path(app_key, settings.KIS_TOKEN_CACHE_DIR)),
            )
        )
    count = len(clients)
    async with contextlib.AsyncExitStack() as stack:
        sessions = [await stack.enter_async_context(client.create_session()) for client in clients]
        for client, session in zip(clients, sessions):
            await client.ensure_token(session)
        semaphores = [asyncio.Semaphore(10) for _ in clients]

        async def _one(index: int, code: str) -> tuple[str, dict[str, Any]]:
            async with semaphores[index]:
                try:
                    res = await call(clients[index], sessions[index], code)
                except Exception as exc:
                    return code, on_error(code, exc)
                return code, res

        tasks = [_one(index, code) for index in range(count) for code in symbols[index::count]]
        return list(await asyncio.gather(*tasks))
