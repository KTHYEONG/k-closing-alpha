from __future__ import annotations

import asyncio

from src.sync.fetcher_program import get_program_history_async


class _Client:
    base_url = "https://example.test"

    async def _handle_request(self, *args, **kwargs):
        return {"rt_cd": "0", "output": [{"stck_bsop_date": "20200102", "whol_smtn_ntby_tr_pbmn": "1234"}]}

    def _get_headers(self, tr_id):
        return {}


class _Session:
    def get(self, *args, **kwargs):
        raise AssertionError("fake request method should not be invoked directly")


def test_program_async_uses_rate_slot() -> None:
    calls = 0

    async def slot() -> None:
        nonlocal calls
        calls += 1

    out = asyncio.run(
        get_program_history_async(
            _Session(), _Client(), "005930", "20200102", "20200102", request_slot=slot
        )
    )
    assert calls == 1
    assert out == {"20200102": 1234.0}
