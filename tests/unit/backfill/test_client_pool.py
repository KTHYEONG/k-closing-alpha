from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd

from src.backfill.altdata.client_pool import fan_out_symbol_calls
from src.backfill.altdata.config import AltDataFetchConfig


def _cfg(**overrides) -> AltDataFetchConfig:
    base = {
        "start": pd.Timestamp("2024-01-02"),
        "end": pd.Timestamp("2024-01-03"),
        "out_dir": Path("x"),
    }
    base.update(overrides)
    return AltDataFetchConfig(**base)


class _FakeClient:
    instances: list[_FakeClient] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.app_key = kwargs.get("app_key", args[0] if args else "")
        self.token_file = kwargs.get("token_file", "")
        self.symbols: list[str] = []
        _FakeClient.instances.append(self)

    def create_session(self):
        session = AsyncMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        mock = AsyncMock()
        mock.__aenter__ = cm.__aenter__
        mock.__aexit__ = cm.__aexit__
        self._session = session
        return mock

    async def ensure_token(self, session) -> str:
        return "tok"


def _patch_pool(monkeypatch, extra_kwargs: dict | None = None):
    from src.backfill.altdata import client_pool

    _FakeClient.instances.clear()
    base_kwargs = {"app_key": "k0", "app_secret": "s0", "account_id": "", "hts_id": "h0", "token_file": "t0"}
    if extra_kwargs:
        base_kwargs.update(extra_kwargs)
    monkeypatch.setattr(client_pool, "kis_data_client_kwargs", lambda: dict(base_kwargs))
    monkeypatch.setattr(client_pool, "KisApiClient", _FakeClient)


def test_fan_out_uses_single_client_without_extra_keys(monkeypatch) -> None:
    import asyncio

    _patch_pool(monkeypatch)
    cfg = _cfg()
    seen: list[tuple[str, str]] = []

    async def _call(client, session, code):
        seen.append((client.app_key, code))
        return {"rt_cd": "0", "code": code}

    out = asyncio.run(fan_out_symbol_calls(cfg, ["a", "b", "c", "d", "e"], _call, lambda code, exc: {"rt_cd": "9"}))
    assert len(out) == 5
    assert len(_FakeClient.instances) == 1
    assert sorted(c for _, c in seen) == ["a", "b", "c", "d", "e"]


def test_fan_out_distributes_symbols_across_keys(monkeypatch) -> None:
    import asyncio

    _patch_pool(monkeypatch)
    cfg = _cfg(extra_client_kwargs=(("k1", "s1", "h1"), ("k2", "s2", "h2")))
    per_client: dict[str, list[str]] = {}

    async def _call(client, session, code):
        per_client.setdefault(client.app_key, []).append(code)
        return {"rt_cd": "0"}

    symbols = ["s0", "s1", "s2", "s3", "s4", "s5"]
    out = asyncio.run(fan_out_symbol_calls(cfg, symbols, _call, lambda code, exc: {"rt_cd": "9"}))
    assert len(out) == 6
    assert len(_FakeClient.instances) == 3
    assert sorted(per_client["k0"]) == sorted(symbols[0::3])
    assert sorted(per_client["k1"]) == sorted(symbols[1::3])
    assert sorted(per_client["k2"]) == sorted(symbols[2::3])
    assert all(len(v) == 2 for v in per_client.values())


def test_fan_out_returns_empty_without_creating_clients(monkeypatch) -> None:
    import asyncio

    from src.backfill.altdata import client_pool

    _patch_pool(monkeypatch)
    with patch.object(_FakeClient, "create_session") as mock_create:
        cfg = _cfg(extra_client_kwargs=(("k1", "s1", "h1"),))
        out = asyncio.run(fan_out_symbol_calls(cfg, [], lambda c, s, code: {"rt_cd": "0"}, lambda code, exc: {}))
    assert out == []
    mock_create.assert_not_called()
    assert _FakeClient.instances == []


def test_fan_out_isolates_individual_failures(monkeypatch) -> None:
    import asyncio

    _patch_pool(monkeypatch)
    cfg = _cfg()

    async def _call(client, session, code):
        if code == "bad":
            raise RuntimeError("boom")
        return {"rt_cd": "0", "code": code}

    def _on_error(code, exc):
        return {"rt_cd": "9", "code": code}

    out = asyncio.run(fan_out_symbol_calls(cfg, ["ok1", "bad", "ok2"], _call, _on_error))
    by_code = dict(out)
    assert by_code["bad"] == {"rt_cd": "9", "code": "bad"}
    assert by_code["ok1"] == {"rt_cd": "0", "code": "ok1"}
    assert by_code["ok2"] == {"rt_cd": "0", "code": "ok2"}


def test_fan_out_uses_token_cache_path_for_extra_keys(monkeypatch) -> None:
    import asyncio

    from src import settings
    from src.api.kis.key_pool import token_cache_path

    _patch_pool(monkeypatch)
    cfg = _cfg(extra_client_kwargs=(("extra-key", "sec", "hts"),))

    async def _call(client, session, code):
        return {"rt_cd": "0"}

    asyncio.run(fan_out_symbol_calls(cfg, ["a"], _call, lambda code, exc: {}))
    assert len(_FakeClient.instances) == 2
    assert _FakeClient.instances[1].token_file == str(token_cache_path("extra-key", settings.KIS_TOKEN_CACHE_DIR))
