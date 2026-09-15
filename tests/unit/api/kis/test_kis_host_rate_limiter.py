"""Auto-generated from contract: kis_key_pool."""

from __future__ import annotations

def test_host_paced_rate_limiter_spaces_slots_across_instances(tmp_path) -> None:
    import asyncio

    import pytest

    from src.api.kis.rate_limit import HostPacedRateLimiter

    now = {"t": 1000.0}
    sleeps: list[float] = []

    async def _sleep(sec: float) -> None:
        sleeps.append(sec)

    state = tmp_path / "tps.state"

    async def _run() -> None:
        a = HostPacedRateLimiter(state, max_rate=4.0, clock=lambda: now["t"], sleep=_sleep)
        b = HostPacedRateLimiter(state, max_rate=4.0, clock=lambda: now["t"], sleep=_sleep)
        # When: 같은 순간 3건
        await a.acquire()
        await b.acquire()
        await a.acquire()
        # Then: 0, 0.25, 0.5 지연 (0은 sleep 호출 없음)
        assert sleeps == [pytest.approx(0.25), pytest.approx(0.5)]
        # And: 충분히 쉰 뒤에는 대기 없음
        now["t"] = 1010.0
        await b.acquire()
        assert len(sleeps) == 2

    asyncio.run(_run())


def test_host_paced_rate_limiter_recovers_from_corrupt_state(tmp_path, caplog) -> None:
    import asyncio
    import logging

    import pytest

    from src.api.kis.rate_limit import HostPacedRateLimiter

    state = tmp_path / "tps.state"
    state.write_text("garbage", encoding="utf-8")
    sleeps: list[float] = []

    async def _sleep(sec: float) -> None:
        sleeps.append(sec)

    limiter = HostPacedRateLimiter(state, max_rate=4.0, clock=lambda: 1000.0, sleep=_sleep)

    with caplog.at_level(logging.WARNING, logger="src.api.kis.rate_limit"):
        asyncio.run(limiter.acquire())

    assert sleeps == []
    assert float(state.read_text(encoding="ascii")) == pytest.approx(1000.25)
    assert any("status=STATE_RESET" in r.getMessage() for r in caplog.records)


def test_host_paced_rate_limiter_rejects_nonpositive_rate(tmp_path) -> None:
    import pytest

    from src.api.kis.rate_limit import HostPacedRateLimiter

    with pytest.raises(ValueError, match="positive"):
        HostPacedRateLimiter(tmp_path / "a.state", max_rate=0.0)
    with pytest.raises(ValueError, match="positive"):
        HostPacedRateLimiter(tmp_path / "b.state", max_rate=18.0, time_period=0.0)


def test_get_host_rate_limiter_caches_per_path_and_client_uses_it(tmp_path, monkeypatch) -> None:
    from src import settings
    from src.api.kis.client import KIS_REST_TPS_PER_APP_KEY, KisApiClient
    from src.api.kis.key_pool import kis_key_id
    from src.api.kis.rate_limit import HostPacedRateLimiter, get_host_rate_limiter

    a = get_host_rate_limiter(tmp_path / "x.state", 18.0)

    assert get_host_rate_limiter(tmp_path / "x.state", 18.0) is a
    assert get_host_rate_limiter(tmp_path / "y.state", 18.0) is not a
    assert isinstance(a, HostPacedRateLimiter)

    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)
    client = KisApiClient(app_key="k-host", app_secret="s")
    expected = get_host_rate_limiter(tmp_path / f"tps_{kis_key_id('k-host')}.state", KIS_REST_TPS_PER_APP_KEY)
    assert client.rate_limiter is expected

