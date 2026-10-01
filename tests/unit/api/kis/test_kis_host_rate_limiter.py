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
    import src.api.kis.rate_limit as rl_mod
    from src import settings
    from src.api.kis.client import KIS_REST_TPS_PER_APP_KEY, KisApiClient
    from src.api.kis.key_pool import kis_key_id
    from src.api.kis.rate_limit import HostPacedRateLimiter, get_host_rate_limiter

    monkeypatch.setattr(settings, "BROKER_ADMISSION_REQUIRE_SHARED", "never")
    monkeypatch.setattr(rl_mod, "_ADMISSION_DIR_VERIFIED", False)

    a = get_host_rate_limiter(tmp_path / "x.state", 18.0)

    assert get_host_rate_limiter(tmp_path / "x.state", 18.0) is a
    assert get_host_rate_limiter(tmp_path / "y.state", 18.0) is not a
    assert isinstance(a, HostPacedRateLimiter)

    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)
    client = KisApiClient(app_key="k-host", app_secret="s")
    expected = get_host_rate_limiter(tmp_path / f"tps_{kis_key_id('k-host')}.state", KIS_REST_TPS_PER_APP_KEY)
    assert client.rate_limiter is expected



def _reset_admission(monkeypatch, tmp_path, **overrides):
    import src.api.kis.rate_limit as rl_mod
    from src import settings

    monkeypatch.setattr(settings, "BROKER_ADMISSION_DIR", tmp_path)
    monkeypatch.setattr(settings, "BROKER_ADMISSION_REQUIRE_SHARED", overrides.get("require", "never"))
    monkeypatch.setattr(rl_mod, "_ADMISSION_DIR_VERIFIED", False)
    rl_mod._HOST_RATE_LIMITERS.clear()
    return rl_mod


def test_bulk_lead_never_books_far_ahead(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.kis.rate_limit import HostPacedRateLimiter

    _reset_admission(monkeypatch, tmp_path)
    now = {"t": 1000.0}
    state = tmp_path / "bucket.state"
    state.write_text("1005.0", encoding="ascii")

    async def _sleep(sec: float) -> None:
        now["t"] += sec

    limiter = HostPacedRateLimiter(state, max_rate=4.0, max_lead_seconds=0.25, clock=lambda: now["t"], sleep=_sleep)
    asyncio.run(limiter.acquire())
    assert float(state.read_text(encoding="ascii")) == 1005.0 + 0.25
    assert now["t"] >= 1005.0 - 0.25


def test_critical_books_past_bulk_backlog(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.kis.rate_limit import HostPacedRateLimiter

    _reset_admission(monkeypatch, tmp_path)
    now = {"t": 2000.0}
    state = tmp_path / "bucket.state"

    async def _tick(sec: float) -> None:
        now["t"] += sec

    bulk = HostPacedRateLimiter(state, max_rate=4.0, max_lead_seconds=0.25, clock=lambda: now["t"], sleep=_tick)
    critical = HostPacedRateLimiter(state, max_rate=4.0, max_lead_seconds=None, clock=lambda: now["t"], sleep=_tick)

    async def _run() -> None:
        for _ in range(100):
            await bulk.acquire()
        arrival = now["t"]
        slot_delay = await _book_delay(critical)
        assert slot_delay <= 0.25 + critical._interval + 1e-9
        _ = arrival

    async def _book_delay(lim) -> float:
        before = now["t"]
        await lim.acquire()
        content = float(state.read_text(encoding="ascii"))
        return content - lim._interval - before

    asyncio.run(_run())


def test_interval_conservation(tmp_path, monkeypatch) -> None:
    import asyncio

    import pytest

    from src.api.kis.rate_limit import HostPacedRateLimiter

    _reset_admission(monkeypatch, tmp_path)
    state = tmp_path / "bucket.state"

    async def _noop(_: float) -> None:
        return None

    async def _run() -> list[float]:
        a = HostPacedRateLimiter(state, max_rate=4.0, clock=lambda: 3000.0, sleep=_noop)
        b = HostPacedRateLimiter(state, max_rate=4.0, clock=lambda: 3000.0, sleep=_noop)
        slots: list[float] = []
        for lim in (a, b, a, b, a, b):
            await lim.acquire()
            slots.append(float(state.read_text(encoding="ascii")) - 0.25)
        return slots

    slots = asyncio.run(_run())
    from itertools import pairwise

    diffs = [b - a for a, b in pairwise(slots)]
    assert all(d == pytest.approx(0.25) for d in diffs)
    assert slots == sorted(slots)


def test_single_poller_per_process(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.kis.rate_limit import HostPacedRateLimiter

    _reset_admission(monkeypatch, tmp_path)
    now = {"t": 4000.0}
    state = tmp_path / "bucket.state"
    reads = {"n": 0}
    import os as _os

    real_read = _os.read

    def _counting(fd, n):
        reads["n"] += 1
        return real_read(fd, n)

    monkeypatch.setattr(_os, "read", _counting)

    async def _sleep(sec: float) -> None:
        now["t"] += sec
        await asyncio.sleep(0)

    async def _run() -> None:
        lim = HostPacedRateLimiter(state, max_rate=4.0, max_lead_seconds=5.0, clock=lambda: now["t"], sleep=_sleep)
        await asyncio.gather(*[lim.acquire() for _ in range(50)])

    asyncio.run(_run())
    assert reads["n"] <= 2 * 50 + 10
    assert reads["n"] >= 50


def test_missing_marker_in_container_refuses(tmp_path, monkeypatch) -> None:
    import pytest
    import src.api.kis.rate_limit as rl_mod
    from src import settings
    from src.api.kis.rate_limit import AdmissionDirNotSharedError, resolve_admission_dir

    monkeypatch.setattr(settings, "BROKER_ADMISSION_DIR", tmp_path)
    monkeypatch.setattr(settings, "BROKER_ADMISSION_REQUIRE_SHARED", "always")
    monkeypatch.setattr(rl_mod, "_ADMISSION_DIR_VERIFIED", False)
    with pytest.raises(AdmissionDirNotSharedError, match="host-shared"):
        resolve_admission_dir()
    (tmp_path / ".host-admission").write_text("", encoding="utf-8")
    assert resolve_admission_dir() == tmp_path


def test_workstation_without_marker_passes(tmp_path, monkeypatch) -> None:
    import src.api.kis.rate_limit as rl_mod
    from src import settings
    from src.api.kis.rate_limit import resolve_admission_dir

    monkeypatch.setattr(settings, "BROKER_ADMISSION_DIR", tmp_path)
    monkeypatch.setattr(settings, "BROKER_ADMISSION_REQUIRE_SHARED", "auto")
    monkeypatch.setattr(rl_mod, "_in_container", lambda: False)
    monkeypatch.setattr(rl_mod, "_ADMISSION_DIR_VERIFIED", False)
    assert resolve_admission_dir() == tmp_path


def test_state_path_deterministic_and_rejects(tmp_path, monkeypatch) -> None:
    import hashlib

    import pytest

    from src.api.kis.rate_limit import host_admission_state_path

    _reset_admission(monkeypatch, tmp_path)
    p = host_admission_state_path("kiwoom", "cred-9", "ka10079")
    digest = hashlib.sha256(b"cred-9").hexdigest()[:12]
    assert p.name == f"admission_kiwoom_{digest}_ka10079.state"
    q = host_admission_state_path("kiwoom", "cred-9")
    assert q.name == f"admission_kiwoom_{digest}.state"
    with pytest.raises(ValueError, match="scope"):
        host_admission_state_path("kiwoom", "cred-9", "../x")
    with pytest.raises(ValueError, match="vendor"):
        host_admission_state_path("../v", "cred-9")
    with pytest.raises(ValueError, match="credential"):
        host_admission_state_path("kiwoom", "")


def test_settings_rejects_inverted_leads() -> None:
    import pytest

    from src.config.admission import AdmissionSettings

    with pytest.raises(Exception, match="bulk lead"):
        AdmissionSettings(_env_file=None, BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS=2.0, BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS=1.0)


def test_max_lead_and_singleton_class(tmp_path, monkeypatch) -> None:
    import src.api.kis.rate_limit as rl_mod
    from src import settings
    from src.api.kis.rate_limit import AdmissionClass, get_host_rate_limiter, max_lead_for

    _reset_admission(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS", 1.0)
    monkeypatch.setattr(settings, "BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS", 0.25)
    assert max_lead_for(AdmissionClass.CRITICAL) is None
    assert max_lead_for(AdmissionClass.STANDARD) == 1.0
    assert max_lead_for(AdmissionClass.BULK) == 0.25
    a = get_host_rate_limiter(tmp_path / "s.state", 4.0, admission_class=AdmissionClass.BULK)
    b = get_host_rate_limiter(tmp_path / "s.state", 4.0, admission_class=AdmissionClass.BULK)
    c = get_host_rate_limiter(tmp_path / "s.state", 4.0, admission_class=AdmissionClass.CRITICAL)
    assert a is b
    assert c is not a
    d = get_host_rate_limiter(tmp_path / "s.state", 4.0)
    assert d is a
    import pytest

    from src.api.kis.rate_limit import HostPacedRateLimiter as _H

    with pytest.raises(ValueError, match="positive"):
        _H(tmp_path / "z.state", 4.0, max_lead_seconds=0.0)
