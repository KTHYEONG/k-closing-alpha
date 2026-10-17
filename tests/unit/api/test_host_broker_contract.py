"""Host broker contract v1 invariant guards (mirrored across krx-alpha / kca / k-stock-engine)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.api.host_broker_contract import (
    AsyncHostPacer,
    AsyncTokenStore,
    CONTRACT_VERSION,
    DEFAULT_RATES,
    HostBrokerDirNotSharedError,
    HostPacer,
    IssuedToken,
    TokenIssueThrottled,
    TokenRecord,
    TokenStore,
    TokenStoreLockTimeout,
    admission_state_path,
    credential_digest,
    issue_guard_path,
    rate_for,
    resolve_host_broker_dir,
    token_record_path,
    validate_vendor,
)

_NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def _async_store(path: Path, vendor: str = "toss", **kw) -> AsyncTokenStore:
    args: dict = {
        "vendor": vendor,
        "lock_timeout_seconds": 5.0,
        "expiry_margin_seconds": 60.0,
        "clock": lambda: _NOW,
        "issue_throttle_policy": "raise",
    }
    args.update(kw)
    return AsyncTokenStore(path, **args)


def test_token_path_naming(tmp_path: Path) -> None:
    p = token_record_path(tmp_path, "dbsec", "K")
    assert p.name == f"token_dbsec_{hashlib.sha256(b'K').hexdigest()[:12]}.json"


def test_rejects_alias_vendor(tmp_path: Path) -> None:
    for vendor in ("db", "DBSEC", "kis", "../x"):
        with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
            token_record_path(tmp_path, vendor, "K")
        with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
            admission_state_path(tmp_path, vendor, "K")
        with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
            issue_guard_path(tmp_path, vendor, "K")
    with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
        token_record_path(tmp_path, "toss", "")
    with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
        validate_vendor("db")
    assert validate_vendor("toss") == "toss"
    assert credential_digest("K") == hashlib.sha256(b"K").hexdigest()[:12]
    with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
        credential_digest("")


def test_scope_case_preserved(tmp_path: Path) -> None:
    p = admission_state_path(tmp_path, "toss", "k", "STOCK_TRADING_TREND")
    assert p.name.endswith("_STOCK_TRADING_TREND.state")
    with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
        admission_state_path(tmp_path, "toss", "k", "*")
    with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
        admission_state_path(tmp_path, "toss", "k", "../x")
    with pytest.raises(ValueError, match=r"invalid|credential|undeclared"):
        admission_state_path(tmp_path, "toss", "")


def test_schema_strict(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    aware = _NOW.isoformat()
    naive = "2026-10-01T12:00:00"
    cases: list[object] = [
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "expires_at": None, "generation": 1, "extra": 1},
        {"schema_version": 2, "access_token": "a", "issued_at": aware, "expires_at": None, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": naive, "expires_at": None, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "expires_at": None, "generation": True},
    ]
    for i, payload in enumerate(cases):
        path = tmp_path / f"bad{i}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        store = _async_store(path)
        with caplog.at_level(logging.WARNING):
            assert store.read() is None
    assert any("SCHEMA_INVALID" in r.getMessage() for r in caplog.records)


def test_single_issuance_under_contention(tmp_path: Path) -> None:
    calls: list[int] = []

    async def _issue() -> IssuedToken:
        calls.append(1)
        await asyncio.sleep(0.05)
        return IssuedToken(access_token="tok-new", expires_in_seconds=3600.0)

    a = _async_store(tmp_path / "t.json")
    b = _async_store(tmp_path / "t.json")

    async def _run():
        return await asyncio.gather(a.get_or_issue(_issue), b.get_or_issue(_issue))

    ra, rb = asyncio.run(_run())
    assert len(calls) == 1
    assert ra.access_token == rb.access_token == "tok-new"
    assert ra.generation == rb.generation == 1


def test_replace_rejected_no_pingpong(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    payload = {
        "schema_version": 1,
        "access_token": "B",
        "issued_at": _NOW.isoformat(),
        "expires_at": (_NOW + timedelta(seconds=3600)).isoformat(),
        "generation": 2,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = _async_store(path)

    async def _issue() -> IssuedToken:
        raise AssertionError("must not issue")

    rec = asyncio.run(store.replace_rejected("A", _issue))
    assert rec.access_token == "B"
    assert rec.generation == 2


def test_unknown_expiry_bounded(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    old = _NOW - timedelta(seconds=43201)
    payload = {
        "schema_version": 1,
        "access_token": "old",
        "issued_at": old.isoformat(),
        "expires_at": None,
        "generation": 1,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    now = {"t": _NOW}
    store = _async_store(path, clock=lambda: now["t"])
    calls = {"n": 0}

    async def _issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken(access_token="fresh", expires_in_seconds=3600.0)

    rec = asyncio.run(store.get_or_issue(_issue))
    assert calls["n"] == 1
    assert rec.generation == 2


def _dbsec_store(
    tmp_path: Path, key: str, now: dict, sleeps: list[float], policy: str = "wait", lock_timeout: float = 120.0
) -> AsyncTokenStore:
    async def _sleep(s: float) -> None:
        sleeps.append(s)
        now["t"] = datetime.fromtimestamp(now["t"].timestamp() + s, tz=UTC)

    token_path = token_record_path(tmp_path, "dbsec", key)
    return AsyncTokenStore(
        token_path,
        vendor="dbsec",
        lock_timeout_seconds=lock_timeout,
        expiry_margin_seconds=600.0,
        clock=lambda: now["t"],
        issue_throttle_policy=policy,  # type: ignore[arg-type]
        sleep=_sleep,
    )


def test_dbsec_min_issue_interval_wait(tmp_path: Path) -> None:
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    now = {"t": base}
    guard = issue_guard_path(tmp_path, "dbsec", "K")
    guard.write_text(repr((base - timedelta(seconds=10)).timestamp()), encoding="ascii")
    sleeps: list[float] = []
    store = _dbsec_store(tmp_path, "K", now, sleeps, policy="wait")
    order: list[str] = []

    async def _issue() -> IssuedToken:
        order.append("issue")
        assert guard.exists()
        return IssuedToken(access_token="tok", expires_in_seconds=86400.0)

    orig_sleep = sleeps.append

    async def _rec_sleep(s: float) -> None:
        order.append(f"sleep:{s:.0f}")
        await _dbsec_sleep(now, s)
        orig_sleep(s)

    async def _dbsec_sleep(now: dict, s: float) -> None:
        now["t"] = datetime.fromtimestamp(now["t"].timestamp() + s, tz=UTC)

    store._sleep = _rec_sleep  # type: ignore[method-assign]
    rec = asyncio.run(store.get_or_issue(_issue))
    assert rec.access_token == "tok"
    assert any(o.startswith("sleep:51") for o in order)
    assert order.index(next(o for o in order if o.startswith("sleep"))) < order.index("issue")


def test_dbsec_min_issue_interval_wait_exceeds_lock_bound(tmp_path: Path) -> None:
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    now = {"t": base}
    guard = issue_guard_path(tmp_path, "dbsec", "K")
    guard.write_text(repr((base - timedelta(seconds=10)).timestamp()), encoding="ascii")
    before_token = token_record_path(tmp_path, "dbsec", "K")
    before_guard_text = guard.read_text(encoding="ascii")
    sleeps: list[float] = []
    store = _dbsec_store(tmp_path, "K", now, sleeps, policy="wait", lock_timeout=30.0)

    async def _issue() -> IssuedToken:
        raise AssertionError("must not issue")

    with pytest.raises(TokenIssueThrottled):
        asyncio.run(store.get_or_issue(_issue))
    assert sleeps == []
    assert not before_token.exists()
    assert guard.read_text(encoding="ascii") == before_guard_text


def test_dbsec_min_issue_interval_raise(tmp_path: Path) -> None:
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    now = {"t": base}
    guard = issue_guard_path(tmp_path, "dbsec", "K")
    guard.write_text(repr((base - timedelta(seconds=10)).timestamp()), encoding="ascii")
    before = guard.read_text(encoding="ascii")
    sleeps: list[float] = []
    store = _dbsec_store(tmp_path, "K", now, sleeps, policy="raise")

    async def _issue() -> IssuedToken:
        raise AssertionError("must not issue")

    with pytest.raises(TokenIssueThrottled):
        asyncio.run(store.get_or_issue(_issue))
    assert sleeps == []
    assert not token_record_path(tmp_path, "dbsec", "K").exists()
    assert guard.read_text(encoding="ascii") == before


def test_attempt_recorded_on_issue_failure(tmp_path: Path) -> None:
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    now = {"t": base}
    sleeps: list[float] = []
    store = _dbsec_store(tmp_path, "K", now, sleeps, policy="raise")

    async def _boom() -> IssuedToken:
        raise RuntimeError("vendor down")

    with pytest.raises(RuntimeError, match="vendor down"):
        asyncio.run(store.get_or_issue(_boom))
    guard = issue_guard_path(tmp_path, "dbsec", "K")
    assert float(guard.read_text(encoding="ascii")) == pytest.approx(base.timestamp())
    assert not token_record_path(tmp_path, "dbsec", "K").exists()


def test_non_dbsec_has_no_guard_file(tmp_path: Path) -> None:
    store = _async_store(tmp_path / "token_toss_x.json", vendor="toss")

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="t", expires_in_seconds=3600.0)

    asyncio.run(store.get_or_issue(_issue))
    assert list(tmp_path.glob("token_issue_toss_*.state")) == []


def test_sync_async_byte_identical(tmp_path: Path) -> None:
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    now_a = {"t": base}
    now_s = {"t": base}
    pa = tmp_path / "a" / "token_toss_k.json"
    ps = tmp_path / "s" / "token_toss_k.json"
    digest = credential_digest("k")
    pa.parent.mkdir(parents=True)
    ps.parent.mkdir(parents=True)
    # Use identical digests by writing to same filename under different dirs; guard derived per-dir.
    a = AsyncTokenStore(
        tmp_path / f"a/token_toss_{digest}.json",
        vendor="toss",
        lock_timeout_seconds=5.0,
        expiry_margin_seconds=60.0,
        clock=lambda: now_a["t"],
        issue_throttle_policy="raise",
        sleep=lambda s: asyncio.sleep(0),
    )
    s = TokenStore(
        tmp_path / f"s/token_toss_{digest}.json",
        vendor="toss",
        lock_timeout_seconds=5.0,
        expiry_margin_seconds=60.0,
        clock=lambda: now_s["t"],
        issue_throttle_policy="raise",
        sleep=lambda s: None,
    )

    async def _ai() -> IssuedToken:
        return IssuedToken(access_token="tok", expires_in_seconds=3600.0)

    asyncio.run(a.get_or_issue(_ai))
    s.get_or_issue(lambda: IssuedToken(access_token="tok", expires_in_seconds=3600.0))
    assert (tmp_path / f"a/token_toss_{digest}.json").read_bytes() == (
        tmp_path / f"s/token_toss_{digest}.json"
    ).read_bytes()

    fa = tmp_path / "a.slot"
    fs = tmp_path / "s.slot"
    fa.write_text("", encoding="ascii")
    fs.write_text("", encoding="ascii")
    pa_lim = AsyncHostPacer(fa, 4.0, clock=lambda: 1000.0, sleep=lambda s: asyncio.sleep(0))
    ps_lim = HostPacer(fs, 4.0, clock=lambda: 1000.0, sleep=lambda s: None)
    asyncio.run(pa_lim.acquire())
    ps_lim.acquire()
    assert fa.read_bytes() == fs.read_bytes()
    assert pa is not None and ps is not None


def test_pacing_shared_slots(tmp_path: Path) -> None:
    state = tmp_path / "bucket.state"
    now = {"t": 1000.0}
    sleeps: list[float] = []

    async def _sleep(s: float) -> None:
        sleeps.append(s)

    async def _run() -> None:
        a = AsyncHostPacer(state, 20.0, clock=lambda: now["t"], sleep=_sleep)
        b = AsyncHostPacer(state, 20.0, clock=lambda: now["t"], sleep=_sleep)
        for lim in [a, b] * 20:
            await lim.acquire()

    asyncio.run(_run())
    assert len(sleeps) == 39
    content = float(state.read_text(encoding="ascii"))
    assert content == pytest.approx(1000.0 + 40 * 0.05)


def test_rate_table_lookup() -> None:
    assert rate_for("kiwoom", "ka10079") == 5.0
    assert rate_for("dbsec", None) == 20.0
    assert rate_for("dbsec", "chart") == 4.0
    assert rate_for("ls", None) == pytest.approx(1.0 / 1.05)
    assert rate_for("toss", "ORDER") == 10.0
    with pytest.raises(ValueError, match="undeclared rate"):
        rate_for("toss", "NOPE")
    with pytest.raises(ValueError, match="undeclared rate"):
        rate_for("dbsec", "chart-nope")
    assert len([k for k in DEFAULT_RATES if k[0] == "toss"]) == 17


def test_container_requires_marker(tmp_path: Path) -> None:
    with pytest.raises(HostBrokerDirNotSharedError, match="host-shared"):
        resolve_host_broker_dir(tmp_path, require_shared="auto", in_container=True)
    assert resolve_host_broker_dir(tmp_path, require_shared="auto", in_container=False) == tmp_path
    (tmp_path / ".host-admission").write_text("", encoding="utf-8")
    assert resolve_host_broker_dir(tmp_path, require_shared="auto", in_container=True) == tmp_path
    assert resolve_host_broker_dir(tmp_path, require_shared="never", in_container=True) == tmp_path
    assert CONTRACT_VERSION == 1


def test_vendored_copy_hash_pinned() -> None:
    from pathlib import Path as _P

    src = _P("src/api/host_broker_contract.py")
    lines = src.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# canonical: krx-alpha/deploy/shared/host_broker_contract.py contract=v1"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest == "21dacdbbcb55c87c2698d9c32a4a9d43935bc2759157d8e93ac40fb6c9eaf31b"
    assert isinstance(TokenRecord(access_token="x", issued_at=_NOW, expires_at=None, generation=1), TokenRecord)
    assert TokenStoreLockTimeout.__name__ == "TokenStoreLockTimeout"


def test_adapter_identity() -> None:
    import src.api.kis.rate_limit as rl_mod
    import src.api.shared_token as st_mod
    from src.api.dbsec.client import DbsecTokenThrottled
    from src.api.host_broker_contract import AsyncHostPacer as _AHP
    from src.api.host_broker_contract import AsyncTokenStore as _ATS
    from src.api.host_broker_contract import TokenIssueThrottled as _TIT

    assert rl_mod.HostPacedRateLimiter is _AHP
    assert st_mod.SharedTokenStore is _ATS
    assert issubclass(DbsecTokenThrottled, _TIT)


def test_store_init_rejects_bad_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lock_timeout"):
        _async_store(tmp_path / "t.json", lock_timeout_seconds=0.0)
    with pytest.raises(ValueError, match="expiry_margin"):
        _async_store(tmp_path / "t.json", expiry_margin_seconds=-1.0)
    with pytest.raises(ValueError, match="unknown_expiry"):
        _async_store(tmp_path / "t.json", unknown_expiry_max_age_seconds=-1.0)
    with pytest.raises(ValueError, match="vendor"):
        _async_store(tmp_path / "t.json", vendor="db")
    with pytest.raises(ValueError, match="issue_throttle_policy"):
        _async_store(tmp_path / "t.json", issue_throttle_policy="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        AsyncHostPacer(tmp_path / "s.state", 0.0)
    with pytest.raises(ValueError, match="positive"):
        HostPacer(tmp_path / "s.state", 1.0, max_lead_seconds=0.0)


def test_guard_open_failure_treated_as_no_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as _os

    import src.api.host_broker_contract as hbc

    monkeypatch.setattr(_os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    assert hbc._read_guard_value(tmp_path / "g.state") is None
    assert hbc._throttle_delay_seconds("toss", None, 1000.0) == 0.0
    assert hbc._throttle_delay_seconds("toss", tmp_path / "g.state", 1000.0) == 0.0


def test_sync_store_usable_and_lock_timeout(tmp_path: Path) -> None:
    import fcntl
    import os as _os

    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    digest = credential_digest("k")
    token_path = tmp_path / f"token_toss_{digest}.json"
    store = TokenStore(
        token_path,
        vendor="toss",
        lock_timeout_seconds=5.0,
        expiry_margin_seconds=60.0,
        clock=lambda: base,
        issue_throttle_policy="raise",
        sleep=lambda s: None,
    )
    rec = store.get_or_issue(lambda: IssuedToken(access_token="tok", expires_in_seconds=3600.0))
    assert rec.generation == 1
    calls = {"n": 0}

    def _never() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken(access_token="other", expires_in_seconds=3600.0)

    assert store.get_or_issue(_never).access_token == "tok"
    assert calls["n"] == 0
    assert (
        store.replace_rejected("tok", lambda: IssuedToken(access_token="tok2", expires_in_seconds=3600.0)).generation
        == 2
    )

    lock_path = Path(str(token_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "access_token": "expired",
                "issued_at": (base - timedelta(seconds=7200)).isoformat(),
                "expires_at": (base - timedelta(seconds=3600)).isoformat(),
                "generation": 9,
            }
        ),
        encoding="utf-8",
    )
    fd = _os.open(str(lock_path), _os.O_RDWR | _os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        contended = TokenStore(
            token_path,
            vendor="toss",
            lock_timeout_seconds=0.05,
            expiry_margin_seconds=60.0,
            clock=lambda: base,
            issue_throttle_policy="raise",
            sleep=lambda s: None,
        )
        with pytest.raises(TokenStoreLockTimeout, match="token lock"):
            contended.get_or_issue(lambda: IssuedToken(access_token="x", expires_in_seconds=1.0))
        with pytest.raises(TokenStoreLockTimeout, match="token lock"):
            contended.replace_rejected("tok", lambda: IssuedToken(access_token="x", expires_in_seconds=1.0))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        _os.close(fd)


def test_sync_dbsec_guard_raise_and_wait(tmp_path: Path) -> None:
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    now = {"t": base}
    key = "K"
    guard = issue_guard_path(tmp_path, "dbsec", key)
    guard.write_text(repr((base - timedelta(seconds=10)).timestamp()), encoding="ascii")
    before = guard.read_text(encoding="ascii")

    def _clock() -> datetime:
        return now["t"]

    raising = TokenStore(
        token_record_path(tmp_path, "dbsec", key),
        vendor="dbsec",
        lock_timeout_seconds=5.0,
        expiry_margin_seconds=600.0,
        clock=_clock,
        issue_throttle_policy="raise",
        sleep=lambda s: None,
    )
    with pytest.raises(TokenIssueThrottled):
        raising.get_or_issue(lambda: IssuedToken(access_token="x", expires_in_seconds=1.0))
    assert guard.read_text(encoding="ascii") == before

    sleeps: list[float] = []

    def _sleep(s: float) -> None:
        sleeps.append(s)
        now["t"] = datetime.fromtimestamp(now["t"].timestamp() + s, tz=UTC)

    waiting = TokenStore(
        token_record_path(tmp_path, "dbsec", key),
        vendor="dbsec",
        lock_timeout_seconds=120.0,
        expiry_margin_seconds=600.0,
        clock=_clock,
        issue_throttle_policy="wait",
        sleep=_sleep,
    )
    rec = waiting.get_or_issue(lambda: IssuedToken(access_token="tok", expires_in_seconds=86400.0))
    assert rec.access_token == "tok"
    assert sleeps and sleeps[0] == pytest.approx(51.0)

    cached = waiting.get_or_issue(lambda: IssuedToken(access_token="other", expires_in_seconds=1.0))
    assert cached.access_token == "tok"

    adopted = waiting.replace_rejected("nope", lambda: IssuedToken(access_token="y", expires_in_seconds=1.0))
    assert adopted.access_token == "tok"


def test_sync_guard_wait_exceeds_lock_bound_and_double_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    now = {"t": base}
    key = "K"
    guard = issue_guard_path(tmp_path, "dbsec", key)
    guard.write_text(repr((base - timedelta(seconds=10)).timestamp()), encoding="ascii")

    def _clock() -> datetime:
        return now["t"]

    tight = TokenStore(
        token_record_path(tmp_path, "dbsec", key),
        vendor="dbsec",
        lock_timeout_seconds=30.0,
        expiry_margin_seconds=600.0,
        clock=_clock,
        issue_throttle_policy="wait",
        sleep=lambda s: None,
    )
    with pytest.raises(TokenIssueThrottled):
        tight.get_or_issue(lambda: IssuedToken(access_token="x", expires_in_seconds=1.0))

    live = TokenRecord(access_token="live", issued_at=base, expires_at=base + timedelta(seconds=3600), generation=5)
    store = TokenStore(
        tmp_path / "t.json",
        vendor="toss",
        lock_timeout_seconds=5.0,
        expiry_margin_seconds=60.0,
        clock=lambda: base,
        issue_throttle_policy="raise",
        sleep=lambda s: None,
    )
    reads = {"n": 0}

    def _fake_read() -> TokenRecord | None:
        reads["n"] += 1
        if reads["n"] == 1:
            return None
        return live

    monkeypatch.setattr(store, "read", _fake_read)
    calls = {"n": 0}

    def _issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken(access_token="other", expires_in_seconds=3600.0)

    assert store.get_or_issue(_issue).access_token == "live"
    assert calls["n"] == 0


def test_pacer_wrappers_and_blocking_paths(tmp_path: Path) -> None:
    import os as _os

    state = tmp_path / "w.state"
    pacer = AsyncHostPacer(state, 4.0, clock=lambda: 1000.0, sleep=lambda s: asyncio.sleep(0))
    fd = _os.open(str(state), _os.O_RDWR | _os.O_CREAT, 0o600)
    try:
        import fcntl as _fl

        _fl.flock(fd, _fl.LOCK_EX)
        try:
            assert pacer._read_next_free_locked(fd) == 0.0
        finally:
            _fl.flock(fd, _fl.LOCK_UN)
    finally:
        _os.close(fd)
    assert pacer._reserve_slot() == 0.0

    with pytest.raises(ValueError, match="positive"):
        HostPacer(tmp_path / "bad.state", 0.0)

    bounded = HostPacer(tmp_path / "b.state", 4.0, max_lead_seconds=5.0, clock=lambda: 1000.0, sleep=lambda s: None)
    booked, _ = bounded._try_book_or_wait()
    assert booked is True
    (tmp_path / "b.state").write_text("1005.0", encoding="ascii")
    now = {"t": 1000.0}
    sleeps: list[float] = []

    def _sleep(s: float) -> None:
        sleeps.append(s)
        now["t"] += s

    waiter = HostPacer(tmp_path / "b.state", 4.0, max_lead_seconds=0.25, clock=lambda: now["t"], sleep=_sleep)
    waiter.acquire()
    assert now["t"] >= 1005.0 - 0.25
    assert sleeps

    unbound_state = tmp_path / "u.state"
    unbound_state.write_text("1000.5", encoding="ascii")
    seen: list[float] = []
    unbound = HostPacer(unbound_state, 4.0, clock=lambda: 1000.0, sleep=lambda s: seen.append(s))
    unbound.acquire()
    assert seen == [pytest.approx(0.5)]


def test_admission_corrupt_state_resets(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    state = tmp_path / "tps.state"
    state.write_text("garbage", encoding="utf-8")
    sleeps: list[float] = []

    async def _sleep(s: float) -> None:
        sleeps.append(s)

    limiter = AsyncHostPacer(state, 4.0, clock=lambda: 1000.0, sleep=_sleep)
    with caplog.at_level(logging.WARNING):
        asyncio.run(limiter.acquire())
    assert sleeps == []
    assert float(state.read_text(encoding="ascii")) == pytest.approx(1000.25)
    assert any("STATE_RESET" in r.getMessage() for r in caplog.records)
    with pytest.raises(ValueError, match="positive"):
        AsyncHostPacer(tmp_path / "z.state", 4.0, max_lead_seconds=0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_limits_rejected(tmp_path: Path, value: float) -> None:
    for pacer_type in (HostPacer, AsyncHostPacer):
        for kwargs in ({"max_rate": value}, {"max_rate": 4.0, "time_period": value},
                       {"max_rate": 4.0, "max_lead_seconds": value}):
            with pytest.raises(ValueError, match="positive"):
                pacer_type(tmp_path / "s.state", **kwargs)
    for name in ("lock_timeout_seconds", "expiry_margin_seconds", "unknown_expiry_max_age_seconds"):
        with pytest.raises(ValueError, match=name):
            _async_store(tmp_path / "t.json", **{name: value})


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", ""])
def test_nonfinite_host_files_recover(tmp_path: Path, raw: str) -> None:
    import src.api.host_broker_contract as hbc

    guard = tmp_path / "guard.state"
    guard.write_text(raw, encoding="ascii")
    assert hbc._read_guard_value(guard) is None
    state = tmp_path / "admission.state"
    state.write_text(raw, encoding="ascii")
    HostPacer(state, 4.0, clock=lambda: 1000.0).acquire()
    assert state.read_text(encoding="ascii") == "1000.25"


def test_malformed_records_and_scope_rejected(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    path.write_text(json.dumps({"schema_version": True, "access_token": "a", "issued_at": _NOW.isoformat(),
                                "expires_at": None, "generation": 1}), encoding="utf-8")
    assert _async_store(path).read() is None
    path.write_bytes(b"\xff")
    assert _async_store(path).read() is None
    with pytest.raises(ValueError, match="scope"):
        admission_state_path(tmp_path, "toss", "k", "STOCK_TRADING_TREND\n")


def test_dbsec_sync_async_guard_bytes_and_atomic_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    paths = [token_record_path(tmp_path / mode, "dbsec", "k") for mode in ("sync", "async")]
    kwargs = {"vendor": "dbsec", "lock_timeout_seconds": 5.0, "expiry_margin_seconds": 0.0,
              "clock": lambda: _NOW, "issue_throttle_policy": "raise"}

    def issue() -> IssuedToken:
        return IssuedToken(access_token="same", expires_in_seconds=3600.0)

    async def async_issue() -> IssuedToken:
        return issue()

    TokenStore(paths[0], **kwargs).get_or_issue(issue)
    asyncio.run(AsyncTokenStore(paths[1], **kwargs).get_or_issue(async_issue))
    guards = [issue_guard_path(path.parent, "dbsec", "k") for path in paths]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert guards[0].read_bytes() == guards[1].read_bytes() == repr(_NOW.timestamp()).encode("ascii")
    assert guards[0].stat().st_mode & 0o777 == 0o600

    old = repr((_NOW - timedelta(seconds=100)).timestamp())
    guards[0].write_text(old, encoding="ascii")
    store = TokenStore(paths[0], **kwargs)
    posts = []

    def forbidden_issue() -> IssuedToken:
        posts.append(1)
        return issue()

    def fail_replace(*args: object) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failed"):
        store.replace_rejected("same", forbidden_issue)
    assert posts == []
    assert guards[0].read_text(encoding="ascii") == old
    assert not list(paths[0].parent.glob("*.tmp"))


def test_toss_group_rates_are_read_only() -> None:
    from src.api.toss.client import TOSS_RATE_LIMIT_GROUPS

    with pytest.raises(TypeError):
        TOSS_RATE_LIMIT_GROUPS["ORDER"] = 100.0
