"""Shared token store invariant guards."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.api.shared_token import (
    IssuedToken,
    SharedTokenStore,
    TokenStoreLockTimeout,
    shared_token_path,
)


def _store(tmp_path: Path, **kw) -> SharedTokenStore:
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    args = {"lock_timeout_seconds": 5.0, "expiry_margin_seconds": 60.0, "clock": lambda: now}
    args.update(kw)
    return SharedTokenStore(tmp_path / "token_toss_abc.json", **args)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_payload(tok: str = "tok-a", gen: int = 1, expires_in: float = 3600.0) -> dict:  # noqa: S107 - test fixture label, not a secret
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    return {
        "schema_version": 1,
        "access_token": tok,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=expires_in)).isoformat() if expires_in is not None else None,
        "generation": gen,
    }


def test_shared_token_path_deterministic_and_rejects(monkeypatch) -> None:
    from src import settings

    monkeypatch.setattr(settings.settings, "BROKER_ADMISSION_REQUIRE_SHARED", "never")
    p = shared_token_path("toss", "cred-1")
    digest = hashlib.sha256(b"cred-1").hexdigest()[:12]
    assert p.name == f"token_toss_{digest}.json"
    with pytest.raises(ValueError, match="vendor"):
        shared_token_path("../x", "cred-1")
    with pytest.raises(ValueError, match="credential"):
        shared_token_path("toss", "")


def test_single_issuance_under_concurrency(tmp_path: Path) -> None:
    calls: list[int] = []

    async def _issue() -> IssuedToken:
        calls.append(1)
        await asyncio.sleep(0.05)
        return IssuedToken(access_token="tok-new", expires_in_seconds=3600.0)

    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    a = SharedTokenStore(tmp_path / "t.json", lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
    b = SharedTokenStore(tmp_path / "t.json", lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    async def _run():
        ra, rb = await asyncio.gather(a.get_or_issue(_issue), b.get_or_issue(_issue))
        return ra, rb

    ra, rb = asyncio.run(_run())
    assert len(calls) == 1
    assert ra.access_token == rb.access_token == "tok-new"
    assert ra.generation == rb.generation == 1


def test_rejection_after_rotation_adopts(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    _write(path, _valid_payload(tok="tok-b", gen=2))
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
    issued = {"n": 0}

    async def _issue() -> IssuedToken:
        issued["n"] += 1
        return IssuedToken(access_token="tok-c", expires_in_seconds=3600.0)

    rec = asyncio.run(store.replace_rejected("tok-a", _issue))
    assert rec.access_token == "tok-b"
    assert rec.generation == 2
    assert issued["n"] == 0


def test_rejection_does_not_adopt_expired_peer_token(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    _write(path, _valid_payload(tok="tok-b", gen=2, expires_in=30.0))
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="tok-c", expires_in_seconds=3600.0)

    rec = asyncio.run(store.replace_rejected("tok-a", _issue))
    assert rec.access_token == "tok-c"
    assert rec.generation == 3


def test_rejection_of_current_rotates(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    _write(path, _valid_payload(tok="tok-a", gen=1))
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="tok-b", expires_in_seconds=3600.0)

    rec = asyncio.run(store.replace_rejected("tok-a", _issue))
    assert rec.access_token == "tok-b"
    assert rec.generation == 2
    assert store.read() is not None and store.read().access_token == "tok-b"


def test_expired_token_reissued(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    _write(path, _valid_payload(tok="tok-old", gen=1, expires_in=-10.0))
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
    calls = {"n": 0}

    async def _issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken(access_token="tok-fresh", expires_in_seconds=3600.0)

    rec = asyncio.run(store.get_or_issue(_issue))
    assert calls["n"] == 1
    assert rec.access_token == "tok-fresh"
    assert rec.generation == 2


def test_failed_issuance_preserves_file(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    payload = _valid_payload(tok="tok-old", gen=1, expires_in=-10.0)
    _write(path, payload)
    before = path.read_bytes()
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    async def _boom() -> IssuedToken:
        raise RuntimeError("vendor down")

    with pytest.raises(RuntimeError, match="vendor down"):
        asyncio.run(store.get_or_issue(_boom))
    assert path.read_bytes() == before


def test_lock_timeout(tmp_path: Path) -> None:
    import fcntl
    import os

    path = tmp_path / "t.json"
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

        async def _no_sleep(_: float) -> None:
            return None

        store = SharedTokenStore(
            path, lock_timeout_seconds=0.05, expiry_margin_seconds=60.0, clock=lambda: now, sleep=_no_sleep
        )

        async def _issue() -> IssuedToken:
            return IssuedToken(access_token="x", expires_in_seconds=1.0)

        with pytest.raises(TokenStoreLockTimeout, match="token lock"):
            asyncio.run(store.get_or_issue(_issue))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_corrupt_file_treated_as_absent(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text("not-json{{{", encoding="utf-8")
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
    assert store.read() is None

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="tok-new", expires_in_seconds=3600.0)

    rec = asyncio.run(store.get_or_issue(_issue))
    assert rec.access_token == "tok-new"
    assert rec.generation == 1


def test_no_secret_persisted_and_exact_keys(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="tok-new", expires_in_seconds=3600.0)

    asyncio.run(store.get_or_issue(_issue))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "access_token", "issued_at", "expires_at", "generation"}
    assert "app_key" not in payload and "secret" not in payload


def test_usable_token_returned_without_issuing(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    _write(path, _valid_payload(tok="tok-live", gen=3))
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
    calls = {"n": 0}

    async def _issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken(access_token="tok-other", expires_in_seconds=3600.0)

    rec = asyncio.run(store.get_or_issue(_issue))
    assert rec.access_token == "tok-live"
    assert calls["n"] == 0


def test_double_checked_read_after_lock(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "t.json"
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
    live_rec = store._build_record(IssuedToken(access_token="tok-live", expires_in_seconds=3600.0), None)
    object.__setattr__(live_rec, "generation", 5)
    reads = {"n": 0}

    def _fake_read():
        reads["n"] += 1
        if reads["n"] == 1:
            return None
        return live_rec

    monkeypatch.setattr(store, "read", _fake_read)
    calls = {"n": 0}

    async def _issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken(access_token="tok-other", expires_in_seconds=3600.0)

    rec = asyncio.run(store.get_or_issue(_issue))
    assert rec.access_token == "tok-live"
    assert calls["n"] == 0


def test_schema_invalid_variants_return_none(tmp_path: Path) -> None:
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    aware = now.isoformat()
    naive = "2026-10-01T12:00:00"
    cases: list[object] = [
        [1, 2],
        {"schema_version": 2, "access_token": "a", "issued_at": aware, "expires_at": None, "generation": 1},
        {"schema_version": 1, "access_token": "", "issued_at": aware, "expires_at": None, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "expires_at": None, "generation": 0},
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "expires_at": None, "generation": True},
        {"schema_version": 1, "access_token": "a", "issued_at": 123, "expires_at": None, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": naive, "expires_at": None, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": "bogus", "expires_at": None, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "expires_at": 123, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "expires_at": naive, "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "expires_at": "bogus", "generation": 1},
        {"schema_version": 1, "access_token": "a", "issued_at": aware, "generation": 1},
    ]
    for i, payload in enumerate(cases):
        path = tmp_path / f"bad{i}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
        assert store.read() is None, i


def test_read_unreadable_logged(tmp_path: Path, monkeypatch) -> None:
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(tmp_path / "t.json", lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    def _boom(*a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", _boom)
    assert store.read() is None


def test_no_expiry_token_is_usable_and_none_expiry_persisted(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="tok-forever", expires_in_seconds=None)

    rec = asyncio.run(store.get_or_issue(_issue))
    assert rec.expires_at is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["expires_at"] is None
    calls = {"n": 0}

    async def _issue2() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken(access_token="other", expires_in_seconds=None)

    rec2 = asyncio.run(store.get_or_issue(_issue2))
    assert rec2.access_token == "tok-forever"
    assert calls["n"] == 0


def test_replace_rejected_when_absent_issues_gen1(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="tok-first", expires_in_seconds=3600.0)

    rec = asyncio.run(store.replace_rejected("stale", _issue))
    assert rec.generation == 1
    assert rec.access_token == "tok-first"


def test_init_rejects_bad_bounds(tmp_path: Path) -> None:
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="lock_timeout"):
        SharedTokenStore(tmp_path / "t.json", lock_timeout_seconds=0.0, expiry_margin_seconds=0.0, clock=lambda: now)
    with pytest.raises(ValueError, match="expiry_margin"):
        SharedTokenStore(tmp_path / "t.json", lock_timeout_seconds=1.0, expiry_margin_seconds=-1.0, clock=lambda: now)


def test_publish_failure_cleans_tmp(tmp_path: Path, monkeypatch) -> None:
    import os as _os

    path = tmp_path / "t.json"
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    store = SharedTokenStore(path, lock_timeout_seconds=5.0, expiry_margin_seconds=60.0, clock=lambda: now)
    monkeypatch.setattr(_os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    async def _issue() -> IssuedToken:
        return IssuedToken(access_token="tok-x", expires_in_seconds=60.0)

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(store.get_or_issue(_issue))
    assert not path.exists()


def test_issued_token_repr_hides_token() -> None:
    from src.api.shared_token import IssuedToken

    text = repr(IssuedToken(access_token="tok-SECRET-0001", expires_in_seconds=60.0))
    assert "tok-SECRET-0001" not in text
    assert "60.0" in text


def test_token_record_repr_hides_token() -> None:
    from datetime import UTC, datetime

    from src.api.shared_token import TokenRecord

    record = TokenRecord(access_token="tok-SECRET-0002", issued_at=datetime.now(UTC), expires_at=None, generation=7)
    text = repr(record)
    assert "tok-SECRET-0002" not in text
    assert "generation=7" in text
