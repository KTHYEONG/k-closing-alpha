"""Invariant guards for the LS/Toss/DB host-shared token client mixin."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from src.api.dbsec.client import DbsecApiClient, DbsecTokenThrottled
from src.api.host_broker_contract import TokenIssueThrottled, issue_guard_path
from src.api.ls.client import LsApiClient
from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
from src.api.toss.client import TossApiClient
from tests.broker_fakes import FakeBrokerResponse, scripted_session


def _seed_admission(tmp_path: Any, monkeypatch: Any) -> None:
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")


class _NoPostSession:
    def __init__(self) -> None:
        self.n = 0

    def post(self, *args: Any, **kwargs: Any) -> Any:
        self.n += 1
        raise AssertionError("token endpoint must not be contacted")


def test_mixin_store_binding(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    digest = hashlib.sha256(b"k").hexdigest()[:12]
    cases = [(LsApiClient, "ls", 300.0), (TossApiClient, "toss", 300.0), (DbsecApiClient, "dbsec", 600.0)]
    for cls, vendor, margin in cases:
        client = cls(app_key="k", app_secret="s")
        store = client._token_store()
        assert store._path.name == f"token_{vendor}_{digest}.json"
        assert store._vendor == vendor
        assert store._expiry_margin_seconds == margin
        assert store._issue_throttle_policy == "raise"
        assert client._token_store() is not store


def test_dbsec_throttle_translated(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = DbsecApiClient(app_key="k", app_secret="s")
    client.token = "keep"
    issue_guard_path(tmp_path, "dbsec", "k").write_text(repr(time.time() - 10.0))
    session = _NoPostSession()

    with pytest.raises(DbsecTokenThrottled) as ensure_exc:
        asyncio.run(client.ensure_token(session))  # type: ignore[arg-type]
    assert type(ensure_exc.value.__cause__) is TokenIssueThrottled
    assert str(ensure_exc.value) == "DB token issuance throttled (1/min)"
    assert client.token == "keep"

    with pytest.raises(DbsecTokenThrottled) as refresh_exc:
        asyncio.run(client._refresh_rejected_token(session, "keep"))  # type: ignore[arg-type]
    assert type(refresh_exc.value.__cause__) is TokenIssueThrottled
    assert str(refresh_exc.value) == "DB token issuance throttled (1/min)"
    assert session.n == 0
    assert client.token == "keep"

    issue_guard_path(tmp_path, "dbsec", "k").unlink(missing_ok=True)
    vendor = scripted_session([], token_bodies=[{"error_code": "IGW00201"}])
    with pytest.raises(DbsecTokenThrottled) as vendor_exc:
        asyncio.run(client.ensure_token(vendor))
    assert vendor_exc.value.__cause__ is None


def test_refresh_uses_sent_token(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)

    async def _seed() -> None:
        store = SharedTokenStore(
            shared_token_path("toss", "k"),
            vendor="toss",
            lock_timeout_seconds=5.0,
            expiry_margin_seconds=0.0,
            issue_throttle_policy="raise",
            clock=lambda: datetime.now(UTC),
        )
        await store.get_or_issue(
            lambda: asyncio.sleep(0, result=IssuedToken(access_token="A", expires_in_seconds=86400))
        )
        await store.replace_rejected(
            "A", lambda: asyncio.sleep(0, result=IssuedToken(access_token="B", expires_in_seconds=86400))
        )

    asyncio.run(_seed())
    client = TossApiClient(app_key="k", app_secret="s")
    # A peer has rotated both the store and client while the request carrying A was in flight.
    client.token = "B"
    session = scripted_session([])
    asyncio.run(client._refresh_rejected_token(session, "A"))
    assert client.token == "B"
    assert session.requests_to("/oauth2/token") == []


def test_single_refresh_replay_unchanged(tmp_path, monkeypatch, caplog) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = TossApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}, status=401),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ],
        token_bodies=[{"access_token": "tok-1", "expires_in": 86400}],
    )
    asyncio.run(client.ensure_token(session))
    before = len(session.requests_to("/oauth2/token"))

    with caplog.at_level(logging.WARNING):
        data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert data == {"result": {"ok": True}}
    assert len(session.requests_to("/oauth2/token")) - before == 1
    assert len([r for r in session.requests if r.method == "GET"]) == 2
    assert sum("TOKEN_REPLACED" in record.message for record in caplog.records) == 1
