from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from src.api.kiwoom.client import KiwoomIssuedToken
from src.tools.kiwoom_token_rotate import expiry_in_protected_window, rotate_kiwoom_token

_SEOUL = ZoneInfo("Asia/Seoul")
_WINDOWS = ("15:10-15:40",)


def _at(hms: str, day: str = "2026-10-02") -> datetime:
    return datetime.strptime(f"{day}{hms}", "%Y-%m-%d%H:%M:%S").replace(tzinfo=_SEOUL)


class _FakeClient:
    def __init__(self, expiries: list[datetime], revoke_code: int = 0) -> None:
        self._expiries = list(expiries)
        self.revoke_code = revoke_code
        self.issue_calls = 0
        self.revoke_calls = 0
        self.tokens = [f"tok-{i + 1}" for i in range(len(expiries))]

    async def issue_token(self, session: Any) -> KiwoomIssuedToken:
        idx = self.issue_calls
        self.issue_calls += 1
        return KiwoomIssuedToken(token=self.tokens[idx], expires_at=self._expiries[idx])

    async def revoke_token(self, session: Any, token: str) -> None:
        self.revoke_calls += 1
        if self.revoke_code != 0:
            raise RuntimeError(f"Kiwoom token revocation failed: {self.revoke_code}")

    def reset_token(self) -> None:
        return None


def test_expiry_inside_decision_window_triggers_rotation() -> None:
    client = _FakeClient([_at("15:20:06"), _at("07:10:01", "2026-10-03")])
    outcome = asyncio.run(
        rotate_kiwoom_token(client, object(), windows=_WINDOWS, dry_run=False)
    )
    assert outcome.status == "ROTATED"
    assert client.revoke_calls == 1
    assert outcome.new_expiry == _at("07:10:01", "2026-10-03")


def test_already_safe_phase_is_untouched() -> None:
    client = _FakeClient([_at("07:10:00")])
    outcome = asyncio.run(
        rotate_kiwoom_token(client, object(), windows=_WINDOWS, dry_run=False)
    )
    assert outcome.status == "UNCHANGED"
    assert client.revoke_calls == 0
    assert outcome.previous_expiry == outcome.new_expiry


def test_dry_run_never_revokes() -> None:
    client = _FakeClient([_at("15:20:06")])
    outcome = asyncio.run(
        rotate_kiwoom_token(client, object(), windows=_WINDOWS, dry_run=True)
    )
    assert outcome.status == "DRY_RUN"
    assert client.revoke_calls == 0
    assert client.issue_calls == 1


def test_reissue_still_in_window_fails() -> None:
    client = _FakeClient([_at("15:20:06"), _at("15:25:00")])
    with pytest.raises(RuntimeError, match="still inside"):
        asyncio.run(
            rotate_kiwoom_token(client, object(), windows=_WINDOWS, dry_run=False)
        )


def test_revoke_failure_surfaces_without_reissue() -> None:
    client = _FakeClient([_at("15:20:06"), _at("07:10:01", "2026-10-03")], revoke_code=1)
    with pytest.raises(RuntimeError, match="revocation failed"):
        asyncio.run(
            rotate_kiwoom_token(client, object(), windows=_WINDOWS, dry_run=False)
        )
    assert client.revoke_calls == 1
    assert client.issue_calls == 1


def test_window_boundaries_are_start_inclusive_end_exclusive() -> None:
    assert expiry_in_protected_window(_at("15:10:00"), _WINDOWS) is True
    assert expiry_in_protected_window(_at("15:40:00"), _WINDOWS) is False
    assert expiry_in_protected_window(_at("15:09:59"), _WINDOWS) is False


def test_every_outcome_logs_sys_stage_line(caplog) -> None:
    import logging

    client = _FakeClient([_at("07:10:00")])
    with caplog.at_level(logging.INFO):
        asyncio.run(
            rotate_kiwoom_token(client, object(), windows=_WINDOWS, dry_run=False)
        )
    assert "[SYS] stage=kiwoom_token_rotate status=UNCHANGED" in caplog.text


def test_cli_exits_nonzero_when_reissue_stays_in_window(monkeypatch) -> None:
    import src.tools.kiwoom_token_rotate as rotate_mod

    client = _FakeClient([_at("15:20:06"), _at("15:25:00")])
    monkeypatch.setattr(rotate_mod, "KiwoomApiClient", lambda: client)
    monkeypatch.setattr(
        rotate_mod.settings, "KIWOOM_TOKEN_PROTECTED_WINDOWS", _WINDOWS
    )

    class _SessionFactory:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    monkeypatch.setattr(rotate_mod.aiohttp, "ClientSession", _SessionFactory)
    assert rotate_mod.main([]) == 1


def _patched_cli(monkeypatch, expiries: list) -> Any:
    import src.tools.kiwoom_token_rotate as rotate_mod

    client = _FakeClient(expiries)
    monkeypatch.setattr(rotate_mod, "KiwoomApiClient", lambda: client)
    monkeypatch.setattr(
        rotate_mod.settings, "KIWOOM_TOKEN_PROTECTED_WINDOWS", _WINDOWS
    )

    class _SessionFactory:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    monkeypatch.setattr(rotate_mod.aiohttp, "ClientSession", _SessionFactory)
    return client


def test_cli_reports_unchanged_and_exits_zero(monkeypatch) -> None:
    import src.tools.kiwoom_token_rotate as rotate_mod

    client = _patched_cli(monkeypatch, [_at("07:10:00")])
    assert rotate_mod.main([]) == 0
    assert client.revoke_calls == 0


def test_cli_dry_run_flag_never_revokes(monkeypatch) -> None:
    import src.tools.kiwoom_token_rotate as rotate_mod

    client = _patched_cli(monkeypatch, [_at("15:20:06")])
    assert rotate_mod.main(["--dry-run"]) == 0
    assert client.revoke_calls == 0


def test_cli_explicit_dry_run_false_rotates(monkeypatch) -> None:
    import src.tools.kiwoom_token_rotate as rotate_mod

    client = _patched_cli(monkeypatch, [_at("15:20:06"), _at("07:10:01", "2026-10-03")])
    assert rotate_mod.main(["--dry-run=false"]) == 0
    assert client.revoke_calls == 1


def test_parse_bool_accepts_flag_spellings() -> None:
    import pytest

    from src.tools.kiwoom_token_rotate import _parse_bool

    assert _parse_bool(True) is True
    assert _parse_bool("true") is True
    assert _parse_bool("false") is False
    assert _parse_bool("0") is False
    with pytest.raises(ValueError, match="invalid boolean value"):
        _parse_bool("maybe")
