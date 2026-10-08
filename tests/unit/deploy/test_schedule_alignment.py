"""Schedule/schedule-constant alignment guards."""

from __future__ import annotations

import re
from datetime import time
from pathlib import Path

import pytest

from src.config.collection import CollectionSettings
from src.config.market_session import KIWOOM_TAPE_DEPTH_DAYS
from src.tools.ops_sentinel import parse_timer_schedule

_ROOT = Path(__file__).resolve().parents[3] / "deploy" / "systemd"


def _oncalendar_hhmmss(name: str) -> str:
    text = (_ROOT / name).read_text(encoding="utf-8")
    match = re.search(r"OnCalendar=.*?(\d{2}):(\d{2}):(\d{2})", text)
    assert match is not None, name
    return "".join(match.groups())


def _to_seconds(hhmmss: str) -> int:
    return int(hhmmss[0:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])


def test_archive_ready_times_match_timers() -> None:
    from src.config.market_session import ARCHIVE_AFTERMARKET_READY_HHMMSS, ARCHIVE_REGULAR_READY_HHMMSS

    assert _oncalendar_hhmmss("kca-archive-intraday.timer") == ARCHIVE_AFTERMARKET_READY_HHMMSS
    assert _oncalendar_hhmmss("kca-archive-intraday-regular.timer") == ARCHIVE_REGULAR_READY_HHMMSS


def test_sweep_deadline_sits_before_evening_price_ingest() -> None:
    from src.config.market_session import PRICE_INGEST_EVENING_HHMMSS, TAPE_SWEEP_DEADLINE_HHMMSS

    assert _to_seconds(TAPE_SWEEP_DEADLINE_HHMMSS) == _to_seconds(PRICE_INGEST_EVENING_HHMMSS) - 15 * 60
    price_slots = parse_timer_schedule((_ROOT / "kca-price-ingest.timer").read_text(encoding="utf-8"))
    evening = time.fromisoformat(
        f"{PRICE_INGEST_EVENING_HHMMSS[:2]}:{PRICE_INGEST_EVENING_HHMMSS[2:4]}:{PRICE_INGEST_EVENING_HHMMSS[4:]}"
    )
    assert evening == max(slot.time_of_day for slot in price_slots)
    assert _to_seconds(_oncalendar_hhmmss("kca-tape-sweep.timer")) < _to_seconds(TAPE_SWEEP_DEADLINE_HHMMSS)


def test_audit_follows_sweep_deadline() -> None:
    from src.config.market_session import TAPE_SWEEP_DEADLINE_HHMMSS

    assert _to_seconds(_oncalendar_hhmmss("kca-daily-audit.timer")) >= _to_seconds(TAPE_SWEEP_DEADLINE_HHMMSS)


def test_lookback_within_depth_and_retention() -> None:
    from src.config.collection import CollectionSettings
    from src.config.market_session import KIWOOM_TAPE_DEPTH_DAYS
    from src.tools.capture_offsite import LOCAL_SEALED_RETENTION_DAYS

    lookback = int(CollectionSettings().COLLECTION_TAPE_LOOKBACK_DAYS)
    assert lookback <= KIWOOM_TAPE_DEPTH_DAYS
    assert lookback + 1 < LOCAL_SEALED_RETENTION_DAYS


def test_expiry_margin_fits_window() -> None:
    from src.config.collection import CollectionSettings
    from src.config.market_session import TAPE_EXPIRY_WARNING_DAYS

    assert int(CollectionSettings().COLLECTION_TAPE_LOOKBACK_DAYS) > TAPE_EXPIRY_WARNING_DAYS


def test_no_literal_duplicates() -> None:
    import src.daily.tick_tape_sweep as sweep
    import src.tools.daily_audit as audit

    assert not hasattr(sweep, "_TAPE_DEPTH_DAYS")
    assert not hasattr(sweep, "_EXPIRY_WARNING_DAYS")
    assert not hasattr(audit, "TAPE_DEPTH_DAYS")
    assert not hasattr(audit, "TAPE_EXPIRY_WARNING_DAYS")


def test_lookback_depth_boundary() -> None:
    from pydantic import ValidationError

    assert CollectionSettings(COLLECTION_TAPE_LOOKBACK_DAYS=KIWOOM_TAPE_DEPTH_DAYS).COLLECTION_TAPE_LOOKBACK_DAYS == KIWOOM_TAPE_DEPTH_DAYS
    with pytest.raises(ValidationError, match="COLLECTION_TAPE_LOOKBACK_DAYS"):
        CollectionSettings(COLLECTION_TAPE_LOOKBACK_DAYS=KIWOOM_TAPE_DEPTH_DAYS + 1)
