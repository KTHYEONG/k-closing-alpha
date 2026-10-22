from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

TABLE_PATH = Path(__file__).resolve().parents[3] / "deploy" / "shared" / "kis_slot_assignments.toml"
CANONICAL_HEADER = "# canonical: krx-alpha/deploy/shared/kis_slot_assignments.toml schema=1"
PINNED_SHA256 = "5064bbb38e126f37befdbd26a80578000bfd061fb0e4dd49478f2e8386e6a751"

_WINDOW_RE = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")


def _load_doc() -> dict:
    return tomllib.loads(TABLE_PATH.read_text(encoding="utf-8"))


def _clock_to_minutes(clock: str) -> int:
    hour, minute = (int(part) for part in clock.split(":"))
    assert 0 <= hour <= 24 and 0 <= minute <= 59 and not (hour == 24 and minute != 0), clock
    return hour * 60 + minute


def _window_to_range(window: str) -> tuple[int, int]:
    assert _WINDOW_RE.fullmatch(window), window
    start, end = (_clock_to_minutes(part) for part in window.split("-"))
    assert start < end <= 24 * 60, window
    return start, end


def _rest_overlaps(rows: list[dict]) -> list[str]:
    rest = [row for row in rows if row["channel"] == "rest"]
    violations = []
    for i, left in enumerate(rest):
        for right in rest[i + 1 :]:
            if left["slot"] != right["slot"] or left["project"] == right["project"]:
                continue
            for left_window in left["windows"]:
                left_range = _window_to_range(left_window)
                for right_window in right["windows"]:
                    right_range = _window_to_range(right_window)
                    if max(left_range[0], right_range[0]) < min(left_range[1], right_range[1]):
                        violations.append(
                            f"slot {left['slot']}: {left['project']} {left_window} overlaps "
                            f"{right['project']} {right_window}"
                        )
    return violations


def test_vendored_table_hash_pinned() -> None:
    raw = TABLE_PATH.read_bytes()

    assert raw.splitlines()[0].decode("utf-8") == CANONICAL_HEADER
    assert hashlib.sha256(raw).hexdigest() == PINNED_SHA256


def test_schema_valid() -> None:
    doc = _load_doc()

    assert type(doc["schema_version"]) is int and doc["schema_version"] == 1
    pool = set(doc["pool"])
    assert pool and all(type(slot) is int and slot > 0 for slot in pool)
    assert doc["host_data_slots"]
    assert all(type(slot) is int for slot in doc["host_data_slots"])
    assert set(doc["host_data_slots"]) <= pool
    for selectors in doc["selectors"].values():
        for value in selectors.values():
            slots = value if isinstance(value, list) else [value]
            assert slots and all(type(slot) is int and slot in pool for slot in slots)
    for row in doc["assignment"]:
        assert type(row["slot"]) is int and row["slot"] in pool
        assert row["channel"] in {"rest", "ws"}
        assert row["project"] in {"kca", "krx", "kse"}
        assert isinstance(row["purpose"], str) and row["purpose"]
        assert row["windows"] and all(_window_to_range(window) for window in row["windows"])


def test_kca_selectors_match_provision_env() -> None:
    from src.tools.provision_env import VPS_SELECTORS

    selectors = _load_doc()["selectors"]["kca"]
    declared = dict(VPS_SELECTORS)

    for name, value in selectors.items():
        assert ",".join(map(str, value)) == declared[name]
    for name in declared:
        if name.endswith("_SLOTS"):
            assert name in selectors


def test_krx_snapshot_slot_matches() -> None:
    from src.tools.provision_env import KRX_SNAPSHOT_DATA_SLOT

    selector = _load_doc()["selectors"]["krx"]["KRX_ALPHA_SNAPSHOT_KIS_DATA_SLOT"]

    assert str(selector) == KRX_SNAPSHOT_DATA_SLOT


def test_decision_shards_lead_with_decision_slot() -> None:
    doc = _load_doc()

    assert doc["selectors"]["kca"]["KIS_DECISION_SHARD_SLOTS"][0] == doc["host_data_slots"][0]


def test_intraday_kca_avoids_krx_snapshot() -> None:
    doc = _load_doc()
    selectors = doc["selectors"]
    snapshot = selectors["krx"]["KRX_ALPHA_SNAPSHOT_KIS_DATA_SLOT"]

    intraday = {
        doc["host_data_slots"][0],
        *selectors["kca"]["KIS_DECISION_SHARD_SLOTS"],
        *selectors["kca"]["COLLECTION_RESEARCH_SLOTS"],
    }
    assert snapshot not in intraday


def test_research_disjoint_from_shards() -> None:
    selectors = _load_doc()["selectors"]["kca"]

    assert set(selectors["COLLECTION_RESEARCH_SLOTS"]).isdisjoint(selectors["KIS_DECISION_SHARD_SLOTS"])


def test_no_cross_project_rest_overlap() -> None:
    assert _rest_overlaps(_load_doc()["assignment"]) == []


def test_overlap_detector_catches_conflict() -> None:
    rows = [
        *_load_doc()["assignment"],
        {"slot": 2, "project": "kca", "channel": "rest", "windows": ["15:00-15:30"]},
    ]

    violations = _rest_overlaps(rows)

    assert len(violations) == 1
    (violation,) = violations
    assert "2" in violation and "kca" in violation and "krx" in violation


def test_selector_slots_have_rows() -> None:
    doc = _load_doc()
    kca_slots = {row["slot"] for row in doc["assignment"] if row["project"] == "kca"}

    for slots in doc["selectors"]["kca"].values():
        assert set(slots) <= kca_slots


@pytest.mark.parametrize(("window", "invalid"), [
    ("24:00-24:01", "24:01"),
    ("12:60-13:00", "12:60"),
    ("23:00-01:00", "23:00-01:00"),
    ("08:00-08:00", "08:00-08:00"),
    ("08:00-09:00\n", "08:00-09:00"),
])
def test_invalid_windows_rejected(window: str, invalid: str) -> None:
    with pytest.raises(AssertionError, match=re.escape(invalid)):
        _window_to_range(window)


def test_overlap_detector_respects_half_open_windows_slots_and_channels() -> None:
    rows = [
        {"slot": 2, "project": "krx", "channel": "rest", "windows": ["08:00-15:39"]},
        {"slot": 2, "project": "kca", "channel": "rest", "windows": ["15:39-24:00"]},
        {"slot": 3, "project": "kca", "channel": "rest", "windows": ["08:00-15:39"]},
        {"slot": 2, "project": "kca", "channel": "ws", "windows": ["08:00-15:39"]},
        {"slot": 2, "project": "krx", "channel": "rest", "windows": ["09:00-10:00"]},
    ]

    assert _rest_overlaps(rows) == []
