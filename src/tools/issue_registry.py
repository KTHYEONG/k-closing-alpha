"""Routing classes for audit findings and advisory persistence tracking.

Research-data degradation no human action can improve stays advisory until it
persists across consecutive audits; everything else warns immediately.
"""

from __future__ import annotations

import enum
import json
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from src import settings
from src.data.io_utils import atomic_write_text

logger = logging.getLogger(__name__)


class IssueTier(enum.StrEnum):
    """Routing class of one audit key. TRADING_CHAIN and DATA_INTEGRITY keys warn immediately; RESEARCH_DEGRADED keys are advisory (heartbeat info note, no mail) until they persist ADVISORY_ESCALATION_AUDITS consecutive audits."""

    TRADING_CHAIN = "trading_chain"
    DATA_INTEGRITY = "data_integrity"
    RESEARCH_DEGRADED = "research_degraded"


ADVISORY_ESCALATION_AUDITS: int = 3

ADVISORY_STREAKS_RELPATH: str = "logs/heartbeat/advisory_streaks.json"
ADVISORY_STREAKS_SCHEMA_VERSION: int = 1

_TRADING_CHAIN_STEPS: frozenset[str] = frozenset(
    {
        "archive",
        "close_confirmed",
        "decision",
        "paper_entry",
        "paper_exit",
        "price_history_fresh",
        "minute_bars",
    }
)

_TRADING_CHAIN_UNITS: frozenset[str] = frozenset(
    {
        "collect",
        "predict",
        "auction-close",
        "finalize-close",
        "paper-entry",
        "paper-exit",
        "kiwoom-token-rotate",
        "kis-token-warmup",
        "price-ingest",
    }
)

_RESEARCH_DEGRADED_CLASSES: frozenset[str] = frozenset(
    {
        "intraday:regular_ticks:volume_gap",
        "intraday:regular_ticks:certified_gap",
        # Same-evening aftermarket tick shortfalls are recovered by the next sweep (session closes at the day boundary);
        # unrecovered ones resurface through tape expiry warnings.
        "intraday:krx_aftermarket_ticks:volume_mismatch",
        "intraday:nxt_aftermarket_ticks:volume_mismatch",
    }
)


def issue_class(key: str) -> str:
    """Strip the volatile count from a key (`intraday:regular_ticks:2:volume_gap` -> `intraday:regular_ticks:volume_gap`) so streaks follow the condition, not its magnitude."""
    parts = key.split(":")
    if len(parts) == 4 and parts[2].isdigit():
        return f"{parts[0]}:{parts[1]}:{parts[3]}"
    return key


def classify_issue_tier(key: str) -> IssueTier:
    """Map a stable issue key to its tier by explicit rule table. Unknown keys are DATA_INTEGRITY (fail loud): a new warning must be consciously downgraded in the rule table. `missing:<trading-chain step>` and `failed_unit:*` map to TRADING_CHAIN when the unit/step is part of the decision-to-fill chain, otherwise DATA_INTEGRITY. Only regular tick `volume_gap`/`certified_gap` and aftermarket tick `volume_mismatch` are RESEARCH_DEGRADED."""
    cls = issue_class(key)
    if cls in _RESEARCH_DEGRADED_CLASSES:
        return IssueTier.RESEARCH_DEGRADED
    if cls.startswith("missing:"):
        step = cls.split(":", 1)[1]
        if step in _TRADING_CHAIN_STEPS:
            return IssueTier.TRADING_CHAIN
        return IssueTier.DATA_INTEGRITY
    if cls.startswith("failed_unit:"):
        unit = cls.split(":", 1)[1]
        if unit.startswith("kca-"):
            unit = unit[len("kca-") :]
        if unit.endswith(".service"):
            unit = unit[: -len(".service")]
        if unit in _TRADING_CHAIN_UNITS:
            return IssueTier.TRADING_CHAIN
        return IssueTier.DATA_INTEGRITY
    return IssueTier.DATA_INTEGRITY


@dataclass(frozen=True)
class AdvisoryStreak:
    """Consecutive-audit persistence of one advisory class (counts and dates only)."""

    first_date: str
    last_date: str
    consecutive_audits: int


def update_advisory_streaks(
    previous: Mapping[str, AdvisoryStreak],
    observed_classes: Collection[str],
    *,
    audit_date: str,
    previous_audit_date: str | None,
) -> tuple[dict[str, AdvisoryStreak], tuple[str, ...]]:
    """Returns the new streak map and the classes escalated (consecutive_audits >= ADVISORY_ESCALATION_AUDITS). A class observed in an audit directly following `previous_audit_date` extends its streak; a gap (class absent in the prior audit) restarts at 1; classes not observed are dropped. Same `audit_date` called twice is idempotent."""
    observed = set(observed_classes)
    new_map: dict[str, AdvisoryStreak] = {}
    for cls in observed:
        prev = previous.get(cls)
        if prev is not None and prev.last_date == audit_date:
            new_map[cls] = prev
        elif prev is not None and previous_audit_date is not None and prev.last_date == previous_audit_date:
            new_map[cls] = AdvisoryStreak(
                first_date=prev.first_date,
                last_date=audit_date,
                consecutive_audits=prev.consecutive_audits + 1,
            )
        else:
            new_map[cls] = AdvisoryStreak(
                first_date=audit_date,
                last_date=audit_date,
                consecutive_audits=1,
            )
    escalated = tuple(sorted(cls for cls, streak in new_map.items() if streak.consecutive_audits >= ADVISORY_ESCALATION_AUDITS))
    return new_map, escalated


def _streaks_path(path: Path | None = None) -> Path:
    return Path(path) if path is not None else Path(settings.DATA_DIR) / ADVISORY_STREAKS_RELPATH


def load_advisory_streaks(path: Path | None = None) -> dict[str, AdvisoryStreak]:
    """Load persisted advisory streaks; empty on absent or corrupt evidence."""
    target = _streaks_path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("[SYS] stage=issue_registry advisory_streaks=LOAD_FAILED reason=%s", type(exc).__name__)
        return {}
    try:
        if not isinstance(raw, dict):
            raise ValueError("streak file must be a mapping")
        node = raw.get("streaks", raw) if "streaks" in raw or "schema_version" in raw else raw
        if not isinstance(node, dict):
            raise ValueError("streaks node must be a mapping")
        streaks: dict[str, AdvisoryStreak] = {}
        for cls, entry in node.items():
            if not isinstance(cls, str) or not isinstance(entry, dict):
                raise ValueError("invalid streak entry")
            first = str(entry["first_date"])
            last = str(entry["last_date"])
            count = int(entry["consecutive_audits"])
            if count < 1:
                raise ValueError("consecutive_audits must be positive")
            streaks[cls] = AdvisoryStreak(first_date=first, last_date=last, consecutive_audits=count)
        return streaks
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("[SYS] stage=issue_registry advisory_streaks=LOAD_FAILED reason=%s", type(exc).__name__)
        return {}


def write_advisory_streaks(streaks: Mapping[str, AdvisoryStreak], path: Path | None = None) -> None:
    """Persist advisory streaks atomically (counts and dates only)."""
    target = _streaks_path(path)
    payload = {
        "schema_version": ADVISORY_STREAKS_SCHEMA_VERSION,
        "streaks": {
            cls: {
                "first_date": streak.first_date,
                "last_date": streak.last_date,
                "consecutive_audits": streak.consecutive_audits,
            }
            for cls, streak in streaks.items()
        },
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2), mode=0o644)
