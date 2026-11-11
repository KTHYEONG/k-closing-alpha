"""Direction-free liquid research roster: pure selection, immutable capture evidence, and reads."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore

logger = logging.getLogger(__name__)

RESEARCH_SUFFIX: str = "-research"
RESEARCH_ROSTER_RULE_VERSION = "liquid_any_direction@v1"
RESEARCH_ROSTER_REASON = "research-roster"
RESEARCH_ROSTER_FRAME_COLUMNS: tuple[str, ...] = (
    "symbol",
    "rank",
    "trade_value_100m",
    "change_pct",
    "last_price",
    "est_market_cap_100m",
    "member",
    "member_order",
    "reason",
)
_ROSTER_METADATA_COLUMNS: tuple[str, ...] = ("observed_at", "as_of_panel_date", "rule_version")

_KRW_PER_100M = 100_000_000
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9]+$")
_ROSTER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class RosterCandidate:
    symbol: str
    rank: int
    trade_value_100m: float
    change_pct: float
    last_price: int


class ResearchRoster(BaseModel):
    """Published roster members and the reasons every other scanned symbol was excluded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trading_date: date
    roster_id: str
    rule_version: str
    observed_at: datetime
    as_of_panel_date: date
    members: tuple[str, ...]
    dropped: dict[str, str]
    truncated: bool

    @field_validator("roster_id", mode="after")
    @classmethod
    def _check_roster_id(cls, v: str) -> str:
        if not _ROSTER_ID_RE.fullmatch(v):
            raise ValueError("roster_id must be path-safe")
        return v

    @field_validator("observed_at", mode="after")
    @classmethod
    def _check_observed_at(cls, v: datetime) -> datetime:
        if v.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return v

    @field_validator("members", mode="after")
    @classmethod
    def _check_members(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("members must be unique")
        if any(not _SYMBOL_RE.fullmatch(member) for member in v):
            raise ValueError("members must be valid symbols")
        return v

    @model_validator(mode="after")
    def _check_disjoint(self) -> Self:
        if set(self.members) & set(self.dropped):
            raise ValueError("members must be disjoint from dropped symbols")
        return self


def parse_trade_value_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[RosterCandidate], dict[str, str]]:
    """Normalize raw ka10032 rows; the first occurrence of a symbol decides its outcome.

    Args:
        rows: Raw vendor rows in vendor order.

    Returns:
        Parsed candidates in vendor order and unparseable rows keyed by symbol or row position.
    """
    candidates: list[RosterCandidate] = []
    dropped: dict[str, str] = {}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        try:
            symbol = str(row["stk_cd"]).split("_")[0].strip().zfill(6)
        except (KeyError, TypeError):
            dropped[f"<row-{index}>"] = _UNPARSEABLE
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        try:
            change_pct = float(row["flu_rt"])
            if not math.isfinite(change_pct):
                raise ValueError("nonfinite change")
            candidates.append(
                RosterCandidate(
                    symbol=symbol,
                    rank=int(row["now_rank"]),
                    trade_value_100m=int(row["trde_prica"]) / 100,
                    change_pct=change_pct,
                    last_price=abs(int(row["cur_prc"])),
                )
            )
        except (KeyError, TypeError, ValueError):
            dropped[symbol] = _UNPARSEABLE
    return candidates, dropped


def select_research_roster(
    candidates: Sequence[RosterCandidate],
    *,
    listed: frozenset[str],
    screenable: frozenset[str] | None,
    shares_by_symbol: Mapping[str, float],
    min_trade_value_100m: float,
    min_market_cap_100m: float,
    max_members: int,
) -> tuple[tuple[str, ...], dict[str, str], dict[str, float]]:
    """Select liquid, listed, screenable candidates whose estimated market cap clears the floor.

    Args:
        candidates: Parsed ranking candidates.
        listed: Symbols eligible in the panel at the prior close.
        screenable: Symbols passing the screenable class filter, or None when the filter is off.
        shares_by_symbol: Shares outstanding per symbol; absent or nonpositive means unknown.
        min_trade_value_100m: Trade-value floor in 100M KRW.
        min_market_cap_100m: Estimated market-cap floor in 100M KRW.
        max_members: Capacity cap applied after ordering.

    Returns:
        Members in (trade value desc, symbol asc) order, drop reasons, and estimated market caps.
    """
    dropped: dict[str, str] = {}
    est_mc: dict[str, float] = {}
    survivors: list[RosterCandidate] = []
    for candidate in candidates:
        shares = shares_by_symbol.get(candidate.symbol)
        if shares is not None and shares > 0:
            est_mc[candidate.symbol] = candidate.last_price * shares / _KRW_PER_100M
        if candidate.trade_value_100m < min_trade_value_100m:
            dropped[candidate.symbol] = "below_trade_value"
        elif candidate.symbol not in listed:
            dropped[candidate.symbol] = "not_listed_in_panel"
        elif screenable is not None and candidate.symbol not in screenable:
            dropped[candidate.symbol] = "non_screenable_class"
        elif shares is None or shares <= 0:
            dropped[candidate.symbol] = "market_cap_unknown"
        elif est_mc[candidate.symbol] < min_market_cap_100m:
            dropped[candidate.symbol] = "below_market_cap"
        else:
            survivors.append(candidate)
    survivors.sort(key=lambda item: (-item.trade_value_100m, item.symbol))
    for candidate in survivors[max_members:]:
        dropped[candidate.symbol] = "over_capacity"
    members = tuple(item.symbol for item in survivors[:max_members])
    return members, dropped, est_mc


def build_roster_frame(
    candidates: Sequence[RosterCandidate],
    members: Sequence[str],
    dropped: Mapping[str, str],
    est_mc: Mapping[str, float],
) -> pd.DataFrame:
    """Lay out one row per scanned symbol so the roster can be reconstructed from evidence.

    Args:
        candidates: Parsed candidates, one row each.
        members: Ordered member symbols.
        dropped: Exclusion reasons, including unparseable rows that never became candidates.
        est_mc: Estimated market caps in 100M KRW for symbols with known shares.

    Returns:
        Frame with RESEARCH_ROSTER_FRAME_COLUMNS.
    """
    order = {symbol: position for position, symbol in enumerate(members)}
    by_symbol = {candidate.symbol: candidate for candidate in candidates}
    records: list[dict[str, Any]] = []
    for symbol, candidate in by_symbol.items():
        records.append(
            {
                "symbol": symbol,
                "rank": candidate.rank,
                "trade_value_100m": candidate.trade_value_100m,
                "change_pct": candidate.change_pct,
                "last_price": candidate.last_price,
                "est_market_cap_100m": est_mc.get(symbol, math.nan),
            }
        )
    records.extend(
        {
            "symbol": symbol,
            "rank": None,
            "trade_value_100m": math.nan,
            "change_pct": math.nan,
            "last_price": None,
            "est_market_cap_100m": math.nan,
        }
        for symbol in dropped
        if symbol not in by_symbol
    )
    for record in records:
        symbol = record["symbol"]
        record["member"] = symbol in order
        record["member_order"] = order.get(symbol)
        record["reason"] = None if symbol in order else dropped.get(symbol)
    frame = pd.DataFrame.from_records(records, columns=list(RESEARCH_ROSTER_FRAME_COLUMNS))
    return frame.astype(
        {
            "rank": "Int64",
            "last_price": "Int64",
            "member_order": "Int64",
            "trade_value_100m": "float64",
            "change_pct": "float64",
            "est_market_cap_100m": "float64",
            "member": "bool",
        }
    )


def publish_research_roster(
    store: CaptureStore,
    roster: ResearchRoster,
    frame: pd.DataFrame,
    *,
    raw_refs: Sequence[ArtifactRef],
    completed_at: datetime,
) -> CaptureManifest:
    """Persist the roster frame and its manifest; the manifest carries no cohort.

    Args:
        store: Owner-local capture store.
        roster: Roster whose members must match the frame's member rows.
        frame: Output of build_roster_frame.
        raw_refs: Verified raw ranking page references.
        completed_at: Aware actual publication time, at or after roster.observed_at.

    Returns:
        The published manifest.

    Raises:
        ValueError: Naive or early completed_at, empty frame, or members not matching the frame.
        RawCaptureError: Durable frame publication failed.
    """
    if completed_at.utcoffset() is None:
        raise ValueError("completed_at must be timezone-aware")
    if completed_at < roster.observed_at:
        raise ValueError("completed_at precedes observed_at")
    if frame.empty:
        raise ValueError("research roster frame is empty")
    framed_members = tuple(frame.loc[frame["member"], "symbol"].astype(str).tolist())
    if set(framed_members) != set(roster.members) or len(framed_members) != len(roster.members):
        raise ValueError("roster members do not match frame member rows")

    enriched = frame.copy()
    enriched["observed_at"] = roster.observed_at.isoformat()
    enriched["as_of_panel_date"] = roster.as_of_panel_date.isoformat()
    enriched["rule_version"] = roster.rule_version
    context = CaptureContext(
        trading_date=roster.trading_date,
        run_id=roster.roster_id,
        dataset=CaptureDataset.SCAN,
        vendor="kiwoom",
        endpoint="trade-value-ranking",
        symbol=None,
        venue="KRX",
        session="regular",
        capture_reason=RESEARCH_ROSTER_REASON,
        cohort_id=None,
        scheduled_at=None,
    )
    frame_ref = store.publish_frame(enriched, context=context)
    status = CaptureStatus.PARTIAL if roster.truncated else CaptureStatus.COMPLETE
    entry = CoverageEntry(
        symbol=None,
        dataset=CaptureDataset.SCAN,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=status,
        rows=len(roster.members),
        first_event_time=None,
        last_event_time=None,
        reason="truncated" if roster.truncated else RESEARCH_ROSTER_REASON,
        raw_refs=tuple(raw_refs),
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=context,
        cohort=None,
        completed_at=completed_at,
        entries=(entry,),
        artifacts=(frame_ref, *raw_refs),
        status=status,
    )
    store.publish_manifest(manifest)
    return manifest


def read_research_roster(store: CaptureStore, snapshot_date: str, *, available_by: datetime) -> ResearchRoster:
    """Return the latest roster published on snapshot_date and completed by available_by.

    Args:
        store: Owner-local capture store.
        snapshot_date: ISO trading date.
        available_by: Aware cutoff of information available to the caller.

    Returns:
        The reconstructed roster.

    Raises:
        FileNotFoundError: No qualifying roster manifest exists.
        ValueError: The cutoff is naive or the persisted evidence is inconsistent.
    """
    if available_by.utcoffset() is None:
        raise ValueError("available_by must be an aware cutoff")
    qualifying = [
        item
        for item in store.read_manifests(snapshot_date)
        if item.context.capture_reason == RESEARCH_ROSTER_REASON
        and item.status in (CaptureStatus.COMPLETE, CaptureStatus.PARTIAL)
        and item.completed_at <= available_by
    ]
    if not qualifying:
        raise FileNotFoundError(f"no qualifying research roster: {snapshot_date!r}")
    chosen = max(qualifying, key=lambda item: (item.completed_at, item.context.run_id))
    frame_refs = [ref for ref in chosen.artifacts if ref.path.endswith(".parquet")]
    if len(frame_refs) != 1:
        raise ValueError("research roster manifest must carry exactly one frame artifact")
    frame = store.read_frame(frame_refs[0])
    required = set(RESEARCH_ROSTER_FRAME_COLUMNS) | set(_ROSTER_METADATA_COLUMNS)
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("research roster frame is incomplete")
    for column in _ROSTER_METADATA_COLUMNS:
        if frame[column].nunique() != 1:
            raise ValueError(f"research roster frame carries inconsistent {column!r}")
    members_rows = frame[frame["member"]].sort_values("member_order")
    dropped_rows = frame[~frame["member"]]
    if any(not isinstance(reason, str) for reason in dropped_rows["reason"]):
        raise ValueError("research roster non-member rows must carry a drop reason")
    return ResearchRoster(
        trading_date=chosen.context.trading_date,
        roster_id=chosen.context.run_id,
        rule_version=str(frame["rule_version"].iat[0]),
        observed_at=datetime.fromisoformat(str(frame["observed_at"].iat[0])),
        as_of_panel_date=date.fromisoformat(str(frame["as_of_panel_date"].iat[0])),
        members=tuple(members_rows["symbol"].astype(str).tolist()),
        dropped={str(row.symbol): str(row.reason) for row in dropped_rows.itertuples(index=False)},
        truncated=chosen.status == CaptureStatus.PARTIAL,
    )


def research_extras(
    store: CaptureStore,
    roster_date: str,
    *,
    available_by: datetime,
    exclude: Collection[str],
) -> tuple[str, ...]:
    """Return roster members not already covered by the primary population, in roster order.

    Args:
        store: Owner-local capture store.
        roster_date: ISO date whose roster is observed.
        available_by: Aware cutoff of information available to the caller.
        exclude: Symbols the primary population already covers.

    Returns:
        Extra symbols; empty when the roster is missing or unreadable.
    """
    try:
        roster = read_research_roster(store, roster_date, available_by=available_by)
    except FileNotFoundError:
        logger.warning("[DATA] stage=research_roster status=MISSING date=%s", roster_date)
        return ()
    except ValueError:
        logger.warning("[DATA] stage=research_roster status=UNREADABLE date=%s", roster_date)
        return ()
    covered = set(exclude)
    return tuple(member for member in roster.members if member not in covered)


def split_research_entries(
    entries: Sequence[CoverageEntry], research: frozenset[str]
) -> tuple[list[CoverageEntry], list[CoverageEntry]]:
    """Partition coverage entries into primary and research groups, preserving order.

    Args:
        entries: Coverage entries of one capture task.
        research: Symbols attributed to the research population.

    Returns:
        Primary entries (including symbol-less entries) and research entries.
    """
    primary: list[CoverageEntry] = []
    extras: list[CoverageEntry] = []
    for entry in entries:
        if entry.symbol is not None and entry.symbol in research:
            extras.append(entry)
        else:
            primary.append(entry)
    return primary, extras


def research_manifest(
    primary_context: CaptureContext,
    entries: Sequence[CoverageEntry],
    *,
    completed_at: datetime,
    force_partial: bool,
) -> CaptureManifest:
    """Build the cohort-less sibling manifest that carries research coverage.

    Args:
        primary_context: Context of the primary manifest this sibling accompanies.
        entries: Research coverage entries; must be non-empty.
        completed_at: Aware publication time.
        force_partial: True marks the manifest PARTIAL regardless of entry states.

    Returns:
        Unpublished manifest whose run id and capture reason carry RESEARCH_SUFFIX.

    Raises:
        ValueError: The entries are empty.
    """
    if not entries:
        raise ValueError("research manifest requires at least one entry")
    context = primary_context.model_copy(
        update={
            "run_id": primary_context.run_id + RESEARCH_SUFFIX,
            "capture_reason": primary_context.capture_reason + RESEARCH_SUFFIX,
            "cohort_id": None,
        }
    )
    complete = not force_partial and all(entry.status in GOOD_ENTRY_STATES for entry in entries)
    return CaptureManifest(
        schema_version=1,
        context=context,
        cohort=None,
        completed_at=completed_at,
        entries=tuple(entries),
        artifacts=tuple(ref for entry in entries for ref in entry.raw_refs),
        status=CaptureStatus.COMPLETE if complete else CaptureStatus.PARTIAL,
    )
