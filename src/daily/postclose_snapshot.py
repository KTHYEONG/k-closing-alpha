"""Post-close capture of trade strength and roster-extra investor estimates (research only, no orders)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd

from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.config.collection import CollectionSettings
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    SEOUL,
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
    SessionClock,
)
from src.data.capture_store import CaptureStore, resolve_capture_root
from src.data.research_roster import research_extras
from src.data.research_store import research_partition_path, write_research_partition
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.trading_calendar import is_kis_trading_day
from src.tools.alerts import redact_for_egress
from src.tools.run_outcome import (
    RUN_OUTCOME_DEGRADED,
    RUN_OUTCOME_OK,
    RUN_OUTCOME_SKIPPED,
    record_run_outcome,
)
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

POSTCLOSE_REASON: str = "postclose-snapshot"
STRENGTH_DATASET: str = "postclose_strength"
INVESTOR_DATASET: str = "investor_estimate_postclose"
PRE_AUCTION_CUTOFF_HHMMSS: str = "152000"

_JOB = "postclose_snapshot"
_TICK_ENDPOINT = "inquire-time-itemconclusion"
_INVESTOR_ENDPOINT = "investor-trend-estimate"
_MANIFEST_ENDPOINT = "postclose-snapshot"
_TICK_FLOOR_HHMMSS = "090000"
_TICK_END_HHMMSS = "153000"
_CLOSE_LAG = timedelta(minutes=1)
_COHORT = "cohort"
_RESEARCH = "research"
_STRENGTH_COLUMNS: tuple[str, ...] = (
    "symbol",
    "population",
    "strength_pre_auction",
    "pre_auction_hour",
    "strength_final",
    "final_hour",
    "n_ticks",
    "observed_at",
    "run_id",
)
_INVESTOR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "bsop_hour_gb",
    "frgn_fake_ntby_qty",
    "orgn_fake_ntby_qty",
    "sum_fake_ntby_qty",
    "observed_at",
    "run_id",
)
_HHMMSS_RE = re.compile(r"^\d{6}$")


def _now() -> datetime:
    return datetime.now(SEOUL)


def _at_hhmmss(trading_day: date, hhmmss: str) -> datetime:
    return datetime(
        trading_day.year,
        trading_day.month,
        trading_day.day,
        int(hhmmss[0:2]),
        int(hhmmss[2:4]),
        int(hhmmss[4:6]),
        tzinfo=SEOUL,
    )


def _int_or_none(raw: Any) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _float_or_none(raw: Any) -> float | None:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _hhmmss_or_none(raw: Any) -> str | None:
    text = str(raw or "").strip()
    return text if _HHMMSS_RE.fullmatch(text) else None


def _text_or_none(raw: Any) -> str | None:
    text = "" if raw is None else str(raw).strip()
    return text or None


def _redacted_reason(exc: BaseException) -> str:
    return redact_for_egress(type(exc).__name__) or "vendor_failure"


def _skip(snapshot_date: str, reason: str) -> None:
    record_run_outcome(_JOB, RUN_OUTCOME_SKIPPED, run_date=snapshot_date, reason=reason)
    logger.info("[DATA] stage=postclose_snapshot status=SKIP reason=%s date=%s", reason, snapshot_date)


def summarize_strength(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one trade-tick page into the as-of-15:20 and closing trade strengths.

    Args:
        rows: output2 rows of one tick page.

    Returns:
        Pre-auction and final strength with their hours, and the row count. Unparseable fields are None.
    """
    ticks: list[tuple[int, str | None, float | None]] = []
    for row in rows:
        volume = _int_or_none(row.get("acml_vol"))
        if volume is not None:
            ticks.append((volume, _hhmmss_or_none(row.get("stck_cntg_hour")), _float_or_none(row.get("tday_rltv"))))

    def _volume_then_hour(item: tuple[int, str | None, float | None]) -> tuple[int, str]:
        return item[0], item[1] or ""

    pre_candidates = [item for item in ticks if item[1] is not None and item[1] < PRE_AUCTION_CUTOFF_HHMMSS]
    pre = max(pre_candidates, key=_volume_then_hour, default=None)
    final = max(ticks, key=_volume_then_hour, default=None)
    return {
        "strength_pre_auction": pre[2] if pre is not None else None,
        "pre_auction_hour": pre[1] if pre is not None else None,
        "strength_final": final[2] if final is not None else None,
        "final_hour": final[1] if final is not None else None,
        "n_ticks": len(rows),
    }


def investor_rows(symbol: str, payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return one investor-estimate row per output2 entry, with unparseable quantities as None."""
    output = (payload or {}).get("output2") or []
    return [
        {
            "symbol": symbol,
            "bsop_hour_gb": _text_or_none(row.get("bsop_hour_gb")),
            "frgn_fake_ntby_qty": _int_or_none(row.get("frgn_fake_ntby_qty")),
            "orgn_fake_ntby_qty": _int_or_none(row.get("orgn_fake_ntby_qty")),
            "sum_fake_ntby_qty": _int_or_none(row.get("sum_fake_ntby_qty")),
        }
        for row in output
    ]


def _entry(
    symbol: str,
    dataset: CaptureDataset,
    status: CaptureStatus,
    reason: str,
    rows: int,
    refs: Sequence[ArtifactRef],
    started: datetime | None,
    finished: datetime | None,
) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol,
        dataset=dataset,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=status,
        rows=rows,
        first_event_time=started,
        last_event_time=finished,
        reason=reason,
        raw_refs=tuple(refs),
    )


def _persist_response(
    store: CaptureStore,
    *,
    trading_day: date,
    run_id: str,
    symbol: str,
    dataset: CaptureDataset,
    endpoint: str,
    started: datetime,
    received: datetime,
    payload: Any,
) -> ArtifactRef:
    context = CaptureContext(
        trading_date=trading_day,
        run_id=run_id,
        dataset=dataset,
        vendor="kis",
        endpoint=endpoint,
        symbol=symbol,
        venue="KRX",
        session="regular",
        capture_reason=POSTCLOSE_REASON,
        cohort_id=None,
        scheduled_at=None,
    )
    is_dict = isinstance(payload, dict)
    succeeded = is_dict and str(payload.get("rt_cd", "")) == "0"
    return store.append_response(
        CapturedResponse(
            context=context,
            request_started_at=started,
            received_at=received,
            payload=dict(payload) if is_dict else None,
            status=CaptureStatus.COMPLETE if succeeded else CaptureStatus.FAILED,
            source_timestamp=None,
            source_published_at=None,
            page_index=0,
            attempt_index=0,
            continuation={},
            error_type=None,
        )
    )


def _strength_row(
    symbol: str,
    population: str,
    summary: Mapping[str, Any] | None,
    observed_at: datetime | None,
    run_id: str,
) -> dict[str, Any]:
    fields: Mapping[str, Any] = summary or {
        "strength_pre_auction": None,
        "pre_auction_hour": None,
        "strength_final": None,
        "final_hour": None,
        "n_ticks": None,
    }
    return {"symbol": symbol, "population": population, **fields, "observed_at": observed_at, "run_id": run_id}


@dataclass(frozen=True)
class _TargetOutcome:
    strength: dict[str, Any]
    investor: list[dict[str, Any]]
    entries: tuple[CoverageEntry, ...]


async def run_postclose_snapshot(
    snapshot_date: str,
    *,
    profile: CollectionSettings,
    store: CaptureStore,
    client: Any,
    session_clock: SessionClock,
    now_fn: Callable[[], datetime] | None = None,
) -> CaptureManifest:
    """Capture post-close trade strength for the cohort and investor estimates for roster extras.

    Args:
        snapshot_date: ISO trading date; must equal session_clock.trading_date.
        profile: Snapshot switches, deadline and concurrency.
        store: Owner-local capture store receiving raw responses, partitions and the manifest.
        client: Client exposing get_intraday_trade_ticks and get_investor_trend_estimate.
        session_clock: Verified session of snapshot_date.
        now_fn: Aware clock; None selects the KST wall clock.

    Returns:
        The published coverage manifest.

    Raises:
        ValueError: Snapshot disabled, date mismatch, or observation before close plus one minute.
        FileExistsError: A snapshot partition for this date already exists; no request is sent.
        OSError: Raw or partition persistence failed.
    """
    if not profile.COLLECTION_POSTCLOSE_SNAPSHOT_ENABLED:
        raise ValueError("postclose snapshot disabled")
    if session_clock.trading_date.isoformat() != snapshot_date:
        raise ValueError("session clock date must equal snapshot_date")
    clock = now_fn or _now
    observed = clock()
    if observed < session_clock.close_at + _CLOSE_LAG:
        raise ValueError("too_early")
    for dataset in (STRENGTH_DATASET, INVESTOR_DATASET):
        if research_partition_path(dataset, snapshot_date).exists():
            raise FileExistsError(f"post-close snapshot already captured: {snapshot_date}")

    trading_day = session_clock.trading_date
    deadline = _at_hhmmss(trading_day, profile.COLLECTION_POSTCLOSE_SNAPSHOT_DEADLINE_HHMMSS)
    cohort_missing = False
    try:
        cohort = store.read_cohort(snapshot_date, available_by=observed).eligible_symbols
    except FileNotFoundError:
        logger.warning("[DATA] stage=postclose_snapshot cohort=MISSING date=%s", snapshot_date)
        cohort = ()
        cohort_missing = True
    extras = research_extras(
        store,
        snapshot_date,
        available_by=observed,
        exclude=set(cohort),
        enabled=profile.COLLECTION_RESEARCH_ROSTER_ENABLED,
    )
    targets = [(symbol, _COHORT) for symbol in cohort] + [(symbol, _RESEARCH) for symbol in extras]
    run_id = f"postclose-{snapshot_date}-{uuid.uuid4().hex[:8]}"
    bound = asyncio.Semaphore(profile.COLLECTION_POSTCLOSE_SNAPSHOT_CONCURRENCY)

    async def _tick_leg(
        session: aiohttp.ClientSession, symbol: str
    ) -> tuple[CoverageEntry, dict[str, Any] | None, datetime | None]:
        if clock() >= deadline:
            return _entry(symbol, CaptureDataset.TRADE_TICKS, CaptureStatus.PARTIAL, "deadline", 0, (), None, None), None, None
        started = clock()
        try:
            payload = await client.get_intraday_trade_ticks(
                session,
                symbol,
                floor_hour=_TICK_FLOOR_HHMMSS,
                end_hour=_TICK_END_HHMMSS,
                market_div_code="J",
                max_pages=1,
            )
        except Exception as exc:
            return (
                _entry(symbol, CaptureDataset.TRADE_TICKS, CaptureStatus.FAILED, _redacted_reason(exc), 0, (), started, clock()),
                None,
                None,
            )
        received = clock()
        ref = _persist_response(
            store,
            trading_day=trading_day,
            run_id=run_id,
            symbol=symbol,
            dataset=CaptureDataset.TRADE_TICKS,
            endpoint=_TICK_ENDPOINT,
            started=started,
            received=received,
            payload=payload,
        )
        if str(payload.get("rt_cd", "")) != "0":
            entry = _entry(symbol, CaptureDataset.TRADE_TICKS, CaptureStatus.FAILED, "vendor_failure", 0, (ref,), started, received)
            return entry, None, received
        rows = list(payload.get("output2") or [])
        summary = summarize_strength(rows)
        if not rows:
            entry = _entry(symbol, CaptureDataset.TRADE_TICKS, CaptureStatus.NO_TRADES, "no_ticks", 0, (ref,), started, received)
        else:
            entry = _entry(
                symbol, CaptureDataset.TRADE_TICKS, CaptureStatus.COMPLETE, POSTCLOSE_REASON, len(rows), (ref,), started, received
            )
        return entry, summary, received

    async def _investor_leg(
        session: aiohttp.ClientSession, symbol: str
    ) -> tuple[CoverageEntry, list[dict[str, Any]]]:
        dataset = CaptureDataset.INVESTOR_ESTIMATE
        if clock() >= deadline:
            return _entry(symbol, dataset, CaptureStatus.PARTIAL, "deadline", 0, (), None, None), []
        started = clock()
        try:
            payload = await client.get_investor_trend_estimate(session, symbol)
        except Exception as exc:
            return _entry(symbol, dataset, CaptureStatus.FAILED, _redacted_reason(exc), 0, (), started, clock()), []
        received = clock()
        ref = _persist_response(
            store,
            trading_day=trading_day,
            run_id=run_id,
            symbol=symbol,
            dataset=dataset,
            endpoint=_INVESTOR_ENDPOINT,
            started=started,
            received=received,
            payload=payload,
        )
        if str(payload.get("rt_cd", "")) != "0":
            return _entry(symbol, dataset, CaptureStatus.FAILED, "vendor_failure", 0, (ref,), started, received), []
        rows = [{**row, "observed_at": received, "run_id": run_id} for row in investor_rows(symbol, payload)]
        if not rows:
            return _entry(symbol, dataset, CaptureStatus.NO_TRADES, "no_ticks", 0, (ref,), started, received), []
        return _entry(symbol, dataset, CaptureStatus.COMPLETE, POSTCLOSE_REASON, len(rows), (ref,), started, received), rows

    async def _capture(session: aiohttp.ClientSession, symbol: str, population: str) -> _TargetOutcome:
        async with bound:
            tick_entry, summary, observed_at = await _tick_leg(session, symbol)
            entries = [tick_entry]
            investor: list[dict[str, Any]] = []
            if population == _RESEARCH:
                investor_entry, investor = await _investor_leg(session, symbol)
                entries.append(investor_entry)
        return _TargetOutcome(
            strength=_strength_row(symbol, population, summary, observed_at, run_id),
            investor=investor,
            entries=tuple(entries),
        )

    async with aiohttp.ClientSession() as session:
        outcomes = await asyncio.gather(*(_capture(session, symbol, population) for symbol, population in targets))

    strength_frame = pd.DataFrame([outcome.strength for outcome in outcomes], columns=list(_STRENGTH_COLUMNS))
    investor_frame = pd.DataFrame(
        [row for outcome in outcomes for row in outcome.investor], columns=list(_INVESTOR_COLUMNS)
    )
    write_research_partition(strength_frame, STRENGTH_DATASET, snapshot_date, overwrite=False)
    write_research_partition(investor_frame, INVESTOR_DATASET, snapshot_date, overwrite=False)

    entries = tuple(entry for outcome in outcomes for entry in outcome.entries)
    status = (
        CaptureStatus.COMPLETE
        if not cohort_missing and all(entry.status in GOOD_ENTRY_STATES for entry in entries)
        else CaptureStatus.PARTIAL
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=trading_day,
            run_id=run_id,
            dataset=CaptureDataset.TRADE_TICKS,
            vendor="kis",
            endpoint=_MANIFEST_ENDPOINT,
            symbol=None,
            venue="KRX",
            session="regular",
            capture_reason=POSTCLOSE_REASON,
            cohort_id=None,
            scheduled_at=None,
        ),
        cohort=None,
        completed_at=clock(),
        entries=entries,
        artifacts=tuple(ref for entry in entries for ref in entry.raw_refs),
        status=status,
    )
    store.publish_manifest(manifest)
    logger.info(
        "[DATA] stage=postclose_snapshot status=%s date=%s targets=%d extras=%d entries=%d",
        status.value,
        snapshot_date,
        len(targets),
        len(extras),
        len(entries),
    )
    return manifest


async def _execute(
    snapshot_date: str, profile: CollectionSettings, client: KisApiClient, session_clock: SessionClock
) -> CaptureManifest | None:
    async with client.create_session() as session:
        await client.ensure_token(session)
        if not await is_kis_trading_day(client, session, snapshot_date):
            return None
    return await run_postclose_snapshot(
        snapshot_date,
        profile=profile,
        store=CaptureStore(resolve_capture_root(profile)),
        client=client,
        session_clock=session_clock,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python -m src.daily.postclose_snapshot [--date YYYY-MM-DD]``."""
    parser = argparse.ArgumentParser(description="KCA post-close trade-strength snapshot (research, no orders)")
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    snapshot_date = args.date or _now().date().isoformat()
    if not profile.COLLECTION_POSTCLOSE_SNAPSHOT_ENABLED:
        _skip(snapshot_date, "disabled")
        return
    session_day = resolve_session_day(date.fromisoformat(snapshot_date), overrides=profile.COLLECTION_SESSION_OVERRIDES)
    if session_day.kind is SessionKind.CLOSED:
        _skip(snapshot_date, "non_trading_day")
        return
    if session_day.kind is SessionKind.UNKNOWN or session_day.clock is None:
        _skip(snapshot_date, "session_unknown")
        return
    try:
        client = KisApiClient(**kis_data_client_kwargs())
        manifest = asyncio.run(_execute(snapshot_date, profile, client, session_day.clock))
    except FileExistsError:
        _skip(snapshot_date, "already_captured")
        return
    except (RuntimeError, ValueError) as exc:
        reason = redact_for_egress(str(exc)) or type(exc).__name__
        logger.warning("[DATA] stage=postclose_snapshot status=FAILED reason=%s date=%s", reason, snapshot_date)
        record_run_outcome(_JOB, RUN_OUTCOME_DEGRADED, run_date=snapshot_date, reason=reason)
        return
    if manifest is None:
        _skip(snapshot_date, "non_trading_day")
        return
    metrics = {
        "entries": len(manifest.entries),
        "complete": sum(1 for entry in manifest.entries if entry.status == CaptureStatus.COMPLETE),
        "partial": sum(1 for entry in manifest.entries if entry.status == CaptureStatus.PARTIAL),
        "failed": sum(1 for entry in manifest.entries if entry.status == CaptureStatus.FAILED),
    }
    if manifest.status == CaptureStatus.COMPLETE:
        record_run_outcome(_JOB, RUN_OUTCOME_OK, run_date=snapshot_date, metrics=metrics)
    else:
        record_run_outcome(_JOB, RUN_OUTCOME_DEGRADED, run_date=snapshot_date, reason="partial", metrics=metrics)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
