"""Research roster capture at the closing-auction boundary (research only, no orders)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from src import settings
from src.api.kiwoom.client import KiwoomApiClient
from src.config.collection import CollectionSettings
from src.daily.collect import load_eligible_codes
from src.daily.security_classification import load_security_classification
from src.data.capture_contracts import (
    SEOUL,
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureStatus,
    SessionClock,
)
from src.data.capture_store import CaptureStore, resolve_capture_root
from src.data.research_roster import (
    RESEARCH_ROSTER_REASON,
    RESEARCH_ROSTER_RULE_VERSION,
    ResearchRoster,
    build_roster_frame,
    parse_trade_value_rows,
    publish_research_roster,
    select_research_roster,
)
from src.data.session_calendar import SessionKind, resolve_session_day
from src.strategy.contract import PRODUCTION_STRATEGY
from src.strategy.store import StrategyStoreError, require_production_store
from src.tools.alerts import redact_for_egress
from src.tools.run_outcome import (
    RUN_OUTCOME_DEGRADED,
    RUN_OUTCOME_OK,
    RUN_OUTCOME_SKIPPED,
    record_run_outcome,
)
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

_JOB = "research_roster"
_KRW_PER_100M = 100_000_000
_CLOSING_AUCTION_LEAD = timedelta(minutes=10)
"""Regular trading ends where the closing auction starts; no trade prints occur after that point."""
_PANEL_LOOKBACK = timedelta(days=14)


def _now() -> datetime:
    return datetime.now(SEOUL)


def _page_observer(
    store: CaptureStore, trading_date: date, roster_id: str, raw_refs: list[ArtifactRef]
) -> Callable[..., None]:
    context = CaptureContext(
        trading_date=trading_date,
        run_id=roster_id,
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

    def _observe(
        payload: Any,
        metadata: Mapping[str, str],
        started: datetime,
        received: datetime,
        page_index: int,
        attempt_index: int,
    ) -> None:
        raw_refs.append(
            store.append_response(
                CapturedResponse(
                    context=context,
                    request_started_at=started,
                    received_at=received,
                    payload=dict(payload) if isinstance(payload, dict) else None,
                    status=CaptureStatus.COMPLETE if isinstance(payload, dict) else CaptureStatus.FAILED,
                    source_timestamp=None,
                    source_published_at=None,
                    page_index=page_index,
                    attempt_index=attempt_index,
                    continuation={"cont-yn": metadata["cont-yn"], "next-key": metadata["next-key"]},
                    error_type=None,
                )
            )
        )

    return _observe


async def run_research_roster(
    snapshot_date: str,
    *,
    profile: CollectionSettings,
    store: CaptureStore,
    kiwoom_client: Any,
    session_clock: SessionClock,
    listed: frozenset[str],
    screenable: frozenset[str] | None,
    shares_by_symbol: Mapping[str, float],
    as_of_panel_date: date,
    now_fn: Callable[[], datetime] | None = None,
) -> ResearchRoster:
    """Capture the direction-free liquid roster from the 15:20 trade-value ranking.

    Args:
        snapshot_date: ISO trading date; must equal session_clock.trading_date.
        profile: Collection limits and roster thresholds.
        store: Owner-local capture store receiving raw pages and the roster evidence.
        kiwoom_client: Client exposing get_trade_value_ranking.
        session_clock: Verified session of snapshot_date.
        listed: Symbols listed on the prior panel date.
        screenable: Symbols passing the screenable class filter, or None when the filter is off.
        shares_by_symbol: Shares outstanding per symbol from the prior panel date.
        as_of_panel_date: Prior panel date the eligibility inputs describe.
        now_fn: Aware clock; None selects the KST wall clock.

    Returns:
        The published roster.

    Raises:
        ValueError: Roster disabled, date mismatch, or observation outside the pre-auction window.
        RuntimeError: The ranking is unavailable and was not truncated; nothing is published.
        RawCaptureError: Raw page persistence failed.
    """
    if not profile.COLLECTION_RESEARCH_ROSTER_ENABLED:
        raise ValueError("research roster disabled")
    if session_clock.trading_date.isoformat() != snapshot_date:
        raise ValueError("session clock date must equal snapshot_date")
    clock = now_fn or _now
    observed = clock()
    if not session_clock.close_at - _CLOSING_AUCTION_LEAD <= observed < session_clock.close_at:
        raise ValueError("outside_pit_window")

    roster_id = f"research-roster-{snapshot_date}-{uuid.uuid4().hex[:8]}"
    raw_refs: list[ArtifactRef] = []
    observer = _page_observer(store, session_clock.trading_date, roster_id, raw_refs)
    async with aiohttp.ClientSession() as session:
        payload = await kiwoom_client.get_trade_value_ranking(
            session,
            max_pages=profile.COLLECTION_RESEARCH_ROSTER_MAX_PAGES,
            min_trade_value_krw=int(profile.COLLECTION_RESEARCH_ROSTER_MIN_TRADE_VALUE_100M * _KRW_PER_100M),
            on_page=observer,
        )
    truncated = bool(payload.get("truncated", False))
    if payload["rt_cd"] != "0" and not truncated:
        raise RuntimeError("ranking_unavailable")

    candidates, unparseable = parse_trade_value_rows(payload["output"])
    members, rejected, est_mc = select_research_roster(
        candidates,
        listed=listed,
        screenable=screenable,
        shares_by_symbol=shares_by_symbol,
        min_trade_value_100m=profile.COLLECTION_RESEARCH_ROSTER_MIN_TRADE_VALUE_100M,
        min_market_cap_100m=profile.COLLECTION_RESEARCH_ROSTER_MIN_MARKET_CAP_100M,
        max_members=profile.COLLECTION_RESEARCH_ROSTER_MAX_MEMBERS,
    )
    dropped = {**unparseable, **rejected}
    roster = ResearchRoster(
        trading_date=session_clock.trading_date,
        roster_id=roster_id,
        rule_version=RESEARCH_ROSTER_RULE_VERSION,
        observed_at=observed,
        as_of_panel_date=as_of_panel_date,
        members=members,
        dropped=dropped,
        truncated=truncated,
    )
    frame = build_roster_frame(candidates, members, dropped, est_mc)
    publish_research_roster(store, roster, frame, raw_refs=raw_refs, completed_at=clock())
    logger.info(
        "[DATA] stage=research_roster status=%s roster_id=%s members=%d dropped=%d pages=%d",
        "TRUNCATED" if truncated else "OK",
        roster_id,
        len(members),
        len(dropped),
        len(raw_refs),
    )
    return roster


def load_prev_day_shares(
    decision_date: pd.Timestamp,
    *,
    prev_trading_day: pd.Timestamp,
    path: str | Path | None = None,
) -> dict[str, float]:
    """Return shares outstanding implied by the prior trading day's market cap and raw close.

    Args:
        decision_date: Decision date; prev_trading_day must precede it.
        prev_trading_day: Prior trading day whose price_history rows are read.
        path: Parquet path; None selects settings.PRICE_HISTORY_PARQUET_PATH.

    Returns:
        Shares per symbol for rows with positive market cap and raw close.

    Raises:
        ValueError: prev_trading_day is not before decision_date.
        FileNotFoundError: The price_history parquet does not exist.
    """
    prev = pd.Timestamp(prev_trading_day).normalize()
    if prev >= pd.Timestamp(decision_date).normalize():
        raise ValueError(f"prev_trading_day {prev.date()} must be before decision_date")
    src_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if path is None else path)
    if not src_path.exists():
        raise FileNotFoundError(f"price_history not found: {src_path}")
    rows = pd.read_parquet(
        src_path,
        columns=["date", "symbol", "market_cap_100m", "close_raw"],
        filters=[("date", "==", prev)],
    )
    rows = rows[(rows["close_raw"] > 0) & (rows["market_cap_100m"] > 0)]
    shares = rows["market_cap_100m"] * _KRW_PER_100M / rows["close_raw"]
    return {str(symbol): float(value) for symbol, value in zip(rows["symbol"], shares, strict=True)}


def latest_panel_date_before(decision_date: pd.Timestamp, *, path: str | Path | None = None) -> pd.Timestamp:
    """Return the latest price_history date strictly before decision_date within a 14-day lookback.

    Args:
        decision_date: Decision date.
        path: Parquet path; None selects settings.PRICE_HISTORY_PARQUET_PATH.

    Returns:
        Normalized panel date.

    Raises:
        FileNotFoundError: The price_history parquet does not exist.
        ValueError: No panel rows fall inside the lookback window.
    """
    panel_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if path is None else path)
    if not panel_path.exists():
        raise FileNotFoundError(f"price_history not found: {panel_path}")
    day = pd.Timestamp(decision_date).normalize()
    rows = pd.read_parquet(
        panel_path,
        columns=["date"],
        filters=[("date", ">=", day - _PANEL_LOOKBACK), ("date", "<", day)],
    )
    if rows.empty:
        raise ValueError("stale price_history")
    return pd.Timestamp(pd.to_datetime(rows["date"]).max()).normalize()


def _skip(snapshot_date: str, reason: str) -> None:
    record_run_outcome(_JOB, RUN_OUTCOME_SKIPPED, run_date=snapshot_date, reason=reason)
    logger.info("[DATA] stage=research_roster status=SKIP reason=%s date=%s", reason, snapshot_date)


def _pages_of(store: CaptureStore, snapshot_date: str, roster_id: str) -> int:
    (manifest,) = (item for item in store.read_manifests(snapshot_date) if item.context.run_id == roster_id)
    return len(manifest.entries[0].raw_refs)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python -m src.daily.roster_capture [--date YYYY-MM-DD]``."""
    parser = argparse.ArgumentParser(description="KCA research roster capture at the closing auction (no orders)")
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    try:
        require_production_store()
    except StrategyStoreError as exc:
        logger.error("[SYS] stage=%s status=FAIL reason=%s", "roster_capture", exc)
        raise SystemExit(1) from exc
    profile = CollectionSettings()
    snapshot_date = args.date or _now().date().isoformat()
    trading_day = date.fromisoformat(snapshot_date)
    if not profile.COLLECTION_RESEARCH_ROSTER_ENABLED:
        _skip(snapshot_date, "disabled")
        return
    session_day = resolve_session_day(trading_day, overrides=profile.COLLECTION_SESSION_OVERRIDES)
    if session_day.kind is SessionKind.CLOSED:
        _skip(snapshot_date, "non_trading_day")
        return
    if session_day.kind is SessionKind.UNKNOWN or session_day.clock is None:
        _skip(snapshot_date, "session_unknown")
        return
    store = CaptureStore(resolve_capture_root(profile))
    try:
        decision_day = pd.Timestamp(trading_day)
        prev = latest_panel_date_before(decision_day)
        listed = load_eligible_codes(decision_day, prev_trading_day=prev)
        screenable = (
            load_security_classification(decision_day, prev_trading_day=prev)
            if PRODUCTION_STRATEGY.universe.exclude_non_screenable_class
            else None
        )
        shares = load_prev_day_shares(decision_day, prev_trading_day=prev)
        if not settings.KIWOOM_APP_KEY:
            _skip(snapshot_date, "no_kiwoom_credentials")
            return
        roster = asyncio.run(
            run_research_roster(
                snapshot_date,
                profile=profile,
                store=store,
                kiwoom_client=KiwoomApiClient(),
                session_clock=session_day.clock,
                listed=listed,
                screenable=screenable,
                shares_by_symbol=shares,
                as_of_panel_date=prev.date(),
            )
        )
    except (RuntimeError, ValueError) as exc:
        reason = redact_for_egress(str(exc)) or type(exc).__name__
        logger.warning("[DATA] stage=research_roster status=FAILED reason=%s date=%s", reason, snapshot_date)
        record_run_outcome(_JOB, RUN_OUTCOME_DEGRADED, run_date=snapshot_date, reason=reason)
        return
    metrics = {
        "members": len(roster.members),
        "dropped": len(roster.dropped),
        "pages": _pages_of(store, snapshot_date, roster.roster_id),
    }
    if roster.truncated:
        record_run_outcome(_JOB, RUN_OUTCOME_DEGRADED, run_date=snapshot_date, reason="truncated", metrics=metrics)
    else:
        record_run_outcome(_JOB, RUN_OUTCOME_OK, run_date=snapshot_date, metrics=metrics)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
