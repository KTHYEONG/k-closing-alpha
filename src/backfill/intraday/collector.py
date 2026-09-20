"""당일 정규세션/NXT 애프터마켓 1분봉 수집기 (워치리스트 스코프)."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd

from src import settings
from src.api.ls.client import LsApiClient  # noqa: F401 - wiring per spec
from src.config.collection import CollectionSettings
from src.config.market_session import (
    INTRADAY_SESSION_REGULAR,
    KRX_AFTERMARKET_HOUR_CEIL,
    KRX_AFTERMARKET_HOUR_FLOOR,
    KRX_AFTERMARKET_START_DATE,
    KRX_CLOSE_MARKET_DIV_CODE,
    KRX_REGULAR_HOUR_CEIL,
    KRX_REGULAR_HOUR_FLOOR,
    NXT_AFTERMARKET_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_FLOOR,
    NXT_MARKET_DIV_CODE,
    NXT_PREMARKET_HOUR_CEIL,
    NXT_PREMARKET_HOUR_FLOOR,
)
from src.data.capture_contracts import (
    SEOUL,
    BrokerPayload,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    ChartBudget,
    CoverageEntry,
    PageObserver,
    SymbolObserver,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame

logger = logging.getLogger(__name__)

DEFAULT_LS_TICK_MAX_PAGES: int = 30

_EXHAUSTED_TERMINALS = frozenset({"exhausted", "crossed_target_date"})
_GOOD_ENTRY_STATES = frozenset({CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE})
_NONCERTIFIED = frozenset({CaptureStatus.PARTIAL, CaptureStatus.FAILED, CaptureStatus.UNKNOWN})
_ERROR_RE = re.compile(r"[^A-Za-z0-9_]+")


def _canonical_kis_bars(rows: list[dict], snapshot_date: str, code: str) -> pd.DataFrame:
    vendor = "kis"
    try:
        return normalize_bar_frame(pd.DataFrame(rows), vendor, snapshot_date, code)
    except Exception as e:
        logger.warning("[DATA] KIS bar normalize failed code=%s: %s", code, e)
        return pd.DataFrame()


def _canonical_kis_ticks(rows: list[dict], snapshot_date: str, code: str) -> pd.DataFrame:
    vendor = "kis"
    prepared = [dict(r) for r in rows]
    if prepared and not any(k in prepared[0] for k in ("cnqn", "cntg_vol")) and "acml_vol" in prepared[0]:
        # acml_vol(누적)만 있는 응답은 정렬 후 1차 차분으로 봉당 체결량(cnqn)을 합성한다.
        # 파싱 실패는 개별 값만 0으로 클램프하고 계속 진행한다 (전체 배치를 버리지 않음).
        ordered = sorted(prepared, key=lambda r: (str(r.get("stck_cntg_hour", "")), str(r.get("acml_vol", "0"))))
        prev = 0
        for rec in ordered:
            try:
                cur = int(str(rec.get("acml_vol", "0")).strip() or "0")
            except ValueError:
                cur = 0
            rec["cnqn"] = str(max(cur - prev, 0))
            prev = cur
        prepared = ordered
    try:
        return normalize_tick_frame(pd.DataFrame(prepared), vendor, snapshot_date, code)
    except Exception as e:
        logger.warning("[DATA] KIS tick normalize failed code=%s: %s", code, e)
        return pd.DataFrame()


def _seoul_now() -> datetime:
    return datetime.now(SEOUL)


def _parse_snapshot_date(snapshot_date: str) -> date:
    try:
        return date.fromisoformat(str(snapshot_date))
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None


def _is_past_date(snapshot_date: str) -> bool:
    return str(snapshot_date) < _seoul_now().date().isoformat()


def _resolve_profile(profile: CollectionSettings | None) -> CollectionSettings:
    return profile if profile is not None else CollectionSettings()


def _capture_root(profile: CollectionSettings) -> Any:
    if profile.COLLECTION_ROOT is not None:
        return profile.COLLECTION_ROOT
    return settings.HISTORY_DIR / "capture"


def _resolve_store(capture_store: CaptureStore | None, profile: CollectionSettings) -> CaptureStore:
    if capture_store is not None:
        return capture_store
    return CaptureStore(_capture_root(profile))


def _new_run_id(run_id: str | None, snapshot_date: str, dataset: CaptureDataset) -> str:
    if run_id is not None:
        if not str(run_id).strip():
            raise ValueError("run_id must be nonempty")
        return str(run_id)
    return f"intraday-{snapshot_date}-{dataset.value.lower()}-{uuid.uuid4().hex[:8]}"


def _redacted_error(exc: BaseException) -> str:
    name = type(exc).__name__ or "error"
    cleaned = _ERROR_RE.sub("-", name).strip("-")
    return cleaned or "error"


def _venue_for(*, vendor: str, endpoint: str, market_div_code: str | None, profile: CollectionSettings) -> str:
    routes = profile.COLLECTION_VERIFIED_CHART_ROUTES or {}
    mapped = routes.get(f"{vendor}:{endpoint}")
    if mapped is not None:
        return str(mapped)
    if vendor == "kis" and market_div_code == KRX_CLOSE_MARKET_DIV_CODE:
        return "KRX"
    return "UNKNOWN"


def _capture_context(
    *,
    trading_day: date,
    run_id: str,
    dataset: CaptureDataset,
    vendor: str,
    endpoint: str,
    symbol: str | None,
    venue: str,
    session: str,
) -> CaptureContext:
    return CaptureContext(
        trading_date=trading_day,
        run_id=run_id,
        dataset=dataset,
        vendor=vendor,
        endpoint=endpoint,
        symbol=symbol,
        venue=venue,
        session=session,
        capture_reason=f"intraday-{dataset.value.lower()}",
        cohort_id=None,
        scheduled_at=None,
    )


def _pending_entry(symbol: str, dataset: CaptureDataset, venue: str, session: str) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol,
        dataset=dataset,
        venue=venue,
        session=session,
        scheduled_at=None,
        status=CaptureStatus.PENDING,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason="acquisition pending",
        raw_refs=(),
    )


def _terminal_entry(
    *,
    symbol: str,
    dataset: CaptureDataset,
    venue: str,
    session: str,
    status: CaptureStatus,
    rows: int,
    reason: str,
    refs: list[Any],
) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol,
        dataset=dataset,
        venue=venue,
        session=session,
        scheduled_at=None,
        status=status,
        rows=int(rows),
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=tuple(refs),
    )


def _observe_pages(
    store: CaptureStore, context: CaptureContext, refs: list[Any], attempt: int
) -> PageObserver:
    def _on_page(
        payload: BrokerPayload | None,
        metadata: Mapping[str, str],
        started: datetime,
        received: datetime,
        page_index: int,
        _retry: int,
    ) -> None:
        continuation = {str(key): str(item) for key, item in dict(metadata).items()}
        ok = payload is not None
        response = CapturedResponse(
            context=context,
            request_started_at=started,
            received_at=received,
            payload=dict(payload) if payload is not None else None,
            status=CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED,
            source_timestamp=None,
            source_published_at=None,
            page_index=int(page_index),
            attempt_index=int(attempt),
            continuation=continuation,
            error_type=None,
        )
        refs.append(store.append_response(response))

    return _on_page


def _stage_fragment(
    store: CaptureStore,
    *,
    run_id: str,
    trading_day: date,
    dataset: CaptureDataset,
    symbol: str,
    session: str,
    frame: pd.DataFrame,
    reason: str,
) -> Any:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", reason).strip("-") or "fragment"
    context = _capture_context(
        trading_day=trading_day,
        run_id=f"{run_id}-{slug}-{uuid.uuid4().hex[:6]}",
        dataset=dataset,
        vendor="owner-local",
        endpoint="staged-fragment",
        symbol=symbol,
        venue="UNKNOWN",
        session=session,
    )
    ref = store.publish_frame(frame, context=context)
    logger.info("[DATA] stage=intraday_fragment symbol=%s dataset=%s reason=%s rows=%d", symbol, dataset.value, reason, len(frame))
    return ref


def _publish_pending_manifest(
    store: CaptureStore,
    *,
    trading_day: date,
    run_id: str,
    dataset: CaptureDataset,
    vendor: str,
    endpoint: str,
    session: str,
    symbols: list[str],
) -> None:
    entries = tuple(_pending_entry(code, dataset, "UNKNOWN", session) for code in symbols)
    manifest = CaptureManifest(
        schema_version=1,
        context=_capture_context(
            trading_day=trading_day, run_id=run_id, dataset=dataset, vendor=vendor,
            endpoint=endpoint, symbol=None, venue="UNKNOWN", session=session,
        ),
        cohort=None,
        completed_at=_seoul_now(),
        entries=entries,
        artifacts=(),
        status=CaptureStatus.PENDING,
    )
    try:
        store.publish_manifest(manifest)
    except ValueError as exc:
        # PENDING 매니페스트는 completed_at이 매번 달라 재실행 시 동일 run_id 경로와
        # 내용이 반드시 어긋난다 -- 실측: 2026-09-18 재시도 실행이 immutable identity
        # 충돌로 아카이브 전체를 실패시킴. 완료 인증(COMPLETE/PARTIAL)이 아닌 진행 중
        # 체크포인트일 뿐이라 발행 실패해도 실제 수집 작업은 계속 진행한다.
        logger.warning("[DATA] stage=intraday_archive status=DEGRADED reason=pending_manifest_conflict run_id=%s detail=%s", run_id, exc)


def _event_key(row: Mapping[str, Any], vendor: str) -> tuple[str, str]:
    if vendor == "ls":
        return (str(row.get("date", "") or ""), str(row.get("time", "") or ""))
    if vendor == "kiwoom":
        tm = str(row.get("cntr_tm", "") or "")
        if len(tm) >= 14:
            return (tm[:8], tm[8:14])
        return ("", "")
    return ("", str(row.get("stck_cntg_hour", "") or ""))


def _split_session_window(
    rows: list[dict], ymd: str, floor: str, ceil: str, vendor: str
) -> tuple[list[dict], list[dict]]:
    regular: list[dict] = []
    other: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        day, hms = _event_key(row, vendor)
        if day in ("", ymd) and len(hms) == 6 and hms.isdigit() and floor <= hms <= ceil:
            regular.append(dict(row))
        else:
            other.append(dict(row))
    return regular, other


def _empty_bar_frame(snapshot_date: str) -> pd.DataFrame:
    return normalize_bar_frame(pd.DataFrame(), "ls", snapshot_date, "000000")


def _empty_tick_frame(snapshot_date: str) -> pd.DataFrame:
    return normalize_tick_frame(pd.DataFrame(), "ls", snapshot_date, "000000")


def _chart_budget(profile: CollectionSettings) -> ChartBudget:
    return ChartBudget(
        max_pages=int(profile.COLLECTION_CHART_MAX_PAGES),
        deadline=None,
        request_timeout_seconds=float(profile.COLLECTION_REQUEST_TIMEOUT_SECONDS),
    )


def _repair_budget(profile: CollectionSettings) -> ChartBudget:
    return ChartBudget(
        max_pages=int(profile.COLLECTION_TICK_REPAIR_MAX_PAGES),
        deadline=None,
        request_timeout_seconds=float(profile.COLLECTION_REQUEST_TIMEOUT_SECONDS),
    )


async def _kis_bar_attempt(
    *,
    client: Any,
    session: Any,
    code: str,
    snapshot_date: str,
    trading_day: date,
    ymd: str,
    bar_interval_minutes: int,
    floor: str,
    ceil: str,
    market_div_code: str,
    session_tag: str,
    dataset: CaptureDataset,
    store: CaptureStore,
    run_id: str,
    profile: CollectionSettings,
    attempt: int,
) -> tuple[pd.DataFrame, CoverageEntry]:
    refs: list[Any] = []
    venue = _venue_for(vendor="kis", endpoint="kis-chart", market_div_code=market_div_code, profile=profile)
    context = _capture_context(
        trading_day=trading_day, run_id=run_id, dataset=dataset, vendor="kis",
        endpoint="kis-chart", symbol=code, venue=venue, session=session_tag,
    )
    started = _seoul_now()
    try:
        if _is_past_date(snapshot_date):
            payload = await client.get_historical_minute_chart(
                session, code, ymd, bar_interval_minutes=bar_interval_minutes,
                end_hour=ceil, floor_hour=floor, market_div_code=market_div_code,
            )
        else:
            payload = await client.get_intraday_minute_chart(
                session, code, bar_interval_minutes=bar_interval_minutes,
                end_hour=ceil, floor_hour=floor, market_div_code=market_div_code,
            )
    except Exception as e:
        logger.warning("[DATA] stage=bars symbol=%s status=FAILED reason=transport:%s", code, _redacted_error(e))
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.FAILED, rows=0, reason=f"transport:{_redacted_error(e)}", refs=refs,
        )
    received = _seoul_now()
    refs.append(store.append_response(_page_response(context, payload, started, received, attempt)))
    if not isinstance(payload, dict) or payload.get("rt_cd") != "0":
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.FAILED, rows=0, reason="vendor_failure", refs=refs,
        )
    rows = [dict(r) for r in (payload.get("output2") or []) if isinstance(r, dict)]
    regular, other = _split_session_window(rows, ymd, floor, ceil, "kis")
    if other:
        frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=dataset,
                               symbol=code, session=session_tag,
                               frame=_canonical_kis_bars(other, snapshot_date, code), reason="out_of_window")
        refs.append(frag)
    if not regular:
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.UNKNOWN, rows=0, reason="empty_without_proof", refs=refs,
        )
    try:
        frame = normalize_bar_frame(pd.DataFrame(regular), "kis", snapshot_date, code)
    except Exception as e:
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.FAILED, rows=0, reason=f"normalize:{_redacted_error(e)}", refs=refs,
        )
    if frame.empty:
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.UNKNOWN, rows=0, reason="empty_without_proof", refs=refs,
        )
    if venue == "UNKNOWN":
        frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=dataset,
                               symbol=code, session=session_tag, frame=frame, reason="uncertified_venue")
        refs.append(frag)
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.UNKNOWN, rows=0, reason="uncertified_venue", refs=refs,
        )
    return frame, _terminal_entry(
        symbol=code, dataset=dataset, venue=venue, session=session_tag,
        status=CaptureStatus.COMPLETE, rows=len(frame), reason=f"exhausted:regular={len(regular)}", refs=refs,
    )


def _page_response(context: CaptureContext, payload: Any, started: datetime, received: datetime, attempt: int) -> CapturedResponse:
    body = dict(payload) if isinstance(payload, dict) else None
    return CapturedResponse(
        context=context,
        request_started_at=started,
        received_at=received,
        payload=body,
        status=CaptureStatus.COMPLETE if body is not None else CaptureStatus.FAILED,
        source_timestamp=None,
        source_published_at=None,
        page_index=0,
        attempt_index=int(attempt),
        continuation={},
        error_type=None,
    )


async def _ls_bar_attempt(
    *,
    ls_client: Any,
    session: Any,
    code: str,
    snapshot_date: str,
    trading_day: date,
    ymd: str,
    floor: str,
    ceil: str,
    session_tag: str,
    dataset: CaptureDataset,
    store: CaptureStore,
    run_id: str,
    profile: CollectionSettings,
) -> tuple[pd.DataFrame, CoverageEntry]:
    refs: list[Any] = []
    venue = _venue_for(vendor="ls", endpoint="t8412", market_div_code=None, profile=profile)
    context = _capture_context(
        trading_day=trading_day, run_id=run_id, dataset=dataset, vendor="ls",
        endpoint="t8412", symbol=code, venue=venue, session=session_tag,
    )
    try:
        payload = await ls_client.get_minute_chart(
            session, code, snapshot_date, budget=_chart_budget(profile),
            on_page=_observe_pages(store, context, refs, 0),
        )
    except Exception as e:
        logger.warning("[DATA] stage=bars symbol=%s status=FAILED reason=transport:%s", code, _redacted_error(e))
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.FAILED, rows=0, reason=f"transport:{_redacted_error(e)}", refs=refs,
        )
    if not isinstance(payload, dict) or payload.get("rt_cd") != "0":
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag,
            status=CaptureStatus.FAILED, rows=0, reason="vendor_failure", refs=refs,
        )
    terminal = str(payload.get("termination_reason", "") or "")
    rows = [dict(r) for r in (payload.get("output2") or []) if isinstance(r, dict)]
    regular, other = _split_session_window(rows, ymd, floor, ceil, "ls")
    if other:
        frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=dataset,
                               symbol=code, session=session_tag,
                               frame=_safe_normalize_bars("ls", other, snapshot_date, code), reason="out_of_window")
        refs.append(frag)
    if terminal in _EXHAUSTED_TERMINALS and venue != "UNKNOWN":
        try:
            frame = normalize_bar_frame(pd.DataFrame(regular), "ls", snapshot_date, code)
        except Exception as e:
            return _empty_bar_frame(snapshot_date), _terminal_entry(
                symbol=code, dataset=dataset, venue=venue, session=session_tag,
                status=CaptureStatus.FAILED, rows=0, reason=f"normalize:{_redacted_error(e)}", refs=refs,
            )
        if frame.empty:
            return _empty_bar_frame(snapshot_date), _terminal_entry(
                symbol=code, dataset=dataset, venue=venue, session=session_tag,
                status=CaptureStatus.UNKNOWN, rows=0, reason="empty_without_proof", refs=refs,
            )
        return frame, _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=session_tag, status=CaptureStatus.COMPLETE,
            rows=len(frame), reason=f"{terminal}:regular={len(regular)}", refs=refs,
        )
    if regular:
        frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=dataset,
                               symbol=code, session=session_tag,
                               frame=_safe_normalize_bars("ls", regular, snapshot_date, code), reason=f"incomplete:{terminal}")
        refs.append(frag)
    return _empty_bar_frame(snapshot_date), _terminal_entry(
        symbol=code, dataset=dataset, venue=venue, session=session_tag,
        status=CaptureStatus.UNKNOWN if terminal in _EXHAUSTED_TERMINALS else CaptureStatus.PARTIAL,
        rows=0, reason=f"incomplete:{terminal}", refs=refs,
    )


def _safe_normalize_bars(vendor: str, rows: list[dict], snapshot_date: str, code: str) -> pd.DataFrame:
    try:
        return normalize_bar_frame(pd.DataFrame(rows), vendor, snapshot_date, code)
    except Exception as e:
        logger.warning("[DATA] bar normalize failed code=%s: %s", code, e)
        return _empty_bar_frame(snapshot_date)


def _safe_normalize_ticks(vendor: str, rows: list[dict], snapshot_date: str, code: str, truncated: bool) -> pd.DataFrame:
    try:
        return normalize_tick_frame(pd.DataFrame(rows), vendor, snapshot_date, code, truncated=truncated)
    except Exception as e:
        logger.warning("[DATA] tick normalize failed code=%s: %s", code, e)
        return _empty_tick_frame(snapshot_date)


async def _acquire_bars_symbol(
    *,
    client: Any,
    session: Any,
    code: str,
    snapshot_date: str,
    trading_day: date,
    ymd: str,
    bar_interval_minutes: int,
    floor: str,
    ceil: str,
    market_div_code: str,
    ls_client: Any | None,
    profile: CollectionSettings,
    store: CaptureStore,
    run_id: str,
) -> tuple[pd.DataFrame, CoverageEntry]:
    if ls_client is not None:
        frame, entry = await _ls_bar_attempt(
            ls_client=ls_client, session=session, code=code, snapshot_date=snapshot_date,
            trading_day=trading_day, ymd=ymd, floor=floor, ceil=ceil, session_tag=INTRADAY_SESSION_REGULAR,
            dataset=CaptureDataset.MINUTE_BARS, store=store, run_id=run_id, profile=profile,
        )
        if entry.status == CaptureStatus.COMPLETE:
            return frame, entry
        repair_frame, repair_entry = await _kis_bar_attempt(
            client=client, session=session, code=code, snapshot_date=snapshot_date, trading_day=trading_day,
            ymd=ymd, bar_interval_minutes=bar_interval_minutes, floor=floor, ceil=ceil,
            market_div_code=market_div_code, session_tag=INTRADAY_SESSION_REGULAR,
            dataset=CaptureDataset.MINUTE_BARS, store=store, run_id=run_id, profile=profile, attempt=1,
        )
        if repair_entry.status == CaptureStatus.COMPLETE:
            return repair_frame, _with_refs(repair_entry, entry)
        staged = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=CaptureDataset.MINUTE_BARS,
                                 symbol=code, session=INTRADAY_SESSION_REGULAR, frame=repair_frame,
                                 reason="repair_incomplete")
        return _empty_bar_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue=repair_entry.venue,
            session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.PARTIAL, rows=0,
            reason=f"unrepaired:{repair_entry.reason}", refs=[*entry.raw_refs, *repair_entry.raw_refs, staged],
        )
    return await _kis_bar_attempt(
        client=client, session=session, code=code, snapshot_date=snapshot_date, trading_day=trading_day,
        ymd=ymd, bar_interval_minutes=bar_interval_minutes, floor=floor, ceil=ceil,
        market_div_code=market_div_code, session_tag=INTRADAY_SESSION_REGULAR,
        dataset=CaptureDataset.MINUTE_BARS, store=store, run_id=run_id, profile=profile, attempt=0,
    )


def _with_refs(entry: CoverageEntry, prior: CoverageEntry) -> CoverageEntry:
    return CoverageEntry(
        symbol=entry.symbol, dataset=entry.dataset, venue=entry.venue, session=entry.session,
        scheduled_at=entry.scheduled_at, status=entry.status, rows=entry.rows,
        first_event_time=entry.first_event_time, last_event_time=entry.last_event_time,
        reason=entry.reason, raw_refs=(*prior.raw_refs, *entry.raw_refs),
    )


async def _tick_source_attempt(
    *,
    vendor: str,
    endpoint: str,
    payload: Any,
    code: str,
    snapshot_date: str,
    trading_day: date,
    ymd: str,
    floor: str,
    ceil: str,
    market_div_code: str | None,
    store: CaptureStore,
    run_id: str,
    profile: CollectionSettings,
    attempt: int,
    refs: list[Any],
) -> tuple[pd.DataFrame, CoverageEntry] | None:
    dataset = CaptureDataset.TRADE_TICKS
    venue = _venue_for(vendor=vendor, endpoint=endpoint, market_div_code=market_div_code, profile=profile)
    if not isinstance(payload, dict) or payload.get("rt_cd") != "0":
        return _empty_tick_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=INTRADAY_SESSION_REGULAR,
            status=CaptureStatus.FAILED, rows=0, reason="vendor_failure", refs=refs,
        )
    terminal = str(payload.get("termination_reason", "") or "exhausted")
    truncated = bool(payload.get("truncated", False))
    rows = [dict(r) for r in (payload.get("output2") or []) if isinstance(r, dict)]
    regular, other = _split_session_window(rows, ymd, floor, ceil, vendor)
    if other:
        frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=dataset,
                               symbol=code, session=INTRADAY_SESSION_REGULAR,
                               frame=_safe_normalize_ticks(vendor, other, snapshot_date, code, truncated),
                               reason="out_of_window")
        refs.append(frag)
    if truncated:
        if regular:
            frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=dataset,
                                   symbol=code, session=INTRADAY_SESSION_REGULAR,
                                   frame=_safe_normalize_ticks(vendor, regular, snapshot_date, code, truncated),
                                   reason="truncated_partial")
            refs.append(frag)
        return None
    if terminal not in _EXHAUSTED_TERMINALS or venue == "UNKNOWN":
        if regular:
            frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=dataset,
                                   symbol=code, session=INTRADAY_SESSION_REGULAR,
                                   frame=_safe_normalize_ticks(vendor, regular, snapshot_date, code, truncated),
                                   reason=f"uncertified:{terminal}")
            refs.append(frag)
        return _empty_tick_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=INTRADAY_SESSION_REGULAR,
            status=CaptureStatus.UNKNOWN, rows=0, reason=f"uncertified:{terminal}", refs=refs,
        )
    frame = _safe_normalize_ticks(vendor, regular, snapshot_date, code, truncated)
    if frame.empty:
        return _empty_tick_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=dataset, venue=venue, session=INTRADAY_SESSION_REGULAR,
            status=CaptureStatus.UNKNOWN, rows=0, reason="empty_without_proof", refs=refs,
        )
    return frame, _terminal_entry(
        symbol=code, dataset=dataset, venue=venue, session=INTRADAY_SESSION_REGULAR,
        status=CaptureStatus.COMPLETE, rows=len(frame), reason=f"{terminal}:regular={len(regular)}", refs=refs,
    )


async def _acquire_ticks_symbol(
    *,
    client: Any,
    session: Any,
    code: str,
    snapshot_date: str,
    trading_day: date,
    ymd: str,
    ls_client: Any | None,
    kiwoom_client: Any | None,
    ls_max_pages: int,
    profile: CollectionSettings,
    store: CaptureStore,
    run_id: str,
) -> tuple[pd.DataFrame, CoverageEntry]:
    refs: list[Any] = []
    failed_transport = False
    today = not _is_past_date(snapshot_date)
    if kiwoom_client is not None and today:
        context = _capture_context(
            trading_day=trading_day, run_id=run_id, dataset=CaptureDataset.TRADE_TICKS, vendor="kiwoom",
            endpoint="ka10079", symbol=code,
            venue=_venue_for(vendor="kiwoom", endpoint="ka10079", market_div_code=None, profile=profile),
            session=INTRADAY_SESSION_REGULAR,
        )
        try:
            payload: Any = await kiwoom_client.get_tick_chart(
                session, code, snapshot_date, max_pages=int(ls_max_pages),
                on_page=_observe_pages(store, context, refs, 0),
            )
        except Exception as e:
            logger.warning("[DATA] stage=ticks symbol=%s status=FAILED reason=transport:%s", code, _redacted_error(e))
            failed_transport = True
            payload = None
        if payload is not None:
            outcome = await _tick_source_attempt(
                vendor="kiwoom", endpoint="ka10079", payload=payload, code=code, snapshot_date=snapshot_date,
                trading_day=trading_day, ymd=ymd, floor=KRX_REGULAR_HOUR_FLOOR, ceil=KRX_REGULAR_HOUR_CEIL,
                market_div_code=None, store=store, run_id=run_id, profile=profile, attempt=0, refs=refs,
            )
            if outcome is not None:
                return outcome
            try:
                repair_payload: Any = await kiwoom_client.get_tick_chart(
                    session, code, snapshot_date, budget=_repair_budget(profile),
                    on_page=_observe_pages(store, context, refs, 1),
                )
            except Exception as e:
                logger.warning("[DATA] stage=ticks symbol=%s status=FAILED reason=repair:%s", code, _redacted_error(e))
                failed_transport = True
                repair_payload = None
            if repair_payload is not None:
                outcome = await _tick_source_attempt(
                    vendor="kiwoom", endpoint="ka10079", payload=repair_payload, code=code, snapshot_date=snapshot_date,
                    trading_day=trading_day, ymd=ymd, floor=KRX_REGULAR_HOUR_FLOOR, ceil=KRX_REGULAR_HOUR_CEIL,
                    market_div_code=None, store=store, run_id=run_id, profile=profile, attempt=1, refs=refs,
                )
                if outcome is not None:
                    return outcome
    if ls_client is not None:
        context = _capture_context(
            trading_day=trading_day, run_id=run_id, dataset=CaptureDataset.TRADE_TICKS, vendor="ls",
            endpoint="t8411", symbol=code,
            venue=_venue_for(vendor="ls", endpoint="t8411", market_div_code=None, profile=profile),
            session=INTRADAY_SESSION_REGULAR,
        )
        try:
            payload = await ls_client.get_tick_chart(
                session, code, snapshot_date, max_pages=int(ls_max_pages),
                on_page=_observe_pages(store, context, refs, 0),
            )
        except Exception as e:
            logger.warning("[DATA] stage=ticks symbol=%s status=FAILED reason=transport:%s", code, _redacted_error(e))
            failed_transport = True
            payload = None
        if payload is not None:
            outcome = await _tick_source_attempt(
                vendor="ls", endpoint="t8411", payload=payload, code=code, snapshot_date=snapshot_date,
                trading_day=trading_day, ymd=ymd, floor=KRX_REGULAR_HOUR_FLOOR, ceil=KRX_REGULAR_HOUR_CEIL,
                market_div_code=None, store=store, run_id=run_id, profile=profile, attempt=0, refs=refs,
            )
            if outcome is not None:
                return outcome
    if today:
        context = _capture_context(
            trading_day=trading_day, run_id=run_id, dataset=CaptureDataset.TRADE_TICKS, vendor="kis",
            endpoint="kis-ticks", symbol=code,
            venue=_venue_for(vendor="kis", endpoint="kis-ticks", market_div_code=KRX_CLOSE_MARKET_DIV_CODE, profile=profile),
            session=INTRADAY_SESSION_REGULAR,
        )
        started = _seoul_now()
        try:
            payload = await client.get_intraday_trade_ticks(
                session, code, floor_hour=KRX_REGULAR_HOUR_FLOOR, end_hour=KRX_REGULAR_HOUR_CEIL,
                market_div_code=KRX_CLOSE_MARKET_DIV_CODE,
            )
        except Exception as e:
            logger.warning("[DATA] stage=ticks symbol=%s status=FAILED reason=transport:%s", code, _redacted_error(e))
            return _empty_tick_frame(snapshot_date), _terminal_entry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=context.venue,
                session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.FAILED, rows=0,
                reason=f"transport:{_redacted_error(e)}", refs=refs,
            )
        received = _seoul_now()
        refs.append(store.append_response(_page_response(context, payload, started, received, 0)))
        rows = [dict(r) for r in (payload.get("output2") or []) if isinstance(payload, dict) and isinstance(r, dict)]
        if not isinstance(payload, dict) or payload.get("rt_cd") != "0":
            return _empty_tick_frame(snapshot_date), _terminal_entry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=context.venue,
                session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.FAILED, rows=0,
                reason="vendor_failure", refs=refs,
            )
        frame = _canonical_kis_ticks(rows, snapshot_date, code)
        if frame.empty:
            return _empty_tick_frame(snapshot_date), _terminal_entry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=context.venue,
                session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.UNKNOWN, rows=0,
                reason="empty_without_proof", refs=refs,
            )
        return frame, _terminal_entry(
            symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=context.venue,
            session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.COMPLETE, rows=len(frame),
            reason=f"exhausted:regular={len(frame)}", refs=refs,
        )
    if failed_transport and not refs:
        return _empty_tick_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue="UNKNOWN",
            session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.FAILED, rows=0,
            reason="transport_failure", refs=refs,
        )
    if refs:
        staged = _stage_fragment(store, run_id=run_id, trading_day=trading_day, dataset=CaptureDataset.TRADE_TICKS,
                                 symbol=code, session=INTRADAY_SESSION_REGULAR,
                                 frame=_empty_tick_frame(snapshot_date), reason="unrepaired")
        refs.append(staged)
        return _empty_tick_frame(snapshot_date), _terminal_entry(
            symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue="UNKNOWN",
            session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.PARTIAL, rows=0,
            reason="unrepaired", refs=refs,
        )
    return _empty_tick_frame(snapshot_date), _terminal_entry(
        symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue="UNKNOWN",
        session=INTRADAY_SESSION_REGULAR, status=CaptureStatus.UNKNOWN, rows=0,
        reason="no_source_attempt", refs=refs,
    )


async def _collect_with_observer(
    *,
    codes: list[str],
    snapshot_date: str,
    dataset: CaptureDataset,
    session_tag: str,
    store: CaptureStore,
    run_id: str,
    acquire: Any,
    on_symbol: SymbolObserver | None,
) -> pd.DataFrame:
    trading_day = _parse_snapshot_date(snapshot_date)
    _publish_pending_manifest(
        store, trading_day=trading_day, run_id=run_id, dataset=dataset,
        vendor="owner-local", endpoint="pending", session=session_tag, symbols=list(codes),
    )
    frames: list[pd.DataFrame] = []
    for code in codes:
        frame, entry = await acquire(code)
        if on_symbol is not None:
            on_symbol(code, frame, entry)
        elif not frame.empty:
            frames.append(frame)
    if on_symbol is not None:
        return _empty_bar_frame(snapshot_date) if dataset == CaptureDataset.MINUTE_BARS else _empty_tick_frame(snapshot_date)
    if not frames:
        return _empty_bar_frame(snapshot_date) if dataset == CaptureDataset.MINUTE_BARS else _empty_tick_frame(snapshot_date)
    return pd.concat(frames, ignore_index=True)


async def _collect_bars(
    client,
    session,
    stock_codes: list[str],
    snapshot_date: str,
    bar_interval_minutes: int,
    end_hour: str,
    floor_hour: str,
    market_div_code: str,
    historical: bool = False,
) -> pd.DataFrame:
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)
    target_date = str(snapshot_date).replace("-", "") if historical else ""

    async def _fetch_one(code: str) -> pd.DataFrame:
        async with sem:
            try:
                if historical:
                    res = await client.get_historical_minute_chart(
                        session,
                        code,
                        target_date,
                        bar_interval_minutes=bar_interval_minutes,
                        end_hour=end_hour,
                        floor_hour=floor_hour,
                        market_div_code=market_div_code,
                    )
                else:
                    res = await client.get_intraday_minute_chart(
                        session,
                        code,
                        bar_interval_minutes=bar_interval_minutes,
                        end_hour=end_hour,
                        floor_hour=floor_hour,
                        market_div_code=market_div_code,
                    )
            except Exception as e:
                logger.warning("Intraday bars failed code=%s: %s", code, e)
                return pd.DataFrame()
            if res.get("rt_cd") != "0":
                return pd.DataFrame()
            rows = res.get("output2") or []
            if not rows:
                return pd.DataFrame()
            return _canonical_kis_bars(rows, snapshot_date, code)

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def _legacy_collect_intraday_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, ls_client: Any | None = None) -> pd.DataFrame:
    if ls_client is None:
        return await _collect_bars(
            client, session, stock_codes, snapshot_date, bar_interval_minutes,
            KRX_REGULAR_HOUR_CEIL, KRX_REGULAR_HOUR_FLOOR, KRX_CLOSE_MARKET_DIV_CODE,
        )
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)
    target_date = snapshot_date

    async def _fetch_one(code: str) -> pd.DataFrame:
        async with sem:
            try:
                res = await ls_client.get_minute_chart(session, code, target_date)
            except Exception as e:
                logger.warning("LS minute chart failed code=%s: %s", code, e)
                res = {"rt_cd": "1", "output2": []}
            if res.get("rt_cd") == "0" and (res.get("output2") or []):
                rows = res.get("output2") or []
                logger.info("Fetched %d minute bars for %s via LS", len(rows), code)
                vendor = str(res.get("vendor", "ls") or "ls")
                try:
                    return normalize_bar_frame(pd.DataFrame(rows), vendor, snapshot_date, code)
                except Exception as e:
                    logger.warning("[DATA] LS bar normalize failed code=%s: %s", code, e)
                    return pd.DataFrame()
            else:
                try:
                    res = await client.get_intraday_minute_chart(
                        session, code, bar_interval_minutes=bar_interval_minutes,
                        end_hour=KRX_REGULAR_HOUR_CEIL, floor_hour=KRX_REGULAR_HOUR_FLOOR,
                        market_div_code=KRX_CLOSE_MARKET_DIV_CODE,
                    )
                except Exception as e:
                    logger.warning("Intraday bars failed code=%s: %s", code, e)
                    return pd.DataFrame()
                if res.get("rt_cd") != "0":
                    return pd.DataFrame()
                rows = res.get("output2") or []
                logger.info("Fetched %d minute bars for %s via KIS fallback", len(rows), code)
            if not rows:
                return pd.DataFrame()
            return _canonical_kis_bars(rows, snapshot_date, code)

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def collect_intraday_bars(client: Any, session: Any, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, ls_client: Any | None = None, *, profile: CollectionSettings | None = None, capture_store: CaptureStore | None = None, run_id: str | None = None, on_symbol: SymbolObserver | None = None) -> pd.DataFrame:
    """Classify complete source attempts before publishing regular-session bars.

    Args:
        client: Existing explicitly routed KIS client for independent repair.
        session: Existing HTTP session.
        stock_codes: Full declared candidate cohort, not a top-K selection.
        snapshot_date: Exact requested market date.
        bar_interval_minutes: Existing declared interval.
        ls_client: Optional paginated primary chart adapter.
        profile: Validated acquisition and verified-route configuration.
        capture_store: Owner-local raw/task persistence.
        run_id: Stable identity for this acquisition.
        on_symbol: Bounded symbol result consumer; suppresses all-symbol accumulation.

    Returns:
        Compatible frame when on_symbol is None, otherwise an empty canonical frame.

    Raises:
        ValueError: Invalid profile, route, or date.
        OSError: Required evidence or publication fails.
    """
    if profile is None and capture_store is None and on_symbol is None and run_id is None:
        return await _legacy_collect_intraday_bars(client, session, stock_codes, snapshot_date, bar_interval_minutes, ls_client)
    if int(bar_interval_minutes) <= 0:
        raise ValueError(f"Invalid bar_interval_minutes: {bar_interval_minutes!r}")
    prof = _resolve_profile(profile)
    trading_day = _parse_snapshot_date(snapshot_date)
    ymd = trading_day.isoformat().replace("-", "")
    store = _resolve_store(capture_store, prof)
    resolved_run = _new_run_id(run_id, str(snapshot_date), CaptureDataset.MINUTE_BARS)
    codes = [str(item) for item in (stock_codes or [])]

    async def _acquire(code: str) -> tuple[pd.DataFrame, CoverageEntry]:
        return await _acquire_bars_symbol(
            client=client, session=session, code=code, snapshot_date=str(snapshot_date),
            trading_day=trading_day, ymd=ymd, bar_interval_minutes=int(bar_interval_minutes),
            floor=KRX_REGULAR_HOUR_FLOOR, ceil=KRX_REGULAR_HOUR_CEIL,
            market_div_code=KRX_CLOSE_MARKET_DIV_CODE, ls_client=ls_client,
            profile=prof, store=store, run_id=resolved_run,
        )

    return await _collect_with_observer(
        codes=codes, snapshot_date=str(snapshot_date), dataset=CaptureDataset.MINUTE_BARS,
        session_tag=INTRADAY_SESSION_REGULAR, store=store, run_id=resolved_run,
        acquire=_acquire, on_symbol=on_symbol,
    )


async def _legacy_collect_nxt_aftermarket_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, kiwoom_client: Any | None = None) -> pd.DataFrame:
    if kiwoom_client is None:
        return await _collect_bars(
            client, session, stock_codes, snapshot_date, bar_interval_minutes,
            NXT_AFTERMARKET_HOUR_CEIL, NXT_AFTERMARKET_HOUR_FLOOR, NXT_MARKET_DIV_CODE,
        )
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)

    async def _fetch_one(code: str) -> pd.DataFrame:
        async with sem:
            try:
                kw_res = await kiwoom_client.get_nxt_minute_chart(session, code, snapshot_date)
            except Exception as e:
                logger.warning("Kiwoom NXT minute chart failed code=%s: %s", code, e)
                kw_res = {"rt_cd": "1", "output2": []}
            if kw_res.get("rt_cd") == "0" and (kw_res.get("output2") or []):
                rows = kw_res.get("output2") or []
                vendor = str(kw_res.get("vendor", "kiwoom") or "kiwoom")
                try:
                    return normalize_bar_frame(pd.DataFrame(rows), vendor, snapshot_date, code)
                except Exception as e:
                    logger.warning("[DATA] Kiwoom NXT bar normalize failed code=%s: %s", code, e)
                    return pd.DataFrame()
            if kw_res.get("rt_cd") == "0":
                return pd.DataFrame()
            try:
                res = await client.get_intraday_minute_chart(
                    session, code, bar_interval_minutes=bar_interval_minutes,
                    end_hour=NXT_AFTERMARKET_HOUR_CEIL, floor_hour=NXT_AFTERMARKET_HOUR_FLOOR,
                    market_div_code=NXT_MARKET_DIV_CODE,
                )
            except Exception as e:
                logger.warning("Intraday bars failed code=%s: %s", code, e)
                return pd.DataFrame()
            if res.get("rt_cd") != "0":
                return pd.DataFrame()
            rows = res.get("output2") or []
            if not rows:
                return pd.DataFrame()
            return _canonical_kis_bars(rows, snapshot_date, code)

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def _acquire_extended_bars_symbol(
    *,
    client: Any,
    session: Any,
    code: str,
    snapshot_date: str,
    trading_day: date,
    ymd: str,
    bar_interval_minutes: int,
    floor: str,
    ceil: str,
    market_div_code: str,
    session_tag: str,
    kiwoom_client: Any | None,
    kiwoom_method: str,
    profile: CollectionSettings,
    store: CaptureStore,
    run_id: str,
) -> tuple[pd.DataFrame, CoverageEntry]:
    refs: list[Any] = []
    if kiwoom_client is not None:
        venue = _venue_for(vendor="kiwoom", endpoint="ka10080", market_div_code=None, profile=profile)
        context = _capture_context(
            trading_day=trading_day, run_id=run_id, dataset=CaptureDataset.MINUTE_BARS, vendor="kiwoom",
            endpoint="ka10080", symbol=code, venue=venue, session=session_tag,
        )
        started = _seoul_now()
        try:
            payload = await getattr(kiwoom_client, kiwoom_method)(session, code, snapshot_date)
            received = _seoul_now()
            refs.append(store.append_response(_page_response(context, payload, started, received, 0)))
        except Exception as e:
            logger.warning("[DATA] stage=bars symbol=%s status=FAILED reason=transport:%s", code, _redacted_error(e))
            payload = {"rt_cd": "1", "msg1": _redacted_error(e)}
        if isinstance(payload, dict) and payload.get("rt_cd") == "0" and (payload.get("output2") or []):
            rows = [dict(r) for r in (payload.get("output2") or []) if isinstance(r, dict)]
            regular, _ = _split_session_window(rows, ymd, floor, ceil, "kiwoom")
            try:
                frame = normalize_bar_frame(pd.DataFrame(regular), "kiwoom", snapshot_date, code)
            except Exception as e:
                return _empty_bar_frame(snapshot_date), _terminal_entry(
                    symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue=venue, session=session_tag,
                    status=CaptureStatus.FAILED, rows=0, reason=f"normalize:{_redacted_error(e)}", refs=refs,
                )
            if not frame.empty and venue != "UNKNOWN":
                return frame, _terminal_entry(
                    symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue=venue, session=session_tag,
                    status=CaptureStatus.COMPLETE, rows=len(frame),
                    reason=f"exhausted:regular={len(regular)}", refs=refs,
                )
            if not frame.empty:
                frag = _stage_fragment(store, run_id=run_id, trading_day=trading_day,
                                       dataset=CaptureDataset.MINUTE_BARS, symbol=code, session=session_tag,
                                       frame=frame, reason="uncertified_venue")
                refs.append(frag)
            return _empty_bar_frame(snapshot_date), _terminal_entry(
                symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue=venue, session=session_tag,
                status=CaptureStatus.UNKNOWN, rows=0, reason="uncertified_or_empty", refs=refs,
            )
    return await _kis_bar_attempt(
        client=client, session=session, code=code, snapshot_date=snapshot_date, trading_day=trading_day,
        ymd=ymd, bar_interval_minutes=bar_interval_minutes, floor=floor, ceil=ceil,
        market_div_code=market_div_code, session_tag=session_tag,
        dataset=CaptureDataset.MINUTE_BARS, store=store, run_id=run_id, profile=profile, attempt=0,
    )


async def collect_nxt_aftermarket_bars(client: Any, session: Any, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, kiwoom_client: Any | None = None, *, profile: CollectionSettings | None = None, capture_store: CaptureStore | None = None, run_id: str | None = None, on_symbol: SymbolObserver | None = None) -> pd.DataFrame:
    """NXT 애프터마켓(15:40-20:00) 전체를 1분봉 연속 시계열로 수집한다. 미상장은 조용히 스킵."""
    if profile is None and capture_store is None and on_symbol is None and run_id is None:
        return await _legacy_collect_nxt_aftermarket_bars(client, session, stock_codes, snapshot_date, bar_interval_minutes, kiwoom_client)
    prof = _resolve_profile(profile)
    trading_day = _parse_snapshot_date(snapshot_date)
    ymd = trading_day.isoformat().replace("-", "")
    store = _resolve_store(capture_store, prof)
    resolved_run = _new_run_id(run_id, str(snapshot_date), CaptureDataset.MINUTE_BARS)
    codes = [str(item) for item in (stock_codes or [])]

    async def _acquire(code: str) -> tuple[pd.DataFrame, CoverageEntry]:
        return await _acquire_extended_bars_symbol(
            client=client, session=session, code=code, snapshot_date=str(snapshot_date),
            trading_day=trading_day, ymd=ymd, bar_interval_minutes=int(bar_interval_minutes),
            floor=NXT_AFTERMARKET_HOUR_FLOOR, ceil=NXT_AFTERMARKET_HOUR_CEIL,
            market_div_code=NXT_MARKET_DIV_CODE, session_tag="nxt_aftermarket",
            kiwoom_client=kiwoom_client, kiwoom_method="get_nxt_minute_chart",
            profile=prof, store=store, run_id=resolved_run,
        )

    return await _collect_with_observer(
        codes=codes, snapshot_date=str(snapshot_date), dataset=CaptureDataset.MINUTE_BARS,
        session_tag="nxt_aftermarket", store=store, run_id=resolved_run,
        acquire=_acquire, on_symbol=on_symbol,
    )


async def _legacy_collect_nxt_premarket_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, kiwoom_client: Any | None = None) -> pd.DataFrame:
    if kiwoom_client is None:
        return await _collect_bars(
            client, session, stock_codes, snapshot_date, bar_interval_minutes,
            NXT_PREMARKET_HOUR_CEIL, NXT_PREMARKET_HOUR_FLOOR, NXT_MARKET_DIV_CODE,
        )
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)

    async def _fetch_one(code: str) -> pd.DataFrame:
        async with sem:
            try:
                kw_res = await kiwoom_client.get_nxt_premarket_chart(session, code, snapshot_date)
            except Exception as e:
                logger.warning("Kiwoom NXT premarket chart failed code=%s: %s", code, e)
                kw_res = {"rt_cd": "1", "output2": []}
            if kw_res.get("rt_cd") == "0" and (kw_res.get("output2") or []):
                rows = kw_res.get("output2") or []
                vendor = str(kw_res.get("vendor", "kiwoom") or "kiwoom")
                try:
                    return normalize_bar_frame(pd.DataFrame(rows), vendor, snapshot_date, code)
                except Exception as e:
                    logger.warning("[DATA] Kiwoom NXT bar normalize failed code=%s: %s", code, e)
                    return pd.DataFrame()
            if kw_res.get("rt_cd") == "0":
                return pd.DataFrame()
            try:
                res = await client.get_intraday_minute_chart(
                    session, code, bar_interval_minutes=bar_interval_minutes,
                    end_hour=NXT_PREMARKET_HOUR_CEIL, floor_hour=NXT_PREMARKET_HOUR_FLOOR,
                    market_div_code=NXT_MARKET_DIV_CODE,
                )
            except Exception as e:
                logger.warning("Intraday bars failed code=%s: %s", code, e)
                return pd.DataFrame()
            if res.get("rt_cd") != "0":
                return pd.DataFrame()
            rows = res.get("output2") or []
            if not rows:
                return pd.DataFrame()
            return _canonical_kis_bars(rows, snapshot_date, code)

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def collect_nxt_premarket_bars(client: Any, session: Any, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, kiwoom_client: Any | None = None, *, profile: CollectionSettings | None = None, capture_store: CaptureStore | None = None, run_id: str | None = None, on_symbol: SymbolObserver | None = None) -> pd.DataFrame:
    """NXT 프리마켓(08:00-08:50) 전체를 1분봉 연속 시계열로 수집한다. 미상장은 조용히 스킵."""
    if profile is None and capture_store is None and on_symbol is None and run_id is None:
        return await _legacy_collect_nxt_premarket_bars(client, session, stock_codes, snapshot_date, bar_interval_minutes, kiwoom_client)
    prof = _resolve_profile(profile)
    trading_day = _parse_snapshot_date(snapshot_date)
    ymd = trading_day.isoformat().replace("-", "")
    store = _resolve_store(capture_store, prof)
    resolved_run = _new_run_id(run_id, str(snapshot_date), CaptureDataset.MINUTE_BARS)
    codes = [str(item) for item in (stock_codes or [])]

    async def _acquire(code: str) -> tuple[pd.DataFrame, CoverageEntry]:
        return await _acquire_extended_bars_symbol(
            client=client, session=session, code=code, snapshot_date=str(snapshot_date),
            trading_day=trading_day, ymd=ymd, bar_interval_minutes=int(bar_interval_minutes),
            floor=NXT_PREMARKET_HOUR_FLOOR, ceil=NXT_PREMARKET_HOUR_CEIL,
            market_div_code=NXT_MARKET_DIV_CODE, session_tag="nxt_premarket",
            kiwoom_client=kiwoom_client, kiwoom_method="get_nxt_premarket_chart",
            profile=prof, store=store, run_id=resolved_run,
        )

    return await _collect_with_observer(
        codes=codes, snapshot_date=str(snapshot_date), dataset=CaptureDataset.MINUTE_BARS,
        session_tag="nxt_premarket", store=store, run_id=resolved_run,
        acquire=_acquire, on_symbol=on_symbol,
    )


async def _legacy_collect_intraday_trade_ticks(
    client,
    session,
    stock_codes: list[str],
    snapshot_date: str,
    ls_client: Any | None = None,
    kiwoom_client: Any | None = None,
    ls_max_pages: int = DEFAULT_LS_TICK_MAX_PAGES,
) -> pd.DataFrame:
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)
    target_date = snapshot_date
    today_str = datetime.now().strftime("%Y-%m-%d")
    use_kiwoom = kiwoom_client is not None and str(snapshot_date) >= today_str

    async def _fetch_one(code: str) -> list[dict]:
        async with sem:
            if use_kiwoom:
                try:
                    kw_res = await kiwoom_client.get_tick_chart(session, code, target_date)
                except Exception as e:
                    logger.warning("Kiwoom tick chart failed code=%s: %s", code, e)
                    kw_res = {"rt_cd": "1", "output2": []}
                if kw_res.get("rt_cd") == "0" and (kw_res.get("output2") or []):
                    rows = kw_res.get("output2") or []
                    truncated = bool(kw_res.get("truncated", False))
                    vendor = str(kw_res.get("vendor", "kiwoom") or "kiwoom")
                    logger.info("Fetched %d ticks for %s via Kiwoom", len(rows), code)
                    try:
                        frame = normalize_tick_frame(pd.DataFrame(rows), vendor, snapshot_date, code, truncated=truncated)
                    except Exception as e:
                        logger.warning("[DATA] Kiwoom tick normalize failed code=%s: %s", code, e)
                        return []
                    return frame.to_dict("records") if not frame.empty else []
            if ls_client is not None:
                try:
                    ls_res = await ls_client.get_tick_chart(session, code, target_date, max_pages=ls_max_pages)
                except Exception as e:
                    logger.warning("LS tick chart failed code=%s: %s", code, e)
                    ls_res = {"rt_cd": "1", "output2": []}
                if ls_res.get("rt_cd") == "0" and (ls_res.get("output2") or []):
                    rows = ls_res.get("output2") or []
                    truncated = bool(ls_res.get("truncated", False))
                    vendor = str(ls_res.get("vendor", "ls") or "ls")
                    logger.info("Fetched %d ticks for %s via LS", len(rows), code)
                    try:
                        frame = normalize_tick_frame(pd.DataFrame(rows), vendor, snapshot_date, code, truncated=truncated)
                    except Exception as e:
                        logger.warning("[DATA] LS tick normalize failed code=%s: %s", code, e)
                        return []
                    return frame.to_dict("records") if not frame.empty else []
            try:
                res = await client.get_intraday_trade_ticks(session, code, floor_hour=KRX_REGULAR_HOUR_FLOOR, end_hour=KRX_REGULAR_HOUR_CEIL, market_div_code=KRX_CLOSE_MARKET_DIV_CODE)
            except Exception as e:
                logger.warning("Intraday trade ticks failed code=%s: %s", code, e)
                return []
            if res.get("rt_cd") != "0":
                return []
            rows = res.get("output2") or []
            if not rows:
                return []
            frame = _canonical_kis_ticks(rows, snapshot_date, code)
            logger.info("Fetched %d ticks for %s via KIS fallback", len(frame), code)
            return frame.to_dict("records") if not frame.empty else []

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    all_rows: list[dict] = []
    for rows in results:
        if rows:
            all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


async def collect_intraday_trade_ticks(client: Any, session: Any, stock_codes: list[str], snapshot_date: str, ls_client: Any | None = None, kiwoom_client: Any | None = None, ls_max_pages: int = DEFAULT_LS_TICK_MAX_PAGES, *, profile: CollectionSettings | None = None, capture_store: CaptureStore | None = None, run_id: str | None = None, on_symbol: SymbolObserver | None = None) -> pd.DataFrame:
    """Deliver one certified whole-symbol attempt without merging broker tapes.

    Args:
        client: Explicit-route KIS fallback client.
        session: Existing HTTP session.
        stock_codes: All expected project candidates.
        snapshot_date: Exact requested market date.
        ls_client: Optional independent LS source.
        kiwoom_client: Optional current-day Kiwoom source.
        ls_max_pages: Compatible explicit LS limit.
        profile: Normal/repair budgets and route certification.
        capture_store: Owner-local raw and partial-attempt storage.
        run_id: Acquisition identity.
        on_symbol: Bounded consumer of frame and task coverage.

    Returns:
        Legacy combined frame without an observer, otherwise empty canonical frame.

    Raises:
        ValueError: Conflicting or unverified classification inputs.
        OSError: Mandatory evidence cannot be preserved.
    """
    if profile is None and capture_store is None and on_symbol is None and run_id is None:
        return await _legacy_collect_intraday_trade_ticks(client, session, stock_codes, snapshot_date, ls_client, kiwoom_client, ls_max_pages)
    if int(ls_max_pages) <= 0:
        raise ValueError(f"Invalid ls_max_pages: {ls_max_pages!r}")
    prof = _resolve_profile(profile)
    trading_day = _parse_snapshot_date(snapshot_date)
    ymd = trading_day.isoformat().replace("-", "")
    store = _resolve_store(capture_store, prof)
    resolved_run = _new_run_id(run_id, str(snapshot_date), CaptureDataset.TRADE_TICKS)
    codes = [str(item) for item in (stock_codes or [])]

    async def _acquire(code: str) -> tuple[pd.DataFrame, CoverageEntry]:
        return await _acquire_ticks_symbol(
            client=client, session=session, code=code, snapshot_date=str(snapshot_date),
            trading_day=trading_day, ymd=ymd, ls_client=ls_client, kiwoom_client=kiwoom_client,
            ls_max_pages=int(ls_max_pages), profile=prof, store=store, run_id=resolved_run,
        )

    return await _collect_with_observer(
        codes=codes, snapshot_date=str(snapshot_date), dataset=CaptureDataset.TRADE_TICKS,
        session_tag=INTRADAY_SESSION_REGULAR, store=store, run_id=resolved_run,
        acquire=_acquire, on_symbol=on_symbol,
    )



async def backfill_regular_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """특정 과거 날짜(snapshot_date, 'YYYY-MM-DD')의 정규세션 1분봉을 FHKST03010230으로 소급 수집."""
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        KRX_REGULAR_HOUR_CEIL, KRX_REGULAR_HOUR_FLOOR, KRX_CLOSE_MARKET_DIV_CODE,
        historical=True,
    )


async def backfill_nxt_aftermarket_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """특정 과거 날짜의 NXT 애프터마켓 1분봉을 FHKST03010230으로 소급 수집.

    FHKST03010230이 애프터마켓 시간대(15:40-20:00)를 실제로 보관하는지는 실측 미검증
    상태이므로 보관하지 않는 것으로 확인되면 이 함수 호출부를 제거한다. 실패 시 정상
    스킵하며 정규세션 백필(backfill_regular_bars)과 완전히 독립적으로 동작한다.
    """
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        NXT_AFTERMARKET_HOUR_CEIL, NXT_AFTERMARKET_HOUR_FLOOR, NXT_MARKET_DIV_CODE,
        historical=True,
    )


async def collect_krx_aftermarket_bars(client: Any, session: Any, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, *, profile: CollectionSettings | None = None, capture_store: CaptureStore | None = None, run_id: str | None = None, on_symbol: SymbolObserver | None = None) -> pd.DataFrame:
    """당일 KRX 애프터마켓(16:00-20:00) 1분봉을 KRX_CLOSE_MARKET_DIV_CODE('J')로 수집한다."""
    if profile is None and capture_store is None and on_symbol is None and run_id is None:
        if str(snapshot_date) < KRX_AFTERMARKET_START_DATE:
            return pd.DataFrame()
        return await _collect_bars(
            client, session, stock_codes, snapshot_date, bar_interval_minutes,
            KRX_AFTERMARKET_HOUR_CEIL, KRX_AFTERMARKET_HOUR_FLOOR, KRX_CLOSE_MARKET_DIV_CODE,
        )
    if str(snapshot_date) < KRX_AFTERMARKET_START_DATE:
        prof = _resolve_profile(profile)
        trading_day = _parse_snapshot_date(snapshot_date)
        store = _resolve_store(capture_store, prof)
        resolved_run = _new_run_id(run_id, str(snapshot_date), CaptureDataset.MINUTE_BARS)
        codes = [str(item) for item in (stock_codes or [])]

        async def _acquire_na(code: str) -> tuple[pd.DataFrame, CoverageEntry]:
            await asyncio.sleep(0)
            return _empty_bar_frame(str(snapshot_date)), _terminal_entry(
                symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue="UNKNOWN", session="krx_aftermarket",
                status=CaptureStatus.UNKNOWN, rows=0, reason="not_applicable_without_proof", refs=[],
            )

        return await _collect_with_observer(
            codes=codes, snapshot_date=str(snapshot_date), dataset=CaptureDataset.MINUTE_BARS,
            session_tag="krx_aftermarket", store=store, run_id=resolved_run,
            acquire=_acquire_na, on_symbol=on_symbol,
        )
    prof = _resolve_profile(profile)
    trading_day = _parse_snapshot_date(snapshot_date)
    ymd = trading_day.isoformat().replace("-", "")
    store = _resolve_store(capture_store, prof)
    resolved_run = _new_run_id(run_id, str(snapshot_date), CaptureDataset.MINUTE_BARS)
    codes = [str(item) for item in (stock_codes or [])]

    async def _acquire(code: str) -> tuple[pd.DataFrame, CoverageEntry]:
        return await _kis_bar_attempt(
            client=client, session=session, code=code, snapshot_date=str(snapshot_date),
            trading_day=trading_day, ymd=ymd, bar_interval_minutes=int(bar_interval_minutes),
            floor=KRX_AFTERMARKET_HOUR_FLOOR, ceil=KRX_AFTERMARKET_HOUR_CEIL,
            market_div_code=KRX_CLOSE_MARKET_DIV_CODE, session_tag="krx_aftermarket",
            dataset=CaptureDataset.MINUTE_BARS, store=store, run_id=resolved_run, profile=prof, attempt=0,
        )

    return await _collect_with_observer(
        codes=codes, snapshot_date=str(snapshot_date), dataset=CaptureDataset.MINUTE_BARS,
        session_tag="krx_aftermarket", store=store, run_id=resolved_run,
        acquire=_acquire, on_symbol=on_symbol,
    )


async def backfill_krx_aftermarket_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """과거 날짜의 KRX 애프터마켓 1분봉을 historical=True(FHKST03010230)로 소급 수집한다."""
    if str(snapshot_date) < KRX_AFTERMARKET_START_DATE:
        return pd.DataFrame()
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        KRX_AFTERMARKET_HOUR_CEIL, KRX_AFTERMARKET_HOUR_FLOOR, KRX_CLOSE_MARKET_DIV_CODE,
        historical=True,
    )
