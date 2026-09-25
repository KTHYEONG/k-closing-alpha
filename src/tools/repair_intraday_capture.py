"""Stage bounded historical repairs without deleting original collection evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.config.collection import CollectionSettings
from src.config.market_session import INTRADAY_SESSION_REGULAR
from src.data.capture_contracts import SEOUL, CaptureStatus, CoverageEntry
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root

logger = logging.getLogger(__name__)

_DATASETS = ("regular_bars", "regular_ticks")
_MAX_REPAIR_SPAN_DAYS = 31
_MAX_REPAIR_ATTEMPTS = 500
_MAX_REPAIR_HISTORY_DAYS = 365


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage bounded historical intraday repairs.")
    parser.add_argument("--start", required=True, help="Inclusive repair start (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive repair end (YYYY-MM-DD).")
    parser.add_argument("--dataset", action="append", default=[], help="Repeatable dataset in {regular_bars, regular_ticks}.")
    parser.add_argument("--symbol", action="append", default=[], help="Repeatable symbol filter.")
    parser.add_argument("--max-pages", type=int, default=None, help="Bounded page budget per attempt.")
    parser.add_argument("--deadline", default=None, help="Aware ISO timestamp bounding repair work.")
    parser.add_argument("--apply", action="store_true", help="Apply certified attempts through partition writers.")
    return parser.parse_args(argv)


def _parse_day(value: str, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid {field} date: {value!r}") from None


def _parse_deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid --deadline timestamp: {value!r}") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"--deadline must be timezone-aware: {value!r}")
    return moment.astimezone(SEOUL)


def _existing_symbols(snapshot_date: str, dataset: str) -> list[str]:
    base = Path(settings.HISTORY_DIR) / "intraday"
    if dataset == "regular_ticks":
        candidates = [base / "ticks" / INTRADAY_SESSION_REGULAR / snapshot_date[:7] / f"{snapshot_date}.parquet"]
    else:
        candidates = sorted((base).glob(f"*/{INTRADAY_SESSION_REGULAR}/{snapshot_date[:7]}/{snapshot_date}.parquet"))
    symbols: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "symbol" in frame.columns:
            for item in frame["symbol"].astype(str).tolist():
                if item not in symbols:
                    symbols.append(item)
    return symbols


def _open_clients() -> tuple[Any, Any, Any | None, Any | None]:
    from src.api.kis.client import KisApiClient, kis_data_client_kwargs
    from src.api.kiwoom.client import KiwoomApiClient
    from src.api.ls.client import LsApiClient

    client = KisApiClient(**kis_data_client_kwargs())  # type: ignore[no-untyped-call]
    ls_client = LsApiClient() if settings.LS_APP_KEY else None
    kiwoom_client = KiwoomApiClient() if settings.KIWOOM_APP_KEY else None
    return client, client.create_session(), ls_client, kiwoom_client


async def _repair_symbol(
    *,
    kind: str,
    client: Any,
    session: Any,
    ls_client: Any | None,
    kiwoom_client: Any | None,
    code: str,
    snapshot_date: str,
    profile: CollectionSettings,
    store: CaptureStore,
    run_id: str,
    apply: bool,
) -> dict[str, Any]:
    from src.backfill.intraday.collector import collect_intraday_bars, collect_intraday_trade_ticks
    from src.data.intraday_store import write_intraday_partition, write_tick_partition

    held: dict[str, Any] = {}
    batch_rows = int(profile.COLLECTION_ARROW_BATCH_ROWS)

    def _on_symbol(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
        held["frame"] = frame
        held["entry"] = entry

    if kind == "regular_bars":
        await collect_intraday_bars(
            client, session, [code], snapshot_date, 1, ls_client=ls_client,
            profile=profile, capture_store=store, run_id=run_id, on_symbol=_on_symbol,
        )
    else:
        await collect_intraday_trade_ticks(
            client, session, [code], snapshot_date, ls_client=ls_client, kiwoom_client=kiwoom_client,
            profile=profile, capture_store=store, run_id=run_id, on_symbol=_on_symbol,
        )
    entry = held["entry"]
    frame = held["frame"]
    applied_rows = 0
    if apply and entry.status == CaptureStatus.COMPLETE and not frame.empty:
        coverage = {code: entry}
        if kind == "regular_bars":
            total = write_intraday_partition(frame, 1, snapshot_date, INTRADAY_SESSION_REGULAR,
                                             coverage=coverage, batch_rows=batch_rows)
        else:
            total = write_tick_partition(frame, snapshot_date, INTRADAY_SESSION_REGULAR,
                                         coverage=coverage, batch_rows=batch_rows)
        if total < len(frame):
            raise RuntimeError(f"Repair publication verification failed symbol={code} date={snapshot_date}")
        applied_rows = len(frame)
    return {
        "symbol": code,
        "status": entry.status.value,
        "rows": len(frame),
        "applied_rows": applied_rows,
        "reason": entry.reason,
    }


def main(argv: list[str] | None = None) -> None:
    """Stage bounded historical repairs without deleting original collection evidence.

    Args:
        argv: Optional CLI arguments for exact date range, dataset, and profile.

    Raises:
        ValueError: Invalid or unbounded repair selection.
        RuntimeError: Required historical data or publication verification fails.
    """
    args = _parse_args(argv)
    start = _parse_day(args.start, "--start")
    end = _parse_day(args.end, "--end")
    if end < start:
        raise ValueError(f"Invalid repair range: {args.start!r}..{args.end!r}")
    span = (end - start).days + 1
    if span > _MAX_REPAIR_SPAN_DAYS:
        raise ValueError(f"Unbounded repair selection: span={span} exceeds {_MAX_REPAIR_SPAN_DAYS} days")
    datasets = list(args.dataset or [])
    if not datasets:
        raise ValueError("Repair selection requires at least one --dataset")
    for item in datasets:
        if item not in _DATASETS:
            raise ValueError(f"Invalid --dataset: {item!r} (expected one of {sorted(_DATASETS)})")
    max_pages = args.max_pages
    if max_pages is not None and int(max_pages) <= 0:
        raise ValueError(f"Invalid --max-pages: {args.max_pages!r}")
    deadline = _parse_deadline(args.deadline)
    profile = CollectionSettings()
    if max_pages is not None:
        profile = profile.model_copy(update={"COLLECTION_CHART_MAX_PAGES": int(max_pages)})
    store = CaptureStore(_capture_root(profile))
    days = [(start + timedelta(days=offset)).isoformat() for offset in range(span)]
    today = datetime.now(SEOUL).date()
    report: dict[str, Any] = {"start": start.isoformat(), "end": end.isoformat(), "apply": bool(args.apply), "attempts": []}
    selections: list[tuple[str, str, str]] = []
    for day in days:
        for kind in datasets:
            symbols = list(args.symbol or [])
            if not symbols:
                symbols = _existing_symbols(day, kind)
            if not symbols:
                raise ValueError(f"Unbounded repair selection: no symbols for {day!r} {kind!r}")
            selections.extend((day, kind, str(code)) for code in symbols)
    if len(selections) > _MAX_REPAIR_ATTEMPTS:
        raise ValueError(f"Unbounded repair selection: attempts={len(selections)} exceeds {_MAX_REPAIR_ATTEMPTS}")

    async def _run() -> None:
        client, session_ctx, ls_client, kiwoom_client = _open_clients()
        async with session_ctx as session:
            await client.ensure_token(session)
            for day, kind, code in selections:
                if deadline is not None and datetime.now(SEOUL) > deadline:
                    report["attempts"].append({"date": day, "dataset": kind, "symbol": code, "status": "deadline-exceeded", "rows": 0})
                    continue
                target = date.fromisoformat(day)
                if (today - target).days > _MAX_REPAIR_HISTORY_DAYS:
                    report["attempts"].append({"date": day, "dataset": kind, "symbol": code, "status": "unsupported-horizon", "rows": 0})
                    continue
                run_id = f"repair-{day}-{kind}-{code}-{uuid.uuid4().hex[:6]}"
                try:
                    attempt = await _repair_symbol(
                        kind=kind, client=client, session=session, ls_client=ls_client,
                        kiwoom_client=kiwoom_client, code=code, snapshot_date=day,
                        profile=profile, store=store, run_id=run_id, apply=bool(args.apply),
                    )
                except OSError as e:
                    raise RuntimeError(f"Repair publication failed symbol={code} date={day}: {e}") from e
                record = {"date": day, "dataset": kind, **attempt}
                if attempt["status"] != CaptureStatus.COMPLETE.value:
                    record["unresolved"] = code
                report["attempts"].append(record)

    asyncio.run(_run())
    report_path = store.root / "staging" / "intraday" / f"repair-{start.isoformat()}-{end.isoformat()}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    unresolved = sorted({str(item["symbol"]) for item in report["attempts"] if "unresolved" in item})
    logger.info("[DATA] stage=repair_report range=%s..%s attempts=%d unresolved=%s", start.isoformat(), end.isoformat(), len(report["attempts"]), unresolved)


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    main()
