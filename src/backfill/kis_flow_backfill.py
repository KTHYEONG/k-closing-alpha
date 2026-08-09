"""Resumable KIS-only historical investor/program flow backfill."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.api.kis_client import KisApiClient
from src.sync.fetcher_investor import get_investor_trade_daily_async
from src.sync.fetcher_program import get_program_history_async

FLOW_COLUMNS = ("foreign_netbuy", "inst_netbuy", "program_netbuy")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowBackfillConfig:
    requests_per_second: float = 10.0
    concurrency: int = 4
    checkpoint_symbols: int = 25


@dataclass(frozen=True)
class FlowBackfillResult:
    planned_symbols: int
    completed_symbols: int
    checkpoint_paths: tuple[Path, ...]


class AsyncRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._next_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_request - now)
            self._next_request = max(now, self._next_request) + self._interval
        if wait:
            await asyncio.sleep(wait)


def plan_missing_flows(source: pd.DataFrame, checkpoint: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Return only dates where at least one flow field is still unavailable."""
    required = {"symbol", "date", *FLOW_COLUMNS}
    if not required.issubset(source.columns):
        missing = sorted(required - set(source.columns))
        raise ValueError(f"source missing required columns: {missing}")
    frame = source[["symbol", "date", *FLOW_COLUMNS]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "date"])
    if checkpoint is not None and not checkpoint.empty:
        cp = checkpoint[["symbol", "date", *FLOW_COLUMNS]].copy()
        cp["symbol"] = cp["symbol"].astype(str).str.zfill(6)
        cp["date"] = pd.to_datetime(cp["date"], errors="coerce")
        cp = cp.dropna(subset=["symbol", "date"]).groupby(["symbol", "date"], as_index=False)[list(FLOW_COLUMNS)].last()
        frame = frame.merge(cp, on=["symbol", "date"], how="left", suffixes=("", "_checkpoint"))
        for col in FLOW_COLUMNS:
            frame[col] = frame[col].fillna(frame.pop(f"{col}_checkpoint"))
    pending = frame.loc[frame[list(FLOW_COLUMNS)].isna().any(axis=1), ["symbol", "date"]]
    return {
        symbol: dates.dt.strftime("%Y%m%d").tolist()
        for symbol, dates in pending.groupby("symbol", sort=True)["date"]
    }


def _plan_missing_fields(source: pd.DataFrame, checkpoint: pd.DataFrame | None = None) -> dict[str, tuple[bool, bool]]:
    """Return per-symbol requirements as ``(investor, program)`` flags."""
    frame = source[["symbol", "date", *FLOW_COLUMNS]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if checkpoint is not None and not checkpoint.empty:
        cp = checkpoint[["symbol", "date", *FLOW_COLUMNS]].copy()
        cp["symbol"] = cp["symbol"].astype(str).str.zfill(6)
        cp["date"] = pd.to_datetime(cp["date"], errors="coerce")
        cp = cp.groupby(["symbol", "date"], as_index=False)[list(FLOW_COLUMNS)].last()
        frame = frame.merge(cp, on=["symbol", "date"], how="left", suffixes=("", "_checkpoint"))
        for col in FLOW_COLUMNS:
            frame[col] = frame[col].fillna(frame.pop(f"{col}_checkpoint"))
    result: dict[str, tuple[bool, bool]] = {}
    for symbol, group in frame.groupby("symbol", sort=True):
        result[symbol] = (
            bool(group[["foreign_netbuy", "inst_netbuy"]].isna().any().any()),
            bool(group["program_netbuy"].isna().any()),
        )
    return result


def _read_checkpoints(checkpoint_dir: Path) -> pd.DataFrame:
    paths = sorted(checkpoint_dir.glob("batch_*.parquet")) if checkpoint_dir.exists() else []
    if not paths:
        return pd.DataFrame(columns=["symbol", "date", *FLOW_COLUMNS])
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


async def _fetch_symbol(
    session: object,
    client: KisApiClient,
    limiter: AsyncRateLimiter,
    symbol: str,
    dates: list[str],
    need_investor: bool = True,
    need_program: bool = True,
) -> pd.DataFrame:
    requests = []
    if need_investor:
        requests.append(get_investor_trade_daily_async(
            session, client, symbol, min(dates), max(dates), target_dates=dates, request_slot=limiter.acquire
        ))
    if need_program:
        requests.append(get_program_history_async(
            session, client, symbol, min(dates), max(dates), target_dates=dates, request_slot=limiter.acquire
        ))
    results = await asyncio.gather(*requests) if requests else []
    investor = results[0] if need_investor else pd.DataFrame()
    program = results[1 if need_investor else 0] if need_program else {}
    base = pd.DataFrame({"symbol": symbol, "date": pd.to_datetime(dates, format="%Y%m%d")})
    if investor.empty:
        investor = pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])
    prog = pd.DataFrame({"date": pd.to_datetime(list(program), format="%Y%m%d"), "program_netbuy": list(program.values())})
    return base.merge(investor, on="date", how="left").merge(prog, on="date", how="left")


async def _fetch_symbol_guarded(
    session: object,
    client: KisApiClient,
    limiter: AsyncRateLimiter,
    semaphore: asyncio.Semaphore,
    field_plan: dict[str, tuple[bool, bool]],
    symbol: str,
    dates: list[str],
) -> pd.DataFrame:
    async with semaphore:
        need_investor, need_program = field_plan[symbol]
        try:
            return await _fetch_symbol(
                session, client, limiter, symbol, dates, need_investor, need_program
            )
        except Exception as exc:
            logger.exception(
                "[DATA] stage=flow_backfill symbol=%s status=SYMBOL_FAIL error=%s",
                symbol, type(exc).__name__,
            )
            return pd.DataFrame({
                "symbol": symbol,
                "date": pd.to_datetime(dates, format="%Y%m%d"),
                **dict.fromkeys(FLOW_COLUMNS, pd.NA),
            })


async def run_kis_flow_backfill(
    parquet_path: Path,
    checkpoint_dir: Path,
    config: FlowBackfillConfig,
    symbols: set[str] | None = None,
) -> FlowBackfillResult:
    source = pd.read_parquet(parquet_path, columns=["symbol", "date", *FLOW_COLUMNS])
    checkpoint = _read_checkpoints(checkpoint_dir)
    plan = plan_missing_flows(source, checkpoint)
    field_plan = _plan_missing_fields(source, checkpoint)
    if symbols is not None:
        requested = {str(symbol).zfill(6) for symbol in symbols}
        plan = {symbol: dates for symbol, dates in plan.items() if symbol in requested}
    if not plan:
        return FlowBackfillResult(0, 0, ())

    await asyncio.to_thread(checkpoint_dir.mkdir, parents=True, exist_ok=True)
    limiter = AsyncRateLimiter(config.requests_per_second)
    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    client = KisApiClient()
    async with client.create_session() as session:
        await client.ensure_token(session)
        tasks = [
            asyncio.create_task(
                _fetch_symbol_guarded(session, client, limiter, semaphore, field_plan, symbol, dates)
            )
            for symbol, dates in plan.items()
        ]
        completed = 0
        pending_frames: list[pd.DataFrame] = []
        paths: list[Path] = []
        batch_number = await asyncio.to_thread(lambda: len(list(checkpoint_dir.glob("batch_*.parquet"))))
        for task in asyncio.as_completed(tasks):
            frame = await task
            completed += 1
            pending_frames.append(frame)
            if len(pending_frames) >= config.checkpoint_symbols:
                path = checkpoint_dir / f"batch_{batch_number:05d}.parquet"
                await asyncio.to_thread(pd.concat(pending_frames, ignore_index=True).to_parquet, path, index=False)
                paths.append(path)
                batch_number += 1
                pending_frames.clear()
        if pending_frames:
            path = checkpoint_dir / f"batch_{batch_number:05d}.parquet"
            await asyncio.to_thread(pd.concat(pending_frames, ignore_index=True).to_parquet, path, index=False)
            paths.append(path)
    return FlowBackfillResult(len(plan), completed, tuple(paths))


def apply_flow_checkpoints(parquet_path: Path, checkpoint_dir: Path) -> int:
    """Fill missing flow fields from checkpoints without changing source row count."""
    source = pd.read_parquet(parquet_path)
    checkpoint = _read_checkpoints(checkpoint_dir)
    if checkpoint.empty:
        return 0
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["symbol"] = source["symbol"].astype(str).str.zfill(6)
    checkpoint["date"] = pd.to_datetime(checkpoint["date"], errors="coerce")
    checkpoint["symbol"] = checkpoint["symbol"].astype(str).str.zfill(6)
    checkpoint = checkpoint.groupby(["symbol", "date"], as_index=False)[list(FLOW_COLUMNS)].last()
    merged = source.merge(checkpoint, on=["symbol", "date"], how="left", suffixes=("", "_checkpoint"))
    filled = 0
    for col in FLOW_COLUMNS:
        before = merged[col].notna()
        merged[col] = merged[col].fillna(merged.pop(f"{col}_checkpoint"))
        filled += int((~before & merged[col].notna()).sum())
    merged.to_parquet(parquet_path, index=False)
    return filled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # pragma: no cover
    parser = argparse.ArgumentParser(description="KIS flow-only historical backfill")
    parser.add_argument("--parquet", default="data/history/price_history.parquet")
    parser.add_argument("--checkpoint-dir", default="data/history/flow_backfill")
    parser.add_argument("--rps", type=float, default=10.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--checkpoint-symbols", type=int, default=25)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--flows-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    parquet = Path(args.parquet)
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.apply:
        logger.info("[DATA] stage=flow_backfill_apply filled=%d", apply_flow_checkpoints(parquet, checkpoint_dir))
        return
    symbols = {value.strip().zfill(6) for value in args.symbols.split(",") if value.strip()} or None
    result = asyncio.run(run_kis_flow_backfill(parquet, checkpoint_dir, FlowBackfillConfig(args.rps, args.concurrency, args.checkpoint_symbols), symbols))
    logger.info(
        "[DATA] stage=flow_backfill planned=%d completed=%d checkpoints=%d",
        result.planned_symbols,
        result.completed_symbols,
        len(result.checkpoint_paths),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
