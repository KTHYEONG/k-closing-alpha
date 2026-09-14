"""PIT rebuild of price_history from KRX daily all-listed rows (basDd) with base-price corporate-action chain, raw close preserved, flows carried from the previous panel, KIS composite index columns."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

from src import settings
from src.backfill.altdata.config import AltDataFetchConfig
from src.daily.price_ingest import (
    FLOW_COLUMNS,
    INDEX_HISTORY_START,
    KIS_INDEX_KOSDAQ_CODE,
    KIS_INDEX_KOSPI_CODE,
    KRX_ROW_COLUMNS,
    PRICE_EVENT_TOLERANCE,
    attach_index_columns,
    compute_index_columns,
    fetch_index_closes,
    fetch_krx_daily,
)
from src.data.panel_integrity import heal_price_history_panel
from src.data.parquet_codec import write_price_history_parquet

logger = logging.getLogger(__name__)

# KRX Open API 일별 전종목 응답이 확인된 첫 거래일이자 기존 패널 시작일
REBUILD_START_DATE: pd.Timestamp = pd.Timestamp("2016-01-04")
# 기존 pykrx 수정주가와 공통행 일치율 하한. 실측 2018-04~05 창 98.7% 종목 정합, 나머지는 정수 반올림
REBUILD_MIN_MATCH_RATE: float = 0.98
# pykrx 정수 반올림(≤0.03%)을 흡수하는 상대 허용오차
REBUILD_MATCH_RTOL: float = 3e-3
_ADJUSTED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "prev_close")
_PANEL_COLUMNS: tuple[str, ...] = ("date", "symbol", "open", "high", "low", "close", "prev_close", "close_raw", "volume", "trade_value_100m", "market_cap_100m", "market")


@dataclass(frozen=True)
class RebuildReport:
    """Summary of one PIT panel rebuild."""

    n_rows: int
    n_symbols: int
    n_dates: int
    match_rate: float
    n_common_rows: int


def fetch_krx_history(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: AltDataFetchConfig,
    checkpoint_dir: Path,
    *,
    fetch_fn: Callable[[pd.Timestamp, AltDataFetchConfig], pd.DataFrame] = fetch_krx_daily,
) -> pd.DataFrame:
    """Fetch every KRX daily all-listed row in [start, end] with per-date checkpoints.

    Args:
        start: First business day to cover.
        end: Last business day to cover.
        cfg: Alt-data config carrying krx_api_key.
        checkpoint_dir: Directory holding one parquet per business day.
        fetch_fn: One-day fetcher used for cache misses.

    Returns:
        Concatenated rows in KRX_ROW_COLUMNS; holidays contribute no rows but
        keep an empty checkpoint so a resumed run never re-fetches them.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    n_fetched = 0
    for day in pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()):
        path = checkpoint_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
        if path.exists():
            rows = pd.read_parquet(path)
        else:
            rows = fetch_fn(day, cfg)
            rows = rows.reindex(columns=list(KRX_ROW_COLUMNS))
            # 휴장일(0행)도 체크포인트로 남겨 재개 시 재호출하지 않는다
            rows.to_parquet(path, index=False)
            n_fetched += 1
        if not rows.empty:
            frames.append(rows)
    logger.info("[DATA] stage=krx_panel_rebuild n_days_fetched=%d n_trading_days=%d", n_fetched, len(frames))
    if frames:
        out = pd.concat(frames, ignore_index=True)[list(KRX_ROW_COLUMNS)]
    else:
        out = pd.DataFrame(columns=list(KRX_ROW_COLUMNS))
    out["date"] = pd.to_datetime(out["date"])
    return out


def derive_adjusted_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the adjusted OHLC/prev_close chain from KRX base prices.

    KRX base price differs from the prior close only on corporate-action
    ex-dates; the backward cumulative product of base/prior_close reproduces
    the adjusted chain. Volume and close_raw are never scaled.

    Args:
        raw: KRX daily rows with KRX_ROW_COLUMNS.

    Returns:
        Panel rows in _PANEL_COLUMNS with close_raw preserved.
    """
    out = raw.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    for col in (*_ADJUSTED_COLUMNS, "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["close_raw"] = out["close"].to_numpy()
    cal = pd.DatetimeIndex(sorted(out["date"].unique()))
    g = out.groupby("symbol", sort=False)
    prior_close = g["close"].shift(1)
    prior_date = g["date"].shift(1)
    pos = cal.searchsorted(out["date"].to_numpy())
    prev_trading = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    has_prev = pos > 0
    prev_trading[has_prev] = cal[pos[has_prev] - 1]
    event = (prior_date == prev_trading) & (prior_close > 0) & ((out["prev_close"] - prior_close).abs() > PRICE_EVENT_TOLERANCE)
    factor = pd.Series(1.0, index=out.index)
    factor[event] = (out["prev_close"] / prior_close)[event]
    rev = factor[::-1].groupby(out["symbol"][::-1], sort=False).cumprod()[::-1]
    scale = (rev / factor).astype("float64")
    touched = scale != 1.0
    for col in _ADJUSTED_COLUMNS:
        out.loc[touched, col] = out.loc[touched, col] * scale[touched]
    logger.info("[DATA] stage=krx_panel_rebuild n_events=%d n_symbols_touched=%d", int(event.sum()), int(out.loc[touched, "symbol"].nunique()))
    return out[list(_PANEL_COLUMNS)]


def carry_existing_flows(panel: pd.DataFrame, old_panel: pd.DataFrame) -> pd.DataFrame:
    """Copy flows from the old panel onto matching (date, symbol) rows; other rows keep NaN flows and the row count is unchanged."""
    src = old_panel[["date", "symbol", *FLOW_COLUMNS]].copy()
    src["date"] = pd.to_datetime(src["date"])
    src["symbol"] = src["symbol"].astype(str)
    src = src.drop_duplicates(["date", "symbol"], keep="last")
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    return out.merge(src, on=["date", "symbol"], how="left", validate="one_to_one")


def validate_rebuild(new: pd.DataFrame, old: pd.DataFrame) -> dict[str, float | int]:
    """Validate a rebuilt panel against the old panel before writing.

    Args:
        new: Rebuilt panel candidate.
        old: Currently live panel.

    Returns:
        Coverage metrics with n_rows, n_symbols, n_common_rows, match_rate, n_dates.

    Raises:
        ValueError: On duplicate keys, per-date shrink, non-positive
            close_raw, or adjusted-close match rate below REBUILD_MIN_MATCH_RATE.
    """
    keys = ["date", "symbol"]
    n_dup = int(new.duplicated(keys).sum())
    if n_dup:
        raise ValueError(f"rebuilt panel has {n_dup} duplicate (date, symbol) rows")
    raw = pd.to_numeric(new["close_raw"], errors="coerce")
    close = pd.to_numeric(new["close"], errors="coerce")
    n_bad = int(((close > 0) & ~(raw > 0)).sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have close > 0 but non-positive close_raw")
    # 새 패널은 KRX 일별 전종목(코스피·코스닥) API만을 원천으로 하므로, 이전 패널에 섞여
    # 들어간 비-코스피/코스닥 표식 행(ETF 등, 후보 스캔 경유로 우연히 백필된 스코프 밖 데이터)은
    # 행수 감소 비교에서 제외한다.
    old_scope = old[old["market"].isin(("KOSPI", "KOSDAQ"))] if "market" in old.columns else old
    old_counts = old_scope.groupby(pd.to_datetime(old_scope["date"])).size()
    new_counts = new.groupby(pd.to_datetime(new["date"])).size()
    common_dates = old_counts.index.intersection(new_counts.index)
    shrunk = [d for d in common_dates if new_counts[d] < old_counts[d]]
    if shrunk:
        raise ValueError(f"rebuilt panel has fewer rows than the old panel on {len(shrunk)} dates, first {shrunk[0].date()}")
    new_k = new[[*keys, "close"]].copy()
    old_k = old[[*keys, "close"]].copy()
    new_k["date"] = pd.to_datetime(new_k["date"])
    old_k["date"] = pd.to_datetime(old_k["date"])
    new_k["symbol"] = new_k["symbol"].astype(str)
    old_k["symbol"] = old_k["symbol"].astype(str)
    m = new_k.merge(old_k, on=keys, suffixes=("_new", "_old"))
    ok = np.isclose(m["close_new"].astype(float), m["close_old"].astype(float), rtol=REBUILD_MATCH_RTOL)
    match_rate = float(ok.mean()) if len(m) else 1.0
    if match_rate < REBUILD_MIN_MATCH_RATE:
        raise ValueError(f"adjusted close match rate {match_rate:.4f} below {REBUILD_MIN_MATCH_RATE}")
    return {"n_rows": len(new), "n_symbols": int(new["symbol"].nunique()), "n_common_rows": int(len(m)), "match_rate": match_rate, "n_dates": int(new_counts.size)}


async def run_panel_rebuild(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    checkpoint_dir: Path,
    out_path: Path,
    old_path: Path | None = None,
    krx_cfg: AltDataFetchConfig | None = None,
    kis: Any | None = None,
) -> RebuildReport:
    """Rebuild the price_history panel from KRX daily rows and write it to out_path.

    Args:
        start: First business day to rebuild.
        end: Last business day to rebuild.
        checkpoint_dir: Per-date KRX checkpoint directory.
        out_path: Destination parquet (never the live path).
        old_path: Old panel parquet; None selects the live path.
        krx_cfg: KRX config; None builds one from settings.KRX_OPENAPI_KEY.
        kis: KIS client; None builds the data-account client.

    Returns:
        RebuildReport describing the validated rebuild.

    Raises:
        FileNotFoundError: When the old panel does not exist.
        ValueError: When KRX returns no rows, or propagated from validate_rebuild.
    """
    old_p = Path(settings.PRICE_HISTORY_PARQUET_PATH if old_path is None else old_path)
    if not await asyncio.to_thread(old_p.exists):
        raise FileNotFoundError(f"old panel not found: {old_p}")
    cfg = krx_cfg or AltDataFetchConfig(start=pd.Timestamp(start), end=pd.Timestamp(end) + pd.Timedelta(days=1), out_dir=Path("."), krx_api_key=settings.KRX_OPENAPI_KEY)
    if kis is None:
        from src.api.kis.client import KisApiClient, kis_data_client_kwargs

        kis = KisApiClient(**kis_data_client_kwargs())
    raw = fetch_krx_history(start, end, cfg, Path(checkpoint_dir), fetch_fn=fetch_krx_daily)
    if raw.empty:
        raise ValueError("KRX returned no rows for the requested range")
    old = pd.read_parquet(old_p)
    async with aiohttp.ClientSession() as session:
        await kis.ensure_token(session)
        kospi = await fetch_index_closes(kis, session, KIS_INDEX_KOSPI_CODE, INDEX_HISTORY_START, pd.Timestamp(end))
        kosdaq = await fetch_index_closes(kis, session, KIS_INDEX_KOSDAQ_CODE, INDEX_HISTORY_START, pd.Timestamp(end))
    panel = carry_existing_flows(derive_adjusted_prices(raw), old)
    panel = heal_price_history_panel(panel)
    panel = attach_index_columns(panel, compute_index_columns(kospi, kosdaq))
    metrics = validate_rebuild(panel, old)
    write_price_history_parquet(panel, Path(out_path))
    logger.info("[DATA] stage=krx_panel_rebuild status=WRITTEN path=%s %s", out_path, metrics)
    return RebuildReport(n_rows=int(metrics["n_rows"]), n_symbols=int(metrics["n_symbols"]), n_dates=int(metrics["n_dates"]), match_rate=float(metrics["match_rate"]), n_common_rows=int(metrics["n_common_rows"]))


def swap_panel(rebuilt: Path, live: Path) -> Path:
    """Atomically replace the live panel with a rebuilt file via a timestamped backup.

    Args:
        rebuilt: Validated rebuild parquet.
        live: Live panel path.

    Returns:
        Backup path holding the previous live file.

    Raises:
        FileNotFoundError: When the rebuilt file does not exist.
    """
    rebuilt = Path(rebuilt)
    live = Path(live)
    if not rebuilt.exists():
        raise FileNotFoundError(f"rebuilt panel not found: {rebuilt}")
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = live.with_name(f"{live.stem}.{stamp}.bak.parquet")
    if live.exists():
        os.replace(live, backup)
    os.replace(rebuilt, live)
    logger.info("[DATA] stage=krx_panel_rebuild status=SWAPPED live=%s backup=%s", live, backup)
    return backup


def main(argv: list[str] | None = None) -> None:
    """Rebuild the panel to a sidecar path and optionally swap it live."""
    parser = argparse.ArgumentParser(description="PIT rebuild of price_history from KRX daily rows")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--swap", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    live = Path(settings.PRICE_HISTORY_PARQUET_PATH)
    start = REBUILD_START_DATE if args.start is None else pd.Timestamp(args.start)
    end = pd.Timestamp.today().normalize() if args.end is None else pd.Timestamp(args.end)
    out = live.with_name("price_history_rebuild.parquet") if args.out is None else Path(args.out)
    report = asyncio.run(run_panel_rebuild(start=start, end=end, checkpoint_dir=Path(args.checkpoint_dir), out_path=out, old_path=live))
    logger.info("[DATA] stage=krx_panel_rebuild report=%s", report)
    if args.swap:
        swap_panel(out, live)


if __name__ == "__main__":  # pragma: no cover
    main()
