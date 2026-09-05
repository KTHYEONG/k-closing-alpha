"""종목 단위 fetch / 전체 backfill 실행 조합."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.backfill.price.config import FetchConfig, _effective_kis_sleep_sec
from src.backfill.price.factors import _merge_index_returns
from src.backfill.price.normalize import (
    _find_col,
    _normalize_symbol_history,
    _to_ymd,
)
from src.backfill.price.sources import (
    _fetch_investor_history_by_date,
    _fetch_kis_daily_ohlcv,
    _fetch_program_history_by_date,
    _safe_get_market_cap_by_date,
    _safe_get_market_ohlcv_by_date,
)
from src.backfill.price.universe import (
    _build_symbol_windows,
    _load_candidate_universe,
)
from src.data.parquet_codec import write_price_history_parquet

logger = logging.getLogger(__name__)

try:
    from pykrx import stock
except ImportError:  # pragma: no cover - dependency availability differs by env
    stock = None


def fetch_one_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    market_hint: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    start_s = _to_ymd(start)
    end_s = _to_ymd(end)
    
    ohlcv = _safe_get_market_ohlcv_by_date(start_s, end_s, symbol, fetch_cfg)
    cap = _safe_get_market_cap_by_date(start_s, end_s, symbol, fetch_cfg)
    
    # Check if we need KIS fallback for missing columns or empty data
    needs_kis = False
    if ohlcv is None or ohlcv.empty:
        needs_kis = True
    else:
        # Check if trading value is missing (it's essential for this backfill)
        has_val = _find_col(ohlcv, ["거래대금", "거래금액", "Value", "TradingValue"])
        if not has_val:
            needs_kis = True
    
    if cap is None or cap.empty:
        needs_kis = True
        
    if needs_kis:
        kis_df = _fetch_kis_daily_ohlcv(symbol, start, end, fetch_cfg)

        if not kis_df.empty:
            kis_indexed = kis_df.set_index("date")
            if ohlcv is None or ohlcv.empty:
                ohlcv = kis_indexed
            else:
                # Merge KIS columns into ohlcv if missing
                for col in ["trade_value_krw", "market_cap_krw", "daily_change_pct"]:
                    if col in kis_df.columns:
                        if col not in ohlcv.columns:
                            ohlcv[col] = kis_indexed[col]
                        else:
                            ohlcv[col] = ohlcv[col].fillna(kis_indexed[col])

            if cap is None or cap.empty:
                cap = kis_df[["date", "market_cap_krw"]].rename(columns={"market_cap_krw": "시가총액"}).set_index("date")

    norm = _normalize_symbol_history(ohlcv, cap, symbol, market_hint)
    if norm.empty:
        return norm

    target_dates = (
        norm["date"].dropna().dt.strftime("%Y%m%d").drop_duplicates().tolist()
        if "date" in norm.columns
        else None
    )
    if fetch_cfg.include_flows:
        flow_norm = _fetch_investor_history_by_date(symbol, start, end, fetch_cfg, target_dates=target_dates)
        prog_norm = _fetch_program_history_by_date(symbol, start, end, fetch_cfg, target_dates=target_dates)
    else:
        flow_norm = pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])
        prog_norm = pd.DataFrame(columns=["date", "program_netbuy"])

    out = norm.merge(flow_norm, on="date", how="left")
    out = out.merge(prog_norm, on="date", how="left")
    for col in ["foreign_netbuy", "inst_netbuy", "program_netbuy"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _to_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        old = pd.read_parquet(parquet_path)
        merged = pd.concat([old, df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.dropna(subset=["date", "symbol"])
        merged = merged.sort_values(["symbol", "date"]).drop_duplicates(
            subset=["symbol", "date"], keep="last"
        )
        write_price_history_parquet(merged, parquet_path)
    else:
        write_price_history_parquet(df, parquet_path)


def _load_existing_symbol_last_dates(parquet_path: Path) -> dict[str, pd.Timestamp]:
    if parquet_path is None or not parquet_path.exists():
        return {}
    try:
        old = pd.read_parquet(parquet_path, columns=["symbol", "date"])
    except Exception as exc:
        logger.warning("[DATA] stage=backfill_date_map path=%s status=FAIL error=%s", parquet_path, exc)
        return {}
    if old is None or old.empty:
        return {}
    if "symbol" not in old.columns or "date" not in old.columns:
        return {}
    old = old.copy()
    old["symbol"] = old["symbol"].astype(str).str.strip().str.zfill(6)
    old["date"] = pd.to_datetime(old["date"], errors="coerce")
    old = old.dropna(subset=["symbol", "date"])
    if old.empty:
        return {}
    max_date = old.groupby("symbol", sort=False)["date"].max()
    return {str(k): pd.Timestamp(v).normalize() for k, v in max_date.items()}


def run_backfill(
    *,
    lookback_trading_days: int,
    max_workers: int,
    kis_rest_limit_per_sec: float,
    kis_rest_safety_ratio: float,
    kis_max_parallel_calls: int,
    pykrx_requests_per_sec: float = 8.0,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    include_flows: bool = True,
    symbol_limit: int | None,
    include_symbols: set[str] | None,
    parquet_out: Path,
) -> pd.DataFrame:
    if stock is None:
        raise RuntimeError("pykrx is required. Install with: python -m pip install pykrx")

    fetch_cfg = FetchConfig(
        lookback_trading_days=max(1, int(lookback_trading_days)),
        calendar_buffer_days=max(30, int(lookback_trading_days) * 3),
        max_workers=max(1, int(max_workers)),
        kis_rest_limit_per_sec=max(1.0, float(kis_rest_limit_per_sec)),
        kis_rest_safety_ratio=min(1.0, max(0.1, float(kis_rest_safety_ratio))),
        kis_max_parallel_calls=max(1, int(kis_max_parallel_calls)),
        pykrx_requests_per_sec=max(0.1, float(pykrx_requests_per_sec)),
        fixed_start_date=pd.Timestamp(start_date) if start_date is not None else pd.Timestamp("2016-01-01"),
        fixed_end_date=pd.Timestamp(end_date) if end_date is not None else pd.Timestamp("2025-12-31"),
        include_flows=bool(include_flows),
        force_full_history=start_date is not None,
    )
    universe = _load_candidate_universe()
    if universe.empty:
        raise RuntimeError("No symbols found in snapshot universe.")

    existing_last_dates = _load_existing_symbol_last_dates(parquet_out)
    windows = _build_symbol_windows(
        universe,
        fetch_cfg=fetch_cfg,
        symbol_limit=symbol_limit,
        include_symbols=include_symbols,
        existing_last_dates=existing_last_dates,
    )
    if not windows:
        logger.info("[DATA] stage=backfill status=no_fetch_windows")
        if parquet_out.exists():
            try:
                old = pd.read_parquet(parquet_out)
                if "date" in old.columns:
                    old["date"] = pd.to_datetime(old["date"], errors="coerce")
                logger.info("[DATA] stage=backfill status=skip_existing path=%s", parquet_out)
                return old
            except Exception as exc:
                logger.warning("[DATA] stage=backfill_existing path=%s status=FAIL error=%s", parquet_out, exc)
        return pd.DataFrame()

    logger.info(
        "[DATA] stage=backfill symbols=%d workers=%d pykrx_rps=%.2f kis_limit=%.2f safety=%.2f kis_parallel=%d kis_sleep=%.3f existing_symbols=%d",
        len(windows), fetch_cfg.max_workers, fetch_cfg.pykrx_requests_per_sec, fetch_cfg.kis_rest_limit_per_sec,
        fetch_cfg.kis_rest_safety_ratio, fetch_cfg.kis_max_parallel_calls,
        _effective_kis_sleep_sec(fetch_cfg), len(existing_last_dates),
    )
    chunks: list[pd.DataFrame] = []
    checkpoint_chunks: list[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=fetch_cfg.max_workers) as ex:
        futures = {
            ex.submit(fetch_one_symbol, sym, s, e, mkt, fetch_cfg): (sym, s, e)
            for sym, s, e, mkt in windows
        }
        for fut in as_completed(futures):
            sym, s, e = futures[fut]
            done += 1
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    chunks.append(df)
                    checkpoint_chunks.append(df)
                    if done % 100 == 0:
                        _to_parquet(pd.concat(checkpoint_chunks, ignore_index=True), parquet_out)
                        checkpoint_chunks.clear()
                        logger.info("[DATA] stage=backfill checkpoint=%d/%d path=%s", done, len(windows), parquet_out)
                logger.info("[DATA] stage=backfill symbol=%s progress=%d/%d rows=%d", sym, done, len(windows), 0 if df is None else len(df))
            except Exception as exc:
                logger.warning("[DATA] stage=backfill symbol=%s progress=%d/%d status=FAIL error=%s", sym, done, len(windows), exc)

    if not chunks:
        raise RuntimeError("No symbol history fetched. Check API/network constraints.")

    history = pd.concat(chunks, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date", "symbol"])
    history = history.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    history = _merge_index_returns(history, fetch_cfg=fetch_cfg)

    _to_parquet(history, parquet_path=parquet_out)
    logger.info("[DATA] stage=backfill path=%s status=saved", parquet_out)

    logger.info(
        "[DATA] stage=backfill status=done rows=%d symbols=%d date_start=%s date_end=%s",
        len(history), history["symbol"].nunique(), history["date"].min().date(), history["date"].max().date(),
    )
    return history


def preview_windows(
    *,
    lookback_trading_days: int,
    symbol_limit: int | None,
    include_symbols: set[str] | None,
    parquet_out: Path | None = None,
) -> pd.DataFrame:
    fetch_cfg = FetchConfig(
        lookback_trading_days=max(1, int(lookback_trading_days)),
        calendar_buffer_days=max(30, int(lookback_trading_days) * 3),
        max_workers=1,
    )
    universe = _load_candidate_universe()
    existing_last_dates = _load_existing_symbol_last_dates(parquet_out) if parquet_out else None
    windows = _build_symbol_windows(
        universe,
        fetch_cfg=fetch_cfg,
        symbol_limit=symbol_limit,
        include_symbols=include_symbols,
        existing_last_dates=existing_last_dates,
    )
    if not windows:
        return pd.DataFrame(columns=["symbol", "fetch_start", "fetch_end", "market"])
    out = pd.DataFrame(windows, columns=["symbol", "fetch_start", "fetch_end", "market"])
    out["fetch_start"] = pd.to_datetime(out["fetch_start"], errors="coerce")
    out["fetch_end"] = pd.to_datetime(out["fetch_end"], errors="coerce")
    return out.sort_values("symbol").reset_index(drop=True)
