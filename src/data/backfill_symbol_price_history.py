"""종목별 과거 가격/거래대금/수급/지수연계 보조 데이터를 로컬 히스토리로 백필하는 스크립트.

다중 종목 히스토리를 병렬 수집해 parquet 저장소를 갱신하며,
롤링 피처 계산에 필요한 과거 컨텍스트를 제공한다.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
try:
    from pykrx import stock
except ImportError:  # pragma: no cover - dependency availability differs by env
    stock = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.pipeline.config import DEFAULT_CONFIG
    from src.pipeline.data import load_or_build_snapshot
except ImportError:
    # Fallback for when pipeline module is missing
    DEFAULT_CONFIG = None
    def load_or_build_snapshot(*args, **kwargs):
        return pd.DataFrame()


KRW_100M = 100_000_000.0


@dataclass(frozen=True)
class FetchConfig:
    lookback_trading_days: int = 40
    calendar_buffer_days: int = 120
    max_workers: int = 4
    retries: int = 3
    retry_sleep_sec: float = 0.8
    request_sleep_sec: float = 0.03
    fixed_start_date: pd.Timestamp = pd.Timestamp("2016-01-01")
    fixed_end_date: pd.Timestamp = pd.Timestamp("2025-12-31")
    # KIS official samples commonly state REST guidance around 20 req/s.
    # Keep a conservative default to protect stability under parallel backfill.
    kis_rest_limit_per_sec: float = 20.0
    kis_rest_safety_ratio: float = 0.6
    kis_max_parallel_calls: int = 1


_PROGRAM_HISTORY_FN = None
_PROGRAM_HISTORY_RESOLVED = False
_INVESTOR_HISTORY_FN = None
_INVESTOR_HISTORY_RESOLVED = False
_KIS_SEMAPHORE: Optional[threading.Semaphore] = None
_KIS_SEMAPHORE_SIZE = 0


def _effective_kis_sleep_sec(fetch_cfg: FetchConfig) -> float:
    safe_rps = max(1e-6, float(fetch_cfg.kis_rest_limit_per_sec) * float(fetch_cfg.kis_rest_safety_ratio))
    return max(float(fetch_cfg.request_sleep_sec), 1.0 / safe_rps)


def _ensure_kis_semaphore(fetch_cfg: FetchConfig) -> threading.Semaphore:
    global _KIS_SEMAPHORE, _KIS_SEMAPHORE_SIZE
    size = max(1, int(fetch_cfg.kis_max_parallel_calls))
    if _KIS_SEMAPHORE is None or _KIS_SEMAPHORE_SIZE != size:
        _KIS_SEMAPHORE = threading.Semaphore(size)
        _KIS_SEMAPHORE_SIZE = size
    return _KIS_SEMAPHORE


@contextmanager
def _kis_slot(fetch_cfg: FetchConfig):
    sem = _ensure_kis_semaphore(fetch_cfg)
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(map(str, df.columns))
    for c in candidates:
        if c in cols:
            return c
    return None


def _sum_present_cols(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return None
    part = df[cols].apply(pd.to_numeric, errors="coerce")
    return part.sum(axis=1, min_count=1)


def _subtract_flow_frames(buy: pd.DataFrame, sell: pd.DataFrame) -> pd.DataFrame:
    if buy is None or sell is None or buy.empty or sell.empty:
        return pd.DataFrame()

    b = buy.copy()
    s = sell.copy()
    b.index = pd.to_datetime(b.index, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    b = b[~b.index.isna()].copy()
    s = s[~s.index.isna()].copy()
    if b.empty or s.empty:
        return pd.DataFrame()

    common_idx = b.index.intersection(s.index)
    common_cols = [c for c in b.columns if c in s.columns]
    if len(common_idx) == 0 or not common_cols:
        return pd.DataFrame()

    out = pd.DataFrame(index=common_idx)
    for c in common_cols:
        out[c] = pd.to_numeric(b.loc[common_idx, c], errors="coerce") - pd.to_numeric(
            s.loc[common_idx, c], errors="coerce"
        )
    out = out.sort_index().dropna(how="all")
    return out


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _load_candidate_universe() -> pd.DataFrame:
    raw = load_or_build_snapshot(config=DEFAULT_CONFIG, rebuild=False, sync_gsheet=False)
    required = ["symbol", "market"]
    for col in required:
        if col not in raw.columns:
            raw[col] = np.nan
    out = raw[required].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.zfill(6)
    out["market"] = out["market"].astype(str).fillna("UNKNOWN")
    out = out[out["symbol"].str.fullmatch(r"\d{6}", na=False)].copy()
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out


def _build_symbol_windows(
    universe: pd.DataFrame,
    fetch_cfg: FetchConfig,
    symbol_limit: Optional[int] = None,
    include_symbols: Optional[Set[str]] = None,
    existing_last_dates: Optional[Dict[str, pd.Timestamp]] = None,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    by_symbol = universe[["symbol"]].dropna().drop_duplicates().sort_values("symbol")
    market_hint = (
        universe[["symbol", "market"]]
        .dropna()
        .drop_duplicates(subset=["symbol"], keep="last")
        .set_index("symbol")["market"]
        .to_dict()
    )

    if symbol_limit is not None and symbol_limit > 0:
        by_symbol = by_symbol.head(symbol_limit)
    if include_symbols:
        wanted = {str(s).strip().zfill(6) for s in include_symbols if str(s).strip()}
        by_symbol = by_symbol[by_symbol["symbol"].isin(wanted)].copy()
    else:
        wanted = set()

    rows: List[Tuple[str, pd.Timestamp, pd.Timestamp, str]] = []
    start_fixed = pd.Timestamp(fetch_cfg.fixed_start_date)
    end_fixed = min(pd.Timestamp(fetch_cfg.fixed_end_date), pd.Timestamp.today().normalize())
    if start_fixed > end_fixed:
        start_fixed, end_fixed = end_fixed, start_fixed

    existing = existing_last_dates or {}
    overlap_days = max(20, int(fetch_cfg.calendar_buffer_days))

    def _resolve_start(symbol: str) -> pd.Timestamp:
        last_dt = existing.get(str(symbol))
        if last_dt is None:
            return start_fixed
        last_ts = pd.to_datetime(last_dt, errors="coerce")
        if pd.isna(last_ts):
            return start_fixed
        return max(start_fixed, pd.Timestamp(last_ts).normalize() - pd.Timedelta(days=overlap_days))

    for rec in by_symbol.itertuples(index=False):
        symbol = str(rec.symbol)
        start_ts = _resolve_start(symbol)
        if start_ts > end_fixed:
            continue
        rows.append((symbol, start_ts, end_fixed, str(market_hint.get(symbol, "UNKNOWN"))))

    # If a user-requested symbol is missing in snapshot universe, still build a fallback
    # range so manual/backfill test can proceed.
    if wanted:
        existing_symbols = {r[0] for r in rows}
        missing = sorted(wanted - existing_symbols)
        if missing:
            for sym in missing:
                start_ts = _resolve_start(sym)
                if start_ts > end_fixed:
                    continue
                rows.append((sym, start_ts, end_fixed, "UNKNOWN"))
    return rows


def _safe_get_market_ohlcv_by_date(
    from_date: str,
    to_date: str,
    symbol: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for attempt in range(fetch_cfg.retries):
        try:
            time.sleep(fetch_cfg.request_sleep_sec)
            return stock.get_market_ohlcv_by_date(from_date, to_date, symbol)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_err = exc
            time.sleep(fetch_cfg.retry_sleep_sec * (attempt + 1))
    raise RuntimeError(f"ohlcv fetch failed for {symbol}: {last_err}")


def _safe_get_market_cap_by_date(
    from_date: str,
    to_date: str,
    symbol: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for attempt in range(fetch_cfg.retries):
        try:
            time.sleep(fetch_cfg.request_sleep_sec)
            return stock.get_market_cap_by_date(from_date, to_date, symbol)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_err = exc
            time.sleep(fetch_cfg.retry_sleep_sec * (attempt + 1))
    # market cap fetch failure is non-fatal for history table.
    print(f"[warn] market cap fetch failed for {symbol}: {last_err}")
    return pd.DataFrame()


def _safe_get_trading_value_by_date(
    from_date: str,
    to_date: str,
    symbol: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    def _request(on: str, detail: bool) -> pd.DataFrame:
        last_err: Optional[Exception] = None
        for attempt in range(fetch_cfg.retries):
            try:
                time.sleep(fetch_cfg.request_sleep_sec)
                return stock.get_market_trading_value_by_date(
                    from_date,
                    to_date,
                    symbol,
                    on=on,
                    detail=detail,
                    freq="d",
                )
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_err = exc
                time.sleep(fetch_cfg.retry_sleep_sec * (attempt + 1))
        print(
            f"[warn] investor flow fetch failed for {symbol} "
            f"(on={on}, detail={detail}): {last_err}"
        )
        return pd.DataFrame()

    net = _request(on="순매수", detail=False)
    if net is not None and not net.empty:
        return net

    net_detail = _request(on="순매수", detail=True)
    if net_detail is not None and not net_detail.empty:
        return net_detail

    buy = _request(on="매수", detail=False)
    sell = _request(on="매도", detail=False)
    delta = _subtract_flow_frames(buy, sell)
    if not delta.empty:
        return delta

    buy_detail = _request(on="매수", detail=True)
    sell_detail = _request(on="매도", detail=True)
    delta_detail = _subtract_flow_frames(buy_detail, sell_detail)
    if not delta_detail.empty:
        return delta_detail

    print(f"[warn] investor flow empty after all fallbacks for {symbol}")
    return pd.DataFrame()


def _resolve_program_history_func():
    global _PROGRAM_HISTORY_FN, _PROGRAM_HISTORY_RESOLVED
    if _PROGRAM_HISTORY_RESOLVED:
        return _PROGRAM_HISTORY_FN
    _PROGRAM_HISTORY_RESOLVED = True
    candidates = [
        "src.etc.program_data",
        "etc.program_data",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "get_program_history", None)
            if callable(fn):
                _PROGRAM_HISTORY_FN = fn
                return _PROGRAM_HISTORY_FN
        except Exception:
            continue
    return None


def _resolve_investor_history_func():
    global _INVESTOR_HISTORY_FN, _INVESTOR_HISTORY_RESOLVED
    if _INVESTOR_HISTORY_RESOLVED:
        return _INVESTOR_HISTORY_FN
    _INVESTOR_HISTORY_RESOLVED = True
    candidates = [
        "src.etc.investor_data",
        "etc.investor_data",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "get_investor_trade_daily", None)
            if callable(fn):
                _INVESTOR_HISTORY_FN = fn
                return _INVESTOR_HISTORY_FN
        except Exception:
            continue
    return None


def _fetch_program_history_by_date(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_cfg: FetchConfig,
    target_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    fn = _resolve_program_history_func()
    if fn is None:
        return pd.DataFrame(columns=["date", "program_netbuy"])
    kis_sleep_sec = _effective_kis_sleep_sec(fetch_cfg)
    try:
        with _kis_slot(fetch_cfg):
            try:
                prog_map = fn(
                    symbol,
                    _to_ymd(start),
                    _to_ymd(end),
                    target_dates=target_dates,
                    sleep_sec=kis_sleep_sec,
                )
            except TypeError:
                try:
                    prog_map = fn(
                        symbol,
                        _to_ymd(start),
                        _to_ymd(end),
                        target_dates=target_dates,
                    )
                except TypeError:
                    prog_map = fn(symbol, _to_ymd(start), _to_ymd(end))
    except Exception as exc:
        print(f"[warn] program flow fetch failed for {symbol}: {exc}")
        return pd.DataFrame(columns=["date", "program_netbuy"])

    rows = []
    for k, v in (prog_map or {}).items():
        dt = pd.to_datetime(str(k), format="%Y%m%d", errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({"date": dt, "program_netbuy": pd.to_numeric(v, errors="coerce")})
    if not rows:
        return pd.DataFrame(columns=["date", "program_netbuy"])
    return pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last")


def _fetch_investor_history_by_date(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_cfg: FetchConfig,
    target_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    fn = _resolve_investor_history_func()
    if fn is not None:
        kis_sleep_sec = _effective_kis_sleep_sec(fetch_cfg)
        try:
            with _kis_slot(fetch_cfg):
                try:
                    out = fn(
                        symbol,
                        _to_ymd(start),
                        _to_ymd(end),
                        target_dates=target_dates,
                        sleep_sec=kis_sleep_sec,
                    )
                except TypeError:
                    try:
                        out = fn(
                            symbol,
                            _to_ymd(start),
                            _to_ymd(end),
                            target_dates=target_dates,
                        )
                    except TypeError:
                        out = fn(symbol, _to_ymd(start), _to_ymd(end))
            if isinstance(out, pd.DataFrame) and not out.empty:
                cols = set(map(str, out.columns))
                if {"date", "foreign_netbuy", "inst_netbuy"}.issubset(cols):
                    part = out[["date", "foreign_netbuy", "inst_netbuy"]].copy()
                    part["date"] = pd.to_datetime(part["date"], errors="coerce")
                    part["foreign_netbuy"] = pd.to_numeric(part["foreign_netbuy"], errors="coerce")
                    part["inst_netbuy"] = pd.to_numeric(part["inst_netbuy"], errors="coerce")
                    part = part.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                    return part
        except Exception as exc:
            print(f"[warn] investor flow KIS fetch failed for {symbol}: {exc}")

    # fallback: pykrx (may be empty due KRX endpoint/session changes)
    flow = _safe_get_trading_value_by_date(_to_ymd(start), _to_ymd(end), symbol, fetch_cfg)
    return _normalize_investor_flow(flow)


def _normalize_investor_flow(flow_df: pd.DataFrame) -> pd.DataFrame:
    if flow_df is None or flow_df.empty:
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    out = flow_df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    out["date"] = out.index

    def _pick_by_keywords(keywords: List[str], exclude: Optional[Set[str]] = None) -> Optional[str]:
        exclude = exclude or set()
        for c in out.columns:
            name = str(c).strip().lower()
            if c in exclude:
                continue
            if any(k in name for k in keywords):
                return c
        return None

    foreign_total_col = _pick_by_keywords(["외국", "foreign", "foreigner"])
    inst_total_col = _pick_by_keywords(["기관", "institution", "inst"])

    foreign_detail_cols = [
        c
        for c in out.columns
        if c != foreign_total_col
        and any(k in str(c).strip().lower() for k in ["외국", "foreign", "foreigner"])
    ]
    inst_detail_cols = [
        c
        for c in out.columns
        if c != inst_total_col
        and any(
            k in str(c).strip().lower()
            for k in [
                "기관",
                "institution",
                "inst",
                "금융",
                "보험",
                "투신",
                "연기금",
                "사모",
                "은행",
                "기타",
            ]
        )
        and not any(k in str(c).strip().lower() for k in ["외국", "foreign", "foreigner"])
    ]

    norm = pd.DataFrame({"date": out["date"]})
    if foreign_total_col is not None:
        norm["foreign_netbuy"] = pd.to_numeric(out[foreign_total_col], errors="coerce")
    elif foreign_detail_cols:
        norm["foreign_netbuy"] = (
            out[foreign_detail_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        )
    else:
        norm["foreign_netbuy"] = np.nan

    if inst_total_col is not None:
        norm["inst_netbuy"] = pd.to_numeric(out[inst_total_col], errors="coerce")
    elif inst_detail_cols:
        norm["inst_netbuy"] = (
            out[inst_detail_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        )
    else:
        norm["inst_netbuy"] = np.nan

    return norm[["date", "foreign_netbuy", "inst_netbuy"]].drop_duplicates(
        subset=["date"],
        keep="last",
    )


def _normalize_symbol_history(
    ohlcv: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    symbol: str,
    market_hint: str,
) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    out = ohlcv.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    out = out.sort_index()
    out["date"] = out.index

    cols = list(out.columns)
    open_col = _find_col(out, ["시가", "Open", "open"]) or (cols[0] if len(cols) > 0 else None)
    high_col = _find_col(out, ["고가", "High", "high"]) or (cols[1] if len(cols) > 1 else None)
    low_col = _find_col(out, ["저가", "Low", "low"]) or (cols[2] if len(cols) > 2 else None)
    close_col = _find_col(out, ["종가", "Close", "close"]) or (cols[3] if len(cols) > 3 else None)
    vol_col = _find_col(out, ["거래량", "Volume", "volume"]) or (cols[4] if len(cols) > 4 else None)
    value_col = _find_col(out, ["거래대금", "거래금액", "Value", "TradingValue", "acml_tr_pbmn", "trade_value_krw"])
    change_col = _find_col(out, ["등락률", "Change", "change", "prdy_ctrt", "daily_change_pct"])
    
    norm = pd.DataFrame(
        {
            "date": out["date"],
            "symbol": symbol,
            "open": pd.to_numeric(out[open_col], errors="coerce"),
            "high": pd.to_numeric(out[high_col], errors="coerce"),
            "low": pd.to_numeric(out[low_col], errors="coerce"),
            "close": pd.to_numeric(out[close_col], errors="coerce"),
            "volume": pd.to_numeric(out[vol_col], errors="coerce") if vol_col else np.nan,
            "trade_value_krw": pd.to_numeric(out[value_col], errors="coerce") if value_col else np.nan,
            "daily_change_pct_raw": pd.to_numeric(out[change_col], errors="coerce") if change_col else np.nan,
            "market": market_hint,
        }
    )

    cap_col = None
    cap_value_col = None
    if market_cap_df is not None and not market_cap_df.empty:
        cap = market_cap_df.copy()
        # If 'date' is in columns, use it; otherwise use index
        if "date" not in cap.columns:
            cap.index = pd.to_datetime(cap.index, errors="coerce")
            cap = cap[~cap.index.isna()].copy().sort_index()
            cap["date"] = cap.index
        else:
            cap["date"] = pd.to_datetime(cap["date"], errors="coerce")
            cap = cap[~cap["date"].isna()].copy().sort_values("date")
        
        cap_col = _find_col(cap, ["시가총액", "MarketCap", "market_cap"])
        cap_value_col = _find_col(cap, ["거래대금", "거래금액", "Value", "TradingValue"])

        if cap_col is not None:
            # Drop index name to avoid ambiguity during merge if it matches column name
            if cap.index.name == "date":
                cap.index.name = None
            cap_part = cap[["date", cap_col]].rename(columns={cap_col: "market_cap_krw"})
            norm = norm.merge(cap_part, on="date", how="left")
        else:
            norm["market_cap_krw"] = np.nan

        if cap_value_col is not None and norm["trade_value_krw"].isna().all():
            cap_value_part = cap[["date", cap_value_col]].rename(columns={cap_value_col: "trade_value_krw_cap"})
            norm = norm.merge(cap_value_part, on="date", how="left")
            norm["trade_value_krw"] = norm["trade_value_krw"].fillna(
                pd.to_numeric(norm["trade_value_krw_cap"], errors="coerce")
            )
            norm = norm.drop(columns=["trade_value_krw_cap"])
    else:
        norm["market_cap_krw"] = np.nan

    raw = pd.to_numeric(norm["daily_change_pct_raw"], errors="coerce")
    if raw.notna().any() and float(raw.abs().median(skipna=True)) > 1.0:
        norm["daily_change_pct"] = raw / 100.0
    else:
        norm["daily_change_pct"] = raw

    norm["trade_value_100m"] = pd.to_numeric(norm["trade_value_krw"], errors="coerce") / KRW_100M
    norm["market_cap_100m"] = pd.to_numeric(norm["market_cap_krw"], errors="coerce") / KRW_100M
    norm["prev_close"] = norm.groupby("symbol", sort=False)["close"].shift(1)

    keep_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "market_cap_100m",
        "trade_value_100m",
        "daily_change_pct",
        "market",
        "volume",
    ]
    norm = norm[keep_cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    norm = norm.drop_duplicates(subset=["symbol", "date"], keep="last")
    return norm


def _fetch_kis_daily_ohlcv(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    """Fetch OHLCV and volume/value from KIS as a fallback using synchronous requests."""
    try:
        import requests
    except ImportError:
        return pd.DataFrame()

    candidates = ["src.sync.kis_common", "src.etc.kis_common", "etc.kis_common", "kis_common"]
    kis_mod = None
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if all(hasattr(mod, k) for k in ["APP_KEY", "APP_SECRET", "URL_BASE", "get_access_token"]):
                kis_mod = mod
                break
        except Exception:
            continue

    if kis_mod is None:
        return pd.DataFrame()

    try:
        token = kis_mod.get_access_token()
    except Exception as exc:
        print(f"[warn] KIS token fetch failed for {symbol} fallback: {exc}")
        return pd.DataFrame()

    url = f"{kis_mod.URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": kis_mod.APP_KEY,
        "appsecret": kis_mod.APP_SECRET,
        "tr_id": "FHKST03010100",
        "custtype": "P",
    }
    params = {
        "fid_cond_mrkt_div_code": "J", # Default to KOSPI, fallback handled by API if wrong
        "fid_input_iscd": symbol,
        "fid_input_date_1": _to_ymd(start),
        "fid_input_date_2": _to_ymd(end),
        "fid_period_div_code": "D",
        "fid_org_adj_prc": "0",
    }

    kis_sleep_sec = _effective_kis_sleep_sec(fetch_cfg)
    last_err = None
    
    for attempt in range(fetch_cfg.retries):
        try:
            with _kis_slot(fetch_cfg):
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                time.sleep(kis_sleep_sec)
            
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                continue
            
            data = resp.json()
            if data.get("rt_cd") != "0":
                # Handle market div code mismatch if possible
                if data.get("msg_cd") == "EGW00123" and params["fid_cond_mrkt_div_code"] == "J":
                    params["fid_cond_mrkt_div_code"] = "W" # Try KOSDAQ
                    continue
                return pd.DataFrame()
            
            output2 = data.get("output2", [])
            if not output2:
                return pd.DataFrame()
            
            out1 = data.get("output1", {})
            # Listed share count to estimate historical market cap
            lstn_stcn = float(out1.get("lstn_stcn", 0)) if out1.get("lstn_stcn") else 0
            
            rows = []
            for item in output2:
                d_str = item.get("stck_bsop_date")
                if not d_str: continue
                close_val = float(item.get("stck_clpr", 0))
                rows.append({
                    "date": pd.to_datetime(d_str, format="%Y%m%d"),
                    "open": float(item.get("stck_oprc", 0)),
                    "high": float(item.get("stck_hgpr", 0)),
                    "low": float(item.get("stck_lwpr", 0)),
                    "close": close_val,
                    "volume": float(item.get("acml_vol", 0)),
                    "trade_value_krw": float(item.get("acml_tr_pbmn", 0)),
                    "daily_change_pct": float(item.get("prdy_ctrt", 0)) / 100.0,
                    "market_cap_krw": close_val * lstn_stcn if lstn_stcn > 0 else np.nan
                })
            
            df = pd.DataFrame(rows)
            return df
            
        except Exception as e:
            last_err = e
            time.sleep(fetch_cfg.retry_sleep_sec * (attempt + 1))
            
    if last_err:
        print(f"[warn] KIS ohlcv fallback failed for {symbol}: {last_err}")
    return pd.DataFrame()

def _fetch_one_symbol(
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
    flow_norm = _fetch_investor_history_by_date(
        symbol,
        start,
        end,
        fetch_cfg,
        target_dates=target_dates,
    )
    prog_norm = _fetch_program_history_by_date(
        symbol,
        start,
        end,
        fetch_cfg,
        target_dates=target_dates,
    )

    out = norm.merge(flow_norm, on="date", how="left")
    out = out.merge(prog_norm, on="date", how="left")
    for col in ["foreign_netbuy", "inst_netbuy", "program_netbuy"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _fetch_index_returns(
    start: pd.Timestamp,
    end: pd.Timestamp,
    code: str,
    out_col: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for attempt in range(fetch_cfg.retries):
        try:
            time.sleep(fetch_cfg.request_sleep_sec)
            idx = stock.get_index_ohlcv_by_date(_to_ymd(start), _to_ymd(end), code)
            if idx is None or idx.empty:
                return pd.DataFrame(columns=["date", out_col])
            idx = idx.copy()
            idx.index = pd.to_datetime(idx.index, errors="coerce")
            idx = idx[~idx.index.isna()].copy()
            close_col = _find_col(idx, ["종가", "Close"])
            if close_col is None:
                return pd.DataFrame(columns=["date", out_col])
            close = pd.to_numeric(idx[close_col], errors="coerce")
            pct = close.pct_change()
            return pd.DataFrame({"date": idx.index, out_col: pct.to_numpy()})
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_err = exc
            time.sleep(fetch_cfg.retry_sleep_sec * (attempt + 1))
    print(f"[warn] index fetch failed for {code}: {last_err}")
    return pd.DataFrame(columns=["date", out_col])


def compute_vkospi_proxy(
    index_close_df: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Build V-KOSPI proxy (historical volatility) from index close prices."""
    if index_close_df is None or index_close_df.empty:
        return pd.DataFrame(columns=["date", "v_kospi"])

    if "date" not in index_close_df.columns or "close" not in index_close_df.columns:
        return pd.DataFrame(columns=["date", "v_kospi"])

    out = index_close_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    if out.empty:
        return pd.DataFrame(columns=["date", "v_kospi"])

    close_ratio = pd.to_numeric(out["close"] / out["close"].shift(1), errors="coerce")
    log_ret = np.where(close_ratio > 0, np.log(close_ratio), np.nan)
    roll_std = pd.Series(log_ret, index=out.index).rolling(
        window=int(window),
        min_periods=int(min_periods),
    ).std(ddof=0)
    out["v_kospi"] = roll_std * np.sqrt(252.0) * 100.0
    return out[["date", "v_kospi"]]


def _fetch_vkospi_proxy(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_cfg: FetchConfig,
    *,
    index_code: str = "1028",
) -> pd.DataFrame:
    def _fetch_pykrx_close(code: str) -> pd.DataFrame:
        last_err: Optional[Exception] = None
        for attempt in range(fetch_cfg.retries):
            try:
                time.sleep(fetch_cfg.request_sleep_sec)
                idx = stock.get_index_ohlcv_by_date(_to_ymd(start), _to_ymd(end), code)
                if idx is None or idx.empty:
                    return pd.DataFrame(columns=["date", "close"])
                idx = idx.copy()
                idx.index = pd.to_datetime(idx.index, errors="coerce")
                idx = idx[~idx.index.isna()].copy()
                close_col = _find_col(idx, ["종가", "Close"])
                if close_col is None:
                    return pd.DataFrame(columns=["date", "close"])
                close = pd.to_numeric(idx[close_col], errors="coerce")
                out = pd.DataFrame({"date": idx.index, "close": close.to_numpy()})
                out = out.dropna(subset=["date", "close"]).sort_values("date")
                out = out.drop_duplicates(subset=["date"], keep="last")
                return out
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_err = exc
                time.sleep(fetch_cfg.retry_sleep_sec * (attempt + 1))
        print(f"[warn] v_kospi pykrx close fetch failed for {code}: {last_err}")
        return pd.DataFrame(columns=["date", "close"])

    def _fetch_kis_close(code: str) -> pd.DataFrame:
        try:
            import requests
        except Exception:
            return pd.DataFrame(columns=["date", "close"])

        candidates = [
            "src.sync.kis_common",
            "src.etc.kis_common",
            "etc.kis_common",
            "kis_common",
        ]
        kis_mod = None
        for mod_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                if all(
                    hasattr(mod, k)
                    for k in ["APP_KEY", "APP_SECRET", "URL_BASE", "get_access_token"]
                ):
                    kis_mod = mod
                    break
            except Exception:
                continue
        if kis_mod is None:
            return pd.DataFrame(columns=["date", "close"])

        try:
            token = kis_mod.get_access_token()
        except Exception as exc:
            print(f"[warn] KIS token fetch failed for v_kospi proxy: {exc}")
            return pd.DataFrame(columns=["date", "close"])

        url = f"{kis_mod.URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": kis_mod.APP_KEY,
            "appsecret": kis_mod.APP_SECRET,
            "tr_id": "FHKUP03500100",
            "custtype": "P",
        }

        rows: List[Dict[str, object]] = []
        cur_start = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        chunk_days = 60
        kis_sleep_sec = _effective_kis_sleep_sec(fetch_cfg)
        with _kis_slot(fetch_cfg):
            while cur_start <= end_ts:
                cur_end = min(cur_start + pd.Timedelta(days=chunk_days), end_ts)
                params = {
                    "fid_cond_mrkt_div_code": "U",
                    "fid_input_iscd": str(code),
                    "fid_input_date_1": cur_start.strftime("%Y%m%d"),
                    "fid_input_date_2": cur_end.strftime("%Y%m%d"),
                    "fid_period_div_code": "D",
                    "fid_org_adj_prc": "0",
                }
                try:
                    resp = requests.get(url, headers=headers, params=params, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data.get("output2") if isinstance(data, dict) else None
                        if isinstance(items, list):
                            for item in items:
                                if not isinstance(item, dict):
                                    continue
                                d_raw = str(item.get("stck_bsop_date", "")).strip()
                                c_raw = item.get("bstp_nmix_prpr")
                                if not d_raw:
                                    continue
                                dt = pd.to_datetime(d_raw, format="%Y%m%d", errors="coerce")
                                close = pd.to_numeric(c_raw, errors="coerce")
                                if pd.notna(dt) and pd.notna(close):
                                    rows.append({"date": dt, "close": float(close)})
                except Exception:
                    pass

                cur_start = cur_end + pd.Timedelta(days=1)
                time.sleep(kis_sleep_sec)

        if not rows:
            return pd.DataFrame(columns=["date", "close"])
        out = pd.DataFrame(rows)
        out = out.dropna(subset=["date", "close"]).sort_values("date")
        out = out.drop_duplicates(subset=["date"], keep="last")
        return out

    close_df = _fetch_pykrx_close(str(index_code))
    if close_df.empty and str(index_code) != "1001":
        close_df = _fetch_pykrx_close("1001")
    if close_df.empty:
        close_df = _fetch_kis_close(str(index_code))
    if close_df.empty and str(index_code) != "1001":
        close_df = _fetch_kis_close("1001")
    if close_df.empty:
        print(f"[warn] v_kospi proxy fetch failed for {index_code} (all sources)")
        return pd.DataFrame(columns=["date", "v_kospi"])
    return compute_vkospi_proxy(close_df, window=20, min_periods=20)


def _merge_index_returns(
    history: pd.DataFrame,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    if history.empty:
        return history
    buffer_days = max(30, int(fetch_cfg.lookback_trading_days) * 3)
    start = pd.Timestamp(history["date"].min()) - pd.Timedelta(days=buffer_days)
    end = pd.Timestamp(history["date"].max())

    kospi = _fetch_index_returns(start, end, code="1001", out_col="kospi_pct", fetch_cfg=fetch_cfg)
    kosdaq = _fetch_index_returns(start, end, code="2001", out_col="kosdaq_pct", fetch_cfg=fetch_cfg)
    vkospi = _fetch_vkospi_proxy(start, end, fetch_cfg=fetch_cfg, index_code="1028")

    out = history.merge(kospi, on="date", how="left")
    out = out.merge(kosdaq, on="date", how="left")
    out = out.merge(vkospi, on="date", how="left")
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
        merged.to_parquet(parquet_path, index=False)
    else:
        df.to_parquet(parquet_path, index=False)


def _load_existing_symbol_last_dates(parquet_path: Path) -> Dict[str, pd.Timestamp]:
    if parquet_path is None or not parquet_path.exists():
        return {}
    try:
        old = pd.read_parquet(parquet_path, columns=["symbol", "date"])
    except Exception as exc:
        print(f"[warn] existing parquet date-map read failed ({parquet_path}): {exc}")
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
    symbol_limit: Optional[int],
    include_symbols: Optional[Set[str]],
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
        print("[backfill] no fetch windows (all target symbols are up-to-date).")
        if parquet_out.exists():
            try:
                old = pd.read_parquet(parquet_out)
                if "date" in old.columns:
                    old["date"] = pd.to_datetime(old["date"], errors="coerce")
                print(f"[backfill] skip fetch, keep existing parquet: {parquet_out}")
                return old
            except Exception as exc:
                print(f"[warn] existing parquet read failed after no-op backfill: {exc}")
        return pd.DataFrame()

    print(
        "[backfill] symbols="
        f"{len(windows)} workers={fetch_cfg.max_workers} "
        f"kis_limit={fetch_cfg.kis_rest_limit_per_sec:.2f}/s "
        f"safety={fetch_cfg.kis_rest_safety_ratio:.2f} "
        f"kis_parallel={fetch_cfg.kis_max_parallel_calls} "
        f"kis_sleep={_effective_kis_sleep_sec(fetch_cfg):.3f}s "
        f"existing_symbols={len(existing_last_dates)}"
    )
    chunks: List[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=fetch_cfg.max_workers) as ex:
        futures = {
            ex.submit(_fetch_one_symbol, sym, s, e, mkt, fetch_cfg): (sym, s, e)
            for sym, s, e, mkt in windows
        }
        for fut in as_completed(futures):
            sym, s, e = futures[fut]
            done += 1
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    chunks.append(df)
                print(f"[backfill] {done}/{len(windows)} {sym} rows={0 if df is None else len(df)}")
            except Exception as exc:
                print(f"[warn] {done}/{len(windows)} {sym} failed: {exc}")

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
    print(f"[backfill] parquet saved: {parquet_out}")

    print(
        "[backfill] done rows="
        f"{len(history)} symbols={history['symbol'].nunique()} "
        f"date_range=({history['date'].min().date()} ~ {history['date'].max().date()})"
    )
    return history


def preview_windows(
    *,
    lookback_trading_days: int,
    symbol_limit: Optional[int],
    include_symbols: Optional[Set[str]],
    parquet_out: Optional[Path] = None,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill per-symbol historical bars for rolling-feature support."
    )
    parser.add_argument("--lookback-days", type=int, default=40, help="target trading-day lookback")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers")
    parser.add_argument(
        "--kis-rest-rps",
        type=float,
        default=20.0,
        help="KIS REST nominal limit (req/s).",
    )
    parser.add_argument(
        "--kis-safety-ratio",
        type=float,
        default=0.6,
        help="KIS safety ratio in (0,1]. effective_rps = kis_rest_rps * ratio.",
    )
    parser.add_argument(
        "--kis-max-parallel",
        type=int,
        default=1,
        help="Maximum parallel KIS sections in this process.",
    )
    parser.add_argument("--limit-symbols", type=int, default=None, help="debug symbol cap")
    parser.add_argument("--symbols", type=str, default="", help="comma separated symbol list, e.g. 005930,000660")
    parser.add_argument("--dry-run", action="store_true", help="print fetch windows only, no API calls")
    parser.add_argument(
        "--parquet-out",
        type=str,
        default="data/history/price_history.parquet",
        help="output parquet path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    parquet_out = Path(args.parquet_out) if str(args.parquet_out).strip() else None
    if parquet_out is None:
        raise ValueError("parquet output path is required.")

    include_symbols = None
    if str(args.symbols).strip():
        include_symbols = {
            s.strip().zfill(6)
            for s in str(args.symbols).split(",")
            if s.strip()
        }
        if not include_symbols:
            include_symbols = None

    if args.dry_run:
        plan = preview_windows(
            lookback_trading_days=args.lookback_days,
            symbol_limit=args.limit_symbols,
            include_symbols=include_symbols,
            parquet_out=parquet_out,
        )
        if plan.empty:
            print("[dry-run] no matching symbols in snapshot universe.")
            return
        print(f"[dry-run] symbols={len(plan)}")
        print(plan.to_string(index=False))
        return

    if stock is None:
        raise RuntimeError("pykrx is required. Install with: python -m pip install pykrx")

    run_backfill(
        lookback_trading_days=args.lookback_days,
        max_workers=args.workers,
        kis_rest_limit_per_sec=args.kis_rest_rps,
        kis_rest_safety_ratio=args.kis_safety_ratio,
        kis_max_parallel_calls=args.kis_max_parallel,
        symbol_limit=args.limit_symbols,
        include_symbols=include_symbols,
        parquet_out=parquet_out,
    )


if __name__ == "__main__":
    main()




