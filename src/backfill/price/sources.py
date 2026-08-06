"""pykrx / KIS 프로바이더 호출 (provider calls)."""

from __future__ import annotations

import importlib
import logging
import threading
import time

import numpy as np
import pandas as pd

from src.backfill.price.config import (
    FetchConfig,
    _effective_kis_sleep_sec,
    _kis_slot,
    _wait_for_pykrx_slot,
)
from src.backfill.price.normalize import (
    _find_col,
    _normalize_investor_flow,
    _subtract_flow_frames,
    _to_ymd,
)

logger = logging.getLogger(__name__)

try:
    from pykrx import stock
except ImportError:  # pragma: no cover - dependency availability differs by env
    stock = None


_PROGRAM_HISTORY_FN = None
_PROGRAM_HISTORY_RESOLVED = False
_INVESTOR_HISTORY_FN = None
_INVESTOR_HISTORY_RESOLVED = False
_KIS_CLIENT = None
_KIS_CLIENT_LOCK = threading.Lock()


def _safe_get_market_ohlcv_by_date(
    from_date: str,
    to_date: str,
    symbol: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(fetch_cfg.retries):
        try:
            _wait_for_pykrx_slot(fetch_cfg)
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
    last_err: Exception | None = None
    for attempt in range(fetch_cfg.retries):
        try:
            _wait_for_pykrx_slot(fetch_cfg)
            return stock.get_market_cap_by_date(from_date, to_date, symbol)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_err = exc
            time.sleep(fetch_cfg.retry_sleep_sec * (attempt + 1))
    # market cap fetch failure is non-fatal for history table.
    logger.warning("[DATA] stage=market_cap symbol=%s status=FAIL error=%s", symbol, last_err)
    return pd.DataFrame()


def _safe_get_trading_value_by_date(
    from_date: str,
    to_date: str,
    symbol: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    def _request(on: str, detail: bool) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(fetch_cfg.retries):
            try:
                _wait_for_pykrx_slot(fetch_cfg)
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
        logger.warning("[DATA] stage=investor_flow symbol=%s on=%s detail=%s status=FAIL error=%s", symbol, on, detail, last_err)
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

    logger.warning("[DATA] stage=investor_flow symbol=%s status=empty_after_fallbacks", symbol)
    return pd.DataFrame()


def _resolve_program_history_func():
    global _PROGRAM_HISTORY_FN, _PROGRAM_HISTORY_RESOLVED
    if _PROGRAM_HISTORY_RESOLVED:
        return _PROGRAM_HISTORY_FN
    _PROGRAM_HISTORY_RESOLVED = True
    candidates = [
        "src.sync.fetcher_program",
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
        "src.sync.fetcher_investor",
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
    target_dates: list[str] | None = None,
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
        logger.warning("[DATA] stage=program_flow symbol=%s status=FAIL error=%s", symbol, exc)
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
    target_dates: list[str] | None = None,
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
            logger.warning("[DATA] stage=investor_flow_kis symbol=%s status=FAIL error=%s", symbol, exc)

    # fallback: pykrx (may be empty due KRX endpoint/session changes)
    flow = _safe_get_trading_value_by_date(_to_ymd(start), _to_ymd(end), symbol, fetch_cfg)
    return _normalize_investor_flow(flow)


def _kis_sync_client():
    """KisApiClient 인스턴스를 생성하고 토큰을 동기적으로 확보합니다.

    레거시 `kis_common` 모듈 의존성을 대체합니다.
    """
    import asyncio

    from src.api.kis_client import KisApiClient

    global _KIS_CLIENT
    with _KIS_CLIENT_LOCK:
        if _KIS_CLIENT is not None and _KIS_CLIENT.token:
            return _KIS_CLIENT
        client = KisApiClient()

        async def _ensure() -> None:
            async with client.create_session() as session:
                await client.ensure_token(session)

        try:
            asyncio.run(_ensure())
        except Exception as exc:
            logger.warning("[DATA] stage=kis_token status=FAIL error=%s", exc)
            return None
        if not client.token:
            return None
        _KIS_CLIENT = client
        return client


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

    client = _kis_sync_client()
    if client is None:
        return pd.DataFrame()

    url = f"{client.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = client._get_headers("FHKST03010100")
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
        logger.warning("[DATA] stage=kis_ohlcv symbol=%s status=FAIL error=%s", symbol, last_err)
    return pd.DataFrame()


def _fetch_index_returns(
    start: pd.Timestamp,
    end: pd.Timestamp,
    code: str,
    out_col: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(fetch_cfg.retries):
        try:
            _wait_for_pykrx_slot(fetch_cfg)
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

    # pykrx 지수 엔드포인트가 KRX 응답 변경으로 실패할 때 KIS를 사용한다.
    close = _fetch_kis_index_close(start, end, code, fetch_cfg)
    if not close.empty:
        pct = close["close"].pct_change()
        return pd.DataFrame({"date": close["date"], out_col: pct.to_numpy()})

    logger.warning("[DATA] stage=index symbol=%s status=FAIL error=%s", code, last_err)
    return pd.DataFrame(columns=["date", out_col])


def _fetch_kis_index_close(
    start: pd.Timestamp,
    end: pd.Timestamp,
    code: str,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    """KIS 일별 지수 차트에서 종가를 가져옵니다."""
    try:
        import requests
    except Exception:
        return pd.DataFrame(columns=["date", "close"])

    client = _kis_sync_client()
    if client is None:
        return pd.DataFrame(columns=["date", "close"])

    url = f"{client.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
    headers = client._get_headers("FHKUP03500100")
    rows: list[dict[str, object]] = []
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
                            dt = pd.to_datetime(
                                str(item.get("stck_bsop_date", "")).strip(),
                                format="%Y%m%d",
                                errors="coerce",
                            )
                            close = pd.to_numeric(item.get("bstp_nmix_prpr"), errors="coerce")
                            if pd.notna(dt) and pd.notna(close):
                                rows.append({"date": dt, "close": float(close)})
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                logger.warning("[DATA] stage=kis_index code=%s status=FAIL error=%s", code, exc)
            cur_start = cur_end + pd.Timedelta(days=1)
            time.sleep(kis_sleep_sec)

    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    return (
        pd.DataFrame(rows)
        .dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _fetch_vkospi_proxy(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_cfg: FetchConfig,
    *,
    index_code: str = "1028",
) -> pd.DataFrame:
    from src.backfill.price.factors import compute_vkospi_proxy

    def _fetch_pykrx_close(code: str) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(fetch_cfg.retries):
            try:
                _wait_for_pykrx_slot(fetch_cfg)
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
        logger.warning("[DATA] stage=vkospi symbol=%s status=FAIL error=%s", code, last_err)
        return pd.DataFrame(columns=["date", "close"])

    close_df = _fetch_pykrx_close(str(index_code))
    if close_df.empty and str(index_code) != "1001":
        close_df = _fetch_pykrx_close("1001")
    if close_df.empty:
        close_df = _fetch_kis_index_close(start, end, str(index_code), fetch_cfg)
    if close_df.empty and str(index_code) != "1001":
        close_df = _fetch_kis_index_close(start, end, "1001", fetch_cfg)
    if close_df.empty:
        logger.warning("[DATA] stage=vkospi symbol=%s status=FAIL", index_code)
        return pd.DataFrame(columns=["date", "v_kospi"])
    return compute_vkospi_proxy(close_df, window=20, min_periods=20)
