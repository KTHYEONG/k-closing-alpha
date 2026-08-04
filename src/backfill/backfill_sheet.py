"""Google Sheets(Trade/Trade2) 행 데이터를 배치 보강하는 백필 스크립트.

가격/거래량/EMA/수급/지수변동성 컬럼의 누락값을 로컬/외부 소스 조회로 채우고,
시트 API 호출은 배치 단위로 수행한다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import settings
from src.backfill.backfill_price import FetchConfig, fetch_one_symbol


@dataclass(frozen=True)
class SheetBackfillConfig:
    sheets: tuple[str, ...] = ("Trade", "Trade2")
    lookback_for_ema_days: int = 260
    workers: int = 4
    batch_size: int = 2000
    fill_price: bool = True
    fill_ema_volume: bool = True
    fill_flow: bool = True
    fill_index_vol: bool = True


COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["(매수날짜)", "매수날짜", "date"],
    "symbol": ["(종목코드)", "종목코드", "symbol"],
    "open": ["(시가)", "시가", "open"],
    "high": ["(고가)", "고가", "high"],
    "low": ["(저가)", "저가", "low"],
    "close": ["(종가)", "종가", "close"],
    "prev_close": ["(전일종가)", "전일종가", "prev_close"],
    "market_cap": ["(시가총액, 억)", "시가총액(억)", "market_cap"],
    "trade_value": ["(거래대금, 억)", "거래대금(억)", "trade_value"],
    "change_rate": ["(등락률)", "등락률", "change_rate"],
    "volume": ["(거래량)", "거래량", "volume"],
    "ema5": ["(ema5)", "ema5"],
    "ema10": ["(ema10)", "ema10"],
    "ema20": ["(ema20)", "ema20"],
    "inst_netbuy": [
        "(기관_순매수)",
        "(기관순매수)",
        "기관_순매수",
        "기관순매수",
        "inst_netbuy",
    ],
    "foreign_netbuy": [
        "(외국인_순매수)",
        "(외국인순매수)",
        "외국인_순매수",
        "외국인순매수",
        "foreign_netbuy",
    ],
    "program_netbuy": [
        "(프로그램_순매수)",
        "(프로그램순매수)",
        "프로그램_순매수",
        "프로그램순매수",
        "program_netbuy",
    ],
    "v_kospi": ["(v-kospi)", "v-kospi", "v_kospi"],
    "v_kosdaq": ["(v-kosdaq)", "v-kosdaq", "v_kosdaq"],
    "buy_price": ["(매수 가격)", "매수 가격", "(매수가격)", "매수가격", "buy_price"],
    "sell_price": ["(매도 가격)", "매도 가격", "(매도가격)", "매도가격", "sell_price"],
}


def _connect_gsheet():
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    key_path = str(settings.GOOGLE_KEY_PATH)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
    return gspread.authorize(creds)


def _find_header_index(headers: Sequence[str], aliases: Sequence[str]) -> int | None:
    normalized = {str(h).strip(): i for i, h in enumerate(headers)}
    for a in aliases:
        if a in normalized:
            return normalized[a]
    return None


def _resolve_columns(headers: Sequence[str], sheet_name: str = "") -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for key, aliases in COLUMN_ALIASES.items():
        out[key] = _find_header_index(headers, aliases)
    
    # Explicit override for 'Trade' sheet due to header encoding issues
    if "Trade" in sheet_name:
        # Based on actual sheet inspection: H=7, I=8, J=9
        out["market_cap"] = 7
        out["trade_value"] = 8
        out["change_rate"] = 9
        # Ensure date and symbol are also correctly mapped if header matching fails
        if out.get("date") is None: out["date"] = 0
        if out.get("symbol") is None: out["symbol"] = 1
            
    return out


def _parse_date_ymd(value: str) -> str | None:
    s = str(value).strip().replace("-", "").replace("/", "").replace(".", "")
    if len(s) != 8 or not s.isdigit():
        return None
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime("%Y%m%d")


def _needs_fill(
    row: Sequence[str],
    cols: dict[str, int | None],
    *,
    fill_price: bool,
    fill_ema_volume: bool,
    fill_flow: bool,
    fill_index_vol: bool,
) -> bool:
    check_keys: list[str] = []
    if fill_price:
        check_keys += [
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "market_cap",
            "trade_value",
            "change_rate",
        ]
    if fill_ema_volume:
        check_keys += ["volume", "ema5", "ema10", "ema20"]
    if fill_flow:
        check_keys += ["inst_netbuy", "foreign_netbuy", "program_netbuy"]
    if fill_index_vol:
        check_keys += ["v_kospi", "v_kosdaq"]

    for k in check_keys:
        idx = cols.get(k)
        if idx is None:
            continue
        if idx >= len(row):
            return True
        val = str(row[idx]).strip()
        if not val:
            return True
        # Additional check for numeric columns that might have '0' or invalid data
        if k in ["market_cap", "trade_value", "change_rate"]:
            try:
                num_val = float(val.replace(",", ""))
                if num_val == 0:
                    return True
            except ValueError:
                return True
    return False


def _to_int_or_none(v: object) -> int | None:
    num = pd.to_numeric(v, errors="coerce")
    if pd.isna(num):
        return None
    return int(round(float(num)))


def _to_float_or_none(v: object, ndigits: int = 2) -> float | None:
    num = pd.to_numeric(v, errors="coerce")
    if pd.isna(num):
        return None
    return round(float(num), ndigits)


def _compute_emas(hist: pd.DataFrame) -> pd.DataFrame:
    out = hist.copy()
    out = out.sort_values("date")
    out["ema5"] = out["close"].ewm(span=5, adjust=False, min_periods=5).mean()
    out["ema10"] = out["close"].ewm(span=10, adjust=False, min_periods=10).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    return out


def _fetch_index_vol_by_dates(
    target_dates: list[str],
    fetch_cfg: FetchConfig,
    lookback_days: int = 120,
) -> pd.DataFrame:
    if not target_dates:
        return pd.DataFrame(columns=["date_ymd", "v_kospi", "v_kosdaq"])

    try:
        from pykrx import stock as krx_stock
    except Exception:
        return pd.DataFrame(columns=["date_ymd", "v_kospi", "v_kosdaq"])

    s = pd.to_datetime(min(target_dates), format="%Y%m%d", errors="coerce")
    e = pd.to_datetime(max(target_dates), format="%Y%m%d", errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return pd.DataFrame(columns=["date_ymd", "v_kospi", "v_kosdaq"])
    start = (s - pd.Timedelta(days=max(60, int(lookback_days)))).strftime("%Y%m%d")
    end = e.strftime("%Y%m%d")

    def _fetch_close(index_code: str) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(max(1, int(fetch_cfg.retries))):
            try:
                idx = krx_stock.get_index_ohlcv_by_date(start, end, index_code)
                if idx is None or idx.empty:
                    return pd.DataFrame(columns=["date", "close"])
                part = idx.copy()
                part.index = pd.to_datetime(part.index, errors="coerce")
                part = part[~part.index.isna()].copy()
                close_col = None
                for c in ["종가", "Close"]:
                    if c in part.columns:
                        close_col = c
                        break
                if close_col is None:
                    if len(part.columns) >= 4:
                        close_col = list(part.columns)[3]
                    else:
                        return pd.DataFrame(columns=["date", "close"])
                out = pd.DataFrame(
                    {
                        "date": part.index,
                        "close": pd.to_numeric(part[close_col], errors="coerce").to_numpy(),
                    }
                )
                out = out.dropna(subset=["date", "close"]).sort_values("date")
                out = out.drop_duplicates(subset=["date"], keep="last")
                return out
            except Exception as exc:
                last_err = exc
                if attempt < int(fetch_cfg.retries) - 1:
                    import time

                    time.sleep(max(0.0, float(fetch_cfg.request_sleep_sec)))
        if last_err is not None:
            print(f"[warn] index close fetch failed code={index_code}: {last_err}")
        return pd.DataFrame(columns=["date", "close"])

    def _to_hv(close_df: pd.DataFrame, out_col: str) -> pd.DataFrame:
        if close_df is None or close_df.empty:
            return pd.DataFrame(columns=["date_ymd", out_col])
        ratio = pd.to_numeric(close_df["close"] / close_df["close"].shift(1), errors="coerce")
        ratio = ratio.where(ratio > 0)
        log_ret = np.log(ratio)
        hv = log_ret.rolling(window=20, min_periods=20).std(ddof=0) * np.sqrt(252.0) * 100.0
        out = pd.DataFrame(
            {
                "date_ymd": pd.to_datetime(close_df["date"], errors="coerce").dt.strftime("%Y%m%d"),
                out_col: hv.to_numpy(),
            }
        )
        out = out.dropna(subset=["date_ymd"]).drop_duplicates(subset=["date_ymd"], keep="last")
        return out

    # Proxy volatility based on KOSPI200 / KOSDAQ150.
    kospi = _to_hv(_fetch_close("1028"), "v_kospi")
    kosdaq = _to_hv(_fetch_close("2203"), "v_kosdaq")
    out = kospi.merge(kosdaq, on="date_ymd", how="outer")
    out = out[out["date_ymd"].isin(set(target_dates))].copy()
    return out.sort_values("date_ymd").reset_index(drop=True)


def _fetch_symbol_history_for_dates(
    symbol: str,
    target_dates: list[str],
    market_hint: str,
    fetch_cfg: FetchConfig,
    ema_lookback_days: int,
) -> pd.DataFrame:
    if not target_dates:
        return pd.DataFrame()
    s = pd.to_datetime(min(target_dates), format="%Y%m%d", errors="coerce")
    e = pd.to_datetime(max(target_dates), format="%Y%m%d", errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return pd.DataFrame()
    start = s - pd.Timedelta(days=max(60, int(ema_lookback_days)))
    hist = fetch_one_symbol(symbol, start, e, market_hint, fetch_cfg)
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = _compute_emas(hist)
    hist["date_ymd"] = pd.to_datetime(hist["date"], errors="coerce").dt.strftime("%Y%m%d")
    return hist


def _build_cells_for_sheet(
    *,
    ws,
    all_values: list[list[str]],
    cols: dict[str, int | None],
    cfg: SheetBackfillConfig,
) -> tuple[list[object], int]:
    import gspread
    tasks: list[tuple[int, str, str]] = []
    for row_idx, row in enumerate(all_values[1:], start=2):
        date_col = cols.get("date")
        symbol_col = cols.get("symbol")
        if date_col is None or symbol_col is None:
            continue
        if date_col >= len(row) or symbol_col >= len(row):
            continue

        ymd = _parse_date_ymd(row[date_col])
        symbol = str(row[symbol_col]).strip().zfill(6)
        
        needs_update = _needs_fill(
            row,
            cols,
            fill_price=cfg.fill_price,
            fill_ema_volume=cfg.fill_ema_volume,
            fill_flow=cfg.fill_flow,
            fill_index_vol=cfg.fill_index_vol,
        )
        
        if ymd is None or not symbol.isdigit() or not needs_update:
            continue
        tasks.append((row_idx, ymd, symbol))

    if not tasks:
        return [], 0

    by_symbol: dict[str, list[str]] = {}
    for _, ymd, symbol in tasks:
        by_symbol.setdefault(symbol, []).append(ymd)

    fetch_cfg = FetchConfig(
        max_workers=max(1, int(cfg.workers)),
    )
    symbol_hist_map: dict[str, pd.DataFrame] = {}
    symbol_items = [
        (symbol, sorted(set(dates)))
        for symbol, dates in by_symbol.items()
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(cfg.workers))) as ex:
        futures = {
            ex.submit(
                _fetch_symbol_history_for_dates,
                symbol=symbol,
                target_dates=dates,
                market_hint="UNKNOWN",
                fetch_cfg=fetch_cfg,
                ema_lookback_days=cfg.lookback_for_ema_days,
            ): symbol
            for symbol, dates in symbol_items
        }
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                symbol_hist_map[symbol] = fut.result()
            except Exception as exc:
                print(f"[warn] sheet symbol history fetch failed symbol={symbol}: {exc}")
                symbol_hist_map[symbol] = pd.DataFrame()

    vindex_by_date: dict[str, dict[str, object]] = {}
    if cfg.fill_index_vol:
        target_dates = sorted({ymd for _, ymd, _ in tasks})
        vindex = _fetch_index_vol_by_dates(
            target_dates=target_dates,
            fetch_cfg=fetch_cfg,
            lookback_days=cfg.lookback_for_ema_days,
        )
        if not vindex.empty:
            for rec in vindex.itertuples(index=False):
                vindex_by_date[str(rec.date_ymd)] = {
                    "v_kospi": getattr(rec, "v_kospi", np.nan),
                    "v_kosdaq": getattr(rec, "v_kosdaq", np.nan),
                }

    cells: list[object] = []
    filled_rows = 0
    for row_idx, ymd, symbol in tasks:
        hist = symbol_hist_map.get(symbol)
        if hist is None or hist.empty:
            continue
        rec = hist.loc[hist["date_ymd"] == ymd]
        if rec.empty:
            continue
        r = rec.iloc[-1]

        values: dict[str, object] = {}
        if cfg.fill_price:
            values["open"] = _to_int_or_none(r.get("open"))
            values["high"] = _to_int_or_none(r.get("high"))
            values["low"] = _to_int_or_none(r.get("low"))
            values["close"] = _to_int_or_none(r.get("close"))
            values["prev_close"] = _to_int_or_none(r.get("prev_close"))
            
            # Use multiple possible keys for market cap and trade value
            mcap = r.get("market_cap_100m")
            if pd.isna(mcap):
                mcap = r.get("market_cap_krw")
            
            if pd.notna(mcap):
                # Probably raw KRW if > 1e9 (10억). Most listed stocks > 10억 cap.
                if mcap > 1e9:
                    mcap = mcap / 1e8
                values["market_cap"] = _to_float_or_none(mcap, 2)

            tv = r.get("trade_value_100m")
            if pd.isna(tv):
                tv = r.get("trade_value_krw")
            
            if pd.notna(tv):
                # If it's unreasonably large for '억' unit, assume raw KRW. 
                # 100,000,000억 is way too much for daily trade value.
                if tv > 1e7: 
                    tv = tv / 1e8
                values["trade_value"] = _to_float_or_none(tv, 2)

            # daily_change_pct is raw decimal (0.01 = 1%), convert to %
            change = r.get("daily_change_pct")
            if change is not None:
                values["change_rate"] = round(float(change) * 100.0, 2)
        if cfg.fill_ema_volume:
            values["volume"] = _to_int_or_none(r.get("volume"))
            values["ema5"] = _to_float_or_none(r.get("ema5"), 2)
            values["ema10"] = _to_float_or_none(r.get("ema10"), 2)
            values["ema20"] = _to_float_or_none(r.get("ema20"), 2)
        if cfg.fill_flow:
            values["inst_netbuy"] = _to_int_or_none(r.get("inst_netbuy"))
            values["foreign_netbuy"] = _to_int_or_none(r.get("foreign_netbuy"))
            values["program_netbuy"] = _to_int_or_none(r.get("program_netbuy"))
        if cfg.fill_index_vol:
            vrec = vindex_by_date.get(ymd, {})
            values["v_kospi"] = _to_float_or_none(vrec.get("v_kospi"), 2)
            values["v_kosdaq"] = _to_float_or_none(vrec.get("v_kosdaq"), 2)

        row_filled = False
        for key, val in values.items():
            col_idx = cols.get(key)
            if col_idx is None or val is None:
                continue
            
            # 무조건 빈 값만 기입 (강제 덮어씌기 방지)
            current_val = ""
            if col_idx < len(row):
                current_val = str(row[col_idx]).strip()
            
            is_empty_or_zero = not current_val
            if not is_empty_or_zero and key in ["market_cap", "trade_value", "change_rate"]:
                try:
                    # 0인 경우도 빈 값으로 간주하여 채움
                    if float(current_val.replace(",", "")) == 0:
                        is_empty_or_zero = True
                except ValueError:
                    is_empty_or_zero = True
            
            if is_empty_or_zero:
                cells.append(gspread.Cell(row=row_idx, col=col_idx + 1, value=val))
                row_filled = True
            
        if row_idx == 2486 or (row_idx > 2530 and row_filled):
             print(f"      [Debug] Row {row_idx} ({symbol}) fill: {values}")
             
        if row_filled:
            filled_rows += 1

    return cells, filled_rows


def run_sheet_backfill(cfg: SheetBackfillConfig) -> None:
    client = _connect_gsheet()
    sh = client.open(settings.GOOGLE_SHEET_NAME)

    for sheet_name in cfg.sheets:
        ws = sh.worksheet(sheet_name)
        all_values = ws.get_all_values()
        if not all_values:
            print(f"[skip] {sheet_name}: empty")
            continue

        headers = all_values[0]
        cols = _resolve_columns(headers, sheet_name=sheet_name)
        if cols.get("date") is None or cols.get("symbol") is None:
            print(f"[skip] {sheet_name}: missing date/symbol columns")
            continue

        print(f"[run] {sheet_name}: rows={max(0, len(all_values)-1)}")
        cells, filled_rows = _build_cells_for_sheet(
            ws=ws,
            all_values=all_values,
            cols=cols,
            cfg=cfg,
        )
        if not cells:
            print(f"[done] {sheet_name}: no updates")
            continue

        batch_size = max(100, int(cfg.batch_size))
        for i in range(0, len(cells), batch_size):
            ws.update_cells(cells[i : i + batch_size], value_input_option="RAW")
        print(f"[done] {sheet_name}: updated_cells={len(cells)} updated_rows={filled_rows}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified sheet-record backfill")
    p.add_argument("--sheets", type=str, default="Trade,Trade2")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--ema-lookback-days", type=int, default=260)
    p.add_argument("--price-only", action="store_true")
    p.add_argument("--ema-only", action="store_true")
    p.add_argument("--no-flow", action="store_true")
    p.add_argument("--flow-only", action="store_true")
    p.add_argument("--no-vindex", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sheets = tuple(s.strip() for s in str(args.sheets).split(",") if s.strip())
    if args.flow_only:
        fill_price = False
        fill_ema = False
        fill_flow = True
        fill_vindex = False
    else:
        fill_price = not bool(args.ema_only)
        fill_ema = not bool(args.price_only)
        fill_flow = not bool(args.no_flow) and not bool(args.price_only or args.ema_only)
        fill_vindex = not bool(args.no_vindex) and not bool(args.price_only or args.ema_only)
    cfg = SheetBackfillConfig(
        sheets=sheets if sheets else ("Trade", "Trade2"),
        workers=max(1, int(args.workers)),
        batch_size=max(100, int(args.batch_size)),
        lookback_for_ema_days=max(60, int(args.ema_lookback_days)),
        fill_price=fill_price,
        fill_ema_volume=fill_ema,
        fill_flow=fill_flow,
        fill_index_vol=fill_vindex,
    )
    run_sheet_backfill(cfg)


if __name__ == "__main__":
    main()
