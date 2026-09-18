"""Nightly price_history ingest: KRX bulk end-of-day rows + KIS flows/index, all listed stocks.

Role split (measured): KRX OpenAPI returns every listed stock for one date in one call
(OHLC, base price, value, market cap); KIS returns 30 trading days of per-stock investor
and program flow per call under the shared 18 rps bucket, and the KOSPI/KOSDAQ composite
index history (KIS codes 0001/1001). Kiwoom ka10059 backs up the investor flow only; Toss
STOCK_TRADING_TREND program-trades backs up the program flow only, under its own
independent rate bucket (never shared with KIS's limiter).
Corporate actions are re-derived from the KRX base price, so no history refetch is needed.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

from src import settings
from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.krx_api import (
    KRX_ENDPOINT_KSQ_DAILY,
    KRX_ENDPOINT_STK_DAILY,
    fetch_krx_openapi_day_strict,
)
from src.data.capture_contracts import (
    BrokerPayload,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureStatus,
    PageObserver,
    RawCaptureError,
)
from src.data.capture_store import CaptureStore
from src.data.panel_integrity import heal_price_history_panel
from src.data.parquet_codec import write_price_history_parquet
from src.strategy.contract import derive_chg_ratio
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_OK, record_run_outcome

logger = logging.getLogger(__name__)

KRX_DAILY_MARKETS: tuple[tuple[str, str], ...] = (
    (KRX_ENDPOINT_STK_DAILY, "KOSPI"),
    (KRX_ENDPOINT_KSQ_DAILY, "KOSDAQ"),
)
KRX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ISU_CD",
    "MKT_NM",
    "TDD_OPNPRC",
    "TDD_HGPRC",
    "TDD_LWPRC",
    "TDD_CLSPRC",
    "CMPPREVDD_PRC",
    "ACC_TRDVOL",
    "ACC_TRDVAL",
    "MKTCAP",
)
KRX_ROW_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "trade_value_100m",
    "market_cap_100m",
    "market",
)
FLOW_COLUMNS: tuple[str, ...] = ("inst_netbuy", "foreign_netbuy", "program_netbuy")
INDEX_COLUMNS: tuple[str, ...] = ("kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq")
# KIS 업종코드: 0001=코스피 종합, 1001=코스닥 종합 (pykrx 코드 1001/2001 과 다르다)
KIS_INDEX_KOSPI_CODE: str = "0001"
KIS_INDEX_KOSDAQ_CODE: str = "1001"
# FHKUP03500100 는 요청 구간의 최신 50행만 반환한다 (실측)
KIS_INDEX_PAGE_ROWS: int = 50
# 패널 시작(2016-01-04) 이전 20거래일 HV 워밍업을 덮는 지수 조회 시작일
INDEX_HISTORY_START: pd.Timestamp = pd.Timestamp("2015-11-01")
# FHPTJ04160001/FHPPG04650201 은 호출 1회에 30거래일을 반환한다 (실측)
FLOW_WINDOW_TRADING_DAYS: int = 30
MIN_FLOW_COVERAGE: float = 0.99
# 가격은 원 단위 정수 격자라 반올림 오차 이상 차이만 기업행사로 본다
PRICE_EVENT_TOLERANCE: float = 0.5
KRW_PER_100M: float = 1e8
_ADJUSTED_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "prev_close")
_INVESTOR_PATH = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
_PROGRAM_PATH = "/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily"


class VendorResponseError(RuntimeError):
    """A vendor answered with a business-level failure code (not a transport error)."""


@dataclass(frozen=True)
class IngestReport:
    """Summary of one ingest run."""

    ingested_dates: list[str]
    n_new_rows: int
    n_corporate_events: int
    investor_sources: dict[str, int] = field(default_factory=dict)
    program_sources: dict[str, int] = field(default_factory=dict)
    flow_coverage: dict[str, float] = field(default_factory=dict)
    program_flow_coverage: dict[str, float] = field(default_factory=dict)
    wrote: bool = False
    flow_shortfall: dict[str, float] = field(default_factory=dict)
    program_flow_shortfall: dict[str, float] = field(default_factory=dict)
    n_flow_repaired: int = 0


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _signed_num(value: Any) -> float:
    # 키움 부호 표기('+123', '--45')를 부동소수로 정규화
    text = str(value if value is not None else "").replace(",", "").replace("--", "-").replace("+", "").strip()
    return float(text) if text not in ("", "-") else float("nan")


def normalize_krx_daily(raw: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    """Map one KRX OpenAPI daily stock block to price_history row columns.

    Args:
        raw: OutBlock_1 rows of stk/ksq_bydd_trd for one date.
        trade_date: The requested basDd.

    Returns:
        Frame with KRX_ROW_COLUMNS; prev_close is the KRX base price
        (close - CMPPREVDD_PRC). Empty input yields an empty frame.

    Raises:
        ValueError: When a required KRX column is missing or a symbol repeats.
    """
    if raw.empty:
        return pd.DataFrame(columns=list(KRX_ROW_COLUMNS))
    missing = [c for c in KRX_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"KRX daily block missing columns: {missing}")
    close = _to_num(raw["TDD_CLSPRC"])
    out = pd.DataFrame({
        "date": pd.Timestamp(trade_date).normalize(),
        "symbol": raw["ISU_CD"].astype(str).str.strip().to_numpy(),
        "open": _to_num(raw["TDD_OPNPRC"]).to_numpy(),
        "high": _to_num(raw["TDD_HGPRC"]).to_numpy(),
        "low": _to_num(raw["TDD_LWPRC"]).to_numpy(),
        "close": close.to_numpy(),
        "prev_close": (close - _to_num(raw["CMPPREVDD_PRC"])).to_numpy(),
        "volume": _to_num(raw["ACC_TRDVOL"]).to_numpy(),
        "trade_value_100m": (_to_num(raw["ACC_TRDVAL"]) / KRW_PER_100M).to_numpy(),
        "market_cap_100m": (_to_num(raw["MKTCAP"]) / KRW_PER_100M).to_numpy(),
        "market": raw["MKT_NM"].astype(str).str.strip().to_numpy(),
    })
    dup = int(out["symbol"].duplicated().sum())
    if dup:
        raise ValueError(f"KRX daily block repeats {dup} symbols on {pd.Timestamp(trade_date).date()}")
    return out


def _capture_root() -> Path:
    root = settings.COLLECTION_ROOT
    if root is not None:
        return Path(root)
    return Path(settings.HISTORY_DIR) / "capture"


def _price_capture_context(trading_day: date, run_id: str, endpoint: str, *, symbol: str | None = None) -> CaptureContext:
    return CaptureContext(
        trading_date=trading_day,
        run_id=run_id,
        dataset=CaptureDataset.PRICE,
        vendor="krx",
        endpoint=endpoint,
        symbol=symbol,
        venue="KRX",
        session="regular",
        capture_reason="price-ingest",
        cohort_id=None,
        scheduled_at=None,
    )


def _price_page_observer(store: CaptureStore, trading_day: date, run_id: str) -> PageObserver:
    def _on_page(
        payload: BrokerPayload | None,
        meta: Mapping[str, str],
        started: datetime,
        received: datetime,
        page_index: int,
        attempt_index: int,
    ) -> None:
        endpoint = str(dict(meta).get("endpoint", "krx-daily"))
        try:
            store.append_response(
                CapturedResponse(
                    context=_price_capture_context(trading_day, run_id, endpoint),
                    request_started_at=started,
                    received_at=received,
                    payload=dict(payload) if isinstance(payload, dict) else None,
                    source_timestamp=None,
                    source_published_at=None,
                    status=CaptureStatus.COMPLETE if isinstance(payload, dict) else CaptureStatus.FAILED,
                    page_index=int(page_index),
                    attempt_index=int(attempt_index),
                    continuation={k: str(v) for k, v in dict(meta).items() if k not in ("endpoint",)},
                    error_type=None if isinstance(payload, dict) else "transport",
                )
            )
        except OSError as exc:
            raise RawCaptureError(str(exc)) from exc

    return _on_page


def fetch_krx_daily(trade_date: pd.Timestamp, cfg: AltDataFetchConfig, *, on_page: PageObserver | None = None) -> pd.DataFrame:
    """Preserve daily source responses before normalization and price adjustment.

    Args:
        trade_date: Existing requested trading date.
        cfg: Existing KRX fetch configuration.
        on_page: Durable observer forwarded to both market endpoints.
    Returns:
        Existing normalized daily market frame.
    Raises:
        RuntimeError: Existing incomplete-market or strict-fetch failure.
        RawCaptureError: Original response preservation failed.
    """
    ymd = pd.Timestamp(trade_date).strftime("%Y%m%d")
    parts = [normalize_krx_daily(fetch_krx_openapi_day_strict(ep, ymd, cfg, on_page=on_page), trade_date) for ep, _ in KRX_DAILY_MARKETS]
    sizes = [len(p) for p in parts]
    if all(n == 0 for n in sizes):
        return pd.DataFrame(columns=list(KRX_ROW_COLUMNS))
    if any(n == 0 for n in sizes):
        raise RuntimeError(f"KRX partial publication on {ymd}: rows per market {sizes}")
    return pd.concat(parts, ignore_index=True)


async def fetch_index_closes(
    client: Any, session: Any, code: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Page KIS FHKUP03500100 backwards to collect daily index closes.

    Args:
        client: KisApiClient-compatible object exposing get_market_index_history.
        session: aiohttp session.
        code: KIS index code (0001 KOSPI, 1001 KOSDAQ).
        start: First date to cover.
        end: Last date to cover.

    Returns:
        Frame with date and close, ascending, unique dates within [start, end].

    Raises:
        RuntimeError: On a non-zero rt_cd or when paging stops making progress.
    """
    start_ts, cursor = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    rows: list[dict[str, Any]] = []
    prev_min: pd.Timestamp | None = None
    while cursor >= start_ts:
        res = await client.get_market_index_history(session, code, start_ts.strftime("%Y%m%d"), cursor.strftime("%Y%m%d"))
        if res.get("rt_cd") != "0":
            raise RuntimeError(f"KIS index {code} failed rt_cd={res.get('rt_cd')} msg={res.get('msg1', '')}")
        page = [r for r in (res.get("output2") or []) if r.get("stck_bsop_date")]
        if not page:
            break
        page_min = min(pd.Timestamp(r["stck_bsop_date"]) for r in page)
        if prev_min is not None and page_min >= prev_min:
            raise RuntimeError(f"KIS index {code} paging stalled at {page_min.date()}")
        rows.extend({"date": pd.Timestamp(r["stck_bsop_date"]), "close": float(r["bstp_nmix_prpr"])} for r in page)
        prev_min = page_min
        if len(page) < KIS_INDEX_PAGE_ROWS:
            break
        cursor = page_min - pd.Timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    return out[(out["date"] >= start_ts) & (out["date"] <= pd.Timestamp(end).normalize())].reset_index(drop=True)


def compute_vkospi_proxy(
    index_close_df: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int = 20,
    output_col: str = "v_kospi",
) -> pd.DataFrame:
    """Build V-KOSPI proxy (historical volatility) from index close prices."""
    if index_close_df is None or index_close_df.empty:
        return pd.DataFrame(columns=["date", output_col])

    if "date" not in index_close_df.columns or "close" not in index_close_df.columns:
        return pd.DataFrame(columns=["date", output_col])

    out = index_close_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    if out.empty:
        return pd.DataFrame(columns=["date", output_col])

    close_ratio = pd.to_numeric(out["close"] / out["close"].shift(1), errors="coerce")
    log_ret = np.where(close_ratio > 0, np.log(close_ratio), np.nan)
    roll_std = pd.Series(log_ret, index=out.index).rolling(
        window=int(window),
        min_periods=int(min_periods),
    ).std(ddof=0)
    out[output_col] = roll_std * np.sqrt(252.0) * 100.0
    return out[["date", output_col]]


def compute_index_columns(kospi: pd.DataFrame, kosdaq: pd.DataFrame) -> pd.DataFrame:
    """Derive the four date-level index columns from composite closes.

    Args:
        kospi: KOSPI composite closes (date, close).
        kosdaq: KOSDAQ composite closes (date, close).

    Returns:
        Frame keyed by date with kospi_pct, kosdaq_pct (close-to-close returns)
        and v_kospi, v_kosdaq (20-day realized-vol proxy, annualized percent).
    """
    parts = []
    for df, pct_col, vol_col in ((kospi, "kospi_pct", "v_kospi"), (kosdaq, "kosdaq_pct", "v_kosdaq")):
        work = df[["date", "close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
        work[pct_col] = work["close"] / work["close"].shift(1) - 1.0
        vol = compute_vkospi_proxy(work[["date", "close"]], output_col=vol_col)
        parts.append(work[["date", pct_col]].merge(vol, on="date", how="left").set_index("date"))
    return parts[0].join(parts[1], how="inner").reset_index()


def attach_index_columns(panel: pd.DataFrame, index_cols: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the panel's index columns from the composite-derived table.

    Args:
        panel: Price history panel.
        index_cols: Output of compute_index_columns.

    Returns:
        Copy of panel whose INDEX_COLUMNS come from index_cols by date.

    Raises:
        ValueError: When a panel trading date has no index row.
    """
    table = index_cols.set_index("date")
    dates = pd.to_datetime(panel["date"])
    missing = sorted(set(dates.unique()) - set(table.index))
    if missing:
        raise ValueError(f"index history missing {len(missing)} panel dates, first {pd.Timestamp(missing[0]).date()}")
    out = panel.copy()
    for col in INDEX_COLUMNS:
        out[col] = dates.map(table[col]).to_numpy(dtype=np.float64)
    return out


def plan_new_dates(panel_max_date: pd.Timestamp, trading_days: list[pd.Timestamp], today: pd.Timestamp) -> list[pd.Timestamp]:
    """Return trading days after the panel's last date up to today, ascending.

    Args:
        panel_max_date: Latest date in the panel.
        trading_days: Trading calendar (KIS composite index dates).
        today: Run date; later days are never planned.

    Returns:
        Ascending candidate dates.

    Raises:
        ValueError: When the gap exceeds FLOW_WINDOW_TRADING_DAYS (the flow
            endpoints cannot cover it; a historical backfill is required).
    """
    last, bound = pd.Timestamp(panel_max_date).normalize(), pd.Timestamp(today).normalize()
    out = sorted(pd.Timestamp(d).normalize() for d in trading_days if last < pd.Timestamp(d).normalize() <= bound)
    if len(out) > FLOW_WINDOW_TRADING_DAYS:
        raise ValueError(f"gap of {len(out)} trading days exceeds flow window {FLOW_WINDOW_TRADING_DAYS}; run a historical backfill")
    return out


def select_tail_rows(krx_rows: pd.DataFrame, panel_last_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Keep only rows strictly after each symbol's last stored date.

    Args:
        krx_rows: Normalized KRX rows across the fetched dates.
        panel_last_dates: Last stored date per symbol.

    Returns:
        Rows that extend each symbol's tail (every row of symbols absent from
        the panel), so existing rows are never overwritten.
    """
    last = krx_rows["symbol"].map(panel_last_dates)
    keep = last.isna() | (pd.to_datetime(krx_rows["date"]) > pd.to_datetime(last))
    return krx_rows[keep.to_numpy()].reset_index(drop=True)


def parse_kis_investor_rows(body: dict) -> pd.DataFrame:
    """Parse FHPTJ04160001 into daily institutional/foreign net buy (KRW 1e6).

    Raises:
        VendorResponseError: On a non-zero rt_cd.
    """
    if body.get("rt_cd") != "0":
        raise VendorResponseError(f"KIS investor rt_cd={body.get('rt_cd')} msg={body.get('msg1', '')}")
    rows = [
        {"date": pd.Timestamp(r["stck_bsop_date"]), "inst_netbuy": _signed_num(r.get("orgn_ntby_tr_pbmn")), "foreign_netbuy": _signed_num(r.get("frgn_ntby_tr_pbmn"))}
        for r in (body.get("output2") or [])
        if r.get("stck_bsop_date")
    ]
    return pd.DataFrame(rows, columns=["date", "inst_netbuy", "foreign_netbuy"]).drop_duplicates("date")


def parse_kis_program_rows(body: dict) -> pd.DataFrame:
    """Parse FHPPG04650201 into daily program net buy (KRW 1e6).

    Raises:
        VendorResponseError: On a non-zero rt_cd.
    """
    if body.get("rt_cd") != "0":
        raise VendorResponseError(f"KIS program rt_cd={body.get('rt_cd')} msg={body.get('msg1', '')}")
    rows = [
        {"date": pd.Timestamp(r["stck_bsop_date"]), "program_netbuy": _signed_num(r.get("whol_smtn_ntby_tr_pbmn"))}
        for r in (body.get("output") or [])
        if r.get("stck_bsop_date")
    ]
    return pd.DataFrame(rows, columns=["date", "program_netbuy"]).drop_duplicates("date")


def parse_toss_program_rows(body: dict) -> pd.DataFrame:
    """Parse Toss `/stocks/{symbol}/program-trades` into daily program net buy (KRW).

    Toss splits program flow into arbitrage/non-arbitrage legs; program_netbuy is the
    sum of both legs' netBuyVolume, matching KIS's whole-market `whol_smtn_ntby_tr_pbmn`
    semantics (parse_kis_program_rows). Toss numeric fields are clean decimal strings
    with no comma or sign-prefix quirk, so a direct float() cast is safe.

    Raises:
        VendorResponseError: When the response is a Toss error envelope.
    """
    if "error" in body:
        err = body["error"]
        raise VendorResponseError(f"Toss program-trades code={err.get('code')} msg={err.get('message', '')}")
    records = (body.get("result") or {}).get("records") or []
    rows = [
        {
            "date": pd.Timestamp(r["date"]),
            "program_netbuy": float(r["arbitrage"]["netBuyVolume"]) + float(r["nonArbitrage"]["netBuyVolume"]),
        }
        for r in records
        if r.get("date")
    ]
    return pd.DataFrame(rows, columns=["date", "program_netbuy"]).drop_duplicates("date")


def parse_kiwoom_investor_rows(data: dict) -> pd.DataFrame:
    """Parse Kiwoom ka10059 (amount mode) into KIS-equivalent investor flows.

    KIS foreign net buy equals Kiwoom frgnr_invsr + natfor (other foreigners).

    Raises:
        VendorResponseError: On a non-zero return_code.
    """
    if data.get("return_code") != 0:
        raise VendorResponseError(f"Kiwoom ka10059 return_code={data.get('return_code')} msg={data.get('return_msg', '')}")
    rows_raw = next((v for v in data.values() if isinstance(v, list)), [])
    rows = [
        {"date": pd.Timestamp(str(r["dt"])), "inst_netbuy": _signed_num(r.get("orgn")), "foreign_netbuy": _signed_num(r.get("frgnr_invsr")) + _signed_num(r.get("natfor"))}
        for r in rows_raw
        if r.get("dt")
    ]
    return pd.DataFrame(rows, columns=["date", "inst_netbuy", "foreign_netbuy"]).drop_duplicates("date")


async def fetch_symbol_flows(
    kis: Any, kiwoom: Any, session: Any, symbol: str, anchor_ymd: str, toss: Any | None = None
) -> tuple[pd.DataFrame, str, str]:
    """Fetch 30 trading days of investor and program flow for one symbol.

    Args:
        kis: KisApiClient-compatible client.
        kiwoom: KiwoomApiClient-compatible client used when KIS investor fails; may be None.
        session: aiohttp session.
        symbol: Stock code.
        anchor_ymd: Latest date to cover (YYYYMMDD).
        toss: TossApiClient-compatible client used when KIS program flow fails; may be None.

    Returns:
        Tuple of (frame with date, symbol and FLOW_COLUMNS; investor source tag among
        "kis"/"kiwoom"/"none"; program source tag among "kis"/"toss"/"none").
    """
    base = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol, "FID_INPUT_DATE_1": anchor_ymd}
    inv_body, prg_body = await asyncio.gather(
        kis._handle_request(session.get, f"{kis.base_url}{_INVESTOR_PATH}", headers=kis._get_headers("FHPTJ04160001"), params={**base, "FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": ""}),
        kis._handle_request(session.get, f"{kis.base_url}{_PROGRAM_PATH}", headers=kis._get_headers("FHPPG04650201"), params=base),
    )
    source = "kis"
    try:
        inv = parse_kis_investor_rows(inv_body)
    except VendorResponseError as exc:
        inv, source = pd.DataFrame(columns=["date", "inst_netbuy", "foreign_netbuy"]), "none"
        if kiwoom is not None:
            data, _ = await kiwoom._post_tr(session, "ka10059", "/api/dostk/stkinfo", {"dt": anchor_ymd, "stk_cd": symbol, "amt_qty_tp": "1", "trde_tp": "0", "unit_tp": "1000"})
            try:
                inv, source = parse_kiwoom_investor_rows(data), "kiwoom"
            except VendorResponseError:
                logger.warning("[DATA] stage=price_ingest symbol=%s investor=none kis_error=%s", symbol, exc)
    program_source = "kis"
    try:
        prg = parse_kis_program_rows(prg_body)
    except VendorResponseError as exc:
        prg, program_source = pd.DataFrame(columns=["date", "program_netbuy"]), "none"
        if toss is not None:
            data = await toss.get_program_trades(session, symbol, count=FLOW_WINDOW_TRADING_DAYS, until=pd.Timestamp(anchor_ymd).strftime("%Y-%m-%d"))
            try:
                prg, program_source = parse_toss_program_rows(data), "toss"
            except VendorResponseError:
                logger.warning("[DATA] stage=price_ingest symbol=%s program=none kis_error=%s", symbol, exc)
    out = inv.merge(prg, on="date", how="outer")
    out["symbol"] = symbol
    return out[["date", "symbol", *FLOW_COLUMNS]], source, program_source


async def fetch_all_flows(
    kis: Any, kiwoom: Any, session: Any, symbols: list[str], anchor_ymd: str, toss: Any | None = None
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Fetch flows for every symbol concurrently under the clients' shared rate limiters.

    Returns:
        Tuple of (concatenated flow frame, {"investor": counts per investor source,
        "program": counts per program source}).
    """
    results = await asyncio.gather(*(fetch_symbol_flows(kis, kiwoom, session, s, anchor_ymd, toss) for s in symbols))
    frames = [f for f, _, _ in results if not f.empty]
    investor_sources: dict[str, int] = {}
    program_sources: dict[str, int] = {}
    for _, inv_src, prg_src in results:
        investor_sources[inv_src] = investor_sources.get(inv_src, 0) + 1
        program_sources[prg_src] = program_sources.get(prg_src, 0) + 1
    flows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", *FLOW_COLUMNS])
    return flows, {"investor": investor_sources, "program": program_sources}


def assemble_new_rows(krx_rows: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Join flows onto the KRX tail rows and derive the change columns.

    Returns:
        Rows with KRX_ROW_COLUMNS, FLOW_COLUMNS, chg_ratio, daily_change_pct
        and close_raw; a missing flow stays NaN.
    """
    work = flows.copy()
    work["date"] = pd.to_datetime(work["date"])
    out = krx_rows.merge(work, on=["date", "symbol"], how="left", validate="one_to_one")
    out["close_raw"] = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64)
    chg = derive_chg_ratio(out["close"].to_numpy(dtype=np.float64), out["prev_close"].to_numpy(dtype=np.float64))
    out["chg_ratio"] = chg
    out["daily_change_pct"] = chg
    return out


def compute_flow_coverage(rows: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    """Compute per-date flow coverage over traded rows.

    Coverage is the fraction of traded rows (volume > 0) with every column
    non-null, per YYYY-MM-DD; no raise.

    Args:
        rows: Rows with volume, date and flow columns.
        columns: Flow columns that must all be non-null to count a row as covered.

    Returns:
        Coverage per date (YYYY-MM-DD).
    """
    traded = rows[pd.to_numeric(rows["volume"], errors="coerce") > 0]
    ok = pd.Series(True, index=traded.index)
    for col in columns:
        ok &= traded[col].notna()
    cov = ok.groupby(pd.to_datetime(traded["date"]).dt.strftime("%Y-%m-%d")).mean()
    return {k: round(float(v), 6) for k, v in cov.items()}


def plan_flow_repairs(panel: pd.DataFrame, window_start: pd.Timestamp) -> list[str]:
    """Plan flow repair symbols for traded window rows with any NaN flow.

    KIS flow endpoints return the last 30 trading days per call, so stored NaN
    flows are repairable only inside that window.

    Args:
        panel: Stored price history panel.
        window_start: First date of the flow window.

    Returns:
        Sorted symbols needing a flow refetch.
    """
    dates = pd.to_datetime(panel["date"])
    traded = pd.to_numeric(panel["volume"], errors="coerce") > 0
    gap = panel[list(FLOW_COLUMNS)].isna().any(axis=1)
    mask = (dates >= pd.Timestamp(window_start).normalize()) & traded & gap
    return sorted(panel.loc[mask, "symbol"].astype(str).unique().tolist())


def apply_flow_repairs(panel: pd.DataFrame, flows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill only null flow cells from matching (date, symbol) flows.

    Never overwrites stored flows; returns the repaired panel and the filled
    cell count.

    Args:
        panel: Stored price history panel.
        flows: Fetched flow frame with date, symbol and FLOW_COLUMNS.

    Returns:
        Tuple of (repaired panel, filled cell count).
    """
    out = panel.copy()
    if flows.empty:
        return out, 0
    src_flows = (
        flows.assign(date=pd.to_datetime(flows["date"]), symbol=flows["symbol"].astype(str))
        .drop_duplicates(["date", "symbol"], keep="last")
        .set_index(["date", "symbol"])
    )
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(out["date"]), out["symbol"].astype(str)])
    n_filled = 0
    for col in FLOW_COLUMNS:
        aligned = pd.Series(
            pd.to_numeric(src_flows[col], errors="coerce").reindex(keys).to_numpy(dtype=np.float64),
            index=out.index,
        )
        mask = out[col].isna() & aligned.notna()
        if mask.any():
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
            out.loc[mask, col] = aligned[mask]
            n_filled += int(mask.sum())
    return out, n_filled


def merge_and_adjust(panel: pd.DataFrame, new_rows: pd.DataFrame, trading_days: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append tail rows and retro-adjust history for corporate actions.

    An event exists on a new row when the stored row of the previous trading
    day disagrees with the KRX base price; every earlier row of that symbol
    is scaled by factor = base / prior close (prices x factor; volume and close_raw are never rescaled),
    matching the stored adjusted-history convention. Gaps are never events.

    Args:
        panel: Stored price history.
        new_rows: Tail rows (dates strictly after each symbol's last stored date).
        trading_days: Trading calendar used to recognize consecutive days.

    Returns:
        Tuple of (merged panel sorted by symbol/date, events frame with
        symbol, date and factor).
    """
    merged = pd.concat([panel, new_rows], ignore_index=True)
    if "close_raw" not in merged.columns:
        merged["close_raw"] = np.nan
    # 신규 행의 원가격은 조정 전 종가 자체다.
    merged["close_raw"] = pd.to_numeric(merged["close_raw"], errors="coerce").fillna(pd.to_numeric(merged["close"], errors="coerce"))
    merged["date"] = pd.to_datetime(merged["date"])
    merged["symbol"] = merged["symbol"].astype(str)
    merged = merged.sort_values(["symbol", "date"], kind="stable").drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)
    # 저장 dtype(Int32/Int64 nullable)은 소급 배율 대입과 NA 마스크를 막으므로 float64로 통일 (쓰기 시 codec이 재다운캐스트)
    for col in (*_ADJUSTED_PRICE_COLUMNS, "volume"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    cal = pd.DatetimeIndex(sorted(pd.Timestamp(d).normalize() for d in trading_days))
    new_keys = set(zip(pd.to_datetime(new_rows["date"]), new_rows["symbol"].astype(str), strict=True))
    g = merged.groupby("symbol", sort=False)
    prior_close = g["close"].shift(1).astype("float64")
    prior_date = g["date"].shift(1)
    pos = cal.searchsorted(merged["date"].to_numpy())
    prev_trading = pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns]")
    has_prev = pos > 0
    prev_trading[has_prev] = cal[pos[has_prev] - 1]
    is_new = pd.Series([(d, s) in new_keys for d, s in zip(merged["date"], merged["symbol"], strict=True)], index=merged.index)
    base = pd.to_numeric(merged["prev_close"], errors="coerce").astype("float64")
    event = is_new & (prior_date == prev_trading) & (prior_close > 0) & ((base - prior_close).abs() > PRICE_EVENT_TOLERANCE)
    factor = pd.Series(1.0, index=merged.index)
    factor[event] = (base[event] / prior_close[event]).to_numpy()
    # 행 자신 이후(엄격히 뒤) 이벤트 계수의 누적곱 = 해당 행에 적용할 소급 계수
    rev = factor[::-1].groupby(merged["symbol"][::-1], sort=False).cumprod()[::-1]
    scale = (rev / factor).astype("float64")
    touched = scale != 1.0
    for col in _ADJUSTED_PRICE_COLUMNS:
        merged.loc[touched, col] = merged.loc[touched, col] * scale[touched]
    events = merged.loc[event, ["symbol", "date"]].assign(factor=factor[event].to_numpy()).reset_index(drop=True)
    return merged, events


async def run_price_ingest(
    *,
    today: pd.Timestamp | None = None,
    path: str | os.PathLike[str] | None = None,
    krx_cfg: AltDataFetchConfig | None = None,
    kis: Any | None = None,
    kiwoom: Any | None = None,
    toss: Any | None = None,
    on_outcome: Callable[..., Any] | None = None,
) -> IngestReport:
    """Ingest every newly published trading day plus stale tails, then rewrite index columns.

    Args:
        today: Run date; None selects today in Asia/Seoul.
        path: Panel parquet; None selects settings.PRICE_HISTORY_PARQUET_PATH.
        krx_cfg: KRX config; None builds one from settings.KRX_OPENAPI_KEY.
        kis: KIS client; None builds the configured KisApiClient.
        kiwoom: Kiwoom client for the investor-flow fallback; None builds one.
        toss: Toss client for the program-flow fallback; None builds one.
        on_outcome: Run outcome recorder.

    Returns:
        IngestReport describing what was written.

    Raises:
        FileNotFoundError: When the panel parquet does not exist.
        RuntimeError: Propagated vendor failures (KRX strict, KIS index).
        ValueError: Gap beyond the flow window or index history missing a panel date.
    """
    run_day = (pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None) if today is None else pd.Timestamp(today)).normalize()
    out_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if path is None else path)
    if not out_path.exists():
        raise FileNotFoundError(f"price_history not found: {out_path}")
    cfg = krx_cfg or AltDataFetchConfig(start=run_day, end=run_day + pd.Timedelta(days=1), out_dir=Path("."), krx_api_key=settings.KRX_OPENAPI_KEY)
    raw_enabled = bool(settings.COLLECTION_RAW_ENABLED)
    store = CaptureStore(_capture_root()) if raw_enabled else None
    run_id = f"price-{run_day.strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}" if store is not None else None
    if kis is None:
        from src.api.kis.client import KisApiClient, kis_data_client_kwargs

        kis = KisApiClient(**kis_data_client_kwargs())
    if kiwoom is None:
        from src.api.kiwoom.client import KiwoomApiClient

        kiwoom = KiwoomApiClient()
    if toss is None:
        from src.api.toss.client import TossApiClient

        toss = TossApiClient()
    panel = pd.read_parquet(out_path)
    panel["symbol"] = panel["symbol"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"])
    panel_last = panel.groupby("symbol")["date"].max().to_dict()
    panel_max = pd.Timestamp(panel["date"].max())
    async with aiohttp.ClientSession() as session:
        await kis.ensure_token(session)
        kospi = await fetch_index_closes(kis, session, KIS_INDEX_KOSPI_CODE, INDEX_HISTORY_START, run_day)
        kosdaq = await fetch_index_closes(kis, session, KIS_INDEX_KOSDAQ_CODE, INDEX_HISTORY_START, run_day)
        trading = sorted(pd.Timestamp(d) for d in kospi["date"])
        fetched: dict[pd.Timestamp, pd.DataFrame] = {}
        for d in plan_new_dates(panel_max, trading, run_day):
            trade_date = d
            krx_cfg = cfg
            observer = _price_page_observer(store, pd.Timestamp(trade_date).normalize().date(), run_id) if store is not None and run_id is not None else None
            krx_rows = fetch_krx_daily(trade_date, krx_cfg, on_page=observer)
            rows = krx_rows
            if rows.empty:
                break  # 미게시: 이후 날짜는 연속성 때문에 시도하지 않는다
            fetched[d] = rows
        anchor = max(fetched) if fetched else panel_max
        window = [d for d in trading if d <= anchor][-FLOW_WINDOW_TRADING_DAYS:]
        new_rows = pd.DataFrame()
        sources: dict[str, dict[str, int]] = {}
        tail = pd.DataFrame(columns=list(KRX_ROW_COLUMNS))
        if fetched:
            latest = anchor
            listed = set(fetched[latest]["symbol"])
            stale = [v for s, v in panel_last.items() if s in listed and window[0] <= v < panel_max]
            for d in (d for d in trading if stale and min(stale) < d <= panel_max):
                trade_date = d
                krx_cfg = cfg
                observer = _price_page_observer(store, pd.Timestamp(trade_date).normalize().date(), run_id) if store is not None and run_id is not None else None
                krx_rows = fetch_krx_daily(trade_date, krx_cfg, on_page=observer)
                rows = krx_rows
                if rows.empty:
                    raise RuntimeError(f"KRX returned no rows for past trading day {d.date()}")
                fetched[d] = rows
            tail = select_tail_rows(pd.concat(list(fetched.values()), ignore_index=True), panel_last)
        repairs = plan_flow_repairs(panel, window[0])
        symbols = sorted(set(tail["symbol"].astype(str)) | set(repairs))
        flows = pd.DataFrame(columns=["date", "symbol", *FLOW_COLUMNS])
        if symbols:
            flows, sources = await fetch_all_flows(kis, kiwoom, session, symbols, anchor.strftime("%Y%m%d"), toss)
        if not tail.empty:
            new_rows = assemble_new_rows(tail, flows)
        # 신규 행이 없어도 창 안 결측 수급을 재조회해 채운다 — 부분 기록이 영구 결측으로 굳지 않게.
        panel, n_repaired = apply_flow_repairs(panel, flows)
    if store is not None and run_id is not None and not tail.empty:
        store.publish_frame(tail, context=_price_capture_context(pd.Timestamp(anchor).normalize().date(), run_id, "price-unadjusted", symbol="unadjusted"))
    index_cols = compute_index_columns(kospi, kosdaq)
    if new_rows.empty:
        # 신규 행이 없으면 행 순서가 같으므로 위치 비교로 지수 컬럼 변경 여부만 본다
        events = pd.DataFrame(columns=["symbol", "date", "factor"])
        merged = attach_index_columns(panel, index_cols)
        changed = n_repaired > 0 or any(
            not np.allclose(pd.to_numeric(panel[c], errors="coerce").to_numpy(dtype=np.float64), merged[c].to_numpy(dtype=np.float64), rtol=1e-6, atol=1e-9, equal_nan=True)
            for c in INDEX_COLUMNS
        )
    else:
        merged, events = merge_and_adjust(panel, new_rows, trading)
        merged = attach_index_columns(merged, index_cols)
        changed = True
    window_rows = merged[pd.to_datetime(merged["date"]) >= window[0]]
    coverage = compute_flow_coverage(window_rows, ("inst_netbuy", "foreign_netbuy"))
    program_coverage = compute_flow_coverage(window_rows, ("program_netbuy",))
    shortfall = {k: v for k, v in coverage.items() if v < MIN_FLOW_COVERAGE}
    program_shortfall = {k: v for k, v in program_coverage.items() if v < MIN_FLOW_COVERAGE}
    wrote = False
    if changed:
        write_price_history_parquet(heal_price_history_panel(merged), out_path)
        wrote = True
        if store is not None and run_id is not None and not new_rows.empty:
            store.publish_frame(merged, context=_price_capture_context(pd.Timestamp(anchor).normalize().date(), run_id, "price-adjusted", symbol="adjusted"))
    report = IngestReport(
        ingested_dates=[d.strftime("%Y-%m-%d") for d in sorted(fetched)],
        n_new_rows=len(new_rows),
        n_corporate_events=len(events),
        investor_sources=sources.get("investor", {}),
        program_sources=sources.get("program", {}),
        flow_coverage=coverage,
        program_flow_coverage=program_coverage,
        wrote=wrote,
        flow_shortfall=shortfall,
        program_flow_shortfall=program_shortfall,
        n_flow_repaired=n_repaired,
    )
    logger.info(
        "[DATA] stage=price_ingest dates=%s rows=%d events=%d sources=%s wrote=%s path=%s repaired=%d shortfall=%s program_shortfall=%s",
        report.ingested_dates,
        report.n_new_rows,
        report.n_corporate_events,
        {"investor": report.investor_sources, "program": report.program_sources},
        report.wrote,
        out_path,
        n_repaired,
        shortfall,
        program_shortfall,
    )
    if shortfall or program_shortfall:
        logger.warning(
            "[DATA] stage=price_ingest status=DEGRADED shortfall=%s program_shortfall=%s",
            shortfall,
            program_shortfall,
        )
    if on_outcome is not None:
        on_outcome(
            RUN_OUTCOME_DEGRADED if shortfall or program_shortfall else RUN_OUTCOME_OK,
            run_date=run_day.strftime("%Y-%m-%d"),
            reason="flow_coverage_below_min" if shortfall or program_shortfall else "",
            metrics={
                "ingested_dates": report.ingested_dates,
                "n_new_rows": report.n_new_rows,
                "n_flow_repaired": n_repaired,
                "flow_shortfall": shortfall,
                "program_flow_shortfall": program_shortfall,
            },
        )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run_price_ingest(on_outcome=functools.partial(record_run_outcome, "price_ingest")))
    from src.strategy.growth_shadow import run_growth_shadow

    run_growth_shadow()
    from src.strategy.t1_attribution import run_t1_attribution

    run_t1_attribution()


if __name__ == "__main__":  # pragma: no cover
    main()
