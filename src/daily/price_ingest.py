"""Nightly price_history ingest: KRX bulk end-of-day rows + KIS flows/index, all listed stocks.

Role split (measured): KRX OpenAPI returns every listed stock for one date in one call
(OHLC, base price, value, market cap); KIS returns 30 trading days of per-stock investor
and program flow per call under the shared 18 rps bucket, and the KOSPI/KOSDAQ composite
index history (KIS codes 0001/1001). Kiwoom ka10059 backs up the investor flow only.
Corporate actions are re-derived from the KRX base price, so no history refetch is needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
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
from src.backfill.price.factors import compute_vkospi_proxy
from src.data.panel_integrity import heal_price_history_panel
from src.data.parquet_codec import write_price_history_parquet
from src.strategy.contract import derive_chg_ratio

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
    flow_coverage: dict[str, float] = field(default_factory=dict)
    wrote: bool = False


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


def fetch_krx_daily(trade_date: pd.Timestamp, cfg: AltDataFetchConfig) -> pd.DataFrame:
    """Fetch every listed KOSPI and KOSDAQ stock for one date from KRX OpenAPI.

    Args:
        trade_date: Trading date to fetch.
        cfg: Alt-data config carrying krx_api_key.

    Returns:
        Normalized rows for both markets, or an empty frame when KRX has not
        published the date (both markets return zero rows).

    Raises:
        RuntimeError: When only one market is published (partial publication)
            or propagated from the strict fetcher (401/404/non-200).
    """
    ymd = pd.Timestamp(trade_date).strftime("%Y%m%d")
    parts = [normalize_krx_daily(fetch_krx_openapi_day_strict(ep, ymd, cfg), trade_date) for ep, _ in KRX_DAILY_MARKETS]
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


async def fetch_symbol_flows(kis: Any, kiwoom: Any, session: Any, symbol: str, anchor_ymd: str) -> tuple[pd.DataFrame, str]:
    """Fetch 30 trading days of investor and program flow for one symbol.

    Args:
        kis: KisApiClient-compatible client.
        kiwoom: KiwoomApiClient-compatible client used when KIS investor fails; may be None.
        session: aiohttp session.
        symbol: Stock code.
        anchor_ymd: Latest date to cover (YYYYMMDD).

    Returns:
        Tuple of (frame with date, symbol and FLOW_COLUMNS; source tag among
        "kis", "kiwoom", "none").
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
    try:
        prg = parse_kis_program_rows(prg_body)
    except VendorResponseError:
        prg = pd.DataFrame(columns=["date", "program_netbuy"])
    out = inv.merge(prg, on="date", how="outer")
    out["symbol"] = symbol
    return out[["date", "symbol", *FLOW_COLUMNS]], source


async def fetch_all_flows(kis: Any, kiwoom: Any, session: Any, symbols: list[str], anchor_ymd: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fetch flows for every symbol concurrently under the clients' shared rate limiters.

    Returns:
        Tuple of (concatenated flow frame, count of symbols per investor source).
    """
    results = await asyncio.gather(*(fetch_symbol_flows(kis, kiwoom, session, s, anchor_ymd) for s in symbols))
    frames = [f for f, _ in results if not f.empty]
    sources: dict[str, int] = {}
    for _, src in results:
        sources[src] = sources.get(src, 0) + 1
    flows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", *FLOW_COLUMNS])
    return flows, sources


def assemble_new_rows(krx_rows: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Join flows onto the KRX tail rows and derive the change columns.

    Returns:
        Rows with KRX_ROW_COLUMNS, FLOW_COLUMNS, chg_ratio and daily_change_pct;
        a missing flow stays NaN.
    """
    work = flows.copy()
    work["date"] = pd.to_datetime(work["date"])
    out = krx_rows.merge(work, on=["date", "symbol"], how="left", validate="one_to_one")
    chg = derive_chg_ratio(out["close"].to_numpy(dtype=np.float64), out["prev_close"].to_numpy(dtype=np.float64))
    out["chg_ratio"] = chg
    out["daily_change_pct"] = chg
    return out


def check_flow_coverage(new_rows: pd.DataFrame, min_coverage: float = MIN_FLOW_COVERAGE) -> dict[str, float]:
    """Fail closed when investor flow is missing for too many traded rows on any date.

    Args:
        new_rows: Output of assemble_new_rows.
        min_coverage: Minimum fraction of traded (volume > 0) rows with both
            institutional and foreign flow.

    Returns:
        Coverage per date (YYYY-MM-DD).

    Raises:
        ValueError: When any date falls below min_coverage.
    """
    traded = new_rows[pd.to_numeric(new_rows["volume"], errors="coerce") > 0]
    ok = traded["inst_netbuy"].notna() & traded["foreign_netbuy"].notna()
    cov = ok.groupby(pd.to_datetime(traded["date"]).dt.strftime("%Y-%m-%d")).mean()
    bad = cov[cov < float(min_coverage)]
    if len(bad):
        raise ValueError(f"investor flow coverage below {min_coverage}: {bad.round(4).to_dict()}")
    return {k: round(float(v), 6) for k, v in cov.items()}


def merge_and_adjust(panel: pd.DataFrame, new_rows: pd.DataFrame, trading_days: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append tail rows and retro-adjust history for corporate actions.

    An event exists on a new row when the stored row of the previous trading
    day disagrees with the KRX base price; every earlier row of that symbol
    is scaled by factor = base / prior close (prices x factor, volume / factor),
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
    merged.loc[touched, "volume"] = np.round(merged.loc[touched, "volume"] / scale[touched])
    events = merged.loc[event, ["symbol", "date"]].assign(factor=factor[event].to_numpy()).reset_index(drop=True)
    return merged, events


async def run_price_ingest(
    *,
    today: pd.Timestamp | None = None,
    path: str | os.PathLike[str] | None = None,
    krx_cfg: AltDataFetchConfig | None = None,
    kis: Any | None = None,
    kiwoom: Any | None = None,
) -> IngestReport:
    """Ingest every newly published trading day plus stale tails, then rewrite index columns.

    Args:
        today: Run date; None selects today in Asia/Seoul.
        path: Panel parquet; None selects settings.PRICE_HISTORY_PARQUET_PATH.
        krx_cfg: KRX config; None builds one from settings.KRX_OPENAPI_KEY.
        kis: KIS client; None builds the configured KisApiClient.
        kiwoom: Kiwoom client for the investor-flow fallback; None builds one.

    Returns:
        IngestReport describing what was written.

    Raises:
        FileNotFoundError: When the panel parquet does not exist.
        RuntimeError: Propagated vendor failures (KRX strict, KIS index).
        ValueError: Gap beyond the flow window, flow coverage below the
            threshold, or index history missing a panel date.
    """
    run_day = (pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None) if today is None else pd.Timestamp(today)).normalize()
    out_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if path is None else path)
    if not out_path.exists():
        raise FileNotFoundError(f"price_history not found: {out_path}")
    cfg = krx_cfg or AltDataFetchConfig(start=run_day, end=run_day + pd.Timedelta(days=1), out_dir=Path("."), krx_api_key=settings.KRX_OPENAPI_KEY)
    if kis is None:
        from src.api.kis.client import KisApiClient

        c = settings.KIS_API_CONFIG
        kis = KisApiClient(c["app_key"], c["app_secret"], c.get("account_id", ""), c.get("hts_id", ""), token_file=str(settings.TOKEN_FILE))
    if kiwoom is None:
        from src.api.kiwoom.client import KiwoomApiClient

        kiwoom = KiwoomApiClient()
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
            rows = fetch_krx_daily(d, cfg)
            if rows.empty:
                break  # 미게시: 이후 날짜는 연속성 때문에 시도하지 않는다
            fetched[d] = rows
        new_rows = pd.DataFrame()
        sources: dict[str, int] = {}
        coverage: dict[str, float] = {}
        if fetched:
            latest = max(fetched)
            window = [d for d in trading if d <= latest][-FLOW_WINDOW_TRADING_DAYS:]
            listed = set(fetched[latest]["symbol"])
            stale = [v for s, v in panel_last.items() if s in listed and window[0] <= v < panel_max]
            for d in (d for d in trading if stale and min(stale) < d <= panel_max):
                rows = fetch_krx_daily(d, cfg)
                if rows.empty:
                    raise RuntimeError(f"KRX returned no rows for past trading day {d.date()}")
                fetched[d] = rows
            tail = select_tail_rows(pd.concat(list(fetched.values()), ignore_index=True), panel_last)
            flows, sources = await fetch_all_flows(kis, kiwoom, session, sorted(tail["symbol"].unique()), latest.strftime("%Y%m%d"))
            new_rows = assemble_new_rows(tail, flows)
            coverage = check_flow_coverage(new_rows)
    index_cols = compute_index_columns(kospi, kosdaq)
    if new_rows.empty:
        # 신규 행이 없으면 행 순서가 같으므로 위치 비교로 지수 컬럼 변경 여부만 본다
        events = pd.DataFrame(columns=["symbol", "date", "factor"])
        merged = attach_index_columns(panel, index_cols)
        changed = any(
            not np.allclose(pd.to_numeric(panel[c], errors="coerce").to_numpy(dtype=np.float64), merged[c].to_numpy(dtype=np.float64), rtol=1e-6, atol=1e-9, equal_nan=True)
            for c in INDEX_COLUMNS
        )
    else:
        merged, events = merge_and_adjust(panel, new_rows, trading)
        merged = attach_index_columns(merged, index_cols)
        changed = True
    wrote = False
    if changed:
        write_price_history_parquet(heal_price_history_panel(merged), out_path)
        wrote = True
    report = IngestReport(
        ingested_dates=[d.strftime("%Y-%m-%d") for d in sorted(fetched)],
        n_new_rows=len(new_rows),
        n_corporate_events=len(events),
        investor_sources=sources,
        flow_coverage=coverage,
        wrote=wrote,
    )
    logger.info(
        "[DATA] stage=price_ingest dates=%s rows=%d events=%d sources=%s wrote=%s path=%s",
        report.ingested_dates, report.n_new_rows, report.n_corporate_events, report.investor_sources, report.wrote, out_path,
    )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run_price_ingest())


if __name__ == "__main__":  # pragma: no cover
    main()
