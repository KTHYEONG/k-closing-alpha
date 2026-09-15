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
# 구 패널과의 공통행 일치율(진단용 지표) 계산 시 정수 반올림을 흡수하는 상대 허용오차
REBUILD_MATCH_RTOL: float = 3e-3
# 소급배율이 이 폭 이상 벗어난 이벤트만 심판 대상. 1% 미만은 주식배당 권리락 등으로 KIS 30/30 일치 실측
REBUILD_EVENT_MATERIAL: float = 0.01
# 두 배율이 같은 이벤트로 간주되는 상대 허용오차(기준가 호가단위 반올림 흡수)
REBUILD_EVENT_RTOL: float = 5e-3
# 상장주식수가 실제로 변했다고 볼 최소 변화율(시총 억원 반올림 노이즈 초과)
REBUILD_SHARE_CHANGE_MIN: float = 5e-3
# 순수 분할/병합/무상감자는 배율 x 주식수비 = 1; 이 허용오차 안이면 주식수가 이벤트를 뒷받침
REBUILD_SHARE_CONSISTENCY_TOL: float = 0.01
# KIS 수정주가 플래그 의미를 매 실행 자기검증할 합의 이벤트(대조군) 수와 최소 일치율
REBUILD_KIS_CONTROL_N: int = 20
REBUILD_KIS_CONTROL_MIN_AGREE: float = 0.9
# KIS 조회 불가(상장폐지 등)이면서 주식수로도 뒷받침되지 않는 불일치 이벤트 허용 상한
REBUILD_MAX_UNVERIFIED: int = 10
# 거래정지로 달력상 비연속인데 기준가가 직전 종가와 다른 행(체인이 반영하지 못함) 허용 상한. 실측 1건
REBUILD_MAX_GAP_JUMPS: int = 5
# KIS 기간별시세 조회창(이벤트일 기준 달력일): 직전 거래일 확보용 앞 12일, 뒤 3일
REBUILD_KIS_WINDOW_BEFORE_DAYS: int = 12
REBUILD_KIS_WINDOW_AFTER_DAYS: int = 3
REBUILD_REFEREE_CHECKPOINT_NAME: str = "kis_event_referee.parquet"
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
    n_disagreements: int = 0
    n_kis_confirmed: int = 0
    n_share_corroborated: int = 0
    n_unverified: int = 0


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


def compute_base_price_factors(raw: pd.DataFrame) -> pd.DataFrame:
    """Classify every KRX row against its prior row as a base-price event, a gap jump, or neither.

    Args:
        raw: KRX daily rows with KRX_ROW_COLUMNS.

    Returns:
        Rows sorted by (symbol, date) with float64 OHLC/prev_close/volume plus
        close_raw, prior_close, prior_date, factor (base/prior_close on events,
        1.0 elsewhere), is_event, is_gap_jump and share_ratio (implied listed
        shares vs the prior row; NaN when unavailable).
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
    jump = (prior_close > 0) & ((out["prev_close"] - prior_close).abs() > PRICE_EVENT_TOLERANCE)
    contiguous = prior_date == prev_trading
    event = contiguous & jump
    factor = pd.Series(1.0, index=out.index)
    factor[event] = (out["prev_close"] / prior_close)[event]
    # 시총(억원)/종가 = 암시 상장주식수. 분할·병합 당일 주식수비로 기준가 이벤트를 교차확인
    mcap = pd.to_numeric(out["market_cap_100m"], errors="coerce")
    shares = (mcap / out["close"]).where((mcap > 0) & (out["close"] > 0))
    out["prior_close"] = prior_close
    out["prior_date"] = prior_date
    out["factor"] = factor.astype("float64")
    out["is_event"] = event.astype(bool)
    out["is_gap_jump"] = (prior_date.notna() & ~contiguous & jump).astype(bool)
    out["share_ratio"] = (shares / shares.groupby(out["symbol"], sort=False).shift(1)).astype("float64")
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
    out = compute_base_price_factors(raw)
    factor = out["factor"]
    rev = factor[::-1].groupby(out["symbol"][::-1], sort=False).cumprod()[::-1]
    scale = (rev / factor).astype("float64")
    touched = scale != 1.0
    for col in _ADJUSTED_COLUMNS:
        out.loc[touched, col] = out.loc[touched, col] * scale[touched]
    logger.info("[DATA] stage=krx_panel_rebuild n_events=%d n_symbols_touched=%d", int(out["is_event"].sum()), int(out.loc[touched, "symbol"].nunique()))
    return out[list(_PANEL_COLUMNS)]


def select_adjudication_events(factors: pd.DataFrame, old: pd.DataFrame, n_controls: int) -> pd.DataFrame:
    """Pick base-price events whose factor disagrees with the old panel, plus agreeing control events.

    The old panel's implied factor on a row is the raw day-over-day ratio
    divided by the old adjusted ratio; it is only defined when the old panel
    holds the same prior row as the KRX series.

    Args:
        factors: Output of compute_base_price_factors.
        old: Currently live panel with date, symbol and close.
        n_controls: Maximum control events, spread evenly over (date, symbol) order.

    Returns:
        Frame with date, symbol, kind ("disagree" or "control"), new_factor,
        old_factor and share_ratio; disagreements first, each kind sorted by (date, symbol).
    """
    keys = ["date", "symbol"]
    o = old[[*keys, "close"]].copy()
    o["date"] = pd.to_datetime(o["date"])
    o["symbol"] = o["symbol"].astype(str)
    o["close"] = pd.to_numeric(o["close"], errors="coerce").astype("float64")
    o = o.drop_duplicates(keys, keep="last").sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    og = o.groupby("symbol", sort=False)
    o["old_prior_close"] = og["close"].shift(1)
    o["old_prior_date"] = og["date"].shift(1)
    o = o.rename(columns={"close": "old_close"})
    m = factors[[*keys, "close_raw", "prior_close", "prior_date", "factor", "is_event", "share_ratio"]].merge(o, on=keys, how="inner")
    valid = (m["old_prior_date"] == m["prior_date"]) & (m["prior_close"] > 0) & (m["old_close"] > 0) & (m["old_prior_close"] > 0)
    m = m[valid].copy()
    m["new_factor"] = m["factor"].astype("float64")
    m["old_factor"] = ((m["close_raw"] / m["prior_close"]) / (m["old_close"] / m["old_prior_close"])).astype("float64")
    agree = np.isclose(m["new_factor"], m["old_factor"], rtol=REBUILD_EVENT_RTOL)
    material = np.maximum((m["new_factor"] - 1).abs(), (m["old_factor"] - 1).abs()) >= REBUILD_EVENT_MATERIAL
    cols = [*keys, "kind", "new_factor", "old_factor", "share_ratio"]
    dis = m[material & ~agree].assign(kind="disagree").sort_values(keys, kind="stable")
    pool = m[agree & m["is_event"] & ((m["new_factor"] - 1).abs() >= REBUILD_EVENT_MATERIAL)].sort_values(keys, kind="stable")
    if len(pool) > n_controls:
        pool = pool.iloc[np.unique(np.linspace(0, len(pool) - 1, n_controls).round().astype(int))]
    ctrl = pool.assign(kind="control")
    return pd.concat([dis[cols], ctrl[cols]], ignore_index=True)


def _kis_closes(res: dict[str, Any]) -> pd.Series:
    rows = (res.get("output2") or []) if res.get("rt_cd") == "0" else []
    pairs = {pd.Timestamp(r["stck_bsop_date"]): float(r["stck_clpr"]) for r in rows if r.get("stck_bsop_date") and float(r.get("stck_clpr") or 0) > 0}
    return pd.Series(pairs, dtype="float64").sort_index()


async def fetch_kis_event_factors(kis: Any, session: Any, events: pd.DataFrame, checkpoint_path: Path) -> pd.DataFrame:
    """Referee each event with KIS daily closes, adjusted (fid_org_adj_prc="0") vs raw ("1").

    The KIS factor is the raw day-over-day ratio divided by the adjusted ratio
    on the event date. Results, including NaN for unavailable symbols, are
    checkpointed per (date, symbol) so a retried rebuild never re-queries them.

    Args:
        kis: KisApiClient-compatible object exposing get_stock_ohlcv_history.
        session: aiohttp session.
        events: Output of select_adjudication_events.
        checkpoint_path: Parquet holding date, symbol, kis_factor.

    Returns:
        events with a float64 kis_factor column (NaN when KIS lacks the event
        date or its prior trading day in both series).
    """
    out = events.copy()
    if out.empty:
        out["kis_factor"] = pd.Series(dtype="float64")
        return out
    checkpoint_path = Path(checkpoint_path)
    if await asyncio.to_thread(checkpoint_path.exists):
        done = await asyncio.to_thread(pd.read_parquet, checkpoint_path)
        done["date"] = pd.to_datetime(done["date"])
        done["symbol"] = done["symbol"].astype(str)
    else:
        done = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "symbol": pd.Series(dtype=str), "kis_factor": pd.Series(dtype="float64")})
    have = set(zip(done["date"], done["symbol"], strict=True))
    for date, symbol in out[["date", "symbol"]].drop_duplicates().itertuples(index=False):
        day = pd.Timestamp(date)
        if (day, symbol) in have:
            continue
        a = (day - pd.Timedelta(days=REBUILD_KIS_WINDOW_BEFORE_DAYS)).strftime("%Y%m%d")
        b = (day + pd.Timedelta(days=REBUILD_KIS_WINDOW_AFTER_DAYS)).strftime("%Y%m%d")
        adj = _kis_closes(await kis.get_stock_ohlcv_history(session, symbol, a, b, adj_price="0"))
        org = _kis_closes(await kis.get_stock_ohlcv_history(session, symbol, a, b, adj_price="1"))
        value = float("nan")
        if day in adj.index and day in org.index:
            i, j = adj.index.get_loc(day), org.index.get_loc(day)
            if i > 0 and j > 0 and adj.index[i - 1] == org.index[j - 1]:
                value = float((org.iloc[j] / org.iloc[j - 1]) / (adj.iloc[i] / adj.iloc[i - 1]))
        done = pd.concat([done, pd.DataFrame({"date": [day], "symbol": [symbol], "kis_factor": [value]})], ignore_index=True)
        have.add((day, symbol))
        await asyncio.to_thread(done.to_parquet, checkpoint_path, index=False)
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    return out.merge(done, on=["date", "symbol"], how="left", validate="many_to_one")


def adjudicate_events(events: pd.DataFrame, *, n_gap_jumps: int) -> dict[str, float | int]:
    """Decide whether rebuilt base-price factors survive the independent KIS referee.

    A disagreement is accepted when KIS reproduces the new factor, or when the
    listed share count moved inversely to the factor (pure split/merge).
    Control events verify the KIS adjusted-price flag semantics on every run.

    Args:
        events: Output of fetch_kis_event_factors.
        n_gap_jumps: Rows whose base price jumped across a suspension gap.

    Returns:
        Metrics n_disagreements, n_kis_confirmed, n_share_corroborated,
        n_unverified, n_controls, control_agree_rate (NaN without finite
        controls) and n_gap_jumps.

    Raises:
        ValueError: On too many gap jumps, failed control verification while
            disagreements exist, any new factor contradicted by KIS without
            share corroboration, or unverified disagreements above REBUILD_MAX_UNVERIFIED.
    """
    if n_gap_jumps > REBUILD_MAX_GAP_JUMPS:
        raise ValueError(f"{n_gap_jumps} gap base-price jumps exceed {REBUILD_MAX_GAP_JUMPS}")
    kis_f = events["kis_factor"].astype("float64")
    finite = np.isfinite(kis_f)
    kis_eq_new = finite & np.isclose(kis_f.fillna(0.0), events["new_factor"].astype("float64"), rtol=REBUILD_EVENT_RTOL)
    ctrl = events["kind"] == "control"
    dis = events["kind"] == "disagree"
    n_ctrl_finite = int((ctrl & finite).sum())
    control_rate = float((ctrl & kis_eq_new).sum() / n_ctrl_finite) if n_ctrl_finite else float("nan")
    n_dis = int(dis.sum())
    if n_dis and not (n_ctrl_finite and control_rate >= REBUILD_KIS_CONTROL_MIN_AGREE):
        raise ValueError(f"KIS referee control verification failed: n_controls={n_ctrl_finite} agree_rate={control_rate}")
    share = events["share_ratio"].astype("float64")
    corroborated = ((share - 1).abs() > REBUILD_SHARE_CHANGE_MIN) & ((events["new_factor"].astype("float64") * share - 1).abs() < REBUILD_SHARE_CONSISTENCY_TOL)
    confirmed = dis & kis_eq_new
    by_shares = dis & ~kis_eq_new & corroborated
    rejected = dis & ~kis_eq_new & finite & ~corroborated
    unverified = dis & ~finite & ~corroborated
    if int(rejected.sum()):
        first = events[rejected].iloc[0]
        raise ValueError(f"{int(rejected.sum())} rebuilt factors contradicted by KIS, first {pd.Timestamp(first['date']).date()} {first['symbol']}")
    if int(unverified.sum()) > REBUILD_MAX_UNVERIFIED:
        raise ValueError(f"{int(unverified.sum())} unverified factor disagreements exceed {REBUILD_MAX_UNVERIFIED}")
    return {"n_disagreements": n_dis, "n_kis_confirmed": int(confirmed.sum()), "n_share_corroborated": int(by_shares.sum()), "n_unverified": int(unverified.sum()),
            "n_controls": n_ctrl_finite, "control_agree_rate": control_rate, "n_gap_jumps": int(n_gap_jumps)}


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
    """Validate a rebuilt panel's structure against the old panel before writing.

    The adjusted-close match rate against the old panel is diagnostic only:
    the old panel carries its own splice seams, so price correctness is
    decided by adjudicate_events instead.

    Args:
        new: Rebuilt panel candidate.
        old: Currently live panel.

    Returns:
        Coverage metrics with n_rows, n_symbols, n_common_rows, match_rate, n_dates.

    Raises:
        ValueError: On duplicate keys, per-date KOSPI/KOSDAQ shrink, or non-positive close_raw.
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
        ValueError: When KRX returns no rows, or propagated from adjudicate_events or validate_rebuild.
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
    factors = compute_base_price_factors(raw)
    events = select_adjudication_events(factors, old, REBUILD_KIS_CONTROL_N)
    n_gap_jumps = int(factors["is_gap_jump"].sum())
    del factors
    async with aiohttp.ClientSession() as session:
        await kis.ensure_token(session)
        kospi = await fetch_index_closes(kis, session, KIS_INDEX_KOSPI_CODE, INDEX_HISTORY_START, pd.Timestamp(end))
        kosdaq = await fetch_index_closes(kis, session, KIS_INDEX_KOSDAQ_CODE, INDEX_HISTORY_START, pd.Timestamp(end))
        events = await fetch_kis_event_factors(kis, session, events, Path(checkpoint_dir) / REBUILD_REFEREE_CHECKPOINT_NAME)
    adjudication = adjudicate_events(events, n_gap_jumps=n_gap_jumps)
    logger.info("[DATA] stage=krx_panel_rebuild adjudication=%s", adjudication)
    panel = carry_existing_flows(derive_adjusted_prices(raw), old)
    panel = heal_price_history_panel(panel)
    panel = attach_index_columns(panel, compute_index_columns(kospi, kosdaq))
    metrics = validate_rebuild(panel, old)
    write_price_history_parquet(panel, Path(out_path))
    logger.info("[DATA] stage=krx_panel_rebuild status=WRITTEN path=%s %s", out_path, metrics)
    return RebuildReport(
        n_rows=int(metrics["n_rows"]), n_symbols=int(metrics["n_symbols"]), n_dates=int(metrics["n_dates"]), match_rate=float(metrics["match_rate"]), n_common_rows=int(metrics["n_common_rows"]),
        n_disagreements=int(adjudication["n_disagreements"]), n_kis_confirmed=int(adjudication["n_kis_confirmed"]),
        n_share_corroborated=int(adjudication["n_share_corroborated"]), n_unverified=int(adjudication["n_unverified"]),
    )


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
