"""Recent-regime candidate panel restoration.

The operator discontinued manual hypothetical-outcome journaling for the daily
candidate pool in 2026 (only the traded pick is logged now). This module
automates what that manual step produced: synthetic buy/sell outcomes for the
rest of each day's already-collected candidate pool (condition_history /
archive), convention-matched against real execution via an measured offset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import settings
from src.daily import archive
from src.ml.dataset import clean_column_names

logger = logging.getLogger(__name__)

CONDITION_HISTORY_COLUMN_ALIAS: dict[str, str] = {
    "스냅샷_날짜": "매수날짜",
    "시가총액(억)": "(시가총액, 억)",
    "거래대금(억)": "(거래대금, 억)",
    "순위": "(선정 순위)",
    "기관_순매수(억)": "(기관_순매수)",
    "외국인_순매수(억)": "(외국인_순매수)",
    "프로그램_순매수(억)": "(프로그램_순매수)",
    "전체종목수": "(총 종목 수)",
    "평균거래대금(억)": "(평균 거래대금)",
    "KOSPI등락률": "(kospi, %)",
    "KOSDAQ등락률": "(kosdaq, %)",
    "(v-kospi)": "v_kospi",
    "(v-kosdaq)": "v_kosdaq",
    "시가": "(시가)",
    "고가": "(고가)",
    "저가": "(저가)",
    "종가": "(종가)",
    "전일종가": "(전일종가)",
    "등락률": "(등락률)",
    "체결강도": "(체결강도)",
    "시장구분": "(시장구분)",
}

_ARCHIVE_COLUMN_ALIAS: dict[str, str] = {
    "스냅샷_날짜": "매수날짜",
    "시가": "(시가)",
    "고가": "(고가)",
    "저가": "(저가)",
    "종가": "(종가)",
    "전일종가": "(전일종가)",
    "시가총액": "(시가총액, 억)",
    "거래대금": "(거래대금, 억)",
    "등락률": "(등락률)",
    "선정순위": "(선정 순위)",
    "기관_순매수": "(기관_순매수)",
    "외국인_순매수": "(외국인_순매수)",
    "프로그램_순매수": "(프로그램_순매수)",
    "체결강도": "(체결강도)",
    "시장구분": "(시장구분)",
    "총_종목수": "(총 종목 수)",
    "평균_거래대금": "(평균 거래대금)",
    "kospi": "(kospi, %)",
    "kosdaq": "(kosdaq, %)",
    "v_kospi": "v_kospi",
    "v_kosdaq": "v_kosdaq",
    "거래량": "(거래량)",
    "테마_섹터": "(테마/섹터)",
    "시나리오": "(차트분석)",
}

LABEL_SOURCE_COLUMN: str = "label_source"
EXECUTED_LABEL_SOURCE: str = "sheet_executed"
RECONSTRUCTED_LABEL_SOURCE: str = "reconstructed"
UNSCORED_SCENARIO_SENTINEL: str = "미분류"

ARCHIVE_SCENARIO_THEME_AUTHENTIC_SINCE: str = "2026-08-04"

NO_THEME_SENTINELS: frozenset[str] = frozenset({"테마 없음", ""})

PANEL_COLUMNS: list[str] = [
    "매수날짜",
    "종목코드",
    "(시가)",
    "(고가)",
    "(저가)",
    "(종가)",
    "(전일종가)",
    "(시가총액, 억)",
    "(거래대금, 억)",
    "(등락률)",
    "(선정 순위)",
    "(기관_순매수)",
    "(외국인_순매수)",
    "(프로그램_순매수)",
    "(체결강도)",
    "(시장구분)",
    "(총 종목 수)",
    "(평균 거래대금)",
    "(kospi, %)",
    "(kosdaq, %)",
    "v_kospi",
    "v_kosdaq",
    "(거래량)",
    "(테마/섹터)",
    "(차트분석)",
]

# Stored OHLCV/flow panel columns use float32 per the codec downcast policy;
# label arithmetic and the offset median stay float64.
_PANEL_FLOAT32_COLUMNS: tuple[str, ...] = (
    "(시가)",
    "(고가)",
    "(저가)",
    "(종가)",
    "(전일종가)",
    "(시가총액, 억)",
    "(거래대금, 억)",
    "(등락률)",
    "(선정 순위)",
    "(기관_순매수)",
    "(외국인_순매수)",
    "(프로그램_순매수)",
    "(체결강도)",
    "(총 종목 수)",
    "(평균 거래대금)",
    "(kospi, %)",
    "(kosdaq, %)",
    "v_kospi",
    "v_kosdaq",
    "(거래량)",
)

_CLOSE_RTOL: float = 2e-3


def _default_condition_history_path() -> Path:
    return settings.HISTORY_DIR / "condition_history_cleaned.parquet"


def _read_condition_history(condition_history_path: Path | None) -> pd.DataFrame:
    path = condition_history_path or _default_condition_history_path()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("[DATA] stage=candidate_panel status=no_condition_history path=%s error=%s", path, exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _resolve_archive_df(archive_df: pd.DataFrame | None) -> pd.DataFrame:
    if archive_df is not None:
        return archive_df
    try:
        df = archive.fetch_archive_snapshot(all_rows=True)
    except Exception as exc:
        logger.warning("[DATA] stage=candidate_panel status=no_archive error=%s", exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def check_price_history_freshness(
    price_history_df: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    max_staleness_days: int = 5,
) -> dict[str, Any]:
    """Report price_history staleness in calendar days (observability only)."""
    try:
        if price_history_df is None or price_history_df.empty or "date" not in price_history_df.columns:
            return {"max_date": "", "staleness_days": int(max_staleness_days) + 1, "is_stale": True}  # pragma: no cover
        ref = as_of if as_of is not None else pd.Timestamp.today().normalize()
        ref = pd.Timestamp(ref).normalize()
        max_date = pd.to_datetime(price_history_df["date"], errors="coerce").max()
        if pd.isna(max_date):
            return {"max_date": "", "staleness_days": int(max_staleness_days) + 1, "is_stale": True}  # pragma: no cover
        max_date = pd.Timestamp(max_date).normalize()
        staleness_days = int((ref - max_date).days)
        return {
            "max_date": str(max_date.date()),
            "staleness_days": staleness_days,
            "is_stale": bool(staleness_days > max_staleness_days),
        }
    except Exception:  # pragma: no cover
        return {"max_date": "", "staleness_days": int(max_staleness_days) + 1, "is_stale": True}  # pragma: no cover


def load_candidate_snapshot_panel(
    condition_history_path: Path | None = None,
    archive_df: pd.DataFrame | None = None,
    theme_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Union condition_history and archive snapshots onto raw trade-log headers."""
    condition_df = _read_condition_history(condition_history_path)
    archive_resolved = _resolve_archive_df(archive_df)

    frames: list[pd.DataFrame] = []
    if not condition_df.empty:
        cond = condition_df.rename(columns=CONDITION_HISTORY_COLUMN_ALIAS)
        cond["_src"] = 0
        cond["(차트분석)"] = UNSCORED_SCENARIO_SENTINEL
        frames.append(cond)
    if archive_resolved is not None and not archive_resolved.empty:
        arch = archive_resolved.rename(columns=_ARCHIVE_COLUMN_ALIAS)
        arch["_src"] = 1
        frames.append(arch)
    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    panel = pd.concat(frames, ignore_index=True, sort=False)
    if "종목코드" in panel.columns:
        panel["종목코드"] = panel["종목코드"].astype(str).str.strip().str.zfill(6)

    if theme_df is not None and not theme_df.empty and {"종목코드", "테마"}.issubset(theme_df.columns):
        theme_map = theme_df.set_index("종목코드")["테마"]
        mapped = panel["종목코드"].map(theme_map) if "종목코드" in panel.columns else pd.Series(np.nan, index=panel.index)
        existing = panel["(테마/섹터)"] if "(테마/섹터)" in panel.columns else pd.Series(np.nan, index=panel.index)
        existing = existing.replace(list(NO_THEME_SENTINELS), np.nan)
        mapped = mapped.replace(list(NO_THEME_SENTINELS), np.nan)
        panel["(테마/섹터)"] = existing.fillna(mapped).fillna("기타")
    elif "(테마/섹터)" in panel.columns:
        panel["(테마/섹터)"] = panel["(테마/섹터)"].replace(list(NO_THEME_SENTINELS), np.nan).fillna("기타")
    else:
        panel["(테마/섹터)"] = "기타"

    for col in ("(거래대금, 억)", "(등락률)"):
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    # _validate_close_morning61_feature requires finite relative_flow_strength,
    # so rows with non-finite 거래대금/등락률 cannot survive featurization.
    if "(거래대금, 억)" in panel.columns:
        panel = panel[np.isfinite(panel["(거래대금, 억)"].to_numpy(dtype=np.float64, na_value=np.nan))]
    if "(등락률)" in panel.columns:
        panel = panel[np.isfinite(panel["(등락률)"].to_numpy(dtype=np.float64, na_value=np.nan))]
    if panel.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    # On duplicate (매수날짜, 종목코드) the archive row wins.
    if "_src" in panel.columns and "매수날짜" in panel.columns and "종목코드" in panel.columns:
        panel = panel.sort_values("_src", kind="stable").drop_duplicates(subset=["매수날짜", "종목코드"], keep="last")
        panel = panel.drop(columns=["_src"])

    panel = panel.reindex(columns=PANEL_COLUMNS)
    for col in _PANEL_FLOAT32_COLUMNS:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("float32")
    return panel.reset_index(drop=True)


def load_candidate_universe_symbols(
    condition_history_path: Path | None = None,
    archive_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Deduplicated (symbol, market) universe from candidate sources (column-pruned)."""
    frames: list[pd.DataFrame] = []
    path = condition_history_path or _default_condition_history_path()
    try:
        cdf = pd.read_parquet(path, columns=["종목코드", "시장구분"])
    except Exception:
        try:
            cdf = pd.read_parquet(path, columns=["종목코드"])
        except Exception as exc:
            logger.warning("[DATA] stage=candidate_universe status=no_condition_history path=%s error=%s", path, exc)
            cdf = pd.DataFrame()
    if cdf is not None and not cdf.empty and "종목코드" in cdf.columns:
        part = pd.DataFrame({"symbol": cdf["종목코드"]})
        part["market"] = cdf["시장구분"] if "시장구분" in cdf.columns else np.nan
        frames.append(part)

    arch = _resolve_archive_df(archive_df)
    if arch is not None and not arch.empty and "종목코드" in arch.columns:
        part = pd.DataFrame({"symbol": arch["종목코드"]})
        part["market"] = arch["시장구분"] if "시장구분" in arch.columns else np.nan
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["symbol", "market"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["symbol"] = out["symbol"].astype(str).str.strip().str.zfill(6)
    out["market"] = out["market"].astype(str).fillna("UNKNOWN")
    out.loc[out["market"].isin({"nan", "None", ""}), "market"] = "UNKNOWN"
    out = out[out["symbol"].str.fullmatch(r"\d{6}", na=False)].copy()
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out[["symbol", "market"]].reset_index(drop=True)


def measure_execution_offset_pct(
    trade_log_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    since: str = "2023-01-01",
    min_rows: int = 500,
) -> float:
    """Median (logged net_return - theoretical next_open/close return) in percentage points."""
    clean = clean_column_names(trade_log_df.copy())
    for col in ("trade_date", "stock_code", "close_price", "net_return"):
        if col not in clean.columns:
            raise ValueError(f"measure_execution_offset_pct requires column {col!r}")
    # Column-pruned price history for the label join.
    ph = price_history_df[["date", "symbol", "open", "close"]].copy()
    ph["symbol"] = ph["symbol"].astype(str).str.strip().str.zfill(6)
    ph["date"] = pd.to_datetime(ph["date"], errors="coerce")
    ph = ph.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"], kind="stable")
    # 거래정지/휴장 등으로 0으로 채워진 자리표시 행(측정: 전체의 1.5%)은 다음 실제
    # 거래일이 아니므로 시프트 전에 제외 -- 포함 시 next_open=0 -> -100% 왜곡 발생.
    ph = ph[ph["open"] > 0.0].copy()
    ph["next_open"] = ph.groupby("symbol", sort=False)["open"].shift(-1)

    work = clean[["trade_date", "stock_code", "close_price", "net_return"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["stock_code"] = work["stock_code"].astype(str).str.strip().str.zfill(6)
    merged = work.merge(ph, left_on=["trade_date", "stock_code"], right_on=["date", "symbol"], how="left")
    # Corporate-action guard: reject rows where the sheet close disagrees
    # with the split-adjusted price_history close beyond rtol=2e-3.
    logged = pd.to_numeric(merged["close_price"], errors="coerce").to_numpy(dtype=np.float64)
    px = pd.to_numeric(merged["close"], errors="coerce").to_numpy(dtype=np.float64)
    nxt_full = pd.to_numeric(merged["next_open"], errors="coerce").to_numpy(dtype=np.float64)
    merged["close_agrees"] = np.isclose(logged, px, rtol=_CLOSE_RTOL, atol=0.0) & np.isfinite(logged) & np.isfinite(px)
    merged["theoretical"] = (nxt_full / px - 1.0) * 100.0
    merged = merged[merged["close_agrees"]].copy()
    merged = merged.dropna(subset=["next_open"])
    merged = merged[merged["trade_date"] >= pd.to_datetime(since)]
    if len(merged) < min_rows:
        raise ValueError(f"execution-offset overlap has {len(merged)} qualifying rows, fewer than min_rows={min_rows}")
    theoretical = pd.to_numeric(merged["theoretical"], errors="coerce").to_numpy(dtype=np.float64)
    logged_ret = pd.to_numeric(merged["net_return"], errors="coerce").to_numpy(dtype=np.float64)
    residual = logged_ret - theoretical
    residual = residual[np.isfinite(residual)]
    if len(residual) < min_rows:
        raise ValueError(f"execution-offset overlap has {len(residual)} finite residuals, fewer than min_rows={min_rows}")
    # Median, not mean: the residual is fat-tailed.
    return float(np.median(residual))


def attach_reconstructed_labels(
    panel_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    execution_offset_pct: float,
) -> pd.DataFrame:
    """Attach mechanical next-open labels (float64) to snapshot rows; drop the labelless."""
    for col in ("매수날짜", "종목코드", "(종가)"):
        if col not in panel_df.columns:
            raise ValueError(f"attach_reconstructed_labels requires column {col!r}")
    work = panel_df.copy()
    work["_date"] = pd.to_datetime(work["매수날짜"], errors="coerce")
    work["_sym"] = work["종목코드"].astype(str).str.strip().str.zfill(6)
    work["_entry_close"] = pd.to_numeric(work["(종가)"], errors="coerce").astype(np.float64)
    # Column-pruned price history for the label join.
    ph = price_history_df[["date", "symbol", "open", "close"]].copy()
    ph["symbol"] = ph["symbol"].astype(str).str.strip().str.zfill(6)
    ph["date"] = pd.to_datetime(ph["date"], errors="coerce")
    ph = ph.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"], kind="stable")
    # 거래정지/휴장 등으로 0으로 채워진 자리표시 행(측정: 전체의 1.5%)은 다음 실제
    # 거래일이 아니므로 시프트 전에 제외 -- 포함 시 next_open=0 -> -100% 왜곡 발생.
    ph = ph[ph["open"] > 0.0].copy()
    ph["next_open"] = pd.to_numeric(ph.groupby("symbol", sort=False)["open"].shift(-1), errors="coerce").astype(np.float64)
    merged = work.merge(
        ph[["date", "symbol", "next_open", "close"]], left_on=["_date", "_sym"], right_on=["date", "symbol"], how="left"
    )
    nxt = pd.to_numeric(merged["next_open"], errors="coerce").to_numpy(dtype=np.float64)
    entry = merged["_entry_close"].to_numpy(dtype=np.float64)
    ph_close = pd.to_numeric(merged["close"], errors="coerce").to_numpy(dtype=np.float64)
    # Corporate-action / source-scale guard: the panel's own close (screening
    # snapshot) must agree with price_history's close for the same (date, code)
    # within _CLOSE_RTOL, mirroring measure_execution_offset_pct's guard --
    # otherwise a stale/unadjusted snapshot close vs an adjusted price_history
    # series can produce a wildly wrong ratio (observed up to +4899%).
    close_agrees = np.isfinite(ph_close) & np.isfinite(entry) & np.isclose(entry, ph_close, rtol=_CLOSE_RTOL, atol=0.0)
    # Rows without a next-trading-day open, a non-positive entry close, or a
    # close disagreement, are dropped -- never forward-filled or substituted.
    keep = np.isfinite(nxt) & np.isfinite(entry) & (entry > 0.0) & close_agrees
    merged = merged[keep].copy()
    nxt = nxt[keep]
    entry = entry[keep]
    offset = float(execution_offset_pct)
    ret = (nxt / entry - 1.0) * 100.0 + offset
    merged["(수익률, %)"] = ret.astype(np.float64)
    merged["(매수 가격)"] = entry.astype(np.float64)
    merged["(매도 가격)"] = (entry * (1.0 + ret / 100.0)).astype(np.float64)
    merged[LABEL_SOURCE_COLUMN] = RECONSTRUCTED_LABEL_SOURCE
    merged = merged.drop(columns=["_date", "_sym", "_entry_close", "date", "symbol", "next_open", "close"], errors="ignore")
    return merged.reset_index(drop=True)


def build_restored_trade_log(
    trade_log_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    condition_history_path: Path | None = None,
    archive_df: pd.DataFrame | None = None,
    theme_df: pd.DataFrame | None = None,
    offset_since: str = "2023-01-01",
    offset_min_rows: int = 500,
) -> pd.DataFrame:
    """Union executed trade-log rows with reconstructed snapshot labels (drop-in raw form)."""
    # Era guard: the offset is measured on the post-2023 window only, where the
    # execution convention sign is stable (pre-2021 years flip sign).
    execution_offset_pct = measure_execution_offset_pct(
        trade_log_df, price_history_df, since=offset_since, min_rows=offset_min_rows
    )
    panel = load_candidate_snapshot_panel(
        condition_history_path=condition_history_path, archive_df=archive_df, theme_df=theme_df
    )
    reconstructed = attach_reconstructed_labels(panel, price_history_df, execution_offset_pct=execution_offset_pct)
    tagged = trade_log_df.copy()
    tagged[LABEL_SOURCE_COLUMN] = EXECUTED_LABEL_SOURCE
    # Executed rows are never deduplicated against each other -- the raw trade
    # log legitimately carries duplicate (date, code) pairs (multiple journaled
    # scenarios for the same stock/day). Only reconstructed rows that collide
    # with an already-executed key are dropped, so no operator-logged row is
    # ever discarded by this restoration step.
    tagged_date = pd.to_datetime(tagged["매수날짜"], errors="coerce")
    tagged_code = tagged["종목코드"].astype(str).str.strip().str.zfill(6)
    tagged_keys = set(zip(tagged_date, tagged_code, strict=True))
    if not reconstructed.empty:
        recon_date = pd.to_datetime(reconstructed["매수날짜"], errors="coerce")
        recon_code = reconstructed["종목코드"].astype(str).str.strip().str.zfill(6)
        already_executed = pd.Series(
            [key in tagged_keys for key in zip(recon_date, recon_code, strict=True)], index=reconstructed.index
        )
        reconstructed = reconstructed.loc[~already_executed].copy()
    combined = pd.concat([tagged, reconstructed], ignore_index=True, sort=False)
    combined["_rk_date"] = pd.to_datetime(combined["매수날짜"], errors="coerce")
    combined["_rk_code"] = combined["종목코드"].astype(str).str.strip().str.zfill(6)
    combined = combined.sort_values(["_rk_date", "_rk_code"], kind="stable").drop(columns=["_rk_date", "_rk_code"])
    combined = combined.reset_index(drop=True)
    is_recon = combined[LABEL_SOURCE_COLUMN].astype(str) == RECONSTRUCTED_LABEL_SOURCE
    recon_dates = combined.loc[is_recon, "매수날짜"].astype(str) if is_recon.any() else pd.Series(dtype=str)
    combined.attrs["panel_restoration"] = {
        "execution_offset_pct": float(execution_offset_pct),
        "restored_rows": int(is_recon.sum()),
        "restored_dates": int(recon_dates.nunique()) if len(recon_dates) else 0,
        "restored_date_min": str(recon_dates.min()) if len(recon_dates) else "",
        "restored_date_max": str(recon_dates.max()) if len(recon_dates) else "",
    }
    return combined
