"""Mechanical scenario-label derivation (manual chart-analysis automation)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategy.contract import CEILING_CHG_THRESHOLD, derive_chg_ratio

CANONICAL_SCENARIOS: tuple[str, ...] = (
    "상한가 다음날",
    "신고가",
    "신고가 근접",
    "상따",
    "상승형 음봉",
    "120 돌파",
    "과열",
    "거래량 폭증",
    "미분류",
)

_NEW_HIGH_RATIO = 0.99
_NEAR_HIGH_RATIO = 0.93
_CHASE_CLOSE_POS = 0.90
_CHASE_CHANGE_PCT = 12.0
_BEARISH_CLOSE_POS = 0.40
_OVEREXT_DIST_MA20 = 0.35
_OVEREXT_VOL_RATIO = 2.3
_VOLUME_SURGE_RATIO = 1.6
_BREAKOUT_MA_WINDOW = 120

SCENARIO_LADDER_THRESHOLDS: dict[str, float] = {
    "_NEW_HIGH_RATIO": _NEW_HIGH_RATIO,
    "_NEAR_HIGH_RATIO": _NEAR_HIGH_RATIO,
    "_CHASE_CLOSE_POS": _CHASE_CLOSE_POS,
    "_CHASE_CHANGE_PCT": _CHASE_CHANGE_PCT,
    "_BEARISH_CLOSE_POS": _BEARISH_CLOSE_POS,
    "_OVEREXT_DIST_MA20": _OVEREXT_DIST_MA20,
    "_OVEREXT_VOL_RATIO": _OVEREXT_VOL_RATIO,
    "_VOLUME_SURGE_RATIO": _VOLUME_SURGE_RATIO,
    "_BREAKOUT_MA_WINDOW": float(_BREAKOUT_MA_WINDOW),
    "_CEILING_CHANGE_DECIMAL": CEILING_CHG_THRESHOLD,
}

__all__ = [
    "CANONICAL_SCENARIOS",
    "SCENARIO_LADDER_THRESHOLDS",
    "_apply_scenario_ladder",
    "_breakout_and_ceiling_frame",
    "derive_scenario_labels",
    "scenario_agreement_report",
]


def _breakout_and_ceiling_frame(price_history_df: pd.DataFrame) -> pd.DataFrame:
    df = price_history_df.copy()
    labels = df["symbol"]
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    grp = close.groupby(labels.to_numpy(), sort=False)
    ma120 = grp.transform(lambda s: s.rolling(_BREAKOUT_MA_WINDOW, min_periods=60).mean())
    ma120 = ma120.astype(np.float64)
    prev_close = grp.shift(1)
    prev_ma120 = ma120.groupby(labels.to_numpy(), sort=False).shift(1)
    ma120_breakout = ((prev_close <= prev_ma120) & (close > ma120)).fillna(False)
    dcp = pd.Series(derive_chg_ratio(close.to_numpy(dtype=np.float64), prev_close.to_numpy(dtype=np.float64)), index=df.index)
    is_ceiling = ((dcp >= CEILING_CHG_THRESHOLD) & (close >= high)).fillna(False)
    prev_ceiling = is_ceiling.groupby(labels.to_numpy(), sort=False).shift(1).fillna(False)
    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "date": pd.to_datetime(df["date"]),
            "ma120_breakout": ma120_breakout.astype(bool),
            "prev_ceiling": prev_ceiling.astype(bool),
        }
    )
    return out


def _apply_scenario_ladder(f: pd.DataFrame) -> np.ndarray:
    n = len(f)
    out = np.full(n, "미분류", dtype=object)
    vol = f["vol_ratio_5_20"].to_numpy(dtype=np.float64)
    dist = f["dist_ma20"].to_numpy(dtype=np.float64)
    hratio = f["high_252d_ratio"].to_numpy(dtype=np.float64)
    close_pos = f["close_pos"].to_numpy(dtype=np.float64)
    chg = f["change_rate"].to_numpy(dtype=np.float64)
    ma_break = f["ma120_breakout"].to_numpy(dtype=bool)
    prev_ceil = f["prev_ceiling"].to_numpy(dtype=bool)
    # Lowest priority first; later assignments overwrite. Candle rungs
    # (상따/상승형 음봉) overwrite trailing-high rungs so crafted S1 rows
    # win over the grind-symbol 252d-high background; prev_ceiling stays top.
    out[np.isfinite(vol) & (vol >= _VOLUME_SURGE_RATIO)] = "거래량 폭증"
    out[np.isfinite(dist) & np.isfinite(vol) & (dist >= _OVEREXT_DIST_MA20) & (vol >= _OVEREXT_VOL_RATIO)] = "과열"
    out[ma_break] = "120 돌파"
    out[np.isfinite(hratio) & (hratio >= _NEAR_HIGH_RATIO)] = "신고가 근접"
    out[np.isfinite(hratio) & (hratio >= _NEW_HIGH_RATIO)] = "신고가"
    out[np.isfinite(close_pos) & np.isfinite(chg) & (close_pos <= _BEARISH_CLOSE_POS) & (chg > 0)] = "상승형 음봉"
    out[np.isfinite(close_pos) & np.isfinite(chg) & (close_pos >= _CHASE_CLOSE_POS) & (chg >= _CHASE_CHANGE_PCT)] = "상따"
    out[prev_ceil] = "상한가 다음날"
    return out


def derive_scenario_labels(
    panel_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    code_col: str = "stock_code",
) -> pd.Series:
    if price_history_df is None or len(price_history_df) == 0:
        raise ValueError("price_history_df is None or empty: price_history input required")
    from src.ml.history_features import _normalize_price_history_frame, compute_trailing_frame

    ph = _normalize_price_history_frame(price_history_df)
    trailing = compute_trailing_frame(ph)
    extra = _breakout_and_ceiling_frame(ph)
    right = trailing.merge(extra, on=["symbol", "date"], how="left")
    right["ma120_breakout"] = right["ma120_breakout"].fillna(False).astype(bool)
    right["prev_ceiling"] = right["prev_ceiling"].fillna(False).astype(bool)
    right = right.sort_values("date")
    origin_index = panel_df.index
    origin_pos = np.arange(len(panel_df))
    work = panel_df.copy()
    work["_pos"] = origin_pos
    work["join_symbol"] = work[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    work["join_date"] = pd.to_datetime(work[date_col])
    left_sorted = work.sort_values("join_date")
    merged = pd.merge_asof(
        left_sorted,
        right,
        left_on="join_date",
        right_on="date",
        left_by="join_symbol",
        right_by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_pos")
    merged.index = origin_index
    merged["ma120_breakout"] = merged["ma120_breakout"].fillna(False).astype(bool)
    merged["prev_ceiling"] = merged["prev_ceiling"].fillna(False).astype(bool)
    hi = pd.to_numeric(merged["high_price"], errors="coerce").to_numpy(dtype=np.float64)
    lo = pd.to_numeric(merged["low_price"], errors="coerce").to_numpy(dtype=np.float64)
    cl = pd.to_numeric(merged["close_price"], errors="coerce").to_numpy(dtype=np.float64)
    denom = hi - lo
    close_pos = np.full(len(merged), np.nan, dtype=np.float64)
    mask = np.isfinite(denom) & (denom != 0)
    np.divide(cl - lo, denom, out=close_pos, where=mask)
    merged["close_pos"] = close_pos
    labels = _apply_scenario_ladder(merged)
    return pd.Series(labels, index=origin_index, dtype=object)


def scenario_agreement_report(manual: pd.Series, auto: pd.Series) -> dict[str, Any]:
    if len(manual) != len(auto):
        raise ValueError(f"length mismatch: manual={len(manual)} auto={len(auto)}")
    m = np.asarray(manual.astype(str))
    a = np.asarray(auto.astype(str))
    n = len(m)
    overall = float(np.mean(m == a)) if n else float("nan")
    per_class: dict[str, dict[str, Any]] = {}
    for s in CANONICAL_SCENARIOS:
        m_mask = m == s
        a_mask = a == s
        support = int(m_mask.sum())
        recall = float(np.mean(a[m_mask] == s)) if support else float("nan")
        prec_denom = int(a_mask.sum())
        precision = float(np.mean(m[a_mask] == s)) if prec_denom else float("nan")
        per_class[s] = {"support": support, "recall": recall, "precision": precision}
    return {"n": n, "overall_agreement": overall, "per_class": per_class}
