"""Fail-closed price_history panel normalizer (data-layer ownership).

Quarantines the mixed-unit vendor ``daily_change_pct`` and the non-PIT tick
table behind one deterministic pass applied at every price_history load
boundary. Never drops rows: invalid rows carry NaN ``chg_ratio``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.execution.cost_model import TICK_REFORM_DATE, tick_cost_bp
from src.strategy.contract import (
    KRX_DAILY_LIMIT_RATIO,
    derive_chg_ratio,
    detect_mixed_unit_rows,
    mark_ceiling,
)

__all__ = [
    "PANEL_INTEGRITY_COLUMNS",
    "REQUIRED_SOURCE_COLUMNS",
    "PanelIntegrityError",
    "PanelProvenance",
    "assert_price_history_units_clean",
    "heal_price_history_panel",
    "load_price_panel",
    "prepare_price_panel",
]

PANEL_INTEGRITY_COLUMNS: frozenset[str] = frozenset(
    {"chg_ratio", "tv_clean", "mc_clean", "is_ceiling", "tick_cost_bp"}
)

REQUIRED_SOURCE_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume",
        "market_cap_100m",
        "trade_value_100m",
    }
)


@dataclass(frozen=True)
class PanelProvenance:
    """Row-level provenance emitted with every prepared panel."""

    n_input: int
    n_symbols: int
    n_days: int
    date_min: str
    date_max: str
    n_unit_mismatch_vendor: int
    n_invalid_prev_close: int
    n_invalid_limit_violation: int
    n_invalid_price: int
    n_valid_chg: int
    n_ceiling: int
    n_symbols_absent_at_end: int
    n_rows_post_tick_reform: int
    n_days_post_tick_reform: int

    def to_dict(self) -> dict[str, Any]:
        """Return this record as a plain dict in field-declaration order."""
        return dataclasses.asdict(self)

    def to_log_kv(self) -> str:
        """Return a flat space-joined key=value string over to_dict()."""
        return " ".join(f"{k}={v}" for k, v in self.to_dict().items())


def prepare_price_panel(
    price_history_df: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    market_col: str = "market",
) -> tuple[pd.DataFrame, PanelProvenance]:
    """Normalize one price_history frame into a PIT-clean panel.

    Args:
        price_history_df: Raw vendor frame (never mutated).
        date_col: Name of the trading-date column.
        symbol_col: Name of the symbol column.
        market_col: Name of the board column (KOSPI/KOSDAQ).

    Returns:
        Tuple of (prepared frame, provenance record).
    """
    required = set(REQUIRED_SOURCE_COLUMNS - {"date", "symbol"}) | {date_col, symbol_col}
    missing = sorted(c for c in required if c not in price_history_df.columns)
    if missing:
        raise ValueError(
            f"prepare_price_panel is missing required columns: {missing}"
        )
    out = price_history_df.copy()
    # 캘린더 정규화: 날짜는 datetime, 심볼은 6자리 문자열.
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[symbol_col] = out[symbol_col].astype(str).str.zfill(6)
    out = out.sort_values([symbol_col, date_col], kind="stable").reset_index(drop=True)

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev = pd.to_numeric(out["prev_close"], errors="coerce").to_numpy(dtype=np.float64)
    # 단일 결정론적 원천: 벤더 컬럼은 값 생성에 절대 사용하지 않는다.
    chg = derive_chg_ratio(close, prev)
    out["chg_ratio"] = np.asarray(chg, dtype=np.float64)
    out["daily_change_pct"] = np.asarray(chg, dtype=np.float64)

    volume = pd.to_numeric(out["volume"], errors="coerce").to_numpy(dtype=np.float64)
    tv_raw = pd.to_numeric(out["trade_value_100m"], errors="coerce").to_numpy(dtype=np.float64)
    out["tv_clean"] = np.where(np.isfinite(tv_raw), tv_raw, close * volume / 1e8).astype(np.float64)
    # 미래 시가총액이 과거로 흐르면 안 되므로 심볼별 ffill만 허용.
    mc_raw = pd.to_numeric(out["market_cap_100m"], errors="coerce")
    out["mc_clean"] = mc_raw.groupby(out[symbol_col]).ffill().to_numpy(dtype=np.float64)

    # NaN chg는 상한이 될 수 없어 False가 된다.
    out["is_ceiling"] = np.asarray(mark_ceiling(out), dtype=bool)

    dates = pd.to_datetime(out[date_col], errors="coerce").to_numpy()
    if market_col in out.columns:
        market_values = out[market_col].astype(str).to_numpy(dtype=object)
    else:
        # UNKNOWN은 KOSPI 밴드로 해석되어 보수적 틱을 준다.
        market_values = np.full(len(out), "UNKNOWN", dtype=object)
    out["tick_cost_bp"] = np.asarray(
        tick_cost_bp(close, dates, market_values), dtype=np.float64
    )

    # --- provenance (입력 프레임 기준, 정렬 불변) ---
    n_input = len(price_history_df)
    sym = out[symbol_col].to_numpy()
    n_symbols = int(pd.Series(sym).nunique()) if len(out) else 0
    dts = pd.to_datetime(out[date_col], errors="coerce")
    n_days = int(dts.nunique()) if len(out) else 0
    if len(out) and dts.notna().any():
        date_min = str(dts.min().strftime("%Y-%m-%d"))
        date_max = str(dts.max().strftime("%Y-%m-%d"))
    else:
        date_min = ""
        date_max = ""
    if "daily_change_pct" in price_history_df.columns:
        v_close = pd.to_numeric(price_history_df["close"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        v_prev = pd.to_numeric(price_history_df["prev_close"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        v_vendor = pd.to_numeric(
            price_history_df["daily_change_pct"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        n_unit_mismatch_vendor = int(detect_mixed_unit_rows(v_close, v_prev, v_vendor).sum())
    else:
        n_unit_mismatch_vendor = 0
    n_invalid_prev_close = int((~np.isfinite(prev) | (prev <= 0.0)).sum())
    # 기업액션(분할/권리) 위반은 NaN으로 두고 하류에서 제외한다.
    both_valid = np.isfinite(prev) & (prev > 0.0) & np.isfinite(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        limit_breach = np.abs(close / prev - 1.0) > float(KRX_DAILY_LIMIT_RATIO)
    n_invalid_limit_violation = int((both_valid & limit_breach).sum())
    n_invalid_price = int((~np.isfinite(close) | (close <= 0.0)).sum())
    n_valid_chg = int(np.isfinite(chg).sum())
    n_ceiling = int(np.asarray(out["is_ceiling"].to_numpy(dtype=bool)).sum())
    if len(out) and dts.notna().any():
        panel_max = dts.max()
        per_sym_max = dts.groupby(pd.Series(sym).to_numpy()).max()
        n_symbols_absent_at_end = int((per_sym_max < panel_max).sum())
    else:
        n_symbols_absent_at_end = 0
    reform = np.datetime64(TICK_REFORM_DATE)
    d64 = dts.to_numpy()
    post_mask = (~pd.isna(dts).to_numpy()) & (d64 >= reform) if len(out) else np.zeros(0, dtype=bool)
    n_rows_post_tick_reform = int(post_mask.sum())
    n_days_post_tick_reform = int(pd.Series(d64[post_mask]).nunique()) if n_rows_post_tick_reform else 0

    provenance = PanelProvenance(
        n_input=n_input,
        n_symbols=n_symbols,
        n_days=n_days,
        date_min=date_min,
        date_max=date_max,
        n_unit_mismatch_vendor=n_unit_mismatch_vendor,
        n_invalid_prev_close=n_invalid_prev_close,
        n_invalid_limit_violation=n_invalid_limit_violation,
        n_invalid_price=n_invalid_price,
        n_valid_chg=n_valid_chg,
        n_ceiling=n_ceiling,
        n_symbols_absent_at_end=n_symbols_absent_at_end,
        n_rows_post_tick_reform=n_rows_post_tick_reform,
        n_days_post_tick_reform=n_days_post_tick_reform,
    )
    out.attrs["panel_provenance"] = provenance.to_dict()
    return out, provenance


def load_price_panel(
    path: str | Path,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    market_col: str = "market",
) -> tuple[pd.DataFrame, PanelProvenance]:
    """Load a price_history parquet through the fail-closed normalizer.

    Args:
        path: Parquet file path.
        date_col: Name of the trading-date column.
        symbol_col: Name of the symbol column.
        market_col: Name of the board column.

    Returns:
        Tuple of (prepared frame, provenance record).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"price_history parquet not found: {path}")
    frame = pd.read_parquet(path)
    return prepare_price_panel(
        frame, date_col=date_col, symbol_col=symbol_col, market_col=market_col
    )


class PanelIntegrityError(ValueError):
    """Raised when a stored change column disagrees with its own prices."""


def heal_price_history_panel(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Re-derive prev_close and change ratios over the whole merged panel.

    Args:
        df: Price history frame carrying close and prev_close (never mutated).
        date_col: Name of the trading-date column.
        symbol_col: Name of the symbol column.

    Returns:
        Sorted copy with healed prev_close, chg_ratio and daily_change_pct.
    """
    out = df.copy()
    # 캘린더 정규화: 날짜는 datetime, 심볼은 6자리 문자열.
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[symbol_col] = out[symbol_col].astype(str).str.zfill(6)
    out = out.sort_values([symbol_col, date_col], kind="stable").reset_index(drop=True)
    # 전체 패널 기준 재도출: 슬라이스 경계는 병합된 전일 종가로 치유한다.
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev_existing = pd.to_numeric(out["prev_close"], errors="coerce").to_numpy(dtype=np.float64)
    keys = out[symbol_col].to_numpy()
    shifted = pd.Series(close).groupby(keys, sort=False).shift(1).to_numpy(dtype=np.float64)
    # 결측 이전 종가만 전일 종가로 메운다; 운반된 유효값은 그대로 둔다.
    prev = np.where(np.isfinite(prev_existing), prev_existing, shifted)
    out["prev_close"] = np.asarray(prev, dtype=np.float64)
    # 단일 결정론적 원천: 벤더 컬럼은 값 생성에 절대 사용하지 않는다.
    chg = derive_chg_ratio(close, prev)
    out["chg_ratio"] = np.asarray(chg, dtype=np.float64)
    out["daily_change_pct"] = np.asarray(chg, dtype=np.float64)
    return out


def assert_price_history_units_clean(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    rtol: float = 1e-6,
) -> None:
    """Refuse a panel whose stored change column disagrees with its prices.

    Args:
        df: Price history frame to validate (never mutated).
        date_col: Name of the trading-date column.
        symbol_col: Name of the symbol column.
        rtol: Relative tolerance for the stored-vs-derived comparison.

    Returns:
        None on success.

    Raises:
        PanelIntegrityError: If a column is missing or any row mismatches.
    """
    required = {date_col, symbol_col, "close", "prev_close", "daily_change_pct"}
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise PanelIntegrityError(
            f"assert_price_history_units_clean is missing required columns: {missing}"
        )
    work = df.copy()
    # 결정론적 비교 순서: 심볼·날짜 정렬본 위에서 진릿값을 계산한다.
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[symbol_col] = work[symbol_col].astype(str).str.zfill(6)
    work = work.sort_values([symbol_col, date_col], kind="stable").reset_index(drop=True)
    close = pd.to_numeric(work["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev = pd.to_numeric(work["prev_close"], errors="coerce").to_numpy(dtype=np.float64)
    stored = pd.to_numeric(work["daily_change_pct"], errors="coerce").to_numpy(dtype=np.float64)
    # 단일 결정론적 원천과의 벡터화 비교; NaN 행은 동등 취급한다.
    truth = derive_chg_ratio(close, prev)
    ok = np.isclose(stored, truth, rtol=float(rtol), atol=0.0, equal_nan=True)
    n_mismatch = int((~np.asarray(ok)).sum())
    if n_mismatch:
        total = len(work)
        syms = work[symbol_col].astype(str).to_numpy()
        dates = pd.to_datetime(work[date_col], errors="coerce")
        bad = np.flatnonzero(~np.asarray(ok))[:3]
        # 최대 3개 행만 보고한다; 연산자는 추가 조회 없이 조치한다.
        parts = [
            f"({syms[i]}, {dates.iloc[i]}, stored={stored[i]!r}, expected={truth[i]!r})"
            for i in bad
        ]
        raise PanelIntegrityError(
            f"price_history panel has {n_mismatch} mismatching rows "
            f"out of {total}: " + "; ".join(parts)
        )
    return None
