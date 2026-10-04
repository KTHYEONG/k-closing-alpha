"""Core research and walk-forward validation engine for Research Validation v3.

Enforces zero lookahead leakage, strict calendar-aware walk-forward validation,
point-in-time universe selection, realistic cost models, and discrete portfolio NAV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.panel_integrity import prepare_price_panel
from src.data.screenable_class import attach_screenable_class, load_classification_panel
from src.strategy.contract import (
    AA_COST,
    DEFAULT_UNIVERSE,
    PA_COST,
    UniverseSpec,
    round_trip_cost_bp,
    select_universe,
)

logger = logging.getLogger("research_v3_engine")

# 실측 왕복비용 상단 시나리오(bp). 라벨이 아니라 스트레스 시나리오 전용이므로 PIT 스케줄과 독립이다.
STRESS_COST_BP: float = 46.0


def load_and_prepare_price_history(
    path: Path | str,
    *,
    classification_path: Path | str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[pd.Timestamp, int]]:
    """Load price history, attach the PIT security-class verdict and index the calendar.

    Args:
        path: price_history parquet.
        classification_path: Classification panel parquet; None selects
            settings.ALTDATA_DIR / SECURITY_CLASSIFICATION_PARQUET_FILENAME.

    Returns:
        (prepared panel carrying SCREENABLE_CLASS_COL and SCREENABLE_SOURCE_COL,
        sorted trading calendar, date-to-index lookup).

    Raises:
        FileNotFoundError: When the classification panel does not exist.
        ValueError: Propagated from attach_screenable_class (coverage gap) or the loader.
    """
    logger.info("Loading price history from %s...", path)
    ph, prov = prepare_price_panel(pd.read_parquet(path))
    logger.info("[DATA] stage=panel_integrity %s", prov.to_log_kv())

    # Baseline trading calendar
    market_dates = np.array(sorted(ph["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}

    classification = load_classification_panel(classification_path)
    ph, class_prov = attach_screenable_class(ph, classification, market_dates=market_dates)
    logger.info("[DATA] stage=screenable_class %s", class_prov.to_log_kv())

    return ph, market_dates, d_to_idx


def build_candidate_universe(
    ph: pd.DataFrame, spec: UniverseSpec = DEFAULT_UNIVERSE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct U0_PIT and U3_PIT universes without future tradability filters.

    Args:
        ph: Prepared price-history panel.
        spec: Universe screen. Defaults to DEFAULT_UNIVERSE; pass COST_AWARE_UNIVERSE
            to add the per-tick cost cap (requires a ``tick_cost_bp`` column).
    """
    logger.info("Filtering PIT candidate universes U0 and U3...")

    # U0_PIT mask: 2% <= chg < 10%, tv >= 100억, mc >= 500억, not ceiling, close > 0, vol > 0
    u0_mask = select_universe(ph, spec)

    u0_df = ph[u0_mask].copy().sort_values(["date", "symbol"]).reset_index(drop=True)

    # U3_PIT mask: U0 + mc >= 5000억 + inst_netbuy > 0 + market index > 0
    is_kosdaq = u0_df["market"].astype(str).str.upper().str.contains("KOSDAQ")
    mkt_idx_pos = np.where(is_kosdaq, u0_df["kosdaq_pct"] > 0, u0_df["kospi_pct"] > 0)
    u3_mask = (u0_df["mc_clean"] >= 5000.0) & (u0_df["inst_netbuy"].fillna(0) > 0) & mkt_idx_pos

    u0_df["is_u3"] = u3_mask
    return u0_df, ph


def attach_forward_exit_paths(
    cands: pd.DataFrame,
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
) -> pd.DataFrame:
    """Attach forward exit prices with suspension tracking and carry rule.

    The exit is the first tradable D+k open (k = 1..20); a candidate that never resumes within 20
    trading days exits at a 50% haircut of its close, dated D+20.

    Args:
        cands: Candidate rows with date, symbol, close and market columns.
        ph: Prepared price-history panel used for the forward lookup.
        market_dates: Full sorted trading calendar.
        d_to_idx: Date-to-index lookup into market_dates.

    Returns:
        cands with d1_tradable, exit_price, exit_status, holding_days, exit_date, gross/net return and
        cost columns. exit_date is the trading date whose open realizes the label
        (market_dates[idx(date) + holding_days]); NaT when no D+1 bar exists or the D+20 horizon of an
        unresolved exit runs past the calendar. Downstream CV uses it to purge label-overlapping rows.

    Raises:
        ValueError: When the market column is missing.
    """
    logger.info("Attaching calendar-aware forward exit paths...")
    lookup = ph.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume"]]
    n_market_dates = len(market_dates)

    date_indices = np.array([d_to_idx[d] for d in cands["date"]])
    valid_d1 = date_indices + 1 < n_market_dates
    d1_dates = np.where(valid_d1, market_dates[np.minimum(date_indices + 1, n_market_dates - 1)], pd.NaT)
    syms = cands["symbol"].to_numpy()

    # D1 lookup
    keys_d1 = list(zip(d1_dates, syms, strict=False))
    idx_tuples_d1 = pd.MultiIndex.from_tuples(keys_d1, names=["date", "symbol"])
    joined_d1 = lookup.reindex(idx_tuples_d1)

    j_open_d1 = joined_d1["open"].to_numpy(dtype=np.float64)
    j_vol_d1 = joined_d1["volume"].to_numpy(dtype=np.float64)
    tradable_d1 = valid_d1 & np.isfinite(j_open_d1) & (j_open_d1 > 0.0) & (j_vol_d1 > 0.0)

    exit_prices = np.full(len(cands), np.nan, dtype=np.float64)
    exit_status = np.full(len(cands), "EXIT_TRADABLE", dtype=object)
    holding_days = np.ones(len(cands), dtype=np.int32)

    # Direct tradable on D1
    exit_prices[tradable_d1] = j_open_d1[tradable_d1]

    # Handle suspended D1 candidates: search up to 20 days ahead
    suspended_indices = np.where(~tradable_d1 & valid_d1)[0]
    logger.info("Resolving %d suspended/untradable candidates...", len(suspended_indices))

    for idx in suspended_indices:
        sym = syms[idx]
        cur_d_idx = date_indices[idx]
        found = False
        for step in range(2, 21):
            if cur_d_idx + step >= n_market_dates:
                break
            nxt_date = market_dates[cur_d_idx + step]
            try:
                row = lookup.loc[(nxt_date, sym)]
                op = float(row["open"]) if not isinstance(row, pd.DataFrame) else float(row.iloc[0]["open"])
                vl = float(row["volume"]) if not isinstance(row, pd.DataFrame) else float(row.iloc[0]["volume"])
                if np.isfinite(op) and op > 0.0 and vl > 0.0:
                    exit_prices[idx] = op
                    holding_days[idx] = step
                    exit_status[idx] = "EXIT_SUSPENDED"
                    found = True
                    break
            except KeyError:
                continue

        if not found:
            # If never resumes in 20 days, mark as unresolved exit
            exit_prices[idx] = cands.iloc[idx]["close"] * 0.50  # Conservative haircut
            holding_days[idx] = 20
            exit_status[idx] = "UNRESOLVED_EXIT"

    cands["d1_tradable"] = tradable_d1
    cands["exit_price"] = exit_prices
    cands["exit_status"] = exit_status
    cands["holding_days"] = holding_days
    # 라벨 실현일: 보유일수만큼 앞선 거래일의 시가가 라벨을 확정한다
    exit_idx = date_indices.astype(np.int64) + holding_days.astype(np.int64)
    calendar = pd.to_datetime(pd.Series(market_dates)).to_numpy(dtype="datetime64[ns]")
    exit_dates = np.full(len(cands), np.datetime64("NaT"), dtype="datetime64[ns]")
    in_range = valid_d1 & (exit_idx < n_market_dates)
    exit_dates[in_range] = calendar[exit_idx[in_range]]
    cands["exit_date"] = exit_dates

    # Calculate returns and costs
    entry_p = cands["close"].to_numpy(dtype=np.float64)
    if "market" not in cands.columns:
        raise ValueError("attach_forward_exit_paths requires a 'market' column for point-in-time tick costing")
    trade_dates = pd.to_datetime(cands["date"]).to_numpy()
    markets = cands["market"].astype(str).to_numpy(dtype=object)
    # 왕복비용의 틱 크기는 원가격 기준, 수익률은 수정주가 체인 기준.
    level_p = _level_close(cands)
    cost_aa_bp = round_trip_cost_bp(level_p, trade_dates, markets, AA_COST)
    cost_pa_bp = round_trip_cost_bp(level_p, trade_dates, markets, PA_COST)
    cost_stress_bp = np.full(len(cands), STRESS_COST_BP, dtype=np.float64)

    gross_ret = exit_prices / entry_p - 1.0
    cands["gross_return"] = gross_ret
    cands["cost_aa_bp"] = cost_aa_bp
    cands["cost_pa_bp"] = cost_pa_bp
    cands["cost_stress_bp"] = cost_stress_bp

    cands["net_return_aa"] = gross_ret - cost_aa_bp / 10000.0
    cands["net_return_pa"] = gross_ret - cost_pa_bp / 10000.0
    cands["net_return_stress"] = gross_ret - cost_stress_bp / 10000.0

    return cands


# 가격 레벨(금액·틱) 계산은 원가격, 수정주가는 비율 피처 전용.
def _level_close(frame: pd.DataFrame) -> np.ndarray:
    """Return the raw price level per row, falling back to close."""
    if "close_raw" in frame.columns:
        return np.asarray(
            pd.to_numeric(frame["close_raw"], errors="coerce").fillna(frame["close"]).to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
    return np.asarray(frame["close"].to_numpy(dtype=np.float64), dtype=np.float64)
