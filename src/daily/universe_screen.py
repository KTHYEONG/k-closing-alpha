import numpy as np
import pandas as pd

from src.execution.cost_model import tick_cost_bp
from src.strategy.contract import (
    PRODUCTION_STRATEGY,
    SCREENABLE_CLASS_COL,
    UniverseSpec,
    derive_chg_ratio,
    mark_ceiling,
    select_universe,
    training_universe,
)

_PRODUCTION_TRAINING_SCREEN: UniverseSpec = training_universe(PRODUCTION_STRATEGY.universe)


def build_screen_frame(df: pd.DataFrame, *, decision_date: pd.Timestamp) -> pd.DataFrame:
    """Map a Korean-column decision snapshot to the select_universe input frame.

    Args:
        df: Decision snapshot (종가/전일종가/고가/거래량/거래대금/시가총액/시장구분, and
            optionally SCREENABLE_CLASS_COL written by assemble_decision_frame).
        decision_date: Decision date for point-in-time tick costing.

    Returns:
        Frame with chg_ratio, is_ceiling, tick_cost_bp, tv_clean, mc_clean, close, volume,
        plus SCREENABLE_CLASS_COL (bool) when the snapshot carries it.

    Raises:
        ValueError: When the snapshot's SCREENABLE_CLASS_COL contains nulls.
    """
    close = pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = pd.to_numeric(df["전일종가"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["고가"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(df["거래량"], errors="coerce").to_numpy(dtype=np.float64)
    tv_clean = pd.to_numeric(df["거래대금"], errors="coerce").to_numpy(dtype=np.float64)
    mc_clean = pd.to_numeric(df["시가총액"], errors="coerce").to_numpy(dtype=np.float64)
    chg_ratio = derive_chg_ratio(close, prev_close)
    is_ceiling = mark_ceiling(pd.DataFrame({"chg_ratio": chg_ratio, "close": close, "high": high}))
    out = pd.DataFrame(
        {
            "chg_ratio": chg_ratio,
            "is_ceiling": is_ceiling,
            "tick_cost_bp": tick_cost_bp(
                close,
                np.full(len(df), np.datetime64(decision_date.strftime("%Y-%m-%d"))),
                df["시장구분"].astype(str).to_numpy(dtype=object),
            ),
            "tv_clean": tv_clean,
            "mc_clean": mc_clean,
            "close": close,
            "volume": volume,
        }
    )
    if SCREENABLE_CLASS_COL in df.columns:
        vals = df[SCREENABLE_CLASS_COL]
        if vals.isna().any():
            raise ValueError(f"build_screen_frame {SCREENABLE_CLASS_COL} contains nulls")
        out[SCREENABLE_CLASS_COL] = np.asarray(vals.astype(bool).to_numpy(), dtype=bool)
    return out


def rank_pool_mask(
    df: pd.DataFrame,
    *,
    decision_date: pd.Timestamp,
    screen: UniverseSpec = _PRODUCTION_TRAINING_SCREEN,
) -> np.ndarray:
    """Return the training rank-pool mask of a decision snapshot.

    Args:
        df: Decision snapshot (build_screen_frame input, carrying SCREENABLE_CLASS_COL).
        decision_date: Decision date for point-in-time tick costing.
        screen: Training screen; defaults to the wide screen of PRODUCTION_STRATEGY.

    Returns:
        Bool mask aligned with df.

    Raises:
        ValueError: When the snapshot lacks a column the screen requires (including
            SCREENABLE_CLASS_COL under a class-filtered screen).
    """
    return np.asarray(select_universe(build_screen_frame(df, decision_date=decision_date), screen), dtype=bool)
