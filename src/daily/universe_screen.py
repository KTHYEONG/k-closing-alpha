import numpy as np
import pandas as pd

from src.execution.cost_model import tick_cost_bp
from src.strategy.contract import DEFAULT_UNIVERSE, UniverseSpec, derive_chg_ratio, mark_ceiling, select_universe


def build_screen_frame(df: pd.DataFrame, *, decision_date: pd.Timestamp) -> pd.DataFrame:
    close = pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = pd.to_numeric(df["전일종가"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["고가"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(df["거래량"], errors="coerce").to_numpy(dtype=np.float64)
    tv_clean = pd.to_numeric(df["거래대금"], errors="coerce").to_numpy(dtype=np.float64)
    mc_clean = pd.to_numeric(df["시가총액"], errors="coerce").to_numpy(dtype=np.float64)
    chg_ratio = derive_chg_ratio(close, prev_close)
    is_ceiling = mark_ceiling(pd.DataFrame({"chg_ratio": chg_ratio, "close": close, "high": high}))
    return pd.DataFrame(
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


def rank_pool_mask(df: pd.DataFrame, *, decision_date: pd.Timestamp, screen: UniverseSpec = DEFAULT_UNIVERSE) -> np.ndarray:
    return np.asarray(select_universe(build_screen_frame(df, decision_date=decision_date), screen), dtype=bool)
