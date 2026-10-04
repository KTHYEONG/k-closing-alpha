"""EOD-observable superset of the 15:20 decision-time screen, used only to choose what to fetch or audit.

A symbol-day passes the 15:20 screen on decision-time values (chg at the last pre-cutoff trade, cumulative
trade value at 15:20, market cap at the 15:20 price). Those values are unknown for past days until their
minute bars are fetched, so acquisition must select from end-of-day values with margins wide enough to
contain the 15:20 screen. The resulting set is a fetch/coverage filter and must never be used as a training
or selection screen: it is defined on EOD values that are not observable at 15:20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.collection import CollectionSettings
from src.strategy.contract import DEFAULT_UNIVERSE, UniverseSpec

_REQUIRED_COLUMNS: tuple[str, ...] = ("symbol", "chg_ratio", "tv_clean", "mc_clean", "volume")


@dataclass(frozen=True)
class EodSupersetScreen:
    """EOD-value bounds whose pass set contains the decision-time training screen.

    Attributes:
        min_change_ratio: Inclusive lower bound on the EOD close/prev_close - 1.
        max_change_ratio: Exclusive upper bound on the EOD close/prev_close - 1.
        min_trade_value_100m: Inclusive floor on EOD trade value (100M KRW).
        min_market_cap_100m: Inclusive floor on EOD market cap (100M KRW).
        common_stock_only: Keep only codes whose last character is '0' and whose first is not '9'.
    """

    min_change_ratio: float
    max_change_ratio: float
    min_trade_value_100m: float
    min_market_cap_100m: float
    common_stock_only: bool

    @classmethod
    def from_profile(cls, profile: CollectionSettings, *, contains: UniverseSpec = DEFAULT_UNIVERSE) -> EodSupersetScreen:
        """Build the screen from CollectionSettings and prove it contains a training screen.

        Args:
            profile: Validated collection settings carrying the COLLECTION_PIT_BACKFILL_* fields.
            contains: Decision-time screen the superset must contain; defaults to the training screen.

        Returns:
            The screen with the profile's bounds.

        Raises:
            ValueError: Naming every bound that is tighter than the contained screen (min chg above
                contains.chg_min, max chg below contains.chg_max, tv floor above
                contains.min_trade_value_100m, mc floor above contains.min_market_cap_100m).

        Note:
            common_stock_only=True while contains.exclude_non_screenable_class is False is a documented
            approximation (OD-4), not an error; it breaks containment for non-common classes only.
        """
        screen = cls(
            min_change_ratio=float(profile.COLLECTION_PIT_BACKFILL_MIN_CHANGE_RATIO),
            max_change_ratio=float(profile.COLLECTION_PIT_BACKFILL_MAX_CHANGE_RATIO),
            min_trade_value_100m=float(profile.COLLECTION_PIT_BACKFILL_MIN_TRADE_VALUE_100M),
            min_market_cap_100m=float(profile.COLLECTION_PIT_BACKFILL_MIN_MARKET_CAP_100M),
            common_stock_only=bool(profile.COLLECTION_PIT_BACKFILL_COMMON_STOCK_ONLY),
        )
        tighter: list[str] = []
        if screen.min_change_ratio > float(contains.chg_min):
            tighter.append("min_change_ratio")
        if screen.max_change_ratio < float(contains.chg_max):
            tighter.append("max_change_ratio")
        if screen.min_trade_value_100m > float(contains.min_trade_value_100m):
            tighter.append("min_trade_value_100m")
        if screen.min_market_cap_100m > float(contains.min_market_cap_100m):
            tighter.append("min_market_cap_100m")
        if tighter:
            raise ValueError(f"EOD superset is tighter than the contained screen: {tighter}")
        return screen


def eod_superset_mask(panel: pd.DataFrame, screen: EodSupersetScreen) -> np.ndarray:
    """Flag prepared-panel rows whose EOD values pass the fetch superset.

    Args:
        panel: Output of src.data.panel_integrity.prepare_price_panel (columns symbol, chg_ratio,
            tv_clean, mc_clean, volume); chg/tv/mc share the training definitions by construction.
        screen: Superset bounds.

    Returns:
        Boolean array aligned to panel rows.

    Raises:
        ValueError: Naming every missing required column.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"eod_superset_mask missing required columns: {missing}")
    chg = pd.to_numeric(panel["chg_ratio"], errors="coerce").to_numpy(dtype=np.float64)
    tv = pd.to_numeric(panel["tv_clean"], errors="coerce").to_numpy(dtype=np.float64)
    mc = pd.to_numeric(panel["mc_clean"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(panel["volume"], errors="coerce").to_numpy(dtype=np.float64)
    mask = (
        (chg >= float(screen.min_change_ratio))
        & (chg < float(screen.max_change_ratio))
        & (tv >= float(screen.min_trade_value_100m))
        & (mc >= float(screen.min_market_cap_100m))
        & (volume > 0.0)
    )
    if screen.common_stock_only:
        from src.data.screenable_class import proxy_screenable

        mask = mask & proxy_screenable(panel["symbol"].astype(str))
    return np.asarray(mask, dtype=bool)
