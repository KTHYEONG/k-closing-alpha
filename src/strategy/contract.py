from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from src.execution.cost_model import spread_cost_bp


class ExecutionMode(StrEnum):
    AA = "AA"
    PA = "PA"


MIN_ROUND_TRIP_TICKS: dict[ExecutionMode, float] = {ExecutionMode.AA: 2.0, ExecutionMode.PA: 1.0}

CEILING_CHG_THRESHOLD: float = 0.29

APPROXIMATE_SPEARMAN_THRESHOLD: float = 0.99


@dataclass(frozen=True)
class CostSpec:
    mode: ExecutionMode = ExecutionMode.AA
    statutory_bp: float = 20.0
    round_trip_ticks: float = 2.0

    def __post_init__(self) -> None:
        ticks = float(self.round_trip_ticks)
        if not math.isfinite(ticks):
            raise ValueError(f"round_trip_ticks must be finite, got {self.round_trip_ticks!r}")
        floor = float(MIN_ROUND_TRIP_TICKS[self.mode])
        if ticks < floor:
            raise ValueError(f"round_trip_ticks {ticks} below floor {floor} for mode {self.mode}")
        if float(self.statutory_bp) < 0.0:
            raise ValueError(f"statutory_bp must be >= 0, got {self.statutory_bp!r}")


@dataclass(frozen=True)
class UniverseSpec:
    chg_min: float = 0.02
    chg_max: float = 0.10
    min_trade_value_100m: float = 100.0
    min_market_cap_100m: float = 500.0
    exclude_ceiling: bool = True


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    top_k: int
    universe: UniverseSpec
    cost: CostSpec

    def fingerprint(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureContractRow:
    feature: str
    historical_source: str
    live_source: str | None
    status: str
    spearman: float | None
    is_cross_sectional_rank: bool
    action: str


AA_COST: CostSpec = CostSpec(mode=ExecutionMode.AA, statutory_bp=20.0, round_trip_ticks=2.0)

PA_COST: CostSpec = CostSpec(mode=ExecutionMode.PA, statutory_bp=20.0, round_trip_ticks=1.0)

DEFAULT_UNIVERSE: UniverseSpec = UniverseSpec()

KCA_TOP3_SHADOW_001: StrategySpec = StrategySpec(
    strategy_id="KCA-TOP3-SHADOW-001", top_k=3, universe=DEFAULT_UNIVERSE, cost=AA_COST
)


def round_trip_cost_bp(price: np.ndarray | float, cost: CostSpec = AA_COST) -> np.ndarray:
    arr = np.asarray(price, dtype=np.float64)
    spread = spread_cost_bp(arr, round_trip_ticks=float(cost.round_trip_ticks))
    return np.asarray(float(cost.statutory_bp) + spread, dtype=np.float64)


def mark_ceiling(df: pd.DataFrame) -> np.ndarray:
    required = ("chg_ratio", "close", "high")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"mark_ceiling missing required columns: {missing}")
    chg = df["chg_ratio"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    return np.asarray((chg >= CEILING_CHG_THRESHOLD) & (close >= high), dtype=bool)


def select_universe(df: pd.DataFrame, spec: UniverseSpec = DEFAULT_UNIVERSE) -> np.ndarray:
    required = ["chg_ratio", "tv_clean", "mc_clean", "close", "volume"]
    if spec.exclude_ceiling:
        required = [*required, "is_ceiling"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"select_universe missing required columns: {missing}")
    chg = df["chg_ratio"].to_numpy(dtype=np.float64)
    tv = df["tv_clean"].to_numpy(dtype=np.float64)
    mc = df["mc_clean"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)
    mask = (
        (chg >= float(spec.chg_min))
        & (chg < float(spec.chg_max))
        & (tv >= float(spec.min_trade_value_100m))
        & (mc >= float(spec.min_market_cap_100m))
        & (close > 0.0)
        & (vol > 0.0)
    )
    if spec.exclude_ceiling:
        ceiling = df["is_ceiling"].to_numpy(dtype=bool)
        mask = mask & (~ceiling)
    return np.asarray(mask, dtype=bool)


def _status_to_action(status: str) -> str:
    if status == "MATCH":
        return "KEEP"
    if status == "APPROXIMATE":
        return "JUSTIFY_OR_ALIGN"
    return "REMOVE_OR_REPLACE"


def classify_feature_contract(
    feature: str,
    historical_source: str,
    live_source: str | None,
    *,
    spearman: float | None = None,
    is_cross_sectional_rank: bool = False,
    populations_match: bool | None = None,
    approximate_threshold: float = APPROXIMATE_SPEARMAN_THRESHOLD,
) -> FeatureContractRow:
    if live_source is None:
        return FeatureContractRow(
            feature=feature,
            historical_source=historical_source,
            live_source=None,
            status="UNAVAILABLE_LIVE",
            spearman=None,
            is_cross_sectional_rank=bool(is_cross_sectional_rank),
            action=_status_to_action("UNAVAILABLE_LIVE"),
        )
    if spearman is None:
        raise ValueError(f"spearman measurement required for feature {feature!r}")
    rho = float(spearman)
    if bool(is_cross_sectional_rank) and populations_match is not True:
        status = "MISMATCH"
    elif rho >= 1.0 - 1e-9:
        status = "MATCH"
    elif rho >= float(approximate_threshold):
        status = "APPROXIMATE"
    else:
        status = "MISMATCH"
    return FeatureContractRow(
        feature=feature,
        historical_source=historical_source,
        live_source=live_source,
        status=status,
        spearman=rho,
        is_cross_sectional_rank=bool(is_cross_sectional_rank),
        action=_status_to_action(status),
    )


def assert_production_feature_set(
    rows: Sequence[FeatureContractRow],
    *,
    justified_approximates: frozenset[str] = frozenset(),
) -> None:
    offending = [
        r.feature
        for r in rows
        if r.status in ("MISMATCH", "UNAVAILABLE_LIVE")
        or (r.status == "APPROXIMATE" and r.feature not in justified_approximates)
    ]
    if offending:
        raise ValueError(f"production feature set rejected: {offending}")
    return None
