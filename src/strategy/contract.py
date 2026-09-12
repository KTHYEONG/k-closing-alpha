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

# tick_cost_bp 와 statutory_bp_asof 는 UniverseSpec.max_tick_cost_bp 가 소비하는
# 비용 컬럼과 PIT 법정비용의 생산자다. 스크린 계약과 함께 쓰이므로 이 모듈에서 재수출한다.
from src.execution.cost_model import statutory_bp_asof, tick_cost_bp

__all__ = [
    "AA_COST",
    "APPROXIMATE_SPEARMAN_THRESHOLD",
    "CAPFREE_UNIVERSE",
    "CEILING_CHG_THRESHOLD",
    "COST_AWARE_UNIVERSE",
    "DEFAULT_REALIZED_VOL",
    "DEFAULT_UNIVERSE",
    "KCA_TOP3_SHADOW_001",
    "KCA_TOPK_CAPFREE_001",
    "KCA_TOPK_COSTAWARE_001",
    "KRX_DAILY_LIMIT_RATIO",
    "LABEL_BAD_THRESHOLD",
    "LABEL_GOOD_THRESHOLD",
    "MAX_TICK_COST_BP",
    "MIN_PATH_WIN_RATE",
    "MIN_ROUND_TRIP_TICKS",
    "PA_COST",
    "CostSpec",
    "ExecutionMode",
    "FeatureContractRow",
    "StrategySpec",
    "UniverseSpec",
    "assert_production_feature_set",
    "classify_feature_contract",
    "derive_chg_ratio",
    "detect_mixed_unit_rows",
    "mark_ceiling",
    "round_trip_cost_bp",
    "select_universe",
    "statutory_bp_asof",
    "tick_cost_bp",
]


class ExecutionMode(StrEnum):
    AA = "AA"
    PA = "PA"


MIN_ROUND_TRIP_TICKS: dict[ExecutionMode, float] = {ExecutionMode.AA: 2.0, ExecutionMode.PA: 1.0}

CEILING_CHG_THRESHOLD: float = 0.29

# 비용인식 스크린의 1틱 비용 상한 (bp)
# 12.0bp 근거: 후보풀 2.6배 확대로 리랭커 선택폭 확보; K=1~8 및 왕복 0~8틱 전 구간에서 7.5bp 지배.
MAX_TICK_COST_BP: float = 12.0
# 실현변동성 폴백 (불확실성 지표가 아닌 시그마 추정용)
DEFAULT_REALIZED_VOL: float = 0.02
# CPCV 경로승률 게이트 (검증 임계값)
MIN_PATH_WIN_RATE: float = 0.60
# 비용차감 후 '좋음' 라벨 경계
LABEL_GOOD_THRESHOLD: float = 0.01
# 비용차감 후 '나쁨' 라벨 경계
LABEL_BAD_THRESHOLD: float = -0.02

KRX_DAILY_LIMIT_RATIO: float = 0.31

APPROXIMATE_SPEARMAN_THRESHOLD: float = 0.99


# 법정비용은 날짜 함수이므로 전략 스펙 상수가 될 수 없다.
@dataclass(frozen=True)
class CostSpec:
    mode: ExecutionMode = ExecutionMode.AA
    round_trip_ticks: float = 2.0

    def __post_init__(self) -> None:
        ticks = float(self.round_trip_ticks)
        if not math.isfinite(ticks):
            raise ValueError(f"round_trip_ticks must be finite, got {self.round_trip_ticks!r}")
        floor = float(MIN_ROUND_TRIP_TICKS[self.mode])
        if ticks < floor:
            raise ValueError(f"round_trip_ticks {ticks} below floor {floor} for mode {self.mode}")


@dataclass(frozen=True)
class UniverseSpec:
    chg_min: float = 0.02
    chg_max: float = 0.10
    min_trade_value_100m: float = 100.0
    min_market_cap_100m: float = 500.0
    exclude_ceiling: bool = True
    max_tick_cost_bp: float | None = None


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


AA_COST: CostSpec = CostSpec(mode=ExecutionMode.AA, round_trip_ticks=2.0)

PA_COST: CostSpec = CostSpec(mode=ExecutionMode.PA, round_trip_ticks=1.0)

DEFAULT_UNIVERSE: UniverseSpec = UniverseSpec()

KCA_TOP3_SHADOW_001: StrategySpec = StrategySpec(
    strategy_id="KCA-TOP3-SHADOW-001", top_k=3, universe=DEFAULT_UNIVERSE, cost=AA_COST
)

COST_AWARE_UNIVERSE: UniverseSpec = UniverseSpec(
    chg_min=0.02,
    chg_max=0.10,
    min_trade_value_100m=100.0,
    min_market_cap_100m=500.0,
    exclude_ceiling=True,
    max_tick_cost_bp=MAX_TICK_COST_BP,
)

KCA_TOPK_COSTAWARE_001: StrategySpec = StrategySpec(
    strategy_id="KCA-TOPK-COSTAWARE-001", top_k=3, universe=COST_AWARE_UNIVERSE, cost=AA_COST
)

# 절대 틱비용 상한은 가격의 계단함수라 레짐마다 다른 가격창을 의미한다. 비용은 net 라벨로만 반영한다.
CAPFREE_UNIVERSE: UniverseSpec = UniverseSpec(
    chg_min=0.02,
    chg_max=0.10,
    min_trade_value_100m=100.0,
    min_market_cap_100m=500.0,
    exclude_ceiling=True,
    max_tick_cost_bp=None,
)

KCA_TOPK_CAPFREE_001: StrategySpec = StrategySpec(
    strategy_id="KCA-TOPK-CAPFREE-001", top_k=3, universe=CAPFREE_UNIVERSE, cost=AA_COST
)


def derive_chg_ratio(close: np.ndarray, prev_close: np.ndarray, *, limit: float = KRX_DAILY_LIMIT_RATIO) -> np.ndarray:
    """Deterministic close/prev_close - 1; bad prev_close and limit violations yield NaN."""
    c = np.asarray(close, dtype=np.float64)
    p = np.asarray(prev_close, dtype=np.float64)
    out = np.full(c.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(c) & np.isfinite(p) & (p > 0.0)
    np.divide(c, p, out=out, where=valid)
    ratio = out - 1.0
    over = valid & np.isfinite(ratio) & (np.abs(ratio) > float(limit))
    ratio[over] = np.nan
    return np.asarray(ratio, dtype=np.float64)


def detect_mixed_unit_rows(
    close: np.ndarray, prev_close: np.ndarray, vendor_change: np.ndarray, *, rtol: float = 1e-4
) -> np.ndarray:
    """Flag rows where vendor_change matches ratio*100 but not ratio (percent-encoded)."""
    v = np.asarray(vendor_change, dtype=np.float64)
    ratio = derive_chg_ratio(np.asarray(close, dtype=np.float64), np.asarray(prev_close, dtype=np.float64))
    computable = np.isfinite(ratio) & np.isfinite(v)
    match_pct = np.isclose(v, ratio * 100.0, rtol=float(rtol), atol=0.0, equal_nan=False)
    match_ratio = np.isclose(v, ratio, rtol=float(rtol), atol=0.0, equal_nan=False)
    return np.asarray(computable & match_pct & (~match_ratio), dtype=bool)


def round_trip_cost_bp(price: np.ndarray | float, trade_date: np.ndarray, market: np.ndarray, cost: CostSpec = AA_COST) -> np.ndarray:
    arr = np.asarray(price, dtype=np.float64)
    per_tick = tick_cost_bp(arr, trade_date, market)
    statutory = statutory_bp_asof(trade_date)
    return np.asarray(statutory + float(cost.round_trip_ticks) * per_tick, dtype=np.float64)


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
    if spec.max_tick_cost_bp is not None:
        required = [*required, "tick_cost_bp"]
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
    if spec.max_tick_cost_bp is not None:
        tick_bp = df["tick_cost_bp"].to_numpy(dtype=np.float64)
        mask = mask & (tick_bp <= float(spec.max_tick_cost_bp))
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
