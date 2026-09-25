"""Shared bundle cost/grade constants for the live path."""

from __future__ import annotations

from src.execution.cost_model import ROUND_TRIP_COST_RATIO  # noqa: F401  # single-source re-export

# 왕복 거래 비용 = 평탄 법정비용 + 평탄 스프레드 + 왕복 수수료로 cost_model이 단일 소유한다.

__all__ = [
    "ROUND_TRIP_COST_RATIO",
    "_GOOD_PCT",
    "_GRADE_MULTIPLIERS",
    "_QUANTILE_ALPHAS",
    "_QUANTILE_COLS",
    "_STRONG_PCT",
    "_WEAK_PCT",
]

_QUANTILE_COLS = ("pred_q10", "pred_q50", "pred_q90")
_QUANTILE_ALPHAS = (0.10, 0.50, 0.90)

_STRONG_PCT = 0.90
_GOOD_PCT = 0.75
_WEAK_PCT = 0.50

_GRADE_MULTIPLIERS: dict[str, float] = {
    "Strong": 1.5,
    "Good": 1.0,
    "Weak": 0.5,
    "Pass": 0.0,
}
