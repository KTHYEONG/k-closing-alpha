"""Shared bundle cost/grade constants for the live path."""

from __future__ import annotations

# 왕복 거래 비용 = KRX 매도 거래세(2026-01-01 시행, 양시장 0.20%) + 검증된 1분봉 패널의 왕복 2틱 스프레드 중앙값(~26bp).
# 진입(~15:19)·청산(~09:00+) 모두 연속거래 체결로 스프레드를 크로싱한다. 결정→동시호가 드리프트(중앙값 0)는 cost_model의 행별 auction_impact_bp로 분리.
_STATUTORY_COST_RATIO: float = 0.0020
_SPREAD_COST_RATIO: float = 0.0026
ROUND_TRIP_COST_RATIO: float = _STATUTORY_COST_RATIO + _SPREAD_COST_RATIO

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
