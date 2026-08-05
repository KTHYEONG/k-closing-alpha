"""Decision-time feature manifest: source, availability rule, unit, panel scope.

`docs/specs/ml_strategy_improvement.md` P0 요구사항: 선정된 모든 피처는 decision
timestamp 보다 늦지 않게 이용 가능해야 하고, 피처 단위와 이용 가능성 규칙이 모델
번들에 영속화되어야 합니다. 매니페스트는 피처 이름으로부터만 결정적으로 유도되며
데이터 의존성이 없습니다.

Availability rule:
    ``at_decision_time``  -> decision 시점에 이용 가능한 피처 (횡단면 순위, 수급
                             밀도, 시장 컨텍스트 등).
    ``needs_snapshot_proof`` -> close/high/low 또는 실현 매수 가격 파생 피처. 스냅샷이
                             주문 전에 캡처됨이 증명될 때까지 프로덕션 피처 집합에서
                             제외 대상입니다.
"""

from __future__ import annotations

import pandas as pd

FEATURE_MANIFEST_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "source_column",
    "availability_rule",
    "unit",
    "panel_scope",
)

# close/high/low/실현 매수가 파생 피처: 주문 시점 이용 가능성이 아직 증명되지 않음.
_CANDLE_PRICE_DERIVED: frozenset[str] = frozenset(
    {
        "close_position",
        "body_ratio",
        "upper_shadow_ratio",
        "intraday_range",
        "intraday_return",
        "buy_price_change_rate",
        "gap_ratio",
        "relative_change_rate",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "prev_close_price",
        "buy_price",
        "sell_price",
    }
)

_PERCENT_FEATURES: frozenset[str] = frozenset(
    {
        "change_rate",
        "kospi_change",
        "kosdaq_change",
        "buy_price_change_rate",
        "gap_ratio",
        "intraday_range",
        "relative_change_rate",
        "relative_change_kospi",
        "relative_change_kosdaq",
        "sector_relative_change",
    }
)

_LOG_AMOUNT_FEATURES: frozenset[str] = frozenset(
    {"log_market_cap_100m", "log_trade_value_100m", "log_volume", "log_avg_trade_value"}
)

_INDEX_LEVEL_FEATURES: frozenset[str] = frozenset({"v_kospi", "v_kosdaq"})

# production_calendar_flow 후보: 요일 one-hot 지표.
_BINARY_INDICATOR_FEATURES: frozenset[str] = frozenset(
    {
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
    }
)

# production_calendar_flow 후보: 부호 합산 개수.
_SIGNED_COUNT_FEATURES: frozenset[str] = frozenset({"flow_consensus"})


def _feature_unit(feature: str) -> str:
    """피처 이름으로부터 결정적 단위 라벨을 반환합니다."""
    if feature in _BINARY_INDICATOR_FEATURES:
        return "binary_indicator"
    if feature in _SIGNED_COUNT_FEATURES:
        return "signed_count"
    if feature.endswith("_z"):
        return "robust_z"
    if feature.endswith("_pct_rank"):
        return "pct_rank"
    if feature in _PERCENT_FEATURES:
        return "percent"
    if feature in _LOG_AMOUNT_FEATURES:
        return "log_won_100m"
    if feature in _INDEX_LEVEL_FEATURES:
        return "index_level"
    if feature in ("close_position", "body_ratio", "upper_shadow_ratio"):
        return "decimal_ratio"
    if feature.endswith("_change") and feature.startswith("v_"):
        return "decimal_change"
    return "decimal_ratio"


def build_feature_manifest(feature_cols: list[str]) -> pd.DataFrame:
    """학습/추론 피처 목록에 대한 결정적 매니페스트 DataFrame 을 생성합니다.

    Returns:
        컬럼: ``feature_name``, ``source_column``, ``availability_rule``,
        ``unit``, ``panel_scope`` (순서 고정).
    """
    if isinstance(feature_cols, str):
        raise TypeError("feature_cols must be a list of feature names, not a string")
    rows: list[dict[str, str]] = []
    for feature in feature_cols:
        rule = "needs_snapshot_proof" if feature in _CANDLE_PRICE_DERIVED else "at_decision_time"
        rows.append(
            {
                "feature_name": feature,
                "source_column": feature,
                "availability_rule": rule,
                "unit": _feature_unit(feature),
                "panel_scope": "candidate_panel",
            }
        )
    return pd.DataFrame(rows, columns=list(FEATURE_MANIFEST_COLUMNS))
