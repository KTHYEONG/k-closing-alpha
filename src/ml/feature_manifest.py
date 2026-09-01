"""Feature manifest deterministic port."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

FEATURE_MANIFEST_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "source_column",
    "availability_rule",
    "unit",
    "panel_scope",
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

_BINARY_INDICATOR_FEATURES: frozenset[str] = frozenset(
    {
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
    }
)

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


def build_feature_manifest(
    feature_cols: list[str],
    catalogue: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """학습/추론 피처 목록에 대한 결정적 매니페스트 DataFrame 을 생성합니다."""
    if isinstance(feature_cols, str):
        raise TypeError("feature_cols must be a list of feature names, not a string")
    catalogue = catalogue or {}
    rows = [
        {
            "feature_name": feature,
            "source_column": _catalogue_get(catalogue, feature, "source_column", feature),
            "availability_rule": _catalogue_get(
                catalogue, feature, "availability_rule", "at_decision_time"
            ),
            "unit": _feature_unit(feature),
            "panel_scope": _catalogue_get(catalogue, feature, "panel_scope", "candidate_panel"),
        }
        for feature in feature_cols
    ]
    return pd.DataFrame(rows, columns=list(FEATURE_MANIFEST_COLUMNS))


def _catalogue_get(
    catalogue: Mapping[str, Mapping[str, str]],
    feature: str,
    key: str,
    default: str,
) -> str:
    """카탈로그 오버라이드에서 값(기본값)을 결정적으로 반환합니다."""
    meta = catalogue.get(feature)
    if meta is None:
        return default
    return meta.get(key, default)
