"""Decision-time feature manifest: source, availability rule, unit, panel scope.

`docs/specs/ml_close_to_morning_quality.md` 계약: 모든 피처는 15:20 KST 공통
소스 스냅샷의 availability 메타데이터로 기록됩니다. 피처 단위와 이용 가능성
규칙이 모델 번들에 영속화되며, 피처 이름으로부터만 결정적으로 유도됩니다
(데이터 의존성 없음). 행 단위 타임스탬프 검증/타임스탬프 파생 규칙은 사용하지
않습니다 — 고정된 업무 원천 규칙입니다.

Availability rule:
    ``at_decision_time``  -> 수용된 공통 15:20 KST 소스 스냅샷에서 decision 시점에
                             이용 가능한 피처 (횡단면 순위, 수급 밀도, 시장 컨텍스트,
                             캔들/실현 매수가 파생 피처 포함).
"""

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


def build_feature_manifest(
    feature_cols: list[str],
    catalogue: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """학습/추론 피처 목록에 대한 결정적 매니페스트 DataFrame 을 생성합니다.

    ``catalogue`` 가 주어지면 해당 피처명의 ``source_column`` / ``availability_rule``
    / ``panel_scope`` 를 오버라이드합니다 (``src.ml.history_features`` 카탈로그
    계약). 카탈로그에 없는 피처는 기존 ``at_decision_time`` 규칙을 유지하므로
    기존 호출 동작은 변경되지 않습니다.

    Returns:
        컬럼: ``feature_name``, ``source_column``, ``availability_rule``,
        ``unit``, ``panel_scope`` (순서 고정).
    """
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
