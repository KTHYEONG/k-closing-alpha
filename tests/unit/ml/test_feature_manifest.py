"""Decision-time feature manifest 단위 테스트.

`docs/specs/ml_strategy_improvement.md` P0 요구사항: 선정 피처의 소스/이용가능성
규칙/단위/패널 범위가 결정적으로 기록되어 모델 번들에 영속화되어야 합니다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.feature_manifest import (
    FEATURE_MANIFEST_COLUMNS,
    build_feature_manifest,
)


def test_build_feature_manifest_deterministic() -> None:
    features = ["change_rate", "close_position", "major_density", "log_market_cap_100m"]
    first = build_feature_manifest(features)
    second = build_feature_manifest(features)
    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == list(FEATURE_MANIFEST_COLUMNS)


def test_build_feature_manifest_classifies_availability_rule() -> None:
    manifest = build_feature_manifest(
        ["close_position", "buy_price_change_rate", "change_rate", "major_density"]
    )
    rules = dict(zip(manifest["feature_name"], manifest["availability_rule"], strict=True))
    # 공통 15:20 KST 소스 스냅샷이 수용 계약이므로 모든 피처가 at_decision_time 입니다.
    assert rules["close_position"] == "at_decision_time"
    assert rules["buy_price_change_rate"] == "at_decision_time"
    assert rules["change_rate"] == "at_decision_time"
    assert rules["major_density"] == "at_decision_time"


def test_build_feature_manifest_units() -> None:
    manifest = build_feature_manifest(
        ["change_rate", "change_rate_z", "trade_value_pct_rank", "v_kospi"]
    )
    units = dict(zip(manifest["feature_name"], manifest["unit"], strict=True))
    assert units["change_rate"] == "percent"
    assert units["change_rate_z"] == "robust_z"
    assert units["trade_value_pct_rank"] == "pct_rank"
    assert units["v_kospi"] == "index_level"


def test_build_feature_manifest_production_calendar_flow_units() -> None:
    """production_calendar_flow 후보는 결정적으로 at_decision_time 이며 단위가 고정됩니다."""
    features = [
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "flow_consensus",
        "flow_alignment_direction",
        "flow_turnover",
        "friday_selection_rank_pct",
    ]
    manifest = build_feature_manifest(features)
    units = dict(zip(manifest["feature_name"], manifest["unit"], strict=True))
    rules = dict(zip(manifest["feature_name"], manifest["availability_rule"], strict=True))
    for name in features:
        assert rules[name] == "at_decision_time"
    assert units["weekday_is_monday"] == "binary_indicator"
    assert units["weekday_is_friday"] == "binary_indicator"
    assert units["flow_consensus"] == "signed_count"
    assert units["flow_alignment_direction"] == "decimal_ratio"
    assert units["flow_turnover"] == "decimal_ratio"
    assert units["friday_selection_rank_pct"] == "decimal_ratio"
    assert (manifest["panel_scope"] == "candidate_panel").all()


def test_build_feature_manifest_empty_input() -> None:
    manifest = build_feature_manifest([])
    assert manifest.empty
    assert manifest.columns.tolist() == list(FEATURE_MANIFEST_COLUMNS)


def test_build_feature_manifest_panel_scope_is_candidate() -> None:
    manifest = build_feature_manifest(["change_rate", "inst_density"])
    assert (manifest["panel_scope"] == "candidate_panel").all()


def test_build_feature_manifest_requires_list_input() -> None:
    with pytest.raises(TypeError, match="not a string"):
        build_feature_manifest("change_rate")  # type: ignore[arg-type]
