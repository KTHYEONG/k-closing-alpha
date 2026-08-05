"""시나리오 행동 패널 정규화 단위 테스트.

`docs/specs/scenario_action_panel_resolution.md` 의 필수 테스트 항목과
`scenario_action_panel_resolution_contract.json` 의 planned scenarios
(SCENARIO_ACTION_KEEP_DISTINCT_SCENARIOS,
SCENARIO_ACTION_REJECT_CONFLICTING_DUPLICATES) 를 검증합니다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.scenario_action_panel import (
    SCENARIO_CONTEXT_FEATURES,
    SCENARIO_ONE_HOT_FEATURES,
    build_scenario_action_panel,
)


def _make_panel_df() -> pd.DataFrame:
    """서로 다른 시나리오의 같은 날짜-종목 행과 단일 행동 종목을 포함한 DataFrame."""
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "stock_code": ["000001", "000001", "000002", "000001", "000002"],
            "chart_analysis": ["상따", "120 돌파", "거래량 폭증", "신고가", "신고가 근접"],
            "buy_price": [100.0, 100.0, 200.0, 300.0, 400.0],
            "sell_price": [105.0, 107.5, 205.0, 300.0, 390.0],
            "net_return": [-1.14, 7.54, 1.0, 0.0, -2.5],
        }
    )


def test_scenario_action_keeps_distinct_scenarios_for_same_date_stock() -> None:
    """서로 다른 시나리오의 같은 날짜-종목은 행동 패널에 모두 보존됩니다."""
    df = _make_panel_df()
    panel, rejects = build_scenario_action_panel(df)

    assert len(rejects) == 0
    assert len(panel) == len(df)
    pair = panel.loc[
        (panel["trade_date"] == pd.Timestamp("2024-01-02")) & (panel["stock_code"] == "000001")
    ]
    assert set(pair["chart_analysis"]) == {"상따", "120 돌파"}


def test_scenario_action_adds_fixed_one_hot_features() -> None:
    """허용된 행동 행에 고정 시나리오 one-hot/context 수치 피처가 추가됩니다."""
    panel, _ = build_scenario_action_panel(_make_panel_df())

    expected = set(SCENARIO_ONE_HOT_FEATURES) | set(SCENARIO_CONTEXT_FEATURES)
    assert expected.issubset(panel.columns)
    sangtta = panel.loc[panel["chart_analysis"] == "상따"].iloc[0]
    assert sangtta["scenario_is_sangtta"] == 1
    assert sangtta["scenario_is_120_breakout"] == 0
    breakout = panel.loc[panel["chart_analysis"] == "120 돌파"].iloc[0]
    assert breakout["scenario_is_120_breakout"] == 1


def test_scenario_action_context_features_for_multi_scenario_stock_date() -> None:
    """상따와 일반 시나리오가 함께 있으면 has_sangtta/is_multi_scenario 가 1 입니다."""
    panel, _ = build_scenario_action_panel(_make_panel_df())

    mask = (panel["trade_date"] == pd.Timestamp("2024-01-02")) & (
        panel["stock_code"] == "000001"
    )
    group = panel.loc[mask]
    assert (group["scenario_count_for_stock_date"] == 2).all()
    assert (group["is_multi_scenario_stock_date"] == 1).all()
    assert (group["has_sangtta_for_stock_date"] == 1).all()

    single = panel.loc[(panel["trade_date"] == pd.Timestamp("2024-01-02"))]
    assert (single.loc[single["stock_code"] == "000002", "scenario_count_for_stock_date"] == 1).all()
    assert (single.loc[single["stock_code"] == "000002", "is_multi_scenario_stock_date"] == 0).all()


def test_scenario_action_collapses_byte_identical_duplicate_keys() -> None:
    """같은 행동 key 의 완전 동일 행은 한 행으로 축소됩니다."""
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "stock_code": ["000001", "000001"],
            "chart_analysis": ["거래량 폭증", "거래량 폭증"],
            "buy_price": [200.0, 200.0],
            "sell_price": [205.0, 205.0],
            "net_return": [1.0, 1.0],
        }
    )
    panel, rejects = build_scenario_action_panel(df)
    assert len(panel) == 1
    assert len(rejects) == 0


def test_scenario_action_rejects_conflicting_duplicate_keys() -> None:
    """같은 행동 key 의 실행/피처 충돌은 reject table 로 이동하고 패널에 남지 않습니다."""
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "stock_code": ["000001", "000001", "000002", "000002"],
            "chart_analysis": ["상따", "상따", "거래량 폭증", "거래량 폭증"],
            "buy_price": [100.0, 100.0, 200.0, 200.0],
            "sell_price": [105.0, 108.0, 205.0, 205.0],
            "net_return": [-1.14, -1.14, 1.0, 1.0],
        }
    )
    panel, rejects = build_scenario_action_panel(df)

    assert len(rejects) == 2
    assert (rejects["reject_reason"] == "conflicting_duplicate_action").all()
    assert set(rejects["chart_analysis"]) == {"상따"}
    # 충돌 행은 학습/정책 패널에 들어가지 않습니다.
    assert len(panel) == 1
    assert set(panel["chart_analysis"]) == {"거래량 폭증"}


def test_scenario_action_reject_is_deterministic_no_first_last_mean() -> None:
    """충돌 시 임의의 first/last/평균 선택 없이 key 전체가 reject 됩니다."""
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "stock_code": ["000001", "000001", "000001"],
            "chart_analysis": ["상따", "상따", "상따"],
            "buy_price": [100.0, 100.0, 100.0],
            "sell_price": [105.0, 108.0, 107.0],
            "net_return": [-1.14, 5.0, 2.0],
        }
    )
    panel, rejects = build_scenario_action_panel(df)
    assert len(panel) == 0
    assert len(rejects) == 3
    assert (rejects["reject_reason"] == "conflicting_duplicate_action").all()


def test_scenario_action_unknown_scenario_goes_to_other() -> None:
    """알 수 없는 시나리오는 scenario_other=1 로 보내집니다."""
    df = _make_panel_df()
    df.loc[df.index[4], "chart_analysis"] = "미정의 시나리오"
    panel, _ = build_scenario_action_panel(df)
    row = panel.loc[panel["chart_analysis"] == "미정의 시나리오"].iloc[0]
    assert row["scenario_other"] == 1
    assert row["scenario_is_sangtta"] == 0
    assert row["scenario_is_120_breakout"] == 0


def test_scenario_action_rejects_missing_key_column() -> None:
    df = _make_panel_df().drop(columns=["stock_code"])
    with pytest.raises(ValueError, match="missing required key columns"):
        build_scenario_action_panel(df)


def test_scenario_action_rejects_null_key_values() -> None:
    df = _make_panel_df()
    df.loc[df.index[0], "chart_analysis"] = None
    with pytest.raises(ValueError, match="contain nulls"):
        build_scenario_action_panel(df)
