"""Persisted policy restoration and BUY/ABSTAIN one-decision behavior tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.serving.realtime.policy import (
    REASON_MISSING_POLICY,
    SingleStockPolicy,
    abstain_decision,
    always_buy_policy,
    load_single_stock_policy,
    margin_quantile_policy,
    resolve_stock_actions,
    select_single_daily_trade,
)


def _scored_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-08-04"] * 3,
            "stock_code": ["000001", "000002", "000003"],
            "chart_analysis": ["거래량 폭증", "신고가", "상따"],
            "rank_score": [0.9, 0.4, 0.7],
        }
    )


def test_load_single_stock_policy_from_bundle_state() -> None:
    policy = always_buy_policy("2026-08-04")
    assert load_single_stock_policy({"single_stock_policy": policy}) is policy

    restored = load_single_stock_policy({"single_stock_policy": policy.model_dump()})
    assert isinstance(restored, SingleStockPolicy)
    assert restored.policy_id == "always_buy_top1"

    assert load_single_stock_policy({"feature_cols": ["f1"]}) is None
    assert load_single_stock_policy({"single_stock_policy": "bogus"}) is None


def test_abstain_decision_missing_policy_is_single_row() -> None:
    decision = abstain_decision(
        REASON_MISSING_POLICY, group_value="2026-08-04"
    )
    assert len(decision) == 1
    assert decision.iloc[0]["decision"] == "ABSTAIN"
    assert decision.iloc[0]["decision_reason"] == "missing_validated_policy"


def test_select_single_daily_trade_returns_one_buy() -> None:
    decision = select_single_daily_trade(
        _scored_panel(), always_buy_policy("2026-08-04"), group_col="date"
    )
    assert len(decision) == 1
    assert decision.iloc[0]["decision"] == "BUY"
    assert decision.iloc[0]["stock_code"] == "000001"
    assert decision.iloc[0]["n_unique_stocks"] == 3


def test_select_single_daily_trade_resolves_multi_scenario() -> None:
    panel = _scored_panel()
    multi = pd.concat([panel, _scored_panel()], ignore_index=True)
    multi.loc[3, "chart_analysis"] = "상따"
    multi.loc[3, "rank_score"] = 0.99
    decision = select_single_daily_trade(
        multi,
        always_buy_policy("2026-08-04"),
        group_col="date",
        score_col="rank_score",
        resolve_mode="score_best_action",
    )
    assert len(decision) == 1
    assert decision.iloc[0]["decision"] == "BUY"


def test_select_single_daily_trade_margin_policy_requires_threshold() -> None:
    invalid = margin_quantile_policy(0.7, "2026-08-04")
    with pytest.raises(ValueError, match="requires margin_threshold"):
        select_single_daily_trade(_scored_panel(), invalid, group_col="date")


def test_resolve_stock_actions_score_best_action_is_deterministic() -> None:
    resolved = resolve_stock_actions(
        _scored_panel(), "date", score_col="rank_score", mode="score_best_action"
    )
    again = resolve_stock_actions(
        _scored_panel(), "date", score_col="rank_score", mode="score_best_action"
    )
    pd.testing.assert_frame_equal(resolved, again)


def test_select_single_daily_trade_empty_panel_abstains() -> None:
    decision = select_single_daily_trade(
        pd.DataFrame(),
        always_buy_policy("2026-08-04"),
        group_col="date",
    )
    assert len(decision) == 1
    assert decision.iloc[0]["decision"] == "ABSTAIN"
    assert decision.iloc[0]["decision_reason"] == "no_executable_candidate"
