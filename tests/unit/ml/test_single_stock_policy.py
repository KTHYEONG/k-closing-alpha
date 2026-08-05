"""단일 종목 일일 매수 + 인과적 관망 정책 단위 테스트.

`docs/specs/ml_single_stock_abstention.md` 계약 검증:
- 매 거래일 정확히 하나의 BUY/ABSTAIN 결정.
- 관망 문턱·정책 선택은 이전 OOF 날짜만 사용(인과성).
- 진입 시퀀스 드로다운 명명 규칙(exit ledger 없이 NAV/MDD 금지).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from src.ml.single_stock_policy import (
    REASON_BELOW_MARGIN,
    REASON_INSUFFICIENT_CROSS_SECTION,
    REASON_INSUFFICIENT_HISTORY,
    REASON_NO_CANDIDATE,
    REASON_TOP1_BUY,
    SingleStockPolicy,
    abstain_decision,
    always_buy_policy,
    default_policy_candidates,
    evaluate_single_stock_policy_oof,
    margin_quantile_policy,
    select_single_daily_trade,
)

GROUP_COL = "trade_date"
STOCK_COL = "stock_code"
SCENARIO_COL = "chart_analysis"
SCORE_COL = "rank_score"
TARGET_COL = "target_return"


def _always_buy() -> SingleStockPolicy:
    return always_buy_policy("2024-12-31")


def _margin(q: float = 0.90) -> SingleStockPolicy:
    return margin_quantile_policy(q, "2024-12-31")


def _scored_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            GROUP_COL: ["2026-08-04"] * 3,
            STOCK_COL: ["000001", "000002", "000003"],
            SCENARIO_COL: ["거래량 폭증", "신고가", "신고가 근접"],
            SCORE_COL: [0.9, 0.5, 0.3],
            TARGET_COL: [0.01, 0.02, -0.01],
        }
    )


def _make_oof(n_groups: int, rows_per_group: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_groups, freq="D")
    scenarios = ["거래량 폭증", "신고가", "신고가 근접", "120 돌파"]
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend(
            {
                GROUP_COL: date,
                STOCK_COL: f"{i + 1:06d}",
                SCENARIO_COL: scenarios[i % len(scenarios)],
                "market_type": "KOSPI" if i % 2 == 0 else "KOSDAQ",
            }
            for i in range(rows_per_group)
        )
    df = pd.DataFrame(rows)
    df[SCORE_COL] = rng.normal(size=len(df))
    df[TARGET_COL] = 0.5 * df[SCORE_COL] + rng.normal(scale=0.05, size=len(df))
    return df


def test_always_buy_returns_one_buy_for_three_stock_panel() -> None:
    decision = select_single_daily_trade(
        _scored_panel(), _always_buy(), GROUP_COL, score_col=SCORE_COL
    )
    assert len(decision) == 1
    row = decision.iloc[0]
    assert row["decision"] == "BUY"
    assert row["decision_reason"] == REASON_TOP1_BUY
    assert row[STOCK_COL] == "000001"
    assert row[SCORE_COL] == pytest.approx(0.9)
    assert row["margin"] is None or np.isfinite(row["margin"])
    assert row["n_unique_stocks"] == 3


def test_no_candidate_panel_returns_one_abstain_record() -> None:
    decision = select_single_daily_trade(
        pd.DataFrame(), _always_buy(), GROUP_COL, score_col=SCORE_COL
    )
    assert len(decision) == 1
    assert decision.iloc[0]["decision"] == "ABSTAIN"
    assert decision.iloc[0]["decision_reason"] == REASON_NO_CANDIDATE


def test_duplicate_same_stock_scenarios_resolve_by_score_not_target() -> None:
    scored = pd.DataFrame(
        {
            GROUP_COL: ["2026-08-04"] * 3,
            STOCK_COL: ["000001", "000001", "000002"],
            SCENARIO_COL: ["상따", "120 돌파", "거래량 폭증"],
            SCORE_COL: [0.2, 0.9, 0.5],
            TARGET_COL: [0.9, -0.5, 0.1],
        }
    )
    decision = select_single_daily_trade(
        scored, _always_buy(), GROUP_COL, score_col=SCORE_COL
    )
    row = decision.iloc[0]
    assert row["decision"] == "BUY"
    assert row[STOCK_COL] == "000001"
    assert row[SCENARIO_COL] == "120 돌파"
    assert row["n_unique_stocks"] == 2


def test_tie_break_by_normalized_stock_code_ascending() -> None:
    scored = pd.DataFrame(
        {
            GROUP_COL: ["2026-08-04"] * 2,
            STOCK_COL: ["000002", "000001"],
            SCENARIO_COL: ["거래량 폭증", "신고가"],
            SCORE_COL: [0.5, 0.5],
            TARGET_COL: [0.1, 0.0],
        }
    )
    decision = select_single_daily_trade(
        scored, _always_buy(), GROUP_COL, score_col=SCORE_COL
    )
    assert decision.iloc[0][STOCK_COL] == "000001"


def test_invalid_identity_or_score_data_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing required"):
        select_single_daily_trade(
            _scored_panel().drop(columns=[STOCK_COL]), _always_buy(), GROUP_COL,
            score_col=SCORE_COL,
        )
    bad = _scored_panel().copy()
    bad.loc[0, STOCK_COL] = None
    with pytest.raises(ValueError, match="contain nulls"):
        select_single_daily_trade(bad, _always_buy(), GROUP_COL, score_col=SCORE_COL)
    bad_scores = _scored_panel().copy()
    bad_scores.loc[0, SCORE_COL] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        select_single_daily_trade(bad_scores, _always_buy(), GROUP_COL, score_col=SCORE_COL)


def test_invalid_policy_state_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid policy state"):
        select_single_daily_trade(_scored_panel(), None, GROUP_COL, score_col=SCORE_COL)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="margin_threshold"):
        select_single_daily_trade(_scored_panel(), _margin(), GROUP_COL, score_col=SCORE_COL)


def test_margin_abstains_below_threshold_and_buys_above() -> None:
    panel = _scored_panel()
    low = margin_quantile_policy(
        0.90, "2024-12-31", margin_threshold=10.0, reference_margin=(1.0,)
    )
    decision = select_single_daily_trade(panel, low, GROUP_COL, score_col=SCORE_COL)
    assert decision.iloc[0]["decision"] == "ABSTAIN"
    assert decision.iloc[0]["decision_reason"] == REASON_BELOW_MARGIN

    high = margin_quantile_policy(
        0.90, "2024-12-31", margin_threshold=0.0, reference_margin=(1.0,)
    )
    decision = select_single_daily_trade(panel, high, GROUP_COL, score_col=SCORE_COL)
    assert decision.iloc[0]["decision"] == "BUY"


def test_margin_single_stock_candidate_abstains_insufficient_cross_section() -> None:
    single = pd.DataFrame(
        {
            GROUP_COL: ["2026-08-04"] * 1,
            STOCK_COL: ["000001"],
            SCENARIO_COL: ["거래량 폭증"],
            SCORE_COL: [0.9],
            TARGET_COL: [0.05],
        }
    )
    policy = margin_quantile_policy(
        0.90, "2024-12-31", margin_threshold=0.5, reference_margin=(1.0,)
    )
    decision = select_single_daily_trade(single, policy, GROUP_COL, score_col=SCORE_COL)
    assert decision.iloc[0]["decision"] == "ABSTAIN"
    assert decision.iloc[0]["decision_reason"] == REASON_INSUFFICIENT_CROSS_SECTION


def test_always_buy_single_stock_still_buys() -> None:
    single = pd.DataFrame(
        {
            GROUP_COL: ["2026-08-04"] * 1,
            STOCK_COL: ["000001"],
            SCENARIO_COL: ["거래량 폭증"],
            SCORE_COL: [0.9],
            TARGET_COL: [0.05],
        }
    )
    decision = select_single_daily_trade(single, _always_buy(), GROUP_COL, score_col=SCORE_COL)
    assert decision.iloc[0]["decision"] == "BUY"


def test_abstain_decision_helper_emits_single_record() -> None:
    decision = abstain_decision(
        "missing_validated_policy", group_col=GROUP_COL, group_value="2026-08-04"
    )
    assert len(decision) == 1
    assert decision.iloc[0]["decision"] == "ABSTAIN"
    assert decision.iloc[0][GROUP_COL] == "2026-08-04"
    assert decision.iloc[0]["policy_id"] == "missing_validated_policy"


def test_single_stock_policy_is_immutable() -> None:
    policy = _always_buy()
    with pytest.raises(pydantic.ValidationError):
        policy.calibration_cutoff = "2025-01-01"  # type: ignore[misc]
    assert policy.calibration_cutoff == "2024-12-31"
    updated = policy.model_copy(update={"calibration_cutoff": "2025-01-01"})
    assert updated.calibration_cutoff == "2025-01-01"
    assert policy.calibration_cutoff == "2024-12-31"


def test_default_policy_candidates_include_always_buy_and_grid() -> None:
    candidates = default_policy_candidates("2024-12-31")
    assert [c.candidate for c in candidates] == [
        "always_buy_top1",
        "margin_quantile.0.70",
        "margin_quantile.0.90",
    ]


def test_evaluate_single_stock_policy_oof_returns_contract_shape() -> None:
    oof = _make_oof(n_groups=30, rows_per_group=4, seed=11)
    evaluation = evaluate_single_stock_policy_oof(
        oof,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        stock_col=STOCK_COL,
        policy_candidates=default_policy_candidates("2024-12-31", score_col=SCORE_COL),
        min_history_dates=5,
        score_col=SCORE_COL,
    )
    assert isinstance(evaluation.selected_policy, SingleStockPolicy)
    assert evaluation.selected_policy.candidate in {
        "always_buy_top1",
        "margin_quantile.0.70",
        "margin_quantile.0.90",
    }
    assert len(evaluation.decisions) == 30
    assert evaluation.decisions["decision"].isin(["BUY", "ABSTAIN"]).all()
    assert evaluation.scheduled_returns.shape == (30,)
    for key in (
        "buy_rate",
        "abstain_rate",
        "reason_counts",
        "scheduled_mean_return",
        "scheduled_sharpe",
        "profit_factor",
        "active_trade_mean_return",
        "active_trade_win_rate",
        "turnover",
        "entry_sequence_drawdown",
    ):
        assert key in evaluation.metrics
    assert "entry_sequence_drawdown" in evaluation.metrics
    assert "max_drawdown" not in evaluation.metrics
    assert "nav" not in evaluation.metrics
    assert evaluation.yearly_breakdown
    assert evaluation.market_type_breakdown
    assert evaluation.candidate_results


def test_evaluate_warmup_dates_are_retained_explicit_abstentions() -> None:
    oof = _make_oof(n_groups=10, rows_per_group=3, seed=3)
    evaluation = evaluate_single_stock_policy_oof(
        oof,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        stock_col=STOCK_COL,
        policy_candidates=default_policy_candidates("2024-12-31", score_col=SCORE_COL),
        min_history_dates=4,
        score_col=SCORE_COL,
    )
    reasons = evaluation.decisions["decision_reason"].to_numpy()
    assert list(reasons[:4]) == [REASON_INSUFFICIENT_HISTORY] * 4
    assert evaluation.metrics["n_scheduled_dates"] == 10
    assert evaluation.metrics["reason_counts"].get(REASON_INSUFFICIENT_HISTORY, 0) == 4


def test_future_only_profitable_gate_cannot_affect_prior_decisions() -> None:
    oof = _make_oof(n_groups=8, rows_per_group=3, seed=5)
    last_date = oof[GROUP_COL].max()
    candidates = (margin_quantile_policy(0.9, "2024-12-31"),)
    baseline = evaluate_single_stock_policy_oof(
        oof,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        stock_col=STOCK_COL,
        policy_candidates=candidates,
        min_history_dates=2,
        score_col=SCORE_COL,
    )
    future = oof.copy()
    future.loc[future[GROUP_COL] == last_date, TARGET_COL] = 10.0
    gated = evaluate_single_stock_policy_oof(
        future,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        stock_col=STOCK_COL,
        policy_candidates=candidates,
        min_history_dates=2,
        score_col=SCORE_COL,
    )
    prior = baseline.decisions[baseline.decisions[GROUP_COL] != last_date].reset_index(drop=True)
    prior_future = gated.decisions[gated.decisions[GROUP_COL] != last_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        prior[["decision", "decision_reason", STOCK_COL, SCORE_COL, "margin"]],
        prior_future[["decision", "decision_reason", STOCK_COL, SCORE_COL, "margin"]],
    )


def test_margin_candidate_better_active_but_worse_objective_cannot_win() -> None:
    """미래에만 이익이 있는 게이트가 이전 결정을 바꾸지 못하며, active 통계가
    우수해도 scheduled-date 목적함수가 낮은 후보는 always_buy 를 대체하지 못합니다."""
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    scores: list[float] = []
    for i, _date in enumerate(dates):
        if i == 0 or i == 7:
            scores.extend([10.0, 0.0, 0.0])
        else:
            scores.extend([3.0, 2.0, 1.0])
    oof = pd.DataFrame(
        {
            GROUP_COL: sorted(list(dates) * 3),
            STOCK_COL: [f"{s:06d}" for _ in range(8) for s in (1, 2, 3)],
            SCENARIO_COL: ["거래량 폭증", "신고가", "신고가 근접"] * 8,
            SCORE_COL: scores,
        }
    )
    oof[TARGET_COL] = [0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.06, 0.0, 0.0, 0.06, 0.0, 0.0,
                       0.06, 0.0, 0.0, 0.06, 0.0, 0.0, 0.06, 0.0, 0.0, 1.5, 0.0, 0.0]

    evaluation = evaluate_single_stock_policy_oof(
        oof,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        stock_col=STOCK_COL,
        policy_candidates=(always_buy_policy("2024-12-31"), margin_quantile_policy(0.9, "2024-12-31")),
        min_history_dates=1,
        score_col=SCORE_COL,
    )
    assert evaluation.selected_policy.candidate == "always_buy_top1"
    margin_stats = evaluation.candidate_results["margin_quantile.0.90"]
    always_stats = evaluation.candidate_results["always_buy_top1"]
    assert margin_stats["active_trade_mean_return"] > always_stats["active_trade_mean_return"]
    assert margin_stats["scheduled_mean_return"] < always_stats["scheduled_mean_return"]


def test_evaluate_single_stock_oof_fails_closed() -> None:
    oof = _make_oof(n_groups=8, rows_per_group=3, seed=9)
    with pytest.raises(ValueError, match="min_history_dates"):
        evaluate_single_stock_policy_oof(
            oof, TARGET_COL, GROUP_COL, STOCK_COL,
            default_policy_candidates("2024-12-31"), 0, score_col=SCORE_COL,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_single_stock_policy_oof(
            oof, TARGET_COL, GROUP_COL, STOCK_COL, (), 5, score_col=SCORE_COL,
        )
    with pytest.raises(ValueError, match="missing required"):
        evaluate_single_stock_policy_oof(
            oof.drop(columns=[TARGET_COL]), TARGET_COL, GROUP_COL, STOCK_COL,
            default_policy_candidates("2024-12-31"), 5, score_col=SCORE_COL,
        )
    bad_target = oof.copy()
    bad_target.loc[0, TARGET_COL] = np.nan
    with pytest.raises(ValueError, match="contain nulls"):
        evaluate_single_stock_policy_oof(
            bad_target, TARGET_COL, GROUP_COL, STOCK_COL,
            default_policy_candidates("2024-12-31"), 5, score_col=SCORE_COL,
        )
    bad_chrono = _make_oof(n_groups=3, rows_per_group=2, seed=2)
    bad_chrono[GROUP_COL] = bad_chrono[GROUP_COL].astype(str)
    bad_chrono.loc[0, GROUP_COL] = "not-a-date"
    with pytest.raises(ValueError, match="chronology"):
        evaluate_single_stock_policy_oof(
            bad_chrono, TARGET_COL, GROUP_COL, STOCK_COL,
            default_policy_candidates("2024-12-31"), 5, score_col=SCORE_COL,
        )


def test_margin_selected_policy_persists_reference_distribution() -> None:
    oof = _make_oof(n_groups=20, rows_per_group=4, seed=21)
    evaluation = evaluate_single_stock_policy_oof(
        oof,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        stock_col=STOCK_COL,
        policy_candidates=default_policy_candidates("2024-12-31", score_col=SCORE_COL),
        min_history_dates=3,
        score_col=SCORE_COL,
    )
    selected = evaluation.selected_policy
    if selected.candidate.startswith("margin_quantile."):
        assert selected.margin_threshold is not None
        assert len(selected.reference_margin) > 0
        assert selected.history_length == len(evaluation.decisions)
    else:
        assert selected.margin_threshold is None
