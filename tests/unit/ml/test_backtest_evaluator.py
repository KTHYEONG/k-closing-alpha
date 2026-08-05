"""Backtest Evaluation (Baseline 비교 & 연도별 안정성 감사) 단위 테스트.

SCENARIO_BACKTEST_BASELINE_COMPARISON
SCENARIO_BACKTEST_YEARLY_BREAKDOWN
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest_evaluator import (
    resolve_stock_actions,
    run_backtest_evaluation,
    simulate_top_k_policy,
)

GROUP_COL = "trade_date"
TARGET_COL = "net_return"

METRIC_KEYS = (
    "top_1_return",
    "top_3_return",
    "win_rate",
    "profit_factor",
    "mean_win",
    "mean_loss",
    "sharpe",
    "cost_adjusted_return",
    "date_weighted_return",
    "capital_weighted_return",
    "turnover",
    "max_drawdown",
)


def _make_oof(
    n_groups: int = 12,
    rows_per_group: int = 6,
    seed: int = 7,
    include_selection_rank: bool = True,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_groups, freq="D")
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({GROUP_COL: date} for _ in range(rows_per_group))
    df = pd.DataFrame(rows)
    df["pred"] = rng.normal(size=len(df))
    df[TARGET_COL] = 0.5 * df["pred"] + rng.normal(scale=0.05, size=len(df))
    if include_selection_rank:
        df["selection_rank"] = (
            df.groupby(GROUP_COL, sort=False)["pred"].rank(method="first", ascending=True).astype(int)
        )
    return df


def test_run_backtest_evaluation_returns_contract_shape() -> None:
    oof = _make_oof()
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    assert "model_metrics" in result
    assert "baseline_metrics" in result
    assert "yearly_breakdown" in result
    assert "regime_breakdown" in result

    model = result["model_metrics"]
    assert set(model) == set(METRIC_KEYS)
    assert result["baseline_metrics"]["selection_rank"] is not None
    assert result["baseline_metrics"]["equal_weight"] is not None
    assert set(result["baseline_metrics"]["selection_rank"]) == set(METRIC_KEYS)
    assert set(result["baseline_metrics"]["equal_weight"]) == set(METRIC_KEYS)


def test_run_backtest_evaluation_fails_closed_when_selection_rank_missing() -> None:
    """selection_rank baseline 은 필수입니다. 누락 시 조용히 생략하지 않고 ValueError 로 실패합니다."""
    oof = _make_oof(include_selection_rank=False)
    with pytest.raises(ValueError, match="selection_rank"):
        run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)


def test_run_backtest_evaluation_yearly_breakdown() -> None:
    oof = _make_oof(n_groups=15, start_date="2023-12-27")
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    yearly = result["yearly_breakdown"]
    assert set(yearly) == {2023, 2024}
    for entry in yearly.values():
        assert entry is not None
        for key in ("top1_return", "top3_return", "win_rate", "profit_factor", "sharpe"):
            assert key in entry
        assert np.isfinite(entry["top1_return"])


def test_run_backtest_evaluation_yearly_small_sample_is_null() -> None:
    oof = _make_oof(n_groups=9, start_date="2023-12-29")
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    assert result["yearly_breakdown"][2023] is None
    assert result["yearly_breakdown"][2024] is not None


def test_run_backtest_evaluation_profit_factor_inf_when_no_loss() -> None:
    oof = _make_oof(n_groups=8, rows_per_group=4)
    oof[TARGET_COL] = 1.0 + oof["pred"] * 0.01
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    assert np.isinf(result["model_metrics"]["profit_factor"])


def test_run_backtest_evaluation_model_beats_baselines() -> None:
    oof = _make_oof(n_groups=30, rows_per_group=6, seed=3)
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    model = result["model_metrics"]
    selection_rank = result["baseline_metrics"]["selection_rank"]
    equal_weight = result["baseline_metrics"]["equal_weight"]
    assert model["top_1_return"] > selection_rank["top_1_return"]
    assert model["top_1_return"] > equal_weight["top_1_return"]
    assert model["win_rate"] > selection_rank["win_rate"]


def test_run_backtest_evaluation_selection_rank_aligned_equals_model() -> None:
    oof = _make_oof(n_groups=10, rows_per_group=4, seed=5)
    oof["selection_rank"] = (
        oof.groupby(GROUP_COL, sort=False)["pred"].rank(method="first", ascending=False).astype(int)
    )
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    model = result["model_metrics"]
    selection_rank = result["baseline_metrics"]["selection_rank"]
    assert selection_rank["top_1_return"] == pytest.approx(model["top_1_return"])
    assert selection_rank["top_3_return"] == pytest.approx(model["top_3_return"])


def test_run_backtest_evaluation_equal_weight_is_daily_group_mean() -> None:
    oof = _make_oof(n_groups=10, rows_per_group=5, seed=9)
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    expected = oof.groupby(GROUP_COL)[TARGET_COL].mean().mean()
    assert result["baseline_metrics"]["equal_weight"]["top_1_return"] == pytest.approx(expected)


def test_run_backtest_evaluation_rejects_missing_required_column() -> None:
    oof = _make_oof().drop(columns=["pred"])
    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)


def _make_policy_oof(
    n_groups: int = 6,
    rows_per_group: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_groups, freq="D")
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({GROUP_COL: date} for _ in range(rows_per_group))
    df = pd.DataFrame(rows)
    df["stock_code"] = [f"{i % rows_per_group:04d}" for i in range(len(df))]
    df["pred"] = rng.normal(size=len(df))
    df[TARGET_COL] = 0.5 * df["pred"] + rng.normal(scale=0.05, size=len(df))
    return df


def test_simulate_top_k_policy_returns_contract_shape() -> None:
    oof = _make_policy_oof()
    result = simulate_top_k_policy(oof, TARGET_COL, GROUP_COL, top_k=2)
    assert result["top_k"] == 2
    assert isinstance(result["daily_returns"], np.ndarray)
    assert len(result["daily_returns"]) == 6
    assert len(result["nav"]) == len(result["daily_returns"])
    assert np.allclose(result["nav"][-1], np.prod(1.0 + result["daily_returns"]))
    for key in ("top_1_return", "win_rate", "profit_factor", "sharpe"):
        assert key in result["metrics"]
    assert "max_drawdown" in result["metrics"]
    assert "yearly_breakdown" in result
    assert np.isfinite(result["turnover"])


def test_simulate_top_k_policy_uses_stock_codes_for_turnover() -> None:
    """turnover 는 DataFrame index 가 아닌 선택된 stock_code 로 계산됩니다.

    pred 를 종목코드 숫자값에 고정하면 매일 동일 종목(코드 0003/0002)이
    재선택되므로 turnover 가 0 이어야 합니다 (index 는 날짜별로 동일).
    """
    oof = _make_policy_oof(n_groups=4, rows_per_group=4)
    oof["pred"] = oof["stock_code"].astype(float)
    result = simulate_top_k_policy(oof, TARGET_COL, GROUP_COL, top_k=2)
    assert result["turnover"] == 0.0


def test_simulate_top_k_policy_target_is_decimal_net_single_cost() -> None:
    """target_col 은 decimal net return 이므로 비용을 재차감하지 않습니다."""
    oof = _make_policy_oof(n_groups=3, rows_per_group=4, seed=11)
    result = simulate_top_k_policy(oof, TARGET_COL, GROUP_COL, top_k=1)
    actual = result["daily_returns"]
    sorted_df = oof.sort_values([GROUP_COL, "pred"], ascending=[True, False])
    expected = sorted_df.groupby(GROUP_COL)[TARGET_COL].head(1).to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    # 등가중 + 비용 미차감: 순수 target 의 합/선택수 와 일치합니다.
    np.testing.assert_allclose(
        result["nav"][-1], np.prod(1.0 + expected), atol=1e-12
    )


def test_simulate_top_k_policy_rejects_top_k_below_one() -> None:
    oof = _make_policy_oof()
    with pytest.raises(ValueError, match="top_k"):
        simulate_top_k_policy(oof, TARGET_COL, GROUP_COL, top_k=0)


def test_simulate_top_k_policy_rejects_missing_columns() -> None:
    oof = _make_policy_oof().drop(columns=["pred"])
    with pytest.raises(ValueError, match="missing required columns"):
        simulate_top_k_policy(oof, TARGET_COL, GROUP_COL)


def test_simulate_top_k_policy_rejects_duplicate_stock_codes_within_date() -> None:
    oof = _make_policy_oof(n_groups=2, rows_per_group=3)
    oof.loc[1, "stock_code"] = oof.loc[0, "stock_code"]
    with pytest.raises(ValueError, match="duplicate stock_code"):
        simulate_top_k_policy(oof, TARGET_COL, GROUP_COL)


def test_simulate_top_k_policy_rejects_missing_stock_codes() -> None:
    oof = _make_policy_oof()
    oof.loc[0, "stock_code"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        simulate_top_k_policy(oof, TARGET_COL, GROUP_COL)


def test_simulate_top_k_policy_rejects_non_finite_selected_returns() -> None:
    oof = _make_policy_oof(n_groups=2, rows_per_group=3)
    oof.loc[oof[GROUP_COL] == oof[GROUP_COL].iloc[0], TARGET_COL] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        simulate_top_k_policy(oof, TARGET_COL, GROUP_COL)


def test_simulate_top_k_policy_rejects_empty_panel_after_nan_filter() -> None:
    oof = _make_policy_oof(n_groups=2, rows_per_group=3)
    oof[TARGET_COL] = np.nan
    with pytest.raises(ValueError, match="no usable rows"):
        simulate_top_k_policy(oof, TARGET_COL, GROUP_COL)


def _make_action_oof() -> pd.DataFrame:
    """날짜-종목별로 여러 시나리오 행동을 가진 OOF."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            GROUP_COL: [dates[0], dates[0], dates[0], dates[1], dates[1]],
            "stock_code": ["000001", "000001", "000002", "000001", "000002"],
            "chart_analysis": ["상따", "120 돌파", "거래량 폭증", "신고가", "신고가 근접"],
            "pred": [0.1, 0.2, 0.3, 0.15, -0.05],
            TARGET_COL: [-0.0114, 0.0754, 0.01, 0.0, -0.025],
        }
    )


def test_resolve_stock_actions_exclude_multi_scenario() -> None:
    """exclude_multi_scenario 는 다중 행동 날짜-종목을 포트폴리오에서 제외합니다."""
    resolved = resolve_stock_actions(_make_action_oof(), GROUP_COL)
    keys = set(zip(resolved[GROUP_COL], resolved["stock_code"], strict=False))
    assert keys == {
        (pd.Timestamp("2024-01-02"), "000002"),
        (pd.Timestamp("2024-01-03"), "000001"),
        (pd.Timestamp("2024-01-03"), "000002"),
    }
    assert not resolved.duplicated(subset=[GROUP_COL, "stock_code"]).any()


def test_resolve_stock_actions_score_best_action() -> None:
    """score_best_action 은 예측 점수로 한 행동만 고르며 실현 수익률을 보지 않습니다."""
    oof = _make_action_oof()
    # 120 돌파 는 상따 보다 수익률이 높지만 점수가 낮게 설정 -> 점수 기준으로만 선택.
    oof.loc[oof["chart_analysis"] == "120 돌파", "pred"] = 0.05
    oof.loc[oof["chart_analysis"] == "상따", "pred"] = 0.9
    resolved = resolve_stock_actions(oof, GROUP_COL, mode="score_best_action")

    pick = resolved.loc[
        (resolved[GROUP_COL] == pd.Timestamp("2024-01-02")) & (resolved["stock_code"] == "000001")
    ]
    assert list(pick["chart_analysis"]) == ["상따"]
    assert not resolved.duplicated(subset=[GROUP_COL, "stock_code"]).any()


def test_resolve_stock_actions_score_best_tie_break_deterministic() -> None:
    """동점은 시나리오명 오름차순으로 결정적 처리됩니다."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-02"])
    df = pd.DataFrame(
        {
            GROUP_COL: [dates[0], dates[0]],
            "stock_code": ["000001", "000001"],
            "chart_analysis": ["거래량 폭증", "상따"],
            "pred": [0.15, 0.15],
            TARGET_COL: [0.01, 0.03],
        }
    )
    resolved = resolve_stock_actions(df, GROUP_COL, mode="score_best_action")
    assert list(resolved["chart_analysis"]) == ["거래량 폭증"]
    # 같은 입력에 대해 반복 호출해도 동일 결과 (결정성).
    again = resolve_stock_actions(df, GROUP_COL, mode="score_best_action")
    assert list(again["chart_analysis"]) == ["거래량 폭증"]


def test_resolve_stock_actions_require_final_action_selects_executable() -> None:
    """require_final_action 은 실행 행동이 정확히 하나인 날짜-종목만 선택합니다."""
    oof = _make_action_oof()
    oof["is_executable_action"] = oof["chart_analysis"] != "120 돌파"
    resolved = resolve_stock_actions(oof, GROUP_COL, mode="require_final_action")
    assert len(resolved) == 4
    assert set(resolved["chart_analysis"]) == {"상따", "거래량 폭증", "신고가", "신고가 근접"}
    assert not resolved.duplicated(subset=[GROUP_COL, "stock_code"]).any()


def test_resolve_stock_actions_require_final_action_fails_without_executable() -> None:
    """실행 행동이 없는 날짜-종목이 있으면 ValueError 로 실패합니다."""
    oof = _make_action_oof()
    oof["is_executable_action"] = False
    with pytest.raises(ValueError, match="exactly one executable action"):
        resolve_stock_actions(oof, GROUP_COL, mode="require_final_action")


def test_resolve_stock_actions_require_final_action_fails_with_multiple_executable() -> None:
    """실행 행동이 둘 이상인 날짜-종목이 있으면 ValueError 로 실패합니다."""
    oof = _make_action_oof()
    oof["is_executable_action"] = True
    with pytest.raises(ValueError, match="exactly one executable action"):
        resolve_stock_actions(oof, GROUP_COL, mode="require_final_action")


def test_resolve_stock_actions_require_final_action_missing_executable_col() -> None:
    """require_final_action 은 is_executable_action 컬럼이 없으면 ValueError 로 실패합니다."""
    oof = _make_action_oof()
    with pytest.raises(ValueError, match="is_executable_action"):
        resolve_stock_actions(oof, GROUP_COL, mode="require_final_action")


def test_resolve_stock_actions_resolution_is_injective_on_date_stock_keys() -> None:
    """세 해소 모드 모두 날짜-종목 key 가 유일한 패널을 반환합니다."""
    oof = _make_action_oof()
    for mode in ("exclude_multi_scenario", "score_best_action", "require_final_action"):
        frame = oof.copy()
        if mode == "require_final_action":
            frame["is_executable_action"] = frame["chart_analysis"] != "120 돌파"
        resolved = resolve_stock_actions(frame, GROUP_COL, mode=mode)
        assert not resolved.duplicated(subset=[GROUP_COL, "stock_code"]).any()


def test_resolve_stock_actions_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        resolve_stock_actions(_make_action_oof(), GROUP_COL, mode="first_row")


def test_resolve_stock_actions_rejects_missing_keys() -> None:
    oof = _make_action_oof().drop(columns=["chart_analysis"])
    with pytest.raises(ValueError, match="missing required columns"):
        resolve_stock_actions(oof, GROUP_COL)


def test_resolve_stock_actions_rejects_null_keys() -> None:
    oof = _make_action_oof()
    oof.loc[0, "stock_code"] = None
    with pytest.raises(ValueError, match="contain nulls"):
        resolve_stock_actions(oof, GROUP_COL)


def test_resolve_stock_actions_feeds_simulate_top_k_policy() -> None:
    """해소된 유일 종목 패널은 simulate_top_k_policy 가 중복을 거부하지 않습니다."""
    oof = _make_action_oof()
    oof["stock_code"] = ["000001", "000001", "000002", "000003", "000004"]
    resolved = resolve_stock_actions(oof, GROUP_COL, mode="score_best_action")
    result = simulate_top_k_policy(resolved, TARGET_COL, GROUP_COL, top_k=1)
    assert len(result["daily_returns"]) == 2
