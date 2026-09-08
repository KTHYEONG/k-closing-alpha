"""Comprehensive verification test suite for K-Closing Alpha Research Validation v3.

Verifies zero leakage, strict walk-forward isolation, calendar-aware discrete NAV,
suspension carry rules, dynamic gate logic, and metrics JSON-to-Markdown consistency.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.research.v3_engine import (
    FEATURE_COLS,
    attach_forward_exit_paths,
    build_candidate_universe,
    compute_derived_features,
    load_and_prepare_price_history,
)
from src.ml.research.v3_metrics import (
    calculate_geometric_cagr,
    calculate_series_metrics,
    deflated_sharpe_ratio,
    moving_block_bootstrap_ci,
    simulate_discrete_portfolio,
)


# research v3 산출물은 sync 단계에서 purge 되는 임시 파일이다(commit a88d292).
# 아티팩트 의존 테스트는 파일이 있을 때만 실행한다.
_V3_METRICS_PATH = Path("docs/research/v3/research_validation_v3_metrics.json")
_V3_REPORT_PATH = Path("docs/research/v3/research_validation_v3.md")
_v3_artifacts_absent = pytest.mark.skipif(
    not _V3_METRICS_PATH.exists(),
    reason="research v3 산출물이 purge됨 — 아티팩트 복원 시에만 검증",
)


@pytest.fixture
def metrics_data() -> dict:
    if not _V3_METRICS_PATH.exists():
        pytest.skip("research_validation_v3_metrics.json purged")
    with open(_V3_METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def report_text() -> str:
    if not _V3_REPORT_PATH.exists():
        pytest.skip("research_validation_v3.md purged")
    return _V3_REPORT_PATH.read_text(encoding="utf-8")


def test_report_numbers_match_metrics_json(metrics_data: dict, report_text: str):
    """Test Section 68 & 78: Markdown report figures match metrics JSON exactly."""
    p1 = metrics_data["pipelines"]["P1"]
    r = metrics_data["ranking_evaluation"]

    # Check Top1 Net
    top1_net_str = f"{p1['net_bp']:.2f}"
    assert top1_net_str in report_text, f"Top1 net {top1_net_str} missing in report"

    # Check Rank IC
    rank_ic_str = f"{r['mean_rank_ic']:.4f}"
    assert rank_ic_str in report_text, f"Rank IC {rank_ic_str} missing in report"

    # Check Sharpe
    sharpe_str = f"{p1['sharpe']:.2f}"
    assert sharpe_str in report_text, f"Sharpe {sharpe_str} missing in report"

    # Check MDD
    mdd_str = f"{p1['mdd_pct']:.2f}%"
    assert mdd_str in report_text, f"MDD {mdd_str} missing in report"

    # Check CAGR
    cagr_str = f"{p1['cagr_pct']:.2f}%"
    assert cagr_str in report_text, f"CAGR {cagr_str} missing in report"

    # Check Verdicts
    assert metrics_data["research_verdict"] in report_text
    assert metrics_data["production_verdict"] in report_text


def test_no_feature_after_decision_timestamp():
    """Test Section 69: Feature set contains only predeclared decision-time features."""
    expected = [
        "chg_ratio",
        "log_tv",
        "log_mc",
        "body_ratio",
        "upper_shadow_ratio",
        "intraday_range",
        "inst_density",
        "foreign_density",
        "kospi_pct",
        "kosdaq_pct",
        "v_kospi",
        "tv_rank",
        "inst_rank",
        "chg_rank",
    ]
    assert expected == FEATURE_COLS


def test_universe_uses_asof_change_not_final_change():
    """Test Section 69: Universe selection criteria strictly bounded by 2% <= chg < 10%."""
    data = {
        "date": [pd.Timestamp("2024-01-02")] * 4,
        "symbol": ["000001", "000002", "000003", "000004"],
        "daily_change_pct": [1.9, 2.5, 9.9, 10.1],
        "trade_value_100m": [200.0, 200.0, 200.0, 200.0],
        "market_cap_100m": [1000.0, 1000.0, 1000.0, 1000.0],
        "close": [10000.0, 10000.0, 10000.0, 10000.0],
        "high": [10000.0, 10000.0, 10000.0, 10000.0],
        "volume": [100000.0, 100000.0, 100000.0, 100000.0],
        "market": ["KOSPI"] * 4,
        "kospi_pct": [0.01] * 4,
        "kosdaq_pct": [0.01] * 4,
        "inst_netbuy": [10.0] * 4,
    }
    df = pd.DataFrame(data)
    df["chg_ratio"] = df["daily_change_pct"] / 100.0
    df["tv_clean"] = df["trade_value_100m"]
    df["mc_clean"] = df["market_cap_100m"]
    df["is_ceiling"] = False

    u0, _ = build_candidate_universe(df)
    assert len(u0) == 2
    assert set(u0["symbol"]) == {"000002", "000003"}


def test_validation_candidates_not_filtered_by_future_tradability(metrics_data: dict):
    """Test Section 10 & 69: Validation candidates contain D+1 untradable/suspended rows."""
    attrition = metrics_data["candidate_attrition"]
    assert attrition["top1_d1_suspended"] > 0, "Suspended Top1 picks must exist and NOT be dropped"
    # Total signals = completed tradable + suspended + unresolved + 1 pending active day
    resolved = attrition["top1_d1_tradable"] + attrition["top1_d1_suspended"] + attrition["top1_unresolved_exits"]
    assert attrition["top1_signals"] in (resolved, resolved + 1)


def test_suspended_top1_is_not_dropped():
    """Test Section 12 & 69: Untradable D+1 candidate is carried forward to next tradable open."""
    market_dates = np.array([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")])
    d_to_idx = {d: i for i, d in enumerate(market_dates)}

    ph_data = [
        # T (2024-01-02)
        {"date": pd.Timestamp("2024-01-02"), "symbol": "000001", "open": 10000.0, "high": 10500.0, "low": 9900.0, "close": 10400.0, "volume": 50000.0},
        # D+1 (2024-01-03): Suspended (volume=0, open=0)
        {"date": pd.Timestamp("2024-01-03"), "symbol": "000001", "open": 0.0, "high": 0.0, "low": 0.0, "close": 10400.0, "volume": 0.0},
        # D+2 (2024-01-04): Resumed trading at 11000.0
        {"date": pd.Timestamp("2024-01-04"), "symbol": "000001", "open": 11000.0, "high": 11200.0, "low": 10900.0, "close": 11100.0, "volume": 60000.0},
    ]
    ph = pd.DataFrame(ph_data)

    cand_df = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-02"), "symbol": "000001", "close": 10400.0}
    ])

    res = attach_forward_exit_paths(cand_df, ph, market_dates, d_to_idx)
    assert len(res) == 1
    assert res.iloc[0]["exit_status"] == "EXIT_SUSPENDED"
    assert res.iloc[0]["exit_price"] == 11000.0
    assert res.iloc[0]["holding_days"] == 2
    assert res.iloc[0]["gross_return"] == pytest.approx(11000.0 / 10400.0 - 1.0)


def test_walk_forward_train_precedes_validation(metrics_data: dict):
    """Test Section 31 & 69: All training folds strictly precede validation folds."""
    folds = metrics_data["walk_forward_folds"]
    for f in folds:
        t_end = pd.Timestamp(f["train_end"])
        v_start = pd.Timestamp(f["val_start"])
        assert t_end < v_start, f"Fold {f['fold']} train_end {t_end} not before val_start {v_start}"


def test_purge_gap_applied(metrics_data: dict):
    """Test Section 29 & 69: Purge gap between train and validation is >= 2 days."""
    folds = metrics_data["walk_forward_folds"]
    for f in folds:
        assert f["purge_gap_days"] >= 2


def test_pa_unfilled_signal_counted_as_zero_return(metrics_data: dict):
    """Test Section 22 & 69: PA overlay includes unfilled signals as 0 return."""
    pa = metrics_data["pipelines"]["P1_PA"]
    fill_rate = pa["fill_rate"]
    filled_net = pa["filled_trade_net_bp"]
    attempted_net = pa["return_per_attempted_signal_bp"]

    expected_attempted = round(fill_rate * filled_net + (1.0 - fill_rate) * 0.0, 2)
    assert attempted_net == pytest.approx(expected_attempted, abs=0.05)


def test_cagr_is_geometric():
    """Test Section 44 & 69: CAGR is strictly geometric compounding, not arithmetic."""
    v0 = 100_000_000.0
    v1 = 150_000_000.0
    days = 730  # ~2 years

    geom = calculate_geometric_cagr(v0, v1, days)
    expected_geom = (v1 / v0) ** (365.25 / days) - 1.0
    assert geom == pytest.approx(expected_geom, rel=1e-6)

    # Arithmetic annual return would be (50% / 730) * 365.25 ≈ 25.0%
    arithmetic = 0.25
    assert geom != pytest.approx(arithmetic)


def test_market_calendar_used_for_nav(metrics_data: dict):
    """Test Section 42 & 69: Market trading days used instead of generic calendar."""
    p1 = metrics_data["pipelines"]["P1"]
    assert p1["n_days"] == 2172  # KRX trading days evaluated in OOF


def test_gate_uses_same_pipeline(metrics_data: dict):
    """Test Section 4 & 50: Gates evaluated for P1 use strictly P1 values."""
    p1 = metrics_data["pipelines"]["P1"]
    g3 = metrics_data["gates"]["Gate_3_Net_Alpha"]
    g10 = metrics_data["gates"]["Gate_10_Portfolio_Risk"]

    assert g3["value_bp"] == p1["net_bp"]
    assert g10["mdd_pct"] == p1["mdd_pct"]


def test_gate8_not_hardcoded(metrics_data: dict):
    """Test Section 59 & 69: Gate 8 robustness is dynamically computed."""
    g8 = metrics_data["gates"]["Gate_8_Strategy_Robustness"]
    assert "details" in g8
    assert "top1_positive" in g8["details"]
    assert "ridge_top1_bp" in g8["details"]
    assert "shallow_top1_bp" in g8["details"]
    assert isinstance(g8["details"]["ridge_top1_bp"], float)


def test_future_mutation_test():
    """Test Section 70: Mutating future post-15:20 data does not alter decision candidate features."""
    cands_original = pd.DataFrame([
        {
            "date": pd.Timestamp("2024-01-02"),
            "symbol": "005930",
            "open": 70000.0,
            "high": 72000.0,
            "low": 69500.0,
            "close": 71500.0,
            "volume": 1000000.0,
            "tv_clean": 715.0,
            "mc_clean": 400000.0,
            "inst_netbuy": 50.0,
            "foreign_netbuy": 30.0,
            "chg_ratio": 0.035,
            "kospi_pct": 0.012,
            "kosdaq_pct": 0.008,
            "v_kospi": 16.5,
        }
    ])
    feats1 = compute_derived_features(cands_original.copy())

    # Mutate post-decision future values: e.g. imagine D+1 price was stored or modified
    cands_mutated = cands_original.copy()
    cands_mutated["future_fake_price"] = 999999.0
    feats2 = compute_derived_features(cands_mutated.copy())

    for col in FEATURE_COLS:
        assert feats1.iloc[0][col] == pytest.approx(feats2.iloc[0][col])


def test_load_and_prepare_price_history_marks_ceiling_via_contract(tmp_path: Path) -> None:
    # Given: a minimal on-disk parquet with one limit-up close among two normal rows
    raw = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02", "2026-01-05"],
            "symbol": ["1", "2", "1"],
            "daily_change_pct": [5.0, 30.0, 4.0],
            "trade_value_100m": [200.0, 200.0, 200.0],
            "volume": [10, 10, 10],
            "close": [1050.0, 1300.0, 1040.0],
            "high": [1060.0, 1300.0, 1050.0],
            "market_cap_100m": [1000.0, 1000.0, 1000.0],
            "prev_close": [1000.0, 1000.0, 1000.0],
            "open": [1000.0, 1000.0, 1000.0],
            "low": [990.0, 990.0, 990.0],
        }
    )
    path = tmp_path / "price_history.parquet"
    raw.to_parquet(path)

    # When
    ph, market_dates, d_to_idx = load_and_prepare_price_history(path)

    # Then: ceiling flag comes from src.strategy.contract.mark_ceiling (chg>=29% and close>=high)
    ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)
    assert ph["symbol"].tolist() == ["000001", "000001", "000002"]
    assert ph["is_ceiling"].tolist() == [False, False, True]
    assert ph["chg_ratio"].to_numpy() == pytest.approx([0.05, 0.04, 0.30])
    assert len(market_dates) == 2
    assert d_to_idx[market_dates[0]] == 0


def test_v3_universe_matches_strategy_contract_selection() -> None:
    # Given
    import numpy as np
    import pandas as pd

    from src.ml.research.v3_engine import build_candidate_universe
    from src.strategy.contract import select_universe

    ph = pd.DataFrame(
        {
            "chg_ratio": [0.02, 0.05, 0.10, 0.01, 0.30, 0.07],
            "tv_clean": [200.0, 200.0, 200.0, 200.0, 200.0, 50.0],
            "mc_clean": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "close": [1000.0, 1000.0, 1000.0, 1000.0, 1300.0, 1000.0],
            "volume": [10, 10, 10, 10, 10, 10],
            "is_ceiling": [False, False, False, False, True, False],
            "market": ["KOSPI"] * 6,
            "date": pd.to_datetime(["2026-01-02"] * 6),
            "symbol": ["000001", "000002", "000003", "000004", "000005", "000006"],
            "inst_netbuy": [1.0] * 6,
            "kospi_pct": [0.5] * 6,
            "kosdaq_pct": [0.5] * 6,
        }
    )

    # When
    u0_df, _ = build_candidate_universe(ph)

    # Then
    expected = ph[select_universe(ph)]["symbol"].tolist()
    assert u0_df["symbol"].tolist() == expected
    assert expected == ["000001", "000002"]
    assert np.all(u0_df["chg_ratio"].to_numpy() < 0.10)


def test_v3_build_candidate_universe_applies_cost_aware_spec() -> None:
    # Given
    import pandas as pd

    from src.ml.research.v3_engine import build_candidate_universe
    from src.strategy.contract import COST_AWARE_UNIVERSE

    ph = pd.DataFrame(
        {
            "chg_ratio": [0.05, 0.05, 0.05],
            "tv_clean": [200.0, 200.0, 200.0],
            "mc_clean": [1000.0, 1000.0, 1000.0],
            "close": [1000.0, 1000.0, 1000.0],
            "volume": [10, 10, 10],
            "is_ceiling": [False, False, False],
            "tick_cost_bp": [5.0, 20.0, float("nan")],
            "market": ["KOSPI"] * 3,
            "date": pd.to_datetime(["2026-01-02"] * 3),
            "symbol": ["000001", "000002", "000003"],
            "inst_netbuy": [1.0] * 3,
            "kospi_pct": [0.5] * 3,
            "kosdaq_pct": [0.5] * 3,
        }
    )

    # When
    u0_df, _ = build_candidate_universe(ph, COST_AWARE_UNIVERSE)

    # Then: 20bp 초과·NaN 비용 종목은 배제, 5bp 종목만 통과
    assert u0_df["symbol"].tolist() == ["000001"]


def test_v3_load_and_prepare_repairs_mixed_unit_and_attaches_tick_cost(tmp_path: Path) -> None:
    # Given: 000001 은 percent 인코딩(오염), 000002 는 ratio 인코딩(정상)
    raw = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-02"],
            "symbol": ["1", "1", "2"],
            "prev_close": [10000.0, 10200.0, 3000.0],
            "close": [10200.0, 10098.0, 3060.0],
            "high": [10300.0, 10200.0, 3100.0],
            "daily_change_pct": [2.0, -1.0, 0.02],
            "trade_value_100m": [200.0, 200.0, 200.0],
            "volume": [10, 10, 10],
            "market_cap_100m": [1000.0, 1000.0, 1000.0],
            "market": ["KOSPI", "KOSPI", "KOSDAQ"],
            "open": [10000.0, 10200.0, 3000.0],
            "low": [9900.0, 10000.0, 2990.0],
        }
    )
    path = tmp_path / "price_history.parquet"
    raw.to_parquet(path)

    # When
    ph, _, _ = load_and_prepare_price_history(path)
    ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Then: 오염 행은 close/prev_close-1 로 교정(0.02, -0.01), 정상 행은 그대로(0.02)
    assert ph["chg_ratio"].to_numpy() == pytest.approx([0.02, -0.01, 0.02], abs=1e-9)
    # 시점정합 1틱 비용: 2023 개편 후 10,200원(10원 틱) -> 9.8bp
    assert ph["tick_cost_bp"].to_numpy()[0] == pytest.approx(10.0 / 10200.0 * 1e4, rel=1e-6)
    assert (ph["tick_cost_bp"].to_numpy() > 0).all()


def test_v3_attach_forward_exit_paths_uses_contract_cost() -> None:
    # Given
    import numpy as np
    import pandas as pd

    from src.ml.research.v3_engine import attach_forward_exit_paths
    from src.strategy.contract import AA_COST, PA_COST, round_trip_cost_bp

    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    market_dates = np.array(dates)
    d_to_idx = {d: i for i, d in enumerate(market_dates)}

    ph = pd.DataFrame(
        {
            "date": [dates[0], dates[1]],
            "symbol": ["000001", "000001"],
            "open": [980.0, 10200.0],
            "high": [1010.0, 10300.0],
            "low": [970.0, 10100.0],
            "close": [10000.0, 10250.0],
            "volume": [100, 100],
        }
    )
    cands = pd.DataFrame({"date": [dates[0]], "symbol": ["000001"], "close": [10000.0]})

    # When
    out = attach_forward_exit_paths(cands, ph, market_dates, d_to_idx)

    # Then
    entry = out["close"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(out["cost_aa_bp"].to_numpy(), round_trip_cost_bp(entry, AA_COST))
    np.testing.assert_allclose(out["cost_pa_bp"].to_numpy(), round_trip_cost_bp(entry, PA_COST))
    np.testing.assert_allclose(out["cost_aa_bp"].to_numpy(), [40.0])
    np.testing.assert_allclose(out["cost_stress_bp"].to_numpy(), [46.0])


def test_v3_metrics_no_longer_duplicates_tick_ladder() -> None:
    # Given
    from src.ml.research import v3_metrics

    # When / Then
    assert not hasattr(v3_metrics, "krx_tick_size")
    assert not hasattr(v3_metrics, "KRX_TICK_BANDS")


@_v3_artifacts_absent
def test_v3_metrics_json_headline_unchanged_after_refactor() -> None:
    # Given
    import json

    metrics_path = _V3_METRICS_PATH

    # When
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    # Then
    assert metrics["ranking_evaluation"]["mean_rank_ic"] == 0.1089
    assert metrics["pipelines"]["P1"]["net_bp"] == 27.57
    assert metrics["pipelines"]["P2"]["net_bp"] == 21.57
    assert metrics["spec"]["primary_exit"].startswith("Open(T+1)")


def test_load_and_prepare_price_history_delegates_to_prepare_price_panel(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import PANEL_INTEGRITY_COLUMNS
    from src.ml.research.v3_engine import load_and_prepare_price_history

    # Given: a percent-encoded price_history parquet spanning two trading days
    ph = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-01", "2023-02-02"]),
        "symbol": ["000001", "000001", "000002", "000002"],
        "open": [10000.0, 10800.0, 20000.0, 20200.0],
        "high": [10900.0, 11100.0, 20300.0, 20500.0],
        "low": [9900.0, 10700.0, 19900.0, 20100.0],
        "close": [10800.0, 11000.0, 20200.0, 20400.0],
        "prev_close": [10000.0, 10800.0, 20000.0, 20200.0],
        "volume": [1e5, 1e5, 1e5, 1e5],
        "market_cap_100m": [900.0, 900.0, 1200.0, 1200.0],
        "trade_value_100m": [300.0, 300.0, 500.0, 500.0],
        "market": ["KOSPI", "KOSPI", "KOSDAQ", "KOSDAQ"],
        "daily_change_pct": [8.0, 1.8518518518518516, 1.0, 0.9900990099009901],
        "inst_netbuy": [0.0, 0.0, 0.0, 0.0],
        "foreign_netbuy": [0.0, 0.0, 0.0, 0.0],
        "program_netbuy": [0.0, 0.0, 0.0, 0.0],
        "kospi_pct": [0.001] * 4,
        "kosdaq_pct": [0.001] * 4,
        "v_kospi": [18.0] * 4,
        "v_kosdaq": [22.0] * 4,
    })
    path = tmp_path / "price_history.parquet"
    ph.to_parquet(path)

    # When
    prepared, market_dates, d_to_idx = load_and_prepare_price_history(path)

    # Then: the calendar contract is unchanged and the integrity columns are present
    assert len(prepared) == 4
    assert PANEL_INTEGRITY_COLUMNS.issubset(set(prepared.columns))
    assert len(market_dates) == 2
    assert d_to_idx[pd.Timestamp("2023-02-01")] == 0
    assert d_to_idx[pd.Timestamp("2023-02-02")] == 1
    # And: the percent-encoded vendor value never reaches the screen
    assert np.isclose(prepared["chg_ratio"].iloc[0], 0.08)
    assert float(np.nanmax(np.abs(prepared["daily_change_pct"].to_numpy(dtype=float)))) < 1.0
    assert np.isfinite(prepared["tick_cost_bp"].to_numpy(dtype=float)).all()
    assert "panel_provenance" in prepared.attrs


