"""v3 리서치 엔진의 생존 함수와 삭제된 파이프라인 평가 함수를 검증하는 회귀 가드."""

from __future__ import annotations


def test_v3_engine_survivors_remain_and_dead_pipelines_are_gone() -> None:
    from src.ml.research import v3_engine

    # 이번 단계에서 삭제된 파이프라인 평가 함수 3종
    for dead in (
        "execute_walk_forward_oof",
        "evaluate_all_pipelines",
        "evaluate_decision_gates",
    ):
        assert not hasattr(v3_engine, dead), f"v3_engine.{dead} should be deleted"
    # 살아남아야 하는 엔진 함수
    for alive in (
        "load_and_prepare_price_history",
        "build_candidate_universe",
        "attach_forward_exit_paths",
        "compute_derived_features",
    ):
        assert callable(getattr(v3_engine, alive)), f"v3_engine.{alive} must remain"


def test_compute_derived_features_and_costs_use_close_raw() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.research.v3_engine import attach_forward_exit_paths, compute_derived_features
    from src.strategy.contract import AA_COST, round_trip_cost_bp

    d0, d1 = pd.Timestamp("2018-04-27"), pd.Timestamp("2018-04-30")
    ph = pd.DataFrame({
        "date": [d0, d1], "symbol": ["005930", "005930"], "open": [52000.0, 53000.0], "high": [53200.0, 53000.0], "low": [51800.0, 53000.0],
        "close": [53000.0, 53000.0], "close_raw": [2650000.0, 2650000.0], "volume": [606216.0, 100.0], "market": ["KOSPI", "KOSPI"],
        "tv_clean": [16112.4, 1.0], "mc_clean": [3402242.0, 3402242.0], "inst_netbuy": [1.0e9, 0.0], "foreign_netbuy": [0.0, 0.0], "chg_ratio": [0.0165, 0.0],
    })
    cands = ph.iloc[[0]].copy().reset_index(drop=True)
    market_dates = np.array([d0, d1])

    # When
    feats = compute_derived_features(cands.copy())
    paths = attach_forward_exit_paths(cands.copy(), ph, market_dates, {d0: 0, d1: 1})

    # Then
    assert feats["inst_density"].iloc[0] == pytest.approx(1.0e9 / (2650000.0 * 606216.0))
    dates = np.array([d0], dtype="datetime64[ns]")
    assert paths["cost_aa_bp"].iloc[0] == pytest.approx(float(round_trip_cost_bp(np.array([2650000.0]), dates, np.array(["KOSPI"], dtype=object), AA_COST)[0]))
    assert paths["gross_return"].iloc[0] == pytest.approx(53000.0 / 53000.0 - 1.0)
    plain = compute_derived_features(cands.drop(columns=["close_raw"]))
    assert plain["inst_density"].iloc[0] == pytest.approx(1.0e9 / (53000.0 * 606216.0))
