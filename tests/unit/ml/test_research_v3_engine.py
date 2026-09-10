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
