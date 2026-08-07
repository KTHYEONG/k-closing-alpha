"""Bounded parallel algorithm-family OOF execution 단위 테스트.

SCENARIO_ENSEMBLE_PERF_04 / SCENARIO_ENSEMBLE_PERF_05
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.ml.training.ensemble_execution import (
    _EXPERT_MEMORY_BUDGET_BYTES,
    _resolve_expert_worker_count,
    run_algorithm_expert_oof_parallel,
)
from src.ml.training.validation import _ALGORITHM_FAMILIES

FEATURE_COLS = ["feature_a", "feature_b"]
TARGET_COL = "net_return"
GROUP_COL = "trade_date"


def _make_dataset(n_groups: int = 12, rows_per_group: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime([f"2024-03-{d:02d}" for d in range(1, n_groups + 1)])
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({"trade_date": date} for _ in range(rows_per_group))
    df = pd.DataFrame(rows)
    df["feature_a"] = rng.normal(size=len(df))
    df["feature_b"] = rng.normal(size=len(df))
    df["net_return"] = 0.01 * df["feature_a"] + rng.normal(scale=0.004, size=len(df))
    df["selection_rank"] = df.groupby(GROUP_COL, sort=False).cumcount() + 1
    df["stock_code"] = [f"{i % 6 + 1:06d}" for i in range(len(df))]
    df["chart_analysis"] = "신고가"
    return df


def test_resolve_expert_worker_count_positive_and_bounded() -> None:
    """(SCENARIO_ENSEMBLE_PERF_04) 워커 해상도는 양수이며, 요청값·가용 CPU·가용
    메모리·전문가 수에 의해 상한됩니다."""
    mem = 11 * 1024**3
    # 용량이 충분하면 전문가 수가 상한입니다.
    assert _resolve_expert_worker_count(None, 4, cpu_count=20, available_memory_bytes=mem) == 4
    # 요청값이 상한입니다.
    assert _resolve_expert_worker_count(2, 4, cpu_count=20, available_memory_bytes=mem) == 2
    # 요청값이 용량을 초과해도 용량·전문가 수로 하향됩니다.
    assert _resolve_expert_worker_count(100, 4, cpu_count=20, available_memory_bytes=mem) == 4
    # 가용 CPU 가 제약입니다.
    assert _resolve_expert_worker_count(None, 4, cpu_count=2, available_memory_bytes=mem) == 2
    # 가용 메모리가 제약입니다.
    small_mem = 3 * _EXPERT_MEMORY_BUDGET_BYTES
    assert _resolve_expert_worker_count(None, 4, cpu_count=100, available_memory_bytes=small_mem) == 3
    # 용량 정보가 전혀 없으면 전문가 수가 상한입니다 (양수 보장).
    assert _resolve_expert_worker_count(None, 4, cpu_count=None, available_memory_bytes=0) == 4
    # 양수가 아닌 요청은 fail-closed 입니다.
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        _resolve_expert_worker_count(0, 4, cpu_count=20, available_memory_bytes=mem)
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        _resolve_expert_worker_count(-3, 4, cpu_count=20, available_memory_bytes=mem)


def test_run_algorithm_expert_oof_parallel_rejects_non_positive_workers() -> None:
    """(SCENARIO_ENSEMBLE_PERF_04) 요청 워커가 양수가 아니면 ValueError 입니다."""
    df = _make_dataset()
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        run_algorithm_expert_oof_parallel(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            n_splits=2,
            purge_gap=1,
            max_workers=0,
        )


def test_run_algorithm_expert_oof_parallel_returns_canonical_order() -> None:
    """(SCENARIO_ENSEMBLE_PERF_05) 병렬 결과는 _ALGORITHM_FAMILIES 순서를 유지하고
    각 전문가 OOF 계약을 충족합니다."""
    df = _make_dataset()
    results, telemetry = run_algorithm_expert_oof_parallel(
        df,
        FEATURE_COLS,
        TARGET_COL,
        GROUP_COL,
        n_splits=2,
        purge_gap=1,
        max_workers=2,
    )
    assert list(results) == list(_ALGORITHM_FAMILIES)
    assert telemetry["n_workers"] == 2
    assert telemetry["expert_wall_seconds"] >= 0.0
    for model_type in _ALGORITHM_FAMILIES:
        oof = results[model_type]["oof_predictions"]
        assert {"pred", "fold", GROUP_COL, TARGET_COL} <= set(oof.columns)
        assert oof["pred"].notna().all()


def test_run_algorithm_expert_oof_parallel_propagates_first_exception() -> None:
    """(SCENARIO_ENSEMBLE_PERF_05) 실패한 전문가 작업은 원본 예외를 그대로
    전파합니다."""

    def _boom(
        df,
        feature_cols,
        target_col,
        group_col,
        *,
        n_splits,
        purge_gap,
        model_type,
        model_params=None,
    ):
        if model_type == "xgb_regressor":
            raise RuntimeError("xgb worker exploded")
        return {"oof_predictions": pd.DataFrame()}

    df = _make_dataset()
    with (
        patch(
            "src.ml.training.ensemble_execution.run_model_pipeline",
            side_effect=_boom,
        ),
        pytest.raises(RuntimeError, match="xgb worker exploded"),
    ):
        run_algorithm_expert_oof_parallel(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            n_splits=2,
            purge_gap=1,
            max_workers=2,
        )


def test_run_algorithm_expert_oof_parallel_auto_resolves_from_measured_capacity() -> None:
    """psutil 로 측정한 CPU/가용 메모리 용량에서 워커 수를 자동 결정하고
    telemetry 에 기록합니다."""
    df = _make_dataset()
    mem = 11 * 1024**3
    fake_memory = type("FakeVirtualMemory", (), {"available": mem})()
    with (
        patch("src.ml.training.ensemble_execution.psutil.cpu_count", return_value=20),
        patch(
            "src.ml.training.ensemble_execution.psutil.virtual_memory",
            return_value=fake_memory,
        ),
    ):
        _results, telemetry = run_algorithm_expert_oof_parallel(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            n_splits=2,
            purge_gap=1,
            max_workers=None,
        )
    assert telemetry["n_workers"] == len(_ALGORITHM_FAMILIES)
