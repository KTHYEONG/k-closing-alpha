"""Algorithm-family ensemble 외부 OOF 의 bounded 병렬 실행.

네 개의 독립적인 return 전문가(fold 학습 자체는 각 library 를 단일 워커로
고정)를 ``ThreadPoolExecutor`` 로 병렬 실행해 벽시계 시간을 줄입니다. thread
방식은 33k 행 패널을 프로세스별로 복사하지 않고, 하위 learner 가 fit 동안
Python 실행을 해제하므로 library 내부 thread pool 이 아닌 executor 가 병렬성을
소유합니다. process pool 이나 DataFrame row-wise apply 는 사용하지 않습니다.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pandas as pd
import psutil

from src.ml.training.pipelines import run_model_pipeline
from src.ml.training.validation import _ALGORITHM_FAMILIES

# 단일 전문가 fit 에 허용하는 메모리 상한 (2 GiB). 가용 메모리를 이 예산으로 나눠
# 워커 수를 제한하므로, 병렬 실행이 호스트 메모리를 초과해 스왑에 빠지지 않습니다.
_EXPERT_MEMORY_BUDGET_BYTES = 2 * 1024**3


def _resolve_expert_worker_count(
    max_workers: int | None,
    n_experts: int,
    cpu_count: int | None,
    available_memory_bytes: int,
) -> int:
    """요청값·가용 CPU·가용 메모리·전문가 수로 워커 수를 결정합니다.

    ``max_workers`` 가 양수가 아니면 ``ValueError`` 로 fail-closed 하며, 실제
    실행 값은 ``min(요청(있으면), 가용 CPU, 가용 메모리 기반, 전문가 수)`` 로
    하한 1 을 보장합니다. 고정 하드웨어 워커 수는 사용하지 않습니다.
    """
    if max_workers is not None and max_workers <= 0:
        raise ValueError(f"max_workers must be a positive integer, got {max_workers}")
    bounds: list[int] = [n_experts]
    if max_workers is not None:
        bounds.append(max_workers)
    if cpu_count is not None and cpu_count > 0:
        bounds.append(cpu_count)
    if available_memory_bytes > 0:
        memory_workers = max(1, int(available_memory_bytes // _EXPERT_MEMORY_BUDGET_BYTES))
        bounds.append(memory_workers)
    return max(1, min(bounds))


def run_algorithm_expert_oof_parallel(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int,
    purge_gap: int,
    max_workers: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, float | int]]:
    """외부 purged splitter 로 네 algorithm-family return 전문가를 병렬 실행합니다.

    각 전문가는 ``run_model_pipeline`` 을 호출하며, 입력 ``df`` 는 변경하지
    않고(내부에서 정렬 복사) fold 학습은 library 단일 워커로 수행합니다.
    LightGBM 은 중첩 병렬성을 막기 위해 ``n_jobs=1`` 을 명시하고, XGBoost/
    CatBoost/RandomForest 는 이미 단일 워커 설정을 유지합니다. 결과는
    ``_ALGORITHM_FAMILIES`` 순서로 반환하며, 작업 실패 시 원본 예외를 그대로
    전파합니다. 모델 artifact 는 영속화하지 않습니다.

    Returns:
        ``(results, telemetry)`` — ``results`` 는 model_type 키의 OOF 결과 dict,
        ``telemetry`` 는 ``n_workers``(결정된 워커 수)와
        ``expert_wall_seconds``(병렬 블록 벽시계 시간) 를 포함합니다.
    """
    resolved = _resolve_expert_worker_count(
        max_workers,
        len(_ALGORITHM_FAMILIES),
        psutil.cpu_count(logical=True),
        psutil.virtual_memory().available,
    )
    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=resolved) as executor:
        futures: dict[str, Future[dict[str, Any]]] = {
            model_type: executor.submit(
                run_model_pipeline,
                df,
                feature_cols,
                target_col,
                group_col,
                n_splits=n_splits,
                purge_gap=purge_gap,
                model_type=model_type,
                model_params={"n_jobs": 1} if model_type == "lgb_regressor" else None,
            )
            for model_type in _ALGORITHM_FAMILIES
        }
        for model_type in _ALGORITHM_FAMILIES:
            results[model_type] = futures[model_type].result()
    wall_seconds = time.perf_counter() - started
    return results, {"n_workers": resolved, "expert_wall_seconds": wall_seconds}
