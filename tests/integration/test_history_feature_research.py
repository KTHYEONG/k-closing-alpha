"""History-feature real-data streaming benchmark (slow, opt-in).

`docs/specs/ml_history_feature_scaling.md` HFS-08: 실제 ``price_history.parquet``
입력으로 peak RSS/elapsed/batch count/OOF 날짜 일치/선정 피처 수를 기록하고,
활성 아티팩트를 승격·덮어쓰지 않습니다.

기본 테스트 스위트에서는 실행하지 않습니다. 의도적으로 실행하려면
``K_CLOSING_RUN_SLOW=1`` 환경 변수를 설정합니다 (예: ``K_CLOSING_RUN_SLOW=1 uv run pytest
tests/integration/test_history_feature_research.py -m slow``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.ml.feature_selection import FeatureSelectionConfig
from src.ml.history_feature_research import run_history_feature_research_experiment
from src.ml.history_features import HistoryFeatureExecutionConfig

HISTORY_PATH = Path("data/history/price_history.parquet")
TRADE_LOG_PATH = Path("data/parquet/trade_log.parquet")

_RUN_SLOW = os.environ.get("K_CLOSING_RUN_SLOW") == "1"


@pytest.mark.skipif(
    not _RUN_SLOW,
    reason="real-data benchmark; set K_CLOSING_RUN_SLOW=1 to run",
)
@pytest.mark.slow
@pytest.mark.skipif(
    not (HISTORY_PATH.is_file() and TRADE_LOG_PATH.is_file()),
    reason="real data files are missing",
)
def test_hfs08_real_data_streaming_benchmark(tmp_path: Path) -> None:
    """실데이터 streaming 실행이 메모리 예산 내에서 불변량을 유지하고 지표를 기록합니다."""
    trade_log = pd.read_parquet(TRADE_LOG_PATH)
    export_dir = tmp_path / "research"
    config = FeatureSelectionConfig()
    execution = HistoryFeatureExecutionConfig(
        memory_budget_bytes=8 * 1024**3,
        enforce_memory_budget=True,
    )
    result = run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history_path=str(HISTORY_PATH),
        n_splits=5,
        purge_gap=1,
        feature_selection_config=config,
        execution_config=execution,
        export_dir=str(export_dir),
    )
    metrics = result["build_metrics"]
    assert metrics["input_history_rows"] > 0
    assert metrics["decision_key_rows"] >= 1
    assert metrics["output_rows"] == metrics["decision_key_rows"]
    assert metrics["batch_count"] >= 1
    assert metrics["peak_rss_bytes"] <= 8 * 1024**3
    assert metrics["elapsed_seconds"] > 0.0
    assert result["comparison"]["identical_oof_dates"] is True
    final_count = len(result["candidate"]["final_features"])
    assert 1 <= final_count <= 500
    # 활성 아티팩트를 승격하거나 덮어쓰지 않습니다.
    assert Path(result["candidate_bundle_path"]).is_file()
