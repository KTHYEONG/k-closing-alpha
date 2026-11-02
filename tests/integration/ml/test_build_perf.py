import os
import time
from pathlib import Path

import pandas as pd
import pytest

from src.ml.dataset import build_ml_dataset


@pytest.mark.slow
def test_build_ml_dataset_perf_budget() -> None:
    # Research journal inputs (gdrive:quant-lake/research/k-closing-alpha/manual_journal_v1/) by explicit opt-in path.
    tl_path = Path(os.environ.get("KCA_RESEARCH_TRADE_LOG", ""))
    th_path = Path(os.environ.get("KCA_RESEARCH_THEME", ""))
    if not tl_path.is_file():
        pytest.skip("set KCA_RESEARCH_TRADE_LOG to the research trade-log parquet")
    tl = pd.read_parquet(tl_path)
    th = pd.read_parquet(th_path) if th_path.is_file() else None
    start = time.perf_counter()
    x, targets, cat, proc = build_ml_dataset(tl, th, feature_set="close_morning61", panel_mode="scenario_action")
    elapsed = time.perf_counter() - start
    assert len(proc) > 30000
    assert elapsed <= 5.0, f"build_ml_dataset compute took {elapsed:.1f}s, budget 5s (pre-change ~5.5s)"
