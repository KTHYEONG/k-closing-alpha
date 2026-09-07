from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.ml.dataset import clean_column_names
from src.ml.scenario_rules import derive_scenario_labels, scenario_agreement_report

pytestmark = pytest.mark.slow

_PANEL = Path("data/parquet/ml_training_panel.parquet")
_PH = Path("data/history/price_history.parquet")


@pytest.mark.skipif(not (_PANEL.exists() and _PH.exists()), reason="local data artifacts required")
def test_auto_scenario_agreement_on_journaled_panel(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    panel_raw = pd.read_parquet(_PANEL)
    panel_raw = panel_raw[panel_raw["label_source"] == "sheet_executed"].copy()
    ph = pd.read_parquet(_PH)
    cleaned = clean_column_names(panel_raw)
    manual = cleaned["chart_analysis"].astype(str).reset_index(drop=True)
    cleaned = cleaned.reset_index(drop=True)

    # Act
    auto = derive_scenario_labels(cleaned, ph).reset_index(drop=True)
    report = scenario_agreement_report(manual, auto)

    # Assert
    with caplog.at_level(logging.INFO):
        logging.getLogger(__name__).info(
            "[EVAL] stage=scenario_auto_agreement n=%s overall=%.3f ceiling_recall=%.3f",
            report["n"], report["overall_agreement"],
            report["per_class"]["상한가 다음날"]["recall"],
        )
    assert report["overall_agreement"] >= 0.45
    assert report["per_class"]["상한가 다음날"]["recall"] >= 0.85
