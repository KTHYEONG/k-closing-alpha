"""Model Pipeline wiring 및 일일 의사결정 로직 단위 테스트.

SCENARIO_MODEL_PIPELINE_TRAIN_EVAL 의 wiring 단계가 일일 예측 진입점에
연결되었는지 확인하고, predict_position.py 의 순수 로직(레이블 인코더 로드,
데이터 전처리, SHAP 설명, GMM 기반 동적 의사결정)을 검증합니다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.daily.predict_position as predict_position


def test_run_model_pipeline_wired_into_daily_predict_position() -> None:
    assert hasattr(predict_position, "run_model_pipeline")


def test_load_label_encoder_map_returns_empty_when_file_missing(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert predict_position.load_label_encoder_map(str(missing)) == {}


def test_load_label_encoder_map_builds_mapping(tmp_path) -> None:
    encoder_file = tmp_path / "enc.json"
    encoder_file.write_text(
        json.dumps({"market_type": ["A", "B", "Unknown"]}), encoding="utf-8"
    )
    result = predict_position.load_label_encoder_map(str(encoder_file))
    assert result["market_type"]["mapping"] == {"A": 0, "B": 1, "Unknown": 2}
    assert result["market_type"]["unknown"] == 2


def test_load_label_encoder_map_handles_corrupt_json(tmp_path) -> None:
    encoder_file = tmp_path / "bad.json"
    encoder_file.write_text("{ not json", encoding="utf-8")
    assert predict_position.load_label_encoder_map(str(encoder_file)) == {}


def test_load_and_preprocess_data_exits_on_missing_file() -> None:
    with (
        patch.object(predict_position.os.path, "exists", return_value=False),
        patch.object(predict_position.sys, "exit", side_effect=SystemExit) as exit_mock,
        pytest.raises(SystemExit),
    ):
        predict_position.load_and_preprocess_data("no_such.xlsx")
    exit_mock.assert_called_once_with(1)


def test_load_and_preprocess_data_exits_on_read_error() -> None:
    with (
        patch.object(predict_position.os.path, "exists", return_value=True),
        patch.object(predict_position.pd, "read_excel", side_effect=OSError("boom")),
        patch.object(predict_position.sys, "exit", side_effect=SystemExit),
        pytest.raises(SystemExit),
    ):
        predict_position.load_and_preprocess_data("fake.xlsx")


def test_load_and_preprocess_data_normalizes_columns() -> None:
    raw = pd.DataFrame(
        {
            "(종목코드)": ["123", 456],
            "(시가총액, 억)": [10.0, 20.0],
            "기관_순매수(억)": [1.0, 2.0],
            "(상장일수)": ["300", "500"],
            "기타": ["x", "y"],
        }
    )
    with (
        patch.object(predict_position.os.path, "exists", return_value=True),
        patch.object(predict_position.pd, "read_excel", return_value=raw),
    ):
        result = predict_position.load_and_preprocess_data("fake.xlsx")
    assert result["종목코드"].tolist() == ["000123", "000456"]
    assert result["기관_순매수"].tolist() == [100_000_000, 200_000_000]
    assert result["시가총액"].tolist() == [10.0, 20.0]
    assert (result["(상장일수)"] >= predict_position.settings.EMA_PERIOD).all()


def test_load_and_preprocess_data_without_listing_days() -> None:
    raw = pd.DataFrame(
        {"종목코드": ["000123"], "거래대금(억)": [3.0], "등락률": [1.0]}
    )
    with (
        patch.object(predict_position.os.path, "exists", return_value=True),
        patch.object(predict_position.pd, "read_excel", return_value=raw),
    ):
        result = predict_position.load_and_preprocess_data("fake.xlsx")
    assert len(result) == 1
    assert result["거래대금"].tolist() == [300_000_000]


def test_load_and_preprocess_data_filters_insufficient_listing_days() -> None:
    raw = pd.DataFrame({"종목코드": ["000001"], "(상장일수)": ["1"]})
    with (
        patch.object(predict_position.os.path, "exists", return_value=True),
        patch.object(predict_position.pd, "read_excel", return_value=raw),
    ):
        result = predict_position.load_and_preprocess_data("fake.xlsx")
    assert result.empty


def test_explain_predictions_with_shap_skips_when_not_installed() -> None:
    with patch.object(predict_position, "HAS_SHAP", False):
        predict_position.explain_predictions_with_shap(None, None, [])


def test_explain_predictions_with_shap_reports_features() -> None:
    X = pd.DataFrame(
        {"f_num": [1.0, 2.0], "f_str": ["high", "low"]}, index=[0, 1]
    )
    shap_values = np.array([[0.4, -0.1], [-0.2, 0.3]])

    class FakeExplainer:
        def __init__(self, model) -> None:
            self.model = model

        def shap_values(self, X_final) -> np.ndarray:
            return shap_values

    fake_shap = SimpleNamespace(TreeExplainer=FakeExplainer)
    with (
        patch.object(predict_position, "HAS_SHAP", True),
        patch.object(predict_position, "shap", fake_shap, create=True),
    ):
        predict_position.explain_predictions_with_shap(
            "model", X, stock_names=["A", "A"], top_n=2
        )


def test_explain_predictions_with_shap_handles_exception() -> None:
    def boom(*args, **kwargs) -> None:
        raise RuntimeError("shap failed")

    fake_shap = SimpleNamespace(TreeExplainer=boom)
    with (
        patch.object(predict_position, "HAS_SHAP", True),
        patch.object(predict_position, "shap", fake_shap, create=True),
    ):
        predict_position.explain_predictions_with_shap("model", None, [])


def test_explain_predictions_with_shap_formats_non_numeric_values() -> None:
    X = pd.DataFrame({"f_obj": [object(), object()]})
    shap_values = np.array([[0.5], [-0.3]])

    class FakeExplainer:
        def __init__(self, model) -> None:
            self.model = model

        def shap_values(self, X_final) -> np.ndarray:
            return shap_values

    fake_shap = SimpleNamespace(TreeExplainer=FakeExplainer)
    with (
        patch.object(predict_position, "HAS_SHAP", True),
        patch.object(predict_position, "shap", fake_shap, create=True),
    ):
        predict_position.explain_predictions_with_shap("model", X, ["A", "B"], top_n=1)


def test_main_returns_when_no_actionable_stocks() -> None:
    df_condition = pd.DataFrame({"종목코드": ["000001"], "종목명": ["AAA"]})
    with (
        patch.object(predict_position.os.path, "exists", return_value=False),
        patch.object(
            predict_position, "load_and_preprocess_data", return_value=df_condition
        ),
        patch.object(predict_position, "load_theme_from_db", return_value={}),
        patch.object(predict_position, "sync_theme_only"),
    ):
        predict_position.main()


def _wide_scores() -> pd.Series:
    return pd.Series(
        [0.20, 0.35, 0.50, 0.62, 0.72, 0.85, 0.95, 1.05, 1.12, 1.20, 1.30, 1.40]
    )


def test_get_decision_batch_uses_static_logic_without_sklearn() -> None:
    scores = pd.Series([0.30, 0.70, 0.90, 1.10])
    with patch.object(predict_position, "HAS_SKLEARN", False):
        decisions = predict_position.get_decision_batch(scores)
    assert decisions.tolist() == ["Reduce", "Neutral", "Expand", "Max_Expand"]


def test_get_decision_batch_static_when_sample_too_small() -> None:
    scores = pd.Series([0.3, 0.9, 1.1])
    with patch.object(predict_position, "HAS_SKLEARN", True):
        decisions = predict_position.get_decision_batch(scores)
    assert decisions.tolist() == ["Reduce", "Expand", "Max_Expand"]


def test_get_decision_batch_static_when_scores_have_low_discrimination() -> None:
    scores = pd.Series([1.0] * 12)
    with patch.object(predict_position, "HAS_SKLEARN", True):
        decisions = predict_position.get_decision_batch(scores)
    assert set(decisions.tolist()) == {"Expand"}


def test_get_decision_batch_gmm_path_applies_safety_logic() -> None:
    scores = _wide_scores()
    with patch.object(predict_position, "HAS_SKLEARN", True):
        decisions = predict_position.get_decision_batch(scores)
    assert set(decisions.tolist()) <= {
        "Reduce",
        "Neutral",
        "Expand",
        "Max_Expand",
    }
    assert decisions[scores < predict_position.ABSOLUTE_MIN_SCORE].tolist() == [
        "Reduce"
    ] * int((scores < predict_position.ABSOLUTE_MIN_SCORE).sum())
    assert decisions[scores >= 1.07].tolist() == ["Max_Expand"] * int(
        (scores >= 1.07).sum()
    )


def test_get_decision_batch_falls_back_to_static_on_gmm_error() -> None:
    scores = _wide_scores()
    with (
        patch.object(predict_position, "HAS_SKLEARN", True),
        patch.object(
            predict_position.GaussianMixture, "fit", side_effect=RuntimeError("gmm")
        ),
    ):
        decisions = predict_position.get_decision_batch(scores)
    assert decisions.tolist() == scores.apply(
        lambda s: (
            "Reduce"
            if s < 0.59
            else "Neutral"
            if s < 0.87
            else "Expand"
            if s < 1.07
            else "Max_Expand"
        )
    ).tolist()
