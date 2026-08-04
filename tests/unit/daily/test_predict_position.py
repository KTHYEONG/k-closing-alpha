"""일일 예측 진입점 wiring 및 Fast Inference 단위 테스트.

SCENARIO_MODEL_PIPELINE_TRAIN_EVAL 의 wiring 단계가 일일 예측 진입점에
연결되었는지 확인하고, 레거시 GMM/Static 의사결정 로직 제거 여부와
저장된 모델 아티팩트 기반 Fast Inference(< 1초) 동작을 검증합니다.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.daily.predict_position as predict_position
from src.ml.sizing_engine import _train_inline_bundle, save_model_artifacts

FEATURE_COLS = ["f1", "f2"]
TARGET_COL = "target_net_return"
GROUP_COL = "date"


def _snapshot_df(n_rows: int = 24, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n_rows)
    f2 = rng.normal(size=n_rows)
    target = 0.02 * f1 + 0.01 * f2 + rng.normal(loc=0.0, scale=0.008, size=n_rows)
    return pd.DataFrame(
        {
            "종목명": [f"종목{i}" for i in range(n_rows)],
            "테마_섹터": ["테마A"] * n_rows,
            "차트분석": ["거래량 폭증_Y"] * n_rows,
            "선정순위": list(range(1, n_rows + 1)),
            GROUP_COL: [f"2026-08-{d:02d}" for d in range(1, n_rows + 1)],
            "f1": f1,
            "f2": f2,
            TARGET_COL: target,
        }
    )


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


def test_legacy_gmm_logic_removed() -> None:
    """레거시 GMM/Static 의사결정 및 하드코딩 Safety Floor 가 제거되었는지 확인한다."""
    assert not hasattr(predict_position, "get_decision_batch")
    assert not hasattr(predict_position, "GaussianMixture")
    assert not hasattr(predict_position, "HAS_SKLEARN")
    assert not hasattr(predict_position, "SAFETY_MAX_FLOOR")
    assert not hasattr(predict_position, "SAFETY_EXPAND_FLOOR")
    assert not hasattr(predict_position, "ABSOLUTE_MIN_SCORE")
    assert not hasattr(predict_position, "MIN_SAMPLES_FOR_GMM")


def test_run_daily_sizing_inference_from_saved_artifacts_within_1s(tmp_path) -> None:
    """저장된 모델 아티팩트(artifacts/models)를 로드하여 Fast Inference 가 1초 이내 완료된다."""
    snapshot = _snapshot_df()
    bundle = _train_inline_bundle(snapshot, FEATURE_COLS, TARGET_COL, GROUP_COL)
    save_model_artifacts(bundle, str(tmp_path))

    start = time.perf_counter()
    loaded = predict_position.load_model_artifacts(str(tmp_path))
    result = predict_position.run_daily_sizing_inference(
        snapshot, loaded, feature_cols=FEATURE_COLS, group_col=GROUP_COL
    )
    elapsed = time.perf_counter() - start

    assert {"utility_score", "grade", "grade_multiplier", "allocation"}.issubset(
        result.columns
    )
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()
    assert result["allocation"].ge(0.0).all()
    assert elapsed < 1.0


def test_run_daily_sizing_inference_adds_missing_group_col() -> None:
    snapshot = _snapshot_df().drop(columns=[GROUP_COL])
    bundle = _train_inline_bundle(_snapshot_df(), FEATURE_COLS, TARGET_COL, GROUP_COL)
    result = predict_position.run_daily_sizing_inference(snapshot, bundle)
    assert GROUP_COL in result.columns
    assert len(result) == len(snapshot)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)


def test_run_daily_sizing_inference_fills_missing_features() -> None:
    snapshot = _snapshot_df().drop(columns=["f2"])
    bundle = _train_inline_bundle(_snapshot_df(), FEATURE_COLS, TARGET_COL, GROUP_COL)
    result = predict_position.run_daily_sizing_inference(snapshot, bundle)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)
    assert len(result) == len(snapshot)


def test_run_daily_sizing_inference_raises_without_feature_cols() -> None:
    with pytest.raises(ValueError, match="feature_cols is empty"):
        predict_position.run_daily_sizing_inference(_snapshot_df(), {"dummy": 1})
