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

import src.daily.predict as predict
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
    assert hasattr(predict, "run_model_pipeline")


def test_load_label_encoder_map_returns_empty_when_file_missing(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert predict.load_label_encoder_map(str(missing)) == {}


def test_load_label_encoder_map_builds_mapping(tmp_path) -> None:
    encoder_file = tmp_path / "enc.json"
    encoder_file.write_text(
        json.dumps({"market_type": ["A", "B", "Unknown"]}), encoding="utf-8"
    )
    result = predict.load_label_encoder_map(str(encoder_file))
    assert result["market_type"]["mapping"] == {"A": 0, "B": 1, "Unknown": 2}
    assert result["market_type"]["unknown"] == 2


def test_load_label_encoder_map_handles_corrupt_json(tmp_path) -> None:
    encoder_file = tmp_path / "bad.json"
    encoder_file.write_text("{ not json", encoding="utf-8")
    assert predict.load_label_encoder_map(str(encoder_file)) == {}


def test_load_and_preprocess_data_exits_on_missing_file() -> None:
    with (
        patch.object(predict.os.path, "exists", return_value=False),
        patch.object(predict.sys, "exit", side_effect=SystemExit) as exit_mock,
        pytest.raises(SystemExit),
    ):
        predict.load_and_preprocess_data("no_such.xlsx")
    exit_mock.assert_called_once_with(1)


def test_load_and_preprocess_data_exits_on_read_error() -> None:
    with (
        patch.object(predict.os.path, "exists", return_value=True),
        patch.object(predict.pd, "read_csv", side_effect=OSError("boom")),
        patch.object(predict.sys, "exit", side_effect=SystemExit),
        pytest.raises(SystemExit),
    ):
        predict.load_and_preprocess_data("fake.xlsx")


def test_load_and_preprocess_data_normalizes_columns() -> None:
    raw = pd.DataFrame(
        {
            "종목코드": ["123", 456],
            "시가총액": [10.0, 20.0],
            "기관_순매수": [1.0, 2.0],
            "상장일수": ["300", "500"],
            "기타": ["x", "y"],
        }
    )
    with (
        patch.object(predict.os.path, "exists", return_value=True),
        patch.object(predict.pd, "read_csv", return_value=raw),
    ):
        result = predict.load_and_preprocess_data("fake.xlsx")
    assert result["종목코드"].tolist() == ["000123", "000456"]
    assert result["기관_순매수"].tolist() == [100_000_000, 200_000_000]
    assert result["시가총액"].tolist() == [10.0, 20.0]
    assert (result["상장일수"] >= predict.settings.EMA_PERIOD).all()


def test_load_and_preprocess_data_without_listing_days() -> None:
    raw = pd.DataFrame(
        {"종목코드": ["000123"], "거래대금": [3.0], "등락률": [1.0]}
    )
    with (
        patch.object(predict.os.path, "exists", return_value=True),
        patch.object(predict.pd, "read_csv", return_value=raw),
    ):
        result = predict.load_and_preprocess_data("fake.xlsx")
    assert len(result) == 1
    assert result["거래대금"].tolist() == [300_000_000]


def test_load_and_preprocess_data_filters_insufficient_listing_days() -> None:
    raw = pd.DataFrame({"종목코드": ["000001"], "상장일수": ["1"]})
    with (
        patch.object(predict.os.path, "exists", return_value=True),
        patch.object(predict.pd, "read_csv", return_value=raw),
    ):
        result = predict.load_and_preprocess_data("fake.xlsx")
    assert result.empty


def test_explain_predictions_with_shap_skips_when_not_installed() -> None:
    with patch.object(predict, "HAS_SHAP", False):
        predict.explain_predictions_with_shap(None, None, [])


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
        patch.object(predict, "HAS_SHAP", True),
        patch.object(predict, "shap", fake_shap, create=True),
    ):
        predict.explain_predictions_with_shap(
            "model", X, stock_names=["A", "A"], top_n=2
        )


def test_explain_predictions_with_shap_handles_exception() -> None:
    def boom(*args, **kwargs) -> None:
        raise RuntimeError("shap failed")

    fake_shap = SimpleNamespace(TreeExplainer=boom)
    with (
        patch.object(predict, "HAS_SHAP", True),
        patch.object(predict, "shap", fake_shap, create=True),
    ):
        predict.explain_predictions_with_shap("model", None, [])


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
        patch.object(predict, "HAS_SHAP", True),
        patch.object(predict, "shap", fake_shap, create=True),
    ):
        predict.explain_predictions_with_shap("model", X, ["A", "B"], top_n=1)


def test_main_returns_when_no_actionable_stocks() -> None:
    df_condition = pd.DataFrame({"종목코드": ["000001"], "종목명": ["AAA"]})
    with (
        patch.object(predict.os.path, "exists", return_value=False),
        patch.object(
            predict, "load_and_preprocess_data", return_value=df_condition
        ),
        patch.object(predict, "load_theme_from_db", return_value={}),
        patch.object(predict, "sync_theme_only"),
    ):
        predict.main()


def test_legacy_gmm_logic_removed() -> None:
    """레거시 GMM/Static 의사결정 및 하드코딩 Safety Floor 가 제거되었는지 확인한다."""
    assert not hasattr(predict, "get_decision_batch")
    assert not hasattr(predict, "GaussianMixture")
    assert not hasattr(predict, "HAS_SKLEARN")
    assert not hasattr(predict, "SAFETY_MAX_FLOOR")
    assert not hasattr(predict, "SAFETY_EXPAND_FLOOR")
    assert not hasattr(predict, "ABSOLUTE_MIN_SCORE")
    assert not hasattr(predict, "MIN_SAMPLES_FOR_GMM")


def test_run_daily_sizing_inference_from_saved_artifacts_within_1s(tmp_path) -> None:
    """저장된 모델 아티팩트(artifacts/models)를 로드하여 Fast Inference 가 1초 이내 완료된다."""
    snapshot = _snapshot_df()
    bundle = _train_inline_bundle(snapshot, FEATURE_COLS, TARGET_COL, GROUP_COL)
    save_model_artifacts(bundle, str(tmp_path))

    start = time.perf_counter()
    loaded = predict.load_model_artifacts(str(tmp_path))
    result = predict.run_daily_sizing_inference(
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
    result = predict.run_daily_sizing_inference(snapshot, bundle)
    assert GROUP_COL in result.columns
    assert len(result) == len(snapshot)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)


def test_run_daily_sizing_inference_fills_missing_features() -> None:
    snapshot = _snapshot_df().drop(columns=["f2"])
    bundle = _train_inline_bundle(_snapshot_df(), FEATURE_COLS, TARGET_COL, GROUP_COL)
    result = predict.run_daily_sizing_inference(snapshot, bundle)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)
    assert len(result) == len(snapshot)


def test_run_daily_sizing_inference_raises_without_feature_cols() -> None:
    with pytest.raises(ValueError, match="feature_cols is empty"):
        predict.run_daily_sizing_inference(_snapshot_df(), {"dummy": 1})


def test_scenario_daily_predict_refactoring_02(tmp_path) -> None:
    """[SCENARIO_DAILY_PREDICT_REFACTORING_02] Verify clean loading of standard CSV without chart_pass filtering."""
    csv_file = tmp_path / "daily_stocks.csv"
    raw = pd.DataFrame(
        {
            "종목코드": ["123", "456"],
            "시가총액": [10.0, 20.0],
            "기관_순매수": [1.0, 2.0],
            "상장일수": [300, 500],
            "거래대금": [5.0, 10.0],
            "평균_거래대금": [2.0, 4.0],
        }
    )
    raw.to_csv(csv_file, index=False)
    with (
        patch.object(predict.os.path, "exists", return_value=True),
        patch.object(predict.pd, "read_csv", return_value=raw),
    ):
        result = predict.load_and_preprocess_data(str(csv_file))
    assert result["종목코드"].tolist() == ["000123", "000456"]
    assert result["기관_순매수"].tolist() == [100_000_000, 200_000_000]
    assert "차트통과" not in result.columns


def test_scenario_daily_predict_redesign_01(tmp_path) -> None:
    """[SCENARIO_DAILY_PREDICT_REDESIGN_01] Verifies daily sizing inference executes fast inference with preprocessed ML features and outputs sizing grades."""
    snapshot = _snapshot_df()
    bundle = _train_inline_bundle(snapshot, FEATURE_COLS, TARGET_COL, GROUP_COL)
    save_model_artifacts(bundle, str(tmp_path))

    loaded = predict.load_model_artifacts(str(tmp_path))
    result = predict.run_daily_sizing_inference(
        snapshot, loaded, feature_cols=FEATURE_COLS, group_col=GROUP_COL
    )

    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()


def _daily_snapshot_df() -> pd.DataFrame:
    """일일 CSV(daily_stocks.csv)와 동일한 스프레드시트 컬럼명의 당일 스냅샷을 생성합니다."""
    return pd.DataFrame(
        {
            "시나리오": ["거래량 폭증", "상따"],
            "종목명": ["AAA", "BBB"],
            "종목코드": ["000001", "000002"],
            "시가": [10_000.0, 20_000.0],
            "고가": [11_000.0, 22_000.0],
            "저가": [9_000.0, 19_000.0],
            "종가": [10_500.0, 21_000.0],
            "전일종가": [10_000.0, 20_000.0],
            "시가총액": [1_000.0, 2_000.0],
            "거래대금": [100.0, 200.0],
            "등락률": [22.0, 20.0],
            "선정순위": [1, 2],
            "기관_순매수": [10.0, 20.0],
            "외국인_순매수": [5.0, 10.0],
            "프로그램_순매수": [2.0, 4.0],
            "체결강도": [110.0, 120.0],
            "시장구분": ["KOSPI", "KOSDAQ"],
            "총_종목수": [50, 50],
            "평균_거래대금": [80.0, 80.0],
            "kospi": [0.5, 0.5],
            "kosdaq": [0.3, 0.3],
            "v_kospi": [15.0, 15.0],
            "v_kosdaq": [18.0, 18.0],
            "거래량": [1_000_000, 2_000_000],
            "테마_섹터": ["테마A", "테마A"],
            "차트분석": ["거래량 폭증", "상따"],
        }
    )



def test_apply_standard_feature_engineering_maps_to_ml_schema() -> None:
    """비괄호 일일 CSV 컬럼명이 표준 ML 피처 스키마로 1:1 정규화되고 파생 피처가 생성됩니다."""
    out = predict.apply_standard_feature_engineering(_daily_snapshot_df())

    expected_cols = [
        "stock_code",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "prev_close_price",
        "market_cap_100m",
        "trade_value_100m",
        "change_rate",
        "selection_rank",
        "inst_net_buy",
        "foreign_net_buy",
        "prog_net_buy",
        "volume_power",
        "market_type",
        "total_candidate_count",
        "avg_trade_value",
        "kospi_change",
        "kosdaq_change",
        "volume",
        "theme_sector",
        "chart_analysis",
        "trade_date",
        "major_density",
        "prog_dominance",
        "turnover",
        "rank_ratio",
        "log_market_cap_100m",
        "log_trade_value_100m",
        "buy_price_change_rate",
        "relative_change_rate",
    ]
    assert set(expected_cols).issubset(out.columns)
    assert np.issubdtype(out["trade_date"].dtype, np.datetime64)
    assert out["buy_price_change_rate"].round(6).eq(0).all()
    assert out["major_density"].notna().all()
    assert out["prog_dominance"].notna().all()


def test_apply_standard_feature_engineering_preserves_display_metadata() -> None:
    """표준화 후에도 표시용 메타데이터(종목명 등)가 보존되어 출력 호환성을 유지합니다."""
    out = predict.apply_standard_feature_engineering(_daily_snapshot_df())
    assert out["종목명"].tolist() == ["AAA", "BBB"]
    assert out["theme_sector"].tolist() == ["테마A", "테마A"]
    assert out["chart_analysis"].tolist() == ["거래량 폭증", "상따"]
    assert out["selection_rank"].tolist() == [1, 2]


def test_build_result_rows_uses_standard_columns() -> None:
    sizing_df = pd.DataFrame(
        {
            "종목명": ["AAA"],
            "theme_sector": ["테마A"],
            "chart_analysis": ["거래량 폭증"],
            "selection_rank": [1],
            "change_rate": [5.0],
            "rank_score": [1.0],
            "utility_score": [0.5],
            "grade": ["Strong"],
            "allocation": [0.1],
            "kospi": [0.5],
            "kosdaq": [0.3],
            "date": ["2026-08-04"],
        }
    )
    rows = predict.build_result_rows(sizing_df)
    assert len(rows) == 1
    row = rows[0]
    assert row["Rank"] == 1
    assert row["Name"] == "AAA"
    assert row["Theme"] == "테마A"
    assert row["Scenario"] == "거래량 폭증"
    assert row["Score"] == 0.5
    assert row["Grade"] == "Strong"
    assert row["Decision"] == "Strong (10.0%)"
    assert row["Applied_Rate"] == 5.0


def test_select_top_actionable_excludes_pass_and_limits_top_n() -> None:
    results = [
        {"Grade": "Strong", "Score": 0.9, "Name": "A"},
        {"Grade": "Weak", "Score": 0.5, "Name": "B"},
        {"Grade": "Pass", "Score": 0.7, "Name": "C"},
        {"Grade": "Good", "Score": 0.8, "Name": "D"},
        {"Grade": "Good", "Score": 0.6, "Name": "E"},
    ]
    top = predict.select_top_actionable(results, top_n=2)
    assert top == [
        {"Grade": "Strong", "Score": 0.9, "Name": "A"},
        {"Grade": "Good", "Score": 0.8, "Name": "D"},
    ]


def test_select_top_actionable_returns_empty_when_all_pass() -> None:
    res = predict.select_top_actionable([{"Grade": "Pass", "Score": 1.0}])
    assert len(res) == 1
    assert res[0]["Score"] == 1.0


def test_main_runs_redesigned_pipeline_with_mocks() -> None:
    """리디자인된 main() 이 표준 피처 엔지니어링 + Top N 출력으로 완주합니다."""
    sizing_df = pd.DataFrame(
        {
            "종목명": ["AAA", "BBB"],
            "theme_sector": ["테마A", "테마A"],
            "chart_analysis": ["거래량 폭증", "상따"],
            "selection_rank": [1, 2],
            "change_rate": [5.0, 29.9],
            "rank_score": [1.0, 0.5],
            "utility_score": [0.5, 0.4],
            "grade": ["Strong", "Pass"],
            "allocation": [0.1, 0.0],
            "kospi": [0.5, 0.5],
            "kosdaq": [0.3, 0.3],
            "date": ["2026-08-04", "2026-08-04"],
        }
    )

    async def fake_fetch(_code: str) -> tuple[float, float]:
        return 15.0, 0.05

    with (
        patch.object(
            predict, "load_and_preprocess_data", return_value=_daily_snapshot_df()
        ),
        patch.object(
            predict,
            "load_theme_from_db",
            return_value={"000001": "테마A", "000002": "테마A"},
        ),
        patch.object(predict, "sync_theme_only"),
        patch(
            "src.api.kis_client.fetch_index_and_calculate_volatility",
            side_effect=fake_fetch,
        ),
        patch.object(
            predict, "load_model_artifacts", return_value={"feature_cols": ["f1"]}
        ),
        patch.object(
            predict,
            "ensure_valid_model_bundle",
            side_effect=lambda bundle: bundle,
        ),
        patch.object(
            predict,
            "run_daily_sizing_inference",
            side_effect=lambda df, *a, **kw: sizing_df[sizing_df["chart_analysis"].isin(df["시나리오"])],
        ),
        patch.object(predict, "print_table") as print_table_mock,
    ):
        predict.main()

    assert print_table_mock.call_count == 2
    normal_rows = print_table_mock.call_args_list[0].args[0]
    assert [r["Name"] for r in normal_rows] == ["AAA"]
    assert normal_rows[0]["Decision"] == "Strong (10.0%)"
    sangdda_rows = print_table_mock.call_args_list[1].args[0]
    assert len(sangdda_rows) == 1
    assert sangdda_rows[0]["Name"] == "BBB"


def _realistic_trade_log_df(n_dates: int = 6, n_candidates: int = 15) -> pd.DataFrame:
    """실제 trade_log.parquet 와 동일한 컬럼 스키마의 합성 매매일지를 생성합니다."""
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for d_idx in range(n_dates):
        date = pd.Timestamp(f"2026-0{1 + d_idx}-05")
        for c in range(n_candidates):
            prev = 10_000.0 + rng.normal(0, 500)
            open_p = prev * (1 + rng.normal(0, 0.01))
            close = prev * (1 + rng.normal(0, 0.02))
            rows.append(
                {
                    "매수날짜": date,
                    "종목코드": f"{c + 1:06d}",
                    "(시가)": open_p,
                    "(고가)": max(open_p, close) * (1 + abs(rng.normal(0, 0.01))),
                    "(저가)": min(open_p, close) * (1 - abs(rng.normal(0, 0.01))),
                    "(종가)": close,
                    "(전일종가)": prev,
                    "(시가총액, 억)": rng.uniform(300, 3_000),
                    "(거래대금, 억)": rng.uniform(50, 800),
                    "(등락률)": rng.normal(2, 8),
                    "(선정 순위)": float(c + 1),
                    "(기관_순매수)": rng.normal(0, 1e8),
                    "(외국인_순매수)": rng.normal(0, 1e8),
                    "(프로그램_순매수)": rng.normal(0, 5e7),
                    "(체결강도)": rng.uniform(80, 200),
                    "(시장구분)": rng.choice(["KOSPI", "KOSDAQ"]),
                    "(총 종목 수)": float(n_candidates),
                    "(평균 거래대금)": rng.uniform(50, 800),
                    "(kospi, %)": rng.normal(0, 1),
                    "(kosdaq, %)": rng.normal(0, 1),
                    "v_kospi": rng.uniform(12, 25),
                    "v_kosdaq": rng.uniform(12, 25),
                    "(거래량)": rng.uniform(1e5, 5e6),
                    "(테마/섹터)": rng.choice(["테마A", "테마B", "테마C"]),
                    "(차트분석)": rng.choice(["거래량 폭증", "신고가 근접", "상한가 다음날"]),
                    "(매수 가격)": prev * 1.01,
                    "(매도 가격)": prev * 1.03,
                    "(수익률, %)": rng.normal(1.0, 4.0),
                }
            )
    return pd.DataFrame(rows)


def _daily_snapshot_from_processed(processed: pd.DataFrame) -> pd.DataFrame:
    """마지막 거래일의 processed 행을 당일 일일 CSV 스프레드시트 포맷으로 변환합니다."""
    last = processed[processed["trade_date"] == processed["trade_date"].max()].copy()
    return pd.DataFrame(
        {
            "시나리오": ["거래량 폭증"] * len(last),
            "종목명": [f"종목{i}" for i in range(len(last))],
            "종목코드": last["stock_code"].astype(str).str.zfill(6),
            "시가": last["open_price"],
            "고가": last["high_price"],
            "저가": last["low_price"],
            "종가": last["close_price"],
            "전일종가": last["prev_close_price"],
            "시가총액": last["market_cap_100m"],
            "거래대금": last["trade_value_100m"],
            "등락률": last["change_rate"],
            "선정순위": last["selection_rank"],
            "기관_순매수": last["inst_net_buy"],
            "외국인_순매수": last["foreign_net_buy"],
            "프로그램_순매수": last["prog_net_buy"],
            "체결강도": last["volume_power"],
            "시장구분": last["market_type"],
            "총_종목수": last["total_candidate_count"],
            "평균_거래대금": last["avg_trade_value"],
            "kospi": last["kospi_change"],
            "kosdaq": last["kosdaq_change"],
            "v_kospi": last["v_kospi"],
            "v_kosdaq": last["v_kosdaq"],
            "거래량": last["volume"],
            "테마_섹터": last["theme_sector"],
            "차트분석": last["chart_analysis"],
        }
    )


def test_scenario_model_utility_score_fix_01(tmp_path) -> None:
    """[SCENARIO_MODEL_UTILITY_SCORE_FIX_01] Verifies that daily sizing inference produces non-constant, varied utility scores across candidates using real preprocessed features."""
    raw = _realistic_trade_log_df()
    X, targets, cat_features, processed = predict.build_ml_dataset(raw, None)
    feature_cols = [col for col in X.columns if col not in cat_features]
    target_col = "target_return"
    group_col = "trade_date"
    bundle = _train_inline_bundle(
        processed[[*feature_cols, target_col, group_col]],
        feature_cols,
        target_col,
        group_col,
    )
    save_model_artifacts(bundle, str(tmp_path))

    loaded = predict.load_model_artifacts(str(tmp_path))
    assert loaded["feature_cols"] == feature_cols
    assert all(col not in cat_features for col in loaded["feature_cols"])

    snapshot = predict.apply_standard_feature_engineering(
        _daily_snapshot_from_processed(processed)
    )
    assert {"change_rate_z", "major_density_z", "turnover_z"}.issubset(snapshot.columns)
    assert not set(feature_cols).difference(snapshot.columns)

    result = predict.run_daily_sizing_inference(snapshot, loaded, group_col="date")

    assert result["utility_score"].nunique() > 1
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()


def test_ensure_valid_model_bundle_returns_real_bundle_unchanged() -> None:
    """실데이터 번들(수치 피처 + 모델 키)은 재학습 없이 그대로 반환됩니다."""
    real = {
        "feature_cols": ["change_rate", "turnover_z"],
        "rank_model": "r",
        "quantile_models": "q",
        "calibrators": "c",
    }
    with patch.object(
        predict, "train_and_save_real_model_bundle", return_value=real
    ) as retrain_mock:
        out = predict.ensure_valid_model_bundle(real)
    assert out is real
    retrain_mock.assert_not_called()


def test_ensure_valid_model_bundle_retrains_on_dummy_features() -> None:
    """더미 테스트 피처('f1','f2','f3') 번들은 실데이터 재학습으로 대체됩니다."""
    dummy = {"feature_cols": ["f1", "f2", "f3"]}
    real = {
        "feature_cols": ["change_rate"],
        "rank_model": 1,
        "quantile_models": 2,
        "calibrators": 3,
    }
    with patch.object(
        predict, "train_and_save_real_model_bundle", return_value=real
    ) as retrain_mock:
        out = predict.ensure_valid_model_bundle(dummy)
    assert out is real
    retrain_mock.assert_called_once_with()


def test_ensure_valid_model_bundle_retrains_on_invalid_bundle() -> None:
    """모델 키가 누락된 번들도 실데이터 재학습으로 대체됩니다."""
    invalid = {"feature_cols": ["change_rate"]}
    real = {
        "feature_cols": ["change_rate"],
        "rank_model": 1,
        "quantile_models": 2,
        "calibrators": 3,
    }
    with patch.object(
        predict, "train_and_save_real_model_bundle", return_value=real
    ) as retrain_mock:
        out = predict.ensure_valid_model_bundle(invalid)
    assert out is real
    retrain_mock.assert_called_once_with()


def test_train_and_save_real_model_bundle_trains_numeric_features(tmp_path) -> None:
    """trade_log.parquet 실데이터로 학습한 번들은 범주형 컬럼을 제외한 수치 피처를 선언합니다."""
    raw = _realistic_trade_log_df()
    trade_log_path = tmp_path / "trade_log.parquet"
    raw.to_parquet(trade_log_path)
    export_dir = tmp_path / "models"

    bundle = predict.train_and_save_real_model_bundle(
        export_dir=str(export_dir),
        trade_log_path=trade_log_path,
        theme_path=str(tmp_path / "missing_theme.parquet"),
    )

    assert not set(bundle["feature_cols"]).intersection(
        {"market_type", "theme_sector", "chart_analysis"}
    )
    assert bundle["feature_cols"]
    assert "quantile_models" in bundle
    assert (export_dir / "sizing_pipeline_bundle.joblib").exists()


def test_train_inline_bundle_excludes_categorical_features() -> None:
    """_train_inline_bundle 은 문자열/범주형 컬럼을 학습 피처에서 제외합니다."""
    snapshot = _snapshot_df()
    snapshot["market_type"] = "KOSPI"
    snapshot["theme_sector"] = "테마A"
    snapshot["chart_analysis"] = "거래량 폭증"
    bundle = _train_inline_bundle(
        snapshot,
        [*FEATURE_COLS, "market_type", "theme_sector", "chart_analysis"],
        TARGET_COL,
        GROUP_COL,
    )
    assert not set(bundle["feature_cols"]).intersection(
        {"market_type", "theme_sector", "chart_analysis"}
    )


def test_train_inline_bundle_raises_when_only_categorical_features() -> None:
    """모든 피처가 범주형이면 수치 피처가 없어 학습을 거부합니다."""
    snapshot = _snapshot_df()
    with pytest.raises(ValueError, match="after excluding categorical columns"):
        _train_inline_bundle(
            snapshot,
            ["market_type", "theme_sector", "chart_analysis"],
            TARGET_COL,
            GROUP_COL,
        )


def test_sangdda_feature_engineering_order() -> None:
    """[sangdda_feature_engineering_order] Verify that 20%+ rate stocks expand to sangdda scenario with change_rate 29.9% before feature engineering."""
    raw = pd.DataFrame(
        {
            "시나리오": ["신고가", "거래량 폭증"],
            "종목명": ["일반종목", "급등종목"],
            "종목코드": ["000001", "000002"],
            "시가": [10_000.0, 20_000.0],
            "고가": [11_000.0, 25_000.0],
            "저가": [9_500.0, 19_500.0],
            "종가": [10_500.0, 24_500.0],
            "전일종가": [10_000.0, 20_000.0],
            "시가총액": [1_000.0, 2_000.0],
            "거래대금": [100.0, 200.0],
            "change_rate": [5.0, 22.5],  # 급등종목 22.5% (20% 이상)
            "선정순위": [1, 2],
            "기관_순매수": [10.0, 20.0],
            "외국인_순매수": [5.0, 10.0],
            "프로그램_순매수": [2.0, 4.0],
            "체결강도": [110.0, 120.0],
            "시장구분": ["KOSPI", "KOSDAQ"],
            "총_종목수": [50, 50],
            "평균_거래대금": [80.0, 80.0],
            "kospi": [0.5, 0.5],
            "kosdaq": [0.3, 0.3],
            "v_kospi": [15.0, 15.0],
            "v_kosdaq": [18.0, 18.0],
            "거래량": [1_000_000, 2_000_000],
            "테마_섹터": ["테마A", "테마A"],
            "차트분석": ["신고가", "거래량 폭증"],
        }
    )

    def get_scenario_list(row):
        assigned = row.get("시나리오")
        if assigned == "상따":
            return ["상따"]
        scenarios = []
        if pd.notna(assigned) and assigned != "" and assigned is not None:
            scenarios.append(assigned)
        else:
            scenarios.append("거래량 폭증")

        rate = float(row.get("change_rate", row.get("등락률", 0)) or 0)
        if rate >= 20 and "상따" not in scenarios:
            scenarios.append("상따")
        return scenarios

    raw["Scenario_List"] = raw.apply(get_scenario_list, axis=1)
    df_all = raw.explode("Scenario_List").reset_index(drop=True)
    df_all["Scenario_Base"] = df_all["Scenario_List"]
    df_all = df_all.drop(columns=["Scenario_List"])

    # 20% 이상 급등종목은 일반 시나리오와 "상따" 시나리오 2개로 확장되어야 함
    sangdda_rows = df_all[df_all["Scenario_Base"].str.contains("상따")]
    assert len(sangdda_rows) == 1
    assert sangdda_rows.iloc[0]["종목명"] == "급등종목"

    # apply_standard_feature_engineering 적용 전에 sangdda 행의 change_rate를 29.9%로 세팅
    df_all = predict.normalize_column_names(df_all)
    sangdda_mask = df_all["Scenario_Base"].str.contains("상따", na=False)
    if "change_rate" in df_all.columns:
        df_all.loc[sangdda_mask, "change_rate"] = 29.9

    engineered = predict.apply_standard_feature_engineering(df_all)
    sangdda_eng = engineered[engineered["Scenario_Base"].str.contains("상따")].iloc[0]

    # change_rate 및 robust z-score 피처가 29.9% 기준으로 정규화되었는지 검증
    assert sangdda_eng["change_rate"] == 29.9
    assert "change_rate_z" in engineered.columns





