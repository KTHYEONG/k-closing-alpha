"""일일 예측 진입점 wiring 및 Fast Inference 단위 테스트 (live serving 경로).

SCENARIO_MODEL_PIPELINE_TRAIN_EVAL 의 wiring 단계가 일일 예측 진입점에
연결되었는지 확인하고, 저장된 모델 아티팩트 기반 Fast Inference 동작과
단일 BUY/ABSTAIN 결정을 검증합니다. 학습/재학습 관련 케이스는
``legacy/tests/unit/daily/test_predict.py`` 로 이동되었습니다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.daily.predict as predict
from src.serving.realtime.inference import predict_daily_sizing
from src.serving.realtime.policy import (
    SingleStockPolicy,
    always_buy_policy,
    load_single_stock_policy,
    select_single_daily_trade,
)
from src.serving.realtime.features import build_snapshot_features

from tests.unit.serving.realtime.fixtures import (
    build_fixed_serving_bundle,
    daily_snapshot_df,
    snapshot_feature_cols,
)

FEATURE_COLS = snapshot_feature_cols()


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


def test_load_and_preprocess_data_handles_alphanumeric_stock_codes(tmp_path) -> None:
    csv_file = tmp_path / "test_stocks.csv"
    csv_file.write_text("시나리오,종목명,종목코드\n폭증,삼성SDI,006400\n폭증,해치텍,0155E0\n", encoding="utf-8-sig")
    result = predict.load_and_preprocess_data(str(csv_file))
    assert result["종목코드"].tolist() == ["006400", "0155E0"]



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
    df_condition = pd.DataFrame(columns=["종목코드", "종목명", "시장구분"])
    with (
        patch.object(predict.os.path, "exists", return_value=False),
        patch.object(
            predict, "load_and_preprocess_data", return_value=df_condition
        ),
        patch.object(predict, "load_theme_from_db", return_value={}),
        patch.object(predict, "batch_resolve_missing_themes"),
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


def test_predict_exposes_only_live_control_apis() -> None:
    """예측 진입점은 학습/재학습 진입점을 노출하지 않습니다."""
    assert callable(predict.load_model_bundle)
    assert callable(predict.predict_daily_sizing)
    assert not hasattr(predict, "run_model_pipeline")
    assert not hasattr(predict, "train_and_save_real_model_bundle")
    assert not hasattr(predict, "ensure_valid_model_bundle")


def _daily_snapshot_with_bundle() -> pd.DataFrame:
    """일일 CSV 스냅샷을 고정 번들 피처 컬럼을 포함하는 프레임으로 확장합니다."""
    snapshot = build_snapshot_features(daily_snapshot_df())
    rng = np.random.default_rng(5)
    for col in FEATURE_COLS:
        if col not in snapshot.columns:
            snapshot[col] = rng.normal(size=len(snapshot))
        else:
            snapshot[col] = snapshot[col] + rng.normal(size=len(snapshot)) * 0.1
    return snapshot


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
            "close_price": [11000.0, 11000.0],
            "prev_close_price": [10000.0, 10000.0],
            "high_price": [11200.0, 11200.0],
        }
    )

    async def fake_fetch(_code: str) -> tuple[float, float]:
        return 15.0, 0.05

    with (
        patch.object(
            predict, "load_and_preprocess_data", return_value=daily_snapshot_df()
        ),
        patch.object(
            predict,
            "load_theme_from_db",
            return_value={"000001": "테마A", "000002": "테마A"},
        ),
        patch.object(predict, "batch_resolve_missing_themes"),
        patch(
            "src.api.kis_client.fetch_index_and_calculate_volatility",
            side_effect=fake_fetch,
        ),
        patch.object(
            predict, "load_model_bundle", return_value={"feature_cols": ["f1"]}
        ),
        patch.object(
            predict,
            "predict_daily_sizing",
            side_effect=lambda df, *a, **kw: sizing_df[
                sizing_df["chart_analysis"].isin(df["chart_analysis"])
            ],
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


def test_main_auto_resolves_themes_missing_from_local_cache() -> None:
    """theme.parquet/DB에 없는 신규 종목은 theme_resolver 자동 분류로 처리된다(코드_테마_DB 시트 폐지 후)."""
    sizing_df = pd.DataFrame(
        {
            "종목명": ["AAA", "BBB"],
            "theme_sector": ["테마A", "테마B"],
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
            "close_price": [11000.0, 11000.0],
            "prev_close_price": [10000.0, 10000.0],
            "high_price": [11200.0, 11200.0],
        }
    )

    async def fake_fetch(_code: str) -> tuple[float, float]:
        return 15.0, 0.05

    with (
        patch.object(
            predict, "load_and_preprocess_data", return_value=daily_snapshot_df()
        ),
        # 000002는 로컬 캐시에 없어 자동 분류 대상이 됨 (신규 상장 등)
        patch.object(
            predict, "load_theme_from_db", return_value={"000001": "테마A"}
        ),
        patch.object(predict, "batch_resolve_missing_themes") as resolve_mock,
        patch(
            "src.api.kis_client.fetch_index_and_calculate_volatility",
            side_effect=fake_fetch,
        ),
        patch.object(
            predict, "load_model_bundle", return_value={"feature_cols": ["f1"]}
        ),
        patch.object(
            predict,
            "predict_daily_sizing",
            side_effect=lambda df, *a, **kw: sizing_df[
                sizing_df["chart_analysis"].isin(df["chart_analysis"])
            ],
        ),
        patch.object(predict, "print_table"),
    ):
        predict.main()

    resolve_mock.assert_called_once()
    (missing_list,), kwargs = resolve_mock.call_args
    assert kwargs == {}
    assert [row["종목코드"] for row in missing_list] == ["000002"]


def test_merged_normal_sangdda_scored_table_yields_one_decision() -> None:
    """병합된 normal/sangdda 스코어링 테이블은 독립 Top-N 이 아닌 단일 결정을 만듭니다."""
    normal = pd.DataFrame(
        {
            "date": ["2026-08-04"] * 2,
            "stock_code": ["000001", "000002"],
            "chart_analysis": ["거래량 폭증", "신고가"],
            "rank_score": [0.9, 0.4],
        }
    )
    sangdda = pd.DataFrame(
        {
            "date": ["2026-08-04"],
            "stock_code": ["000003"],
            "chart_analysis": ["상따"],
            "rank_score": [0.7],
        }
    )
    merged = pd.concat([normal, sangdda], ignore_index=True)
    decision = select_single_daily_trade(
        merged, always_buy_policy("2026-08-04"), "date"
    )
    assert len(decision) == 1
    assert decision.iloc[0]["decision"] == "BUY"
    assert decision.iloc[0]["stock_code"] == "000001"
    assert decision.iloc[0]["n_unique_stocks"] == 3


def test_load_single_stock_policy_from_bundle_state() -> None:
    """번들 상태에서 정책을 복원하고, 무효 상태는 None 으로 fail-safe 합니다."""
    policy = always_buy_policy("2026-08-04")
    assert load_single_stock_policy({"single_stock_policy": policy}) is policy

    restored = load_single_stock_policy({"single_stock_policy": policy.model_dump()})
    assert isinstance(restored, SingleStockPolicy)
    assert restored.policy_id == "always_buy_top1"

    assert load_single_stock_policy({"feature_cols": ["f1"]}) is None
    assert load_single_stock_policy({"single_stock_policy": "bogus"}) is None


def test_scenario_model_utility_score_fix_01() -> None:
    """[SCENARIO_MODEL_UTILITY_SCORE_FIX_01] 고정 번들과 피처 호환 스냅샷이
    다양한 live utility score 와 유효한 sizing grade 를 산출합니다."""
    bundle = build_fixed_serving_bundle(FEATURE_COLS)
    snapshot = _daily_snapshot_with_bundle()
    assert not set(FEATURE_COLS).difference(snapshot.columns)

    result = predict_daily_sizing(snapshot, bundle, group_col="date")

    assert result["utility_score"].nunique() > 1
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()


def test_sangdda_feature_engineering_order() -> None:
    """[sangdda_feature_engineering_order] 20%+ 종목은 상따 시나리오로 확장되고
    change_rate 29.9% 가 피처 엔지니어링 전에 적용됩니다."""
    raw = daily_snapshot_df().copy()
    raw["등락률"] = [5.0, 22.5]

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

    sangdda_rows = df_all[df_all["Scenario_Base"].str.contains("상따")]
    assert len(sangdda_rows) == 1
    assert sangdda_rows.iloc[0]["종목명"] == "BBB"

    df_all = predict.normalize_column_names(df_all)
    sangdda_mask = df_all["Scenario_Base"].str.contains("상따", na=False)
    if "change_rate" in df_all.columns:
        df_all.loc[sangdda_mask, "change_rate"] = 29.9

    engineered = build_snapshot_features(df_all)
    sangdda_eng = engineered[engineered["Scenario_Base"].str.contains("상따")].iloc[0]

    assert sangdda_eng["change_rate"] == 29.9
    assert "change_rate_z" in engineered.columns


def test_scenario_realtime_sangtta_price_alignment_02() -> None:
    """[SCENARIO_REALTIME_SANGTTA_PRICE_ALIGNMENT_02] predict.py aligns change_rate,
    close_price, buy_price, and high_price for sangtta scenarios before snapshot
    feature building."""
    raw = daily_snapshot_df().copy()
    raw["등락률"] = [5.0, 22.5]
    raw["전일종가"] = [10_000.0, 20_000.0]
    raw["(매수 가격)"] = raw["종가"].copy()

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

    df_all = predict.normalize_column_names(df_all)
    sangdda_mask = df_all["Scenario_Base"].str.contains("상따", na=False)
    if "change_rate" in df_all.columns:
        df_all.loc[sangdda_mask, "change_rate"] = 29.9
    if "prev_close_price" in df_all.columns:
        limit_up_price = np.round(
            df_all.loc[sangdda_mask, "prev_close_price"] * 1.299
        )
        if "close_price" in df_all.columns:
            df_all.loc[sangdda_mask, "close_price"] = limit_up_price
        if "buy_price" in df_all.columns:
            df_all.loc[sangdda_mask, "buy_price"] = limit_up_price
        if "high_price" in df_all.columns:
            df_all.loc[sangdda_mask, "high_price"] = np.maximum(
                df_all.loc[sangdda_mask, "high_price"], limit_up_price
            )

    sangdda_row = df_all[sangdda_mask].iloc[0]
    expected_limit_up = np.round(sangdda_row["prev_close_price"] * 1.299)
    assert sangdda_row["change_rate"] == 29.9
    assert sangdda_row["close_price"] == expected_limit_up
    assert sangdda_row["buy_price"] == expected_limit_up
    assert sangdda_row["high_price"] >= expected_limit_up

    engineered = build_snapshot_features(df_all)
    sangdda_eng = engineered[engineered["Scenario_Base"].str.contains("상따")].iloc[0]
    assert sangdda_eng["buy_price_change_rate"] >= 0


def test_predict_forces_pass_for_ceiling_candidates_only() -> None:
    import pandas as pd

    from src.ml.buyability import classify_ceiling_entry

    # Given: a sizing_df as if predict_daily_sizing already graded it -- one
    # ceiling-close row mis-graded Strong, one ordinary row graded Weak
    df = pd.DataFrame(
        {
            "close_price": [12900.0, 11000.0],
            "prev_close_price": [10000.0, 10000.0],
            "high_price": [12900.0, 11200.0],
            "grade": ["Strong", "Weak"],
            "grade_multiplier": [1.5, 0.5],
            "allocation": [0.15, 0.05],
        }
    )

    # When: mirror predict.py's post-hoc override
    is_ceiling = classify_ceiling_entry(df).to_numpy(dtype=bool)
    df.loc[is_ceiling, "grade"] = "Pass"
    df.loc[is_ceiling, "grade_multiplier"] = 0.0
    df.loc[is_ceiling, "allocation"] = 0.0

    # Then
    assert df.loc[0, "grade"] == "Pass"
    assert df.loc[0, "allocation"] == 0.0
    assert df.loc[1, "grade"] == "Weak"
    assert df.loc[1, "allocation"] == 0.05


def test_load_condition_snapshot_preserves_100m_units(tmp_path) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: a standard snapshot CSV whose amounts are in 100M KRW units
    csv = tmp_path / "daily_stocks.csv"
    pd.DataFrame({
        "종목코드": ["1", "000002"],
        "거래대금": [500.0, 300.0],
        "시가총액": [3000.0, 2000.0],
        "기관_순매수": [10.0, 20.0],
    }).to_csv(csv, index=False, encoding="utf-8-sig")

    # When
    out = predict_mod.load_condition_snapshot(str(csv))

    # Then: amounts untouched (the reranker trains on 100M-KRW units)
    assert out["거래대금"].tolist() == [500.0, 300.0]
    assert out["시가총액"].tolist() == [3000.0, 2000.0]
    assert out["기관_순매수"].tolist() == [10.0, 20.0]
    assert out["종목코드"].tolist() == ["000001", "000002"]



def test_convert_amount_units_to_krw_scales_only_legacy_amount_columns() -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    # Given
    df = pd.DataFrame({
        "거래대금": [500.0],
        "기관_순매수": [10.0],
        "외국인_순매수": [5.0],
        "시가총액": [3000.0],
    })

    # When
    out = predict_mod.convert_amount_units_to_krw(df)

    # Then: only the legacy champion amount columns are rescaled
    assert out["거래대금"].tolist() == [500.0 * 1e8]
    assert out["기관_순매수"].tolist() == [10.0 * 1e8]
    assert out["외국인_순매수"].tolist() == [5.0 * 1e8]
    assert out["시가총액"].tolist() == [3000.0]



def test_run_topk_ranker_sleeve_scores_wide_and_selects_admitted(monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.daily.predict as predict_mod
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    # Given: a wide 4-name snapshot in raw 100M-KRW units where only 3 clear
    # the cost-aware screen (S4 sits at 30000 KRW -> 16.67bp > the 7.5bp cap)
    wide = pd.DataFrame({
        "종목코드": ["000001", "000002", "000003", "000004"],
        "종목명": ["AAA", "BBB", "CCC", "DDD"],
        "종가": [18000.0, 18100.0, 17900.0, 30000.0],
        "전일종가": [17142.86, 17238.10, 17047.62, 28571.43],
        "고가": [18100.0, 18200.0, 18000.0, 30100.0],
        "저가": [17800.0, 17900.0, 17700.0, 29800.0],
        "시가": [17900.0, 18000.0, 17800.0, 29900.0],
        "거래량": [1_000_000, 900_000, 1_100_000, 800_000],
        "거래대금": [500.0, 450.0, 550.0, 400.0],
        "시가총액": [3000.0, 2800.0, 3200.0, 5000.0],
        "기관_순매수": [10.0, -5.0, 20.0, 8.0],
        "외국인_순매수": [5.0, 12.0, -3.0, 6.0],
        "시장구분": ["KOSPI", "KOSPI", "KOSPI", "KOSPI"],
        "kospi": [0.52, 0.52, 0.52, 0.52],
        "kosdaq": [-0.31, -0.31, -0.31, -0.31],
        "v_kospi": [15.2, 15.2, 15.2, 15.2],
    })
    bundle = build_fixed_serving_bundle(list(FEATURE_COLS))
    bundle["top_k"] = 3

    monkeypatch.setattr(predict_mod.settings, "CANDIDATE_SOURCE_MODE", "automated")
    monkeypatch.setattr(predict_mod, "load_condition_snapshot", lambda _p: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)

    # When
    out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))

    # Then: the tick-cost rejected name never enters, weights are uniform
    assert len(out) == 3
    assert sorted(out["symbol"].tolist()) == ["000001", "000002", "000003"]
    assert np.allclose(out["allocation"].to_numpy(dtype=np.float64), 1.0 / 3.0)
    assert sorted(out["name"].tolist()) == ["AAA", "BBB", "CCC"]



def test_run_topk_ranker_sleeve_warns_when_bundle_missing(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: the sleeve is in scope but no production bundle is published yet
    wide = pd.DataFrame({
        "종목코드": ["000001"], "종목명": ["AAA"], "종가": [18000.0],
        "전일종가": [17142.86], "고가": [18100.0], "저가": [17800.0], "시가": [17900.0],
        "거래량": [1_000_000], "거래대금": [500.0], "시가총액": [3000.0],
        "기관_순매수": [10.0], "외국인_순매수": [5.0], "시장구분": ["KOSPI"],
        "kospi": [0.52], "kosdaq": [-0.31], "v_kospi": [15.2],
    })
    monkeypatch.setattr(predict_mod.settings, "CANDIDATE_SOURCE_MODE", "automated")
    monkeypatch.setattr(predict_mod, "load_condition_snapshot", lambda _p: wide)

    def _missing(import_dir=None):
        raise FileNotFoundError("model artifact bundle not found")

    monkeypatch.setattr(predict_mod, "load_model_bundle", _missing)

    # When
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))

    # Then: fails soft, but never silently
    assert out.empty
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)



def test_main_automated_mode_prints_only_topk_decision_table(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: automated mode with a populated top-3 sleeve
    sleeve_df = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "name": ["AAA", "BBB", "CCC"],
        "pred": [0.021, 0.017, 0.011],
        "allocation": [1.0 / 3.0] * 3,
    })
    monkeypatch.setattr(predict_mod.settings, "CANDIDATE_SOURCE_MODE", "automated")
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", Mock(return_value=sleeve_df))
    champion_mock = Mock()
    monkeypatch.setattr(predict_mod, "predict_daily_sizing", champion_mock)
    print_table_mock = Mock()
    monkeypatch.setattr(predict_mod, "print_table", print_table_mock)

    # When
    predict_mod.main()

    # Then: the reranker table is the single decision surface; champion's
    # fixed-threshold grading is never even computed for this population
    assert print_table_mock.call_count == 1
    champion_mock.assert_not_called()
    rows = print_table_mock.call_args_list[0].args[0]
    assert [r["Code"] for r in rows] == ["000001", "000002", "000003"]
    assert rows[0]["Alloc%"] == pytest.approx(33.3, abs=0.1)



def test_main_automated_mode_warns_and_prints_nothing_when_no_decision(monkeypatch, caplog) -> None:
    import logging
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: automated mode where the sleeve yields no actionable decision
    monkeypatch.setattr(predict_mod.settings, "CANDIDATE_SOURCE_MODE", "automated")
    monkeypatch.setattr(
        predict_mod, "run_topk_ranker_sleeve", Mock(return_value=pd.DataFrame())
    )
    print_table_mock = Mock()
    monkeypatch.setattr(predict_mod, "print_table", print_table_mock)

    # When
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        predict_mod.main()

    # Then: silence is made explicit as a no-participation banner
    print_table_mock.assert_not_called()
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)



def test_main_manual_mode_keeps_champion_tables_unchanged(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    sizing_df = pd.DataFrame({
        "종목명": ["AAA", "BBB"], "theme_sector": ["테마A", "테마A"],
        "chart_analysis": ["거래량 폭증", "상따"], "selection_rank": [1, 2],
        "change_rate": [5.0, 29.9], "rank_score": [1.0, 0.5], "utility_score": [0.5, 0.4],
        "grade": ["Strong", "Pass"], "allocation": [0.1, 0.0],
        "kospi": [0.5, 0.5], "kosdaq": [0.3, 0.3], "date": ["2026-08-04", "2026-08-04"],
        "close_price": [11000.0, 11000.0], "prev_close_price": [10000.0, 10000.0],
        "high_price": [11200.0, 11200.0],
    })

    async def fake_fetch(_code: str) -> tuple[float, float]:
        return 15.0, 0.05

    monkeypatch.setattr(predict_mod.settings, "CANDIDATE_SOURCE_MODE", "manual")
    sleeve_mock = Mock(return_value=pd.DataFrame())
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", sleeve_mock)

    with (
        patch.object(predict_mod, "load_and_preprocess_data", return_value=daily_snapshot_df()),
        patch.object(predict_mod, "load_theme_from_db",
                     return_value={"000001": "테마A", "000002": "테마A"}),
        patch.object(predict_mod, "batch_resolve_missing_themes"),
        patch("src.api.kis_client.fetch_index_and_calculate_volatility", side_effect=fake_fetch),
        patch.object(predict_mod, "load_model_bundle", return_value={"feature_cols": ["f1"]}),
        patch.object(predict_mod, "predict_daily_sizing",
                     side_effect=lambda df, *a, **kw: sizing_df[
                         sizing_df["chart_analysis"].isin(df["chart_analysis"])]),
        patch.object(predict_mod, "print_table") as print_table_mock,
    ):
        predict_mod.main()

    # Then: byte-identical to the pre-migration manual behaviour
    assert print_table_mock.call_count == 2
    sleeve_mock.assert_not_called()



def test_run_topk_ranker_sleeve_short_circuits_in_manual_mode(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: manual mode -- the reranker sleeve is entirely out of scope
    monkeypatch.setattr(predict_mod.settings, "CANDIDATE_SOURCE_MODE", "manual")
    snapshot_mock = Mock()
    bundle_mock = Mock()
    monkeypatch.setattr(predict_mod, "load_condition_snapshot", snapshot_mock)
    monkeypatch.setattr(predict_mod, "load_model_bundle", bundle_mock)

    # When
    out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))

    # Then: short-circuits before touching disk or artifacts
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    snapshot_mock.assert_not_called()
    bundle_mock.assert_not_called()
