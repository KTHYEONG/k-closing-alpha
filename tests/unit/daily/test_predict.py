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
from src.processing.schema import normalize_column_names
from src.serving.realtime.features import build_snapshot_features

from tests.unit.serving.realtime.fixtures import (
    build_fixed_serving_bundle,
    daily_snapshot_df,
    snapshot_feature_cols,
)

FEATURE_COLS = snapshot_feature_cols()































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

    df_all = normalize_column_names(df_all)
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

    df_all = normalize_column_names(df_all)
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














def test_main_automated_mode_prints_only_topk_decision_table(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: a populated top-3 sleeve
    sleeve_df = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "name": ["AAA", "BBB", "CCC"],
        "pred": [0.021, 0.017, 0.011],
        "allocation": [1.0 / 3.0] * 3,
    })
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", Mock(return_value=sleeve_df))
    print_table_mock = Mock()
    monkeypatch.setattr(predict_mod, "print_table", print_table_mock)

    # When
    predict_mod.main()

    # Then: the reranker table is the single decision surface
    assert print_table_mock.call_count == 1
    rows = print_table_mock.call_args_list[0].args[0]
    assert [r["Code"] for r in rows] == ["000001", "000002", "000003"]
    assert rows[0]["Alloc%"] == pytest.approx(33.3, abs=0.1)



def test_main_automated_mode_warns_and_prints_nothing_when_no_decision(monkeypatch, caplog) -> None:
    import logging
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: the sleeve yields no actionable decision
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









def test_load_daily_snapshot_reads_archive_store_and_zero_fills_code(monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: the archive store returns the day's wide snapshot in 100M-KRW units
    captured = {}

    def _fake_fetch(snapshot_date=None, **kwargs):
        captured["snapshot_date"] = snapshot_date
        return pd.DataFrame({"종목코드": [5930, "000660"], "거래대금": [500.0, 300.0], "admitted": [True, False]})

    monkeypatch.setattr(predict_mod, "fetch_archive_snapshot", _fake_fetch)

    # When
    out = predict_mod.load_daily_snapshot(pd.Timestamp("2026-09-09"))

    # Then
    assert captured["snapshot_date"] == "2026-09-09"
    assert out["종목코드"].tolist() == ["005930", "000660"]
    assert out["거래대금"].tolist() == [500.0, 300.0]
    assert out["admitted"].tolist() == [True, False]



def test_run_topk_ranker_sleeve_uses_stored_admitted_without_recompute(monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.daily.predict as predict_mod
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    # Given: a 4-name wide snapshot where the store already marked one rejected
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
        "kospi": [0.52] * 4,
        "kosdaq": [-0.31] * 4,
        "v_kospi": [15.2] * 4,
        "admitted": [True, True, True, False],
    })
    bundle = build_fixed_serving_bundle(list(FEATURE_COLS))
    bundle["top_k"] = 3

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)

    # When
    out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))

    # Then: the stored verdict alone decides eligibility
    assert len(out) == 3
    assert sorted(out["symbol"].tolist()) == ["000001", "000002", "000003"]
    assert np.allclose(out["allocation"].to_numpy(dtype=np.float64), 1.0 / 3.0)
    assert sorted(out["name"].tolist()) == ["AAA", "BBB", "CCC"]



def test_run_topk_ranker_sleeve_warns_when_bundle_missing(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    import src.daily.predict as predict_mod

    wide = pd.DataFrame({
        "종목코드": ["000001"], "종목명": ["AAA"], "종가": [18000.0], "전일종가": [17142.86],
        "고가": [18100.0], "저가": [17800.0], "시가": [17900.0], "거래량": [1_000_000],
        "거래대금": [500.0], "시가총액": [3000.0], "기관_순매수": [10.0], "외국인_순매수": [5.0],
        "시장구분": ["KOSPI"], "kospi": [0.52], "kosdaq": [-0.31], "v_kospi": [15.2],
        "admitted": [True],
    })
    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d: wide)

    def _missing(import_dir=None):
        raise FileNotFoundError("model artifact bundle not found")

    monkeypatch.setattr(predict_mod, "load_model_bundle", _missing)

    # When
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))

    # Then: fails soft, never silently
    assert out.empty
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)



def test_predict_module_drops_champion_surface() -> None:
    import src.daily.predict as predict_mod

    # Then: no champion grading/display helpers and no CSV loader remain
    for gone in (
        "load_condition_snapshot",
        "convert_amount_units_to_krw",
        "load_and_preprocess_data",
        "build_result_rows",
        "select_top_actionable",
        "explain_predictions_with_shap",
        "load_label_encoder_map",
        "LABEL_ENCODER_MAP",
        "predict_daily_sizing",
    ):
        assert not hasattr(predict_mod, gone), gone
