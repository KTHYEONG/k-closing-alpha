"""preprocessor 데이터 복구/ML 전처리 파이프라인 단위 테스트.

`docs/specs/preprocessor_data_recovery_contract.json`,
`docs/specs/ml_data_preprocessing_contract.json`,
`docs/specs/preprocessor_refactor_contract.json`의 시나리오 기반 검증입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.preprocessor import (
    _ROBUST_Z_COLUMNS,
    _apply_robust_z,
    _validate_close_morning61_feature,
    build_ml_dataset,
    clean_column_names,
    create_multi_targets,
    engineer_features,
)


def test_scenario_preprocessor_percent_cleaning() -> None:
    """SCENARIO_PREPROCESSOR_PERCENT_CLEANING: %와 , 기호가 제거되어 수치 변환 후 전체 행이 보존됩니다.

    trade_log.parquet의 33,934건 중 약 92.4%(31,352건)가 `'5.95%'` 형태의
    퍼센트 문자열로 저장되어 있어, 정제 없이는 pd.to_numeric이 NaN으로 강제 변환되어
    dropna(subset=["net_return"]) 단계에서 대량 유실됩니다. % 기호 제거 후
    net_return 전 행이 유효한 수치로 복구되는지 검증합니다.
    """
    df = pd.DataFrame(
        {
            "net_return": ["5.95%", "-1.96%", "1,234.50%", "0.00%", "10.1%"],
            "trade_date": pd.to_datetime(["2024-01-02"] * 5),
        }
    )
    cleaned = clean_column_names(df)
    assert len(cleaned) == 5
    assert cleaned["net_return"].notna().all()
    assert cleaned["net_return"].iloc[0] == 5.95
    assert cleaned["net_return"].iloc[1] == -1.96
    assert cleaned["net_return"].iloc[2] == 1234.50


def test_percent_cleaning_full_dataset_recovery() -> None:
    """퍼센트 문자열 net_return 전량이 NaN 없이 복구되어 100% 행 보존을 보장합니다.

    실제 trade_log.parquet의 분포(33,934행 중 31,352행이 '%' 문자열)를 반영해
    전체 행이 퍼센트 문자열인 경우에도 유실이 발생하지 않음을 검증합니다.
    """
    rng = np.random.default_rng(7)
    n_rows = 200
    df = pd.DataFrame(
        {
            "net_return": [f"{float(v):.2f}%" for v in rng.normal(0.0, 2.0, n_rows)],
            "trade_date": pd.to_datetime(
                [f"2024-0{1 + i % 9}-0{1 + (i * 7) % 9}" for i in range(n_rows)]
            ),
        }
    )
    cleaned = clean_column_names(df.copy())
    assert cleaned["net_return"].notna().all()
    assert len(cleaned.dropna(subset=["net_return"])) == n_rows
    assert cleaned["net_return"].dtype.kind == "f"


def _build_raw_df(n_per_date: list[int] | None = None) -> pd.DataFrame:
    """괄호/단위 특수문자가 포함된 스프레드시트 헤더 형태의 원본 DataFrame을 생성합니다.

    ``n_per_date``로 거래일별 종목 수를 조정해 그룹 크기 엣지케이스 검증을 지원합니다.
    """
    rng = np.random.default_rng(42)
    if n_per_date is None:
        n_per_date = [8, 8, 3]
    dates = pd.to_datetime([f"2024-01-0{i + 2}" for i in range(len(n_per_date))])
    rows: list[dict[str, object]] = []
    for date, n_stocks in zip(dates, n_per_date, strict=True):
        for i in range(n_stocks):
            code = f"{i:06d}"
            base = 10_000 + i * 1_000
            change = float(rng.normal(0.5, 2.0))
            net_return = float(rng.normal(0.2, 1.2))
            rows.append(
                {
                    "매수날짜": date,
                    "종목코드": code,
                    "(시가)": base,
                    "(고가)": int(base * 1.05),
                    "(저가)": int(base * 0.97),
                    "(종가)": int(base * (1 + change / 100)),
                    "(전일종가)": int(base * 0.99),
                    "(시가총액, 억)": 1_000.0 + i * 100,
                    "(거래대금, 억)": 100.0 + i * 20,
                    "(등락률)": change,
                    "(선정 순위)": float(i + 1),
                    "(기관_순매수)": float((i - 2) * 50),
                    "(외국인_순매수)": float(i * 30),
                    "(프로그램_순매수)": float((i - 1) * 10),
                    "(체결강도)": 110.0 + i,
                    "(시장구분)": "KOSPI" if i % 2 == 0 else "KOSDAQ",
                    "(총 종목 수)": float(n_stocks),
                    "(평균 거래대금)": 90.0,
                    "(kospi, %)": 0.3,
                    "(kosdaq, %)": 0.1,
                    "v_kospi": 15.0,
                    "v_kosdaq": 18.0,
                    "(거래량)": 1_000_000 + i * 1_000,
                    "(테마/섹터)": f"theme{i % 3}",
                    "(차트분석)": "신고가 근접",
                    "(매수 가격)": float(int(base * (1 + change / 100)) * 0.99),
                    "(매도 가격)": float(int(base * (1 + change / 100)) * 1.02),
                    "(수익률, %)": f"{net_return:.4f}",
                }
            )
    return pd.DataFrame(rows)


def test_clean_column_mapping() -> None:
    """괄호/단위 특수문자 컬럼명이 snake_case 식별자로 정규화되는지 검증합니다."""
    cleaned = clean_column_names(_build_raw_df())
    assert "market_cap_100m" in cleaned.columns
    assert "trade_value_100m" in cleaned.columns
    assert "trade_date" in cleaned.columns
    assert np.issubdtype(cleaned["trade_date"].dtype, np.datetime64)
    assert cleaned["stock_code"].str.len().eq(6).all()
    assert cleaned["net_return"].dtype.kind == "f"


def test_multi_target_generation() -> None:
    """3종 타깃이 NaN 없이 생성되는지 검증합니다."""
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    assert len(X) == len(processed)
    assert targets["target_rank"].between(0, 4).all()
    assert targets["target_rank"].dtype.kind == "i"
    assert targets["target_return"].between(-0.10, 0.10).all()
    assert set(targets["target_good"].unique()).issubset({0, 1})
    assert set(targets["target_bad"].unique()).issubset({0, 1})
    for series in targets.values():
        assert series.notna().all()
        assert len(series) == len(processed)


def test_target_return_clipped() -> None:
    """target_return이 decimal net 단위로 ±10% 소수 클리핑되고, 라벨이 동일 단위 임계값에서 파생됩니다.

    SCENARIO: test_target_return_clipped — Targets and classifier labels share
    decimal-net units and one cost deduction.
    """
    df = _build_raw_df()
    df.loc[df.index[0], "(수익률, %)"] = "25.0"
    df.loc[df.index[1], "(수익률, %)"] = "-15.0"
    _, targets, _, _ = build_ml_dataset(df)
    assert targets["target_return"].max() <= 0.10
    assert targets["target_return"].min() >= -0.10


def test_target_return_decimal_net_single_cost_deduction() -> None:
    """원본 퍼센트 수익률이 정확히 1회 decimal 변환되고 비용이 정확히 1회 차감됩니다.

    - 1.0% gross -> decimal net = 0.01 - 0.002 = 0.008 (percentage-point 혼용 금지)
    - 2.0% gross -> 0.018 >= +1% 임계값 -> target_good = 1
    - -3.0% gross -> -0.032 <= -2% 임계값 -> target_bad = 1
    """
    from src.ml.sizing_engine import ROUND_TRIP_COST_RATIO

    df = _build_raw_df(n_per_date=[1])
    df["(수익률, %)"] = "1.0"
    _, targets, _, _ = build_ml_dataset(df)
    assert targets["target_return"].iloc[0] == pytest.approx(0.01 - ROUND_TRIP_COST_RATIO)

    df = _build_raw_df(n_per_date=[1])
    df["(수익률, %)"] = "2.0"
    _, targets, _, _ = build_ml_dataset(df)
    assert targets["target_good"].iloc[0] == 1
    assert targets["target_return"].iloc[0] == pytest.approx(0.02 - ROUND_TRIP_COST_RATIO)

    df = _build_raw_df(n_per_date=[1])
    df["(수익률, %)"] = "-3.0"
    _, targets, _, _ = build_ml_dataset(df)
    assert targets["target_bad"].iloc[0] == 1


def test_build_ml_dataset_attaches_feature_manifest() -> None:
    """훈련 입력에 결정적 피처 매니페스트가 부착됩니다."""
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    manifest = processed.attrs["feature_manifest"]
    assert set(manifest.columns) == {
        "feature_name",
        "source_column",
        "availability_rule",
        "unit",
        "panel_scope",
    }
    assert set(manifest["feature_name"]) == set(X.columns)
    assert (manifest["availability_rule"] == "at_decision_time").all()


def test_production_feature_set_excludes_candle_price_derived_features() -> None:
    """P0: base40 은 close/high/low/실현 매수가 파생 피처를 X 에서 제외합니다.

    base40 은 이전 conservative feature 집합과 동일해야 하므로
    캔들/실현 매수가 파생 피처와 그 robust-z 변환을 포함하지 않습니다.
    """
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    assert X.attrs["feature_set"] == "base40"
    candle_derived = {
        "close_position",
        "body_ratio",
        "upper_shadow_ratio",
        "intraday_range",
        "buy_price_change_rate",
        "gap_ratio",
        "relative_change_rate",
    }
    assert candle_derived.issubset(set(processed.columns))
    assert not candle_derived.intersection(X.columns)
    # 캔들/실현 매수가 파생 피처의 robust-z 변환도 X 에서 제외됩니다.
    for z_col in ("buy_price_change_rate_z", "gap_ratio_z"):
        assert z_col in processed.columns
        assert z_col not in X.columns


def test_snapshot49_feature_set_promotes_nine_documented_features() -> None:
    """snapshot49 는 정확히 9개 내부 피처를 X 에 승격합니다."""
    X, targets, cat_features, processed = build_ml_dataset(
        _build_raw_df(), feature_set="snapshot49"
    )
    assert X.attrs["feature_set"] == "snapshot49"
    promoted = {
        "close_position",
        "body_ratio",
        "upper_shadow_ratio",
        "intraday_range",
        "buy_price_change_rate",
        "gap_ratio",
        "relative_change_rate",
        "buy_price_change_rate_z",
        "gap_ratio_z",
    }
    assert promoted.issubset(set(processed.columns))
    assert promoted.issubset(X.columns)
    # 원시 가격/매수가/결과 열은 여전히 X 에서 제외됩니다.
    assert not {"open_price", "close_price", "buy_price", "sell_price", "net_return"}.intersection(
        X.columns
    )


def test_interaction53_feature_set_adds_four_interaction_features() -> None:
    """interaction53 은 snapshot49 에 4개 상호작용 피처만 추가합니다."""
    X, targets, cat_features, processed = build_ml_dataset(
        _build_raw_df(), feature_set="interaction53"
    )
    assert X.attrs["feature_set"] == "interaction53"
    interactions = {"candle_strength", "range_efficiency", "flow_turnover", "relative_flow_strength"}
    assert interactions.issubset(set(processed.columns))
    assert interactions.issubset(X.columns)
    snapshot49 = {
        "close_position",
        "body_ratio",
        "upper_shadow_ratio",
        "intraday_range",
        "buy_price_change_rate",
        "gap_ratio",
        "relative_change_rate",
        "buy_price_change_rate_z",
        "gap_ratio_z",
    }
    assert snapshot49.issubset(X.columns)
    # X 에 결과/식별자/원시 가격 열이 누출되지 않아야 합니다.
    assert not {"stock_code", "sell_price", "net_return", "intraday_return"}.intersection(X.columns)


def test_production_calendar_flow_feature_set_adds_nine_features() -> None:
    """production_calendar_flow 는 base40 에 정확히 9개 캘린더/수급 후보를 추가합니다."""
    X_base, _, _, _ = build_ml_dataset(
        _build_scenario_raw_df(), panel_mode="scenario_action"
    )
    X_pcf, _, _, processed = build_ml_dataset(
        _build_scenario_raw_df(),
        feature_set="production_calendar_flow",
        panel_mode="scenario_action",
    )
    assert processed.attrs["feature_set"] == "production_calendar_flow"
    assert X_pcf.attrs["feature_set"] == "production_calendar_flow"
    assert _PRODUCTION_CALENDAR_FLOW_NINE.issubset(set(processed.columns))
    assert _PRODUCTION_CALENDAR_FLOW_NINE.issubset(X_pcf.columns)
    # base40 X 에는 후보 피처가 하나도 없고, 차이는 정확히 9종입니다.
    assert not _PRODUCTION_CALENDAR_FLOW_NINE.intersection(X_base.columns)
    assert set(X_pcf.columns) - set(X_base.columns) == _PRODUCTION_CALENDAR_FLOW_NINE
    # 캔들/실현 매수가 파생 피처는 여전히 X 에서 제외됩니다.
    candle_derived = {
        "close_position",
        "body_ratio",
        "upper_shadow_ratio",
        "intraday_range",
        "buy_price_change_rate",
        "gap_ratio",
        "relative_change_rate",
    }
    assert not candle_derived.intersection(X_pcf.columns)


def test_close_morning61_feature_set_is_snapshot49_plus_relative_flow_strength() -> None:
    """close_morning61 은 snapshot49 전체 + relative_flow_strength 정확히 1개입니다.

    rejected 상호작용(range_efficiency/flow_turnover), 캘린더 흐름, 타깃, 매도가,
    행 단위 타임스탬프는 X 에 포함되지 않습니다.
    """
    X_snap, _, _, _ = build_ml_dataset(_build_raw_df(), feature_set="snapshot49")
    X_cm, _, _, processed = build_ml_dataset(_build_raw_df(), feature_set="close_morning61")
    assert processed.attrs["feature_set"] == "close_morning61"
    assert X_cm.attrs["feature_set"] == "close_morning61"
    assert set(X_cm.columns) == set(X_snap.columns) | {"relative_flow_strength"}
    assert "relative_flow_strength" in X_cm.columns
    assert not {"range_efficiency", "flow_turnover"}.intersection(X_cm.columns)
    assert not _PRODUCTION_CALENDAR_FLOW_NINE.intersection(X_cm.columns)
    assert not {"sell_price", "net_return"}.intersection(X_cm.columns)
    assert not {"decision_timestamp", "feature_available_timestamp"}.intersection(
        X_cm.columns
    )
    assert X_cm["relative_flow_strength"].between(0.0, 1.0).all()


def test_relative_flow_strength_deterministic_and_bounded() -> None:
    """relative_flow_strength 는 결정적이며 [0, 1] 로 유한하게 묶입니다."""
    _, _, _, first = build_ml_dataset(_build_raw_df(), feature_set="close_morning61")
    _, _, _, second = build_ml_dataset(_build_raw_df(), feature_set="close_morning61")
    pd.testing.assert_series_equal(
        first["relative_flow_strength"], second["relative_flow_strength"]
    )
    rfs = first["relative_flow_strength"]
    assert rfs.notna().all()
    assert rfs.between(0.0, 1.0).all()


def test_close_morning61_fails_closed_on_missing_required_source() -> None:
    """champion 피처셋의 필수 원천(등락률)이 없으면 ValueError 로 fail-closed 합니다."""
    df = _build_raw_df().drop(columns=["(등락률)"])
    with pytest.raises(ValueError, match="relative_flow_strength"):
        build_ml_dataset(df, feature_set="close_morning61")


def test_validate_close_morning61_feature_fails_closed() -> None:
    """유효성 검증 헬퍼는 피처 부재와 비유한/범위 밖 값을 각각 ValueError 로 거부합니다."""
    with pytest.raises(ValueError, match="relative_flow_strength"):
        _validate_close_morning61_feature(
            pd.DataFrame({"change_rate": [1.0]})
        )
    out_of_bounds = pd.DataFrame({"relative_flow_strength": [1.5]})
    with pytest.raises(ValueError, match="finite within \\[0, 1\\]"):
        _validate_close_morning61_feature(out_of_bounds)
    non_finite = pd.DataFrame({"relative_flow_strength": [np.inf]})
    with pytest.raises(ValueError, match="finite within \\[0, 1\\]"):
        _validate_close_morning61_feature(non_finite)


def test_validate_close_morning61_feature_accepts_bounded_feature() -> None:
    """유한하고 [0, 1] 범위의 피처는 통과합니다."""
    _validate_close_morning61_feature(pd.DataFrame({"relative_flow_strength": [0.0, 0.5, 1.0]}))


def test_build_ml_dataset_rejects_unknown_feature_set() -> None:
    """허용되지 않는 feature_set 은 ValueError 를 발생시킵니다."""
    with pytest.raises(ValueError, match="feature_set"):
        build_ml_dataset(_build_raw_df(), feature_set="leaky80")


def test_build_ml_dataset_rejects_unknown_panel_mode() -> None:
    """허용되지 않는 panel_mode 는 ValueError 를 발생시킵니다."""
    with pytest.raises(ValueError, match="panel_mode"):
        build_ml_dataset(_build_raw_df(), panel_mode="raw_rows_legacy")


def _build_scenario_raw_df() -> pd.DataFrame:
    """날짜-종목별로 서로 다른 시나리오를 가진 원본 헤더 형태의 DataFrame."""
    df = _build_raw_df(n_per_date=[3, 3])
    df["(차트분석)"] = ["상따", "120 돌파", "거래량 폭증", "상한가 다음날", "신고가 근접", "상승형 음봉"]
    return df


def test_build_ml_dataset_scenario_action_panel_mode() -> None:
    """scenario_action 모드는 행동 패널을 거쳐 고정 시나리오 one-hot 피처를 X 에 포함합니다."""
    X, targets, cat_features, processed = build_ml_dataset(
        _build_scenario_raw_df(), panel_mode="scenario_action"
    )
    assert processed.attrs["panel_mode"] == "scenario_action"
    assert X.attrs["panel_mode"] == "scenario_action"
    assert len(processed) == 6
    scenario_cols = {
        "scenario_is_sangtta",
        "scenario_is_120_breakout",
        "scenario_is_volume_surge",
        "scenario_is_new_high",
        "scenario_is_near_new_high",
        "scenario_is_limitup_next_day",
        "scenario_is_rising_bearish",
        "scenario_other",
        "scenario_count_for_stock_date",
        "has_sangtta_for_stock_date",
        "is_multi_scenario_stock_date",
    }
    assert scenario_cols.issubset(X.columns)
    # chart_analysis 원문은 one-hot 수치 피처를 대신 사용하므로 X 에서 제외됩니다.
    assert "chart_analysis" not in X.columns
    assert "chart_analysis" not in cat_features
    # X 에 결과/식별자/원시 가격이 누출되지 않습니다.
    assert not {"sell_price", "net_return", "stock_code", "trade_date"}.intersection(X.columns)


def test_build_ml_dataset_scenario_action_excludes_conflicting_rejects() -> None:
    """충돌 중복 행동은 reject 로 노출되고 X/타깃/학습에서 제외됩니다."""
    df = _build_raw_df(n_per_date=[2, 2])
    df["(차트분석)"] = ["상따", "상따", "거래량 폭증", "신고가"]
    # 같은 날짜-종목-시나리오 key 의 실행값 충돌 생성.
    df.loc[1, "종목코드"] = df.loc[0, "종목코드"]
    df.loc[1, "(수익률, %)"] = "9.99%"

    X, targets, cat_features, processed = build_ml_dataset(df, panel_mode="scenario_action")
    rejects = processed.attrs["scenario_action_rejects"]
    assert len(rejects) == 2
    assert (rejects["reject_reason"] == "conflicting_duplicate_action").all()
    assert len(processed) == 2
    assert len(X) == len(processed)
    assert len(targets["target_return"]) == len(processed)


def test_build_ml_dataset_scenario_action_context_for_sangtta_stock_date() -> None:
    """상따가 포함된 날짜-종목은 has_sangtta/is_multi_scenario 가 1 이 됩니다."""
    df = _build_raw_df(n_per_date=[3, 3])
    df["(차트분석)"] = ["상따", "120 돌파", "거래량 폭증", "상한가 다음날", "신고가 근접", "상승형 음봉"]
    # 날짜1의 000000 에 상따 + 120 돌파 두 행동을 생성.
    df.loc[1, "종목코드"] = df.loc[0, "종목코드"]
    X, targets, cat_features, processed = build_ml_dataset(df, panel_mode="scenario_action")
    group = processed.loc[processed["stock_code"] == "000000"]
    date1 = group["trade_date"].iloc[0]
    date1_group = group.loc[group["trade_date"] == date1]
    assert (date1_group["scenario_count_for_stock_date"] == 2).all()
    assert (date1_group["has_sangtta_for_stock_date"] == 1).all()
    assert (date1_group["is_multi_scenario_stock_date"] == 1).all()


def test_build_ml_dataset_raw_rows_keeps_chart_analysis_categorical() -> None:
    """raw_rows 모드는 chart_analysis 를 기존 범주형 피처로 유지합니다."""
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    assert processed.attrs["panel_mode"] == "raw_rows"
    assert "chart_analysis" in X.columns
    assert "chart_analysis" in cat_features
    assert "scenario_is_sangtta" not in X.columns


def test_engineer_features_created() -> None:
    """상대 비율/로그/횡단면 백분위 피처가 생성되는지 검증합니다."""
    cleaned = clean_column_names(_build_raw_df())
    engineered = engineer_features(cleaned)
    expected = {
        "buy_price_change_rate",
        "gap_ratio",
        "intraday_return",
        "major_density",
        "prog_dominance",
        "rank_ratio",
        "relative_change_rate",
        "trade_value_pct_rank",
        "inst_net_buy_pct_rank",
        "foreign_net_buy_pct_rank",
        "change_rate_pct_rank",
    }
    assert expected.issubset(set(engineered.columns))
    # production_calendar_flow 연구 후보 9종이 engineer_features 에서 생성됩니다.
    candidate = {
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "flow_consensus",
        "flow_alignment_direction",
        "flow_turnover",
        "friday_selection_rank_pct",
    }
    assert candidate.issubset(set(engineered.columns))


_PRODUCTION_CALENDAR_FLOW_NINE: set[str] = {
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "flow_consensus",
    "flow_alignment_direction",
    "flow_turnover",
    "friday_selection_rank_pct",
}


def test_production_calendar_flow_weekday_one_hot_and_friday_rank() -> None:
    """요일 one-hot 은 정확히 하나만 1 이고, 금요일 랭킹 상호작용은 금요일만 0 이 아닙니다."""
    df = _build_raw_df(n_per_date=[2, 2])
    monday = pd.Timestamp("2024-01-08")
    friday = pd.Timestamp("2024-01-12")
    df["매수날짜"] = [monday, monday, friday, friday]
    engineered = engineer_features(clean_column_names(df))

    weekday_cols = [
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
    ]
    assert (engineered[weekday_cols].sum(axis=1) == 1).all()
    assert engineered[weekday_cols].isin([0.0, 1.0]).all().all()
    assert engineered[weekday_cols].dtypes.eq("float64").all()

    monday_mask = engineered["trade_date"] == monday
    friday_mask = engineered["trade_date"] == friday
    assert (engineered.loc[monday_mask, "weekday_is_monday"] == 1.0).all()
    assert (engineered.loc[monday_mask, "weekday_is_friday"] == 0.0).all()
    assert (engineered.loc[friday_mask, "weekday_is_friday"] == 1.0).all()

    # 금요일 랭킹 상호작용: 금요일은 1 - rank_ratio, 그 외 요일은 정확히 0.
    assert (engineered.loc[monday_mask, "friday_selection_rank_pct"] == 0.0).all()
    expected_friday = (1 - engineered.loc[friday_mask, "rank_ratio"]).round(12)
    actual_friday = engineered.loc[friday_mask, "friday_selection_rank_pct"].round(12)
    assert (actual_friday == expected_friday).all()
    # 금요일인데 rank_ratio < 1 인 행은 0 이 아닙니다 (정확히 그 요일에만 시그널).
    assert (actual_friday < 1.0).all()
    assert (engineered.loc[friday_mask, "rank_ratio"] < 1).any()
    assert (actual_friday[engineered.loc[friday_mask, "rank_ratio"] < 1] > 0).all()


def test_production_calendar_flow_zero_flow_and_alignment_formulas() -> None:
    """전 수급이 0/결측이면 consensus 0, 정렬 0 이고, 일치 방향은 ±1 입니다."""
    df = _build_raw_df(n_per_date=[3])
    df["(기관_순매수)"] = 0.0
    df["(외국인_순매수)"] = 0.0
    df["(프로그램_순매수)"] = 0.0
    engineered = engineer_features(clean_column_names(df))
    assert (engineered["flow_consensus"] == 0).all()
    assert (engineered["flow_alignment_direction"] == 0.0).all()

    aligned = _build_raw_df(n_per_date=[3])
    aligned["(기관_순매수)"] = 100.0
    aligned["(외국인_순매수)"] = 50.0
    aligned["(프로그램_순매수)"] = 25.0
    engineered = engineer_features(clean_column_names(aligned))
    assert (engineered["flow_consensus"] == 3).all()
    assert (engineered["flow_alignment_direction"] == 1.0).all()
    assert engineered["flow_alignment_direction"].between(-1.0, 1.0).all()
    assert engineered["flow_consensus"].between(-3, 3).all()


def test_no_data_leakage_in_features() -> None:
    """피처 행렬 X에 미래 정보/메타데이터가 유출되지 않는지 검증합니다."""
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    leak_cols = {"buy_price", "sell_price", "net_return"} | set(targets.keys())
    assert not leak_cols.intersection(X.columns)


def test_categorical_features_reported() -> None:
    """범주형 피처가 cat_features에 노출되고 X에 포함되는지 검증합니다."""
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    assert cat_features
    assert set(cat_features).issubset(X.columns)


def test_build_ml_dataset_nonempty() -> None:
    """SCENARIO_PREPROCESS_ML_DATASET: build_ml_dataset이 X/y/cat_features 및 multi-target 컬럼을 반환합니다."""
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    assert len(X) > 0
    assert len(targets) == 4
    assert list(targets.keys()) == [
        "target_return",
        "target_rank",
        "target_good",
        "target_bad",
    ]


def test_theme_df_fallback_fill() -> None:
    """theme_df 기반 theme_sector/market_type 결측치 보강을 검증합니다."""
    df = _build_raw_df()
    df.loc[df.index[0], "(테마/섹터)"] = np.nan
    df.loc[df.index[1], "(시장구분)"] = np.nan
    theme_df = pd.DataFrame(
        {
            "종목코드": sorted(df["종목코드"].unique()),
            "테마": "반도체",
            "시장구분": "KOSPI",
        }
    )
    X, targets, cat_features, processed = build_ml_dataset(df, theme_df)
    assert processed["theme_sector"].isna().sum() == 0
    assert processed["market_type"].isna().sum() == 0


def test_create_multi_targets_standalone() -> None:
    """단일 함수 호출로도 타깃이 생성되는지 검증합니다."""
    cleaned = clean_column_names(_build_raw_df())
    cleaned = cleaned.dropna(subset=["net_return"])
    out = create_multi_targets(cleaned)
    assert out["target_rank"].between(0, 4).all()
    assert out["target_rank"].notna().all()


def test_target_rank_single_date_group() -> None:
    """단일 거래일 그룹(그룹 1개)에서도 target_rank가 Series로 할당되는지 검증합니다.

    pandas 2.x의 groupby.apply가 단일 그룹에서 DataFrame을 반환해
    target_rank 할당이 깨지는 회귀를 방지합니다.
    """
    df = _build_raw_df()
    df = df[df["매수날짜"] == df["매수날짜"].iloc[0]]
    X, targets, cat_features, processed = build_ml_dataset(df)
    assert targets["target_rank"].notna().all()
    assert targets["target_rank"].between(0, 4).all()
    assert len(targets["target_rank"]) == len(processed)

def test_s1_unit_consistency() -> None:
    """S1_unit_consistency: buy_price_change_rate와 kospi_change의 단위가 %로 통일되어 차감이 유효합니다.

    매수가 70000, 전일종가 65000 → buy_price_change_rate=7.69%, kospi_change=0.3%
    → relative_change_rate ≈ 7.39%. 단위 불일치 시 ~700% 수준으로 왜곡됩니다.
    """
    df = _build_raw_df(n_per_date=[1])
    df["(매수 가격)"] = 70_000.0
    df["(전일종가)"] = 65_000.0
    df["(시가)"] = 65_000.0
    df["(kospi, %)"] = 0.3
    df["(시장구분)"] = "KOSPI"
    engineered = engineer_features(clean_column_names(df))
    row = engineered.iloc[0]
    assert row["buy_price_change_rate"] == pytest.approx(5000 / 65_000 * 100, rel=1e-6)
    assert row["gap_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert row["relative_change_rate"] == pytest.approx(
        5000 / 65_000 * 100 - 0.3, rel=1e-6
    )


def test_s2_no_leakage() -> None:
    """S2_no_leakage: build_ml_dataset() 반환 X에 당일 OHLC/intraday_return 등 미래 정보 미포함."""
    X, targets, cat_features, processed = build_ml_dataset(_build_raw_df())
    leak_cols = {
        "intraday_return",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "prev_close_price",
        "buy_price",
        "sell_price",
        "market_cap_100m",
        "trade_value_100m",
        "net_return",
        "trade_date",
        "stock_code",
    }
    assert not leak_cols.intersection(X.columns)


def test_s3_rank_edge_cases() -> None:
    """S3_rank_edge_cases: 거래일별 종목 수 1/2/3/4개일 때 target_rank가 0~4 범위에서 max=4 생성."""
    df = _build_raw_df(n_per_date=[1, 2, 3, 4])
    _, _, _, processed = build_ml_dataset(df)
    rank = processed["target_rank"]
    assert rank.between(0, 4).all()
    assert rank.max() == 4
    groups: dict[int, set[int]] = {}
    for date, size in zip(sorted(processed["trade_date"].unique()), [1, 2, 3, 4], strict=True):
        groups[size] = set(processed.loc[processed["trade_date"] == date, "target_rank"])
    assert groups[1] == {2}
    assert groups[2] == {0, 4}
    assert groups[3] == {0, 2, 4}
    assert groups[4] == {0, 1, 3, 4}


def test_s4_nan_safety() -> None:
    """S4_nan_safety: inst_net_buy / total_candidate_count NaN 시 major_density, rank_ratio 유효."""
    df = _build_raw_df(n_per_date=[3])
    df["(기관_순매수)"] = np.nan
    df["(총 종목 수)"] = np.nan
    engineered = engineer_features(clean_column_names(df))
    assert engineered["major_density"].notna().all()
    assert engineered["inst_density"].notna().all()
    assert engineered["foreign_density"].notna().all()
    assert engineered["rank_ratio"].notna().all()
    assert (engineered["rank_ratio"] > 0).all()


def test_s5_robust_z_bounds() -> None:
    """S5_robust_z_bounds: Robust Z-Score가 [-5, 5] 범위 내이고 MAD=0 그룹은 NaN으로 안전 처리."""
    df = _build_raw_df(n_per_date=[3, 3])
    date1 = df["매수날짜"].iloc[0]
    df.loc[df["매수날짜"] == date1, "(등락률)"] = 1.0
    engineered = engineer_features(clean_column_names(df))
    out = _apply_robust_z(engineered, _ROBUST_Z_COLUMNS)
    for col in _ROBUST_Z_COLUMNS:
        z_col = f"{col}_z"
        assert z_col in out.columns
        vals = out[z_col].dropna()
        assert vals.between(-5, 5).all()
    assert not np.isinf(out["change_rate_z"]).any()
    assert out["change_rate_z"].notna().sum() > 0


def test_s6_log_original_preserved() -> None:
    """S6_log_original_preserved: log 변환 후 원본 금액 컬럼(market_cap_100m) 유지 + log_ 접두사 컬럼 생성."""
    cleaned = clean_column_names(_build_raw_df())
    original_cap = cleaned["market_cap_100m"].copy()
    engineered = engineer_features(cleaned)
    assert "log_market_cap_100m" in engineered.columns
    assert "log_trade_value_100m" in engineered.columns
    pd.testing.assert_series_equal(engineered["market_cap_100m"], original_cap)
    assert engineered["log_market_cap_100m"].notna().all()


def test_s7_vkospi_zero() -> None:
    """S7_vkospi_zero: v_kospi=0 입력 시 NaN 변환 후 ffill/bfill 보간, v_kospi_change 정상 생성."""
    df = _build_raw_df(n_per_date=[2, 2, 2])
    date1 = df["매수날짜"].iloc[0]
    date3 = df["매수날짜"].iloc[-1]
    df.loc[df["매수날짜"] == date1, "v_kospi"] = 0
    df.loc[df["매수날짜"] == date3, "v_kospi"] = 20.0
    engineered = engineer_features(clean_column_names(df))
    assert engineered["v_kospi"].notna().all()
    assert (engineered["v_kospi"] > 0).all()
    assert "v_kospi_change" in engineered.columns
    assert engineered["v_kospi_change"].notna().all()
    assert not np.isinf(engineered["v_kospi_change"]).any()


def test_s8_pct_rank_missing_col() -> None:
    """S8_pct_rank_missing_col: _PCT_RANK_COLUMNS 대상 컬럼 부재 시 KeyError 미발생, 해당 rank 컬럼만 스킵."""
    df = _build_raw_df(n_per_date=[3])
    cleaned = clean_column_names(df)
    cleaned = cleaned.drop(columns=["foreign_net_buy", "v_kosdaq"])
    engineered = engineer_features(cleaned)
    assert "foreign_net_buy_pct_rank" not in engineered.columns
    assert "inst_net_buy_pct_rank" in engineered.columns
    assert "major_density" in engineered.columns
    assert engineered["major_density"].notna().all()
    assert "v_kosdaq_change" not in engineered.columns
    out = _apply_robust_z(engineered, _ROBUST_Z_COLUMNS)
    assert "foreign_density_z" not in out.columns
    assert "major_density_z" in out.columns
    assert out["major_density_z"].notna().all()


def _price_history_for(raw_df: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """``_build_raw_df`` 종목/날짜를 커버하는 표준 가격 이력 스키마를 합성합니다."""
    cleaned = clean_column_names(raw_df)
    rng = np.random.default_rng(11)
    symbols = sorted(cleaned["stock_code"].unique())
    date_min = cleaned["trade_date"].min() - pd.Timedelta(days=lookback_days)
    dates = pd.date_range(date_min, cleaned["trade_date"].max(), freq="D")
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        close = 50000.0
        for date in dates:
            close = close * (1 + rng.normal(0, 0.02))
            open_ = close * (1 + rng.normal(0, 0.005))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            volume = abs(rng.normal(1_000_000, 200_000))
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "prev_close": close / (1 + rng.normal(0, 0.02)),
                    "volume": volume,
                    "trade_value_100m": volume * close / 100_000_000,
                    "market_cap_100m": close * 1_000_000 / 100_000_000,
                    "daily_change_pct": rng.normal(0.0, 2.0),
                    "market": "KOSPI",
                    "inst_net_buy": rng.normal(0, 1e8),
                    "foreign_net_buy": rng.normal(0, 1e8),
                    "prog_net_buy": rng.normal(0, 1e8),
                }
            )
    return pd.DataFrame(rows)


def test_causal_expanded_v1_requires_price_history() -> None:
    """SCENARIO_PREPROCESS_ML_DATASET: causal_expanded_v1 은 이력이 없으면 fail-closed."""
    raw = _build_raw_df(n_per_date=[8, 8, 8])
    with pytest.raises(ValueError, match="price_history_df"):
        build_ml_dataset(raw, feature_set="causal_expanded_v1")


def test_build_ml_dataset_causal_expanded_v1_returns_catalog() -> None:
    """SCENARIO_PREPROCESS_ML_DATASET: causal_expanded_v1 은 카탈로그 행렬을 반환합니다.

    기존 데이터셋 구성은 하위 호환 기준선으로 유지되고, causal_expanded_v1 은
    opt-in 경로입니다.
    """
    raw = _build_raw_df(n_per_date=[8, 8, 8])
    history = _price_history_for(raw)
    X, targets, cat_features, processed = build_ml_dataset(
        raw, feature_set="causal_expanded_v1", panel_mode="raw_rows", price_history_df=history
    )
    assert X.columns.is_unique
    assert 600 <= X.shape[1] <= 1000
    manifest = X.attrs["feature_manifest"]
    assert {"family", "source_columns", "lookback_groups", "availability_rule"}.issubset(
        set(manifest.columns)
    )
    assert set(manifest["feature_name"]) == set(X.columns)
    assert X.attrs["catalog_version"] == "causal_expanded_v1"
    assert X.attrs["catalog_hash"]
    assert processed.attrs["catalog_hash"] == X.attrs["catalog_hash"]
    assert "snap_log_market_cap_100m" in processed.columns
    assert "target_return" in targets


def test_build_ml_dataset_causal_expanded_v1_scenario_action_panel() -> None:
    """causal_expanded_v1 은 scenario_action 패널 모드에서도 카탈로그 행렬을 생성합니다."""
    raw = _build_raw_df(n_per_date=[8, 8, 8])
    history = _price_history_for(raw)
    X, targets, cat_features, processed = build_ml_dataset(
        raw,
        feature_set="causal_expanded_v1",
        panel_mode="scenario_action",
        price_history_df=history,
    )
    assert 600 <= X.shape[1] <= 1000
    assert processed.attrs["panel_mode"] == "scenario_action"
    assert "chart_analysis" not in X.columns


def test_legacy_feature_sets_remain_unchanged_without_history() -> None:
    """SCENARIO_PREPROCESS_ML_DATASET: 레거시 피처셋은 이력 없이 기존 동작을 유지합니다."""
    raw = _build_raw_df(n_per_date=[8, 8, 8])
    legacy = build_ml_dataset(raw, feature_set="close_morning61", panel_mode="raw_rows")
    snapshot = build_ml_dataset(raw, feature_set="snapshot49", panel_mode="raw_rows")
    assert set(legacy[0].columns) == set(snapshot[0].columns) | {"relative_flow_strength"}
    assert "feature_manifest" in legacy[0].attrs
    assert legacy[0].attrs["feature_manifest"].shape[1] == 5
