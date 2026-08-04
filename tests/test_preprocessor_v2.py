"""preprocessor_v2 ML 데이터 전처리 파이프라인 검증 테스트.

`docs/specs/ml_data_preprocessing_contract.json`의 시나리오 기반 검증입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.preprocessor_v2 import (
    _ROBUST_Z_COLUMNS,
    _apply_robust_z,
    build_ml_dataset,
    clean_column_names,
    create_multi_targets,
    engineer_features,
)


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
    assert targets["target_return"].between(-10.0, 10.0).all()
    assert set(targets["target_good"].unique()).issubset({0, 1})
    assert set(targets["target_bad"].unique()).issubset({0, 1})
    for series in targets.values():
        assert series.notna().all()
        assert len(series) == len(processed)


def test_target_return_clipped() -> None:
    """target_return이 ±10%로 클리핑되는지 검증합니다."""
    df = _build_raw_df()
    df.loc[df.index[0], "(수익률, %)"] = "25.0"
    df.loc[df.index[1], "(수익률, %)"] = "-15.0"
    _, targets, _, _ = build_ml_dataset(df)
    assert targets["target_return"].max() <= 10.0
    assert targets["target_return"].min() >= -10.0


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
    """python_assertion: build_ml_dataset(sample_df)[0] 길이가 0보다 커야 합니다."""
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
