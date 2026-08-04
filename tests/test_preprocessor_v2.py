"""preprocessor_v2 ML 데이터 전처리 파이프라인 검증 테스트.

`docs/specs/ml_data_preprocessing_contract.json`의 시나리오 기반 검증입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.preprocessor_v2 import (
    build_ml_dataset,
    clean_column_names,
    create_multi_targets,
    engineer_features,
)


def _build_raw_df() -> pd.DataFrame:
    """괄호/단위 특수문자가 포함된 스프레드시트 헤더 형태의 원본 DataFrame을 생성합니다."""
    rng = np.random.default_rng(42)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rows: list[dict[str, object]] = []
    for date in dates:
        n_stocks = 3 if date.day == 4 else 8
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
