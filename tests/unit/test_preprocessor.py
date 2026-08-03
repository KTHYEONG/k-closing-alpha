"""피처 엔지니어링 로직 검증 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src.processing.preprocessor import (
    RENAME_MAP,
    _add_technical_features,
    preprocess_data,
)


def _build_raw_df() -> pd.DataFrame:
    """괄호 포함 스프레드시트 헤더 형태의 원본 DataFrame을 생성합니다."""
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    n = len(dates)
    df = pd.DataFrame(
        {
            "(매수날짜)": dates,
            "(종목코드)": ["005930"] * n,
            "(시가)": [70_000 + i for i in range(n)],
            "(고가)": [71_000 + i for i in range(n)],
            "(저가)": [69_000 + i for i in range(n)],
            "(종가)": [70_500 + i for i in range(n)],
            "(전일종가)": [69_500 + i for i in range(n)],
            "(시가총액, 억)": [400_000] * n,
            "(거래대금, 억)": [10_000] * n,
            "(등락률)": [1.0] * n,
            "(선정 순위)": list(range(1, n + 1)),
            "(기관_순매수)": [1000] * n,
            "(외국인_순매수)": [2000] * n,
            "(프로그램_순매수)": [500] * n,
            "(체결강도)": [120] * n,
            "(시장구분)": ["KOSPI"] * n,
            "(총 종목 수)": [200] * n,
            "(평균 거래대금)": [9_000] * n,
            "(kospi, %)": [0.5] * n,
            "(kosdaq, %)": [0.3] * n,
            "(v-kospi)": [15.0] * n,
            "(v-kosdaq)": [18.0] * n,
            "(ema5)": [70_100 + i for i in range(n)],
            "(ema10)": [70_050 + i for i in range(n)],
            "(ema20)": [70_000 + i for i in range(n)],
            "(거래량)": [1_000_000] * n,
            "(Win)": [1] * n,
            "(수익률, %)": [2.0] * n,
            "(차트통과)": ["Y"] * n,
            "(차트분석)": ["신고가"] * n,
        }
    )
    return df


def test_rename_map_applies() -> None:
    df = _build_raw_df()
    renamed = df.rename(columns=RENAME_MAP)
    assert "매수날짜" in renamed.columns
    assert "종목코드" in renamed.columns
    assert "ema5" in renamed.columns


def test_add_technical_features_outputs() -> None:
    df = _build_raw_df().rename(columns=RENAME_MAP)
    out = _add_technical_features(df.copy())
    expected = {
        "disparity_5",
        "momentum_3d",
        "volatility_5d",
        "volume_ratio",
        "momentum_5d",
        "trend_score",
        "ema_oscillator",
        "position_20d",
        "directional_20d",
        "macd_fast",
        "rsi_14",
    }
    assert expected.issubset(set(out.columns))
    assert out.isna().sum().sum() == 0


def test_preprocess_data_classification_shape() -> None:
    df = _build_raw_df()
    X, y, cat_features, processed = preprocess_data(df, task="classification")
    assert len(X) == len(y) == len(processed)
    assert set(cat_features).issubset(X.columns)
    assert "매수날짜" not in X.columns


def test_preprocess_data_requires_date_col() -> None:
    with pytest.raises(ValueError, match="필수 컬럼"):
        preprocess_data(pd.DataFrame({"foo": [1]}))
