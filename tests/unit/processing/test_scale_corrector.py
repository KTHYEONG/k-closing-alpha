"""스케일 보정 알고리즘 검증 테스트."""

from __future__ import annotations

import pandas as pd

from src.processing.scale_corrector import (
    apply_scale_correction,
    detect_price_scale_mismatch,
    find_closest_valid_scale,
)


def test_find_closest_valid_scale_normal() -> None:
    assert find_closest_valid_scale(1.0) == 1.0
    assert find_closest_valid_scale(0.9) == 1.0
    assert find_closest_valid_scale(1.3) == 1.0


def test_find_closest_valid_scale_tenfold() -> None:
    assert find_closest_valid_scale(10.0) == 10.0
    assert find_closest_valid_scale(0.1) == 0.1


def test_find_closest_valid_scale_rejects_natural_volatility() -> None:
    # 2배/0.5배는 자연스러운 변동으로 오진 금지
    assert find_closest_valid_scale(2.0) == 1.0
    assert find_closest_valid_scale(0.5) == 1.0


def test_detect_price_scale_mismatch_detects_buy() -> None:
    df = pd.DataFrame(
        {
            "종목코드": ["005930", "000660"],
            "매수날짜": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "종가": [70_000, 30_000],
            "매수가격": [7_000, 30_000],  # 첫 행은 1/10 스케일 오류
            "매도가격": [7_200, 31_000],
        }
    )
    mismatched = detect_price_scale_mismatch(df)
    assert ("005930", pd.Timestamp("2024-01-02")) in mismatched
    assert ("000660", pd.Timestamp("2024-01-03")) not in mismatched


def test_detect_price_scale_mismatch_missing_cols() -> None:
    df = pd.DataFrame({"종목코드": ["005930"]})
    assert detect_price_scale_mismatch(df) == {}


def test_apply_scale_correction(sample_trade_df) -> None:
    mismatched = detect_price_scale_mismatch(sample_trade_df)
    corrected = apply_scale_correction(sample_trade_df, mismatched)
    row = corrected.loc[corrected["종목코드"] == "005930"].iloc[0]
    assert row["매수가격"] == 70_000
    assert row["매도가격"] == 72_000


def test_apply_scale_correction_noop_when_clean(sample_trade_df) -> None:
    clean = sample_trade_df.copy()
    clean["매수가격"] = clean["종가"]
    clean["매도가격"] = clean["종가"]
    mismatched = detect_price_scale_mismatch(clean)
    corrected = apply_scale_correction(clean, mismatched)
    pd.testing.assert_frame_equal(corrected, clean)
