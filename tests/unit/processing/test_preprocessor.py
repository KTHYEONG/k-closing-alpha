"""live 스냅샷 피처 프레임 변환(serving/realtime/features) 단위 테스트.

학습/데이터셋/타깃/가용성 증명 케이스는 ``legacy/tests/unit/processing/test_preprocessor.py``
로 이동되었습니다. 이 파일은 실시간 추론 경로가 소비하는 피처 변환만 검증합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.serving.realtime.features import (
    _ROBUST_Z_COLUMNS,
    _apply_robust_z,
    build_snapshot_features,
    engineer_features,
)


def _live_snapshot_raw() -> pd.DataFrame:
    """일일 CSV 스프레드시트 헤더 형태의 당일 스냅샷."""
    return pd.DataFrame(
        {
            "종목코드": ["000001", "000002"],
            "시가": [10_000.0, 20_000.0],
            "고가": [11_000.0, 22_000.0],
            "저가": [9_000.0, 19_000.0],
            "종가": [10_500.0, 21_000.0],
            "전일종가": [10_000.0, 20_000.0],
            "시가총액": [1_000.0, 2_000.0],
            "거래대금": [100.0, 200.0],
            "등락률": [5.0, 2.5],
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


def test_build_snapshot_features_derives_standard_ml_frame() -> None:
    """당일 스냅샷이 표준 ML 피처 프레임으로 변환되고 파생 피처가 생성됩니다."""
    out = build_snapshot_features(
        _live_snapshot_raw(), decision_date=pd.Timestamp("2026-08-04")
    )
    expected = [
        "stock_code",
        "open_price",
        "close_price",
        "prev_close_price",
        "change_rate",
        "selection_rank",
        "trade_date",
        "major_density",
        "prog_dominance",
        "turnover",
        "rank_ratio",
        "log_market_cap_100m",
        "change_rate_z",
    ]
    assert set(expected).issubset(out.columns)
    assert np.issubdtype(out["trade_date"].dtype, np.datetime64)
    assert out["major_density"].notna().all()


def test_engineer_features_creates_cross_sectional_features() -> None:
    """상대 비율/로그/횡단면 백분위 피처가 생성되는지 검증합니다."""
    df = _live_snapshot_raw().rename(
        columns={
            "등락률": "change_rate",
            "시가": "open_price",
            "고가": "high_price",
            "저가": "low_price",
            "종가": "close_price",
            "전일종가": "prev_close_price",
            "시가총액": "market_cap_100m",
            "거래대금": "trade_value_100m",
            "선정순위": "selection_rank",
            "기관_순매수": "inst_net_buy",
            "외국인_순매수": "foreign_net_buy",
            "프로그램_순매수": "prog_net_buy",
            "시장구분": "market_type",
            "총_종목수": "total_candidate_count",
            "평균_거래대금": "avg_trade_value",
            "kospi": "kospi_change",
            "kosdaq": "kosdaq_change",
            "테마_섹터": "theme_sector",
            "차트분석": "chart_analysis",
        }
    )
    df["trade_date"] = pd.Timestamp("2026-08-04")
    df["buy_price"] = df["close_price"]
    engineered = engineer_features(df)
    expected = {
        "buy_price_change_rate",
        "gap_ratio",
        "intraday_return",
        "major_density",
        "prog_dominance",
        "rank_ratio",
        "relative_change_rate",
        "change_rate_pct_rank",
        "log_market_cap_100m",
    }
    assert expected.issubset(set(engineered.columns))


def test_apply_robust_z_produces_clipped_bounded_scores() -> None:
    """Robust Z-Score 가 [-5, 5] 범위 내에서 생성됩니다."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "trade_date": ["2026-08-04"] * 12,
            "change_rate": rng.normal(size=12),
            "major_density": rng.uniform(0, 1, size=12),
        }
    )
    out = _apply_robust_z(df, _ROBUST_Z_COLUMNS)
    for col in _ROBUST_Z_COLUMNS:
        if col not in df.columns:
            continue
        vals = out[f"{col}_z"].dropna()
        assert vals.between(-5, 5).all()
    assert not np.isinf(out["change_rate_z"]).any()
    assert out["change_rate_z"].notna().sum() > 0
