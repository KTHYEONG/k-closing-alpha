"""Unit tests for daily data collection.

SCENARIO_DAILY_COLLECT_REFACTORING_01:
Verifies that collect.py saves collected condition data directly
without chart_pass_cache.json or parenthesis column renaming.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.daily.collect import save_collected_condition_data


def test_scenario_daily_collect_refactoring_01(tmp_path: Path) -> None:
    """[SCENARIO_DAILY_COLLECT_REFACTORING_01] Verify clean standard CSV saving without chart_pass."""
    sample_df = pd.DataFrame(
        {
            "종목명": ["테스트종목"],
            "종목코드": ["123"],
            "시가": [1000],
            "고가": [1100],
            "저가": [990],
            "종가": [1050],
            "전일종가": [1000],
            "시가총액": [500.0],
            "거래대금": [100.0],
            "등락률": [5.0],
            "선정순위": [1],
            "기관_순매수": [10.0],
            "외국인_순매수": [20.0],
            "프로그램_순매수": [5.0],
            "체결강도": [120.0],
            "시장구분": ["KOSDAQ"],
            "총_종목수": [100],
            "평균_거래대금": [50.0],
            "kospi": [0.5],
            "kosdaq": [1.2],
            "v_kospi": [15.0],
            "v_kosdaq": [20.0],
            "거래량": [10000],
            "시나리오": ["거래량 폭증"],
        }
    )

    csv_path = tmp_path / "daily" / "daily_stocks.csv"
    res_path = save_collected_condition_data(sample_df, csv_path)

    assert res_path.exists()
    saved_df = pd.read_csv(res_path, dtype={"종목코드": str})
    assert saved_df["종목코드"].iloc[0] == "000123"
    assert "차트통과" not in saved_df.columns
    assert "(차트통과)" not in saved_df.columns
