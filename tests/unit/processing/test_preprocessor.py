"""preprocessor 데이터 복구 파이프라인 단위 테스트.

`docs/specs/preprocessor_data_recovery_contract.json`의 시나리오 기반 검증입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.preprocessor import clean_column_names


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
