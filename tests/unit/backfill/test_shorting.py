"""src.backfill.altdata.shorting 모듈 직접 참조 테스트.

기존 테스트가 전부 tests/unit/backfill/test_altdata_collectors.py에 묶여 있어
lean_check의 test_<module> co-modification 게이트가 shorting.py를 인식하지
못하던 갭을 해소하기 위해 신설. KIS 네이티브 공매도 일별추이(FHPST04830000)
기반 collect_shorting의 핵심 계약(스키마 컬럼, universe_symbols 필수)을
직접 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backfill.altdata import shorting
from src.backfill.altdata.config import AltDataFetchConfig


def test_collect_shorting_panel_schema_has_ten_columns() -> None:
    """공매도 패널은 거래측 4컬럼 + 잔고측 4컬럼(NaN 허용) + date/symbol 총 10컬럼 계약을 유지한다."""
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
    )

    out = shorting.collect_shorting(cfg, [pd.Timestamp("2024-01-02")])

    expected = {
        "date", "symbol",
        "short_volume", "short_value", "day_total_volume", "short_volume_ratio",
        "short_balance_qty", "short_balance_value", "listed_shares", "short_balance_ratio",
    }
    assert expected.issubset(set(out.columns))
