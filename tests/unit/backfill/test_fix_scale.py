"""fix_scale 스케일 보정 스크립트의 스키마 연동 단위 테스트.

`docs/specs/spreadsheet_column_refactor_contract.json`의
SCENARIO_LEGACY_MAPPING_FILE_REMOVAL 관련 검증을 포함합니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.backfill.fix_scale as fix_scale
from src.processing.schema import (
    LEGACY_RAW_TO_KOREAN_MAP,
    RAW_TO_STANDARD_MAP,
    STANDARD_TO_KOREAN_MAP,
)


def test_fix_scale_imports_schema_not_legacy_mapping() -> None:
    """fix_scale은 legacy_mapping 대신 schema 기반 매핑을 사용합니다."""
    source = Path(fix_scale.__file__).read_text(encoding="utf-8")
    assert "legacy_mapping" not in source
    assert "from src.processing.schema import RAW_TO_STANDARD_MAP, STANDARD_TO_KOREAN_MAP" in source
    assert "LEGACY_RAW_TO_KOREAN_MAP" in source


def test_rename_roundtrip_preserves_legacy_behavior() -> None:
    """fix_scale이 수행하는 정규화(표준 영문 -> 한글 표준)는 기존 괄호 매핑과 동일한 한글 컬럼을 산출합니다."""
    raw_cols = ["(매수날짜)", "(종목코드)", "(종가)", "(매수 가격)", "(매도 가격)", "(수익률, %)"]
    composed = (
        pd.DataFrame(columns=raw_cols)
        .rename(columns=RAW_TO_STANDARD_MAP)
        .rename(columns=STANDARD_TO_KOREAN_MAP)
    )
    legacy = pd.DataFrame(columns=raw_cols).rename(columns=LEGACY_RAW_TO_KOREAN_MAP)
    assert list(composed.columns) == list(legacy.columns)
    assert list(composed.columns) == ["매수날짜", "종목코드", "종가", "매수가격", "매도가격", "수익률"]


def test_inverse_rename_restores_bracket_headers() -> None:
    """DB 업데이트용 역변환(한글 -> 괄호)이 원래 괄호 헤더를 복원합니다."""
    inverse = {v: k for k, v in LEGACY_RAW_TO_KOREAN_MAP.items()}
    df = pd.DataFrame(columns=["매수날짜", "종목코드", "수익률", "종가"])
    restored = df.rename(columns=inverse)
    assert list(restored.columns) == ["(매수날짜)", "(종목코드)", "(수익률, %)", "(종가)"]
