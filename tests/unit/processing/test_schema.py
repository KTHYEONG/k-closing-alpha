"""schema 중앙 컬럼 스키마 모듈 단위 테스트.

`docs/specs/spreadsheet_column_refactor_contract.json`의 시나리오 기반 검증입니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.processing.schema import (
    ARCHIVE_COLUMN_ORDER,
    LEGACY_RAW_TO_KOREAN_MAP,
    RAW_TO_STANDARD_MAP,
    STANDARD_TO_KOREAN_MAP,
    StandardColumns,
    normalize_column_names,
)


def test_normalize_raw_bracket_columns() -> None:
    """SCENARIO_NORMALIZE_RAW_BRACKET_COLUMNS: 괄호 폼 컬럼이 표준 영문 컬럼명으로 매핑됩니다."""
    df = pd.DataFrame(columns=["(매수날짜)", "(수익률, %)"])
    res = normalize_column_names(df)
    assert list(res.columns) == ["trade_date", "net_return"]


def test_normalize_mixed_korean_and_bracket_columns() -> None:
    """한글 폼과 괄호 폼이 섞인 컬럼도 일괄 표준 영문 컬럼명으로 정규화됩니다."""
    df = pd.DataFrame(columns=["매수날짜", "종목코드", "(시가)", "(고가)"])
    res = normalize_column_names(df)
    assert list(res.columns) == ["trade_date", "stock_code", "open_price", "high_price"]


def test_normalize_unknown_columns_unchanged() -> None:
    """매핑에 없는 컬럼은 원래 이름을 유지합니다."""
    df = pd.DataFrame(columns=["알수없음", "(종목코드)"])
    res = normalize_column_names(df)
    assert list(res.columns) == ["알수없음", "stock_code"]


def test_standard_columns_constants() -> None:
    """표준 영문 컬럼 상수가 정의되어 있습니다."""
    assert StandardColumns.TRADE_DATE == "trade_date"
    assert StandardColumns.STOCK_CODE == "stock_code"
    assert StandardColumns.NET_RETURN == "net_return"
    assert StandardColumns.WIN_CLASSIFICATION == "Win"




def test_korean_to_standard_roundtrip() -> None:
    """괄호/한글 폼을 표준 영문으로 정규화 후 한글 표준으로 변환하면 원래 한글 폼이 복원됩니다."""
    df = pd.DataFrame(columns=["(매수날짜)", "(종목코드)", "(수익률, %)", "(종가)"])
    res = normalize_column_names(df).rename(columns=STANDARD_TO_KOREAN_MAP)
    assert list(res.columns) == ["매수날짜", "종목코드", "수익률", "종가"]


def test_legacy_map_compat_with_standard() -> None:
    """LEGACY_RAW_TO_KOREAN_MAP은 표준 변환 왕복과 동일한 결과를 산출합니다."""
    raw_cols = ["(매수날짜)", "(종목코드)", "(매수 가격)", "(매도 가격)", "(수익률, %)"]
    roundtrip = normalize_column_names(pd.DataFrame(columns=raw_cols)).rename(
        columns=STANDARD_TO_KOREAN_MAP
    )
    legacy = pd.DataFrame(columns=raw_cols).rename(columns=LEGACY_RAW_TO_KOREAN_MAP)
    assert list(roundtrip.columns) == list(legacy.columns)


def test_legacy_mapping_file_removal() -> None:
    """SCENARIO_LEGACY_MAPPING_FILE_REMOVAL: legacy_mapping.py가 제거되고 의존 모듈은 schema.py를 참조합니다."""
    legacy_path = Path("src/processing/legacy_mapping.py")
    assert not legacy_path.exists()

    # 실시간 피처 프레임은 legacy_mapping 을 참조하지 않습니다(champion 전용
    # normalize_column_names 소비자인 build_snapshot_features 는 제거됨).
    features_src = Path("src/serving/realtime/features.py").read_text(encoding="utf-8")
    assert "legacy_mapping" not in features_src

    # 아카이브 데이터셋 빌더는 RAW_TO_STANDARD_MAP 을 schema 로부터 참조합니다 (존재하는 경우).
    legacy_dataset_path = Path("legacy/ml_research/features/dataset.py")
    if legacy_dataset_path.exists():
        dataset_src = legacy_dataset_path.read_text(encoding="utf-8")
        assert "legacy_mapping" not in dataset_src
        assert "from src.processing.schema import RAW_TO_STANDARD_MAP" in dataset_src

    # src/backfill/fix_scale.py 는 스프레드시트 매매일지 생태계 제거로 함께
    # 삭제되었으므로(spreadsheet_legacy_removal), 더 이상 legacy_mapping 참조
    # 여부를 검증할 대상이 아닙니다.
    assert RAW_TO_STANDARD_MAP["(매수날짜)"] == "trade_date"





def test_archive_column_order_is_minimal_and_carries_admitted() -> None:
    from src.processing.schema import ARCHIVE_COLUMN_ORDER

    # Then: the reranker-required set plus the PIT admission verdict and data-quality flags
    assert ARCHIVE_COLUMN_ORDER == [
        "스냅샷_날짜", "종목코드", "종목명", "시장구분", "시가", "고가", "저가", "종가",
        "전일종가", "거래량", "거래대금", "시가총액", "기관_순매수", "외국인_순매수",
        "등락률", "kospi", "kosdaq", "v_kospi", "admitted", "수급_실패", "지수_실패",
        "현재가_실패", "결정_종가", "종가_확정",
    ]

    # Then: champion-era and flat level-1 orderbook columns are gone (the full
    # ladder lives in the orderbook partition store instead)
    dropped = {
        "시나리오", "테마_섹터", "선정순위", "총_종목수", "평균_거래대금",
        "체결강도", "프로그램_순매수", "v_kosdaq", "sor_effective_price",
        "krx_현재가", "nxt_현재가", "krx_매도호가1", "nxt_매수잔량",
    }
    assert dropped.isdisjoint(ARCHIVE_COLUMN_ORDER)



def test_standard_column_order_is_removed_from_schema() -> None:
    import src.processing.schema as schema

    # Then: the spreadsheet copy-paste column order no longer exists
    assert not hasattr(schema, "STANDARD_COLUMN_ORDER")
def test_archive_column_order_registers_finalization_columns_at_tail() -> None:
    from src.processing.schema import (
        ARCHIVE_COLUMN_ORDER,
        CLOSE_CONFIRMED_COL,
        DECISION_CLOSE_COL,
        RAW_TO_STANDARD_MAP,
        STANDARD_TO_KOREAN_MAP,
    )

    assert ARCHIVE_COLUMN_ORDER[-2:] == [DECISION_CLOSE_COL, CLOSE_CONFIRMED_COL]
    assert ARCHIVE_COLUMN_ORDER.index("종가") < ARCHIVE_COLUMN_ORDER.index(DECISION_CLOSE_COL)
    assert ARCHIVE_COLUMN_ORDER.count(DECISION_CLOSE_COL) == 1
    assert RAW_TO_STANDARD_MAP[DECISION_CLOSE_COL] == "decision_close_price"
    assert RAW_TO_STANDARD_MAP[CLOSE_CONFIRMED_COL] == "close_confirmed"
    assert STANDARD_TO_KOREAN_MAP["decision_close_price"] == DECISION_CLOSE_COL
    assert STANDARD_TO_KOREAN_MAP["close_confirmed"] == CLOSE_CONFIRMED_COL


