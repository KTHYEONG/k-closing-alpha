"""중앙 컬럼 스키마 모듈.

스프레드시트 원본 컬럼명(괄호 폼/한글 폼)과 표준 영문 컬럼명 간의
단일 원천(Single Source of Truth) 매핑을 정의하고, DataFrame 컬럼명
정규화 유틸리티를 제공합니다.

`docs/specs/spreadsheet_column_refactor.md` 설계를 구현합니다.
"""

from __future__ import annotations

import pandas as pd


class StandardColumns:
    """표준 영문 컬럼 상수 네임스페이스.

    ML 학습 파이프라인과 스프레드시트 표준 출력이 공동으로 참조하는
    정규화된 영문 컬럼명을 정의합니다.
    """

    TRADE_DATE = "trade_date"
    STOCK_CODE = "stock_code"
    NET_RETURN = "net_return"
    WIN_CLASSIFICATION = "Win"


# 스프레드시트 원본(괄호/한글 폼) -> 표준 영문 컬럼명 매핑
RAW_TO_STANDARD_MAP: dict[str, str] = {
    "매수날짜": StandardColumns.TRADE_DATE,
    "종목코드": StandardColumns.STOCK_CODE,
    "(매수날짜)": StandardColumns.TRADE_DATE,
    "(종목코드)": StandardColumns.STOCK_CODE,
    "시가": "open_price",
    "(시가)": "open_price",
    "고가": "high_price",
    "(고가)": "high_price",
    "저가": "low_price",
    "(저가)": "low_price",
    "종가": "close_price",
    "(종가)": "close_price",
    "전일종가": "prev_close_price",
    "(전일종가)": "prev_close_price",
    "시가총액": "market_cap_100m",
    "(시가총액, 억)": "market_cap_100m",
    "거래대금": "trade_value_100m",
    "(거래대금, 억)": "trade_value_100m",
    "등락률": "change_rate",
    "(등락률)": "change_rate",
    "선정순위": "selection_rank",
    "(선정 순위)": "selection_rank",
    "기관_순매수": "inst_net_buy",
    "(기관_순매수)": "inst_net_buy",
    "외국인_순매수": "foreign_net_buy",
    "(외국인_순매수)": "foreign_net_buy",
    "프로그램_순매수": "prog_net_buy",
    "(프로그램_순매수)": "prog_net_buy",
    "체결강도": "volume_power",
    "(체결강도)": "volume_power",
    "시장구분": "market_type",
    "(시장구분)": "market_type",
    "총_종목수": "total_candidate_count",
    "(총 종목 수)": "total_candidate_count",
    "평균_거래대금": "avg_trade_value",
    "(평균 거래대금)": "avg_trade_value",
    "kospi": "kospi_change",
    "(kospi, %)": "kospi_change",
    "kosdaq": "kosdaq_change",
    "(kosdaq, %)": "kosdaq_change",
    "v_kospi": "v_kospi",
    "(v-kospi)": "v_kospi",
    "v_kosdaq": "v_kosdaq",
    "(v-kosdaq)": "v_kosdaq",
    "거래량": "volume",
    "(거래량)": "volume",
    "테마_섹터": "theme_sector",
    "(테마/섹터)": "theme_sector",
    "차트분석": "chart_analysis",
    "(차트분석)": "chart_analysis",
    "매수가격": "buy_price",
    "(매수 가격)": "buy_price",
    "매도가격": "sell_price",
    "(매도 가격)": "sell_price",
    "수익률": StandardColumns.NET_RETURN,
    "(수익률, %)": StandardColumns.NET_RETURN,
    "(Win)": StandardColumns.WIN_CLASSIFICATION,
    "(차트통과)": "차트통과",
    "(수익 구간)": "수익_구간",
    "(중요 손실 지표)": "중요_손실_지표",
    "(ema5)": "ema5",
    "(ema10)": "ema10",
    "(ema20)": "ema20",
}


# 구글 시트 직접 동기화 헤더(괄호 폼) -> 한글 표준 컬럼명 매핑
# (backfill/fix_scale 등 한글 컬럼 기반 파이프라인의 레거시 헤더 정규화 및 원복용)
LEGACY_RAW_TO_KOREAN_MAP: dict[str, str] = {
    "(매수날짜)": "매수날짜",
    "(종목코드)": "종목코드",
    "(시가총액, 억)": "시가총액",
    "(거래대금, 억)": "거래대금",
    "(등락률)": "등락률",
    "(선정 순위)": "선정순위",
    "(기관_순매수)": "기관_순매수",
    "(외국인_순매수)": "외국인_순매수",
    "(테마/섹터)": "테마_섹터",
    "(차트통과)": "차트통과",
    "(차트분석)": "차트분석",
    "(매수 가격)": "매수가격",
    "(매도 가격)": "매도가격",
    "(총 종목 수)": "총_종목수",
    "(평균 거래대금)": "평균_거래대금",
    "(수익률, %)": "수익률",
    "(Win)": "Win",
    "(kospi, %)": "kospi",
    "(kosdaq, %)": "kosdaq",
    "(시장구분)": "시장구분",
    "(수익 구간)": "수익_구간",
    "(중요 손실 지표)": "중요_손실_지표",
    "(프로그램_순매수)": "프로그램_순매수",
    "(체결강도)": "체결강도",
    "(v-kospi)": "v_kospi",
    "(v-kosdaq)": "v_kosdaq",
    "(시가)": "시가",
    "(고가)": "고가",
    "(저가)": "저가",
    "(종가)": "종가",
    "(전일종가)": "전일종가",
    "(ema5)": "ema5",
    "(ema10)": "ema10",
    "(ema20)": "ema20",
    "(거래량)": "거래량",
}


# 표준 영문 컬럼명 -> 스프레드시트 한글 표준 컬럼명 매핑
# (일부 파이프라인이 표준 영문으로 정규화 후 한글 표준으로 왕복 변환 시 사용)
STANDARD_TO_KOREAN_MAP: dict[str, str] = {
    StandardColumns.TRADE_DATE: "매수날짜",
    StandardColumns.STOCK_CODE: "종목코드",
    "open_price": "시가",
    "high_price": "고가",
    "low_price": "저가",
    "close_price": "종가",
    "prev_close_price": "전일종가",
    "market_cap_100m": "시가총액",
    "trade_value_100m": "거래대금",
    "change_rate": "등락률",
    "selection_rank": "선정순위",
    "inst_net_buy": "기관_순매수",
    "foreign_net_buy": "외국인_순매수",
    "prog_net_buy": "프로그램_순매수",
    "volume_power": "체결강도",
    "market_type": "시장구분",
    "total_candidate_count": "총_종목수",
    "avg_trade_value": "평균_거래대금",
    "kospi_change": "kospi",
    "kosdaq_change": "kosdaq",
    "v_kospi": "v_kospi",
    "v_kosdaq": "v_kosdaq",
    "volume": "거래량",
    "theme_sector": "테마_섹터",
    "chart_analysis": "차트분석",
    "buy_price": "매수가격",
    "sell_price": "매도가격",
    StandardColumns.NET_RETURN: "수익률",
    StandardColumns.WIN_CLASSIFICATION: "Win",
}


# 구글 스프레드시트(조건검색) 26개 열과 1:1 대응하는 표준 아카이브 컬럼 순서
ARCHIVE_COLUMN_ORDER: list[str] = [
    "스냅샷_날짜",
    "종목코드",
    "종목명",
    "시가",
    "고가",
    "저가",
    "종가",
    "전일종가",
    "시가총액",
    "거래대금",
    "등락률",
    "선정순위",
    "기관_순매수",
    "외국인_순매수",
    "프로그램_순매수",
    "체결강도",
    "시장구분",
    "총_종목수",
    "평균_거래대금",
    "kospi",
    "kosdaq",
    "v_kospi",
    "v_kosdaq",
    "거래량",
    "테마_섹터",
    "시나리오",
    "krx_현재가",
    "nxt_현재가",
    "sor_effective_price",
    "krx_매도호가1",
    "krx_매수호가1",
    "krx_매도잔량",
    "krx_매수잔량",
    "nxt_매도호가1",
    "nxt_매수호가1",
    "nxt_매도잔량",
    "nxt_매수잔량",
]


# 조건검색 수집 데이터 표준 열 순서 (스프레드시트 복사/붙여넣기 호환)
STANDARD_COLUMN_ORDER: list[str] = [
    "시나리오",
    "종목명",
    "종목코드",
    "시가",
    "고가",
    "저가",
    "종가",
    "전일종가",
    "시가총액",
    "거래대금",
    "등락률",
    "선정순위",
    "기관_순매수",
    "외국인_순매수",
    "프로그램_순매수",
    "체결강도",
    "시장구분",
    "총_종목수",
    "평균_거래대금",
    "kospi",
    "kosdaq",
    "v_kospi",
    "v_kosdaq",
    "거래량",
    "krx_현재가",
    "nxt_현재가",
    "sor_effective_price",
    "krx_매도호가1",
    "krx_매수호가1",
    "krx_매도잔량",
    "krx_매수잔량",
    "nxt_매도호가1",
    "nxt_매수호가1",
    "nxt_매도잔량",
    "nxt_매수잔량",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 컬럼명을 표준 영문 이름으로 일괄 변환합니다.

    스프레드시트의 괄호 폼(``(매수날짜)`` 등) 및 한글 폼(``매수날짜``) 컬럼명을
    ``RAW_TO_STANDARD_MAP`` 기준 표준 영문 컬럼명으로 매핑합니다. 매핑되지 않는
    컬럼은 원래 이름을 유지합니다.
    """
    return df.rename(columns=RAW_TO_STANDARD_MAP)
