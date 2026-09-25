"""Central column-name contracts: vendor/archive raw → standard English map, archive column order, and fail-closed flag columns."""

from __future__ import annotations

# 결정시점(15:20~15:30 동결) 가격 보존 컬럼 (라이브 판정이 실제 사용한 값)
DECISION_CLOSE_COL: str = "결정_종가"
# 종가단일가 확정 이후 값 여부 (fail-closed 플래그; 미확정은 EOD 진실이 아님)
CLOSE_CONFIRMED_COL: str = "종가_확정"


# 현재가 TR 실패 표식 (fail-closed 플래그; 0값 위조 금지)
QUOTE_FAILED_COL: str = "현재가_실패"
# 벤더 rt_cd 성공에도 값이 비정상(0/일관성 붕괴)인 표식 (fail-closed 플래그; 현재가_실패와 별개로 값 자체를 검증)
PRICE_ANOMALY_COL: str = "가격_비정상"


# 스프레드시트 원본(괄호/한글 폼) -> 표준 영문 컬럼명 매핑
RAW_TO_STANDARD_MAP: dict[str, str] = {
    "매수날짜": "trade_date",
    "종목코드": "stock_code",
    "(매수날짜)": "trade_date",
    "(종목코드)": "stock_code",
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
    "수익률": "net_return",
    "(수익률, %)": "net_return",
    "(Win)": "Win",
    "(차트통과)": "차트통과",
    "(수익 구간)": "수익_구간",
    "(중요 손실 지표)": "중요_손실_지표",
    "(ema5)": "ema5",
    "(ema10)": "ema10",
    "(ema20)": "ema20",
    DECISION_CLOSE_COL: "decision_close_price",
    CLOSE_CONFIRMED_COL: "close_confirmed",
}


# 구글 스프레드시트(조건검색) 26개 열과 1:1 대응하는 표준 아카이브 컬럼 순서
ARCHIVE_COLUMN_ORDER: list[str] = [
    "스냅샷_날짜",
    "종목코드",
    "종목명",
    "시장구분",
    "시가",
    "고가",
    "저가",
    "종가",
    "전일종가",
    "거래량",
    "거래대금",
    "시가총액",
    "기관_순매수",
    "외국인_순매수",
    "등락률",
    "kospi",
    "kosdaq",
    "v_kospi",
    "market_breadth",
    "admitted",
    "수급_실패",
    "지수_실패",
    "시장폭_실패",
    QUOTE_FAILED_COL,
    PRICE_ANOMALY_COL,
    DECISION_CLOSE_COL,
    CLOSE_CONFIRMED_COL,
]

