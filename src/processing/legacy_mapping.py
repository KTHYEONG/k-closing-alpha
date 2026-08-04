"""Legacy v1 전처리기 전용 컬럼/매핑 상수.

v1 `src/processing/preprocessor.py`에서만 사용되던 구형 데이터 매핑 상수를
분리 보관합니다. 메인 전처리 파이프라인(`src/processing/preprocessor.py`)에서
re-export하여 하위 호환성을 보장하고, `legacy/preprocessor.py`에서도 재사용합니다.
"""

from __future__ import annotations

# 컬럼 이름 상수로 관리
DATE_COL = "매수날짜"
TARGET_CLASSIFICATION = "Win"
TARGET_REGRESSION = "수익률"

RENAME_MAP = {
    "(매수날짜)": DATE_COL,
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
    "(수익률, %)": TARGET_REGRESSION,
    "(Win)": TARGET_CLASSIFICATION,
    "(kospi, %)": "kospi",
    "(kosdaq, %)": "kosdaq",
    "(시장구분)": "시장구분",
    "(수익 구간)": "수익_구간",
    "(중요 손실 지표)": "중요_손실_지표",
    "(프로그램_순매수)": "프로그램_순매수",
    "(체결강도)": "체결강도",
    "(v-kospi)": "v_kospi",
    "(v-kosdaq)": "v_kosdaq",
    # OHLC 데이터 추가
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
