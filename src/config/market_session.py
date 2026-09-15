"""장 구분/세션 상수 (KRX 정규세션, NXT 애프터마켓, 의사결정 듀얼벤뉴)."""

from __future__ import annotations

KRX_CLOSE_MARKET_DIV_CODE: str = "J"
NXT_MARKET_DIV_CODE: str = "NX"
DECISION_WINDOW_START_HHMMSS: str = "152000"
DECISION_WINDOW_END_HHMMSS: str = "153000"
# 정수형 비교용 파생값 (문자열과 드리프트 방지)
DECISION_WINDOW_START_HMS: int = int(DECISION_WINDOW_START_HHMMSS)
# 정수형 비교용 파생값 (문자열과 드리프트 방지)
DECISION_WINDOW_END_HMS: int = int(DECISION_WINDOW_END_HHMMSS)
# 실패 시세 재조회는 스냅샷 영속이 결정창 종료(15:30:00) 전에 끝나도록 이 시각 이전에만 시작한다.
REALTIME_REQUOTE_DEADLINE_HHMMSS: str = "152800"
# 종가단일가 체결 확정 상태 코드 (KIS antc_mkop_cls_code; 결정창 진행중은 '121')
CLOSING_AUCTION_CONFIRMED_MKOP_CODE: str = "112"
# 확정 승격을 허용하는 최소 시각 (결정창 종료시각과 동일하므로 파생 정의)
CLOSING_AUCTION_CONFIRM_EARLIEST_HHMMSS: str = DECISION_WINDOW_END_HHMMSS
# 확정 재폴링 데드라인 (VI 임의연장 + 벤더 반영 지연 포괄)
CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS: str = "153300"
DEFAULT_BAR_INTERVAL_MINUTES: int = 1
INTRADAY_SESSION_REGULAR: str = "regular"
INTRADAY_SESSION_NXT_AFTERMARKET: str = "nxt_aftermarket"
INTRADAY_SESSION_NXT_PREMARKET: str = "nxt_premarket"
INTRADAY_SESSION_KRX_AFTERMARKET: str = "krx_aftermarket"
KRX_REGULAR_HOUR_FLOOR: str = "090000"
KRX_REGULAR_HOUR_CEIL: str = "153000"
KRX_AFTERMARKET_HOUR_FLOOR: str = "160000"
KRX_AFTERMARKET_HOUR_CEIL: str = "200000"
KRX_AFTERMARKET_START_DATE: str = "2026-09-14"
NXT_AFTERMARKET_HOUR_FLOOR: str = "154000"
NXT_AFTERMARKET_HOUR_CEIL: str = "200000"
NXT_PREMARKET_HOUR_FLOOR: str = "080000"
NXT_PREMARKET_HOUR_CEIL: str = "085000"
# 주문 결정 시각(결정창 시작, 룩어헤드 하한). 체결은 finalize_close 확정종가가 오라클이다.
PAPER_ENTRY_HHMMSS: str = "152000"
# D+1 청산은 모델 라벨(익일 시가)과 동일하게 KRX 시가단일가 체결가로 한다
PAPER_EXIT_OPEN_AUCTION_HHMMSS: str = KRX_REGULAR_HOUR_FLOOR
# 현재가 API의 stck_oprc는 시가단일가 체결 직후 반영 지연이 있어 이 시각 이후 조회만 시가로 신뢰한다
PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS: str = "090030"
