"""장 구분/세션 상수 (KRX 정규세션, NXT 애프터마켓, 의사결정 듀얼벤뉴)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

KRX_CLOSE_MARKET_DIV_CODE: str = "J"
NXT_MARKET_DIV_CODE: str = "NX"
# endpoint→venue은 고정된 벤더 계약이다; 빈 env 기본값은 레거시 경로 제거 이후 모든 Kiwoom/LS 시도를 조용히 비인증 처리했다.
# "toss:toss-candles"→KRX mapping is valid only for bars accepted by the basis gate (toss_basis_verdict).
VERIFIED_CHART_ROUTES: Mapping[str, str] = MappingProxyType(
    {
        "kiwoom:ka10079": "KRX",
        "kiwoom:ka10079-nx": "NXT",
        "kiwoom:ka10080": "NXT",
        "ls:t8411": "KRX",
        "ls:t8412": "KRX",
        "toss:toss-candles": "KRX",
    }
)
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
BAR_STAMP_START: str = "start"
BAR_STAMP_END: str = "end"
# Measured on stored partitions (2026-10-04): KIS/Kiwoom stamp a 1m bar at its minute start (regular 090000..151900,
# NXT 154000..), LS at its minute end (090100..152000). Measured 2026-10-04: Toss 09:01 equals the KIS 09:00 bar
# (END-stamped, 090100..153000); OHLC/volume matched KIS on 29/40 sampled symbol-days. Unlisted vendors must fail closed, never default.
INTRADAY_BAR_STAMP_CONVENTION: Mapping[str, str] = MappingProxyType(
    {"kis": BAR_STAMP_START, "kiwoom": BAR_STAMP_START, "ls": BAR_STAMP_END, "toss": BAR_STAMP_END}
)
INTRADAY_SESSION_REGULAR: str = "regular"
INTRADAY_SESSION_REGULAR_CONSOLIDATED: str = "regular_consolidated"
INTRADAY_SESSION_NXT_AFTERMARKET: str = "nxt_aftermarket"
INTRADAY_SESSION_NXT_PREMARKET: str = "nxt_premarket"
INTRADAY_SESSION_KRX_AFTERMARKET: str = "krx_aftermarket"
# Every partition session the store verifies; unknown tags fail closed at write/path time.
INTRADAY_VERIFIED_SESSIONS: frozenset[str] = frozenset(
    {
        INTRADAY_SESSION_REGULAR,
        INTRADAY_SESSION_REGULAR_CONSOLIDATED,
        INTRADAY_SESSION_NXT_PREMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_KRX_AFTERMARKET,
    }
)
KRX_REGULAR_HOUR_FLOOR: str = "090000"
KRX_REGULAR_HOUR_CEIL: str = "153000"
KRX_AFTERMARKET_HOUR_FLOOR: str = "160000"
KRX_AFTERMARKET_HOUR_CEIL: str = "200000"
KRX_AFTERMARKET_START_DATE: str = "2026-09-14"
# 2026-09-29 이후 세션부터 애프터마켓 틱 매니페스트가 완결성 조건에 필요하다(이전 날짜에는 존재한 적이 없음)
AFTERMARKET_TICKS_START_DATE: str = "2026-09-29"
# 정규장 분봉·틱은 15:30 종가 확정 후 정산되므로 15:40 이후에만 당일분으로 아카이브한다(kca-archive-intraday-regular.timer와 동일)
ARCHIVE_REGULAR_READY_HHMMSS: str = "154000"
# NXT/KRX 애프터마켓은 20:00 종료 후 20:05 이후에만 당일분으로 아카이브한다(kca-archive-intraday.timer와 동일)
ARCHIVE_AFTERMARKET_READY_HHMMSS: str = "200500"
# kca-price-ingest.timer의 평일 저녁 슬롯(KIS flows/index); 테이프 스윕 기본 데드라인 산정의 기준점이다.
PRICE_INGEST_EVENING_HHMMSS: str = "213000"
# 테이프 스윕(kca-tape-sweep.timer, 20:35 발화)은 이 시각 이후 새 Kiwoom 워크를 열지 않는다 — 21:30 가격 수집과
# 겹치지 않도록 15분 여유를 둔다. TimeoutStartSec=3600이 20:35 발화 기준 외곽 안전판(21:35)으로 남는다.
TAPE_SWEEP_DEADLINE_HHMMSS: str = "211500"
NXT_AFTERMARKET_HOUR_FLOOR: str = "154000"
NXT_AFTERMARKET_HOUR_CEIL: str = "200000"
NXT_PREMARKET_HOUR_FLOOR: str = "080000"
NXT_PREMARKET_HOUR_CEIL: str = "085000"
# NXT 개시일; 이전 날짜에는 NXT 봉이 존재하지 않는다.
NXT_START_DATE: str = "2025-03-04"
# 주문 결정 시각(결정창 시작, 룩어헤드 하한). 체결은 finalize_close 확정종가가 오라클이다.
PAPER_ENTRY_HHMMSS: str = "152000"
# D+1 청산은 모델 라벨(익일 시가)과 동일하게 KRX 시가단일가 체결가로 한다
PAPER_EXIT_OPEN_AUCTION_HHMMSS: str = KRX_REGULAR_HOUR_FLOOR
# 현재가 API의 stck_oprc는 시가단일가 체결 직후 반영 지연이 있어 이 시각 이후 조회만 시가로 신뢰한다
PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS: str = "090030"
# 가상 청산은 시가단일가 체결 이후 이 시각까지만 기록한다(그 이후 기록은 실계좌가 재현할 수 없는 시가 체결이다)
PAPER_EXIT_WINDOW_END_HHMMSS: str = "093000"
# 이 시간보다 이르게 기동되면 대기 대신 SKIP한다 — 당일 09:01 정규 발화가 처리하며, 대기가 TimeoutStartSec(15분)를 넘지 않게 한다
PAPER_EXIT_MAX_PRESTART_WAIT_SECONDS: int = 600
# 확정 종가는 결정창 종료 이전에 존재할 수 없으므로 진입 기록의 하한은 결정창 종료와 같다(파생 정의)
PAPER_ENTRY_EARLIEST_HHMMSS: str = DECISION_WINDOW_END_HHMMSS
# 다음 달력일 이 시각 이후 진입 기록은 거부한다 — 다음 시가 청산(최조 08:50:30 기동)보다 앞서야 로트가 청산 대상에 포함된다
PAPER_ENTRY_CATCHUP_DEADLINE_HHMMSS: str = "083000"
# Calendar-day bound for previous-trading-day search, spanning the longest Seollal/Chuseok closures.
MAX_PREV_TRADING_DAY_LOOKBACK: int = 15
