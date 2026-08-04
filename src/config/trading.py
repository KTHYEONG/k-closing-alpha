"""트레이딩 설정 도메인 (TradingSettings).

조건검색 조건명, 차트 판단 임계값, 시나리오 우선순위, API 요청 제한 및
조건명에 의존하는 파생 경로를 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TradingSettings(BaseSettings):
    """당일 트레이딩 운영(조건검색/차트 판단/시나리오) 설정."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------
    # [데이터 수집 설정 (collect)]
    # ---------------------------------------------------------
    TARGET_CONDITION_NAME: str = "종가매매"
    OVERHEATED_CONDITION_NAME: str = "단기과열"
    NEW_HIGH_CONDITION_NAME: str = "신고가"
    NEAR_NEW_HIGH_CONDITION_NAME: str = "신고가 근접"
    UPPER_LIMIT_NEXT_DAY_CONDITION_NAME: str = "상한가 다음날"
    UPPER_LIMIT_CONDITION_NAME: str = "상한가"

    # API 요청 제한 및 지연 시간
    API_SEMAPHORE_LIMIT: int = 4
    API_SLEEP_INTERVAL: float = 0.2

    # 차트 필터링 설정
    EMA_PERIOD: int = 20
    SMA_PERIOD: int = 120
    SMA60_PERIOD: int = 60
    CANDLE_BODY_RATIO_THRESHOLD: float = 0.5
    GAP_UP_THRESHOLD: float = 0.1
    SMA_LOOKBACK_DAYS: int = 200
    SMA60_LOOKBACK_DAYS: int = 120
    EMA_LOOKBACK_DAYS: int = 60

    # AI 분석 기본 시나리오 (시트에서 새로운 유형 기록시 추가 필요)
    DEFAULT_SCENARIOS: list[str] = [
        "신고가",
        "상따",
        "신고가 근접",
        "거래량 폭증",
        "상한가 다음날",
        "120 돌파",
        "상승형 음봉",
    ]

    # 한글 요일 매핑
    DAY_NAME_MAP: dict[int, str] = {
        0: "월요일",
        1: "화요일",
        2: "수요일",
        3: "목요일",
        4: "금요일",
        5: "토요일",
        6: "일요일",
    }
