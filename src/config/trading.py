"""Trading-operation settings domain (TradingSettings): API concurrency limit and paper-trading sizing."""

from __future__ import annotations

from pydantic import Field

from src.config._env import EnvSettings


class TradingSettings(EnvSettings):
    """당일 운영 API 동시성 한도 및 모의 운용 사이징 설정."""

    # API 요청 제한
    API_SEMAPHORE_LIMIT: int = 8

    # 모의 운용 시드(원). 정수 주식수 산정의 분자.
    PAPER_SEED_CAPITAL: int = 10_000_000

    # 결정가(15:20 스냅샷) 대비 종가 상승분을 흡수하는 사이징 여유(bp). 0이면 결정가 그대로 사이징한다.
    PAPER_ENTRY_SIZING_BUFFER_BP: float = Field(default=0.0, ge=0.0)
