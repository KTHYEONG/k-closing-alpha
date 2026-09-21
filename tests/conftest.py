"""공통 pytest fixture 모듈.

Settings 경로 해석, 샘플 매매일지 DataFrame 등을 제공합니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_kis_token_state(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    from src import settings as live_settings

    monkeypatch.setattr(live_settings, "KIS_TOKEN_CACHE_DIR", tmp_path_factory.mktemp("kis_cache"))
    # resolve_host_data_credentials는 풀 미선언시 fail-closed(ValueError)이므로, 그냥
    # 지우기만 하면 kis_data_client_kwargs()의 기본(무인자) 호출부를 쓰는 테스트가
    # 전부 이 예외로 깨진다. 실제 호스트 시크릿이 새어들지 않도록 결정론적 더미
    # 풀 1개로 고정해, 격리 의도(진짜 자격증명 미사용)는 유지하면서 기본 호출부가
    # 항상 해석 가능하게 한다.
    monkeypatch.setenv("KIS_DATA_SLOTS", "1")
    monkeypatch.setenv("KIS_HOST_DATA_SLOTS", "1")
    monkeypatch.setenv("KIS_DATA_1_APP_KEY", "test-data-key")
    monkeypatch.setenv("KIS_DATA_1_APP_SECRET", "test-data-secret")
    monkeypatch.delenv("KIS_DATA_1_HTS_ID", raising=False)
    monkeypatch.delenv("KIS_DATA_ROLE", raising=False)
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    monkeypatch.delenv("KIS_HTS_ID", raising=False)
    # 실호스트의 KIS_DECISION_SHARD_SLOTS가 새어들면 더미 풀(슬롯 1개)과 충돌해
    # parse_decision_shard_credentials가 fail-closed ValueError를 던진다(실측:
    # 2026-09-17 code-sync가 이 누락으로 매일 실패).
    monkeypatch.delenv("KIS_DECISION_SHARD_SLOTS", raising=False)
    # CollectionSettings(**partial)/`_env_file=None` 은 클래스 전용 dotenv만 끄고
    # OS 환경변수 소스는 그대로 남긴다. 실호스트의 COLLECTION_AUCTION_ENABLED/
    # COLLECTION_ALTDATA_ENABLED/COLLECTION_RESEARCH_SLOTS 등이 새어들면
    # model_validator("legacy operating mode...")와 충돌한다(실측: 2026-09-20
    # kca-code-sync가 이 누락으로 며칠간 test_gate_failed 반복, 배포 정지).
    monkeypatch.delenv("COLLECTION_RAW_ENABLED", raising=False)
    monkeypatch.delenv("COLLECTION_AUCTION_ENABLED", raising=False)
    monkeypatch.delenv("COLLECTION_ALTDATA_ENABLED", raising=False)
    monkeypatch.delenv("COLLECTION_RESEARCH_SLOTS", raising=False)
    monkeypatch.delenv("COLLECTION_ALTDATA_EXTRA_SLOTS", raising=False)


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """임시 디렉토리를 가리키는 Settings 인스턴스를 반환합니다."""
    return Settings(
        BASE_DIR=tmp_path,
        DATA_DIR=tmp_path / "data",
        CONFIGS_DIR=tmp_path / "configs",
    )


@pytest.fixture
def sample_trade_df() -> pd.DataFrame:
    """스케일 보정 테스트용 샘플 매매일지 DataFrame."""
    return pd.DataFrame(
        {
            "종목코드": ["005930", "000660", "005930"],
            "매수날짜": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "종가": [70_000, 30_000, 70_000],
            "매수가격": [7_000, 30_000, 7_000],  # 005930: 1/10 스케일 오류
            "매도가격": [7_200, 31_000, 7_200],
        }
    )
