"""한투 KIS API 설정 도메인 (KisSettings).

AppKey/Secret/Account/HTS_ID 및 KIS 접속 정보 딕셔너리를 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field

from src.config._env import EnvSettings


class KisSettings(EnvSettings):
    """한투 KIS(한국투자증권) OpenAPI 접속 설정.

    Attributes:
        KIS_TOKEN_LOCK_TIMEOUT_SECONDS: Bound on waiting for the host-wide KIS token
            lock. A healthy holder (another KCA unit or the external krx-alpha consumer)
            keeps the lock for its whole issuance: up to 3 POST attempts under a 60 s
            session total plus backoff (~183 s). The bound must exceed that so a slow but
            healthy issuance is adopted from the cache instead of failing the waiter, and
            stays finite so a hung holder fails closed instead of blocking forever.
    """

    KIS_APP_KEY: str = Field(default="")
    KIS_APP_SECRET: str = Field(default="")
    KIS_ACCOUNT_ID: str = Field(default="")
    KIS_HTS_ID: str = Field(default="")
    KIS_DATA_ROLE: str = Field(default="batch")
    KIS_TOKEN_CACHE_DIR: Path = Field(default_factory=lambda: Path.home() / ".cache" / "kis")
    KIS_BASE_URL: str = "https://openapi.koreainvestment.com:9443"
    KIS_TOKEN_LOCK_TIMEOUT_SECONDS: float = Field(default=240.0, gt=0, allow_inf_nan=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def KIS_API_CONFIG(self) -> dict[str, str]:
        """KIS API 접속 정보 딕셔너리 (app_key/app_secret/account_id/hts_id)."""
        return {
            "app_key": self.KIS_APP_KEY,
            "app_secret": self.KIS_APP_SECRET,
            "account_id": self.KIS_ACCOUNT_ID,
            "hts_id": self.KIS_HTS_ID,
        }
