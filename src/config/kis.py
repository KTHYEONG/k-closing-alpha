"""한투 KIS API 설정 도메인 (KisSettings).

AppKey/Secret/Account/HTS_ID 및 KIS 접속 정보 딕셔너리를 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class KisSettings(BaseSettings):
    """한투 KIS(한국투자증권) OpenAPI 접속 설정."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    KIS_APP_KEY: str = Field(default="")
    KIS_APP_SECRET: str = Field(default="")
    KIS_ACCOUNT_ID: str = Field(default="")
    KIS_HTS_ID: str = Field(default="")
    KIS_BASE_URL: str = "https://openapi.koreainvestment.com:9443"

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

    # ---------------------------------------------------------
    # [스펙 하위 호환 별칭] (spec: kis_app_key, kis_app_secret, ...)
    # ---------------------------------------------------------
    @property
    def kis_app_key(self) -> str:
        return self.KIS_APP_KEY

    @property
    def kis_app_secret(self) -> str:
        return self.KIS_APP_SECRET

    @property
    def kis_account_id(self) -> str:
        return self.KIS_ACCOUNT_ID
