"""DART API 설정 도메인 (AltDataSettings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AltDataSettings(BaseSettings):
    """Alt-data(대체데이터) 백필 접속 설정: OpenDART · KRX Open API 인증 키."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenDART: 암호화 볼트(.env.enc)는 OPENDART_API_KEY 를 쓰고, 구 설정은 DART_API_KEY 를 씀.
    OPENDART_API_KEY: str = Field(default="")
    DART_API_KEY: str = Field(default="")
    # KRX Open API (data-dbg.krx.co.kr, AUTH_KEY 헤더). 파생·지수 일별매매정보 주 경로.
    KRX_OPENAPI_KEY: str = Field(default="")

    @property
    def dart_key(self) -> str:
        """OpenDART 유효 키 (OPENDART_API_KEY 우선, 없으면 DART_API_KEY)."""
        return (self.OPENDART_API_KEY or self.DART_API_KEY).strip()
