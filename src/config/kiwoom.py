"""키움증권 REST API 설정 도메인 (KiwoomSettings)."""

from __future__ import annotations

from pydantic import AliasChoices, Field

from src.config._env import EnvSettings


class KiwoomSettings(EnvSettings):
    """Kiwoom Securities REST API connection settings.

    Field names use the vendor's correct spelling (KIWOOM). The historical
    misspelled environment names (KIWOM_*) are still accepted as aliases because
    the workstation secret source is shared with another repository and the VPS
    env file is re-provisioned separately from code deploys; the new name wins
    when both are present.
    """

    KIWOOM_APP_KEY: str = Field(default="", validation_alias=AliasChoices("KIWOOM_APP_KEY", "KIWOM_APP_KEY"))
    KIWOOM_SECRET_KEY: str = Field(default="", validation_alias=AliasChoices("KIWOOM_SECRET_KEY", "KIWOM_SECRET_KEY"))
    KIWOOM_BASE_URL: str = Field(default="https://api.kiwoom.com", validation_alias=AliasChoices("KIWOOM_BASE_URL", "KIWOM_BASE_URL"))
