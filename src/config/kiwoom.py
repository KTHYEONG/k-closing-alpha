"""키움증권 REST API 설정 도메인 (KiwoomSettings)."""

from __future__ import annotations

import re
from typing import Any

from pydantic import AliasChoices, Field, field_validator

from src.config._env import EnvSettings

_WINDOW_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


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
    KIWOOM_TOKEN_PROTECTED_WINDOWS: tuple[str, ...] = Field(default=("15:10-15:40",))
    """KST HH:MM-HH:MM windows in which the shared Kiwoom token must not expire (decision capture)."""

    @field_validator("KIWOOM_TOKEN_PROTECTED_WINDOWS", mode="before")
    @classmethod
    def _parse_protected_windows(cls, v: Any) -> Any:
        if isinstance(v, str):
            text = v.strip()
            if not text:
                return ()
            if text.startswith("["):
                import json

                items = json.loads(text)
                return tuple(str(item).strip() for item in items if str(item).strip())
            return tuple(item.strip() for item in text.split(",") if item.strip())
        return v

    @field_validator("KIWOOM_TOKEN_PROTECTED_WINDOWS", mode="after")
    @classmethod
    def _check_protected_windows(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for entry in v:
            match = _WINDOW_RE.fullmatch(str(entry))
            if match is None:
                raise ValueError(f"protected windows must be HH:MM-HH:MM: {entry!r}")
            sh, sm, eh, em = (int(g) for g in match.groups())
            if not (0 <= sh < 24 and 0 <= eh < 24 and 0 <= sm < 60 and 0 <= em < 60):
                raise ValueError(f"protected windows must be HH:MM-HH:MM: {entry!r}")
            if (sh, sm) >= (eh, em):
                raise ValueError(f"protected window start must precede end: {entry!r}")
        return v
