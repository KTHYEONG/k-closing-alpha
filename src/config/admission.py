"""Host-wide broker admission configuration (AdmissionSettings)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from src.config._env import EnvSettings


class AdmissionSettings(EnvSettings):
    """Host-wide broker admission configuration.

    Admission state is shared through files in a host directory mounted into every
    container. A process's admission class bounds how far ahead it may book a slot:
    critical work books without bound, while standard/bulk work may only book within
    a short lead, so a critical arrival waits at most that lead even when bulk jobs
    hold thousands of queued requests.

    Attributes:
        BROKER_ADMISSION_DIR: Shared state directory; None means KIS_TOKEN_CACHE_DIR
            (already mounted host-wide as ~/.cache/kis for KCA units and krx-collector).
        BROKER_ADMISSION_CLASS: This process's class; unset processes (ad-hoc tools,
            workstation runs) default to the lowest class.
        BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS: Maximum reservation lead for standard work.
        BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS: Maximum reservation lead for bulk work.
        BROKER_ADMISSION_REQUIRE_SHARED: "auto" enforces the host marker only inside a
            container (/.dockerenv present); "always"/"never" force the check on/off.
        BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS: Bound on waiting for a token-store lock.
    """

    BROKER_ADMISSION_DIR: Path | None = Field(default=None)
    BROKER_ADMISSION_CLASS: Literal["critical", "standard", "bulk"] = Field(default="bulk")
    BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    BROKER_ADMISSION_REQUIRE_SHARED: Literal["auto", "always", "never"] = Field(default="auto")
    BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_lead_order(self) -> Self:
        if self.BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS > self.BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS:
            raise ValueError("bulk lead must not exceed standard lead")
        return self
