"""Acquisition limits for optional research work (CollectionSettings)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.data.capture_contracts import SessionClock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CollectionSettings(BaseSettings):
    """Define acquisition limits independently of trading selection thresholds.

    These settings bound optional research work; they must never silently reduce
    the all-candidate baseline or change a deployed model's mandatory inputs.

    Attributes:
        COLLECTION_ROOT: Optional owner-local override, default None; resolved beneath HISTORY_DIR/capture.
        COLLECTION_RAW_ENABLED: Enable first-party provenance, default True.
        COLLECTION_AUCTION_ENABLED: Enable independently budgeted sweeps, default False.
        COLLECTION_ALTDATA_ENABLED: Enable incremental slow-data jobs, default False.
        COLLECTION_RESEARCH_SLOTS: Explicit research key slots, default empty.
        COLLECTION_ALTDATA_EXTRA_SLOTS: Extra KIS data slots for alt-data fan-out, default empty.
        COLLECTION_AUCTION_INTERVAL_SECONDS: Closing sweep interval, default 60.
        COLLECTION_REQUEST_TIMEOUT_SECONDS: Total call timeout, default 5.0.
        COLLECTION_CONCURRENCY_PER_KEY: In-flight limit, default 8.
        COLLECTION_CHART_MAX_PAGES: Normal page budget, default 30.
        COLLECTION_TICK_REPAIR_MAX_PAGES: Explicit total repair budget, default 120.
        COLLECTION_ARROW_BATCH_ROWS: Bounded rewrite batch size, default 65536.
        COLLECTION_MAX_RSS_MIB: Measured batch-worker acceptance ceiling, default 1024.
        COLLECTION_ALTDATA_LOOKBACK_DAYS: Rolling re-observation window, default 30.
        COLLECTION_VERIFIED_CHART_ROUTES: Explicit vendor chart-to-venue mapping, default empty.
        COLLECTION_OPEN_CONFIRM_SECONDS: Opening confirmation budget, default 180.
        COLLECTION_SESSION_OVERRIDES: Verified date-specific session times, default empty.

    Raises:
        ValueError: Invalid limits, duplicate slots, or enabled auctions without
            declared research credentials.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    COLLECTION_ROOT: Path | None = Field(default=None)
    COLLECTION_RAW_ENABLED: bool = Field(default=True)
    COLLECTION_AUCTION_ENABLED: bool = Field(default=False)
    COLLECTION_ALTDATA_ENABLED: bool = Field(default=False)
    COLLECTION_RESEARCH_SLOTS: tuple[str, ...] = Field(default=())
    COLLECTION_ALTDATA_EXTRA_SLOTS: tuple[str, ...] = Field(default=())
    COLLECTION_AUCTION_INTERVAL_SECONDS: int = Field(default=60, gt=0)
    COLLECTION_REQUEST_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    COLLECTION_CONCURRENCY_PER_KEY: int = Field(default=8, gt=0)
    COLLECTION_CHART_MAX_PAGES: int = Field(default=30, gt=0)
    COLLECTION_TICK_REPAIR_MAX_PAGES: int = Field(default=120, gt=0)
    COLLECTION_ARROW_BATCH_ROWS: int = Field(default=65536, gt=0)
    COLLECTION_MAX_RSS_MIB: int = Field(default=1024, gt=0)
    COLLECTION_ALTDATA_LOOKBACK_DAYS: int = Field(default=30, gt=0)
    COLLECTION_VERIFIED_CHART_ROUTES: dict[str, str] = Field(default_factory=dict)
    COLLECTION_OPEN_CONFIRM_SECONDS: int = Field(default=180, gt=30)
    COLLECTION_SESSION_OVERRIDES: dict[str, SessionClock] = Field(default_factory=dict)

    @field_validator("COLLECTION_RESEARCH_SLOTS", "COLLECTION_ALTDATA_EXTRA_SLOTS", mode="after")
    @classmethod
    def _check_slots(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for slot in v:
            if not isinstance(slot, str) or not slot or not slot.isascii() or not slot.isdecimal():
                raise ValueError("research slots must be nonempty decimal pool identifiers")
        if len(set(v)) != len(v):
            raise ValueError("research slots must be unique")
        return v

    @field_validator("COLLECTION_VERIFIED_CHART_ROUTES", mode="after")
    @classmethod
    def _check_routes(cls, v: dict[str, str]) -> dict[str, str]:
        for key, item in v.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip():
                raise ValueError("chart routes must map vendor:endpoint to declared venue")
        return v

    @model_validator(mode="after")
    def _check_profile(self) -> Self:
        if self.COLLECTION_TICK_REPAIR_MAX_PAGES < self.COLLECTION_CHART_MAX_PAGES:
            raise ValueError("repair budget must be at least the normal page budget")
        if self.COLLECTION_AUCTION_ENABLED and len(self.COLLECTION_RESEARCH_SLOTS) == 0:
            raise ValueError("enabled auctions require declared research credentials")
        if not self.COLLECTION_RAW_ENABLED and (self.COLLECTION_AUCTION_ENABLED or self.COLLECTION_ALTDATA_ENABLED):
            raise ValueError("legacy operating mode cannot publish independent auctions or slow-data jobs")
        for key, clock in self.COLLECTION_SESSION_OVERRIDES.items():
            try:
                parsed = date.fromisoformat(key)
            except ValueError:
                raise ValueError("session override keys must be ISO dates") from None
            if parsed != clock.trading_date:
                raise ValueError("session override key must equal its trading_date")
        return self
