"""Acquisition limits for optional research work (CollectionSettings)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import NoDecode

from src.config._env import EnvSettings
from src.config.market_session import NXT_AFTERMARKET_HOUR_CEIL, NXT_AFTERMARKET_HOUR_FLOOR, VERIFIED_CHART_ROUTES
from src.data.capture_contracts import SessionClock


class CollectionSettings(EnvSettings):
    """Define acquisition limits independently of trading selection thresholds.

    These settings bound optional research work; they must never silently reduce
    the all-candidate baseline or change a deployed model's mandatory inputs.

    Attributes:
        COLLECTION_ROOT: Optional owner-local override, default None; resolved beneath HISTORY_DIR/capture.
        COLLECTION_AUCTION_ENABLED: Enable independently budgeted sweeps, default False.
        COLLECTION_ALTDATA_ENABLED: Enable incremental slow-data jobs, default False.
        COLLECTION_RESEARCH_SLOTS: Explicit research key slots, default empty.
        COLLECTION_ALTDATA_EXTRA_SLOTS: Extra KIS data slots for alt-data fan-out, default empty.
        COLLECTION_AUCTION_INTERVAL_SECONDS: Closing sweep interval, default 60.
        COLLECTION_REQUEST_TIMEOUT_SECONDS: Total call timeout, default 5.0.
        COLLECTION_CONCURRENCY_PER_KEY: In-flight limit, default 8.
        COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE: Symbols buffered before an intraday partition flush,
            default 25.
        COLLECTION_CHART_MAX_PAGES: Normal page budget, default 30.
        COLLECTION_TICK_REPAIR_MAX_PAGES: Explicit total repair budget, default 400.
            Measured worst case is ~180 pages for the most active symbol; the budget
            must cover a full session with headroom, otherwise the truncation falls
            through to weaker sources.
        COLLECTION_ARROW_BATCH_ROWS: Bounded rewrite batch size, default 65536.
        COLLECTION_MAX_RSS_MIB: Measured batch-worker acceptance ceiling, default 1024.
        COLLECTION_ALTDATA_LOOKBACK_DAYS: Rolling re-observation window, default 30.
        COLLECTION_VERIFIED_CHART_ROUTES: Vendor chart-to-venue mapping,
            default market_session.VERIFIED_CHART_ROUTES; an env value replaces it wholesale.
        COLLECTION_BACKFILL_SLOTS: Explicit backfill key slots, default empty.
        COLLECTION_BACKFILL_MIN_CHANGE_RATIO: Entry-day reconstruction threshold, default 0.02.
        COLLECTION_KIS_MINUTE_RETENTION_DAYS: KIS minute history window, default 365.
        COLLECTION_BACKFILL_STOP_HHMMSS: KST wall time after which no new task starts, default 065000.
        COLLECTION_OPEN_CONFIRM_SECONDS: Opening confirmation budget, default 180.
        COLLECTION_SESSION_OVERRIDES: Operator emergency session clocks resolved by src.data.session_calendar.resolve_session_day; not a second calendar.
        COLLECTION_AFTERMARKET_BOOK_ENABLED: Enable evening aftermarket order-book capture, default False.
        COLLECTION_AFTERMARKET_BOOK_SLOTS: Explicit aftermarket book key slots, default empty.
        COLLECTION_AFTERMARKET_BOOK_DENSE_SECONDS: Dense aftermarket book spacing, default 60.
        COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES: Full-cohort sweep instants, default eleven evening times.
        COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS: Rounds buffered before a normalized flush, default 15.
        COLLECTION_TRANSPORT_RETRIES: Extra tries after a transient transport error, default 2.
        COLLECTION_TRANSPORT_BACKOFF_SECONDS: Delay before the first retry, doubled per retry, default 3.0.

    Raises:
        ValueError: Invalid limits, duplicate slots, or enabled auctions without
            declared research credentials.
    """

    COLLECTION_ROOT: Path | None = Field(default=None)
    COLLECTION_AUCTION_ENABLED: bool = Field(default=False)
    COLLECTION_ALTDATA_ENABLED: bool = Field(default=False)
    COLLECTION_RESEARCH_SLOTS: Annotated[tuple[str, ...], NoDecode] = Field(default=())
    COLLECTION_ALTDATA_EXTRA_SLOTS: Annotated[tuple[str, ...], NoDecode] = Field(default=())
    COLLECTION_AUCTION_INTERVAL_SECONDS: int = Field(default=60, gt=0)
    COLLECTION_REQUEST_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    COLLECTION_CONCURRENCY_PER_KEY: int = Field(default=8, gt=0)
    COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE: int = Field(default=25, gt=0)
    COLLECTION_CHART_MAX_PAGES: int = Field(default=30, gt=0)
    COLLECTION_TICK_REPAIR_MAX_PAGES: int = Field(default=400, gt=0)
    COLLECTION_ARROW_BATCH_ROWS: int = Field(default=65536, gt=0)
    COLLECTION_MAX_RSS_MIB: int = Field(default=1024, gt=0)
    COLLECTION_ALTDATA_LOOKBACK_DAYS: int = Field(default=30, gt=0)
    COLLECTION_VERIFIED_CHART_ROUTES: dict[str, str] = Field(default_factory=lambda: dict(VERIFIED_CHART_ROUTES))
    COLLECTION_BACKFILL_SLOTS: Annotated[tuple[str, ...], NoDecode] = Field(default=())
    COLLECTION_BACKFILL_MIN_CHANGE_RATIO: float = Field(default=0.02, ge=0.0, allow_inf_nan=False)
    COLLECTION_KIS_MINUTE_RETENTION_DAYS: int = Field(default=365, gt=0)
    COLLECTION_BACKFILL_STOP_HHMMSS: str = Field(default="065000", pattern=r"^\d{6}$")
    COLLECTION_OPEN_CONFIRM_SECONDS: int = Field(default=180, gt=30)
    COLLECTION_SESSION_OVERRIDES: dict[str, SessionClock] = Field(default_factory=dict)
    COLLECTION_AFTERMARKET_BOOK_ENABLED: bool = Field(default=False)
    COLLECTION_AFTERMARKET_BOOK_SLOTS: Annotated[tuple[str, ...], NoDecode] = Field(default=())
    COLLECTION_AFTERMARKET_BOOK_DENSE_SECONDS: int = Field(default=60, gt=0)
    COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES: tuple[str, ...] = Field(
        default=("154500", "160500", "163000", "170000", "173000", "180000", "183000", "190000", "193000", "195000", "195800")
    )
    COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS: int = Field(default=15, gt=0)
    COLLECTION_TRANSPORT_RETRIES: int = Field(default=2, ge=0)
    COLLECTION_TRANSPORT_BACKOFF_SECONDS: float = Field(default=3.0, ge=0, allow_inf_nan=False)

    @field_validator("COLLECTION_RESEARCH_SLOTS", "COLLECTION_ALTDATA_EXTRA_SLOTS", "COLLECTION_BACKFILL_SLOTS", "COLLECTION_AFTERMARKET_BOOK_SLOTS", mode="before")
    @classmethod
    def _parse_slot_env(cls, v: Any) -> Any:
        """Accept the slot-list spelling every env loader agrees on.

        docker ``--env-file`` keeps quotes literally while systemd
        ``EnvironmentFile`` strips them, so a JSON list such as ``["3"]`` reads
        as ``[3]`` under systemd. The comma spelling (``2,3``, the same as
        ``KIS_DATA_SLOTS``) is identical under both loaders; JSON lists remain
        accepted for existing deployments.
        """
        if not isinstance(v, str):
            return v
        text = v.strip()
        items = json.loads(text) if text.startswith("[") else text.split(",")
        return tuple(str(item).strip() for item in items if str(item).strip())

    @field_validator("COLLECTION_RESEARCH_SLOTS", "COLLECTION_ALTDATA_EXTRA_SLOTS", "COLLECTION_BACKFILL_SLOTS", "COLLECTION_AFTERMARKET_BOOK_SLOTS", mode="after")
    @classmethod
    def _check_slots(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for slot in v:
            if not isinstance(slot, str) or not slot or not slot.isascii() or not slot.isdecimal():
                raise ValueError("research slots must be nonempty decimal pool identifiers")
        if len(set(v)) != len(v):
            raise ValueError("research slots must be unique")
        return v

    @field_validator("COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES", mode="after")
    @classmethod
    def _check_sparse_times(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        prev = ""
        for item in v:
            if not isinstance(item, str) or len(item) != 6 or not item.isdecimal():
                raise ValueError("sparse times must be HHMMSS strings")
            if item <= prev:
                raise ValueError("sparse times must be strictly increasing")
            if not (NXT_AFTERMARKET_HOUR_FLOOR <= item < NXT_AFTERMARKET_HOUR_CEIL):
                raise ValueError("sparse times must lie inside the NXT aftermarket window")
            prev = item
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
        for key, clock in self.COLLECTION_SESSION_OVERRIDES.items():
            try:
                parsed = date.fromisoformat(key)
            except ValueError:
                raise ValueError("session override keys must be ISO dates") from None
            if parsed != clock.trading_date:
                raise ValueError("session override key must equal its trading_date")
        return self
