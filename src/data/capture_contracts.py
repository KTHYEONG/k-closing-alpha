"""Deterministic capture, pagination, and merge contracts for research acquisition."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, Self
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SEOUL = ZoneInfo("Asia/Seoul")

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list[Any] | dict[str, Any]
BrokerPayload = dict[str, Any]

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9]+$")
_CREDENTIAL_KEY_RE = re.compile(r"authorization|appkey|app_key|secret|token", re.IGNORECASE)


class CaptureStatus(StrEnum):
    """Persisted outcome of a single capture attempt."""

    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NO_TRADES = "NO_TRADES"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


GOOD_ENTRY_STATES: frozenset[CaptureStatus] = frozenset(
    {CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE}
)
"""Coverage statuses that certify a symbol's capture as usable.

NO_TRADES and NOT_APPLICABLE are terminal successes (the venue produced nothing to capture), so a
manifest whose entries are all in this set is COMPLETE; any other status makes it PARTIAL.
"""


class CaptureDataset(StrEnum):
    """Declared acquisition dataset of a capture context."""

    SCAN = "SCAN"
    PRICE = "PRICE"
    INVESTOR_ESTIMATE = "INVESTOR_ESTIMATE"
    ORDERBOOK = "ORDERBOOK"
    PROGRAM = "PROGRAM"
    MINUTE_BARS = "MINUTE_BARS"
    TRADE_TICKS = "TRADE_TICKS"
    DAILY_BARS = "DAILY_BARS"
    DISCLOSURE = "DISCLOSURE"
    SHORTING = "SHORTING"
    CREDIT_BALANCE = "CREDIT_BALANCE"
    DERIVATIVES_BASIS = "DERIVATIVES_BASIS"
    PROGRAM_DAILY = "PROGRAM_DAILY"


def _ensure_aware(value: datetime, field: str) -> datetime:
    """Normalize an aware datetime to Asia/Seoul without changing the instant."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(SEOUL)


def _require_relative_path(value: str, field: str) -> str:
    """Require a safe root-relative artifact path with slash-separated segments."""
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError(f"{field} must be nonempty")
    if value.startswith("/") or value.startswith("\\") or "\x00" in value or "\\" in value:
        raise ValueError(f"{field} must be path-safe")
    segments = value.split("/")
    if any(part in ("", ".", "..") for part in segments):
        raise ValueError(f"{field} must be path-safe")
    return value


def _require_path_safe(value: str, field: str) -> str:
    """Require a nonempty identifier without path separators or parent traversal."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty")
    if "/" in value or "\\" in value or "\x00" in value or value.strip() != value:
        raise ValueError(f"{field} must be path-safe")
    if value in (".", "..") or ".." in value:
        raise ValueError(f"{field} must be path-safe")
    return value


def _require_symbol(value: str | None, field: str) -> str | None:
    """Require security identifiers as strings preserving leading zeros."""
    if value is None:
        return None
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise ValueError(f"{field} must be an alphanumeric security identifier string")
    return value


def _validate_json_value(value: Any) -> None:
    """Reject non-JSON types and nonfinite floats in persisted raw payloads."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("payload must not contain nonfinite floats")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("payload object keys must be strings")
            _validate_json_value(item)
        return
    raise ValueError("payload must contain only JSON scalar/list/object values")


class SessionClock(BaseModel):
    """Verified trading session times sharing a single trading date."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trading_date: date
    open_at: datetime
    close_at: datetime
    close_confirmation_deadline: datetime
    provenance: str

    @field_validator("open_at", "close_at", "close_confirmation_deadline", mode="after")
    @classmethod
    def _normalize_clock(cls, v: datetime) -> datetime:
        return _ensure_aware(v, "session clock")

    @field_validator("provenance", mode="after")
    @classmethod
    def _check_provenance(cls, v: str) -> str:
        return _require_path_safe(v, "provenance")

    @model_validator(mode="after")
    def _check_session(self) -> Self:
        if not (self.open_at < self.close_at <= self.close_confirmation_deadline):
            raise ValueError("session requires open_at < close_at <= close_confirmation_deadline")
        for label, moment in (
            ("open_at", self.open_at),
            ("close_at", self.close_at),
            ("close_confirmation_deadline", self.close_confirmation_deadline),
        ):
            if moment.date() != self.trading_date:
                raise ValueError(f"{label} must share trading_date")
        return self

    @classmethod
    def standard(cls, trading_date: date) -> SessionClock:
        """Build the standard 09:00/15:30/15:33 profile from market-session constants."""
        from src.config.market_session import (
            CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS,
            KRX_REGULAR_HOUR_CEIL,
            KRX_REGULAR_HOUR_FLOOR,
        )

        def _at(hhmmss: str) -> datetime:
            return datetime(
                trading_date.year,
                trading_date.month,
                trading_date.day,
                int(hhmmss[0:2]),
                int(hhmmss[2:4]),
                int(hhmmss[4:6]),
                tzinfo=SEOUL,
            )

        return cls(
            trading_date=trading_date,
            open_at=_at(KRX_REGULAR_HOUR_FLOOR),
            close_at=_at(KRX_REGULAR_HOUR_CEIL),
            close_confirmation_deadline=_at(CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS),
            provenance="standard_profile",
        )


class CaptureContext(BaseModel):
    """Identity of a single declared acquisition task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trading_date: date
    run_id: str
    dataset: CaptureDataset
    vendor: str
    endpoint: str
    symbol: str | None
    venue: str
    session: str
    capture_reason: str
    cohort_id: str | None
    scheduled_at: datetime | None

    @field_validator("run_id", "vendor", "endpoint", "venue", "session", "capture_reason", mode="after")
    @classmethod
    def _check_route(cls, v: str) -> str:
        return _require_path_safe(v, "capture route")

    @field_validator("cohort_id", mode="after")
    @classmethod
    def _check_cohort(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_path_safe(v, "cohort_id")

    @field_validator("symbol", mode="after")
    @classmethod
    def _check_symbol(cls, v: str | None) -> str | None:
        return _require_symbol(v, "symbol")

    @field_validator("scheduled_at", mode="after")
    @classmethod
    def _normalize_scheduled(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        return _ensure_aware(v, "scheduled_at")


class CapturedResponse(BaseModel):
    """Evidence of one broker call bound to its declared context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: CaptureContext
    request_started_at: datetime
    received_at: datetime
    payload: BrokerPayload | None
    status: CaptureStatus
    source_timestamp: datetime | None
    source_published_at: datetime | None
    page_index: int = Field(ge=0)
    attempt_index: int = Field(ge=0)
    continuation: dict[str, str]
    error_type: str | None

    @field_validator("request_started_at", "received_at", mode="after")
    @classmethod
    def _normalize_observation(cls, v: datetime) -> datetime:
        return _ensure_aware(v, "observation clock")

    @field_validator("source_timestamp", "source_published_at", mode="after")
    @classmethod
    def _normalize_source(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        return _ensure_aware(v, "source clock")

    @field_validator("payload", mode="before")
    @classmethod
    def _copy_payload(cls, v: BrokerPayload | None) -> BrokerPayload | None:
        if v is None:
            return None
        copied = copy.deepcopy(v)
        _validate_json_value(copied)
        return copied

    @field_validator("continuation", mode="after")
    @classmethod
    def _check_continuation(cls, v: dict[str, str]) -> dict[str, str]:
        for key, item in v.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
                raise ValueError("continuation must map cursor names to strings")
            if _CREDENTIAL_KEY_RE.search(key):
                raise ValueError("continuation must not carry credentials")
        return v

    @field_validator("error_type", mode="after")
    @classmethod
    def _check_error(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_path_safe(v, "error_type")

    @model_validator(mode="after")
    def _check_chronology(self) -> Self:
        if self.received_at < self.request_started_at:
            raise ValueError("received_at must not precede request_started_at")
        return self


class ArtifactRef(BaseModel):
    """Pointer to a persisted raw artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    bytes: int = Field(ge=0)
    rows: int | None = Field(default=None, ge=0)

    @field_validator("path", mode="after")
    @classmethod
    def _check_path(cls, v: str) -> str:
        return _require_relative_path(v, "path")

    @field_validator("sha256", mode="after")
    @classmethod
    def _check_sha(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("sha256 must be nonempty")
        return v


class CoverageEntry(BaseModel):
    """Per-symbol acquisition coverage for one declared dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str | None
    dataset: CaptureDataset
    venue: str
    session: str
    scheduled_at: datetime | None
    status: CaptureStatus
    rows: int = Field(ge=0)
    first_event_time: datetime | None
    last_event_time: datetime | None
    reason: str
    raw_refs: tuple[ArtifactRef, ...]

    @field_validator("symbol", mode="after")
    @classmethod
    def _check_symbol(cls, v: str | None) -> str | None:
        return _require_symbol(v, "symbol")

    @field_validator("venue", "session", mode="after")
    @classmethod
    def _check_route(cls, v: str) -> str:
        return _require_path_safe(v, "coverage route")

    @field_validator("scheduled_at", "first_event_time", "last_event_time", mode="after")
    @classmethod
    def _normalize_clock(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        return _ensure_aware(v, "coverage clock")

    @model_validator(mode="after")
    def _check_certification(self) -> Self:
        if self.status in (CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE) and (
            not self.reason.strip() or len(self.raw_refs) == 0
        ):
            raise ValueError("NO_TRADES and NOT_APPLICABLE require explicit evidence in raw_refs/reason")
        if self.venue == "UNKNOWN" and self.status == CaptureStatus.COMPLETE:
            raise ValueError("UNKNOWN venue cannot certify a venue-specific dataset")
        return self


class Cohort(BaseModel):
    """Complete research population independent of final stock picks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trading_date: date
    cohort_id: str
    eligible_symbols: tuple[str, ...]
    scanned_symbols: tuple[str, ...]
    eligibility_rule_version: str
    rejections: dict[str, str]

    @field_validator("cohort_id", "eligibility_rule_version", mode="after")
    @classmethod
    def _check_identifier(cls, v: str) -> str:
        return _require_path_safe(v, "cohort identifier")

    @field_validator("eligible_symbols", "scanned_symbols", mode="after")
    @classmethod
    def _check_members(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for member in v:
            _require_symbol(member, "cohort member")
        return v

    @field_validator("rejections", mode="after")
    @classmethod
    def _check_rejections(cls, v: dict[str, str]) -> dict[str, str]:
        for key, reason in v.items():
            _require_symbol(key, "rejection symbol")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("rejection reasons must be nonempty")
        return v

    @model_validator(mode="after")
    def _check_partition(self) -> Self:
        eligible = set(self.eligible_symbols)
        rejected = set(self.rejections)
        scanned = set(self.scanned_symbols)
        if eligible & rejected:
            raise ValueError("eligible and rejected branches must be disjoint")
        if eligible | rejected != scanned:
            raise ValueError("every scanned instrument must have one consistent branch")
        return self


class CaptureManifest(BaseModel):
    """Completed acquisition task with its cohort and coverage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    context: CaptureContext
    cohort: Cohort | None
    completed_at: datetime
    entries: tuple[CoverageEntry, ...]
    artifacts: tuple[ArtifactRef, ...]
    status: CaptureStatus

    @field_validator("completed_at", mode="after")
    @classmethod
    def _normalize_completed(cls, v: datetime) -> datetime:
        return _ensure_aware(v, "completed_at")


class ChartBudget(BaseModel):
    """Bounded page budget for one chart sweep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_pages: int = Field(gt=0)
    deadline: datetime | None
    request_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("deadline", mode="after")
    @classmethod
    def _normalize_deadline(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        return _ensure_aware(v, "deadline")


PageObserver = Callable[[BrokerPayload | None, Mapping[str, str], datetime, datetime, int, int], None]
SymbolObserver = Callable[[str, pd.DataFrame, CoverageEntry], None]


class RawCaptureError(OSError):
    """Durable observer failure while persisting raw capture evidence."""


def build_cohort(
    trading_date: date,
    scanned_symbols: Sequence[str],
    eligible_symbols: Sequence[str],
    rejections: Mapping[str, str],
    *,
    eligibility_rule_version: str,
) -> Cohort:
    """Identify the complete research population independently of final stock picks.

    Args:
        trading_date: Actual market date of the scan.
        scanned_symbols: All observed security identifiers before eligibility checks.
        eligible_symbols: Instruments the project can quote under its declared rules.
        rejections: Explicit reasons for scanned but ineligible instruments.
        eligibility_rule_version: Stable version of the eligibility rule contract.

    Returns:
        Canonically identified cohort retaining admitted and non-admitted candidates.

    Raises:
        ValueError: Inconsistent membership or unidentified rejection reasons.
    """
    _require_path_safe(eligibility_rule_version, "eligibility_rule_version")
    scanned = [str(item) for item in scanned_symbols]
    eligible = [str(item) for item in eligible_symbols]
    reasons = {str(key): str(item) for key, item in dict(rejections).items()}
    for member in [*scanned, *eligible, *reasons]:
        _require_symbol(member, "cohort member")
    for reason in reasons.values():
        if not reason.strip():
            raise ValueError("rejection reasons must be nonempty")
    scanned_set = set(scanned)
    eligible_set = set(eligible)
    rejected_set = set(reasons)
    if eligible_set & rejected_set:
        raise ValueError("eligible and rejected branches must be disjoint")
    if eligible_set | rejected_set != scanned_set:
        raise ValueError("every scanned instrument must have one consistent branch")
    canonical = "|".join(
        [
            trading_date.isoformat(),
            ",".join(sorted(eligible_set)),
            ",".join(sorted(scanned_set)),
            ",".join(f"{key}={reasons[key]}" for key in sorted(reasons)),
            eligibility_rule_version,
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return Cohort(
        trading_date=trading_date,
        cohort_id=f"cohort-{trading_date.isoformat()}-{digest}",
        eligible_symbols=tuple(sorted(eligible_set)),
        scanned_symbols=tuple(sorted(scanned_set)),
        eligibility_rule_version=eligibility_rule_version,
        rejections={key: reasons[key] for key in sorted(reasons)},
    )
