"""Host-wide admission pacing adapter (settings-aware thin wrapper over the canonical contract)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from src.api.host_broker_contract import (
    HOST_ADMISSION_MARKER,
    AsyncHostPacer,
    HostBrokerDirNotSharedError,
    admission_state_path,
    rate_for,
    resolve_host_broker_dir,
)

HostPacedRateLimiter = AsyncHostPacer
AdmissionDirNotSharedError = HostBrokerDirNotSharedError


class AdmissionClass(StrEnum):
    CRITICAL = "critical"
    STANDARD = "standard"
    BULK = "bulk"


def _in_container() -> bool:
    return Path("/.dockerenv").exists()


def resolve_admission_dir() -> Path:
    """Return settings.BROKER_ADMISSION_DIR or KIS_TOKEN_CACHE_DIR; raise AdmissionDirNotSharedError when enforced and unmarked."""
    from src import settings as _settings

    raw = _settings.BROKER_ADMISSION_DIR
    configured: Path | None = Path(raw) if raw is not None else Path(_settings.KIS_TOKEN_CACHE_DIR)
    mode = str(_settings.BROKER_ADMISSION_REQUIRE_SHARED)
    if mode not in ("auto", "always", "never"):
        mode = "auto"
    return resolve_host_broker_dir(
        configured,
        require_shared=mode,  # type: ignore[arg-type]
        in_container=_in_container(),
    )


def host_admission_state_path(vendor: str, credential: str, scope: str | None = None) -> Path:
    """Return the canonical admission path under resolve_admission_dir(); raise ValueError on bad names."""
    return admission_state_path(resolve_admission_dir(), vendor, credential, scope)


def max_lead_for(admission_class: AdmissionClass) -> float | None:
    """Map a class to its configured reservation lead; CRITICAL maps to None (unbounded)."""
    from src import settings as _settings

    if admission_class == AdmissionClass.CRITICAL:
        return None
    if admission_class == AdmissionClass.STANDARD:
        return float(_settings.BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS)
    return float(_settings.BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS)


def _resolve_class(admission_class: AdmissionClass | str | None) -> AdmissionClass:
    if admission_class is None:
        from src import settings as _settings

        return AdmissionClass(str(_settings.BROKER_ADMISSION_CLASS))
    return admission_class if isinstance(admission_class, AdmissionClass) else AdmissionClass(str(admission_class))


_HOST_RATE_LIMITERS: dict[tuple[str, float, float, str], HostPacedRateLimiter] = {}

_ADMISSION_DIR_VERIFIED = False


def _ensure_admission_dir_shared() -> None:
    global _ADMISSION_DIR_VERIFIED
    if not _ADMISSION_DIR_VERIFIED:
        resolve_admission_dir()
        _ADMISSION_DIR_VERIFIED = True


def get_host_rate_limiter(
    state_path: Path, max_rate: float, time_period: float = 1.0, *, admission_class: AdmissionClass | str | None = None
) -> HostPacedRateLimiter:
    """Return the process-singleton limiter for (path, rate, class); class None means settings.BROKER_ADMISSION_CLASS."""
    _ensure_admission_dir_shared()
    resolved = _resolve_class(admission_class)
    key = (str(Path(state_path)), float(max_rate), float(time_period), str(resolved.value))
    limiter = _HOST_RATE_LIMITERS.get(key)
    if limiter is None:
        limiter = HostPacedRateLimiter(
            Path(state_path),
            max_rate,
            time_period,
            max_lead_seconds=max_lead_for(resolved),
        )
        _HOST_RATE_LIMITERS[key] = limiter
    return limiter


def vendor_limiter(
    vendor: str, credential: str, scope: str | None = None, *, admission_class: AdmissionClass | str | None = None
) -> HostPacedRateLimiter:
    """Return the process-singleton limiter for the canonical (vendor, scope) path and rate_for() rate."""
    return get_host_rate_limiter(
        host_admission_state_path(vendor, credential, scope),
        rate_for(vendor, scope),
        admission_class=admission_class,
    )


__all__ = [
    "HOST_ADMISSION_MARKER",
    "AdmissionClass",
    "AdmissionDirNotSharedError",
    "HostPacedRateLimiter",
    "get_host_rate_limiter",
    "host_admission_state_path",
    "max_lead_for",
    "resolve_admission_dir",
    "vendor_limiter",
]
