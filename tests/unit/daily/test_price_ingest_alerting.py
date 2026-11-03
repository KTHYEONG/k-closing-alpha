"""price_ingest KRX retry budget and the stale-panel page after a failed safety run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.daily.price_ingest as mod
from src.backfill.altdata.krx_api import KRX_HTTP_TIMEOUT_SECONDS

# kca-price-ingest.service TimeoutStartSec=45min; the measured run max is ~493 s.
_UNIT_TIMEOUT_SECONDS: float = 45 * 60
_MEASURED_RUN_MAX_SECONDS: float = 493.0


def test_krx_retry_budget_rides_out_about_ten_minutes_within_unit_timeout() -> None:
    budget = mod.PRICE_INGEST_KRX_RETRY
    worst = budget.worst_case_seconds(KRX_HTTP_TIMEOUT_SECONDS)

    assert 8 * 60 <= worst <= 12 * 60
    assert _MEASURED_RUN_MAX_SECONDS + 2 * worst < _UNIT_TIMEOUT_SECONDS
    cfg = mod.price_ingest_krx_config(pd.Timestamp("2026-11-03"))
    assert (cfg.retries, cfg.retry_sleep_sec) == (budget.retries, budget.retry_sleep_sec)
    assert mod.KrxRetryBudget(retries=4, retry_sleep_sec=1.0).worst_case_seconds(20.0) == 86.0


def test_price_ingest_unit_expected_runtime_covers_retry_budget() -> None:
    text = (Path(__file__).resolve().parents[3] / "deploy" / "systemd" / "kca-price-ingest.service").read_text(encoding="utf-8")
    expected = int(next(line for line in text.splitlines() if line.startswith("X-ExpectedRuntimeSec=")).split("=", 1)[1])
    assert expected >= 900 + mod.PRICE_INGEST_KRX_RETRY.worst_case_seconds(KRX_HTTP_TIMEOUT_SECONDS)


def _standard_days(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from src.data.session_calendar import SessionKind

    def _resolve(day: Any) -> SimpleNamespace:
        kind = SessionKind.CLOSED if pd.Timestamp(day).weekday() >= 5 else SessionKind.STANDARD
        return SimpleNamespace(kind=kind)

    monkeypatch.setattr(mod, "resolve_session_day", _resolve)


def test_record_stale_after_failure_only_after_alert_time_and_when_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _standard_days(monkeypatch)
    panel = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-10-30", "2026-11-02"])}).to_parquet(panel)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _record(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    assert not mod.record_stale_after_failure(pd.Timestamp("2026-11-03 08:31"), on_outcome=_record, path=panel)
    assert not mod.record_stale_after_failure(pd.Timestamp("2026-11-03 11:31"), on_outcome=_record, path=panel)
    assert not mod.record_stale_after_failure(pd.Timestamp("2026-11-08 11:31"), on_outcome=_record, path=panel)
    assert calls == []

    assert mod.record_stale_after_failure(pd.Timestamp("2026-11-04 11:31"), on_outcome=_record, path=panel)
    assert mod.record_stale_after_failure(pd.Timestamp("2026-11-04 11:31"), on_outcome=_record, path=tmp_path / "missing.parquet")
    (args, kwargs), (_, missing_kwargs) = calls
    assert args == (mod.RUN_OUTCOME_DEGRADED,)
    assert kwargs["reason"] == "stale_panel" and kwargs["run_date"] == "2026-11-04"
    assert kwargs["metrics"] == {"ingest_failed": True, "panel_max_date": "2026-11-02", "expected_panel_date": "2026-11-03"}
    assert missing_kwargs["metrics"]["panel_max_date"] == ""


def test_main_records_stale_outcome_then_reraises_on_ingest_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[pd.Timestamp] = []

    async def _boom(**kwargs: Any) -> None:
        raise RuntimeError("krx http_status=unknown (all retries failed)")

    monkeypatch.setattr(mod, "run_price_ingest", _boom)
    monkeypatch.setattr(mod, "record_stale_after_failure", lambda run_at, **kw: seen.append(run_at) or True)

    with pytest.raises(RuntimeError, match="all retries failed"):
        mod.main()
    assert len(seen) == 1
