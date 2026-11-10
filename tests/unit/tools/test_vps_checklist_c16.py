"""C16 (auction-close capacity) probe in the checklist's E window, executed against outcome fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

DOC = Path(__file__).resolve().parents[3] / "docs" / "vps-checklist.md"


def _c16_block() -> str:
    text = DOC.read_text()
    tail = text.split('\n@check("C16")\n', 1)[1]
    return '@check("C16")\n' + tail.split("\np_ledger(D, P, morning=False)", 1)[0]


def _run_c16(outcome: dict[str, Any] | None, *, timer_on: bool = True) -> list[tuple[str, str, dict[str, Any]]]:
    emitted: list[tuple[str, str, dict[str, Any]]] = []
    namespace: dict[str, Any] = {
        "o": {} if outcome is None else {"auction_close": outcome},
        "timer_active": lambda name: timer_on,
        "check": lambda cid: lambda fn: fn(),
        "out": lambda cid, st, **detail: emitted.append((cid, st, detail)),
        "lv": lambda fail, warn=False: "FAIL" if fail else "WARN" if warn else "PASS",
    }
    exec(compile(_c16_block(), "C16", "exec"), namespace)  # noqa: S102 - execute the trusted repository probe
    return emitted


def _record(outcome: str = "OK", reason: str = "", **metrics: Any) -> dict[str, Any]:
    base = {
        "roster": 449, "capacity_per_round": 1836, "demand_per_round": 562, "status": "OK", "start_lag_s": -60.0,
        "n_deadline": 0, "rounds_with_deadline": 0, "max_missed_rounds_per_symbol": 0,
    }
    base.update(metrics)
    return {"outcome": outcome, "reason": reason, "metrics": base}


def test_c16_passes_without_deadline() -> None:
    ((cid, status, detail),) = _run_c16(_record())
    assert (cid, status) == ("C16", "PASS")
    assert detail["roster"] == 449 and detail["demand_per_round"] == 562


@pytest.mark.parametrize(
    ("metrics", "cause"),
    [
        ({"status": "OVERCOMMIT", "n_deadline": 3, "rounds_with_deadline": 1}, "overcommit"),
        ({"start_lag_s": 24.0, "n_deadline": 2, "rounds_with_deadline": 1}, "late_start"),
        ({"n_deadline": 2, "rounds_with_deadline": 1}, "vendor"),
    ],
)
def test_c16_warns_with_cause(metrics: dict[str, Any], cause: str) -> None:
    ((cid, status, detail),) = _run_c16(_record(outcome="DEGRADED", reason="deadline", **metrics))
    assert (cid, status, detail["cause"]) == ("C16", "WARN", cause)
    assert detail["rounds_with_deadline"] == metrics["rounds_with_deadline"]


def test_c16_absent_outcome_warns_and_timer_off_skips() -> None:
    ((_, status, detail),) = _run_c16(None)
    assert status == "WARN" and detail == {"outcome": "absent"}
    ((_, status, _),) = _run_c16(_record(), timer_on=False)
    assert status == "SKIP(timer_off)"


def test_c16_row_and_window_list_match_probe() -> None:
    text = DOC.read_text()
    assert "| C16 | EXT | E |" in text
    assert "C14 C16 Q01" in text
