"""Guard tests for part-1 transient-failure ledger semantics."""

from __future__ import annotations

import asyncio
from datetime import datetime
from itertools import permutations

import pandas as pd
import pytest

from src.backfill.intraday.collector import (
    _classify_kis_vendor_failure,
    backfill_extended_session_bars,
    collect_intraday_bars,
)
from src.backfill.intraday.extended_session_backfill import (
    ExtendedBackfillLedger,
    _LEDGER_COLUMNS,
    failure_reason_class,
    is_deterministic_failure,
    is_transient_failure,
)
from src.data.capture_contracts import SEOUL, CaptureDataset, CaptureStatus, CoverageEntry
from src.config.collection import CollectionSettings
from src.data.capture_store import CaptureStore


def _failed(symbol: str, reason: str) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="NXT",
        session="nxt_aftermarket", scheduled_at=None, status=CaptureStatus.FAILED,
        rows=0, first_event_time=None, last_event_time=None, reason=reason, raw_refs=(),
    )


def test_classifier_branches() -> None:
    assert _classify_kis_vendor_failure(None) == "vendor_failure"
    assert _classify_kis_vendor_failure({"rt_cd": "9", "msg1": "x"}) == "vendor_failure:transient:none"
    assert _classify_kis_vendor_failure(
        {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과"}
    ) == "vendor_failure:transient:EGW00201"
    assert _classify_kis_vendor_failure(
        {"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "만료된 token"}
    ) == "vendor_failure:transient:EGW00123"
    assert _classify_kis_vendor_failure(
        {"rt_cd": "1", "msg_cd": "OPSQ0002", "msg1": "retention"}
    ) == "vendor_failure:rejected:OPSQ0002"
    tagged = _classify_kis_vendor_failure({"rt_cd": "1", "msg_cd": "AB:12 한글", "msg1": "other"})
    assert tagged == "vendor_failure:rejected:AB-12"
    assert len(tagged.split(":")) == 3


def test_predicates_and_classes() -> None:
    for reason in (
        "transport:RuntimeError",
        "raw_basis_transport:TimeoutError",
        "raw_basis_vendor_error",
        "raw_basis_vendor_error:9",
        "vendor_failure:transient:EGW00123",
        "price_basis_raw_basis_transport:TimeoutError",
        "price_basis_raw_basis_vendor_error",
        "exhausted:transport:X",
    ):
        assert is_transient_failure(reason) is True
    for reason in (
        "vendor_failure",
        "vendor_failure:rejected:OPSQ0002",
        "raw_basis_unavailable",
        "price_basis_raw_basis_unavailable",
        "price_basis_no_raw_source",
        "price_basis_unknown",
        "nonpositive_price",
        "seed",
        "",
    ):
        assert is_transient_failure(reason) is False
    assert is_deterministic_failure("reopened:exhausted:price_basis_raw_basis_unavailable") is True
    assert is_deterministic_failure("exhausted:price_basis_raw_basis_unavailable") is True
    assert is_deterministic_failure("raw_basis_unavailable") is True
    assert is_deterministic_failure("vendor_failure:rejected:X") is True
    assert is_deterministic_failure("transport:X") is False
    assert is_deterministic_failure("raw_basis_normalize:E") is False
    assert failure_reason_class("transport:A") == "transport"
    assert failure_reason_class("exhausted:price_basis_raw_basis_unavailable") == "price_basis_raw_basis_unavailable"
    assert failure_reason_class("vendor_failure:rejected:X") == "vendor_failure:rejected"
    assert failure_reason_class("") == "empty"
    assert failure_reason_class(":foo") == "empty"
    assert failure_reason_class("price_basis_raw_basis_transport:TimeoutError") == "price_basis_raw_basis_transport"


def test_record_transient_never_exhausts(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    now = datetime.now(SEOUL)
    for _ in range(4):
        returned = ledger.record(
            "2026-03-02", "nxt_aftermarket", [_failed("000001", "transport:X")],
            run_id="r", attempted_at=now, is_transient=is_transient_failure,
        )
        assert returned.newly_exhausted == 0
        assert returned.newly_exhausted_actionable == 0
    frame = ledger._read_all()
    latest = frame.iloc[-1]
    assert latest["status"] == "FAILED" and int(latest["attempts"]) == 0


def test_record_mixed_counts_and_returns(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    now = datetime.now(SEOUL)
    assert ledger.record("2026-03-02", "s", [], run_id="r", attempted_at=now).newly_exhausted == 0
    for _ in range(2):
        assert ledger.record(
            "2026-03-02", "s", [_failed("000001", "transport:X")],
            run_id="r", attempted_at=now, is_transient=is_transient_failure,
        ).newly_exhausted == 0
    assert ledger.record(
        "2026-03-02", "s", [_failed("000001", "raw_basis_unavailable")],
        run_id="r", attempted_at=now, is_transient=is_transient_failure,
    ).newly_exhausted == 0
    assert ledger.record(
        "2026-03-02", "s", [_failed("000001", "raw_basis_unavailable")],
        run_id="r", attempted_at=now, is_transient=is_transient_failure,
    ).newly_exhausted == 0
    assert ledger.record(
        "2026-03-02", "s", [_failed("000001", "raw_basis_unavailable")],
        run_id="r", attempted_at=now, is_transient=is_transient_failure,
    ).newly_exhausted == 1
    frame = ledger._read_all()
    assert frame.iloc[-1]["status"] == "EXHAUSTED"
    assert list(frame.columns) == list(_LEDGER_COLUMNS)


@pytest.mark.parametrize("sequence", sorted(set(permutations("TTNNN"))))
def test_interleaved_failures_exhaust_only_on_third_counting_outcome(tmp_path, sequence) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    counted = 0
    for kind in sequence:
        reason = "transport:X" if kind == "T" else "price_basis_nonpositive_price"
        counted += kind == "N"
        result = ledger.record(
            "2026-03-02", "nxt_aftermarket", [_failed("000001", reason)],
            run_id="r", attempted_at=datetime.now(SEOUL), is_transient=is_transient_failure,
        )
        row = ledger._read_all().iloc[0]
        assert row["attempts"] == counted
        assert result.newly_exhausted == int(counted == 3)
        assert result.newly_exhausted_actionable == 0
        assert row["status"] == ("EXHAUSTED" if counted == 3 else "FAILED")
        assert bool(ledger.terminal_symbols("2026-03-02", "nxt_aftermarket")) == (counted == 3)
        if counted == 3:
            assert row["reason"] == f"exhausted:{reason}"
            break


@pytest.mark.parametrize("previous", ["COMPLETE", "EXHAUSTED"])
@pytest.mark.parametrize(("reason", "expected"), [("transport:X", 0), ("nonpositive_price", 1)])
def test_failure_after_nonfailed_status_restarts_attempts(tmp_path, previous, reason, expected) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    entry = _failed("000001", "nonpositive_price")
    for _ in range(3):
        ledger.record("2026-03-02", "s", [entry], run_id="seed", attempted_at=datetime.now(SEOUL))
    if previous == "COMPLETE":
        ledger.record(
            "2026-03-02", "s", [entry.model_copy(update={"status": CaptureStatus.COMPLETE})],
            run_id="success", attempted_at=datetime.now(SEOUL),
        )
        assert ledger._read_all().iloc[0]["attempts"] == 0
    assert ledger.record(
        "2026-03-02", "s", [_failed("000001", reason)], run_id="r",
        attempted_at=datetime.now(SEOUL), is_transient=is_transient_failure,
    ).newly_exhausted == 0
    row = ledger._read_all().iloc[0]
    assert row["status"] == "FAILED" and row["attempts"] == expected


def test_legacy_attempts_and_predicate_failure_preserve_ledger(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    entry = _failed("000001", "seed")
    ledger.record("2026-03-02", "s", [entry], run_id="seed", attempted_at=datetime.now(SEOUL))
    legacy = ledger._read_all().drop(columns="attempts")
    legacy.to_parquet(ledger.path, index=False)
    ledger.record(
        "2026-03-02", "s", [_failed("000001", "transport:X")], run_id="r",
        attempted_at=datetime.now(SEOUL), is_transient=is_transient_failure,
    )
    assert ledger._read_all().iloc[0]["attempts"] == 1
    assert list(pd.read_parquet(ledger.path).columns) == list(_LEDGER_COLUMNS)
    before = ledger.path.read_bytes()
    seen = []

    def fail_on_second(reason: str) -> bool:
        seen.append(reason)
        if len(seen) == 2:
            raise RuntimeError("predicate failure")
        return False

    with pytest.raises(RuntimeError, match="predicate failure"):
        ledger.record(
            "2026-03-02", "s", [entry, _failed("000002", "transport:X")], run_id="r",
            attempted_at=datetime.now(SEOUL), is_transient=fail_on_second,
        )
    assert seen == ["seed", "transport:X"]
    assert ledger.path.read_bytes() == before


def test_default_predicate_counts_transport_and_empty_records_do_not_write(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    for entries in ([], [_failed("000001", "x").model_copy(update={"symbol": None})]):
        assert ledger.record("2026-03-02", "s", entries, run_id="r", attempted_at=datetime.now(SEOUL)).newly_exhausted == 0
        assert not ledger.path.exists()
    for expected in (0, 0, 1):
        assert ledger.record(
            "2026-03-02", "s", [_failed("000001", "transport:X")],
            run_id="r", attempted_at=datetime.now(SEOUL),
        ).newly_exhausted == expected
    assert ledger._read_all().iloc[0]["attempts"] == 3


@pytest.mark.parametrize(("payload", "reason"), [
    ({"rt_cd": "9", "msg1": "network exhausted"}, "vendor_failure:transient:none"),
    ({"rt_cd": "9", "msg1": "최대 재시도 횟수 초과 (TPS 제한)"}, "vendor_failure:transient:none"),
    ({"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수 초과"}, "vendor_failure:transient:EGW00201"),
    ({"rt_cd": "1", "msg_cd": "EGW00123"}, "vendor_failure:transient:EGW00123"),
    ({"rt_cd": "1", "msg_cd": "OPSQ0002", "msg1": "retention"}, "vendor_failure:rejected:OPSQ0002"),
    (None, "vendor_failure"),
])
def test_extended_route_classifies_and_retains_evidence_without_changing_live_route(tmp_path, payload, reason) -> None:
    class Kis:
        calls = 0

        async def get_historical_minute_chart(self, *args, **kwargs):
            self.calls += 1
            return payload

    client = Kis()
    profile = CollectionSettings(COLLECTION_ROOT=tmp_path / "capture")
    store = CaptureStore(tmp_path / "capture")
    for extended in (True, False):
        observed = []
        kwargs = {"session_tag": "nxt_aftermarket"} if extended else {}
        collect = backfill_extended_session_bars if extended else collect_intraday_bars
        frame = asyncio.run(collect(
            client, None, ["000001"], "2026-03-02", profile=profile, capture_store=store,
            run_id=f"route-{extended}", on_symbol=lambda symbol, frame, entry, outcomes=observed: outcomes.append(entry), **kwargs,
        ))
        assert frame.empty
        assert len(observed) == 1
        entry = observed[0]
        assert entry.status == CaptureStatus.FAILED and entry.rows == 0
        assert entry.reason == (reason if extended else "vendor_failure")
        assert len(entry.raw_refs) == 1
    assert client.calls == 2
