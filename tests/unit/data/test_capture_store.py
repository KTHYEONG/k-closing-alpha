"""CaptureStore atomic-storage invariant guards."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.data.capture_store as store_module
from src.data.capture_contracts import (
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CapturedResponse,
    Cohort,
    CoverageEntry,
    RawCaptureError,
    build_cohort,
)
from src.data.capture_store import CaptureStore
from src.utils.file_lock import sidecar_lock_path

SEOUL = ZoneInfo("Asia/Seoul")
DAY = date(2026, 9, 17)
START = datetime(2026, 9, 17, 15, 30, tzinfo=SEOUL)


def _context(**overrides: Any) -> CaptureContext:
    base: dict[str, Any] = {
        "trading_date": DAY,
        "run_id": "run-1",
        "dataset": CaptureDataset.PRICE,
        "vendor": "kis",
        "endpoint": "price",
        "symbol": "005930",
        "venue": "KRX",
        "session": "regular",
        "capture_reason": "closing-sweep",
        "cohort_id": None,
        "scheduled_at": None,
    }
    base.update(overrides)
    return CaptureContext(**base)


def _response(**overrides: Any) -> CapturedResponse:
    base: dict[str, Any] = {
        "context": _context(),
        "request_started_at": START,
        "received_at": START + timedelta(seconds=1),
        "payload": {"output": {"code": "005930", "price": "72000"}},
        "status": CaptureStatus.COMPLETE,
        "source_timestamp": None,
        "source_published_at": None,
        "page_index": 0,
        "attempt_index": 0,
        "continuation": {},
        "error_type": None,
    }
    base.update(overrides)
    return CapturedResponse(**base)


def _cohort() -> Cohort:
    return build_cohort(
        DAY,
        ["005930", "000660", "035420"],
        ["005930", "000660"],
        {"035420": "suspended"},
        eligibility_rule_version="v1",
    )


def _frame(extra_symbol: str | None = None) -> pd.DataFrame:
    symbols = ["005930", "000660"] if extra_symbol is None else ["005930", "000660", extra_symbol]
    admitted = [True, False] if extra_symbol is None else [True, False, True]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "admitted": admitted,
            "close": [72000.0, float("nan"), 150000.0][: len(symbols)],
            "snapshot_timestamp": [START] * len(symbols),
            "feature_available_timestamp": [START] * len(symbols),
        }
    )


def _entry(ref: ArtifactRef, **overrides: Any) -> CoverageEntry:
    base: dict[str, Any] = {
        "symbol": "005930",
        "dataset": CaptureDataset.PRICE,
        "venue": "KRX",
        "session": "regular",
        "scheduled_at": None,
        "status": CaptureStatus.COMPLETE,
        "rows": 1,
        "first_event_time": None,
        "last_event_time": None,
        "reason": "sweep done",
        "raw_refs": (ref,),
    }
    base.update(overrides)
    return CoverageEntry(**base)


def _publish_decision(
    store: CaptureStore, run_id: str, completed: datetime, entries: list[CoverageEntry] | None = None
) -> Any:
    refs = entries
    if refs is None:
        raw = store.append_response(_response(context=_context(run_id=run_id)))
        refs = [_entry(raw)]
    return store.publish_decision(_frame(), cohort=_cohort(), run_id=run_id, completed_at=completed, entries=refs)


def test_appended_raw_payload_round_trips(tmp_path: Path) -> None:
    """Raw payload is preserved."""
    store = CaptureStore(tmp_path / "capture")
    ref = store.append_response(_response())
    assert ref.path == "raw/2026-09-17/kis/PRICE/price/run-1/005930-p0000-a00.json.gz"
    envelope = store.read_artifact(ref)
    assert envelope["payload"] == {"output": {"code": "005930", "price": "72000"}}
    assert envelope["source_published_at"] is None
    assert envelope["source_timestamp"] is None
    assert envelope["received_at"] == (START + timedelta(seconds=1)).isoformat()


def test_same_run_distinct_endpoints_do_not_collide(tmp_path: Path) -> None:
    """price_ingest fetches KOSPI/KOSDAQ under one run_id with no symbol; endpoint must disambiguate the path.

    실측: 2026-09-20 두 마켓의 첫 페이지가 같은 경로로 써져
    conflicting immutable artifact identity로 크래시했다.
    """
    store = CaptureStore(tmp_path / "capture")
    kospi = store.append_response(
        _response(context=_context(endpoint="stk-bydd-trd", symbol=None), payload={"output": {"market": "KOSPI"}})
    )
    kosdaq = store.append_response(
        _response(context=_context(endpoint="ksq-bydd-trd", symbol=None), payload={"output": {"market": "KOSDAQ"}})
    )
    assert kospi.path != kosdaq.path
    assert store.read_artifact(kospi)["payload"] == {"output": {"market": "KOSPI"}}
    assert store.read_artifact(kosdaq)["payload"] == {"output": {"market": "KOSDAQ"}}


def test_concurrent_appends_lose_no_response(tmp_path: Path) -> None:
    """Concurrent append loses no response."""
    store = CaptureStore(tmp_path / "capture")
    first = _response(page_index=0)
    second = _response(
        page_index=1,
        context=_context(scheduled_at=START),
        continuation={"next-cursor": "abc"},
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        refs = list(pool.map(store.append_response, [first, second]))
    assert len({ref.path for ref in refs}) == 2
    assert store.read_artifact(refs[0])["page_index"] == 0
    assert store.read_artifact(refs[1])["continuation"] == {"next-cursor": "abc"}


def test_identical_publication_is_idempotent(tmp_path: Path) -> None:
    """Idempotent publication."""
    store = CaptureStore(tmp_path / "capture")
    first = store.append_response(_response())
    second = store.append_response(_response())
    assert first == second
    assert store.root == tmp_path / "capture"
    assert len(list((tmp_path / "capture" / "raw").rglob("*.json.gz"))) == 1


def test_conflicting_identity_fails(tmp_path: Path) -> None:
    """Conflicting immutable identity fails."""
    store = CaptureStore(tmp_path / "capture")
    ref = store.append_response(_response())
    with pytest.raises(ValueError, match="conflicting immutable artifact"):
        store.append_response(_response(payload={"output": {"code": "005930", "price": "73000"}}))
    assert store.read_artifact(ref)["payload"] == {"output": {"code": "005930", "price": "72000"}}
    assert store.append_response(_response()) == ref


def test_corrupt_evidence_never_qualifies(tmp_path: Path) -> None:
    """Corrupt evidence never qualifies."""
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    _publish_decision(store, "run-1", completed)
    raw_path = tmp_path / "capture" / "raw/2026-09-17/kis/PRICE/price/run-1/005930-p0000-a00.json.gz"
    tampered = raw_path.read_bytes() + b"\x00"
    raw_path.write_bytes(tampered)
    with pytest.raises(ValueError, match="hash-inconsistent"):
        store.read_manifests("2026-09-17")
    with pytest.raises(ValueError, match="hash-inconsistent"):
        store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1))
    assert raw_path.read_bytes() == tampered
    with pytest.raises(ValueError, match="conflicting immutable artifact"):
        store.append_response(_response())


def test_future_outcome_leaves_decision_unchanged(tmp_path: Path) -> None:
    """Future outcomes cannot alter inputs."""
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    manifest_ref = _publish_decision(store, "run-1", completed)
    input_path = tmp_path / "capture" / "decision/2026-09-17/run-1/input.parquet"
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()
    outcome = pd.DataFrame({"symbol": ["005930"], "next_open": [73000.0]})
    store.publish_frame(outcome, context=_context(dataset=CaptureDataset.DAILY_BARS))
    replayed = store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1))
    assert replayed["close"].tolist()[0] == 72000.0
    assert pd.isna(replayed["close"].tolist()[1])
    assert replayed["admitted"].tolist() == [True, False]
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == before
    assert manifest_ref.bytes > 0


def test_cutoff_excludes_late_run(tmp_path: Path) -> None:
    """Cutoff excludes late observations."""
    store = CaptureStore(tmp_path / "capture")
    early = START + timedelta(minutes=1)
    late = START + timedelta(minutes=5)
    _publish_decision(store, "run-1", early)
    raw = store.append_response(_response(context=_context(run_id="run-2")))
    store.publish_decision(_frame(), cohort=_cohort(), run_id="run-2", completed_at=late, entries=[_entry(raw)])
    picked = store.read_decision("2026-09-17", available_by=early + timedelta(seconds=30))
    assert picked["capture_run_id"].tolist() == ["run-1", "run-1"]
    latest = store.read_decision("2026-09-17", available_by=late + timedelta(seconds=30))
    assert latest["capture_run_id"].tolist() == ["run-2", "run-2"]
    exact = store.read_decision("2026-09-17", available_by=late + timedelta(seconds=30), run_id="run-1")
    assert exact["capture_run_id"].tolist() == ["run-1", "run-1"]


def test_partial_run_does_not_displace_complete(tmp_path: Path) -> None:
    """Incomplete run does not displace complete run."""
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    _publish_decision(store, "run-1", completed)
    raw = store.append_response(_response(context=_context(run_id="run-2")))
    failed = _entry(raw, status=CaptureStatus.FAILED, reason="vendor outage", rows=0)
    partial = store.publish_decision(
        _frame(), cohort=_cohort(), run_id="run-2", completed_at=completed + timedelta(minutes=1), entries=[failed]
    )
    assert partial.path.endswith("manifest-partial.json")
    picked = store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=5))
    assert picked["capture_run_id"].tolist() == ["run-1", "run-1"]
    with pytest.raises(FileNotFoundError, match="no qualifying verified decision"):
        store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=5), run_id="run-2")


def test_cohort_rejections_survive_round_trip(tmp_path: Path) -> None:
    """Cohort rejections are retained."""
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    _publish_decision(store, "run-1", completed)
    cohort = store.read_cohort("2026-09-17", available_by=completed + timedelta(minutes=1))
    assert set(cohort.eligible_symbols) == {"005930", "000660"}
    assert set(cohort.scanned_symbols) == {"005930", "000660", "035420"}
    assert cohort.rejections == {"035420": "suspended"}
    with pytest.raises(FileNotFoundError, match="no qualifying cohort"):
        store.read_cohort("2026-09-18", available_by=completed + timedelta(minutes=1))


def test_unsafe_paths_rejected(tmp_path: Path) -> None:
    """Unsafe paths are rejected."""
    store = CaptureStore(tmp_path / "capture")
    ref = store.append_response(_response())
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    absolute = ArtifactRef.model_construct(path=str(outside), sha256="0" * 64, bytes=6)
    with pytest.raises(ValueError, match="unsafe artifact path"):
        store.read_artifact(absolute)
    traversal = ArtifactRef.model_construct(path="../outside.txt", sha256="0" * 64, bytes=6)
    with pytest.raises(ValueError, match="unsafe artifact path"):
        store.read_artifact(traversal)
    link = tmp_path / "capture" / "raw" / "escape.json.gz"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    escaping = ArtifactRef(path="raw/escape.json.gz", sha256="0" * 64, bytes=6)
    with pytest.raises(ValueError, match="escapes capture root"):
        store.read_artifact(escaping)
    assert outside.read_text() == "secret"
    assert store.read_artifact(ref)["payload"]["output"]["code"] == "005930"


def test_interrupted_staging_unpublished(tmp_path: Path, monkeypatch: Any) -> None:
    """Interrupted staging remains unpublished."""
    store = CaptureStore(tmp_path / "capture")

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("rename interrupted")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="rename interrupted"):
        store.append_response(_response())
    assert store.read_manifests("2026-09-17") == ()
    assert list((tmp_path / "capture" / "manifests").rglob("*.json")) == []


def test_no_other_project_dependency(tmp_path: Path) -> None:
    """No other-project dependency."""
    store = CaptureStore(tmp_path / "capture")
    ref = store.append_response(_response(context=_context(symbol=None)))
    assert "nosymbol" in ref.path
    assert store.read_artifact(ref)["context"]["symbol"] is None
    source = (Path(__file__).resolve().parents[3] / "src" / "data" / "capture_store.py").read_text()
    assert "krx_alpha" not in source
    assert "krx-alpha" not in source
    assert "from src.api" not in source


def test_decision_rejects_unreconciled_inputs(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    cohort = _cohort()
    raw = store.append_response(_response())
    entries = [_entry(raw)]
    no_symbol = _frame().drop(columns=["symbol"])
    with pytest.raises(ValueError, match="symbol column"):
        store.publish_decision(no_symbol, cohort=cohort, run_id="run-1", completed_at=completed, entries=entries)
    no_flag = _frame().drop(columns=["admitted"])
    with pytest.raises(ValueError, match="admission flags"):
        store.publish_decision(no_flag, cohort=cohort, run_id="run-1", completed_at=completed, entries=entries)
    no_stamp = _frame().drop(columns=["feature_available_timestamp"])
    with pytest.raises(ValueError, match="provenance columns"):
        store.publish_decision(no_stamp, cohort=cohort, run_id="run-1", completed_at=completed, entries=entries)
    with pytest.raises(ValueError, match="exactly once"):
        store.publish_decision(
            _frame(extra_symbol="000660"), cohort=cohort, run_id="run-1", completed_at=completed, entries=entries
        )
    with pytest.raises(ValueError, match="exactly once"):
        store.publish_decision(
            _frame(extra_symbol="035421"), cohort=cohort, run_id="run-1", completed_at=completed, entries=entries
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        store.publish_decision(
            _frame(), cohort=cohort, run_id="run-1", completed_at=datetime(2026, 9, 17, 15, 34), entries=entries
        )
    with pytest.raises(ValueError, match="precedes decision-time evidence"):
        store.publish_decision(
            _frame(), cohort=cohort, run_id="run-1", completed_at=START - timedelta(seconds=1), entries=entries
        )


def test_decision_accepts_naive_frame_clocks(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    frame = _frame()
    frame["snapshot_timestamp"] = [START.replace(tzinfo=None)] * len(frame)
    frame["feature_available_timestamp"] = [START.replace(tzinfo=None)] * len(frame)
    raw = store.append_response(_response())
    ref = store.publish_decision(frame, cohort=_cohort(), run_id="run-1", completed_at=completed, entries=[_entry(raw)])
    assert ref.path.endswith("manifest-complete.json")


def test_decision_accepts_korean_symbol_column(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    frame = _frame().rename(columns={"symbol": "종목코드"})
    raw = store.append_response(_response())
    ref = store.publish_decision(frame, cohort=_cohort(), run_id="run-1", completed_at=completed, entries=[_entry(raw)])
    assert ref.path.endswith("manifest-complete.json")
    replayed = store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1))
    assert replayed["종목코드"].tolist() == ["005930", "000660"]


def test_publish_decision_without_entries(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    ref = store.publish_decision(_frame(), cohort=_cohort(), run_id="run-1", completed_at=completed, entries=[])
    assert ref.path.endswith("manifest-complete.json")
    manifests = store.read_manifests("2026-09-17")
    assert len(manifests) == 1
    assert manifests[0].context.dataset == CaptureDataset.SCAN


def test_publish_manifest_rejects_inconsistent_coverage(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    bad = CoverageEntry(
        symbol="005930",
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=CaptureStatus.FAILED,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason="vendor outage",
        raw_refs=(),
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=_context(),
        cohort=_cohort(),
        completed_at=START + timedelta(minutes=4),
        entries=(bad,),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    with pytest.raises(ValueError, match="cannot certify failed or pending"):
        store.publish_manifest(manifest)


def test_publish_manifest_rejects_missing_reference(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    ghost = ArtifactRef(path="raw/2026-09-17/kis/PRICE/price/run-9/ghost-p0000-a00.json.gz", sha256="0" * 64, bytes=10)
    entry = _entry(ghost, status=CaptureStatus.NO_TRADES, reason="market holiday proof")
    manifest = CaptureManifest(
        schema_version=1,
        context=_context(run_id="run-9"),
        cohort=_cohort(),
        completed_at=START + timedelta(minutes=4),
        entries=(entry,),
        artifacts=(),
        status=CaptureStatus.PARTIAL,
    )
    with pytest.raises(ValueError, match="missing referenced evidence"):
        store.publish_manifest(manifest)


def test_read_decision_rejects_inconsistent_persisted_input(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    _publish_decision(store, "run-1", completed)
    foreign = _frame().drop(columns=["symbol"])
    blob_dir = tmp_path / "capture" / "decision/2026-09-17/run-2"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / "input.parquet"
    foreign.to_parquet(blob_path, index=False)
    blob = blob_path.read_bytes()
    blob_ref = ArtifactRef(
        path="decision/2026-09-17/run-2/input.parquet",
        sha256=hashlib.sha256(blob).hexdigest(),
        bytes=len(blob),
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=_context(run_id="run-2"),
        cohort=_cohort(),
        completed_at=completed,
        entries=(),
        artifacts=(blob_ref,),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(manifest)
    with pytest.raises(ValueError, match="no symbol column"):
        store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1), run_id="run-2")
    drifted = _frame(extra_symbol="035421")
    drift_dir = tmp_path / "capture" / "decision/2026-09-17/run-3"
    drift_dir.mkdir(parents=True, exist_ok=True)
    drift_path = drift_dir / "input.parquet"
    drifted.to_parquet(drift_path, index=False)
    drift_blob = drift_path.read_bytes()
    drift_ref = ArtifactRef(
        path="decision/2026-09-17/run-3/input.parquet",
        sha256=hashlib.sha256(drift_blob).hexdigest(),
        bytes=len(drift_blob),
    )
    drifted_manifest = CaptureManifest(
        schema_version=1,
        context=_context(run_id="run-3"),
        cohort=_cohort(),
        completed_at=completed,
        entries=(),
        artifacts=(drift_ref,),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(drifted_manifest)
    with pytest.raises(ValueError, match="cannot be reconciled"):
        store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1), run_id="run-3")


def test_read_decision_rejects_cohortless_or_inputless_manifest(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    _publish_decision(store, "run-1", completed)
    input_ref = store.read_manifests("2026-09-17")[0].artifacts[0]
    cohortless = CaptureManifest(
        schema_version=1,
        context=_context(run_id="run-2"),
        cohort=None,
        completed_at=completed,
        entries=(),
        artifacts=(input_ref,),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(cohortless)
    with pytest.raises(ValueError, match="no cohort"):
        store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1), run_id="run-2")
    inputless = CaptureManifest(
        schema_version=1,
        context=_context(run_id="run-3"),
        cohort=_cohort(),
        completed_at=completed,
        entries=(),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(inputless)
    with pytest.raises(ValueError, match="no decision input"):
        store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1), run_id="run-3")


def test_read_validates_cutoff_date_and_identity(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    _publish_decision(store, "run-1", completed)
    good = completed + timedelta(minutes=1)
    with pytest.raises(ValueError, match="aware cutoff"):
        store.read_decision("2026-09-17", available_by=datetime(2026, 9, 17, 15, 35))
    with pytest.raises(ValueError, match="aware cutoff"):
        store.read_cohort("2026-09-17", available_by=datetime(2026, 9, 17, 15, 35))
    with pytest.raises(ValueError, match="Invalid isoformat"):
        store.read_decision("not-a-date", available_by=good)
    with pytest.raises(FileNotFoundError, match="no qualifying verified decision"):
        store.read_decision("2026-09-18", available_by=good)
    with pytest.raises(FileNotFoundError, match="no qualifying"):
        store.read_decision("2026-09-17", available_by=START)
    with pytest.raises(FileNotFoundError, match="no qualifying verified decision"):
        store.read_decision("2026-09-17", available_by=good, run_id="run-9")


def test_manifest_store_rejects_unreadable_evidence(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    _publish_decision(store, "run-1", completed)
    bad_dir = tmp_path / "capture" / "manifests/2026-09-17/run-9"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "manifest-complete.json").write_bytes(b"\x00\x01 not json")
    with pytest.raises(ValueError, match="unreadable manifest"):
        store.read_manifests("2026-09-17")
    (bad_dir / "manifest-complete.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="invalid manifest"):
        store.read_manifests("2026-09-17")


def test_bounded_artifact_rejects_oversize(tmp_path: Path, monkeypatch: Any) -> None:
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(store_module, "_MAX_ARTIFACT_BYTES", 10)
    with pytest.raises(ValueError, match="bounded size"):
        store.append_response(_response())


def test_publish_frame_failure_surfaces(tmp_path: Path, monkeypatch: Any) -> None:
    store = CaptureStore(tmp_path / "capture")

    def _boom(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise OSError("disk unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    with pytest.raises(RawCaptureError, match="disk unavailable"):
        store.publish_frame(pd.DataFrame({"a": [1]}), context=_context())


def test_read_artifact_rejects_malformed(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    raw_dir = tmp_path / "capture" / "raw/2026-09-17/kis/PRICE/price/run-1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    garbage = raw_dir / "garbage.json.gz"
    garbage.write_bytes(b"not-gzip-bytes")
    garbage_ref = ArtifactRef(
        path="raw/2026-09-17/kis/PRICE/price/run-1/garbage.json.gz",
        sha256=hashlib.sha256(b"not-gzip-bytes").hexdigest(),
        bytes=len(b"not-gzip-bytes"),
    )
    with pytest.raises(ValueError, match="corrupt raw evidence"):
        store.read_artifact(garbage_ref)
    empty_gz = gzip.compress(b"{}", mtime=0)
    empty_path = raw_dir / "empty.json.gz"
    empty_path.write_bytes(empty_gz)
    empty_ref = ArtifactRef(
        path="raw/2026-09-17/kis/PRICE/price/run-1/empty.json.gz",
        sha256=hashlib.sha256(empty_gz).hexdigest(),
        bytes=len(empty_gz),
    )
    with pytest.raises(ValueError, match="malformed raw evidence"):
        store.read_artifact(empty_ref)
    missing = ArtifactRef(path="raw/2026-09-17/kis/PRICE/price/run-1/missing.json.gz", sha256="0" * 64, bytes=1)
    with pytest.raises(FileNotFoundError):
        store.read_artifact(missing)
    ref = store.append_response(_response())
    tampered_ref = ArtifactRef(path=ref.path, sha256="0" * 64, bytes=ref.bytes)
    with pytest.raises(ValueError, match="hash-inconsistent"):
        store.read_artifact(tampered_ref)
    with pytest.raises(ValueError, match="path-safe"):
        ArtifactRef(path="/absolute/path.json.gz", sha256="0" * 64, bytes=1)
    with pytest.raises(ValueError, match="path-safe"):
        ArtifactRef(path="raw/./dot.json.gz", sha256="0" * 64, bytes=1)


def test_publish_lock_timeout_fails_explicitly(tmp_path: Path, monkeypatch: Any) -> None:
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(store_module, "_LOCK_TIMEOUT_SECONDS", 0.05)
    response = _response()
    target = store._root / store._raw_rel(response)
    target.parent.mkdir(parents=True, exist_ok=True)
    holder_fd = os.open(sidecar_lock_path(target), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(OSError, match="timed out acquiring publish lock"):
            store.append_response(response)
    finally:
        os.close(holder_fd)
    assert not target.exists()


def test_stale_publish_sidecar_does_not_block(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    response = _response()
    target = store._root / store._raw_rel(response)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_lock_path(target)
    sidecar.write_text("held", encoding="utf-8")
    ref = store.append_response(response)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == ref.sha256
    assert not sidecar.exists()


def test_immutable_identity_preserved_under_new_lock(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    first = store.append_response(_response())
    assert store.append_response(_response()) == first
    with pytest.raises(ValueError, match="conflicting immutable artifact identity"):
        store.append_response(_response(payload={"output": {"code": "005930", "price": "73000"}}))
    assert list((tmp_path / "capture").rglob("*.lock")) == []


def _partial_entry() -> CoverageEntry:
    return CoverageEntry(
        symbol=None,
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=CaptureStatus.PARTIAL,
        rows=2,
        first_event_time=None,
        last_event_time=None,
        reason="coverage_below_threshold:0.9800",
        raw_refs=(),
    )


def test_degraded_decision_is_unreadable_for_trading(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    store.publish_decision(_frame(), cohort=_cohort(), run_id="run-1", completed_at=completed, entries=[_partial_entry()])
    with pytest.raises(FileNotFoundError):
        store.read_decision("2026-09-17", available_by=completed + timedelta(minutes=1))


def test_degraded_decision_still_declares_cohort(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    completed = START + timedelta(minutes=4)
    store.publish_decision(_frame(), cohort=_cohort(), run_id="run-1", completed_at=completed, entries=[_partial_entry()])
    cohort = store.read_cohort("2026-09-17", available_by=completed + timedelta(minutes=1))
    assert cohort.cohort_id == _cohort().cohort_id


def test_complete_decision_wins_over_earlier_partial(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    early = START + timedelta(seconds=40)
    late = START + timedelta(seconds=70)
    store.publish_decision(_frame(), cohort=_cohort(), run_id="run-early", completed_at=early, entries=[_partial_entry()])
    _publish_decision(store, "run-late", late)
    cutoff = late + timedelta(minutes=1)
    assert store.read_cohort("2026-09-17", available_by=cutoff).cohort_id == _cohort().cohort_id
    frame = store.read_decision("2026-09-17", available_by=cutoff)
    assert frame["capture_run_id"].iloc[0] == "run-late"


def test_resolve_capture_root_profile_override_wins(tmp_path: Path) -> None:
    from src import settings as app_settings
    from src.config.collection import CollectionSettings
    from src.data.capture_store import resolve_capture_root

    old = app_settings.COLLECTION_ROOT
    app_settings.COLLECTION_ROOT = tmp_path / "b"
    try:
        assert resolve_capture_root(CollectionSettings(COLLECTION_ROOT=tmp_path / "a")) == tmp_path / "a"
    finally:
        app_settings.COLLECTION_ROOT = old


def test_resolve_capture_root_settings_override_without_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import settings as app_settings
    from src.data.capture_store import resolve_capture_root

    monkeypatch.setattr(app_settings, "COLLECTION_ROOT", tmp_path / "b", raising=False)

    assert resolve_capture_root() == tmp_path / "b"


def test_resolve_capture_root_history_fallback_for_both_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src import settings as app_settings
    from src.config.collection import CollectionSettings
    from src.data.capture_store import resolve_capture_root

    monkeypatch.setattr(app_settings, "COLLECTION_ROOT", None, raising=False)
    monkeypatch.setattr(app_settings, "HISTORY_DIR", tmp_path / "h", raising=False)

    assert resolve_capture_root() == tmp_path / "h" / "capture"
    assert resolve_capture_root(CollectionSettings(COLLECTION_ROOT=None)) == tmp_path / "h" / "capture"


def test_resolve_capture_root_string_override_normalized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pathlib import Path as _Path

    from src.config.collection import CollectionSettings
    from src.data.capture_store import resolve_capture_root

    profile = CollectionSettings(COLLECTION_ROOT=_Path(str(tmp_path / "s")))
    result = resolve_capture_root(profile)

    assert isinstance(result, _Path)


def test_resolve_capture_root_has_no_filesystem_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src import settings as app_settings
    from src.data.capture_store import resolve_capture_root

    target = tmp_path / "no-such-dir" / "capture-root"
    assert not target.exists()
    monkeypatch.setattr(app_settings, "COLLECTION_ROOT", target, raising=False)

    assert resolve_capture_root() == target
    assert not target.exists()


def test_capture_root_has_single_owner_across_consumers() -> None:
    import ast
    import importlib
    from pathlib import Path as _Path

    from src.data.capture_store import resolve_capture_root

    consumers = [
        "src.daily.collect",
        "src.daily.price_ingest",
        "src.daily.predict",
        "src.data.intraday_store",
        "src.daily.auction_capture",
        "src.daily.archive_intraday",
        "src.daily.altdata_capture",
        "src.tools.repair_intraday_capture",
        "src.backfill.intraday.collector",
        "src.tools.daily_audit",
        "src.tools.backup_prune",
        "src.tools.offsite_backup",
    ]
    for name in consumers:
        module = importlib.import_module(name)
        assert module._capture_root is resolve_capture_root, name

    root = _Path("src")
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        offenders.extend(
            str(path)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_capture_root"
        )
    assert offenders == []
