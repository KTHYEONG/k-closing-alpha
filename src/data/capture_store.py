"""Owner-local atomic raw capture and immutable decision storage."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.capture_contracts import (
    ArtifactRef,
    BrokerPayload,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    Cohort,
    CoverageEntry,
    RawCaptureError,
)

__all__ = ["CaptureStore"]

_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_LOCK_TIMEOUT_SECONDS = 30.0
_GOOD_ENTRY_STATES = frozenset({CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE})


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


class CaptureStore:
    """Preserve first-party acquisition evidence before lossy normalization.

    Raw responses and published snapshots are append-only artifacts. Manifests
    certify complete owner-local tasks without treating another collector's
    configuration or a filename as evidence of coverage.

    Args:
        root: Configured capture directory within project data storage.

    Raises:
        OSError: Storage or locking failure.
        ValueError: Unsafe artifact paths or inconsistent persisted evidence.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._staging = self._root / "staging"
        self._staging.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Configured capture directory owning every artifact."""
        return self._root

    def _resolve(self, rel: str) -> Path:
        if os.path.isabs(rel) or ".." in Path(rel).parts:
            raise ValueError(f"unsafe artifact path: {rel!r}")
        full = self._root / rel
        base = os.path.realpath(self._root)
        target = os.path.realpath(full)
        if target != base and not target.startswith(base + os.sep):
            raise ValueError(f"artifact path escapes capture root: {rel!r}")
        return full

    def _acquire(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = target.parent / (target.name + ".lock")
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lock
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise OSError(f"timed out acquiring publish lock: {lock}") from None
                time.sleep(0.01)

    def _publish_bytes(self, rel: str, data: bytes, rows: int | None) -> ArtifactRef:
        if len(data) > _MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds bounded size: {rel!r}")
        target = self._root / rel
        lock = self._acquire(target)
        try:
            if target.exists():
                existing = target.read_bytes()
                if existing == data:
                    return ArtifactRef(path=rel, sha256=_sha256(existing), bytes=len(existing), rows=rows)
                raise ValueError(f"conflicting immutable artifact identity: {rel!r}")
            tmp_path = target.parent / f"stage-{uuid.uuid4().hex}.tmp"
            try:
                tmp_path.write_bytes(data)
                os.replace(tmp_path, target)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            return ArtifactRef(path=rel, sha256=_sha256(data), bytes=len(data), rows=rows)
        finally:
            if lock.exists():
                lock.unlink()

    def _check_ref(self, ref: ArtifactRef) -> bytes:
        full = self._resolve(ref.path)
        try:
            data = full.read_bytes()
        except FileNotFoundError:
            raise ValueError(f"missing referenced evidence: {ref.path!r}") from None
        if len(data) != ref.bytes or _sha256(data) != ref.sha256:
            raise ValueError(f"hash-inconsistent evidence: {ref.path!r}")
        return data

    @staticmethod
    def _envelope(response: CapturedResponse) -> dict[str, Any]:
        context = response.context
        scheduled = context.scheduled_at
        source_ts = response.source_timestamp
        source_pub = response.source_published_at
        return {
            "attempt_index": response.attempt_index,
            "context": {
                "capture_reason": context.capture_reason,
                "cohort_id": context.cohort_id,
                "dataset": context.dataset.value,
                "endpoint": context.endpoint,
                "run_id": context.run_id,
                "scheduled_at": scheduled.isoformat() if scheduled is not None else None,
                "session": context.session,
                "symbol": context.symbol,
                "trading_date": context.trading_date.isoformat(),
                "vendor": context.vendor,
                "venue": context.venue,
            },
            "continuation": dict(response.continuation),
            "error_type": response.error_type,
            "page_index": response.page_index,
            "payload": response.payload,
            "received_at": response.received_at.isoformat(),
            "request_started_at": response.request_started_at.isoformat(),
            "source_published_at": source_pub.isoformat() if source_pub is not None else None,
            "source_timestamp": source_ts.isoformat() if source_ts is not None else None,
            "status": response.status.value,
        }

    @staticmethod
    def _raw_rel(response: CapturedResponse) -> str:
        context = response.context
        symbol = context.symbol if context.symbol is not None else "nosymbol"
        name = f"{symbol}-p{response.page_index:04d}-a{response.attempt_index:02d}.json.gz"
        # 동일 run이 여러 endpoint(예: PRICE의 KOSPI/KOSDAQ)를 순회할 때 심볼 없는
        # 첫 페이지 이름이 겹칠 수 있으므로 endpoint를 항상 경로에 반영한다.
        # 실측: 2026-09-20 price_ingest가 KOSPI/KOSDAQ 응답을 같은 경로로 써서
        # conflicting immutable artifact identity로 크래시.
        endpoint = re.sub(r"[^A-Za-z0-9_-]+", "-", str(context.endpoint)).strip("-") or "endpoint"
        return "/".join(
            [
                "raw",
                context.trading_date.isoformat(),
                context.vendor,
                context.dataset.value,
                endpoint,
                context.run_id,
                name,
            ]
        )

    def append_response(self, response: CapturedResponse) -> ArtifactRef:
        """Persist an unmodified market-response payload and its actual observation clocks.

        Args:
            response: Validated payload, route, cursor, and observation metadata.

        Returns:
            Content-verified reference to an immutable compressed artifact.

        Raises:
            OSError: Artifact publication failed.
            ValueError: Invalid JSON evidence or unsafe context.
        """
        data = gzip.compress(_canonical_json(self._envelope(response)), compresslevel=9, mtime=0)
        return self._publish_bytes(self._raw_rel(response), data, None)

    def _verify_manifest_refs(self, manifest: CaptureManifest) -> None:
        for ref in manifest.artifacts:
            self._check_ref(ref)
        for entry in manifest.entries:
            for ref in entry.raw_refs:
                self._check_ref(ref)

    def publish_manifest(self, manifest: CaptureManifest) -> ArtifactRef:
        """Publish coverage only after its referenced evidence has been verified.

        Args:
            manifest: Task result including every declared expected entry.

        Returns:
            Verified immutable manifest reference.

        Raises:
            ValueError: Missing references, inconsistent coverage, or identity conflict.
            OSError: Publication or lock failure.
        """
        if manifest.status == CaptureStatus.COMPLETE and any(
            entry.status not in _GOOD_ENTRY_STATES for entry in manifest.entries
        ):
            raise ValueError("COMPLETE manifest cannot certify failed or pending entries")
        self._verify_manifest_refs(manifest)
        trading_date = manifest.context.trading_date.isoformat()
        rel = "/".join(
            ["manifests", trading_date, manifest.context.run_id, f"manifest-{manifest.status.value.lower()}.json"]
        )
        return self._publish_bytes(rel, _canonical_json(manifest.model_dump(mode="json")), None)

    @staticmethod
    def _decision_frame(frame: pd.DataFrame, cohort: Cohort) -> str:
        columns = list(frame.columns)
        symbol_col = "symbol" if "symbol" in columns else ("종목코드" if "종목코드" in columns else None)
        if symbol_col is None:
            raise ValueError("decision frame must carry a symbol column")
        if "admitted" not in columns:
            raise ValueError("decision frame must carry unchanged admission flags")
        if not {"snapshot_timestamp", "feature_available_timestamp"} <= set(columns):
            raise ValueError("decision frame must carry decision-time provenance columns")
        symbols = frame[symbol_col].astype(str).tolist()
        if set(symbols) != set(cohort.eligible_symbols) or len(symbols) != len(set(symbols)):
            raise ValueError("decision frame must retain every eligible cohort member exactly once")
        return symbol_col

    def publish_decision(
        self,
        frame: pd.DataFrame,
        *,
        cohort: Cohort,
        run_id: str,
        completed_at: datetime,
        entries: Sequence[CoverageEntry],
    ) -> ArtifactRef:
        """Freeze all decision-time values independently of later market outcomes.

        Args:
            frame: Enriched wide candidate input, including admission failures.
            cohort: Owner-local population and explicit eligibility rejections.
            run_id: Acquisition identity shared with raw response references.
            completed_at: Actual completion time of enrichment and persistence inputs.
            entries: Raw-response coverage for the declared acquisition task.

        Returns:
            Manifest reference to the immutable decision input and cohort.

        Raises:
            ValueError: Membership, timestamps, or evidence cannot be reconciled.
            OSError: Snapshot publication failed.
        """
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")
        self._decision_frame(frame, cohort)
        stamps = pd.to_datetime(
            pd.concat([frame["snapshot_timestamp"], frame["feature_available_timestamp"]], ignore_index=True),
            errors="raise",
        )
        if stamps.dt.tz is None:
            stamps = stamps.dt.tz_localize("Asia/Seoul")
        floor = stamps.max().to_pydatetime()
        if completed_at < floor:
            raise ValueError("completed_at precedes decision-time evidence")
        stamped = frame.copy()
        stamped["capture_run_id"] = run_id
        stamped["cohort_id"] = cohort.cohort_id
        input_rel = "/".join(["decision", cohort.trading_date.isoformat(), run_id, "input.parquet"])
        input_ref = self._publish_bytes(input_rel, self._frame_bytes(stamped), len(stamped))
        listed = list(entries)
        status = (
            CaptureStatus.COMPLETE
            if all(entry.status in _GOOD_ENTRY_STATES for entry in listed)
            else CaptureStatus.PARTIAL
        )
        first = listed[0] if listed else None
        manifest = CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=cohort.trading_date,
                run_id=run_id,
                dataset=first.dataset if first is not None else CaptureDataset.SCAN,
                vendor="owner-local",
                endpoint="decision-input",
                symbol=None,
                venue=first.venue if first is not None else "UNKNOWN",
                session=first.session if first is not None else "regular",
                capture_reason="decision-input",
                cohort_id=cohort.cohort_id,
                scheduled_at=None,
            ),
            cohort=cohort,
            completed_at=completed_at,
            entries=tuple(listed),
            artifacts=(input_ref,),
            status=status,
        )
        return self.publish_manifest(manifest)

    def _frame_bytes(self, frame: pd.DataFrame) -> bytes:
        tmp_path = self._staging / f"stage-{uuid.uuid4().hex}.parquet"
        try:
            frame.to_parquet(tmp_path, index=False)
            return tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _cutoff(available_by: datetime) -> datetime:
        if not isinstance(available_by, datetime) or available_by.tzinfo is None or available_by.utcoffset() is None:
            raise ValueError("available_by must be an aware cutoff")
        return available_by

    def read_manifests(self, snapshot_date: str) -> tuple[CaptureManifest, ...]:
        """Expose verified owner-local task states for coverage auditing.

        Args:
            snapshot_date: Market date whose artifacts must be checked.

        Returns:
            Validated manifests ordered by completion time and run identity.

        Raises:
            ValueError: Existing evidence is unreadable or hash-inconsistent.
            OSError: The store cannot be read.
        """
        day = date.fromisoformat(snapshot_date)
        day_dir = self._root / "manifests" / day.isoformat()
        if not day_dir.exists():
            return ()
        found: list[CaptureManifest] = []
        for path in sorted(day_dir.rglob("manifest-*.json")):
            try:
                raw = json.loads(path.read_bytes())
            except (OSError, ValueError) as exc:
                raise ValueError(f"unreadable manifest evidence: {path.name!r}") from exc
            try:
                manifest = CaptureManifest.model_validate(raw)
            except ValueError as exc:
                raise ValueError(f"invalid manifest evidence: {path.name!r}") from exc
            self._verify_manifest_refs(manifest)
            found.append(manifest)
        found.sort(key=lambda item: (item.completed_at, item.context.run_id))
        return tuple(found)

    def _select_decision(
        self, snapshot_date: str, available_by: datetime, run_id: str | None
    ) -> CaptureManifest:
        cutoff = self._cutoff(available_by)
        qualifying = [
            item
            for item in self.read_manifests(snapshot_date)
            if item.status == CaptureStatus.COMPLETE
            and item.completed_at <= cutoff
            and (run_id is None or item.context.run_id == run_id)
        ]
        if not qualifying:
            raise FileNotFoundError(f"no qualifying verified decision capture: {snapshot_date!r}")
        return max(qualifying, key=lambda item: (item.completed_at, item.context.run_id))

    def read_decision(
        self, snapshot_date: str, *, available_by: datetime, run_id: str | None = None
    ) -> pd.DataFrame:
        """Load decision evidence observable before an explicit inference cutoff.

        Args:
            snapshot_date: Market date to replay.
            available_by: Aware cutoff of information available to the caller.
            run_id: Exact capture identity; None selects the latest qualifying run.

        Returns:
            A fresh frame retaining original decision values and provenance columns.

        Raises:
            FileNotFoundError: No qualifying verified decision capture exists.
            ValueError: Artifact hashes, clocks, or schema validation fail.
        """
        chosen = self._select_decision(snapshot_date, available_by, run_id)
        inputs = [ref for ref in chosen.artifacts if ref.path.endswith("/input.parquet")]
        if not inputs:
            raise ValueError("qualifying manifest carries no decision input")
        if chosen.cohort is None:
            raise ValueError("qualifying manifest carries no cohort")
        data = self._check_ref(inputs[0])
        tmp_path = self._staging / f"stage-{uuid.uuid4().hex}.parquet"
        try:
            tmp_path.write_bytes(data)
            frame = pd.read_parquet(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        cohort = chosen.cohort
        symbol_col = "symbol" if "symbol" in frame.columns else ("종목코드" if "종목코드" in frame.columns else None)
        if symbol_col is None:
            raise ValueError("persisted decision input carries no symbol column")
        symbols = frame[symbol_col].astype(str).tolist()
        if set(symbols) != set(cohort.eligible_symbols) or len(symbols) != len(set(symbols)):
            raise ValueError("persisted decision input cannot be reconciled with its cohort")
        return frame.copy()

    def read_cohort(self, snapshot_date: str, *, available_by: datetime) -> Cohort:
        """Resolve a declared cohort without inferring it from trading picks or stale days.

        A PARTIAL decision manifest still declares the day's population; quote
        coverage does not change membership. Trading readers use read_decision,
        which accepts COMPLETE only.

        Args:
            snapshot_date: Exact actual trading date requested by the caller.
            available_by: Latest permitted capture publication time.

        Returns:
            Latest verified eligible and rejected cohort for that exact date.

        Raises:
            FileNotFoundError: The requested date has no qualifying cohort.
            ValueError: Persisted identity or references are inconsistent.
        """
        cutoff = self._cutoff(available_by)
        qualifying = [
            item
            for item in self.read_manifests(snapshot_date)
            if item.status in (CaptureStatus.COMPLETE, CaptureStatus.PARTIAL)
            and item.completed_at <= cutoff
            and item.cohort is not None
        ]
        if not qualifying:
            raise FileNotFoundError(f"no qualifying cohort: {snapshot_date!r}")
        chosen = max(qualifying, key=lambda item: (item.completed_at, item.context.run_id))
        cohort = chosen.cohort
        assert cohort is not None
        return cohort

    def publish_frame(self, frame: pd.DataFrame, *, context: CaptureContext) -> ArtifactRef:
        """Retain an immutable normalized outcome independently of decision inputs.

        Args:
            frame: Normalized outcome rows with preserved nulls.
            context: Dated owner-local acquisition identity.
        Returns:
            Verified immutable Parquet reference.
        Raises:
            RawCaptureError: Durable publication fails.
            ValueError: Identity conflicts with an existing artifact.
        """
        symbol = context.symbol if context.symbol is not None else "nosymbol"
        rel = "/".join(
            [
                "normalized",
                context.trading_date.isoformat(),
                context.run_id,
                f"{context.dataset.value}-{symbol}.parquet",
            ]
        )
        try:
            return self._publish_bytes(rel, self._frame_bytes(frame), len(frame))
        except OSError as exc:
            raise RawCaptureError(str(exc)) from exc

    def read_artifact(self, ref: ArtifactRef) -> BrokerPayload:
        """Verify and read a serialized raw-response envelope without changing its clocks.

        Args:
            ref: Root-relative expected path, checksum and byte length.
        Returns:
            Decoded CapturedResponse envelope fields with unknown clocks unchanged.
        Raises:
            ValueError: Unsafe path, corrupt bytes or malformed serialized evidence.
            OSError: The artifact cannot be read.
        """
        full = self._resolve(ref.path)
        data = full.read_bytes()
        if len(data) != ref.bytes or _sha256(data) != ref.sha256:
            raise ValueError(f"hash-inconsistent evidence: {ref.path!r}")
        try:
            envelope: dict[str, Any] = json.loads(gzip.decompress(data).decode("utf-8"))
        except OSError as exc:
            raise ValueError(f"corrupt raw evidence: {ref.path!r}") from exc
        if not isinstance(envelope, dict) or "payload" not in envelope or "context" not in envelope:
            raise ValueError(f"malformed raw evidence: {ref.path!r}")
        return envelope
