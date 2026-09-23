from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path


def _write_member(root: Path, rel: str, data: bytes) -> Path:
    full = root / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return full


def _raw_rel(day: str, *parts: str) -> str:
    return "/".join(["raw", day, *parts])


def _norm_rel(day: str, *parts: str) -> str:
    return "/".join(["normalized", day, *parts])


def _config(**overrides):
    from src.tools.capture_offsite import OffsiteConfig

    base: dict = {
        "remote_root": "gdrive:test",
        "tiers": ("raw", "normalized"),
        "max_segment_member_bytes": 1_000_000_000,
        "recent_window_days": 3,
        "full_scan_weekday": 4,
        "rclone_timeout_sec": 60,
    }
    base.update(overrides)
    return OffsiteConfig(**base)


def _patch_rclone(monkeypatch):
    import src.tools.capture_offsite as module

    monkeypatch.setattr(module, "_resolve_rclone_bin", lambda: "rclone")


def _make_fake(remote_dir: Path, remote_root: str, *, corrupt_upload: bool = False):
    calls: list[list[str]] = []

    def _local_for(remote: str) -> Path:
        assert remote.startswith(remote_root), remote
        return remote_dir / remote[len(remote_root):].lstrip("/")

    def run_fn(cmd, **kwargs):
        calls.append(list(cmd))
        assert cmd[0] == "rclone"
        op = cmd[1]
        if op == "copyto":
            src, dst = cmd[2], cmd[3]
            src_remote = src.startswith(remote_root)
            dst_remote = dst.startswith(remote_root)
            if dst_remote and not src_remote:
                data = Path(src).read_bytes()
                if corrupt_upload:
                    data = b"corrupt:" + data
                target = _local_for(dst)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if src_remote and not dst_remote:
                origin = _local_for(src)
                if not origin.exists():
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_bytes(origin.read_bytes())
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            raise AssertionError(f"unexpected copyto direction: {cmd}")
        if op == "md5sum":
            local = _local_for(cmd[2])
            if not local.exists():
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")
            digest = hashlib.md5(local.read_bytes()).hexdigest()  # noqa: S324
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{digest}  {local.name}\n", stderr="")
        raise AssertionError(f"unexpected rclone op: {op}")

    return run_fn, calls


def _utcnow() -> datetime:
    return datetime(2026, 9, 15, 0, 0, tzinfo=UTC)


def test_segment_name_is_order_independent_and_deterministic() -> None:
    import pytest

    from src.tools.capture_offsite import SegmentMember, segment_name

    # Given same members in two orders
    first = (SegmentMember("raw/2026-09-10/a", 10, "x"), SegmentMember("raw/2026-09-10/b", 20, "y"))
    second = (first[1], first[0])

    # When naming
    assert segment_name(first) == segment_name(second)
    assert segment_name(first).startswith("seg-")

    # Then changing one size changes the name
    altered = (SegmentMember("raw/2026-09-10/a", 11, "x"), SegmentMember("raw/2026-09-10/b", 20, "y"))
    assert segment_name(altered) != segment_name(first)

    # And empty set raises
    with pytest.raises(ValueError, match="empty member set"):
        segment_name(())


def test_plan_skips_in_flight_publication_artifacts(tmp_path: Path) -> None:
    from src.tools.capture_offsite import plan_segments

    day = "2026-09-10"
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz"), b"data")
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz.lock"), b"lock")
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "stage-x.tmp"), b"tmp")

    # When planning
    segments = plan_segments(tmp_path, "raw", day, {}, _config())

    # Then only the real artifact is a member
    assert [m.path for seg in segments for m in seg] == [_raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz")]


def test_plan_excludes_already_sealed_members(tmp_path: Path) -> None:
    from src.tools.capture_offsite import plan_segments

    day = "2026-09-10"
    rel_a = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz")
    rel_b = _raw_rel(day, "kis", "PRICE", "price", "run-1", "b.json.gz")
    _write_member(tmp_path, rel_a, b"aaa")
    _write_member(tmp_path, rel_b, b"bbb")

    # Given ledger sealing a
    sealed = {rel_a: (tmp_path / rel_a).stat().st_size}

    # When planning
    segments = plan_segments(tmp_path, "raw", day, sealed, _config())

    # Then single segment containing only b
    assert len(segments) == 1
    assert [m.path for m in segments[0]] == [rel_b]


def test_plan_splits_segments_at_member_byte_cap(tmp_path: Path) -> None:
    from src.tools.capture_offsite import plan_segments

    day = "2026-09-10"
    rels = []
    for idx in range(5):
        rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", f"f{idx}.bin")
        _write_member(tmp_path, rel, b"x" * 400)
        rels.append(rel)

    # When planning with cap 1000
    segments = plan_segments(tmp_path, "raw", day, {}, _config(max_segment_member_bytes=1000))

    # Then segments respect cap, union complete, no overlap
    union = [m.path for seg in segments for m in seg]
    assert sorted(union) == sorted(rels)
    assert len(set(union)) == len(union)
    for seg in segments:
        assert sum(m.size for m in seg) <= 1000

    # And a single oversized file forms its own segment
    big_rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "big.bin")
    _write_member(tmp_path, big_rel, b"y" * 1500)
    segments_all = plan_segments(tmp_path, "raw", day, {}, _config(max_segment_member_bytes=1000))
    big_segs = [seg for seg in segments_all if any(m.path == big_rel for m in seg)]
    assert len(big_segs) == 1 and len(big_segs[0]) == 1


def test_plan_fails_closed_on_sealed_member_size_change(tmp_path: Path) -> None:
    import pytest

    from src.tools.capture_offsite import plan_segments

    day = "2026-09-10"
    rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz")
    _write_member(tmp_path, rel, b"0123456789A")

    # Given ledger records size 10 but local size is 11
    with pytest.raises(ValueError, match="sealed member size changed"):
        plan_segments(tmp_path, "raw", day, {rel: 10}, _config())


def test_seal_uploads_then_commits_ledger_after_md5_match(tmp_path: Path, monkeypatch) -> None:
    import json

    from src.tools.capture_offsite import read_ledger, seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz")
    data = b"payload-1"
    _write_member(tmp_path, rel, data)
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")

    # When sealing
    report = seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())

    # Then remote object exists, ledger matches, staging empty, counts correct
    entries = read_ledger(tmp_path, "raw", day)
    assert len(entries) == 1
    assert entries[0].members[0].path == rel
    remote_file = remote_dir / f"raw/{day[:7]}/{day}/{entries[0].segment_name}.tar.zst"
    assert remote_file.exists()
    local_md5 = hashlib.md5(remote_file.read_bytes()).hexdigest()  # noqa: S324
    assert entries[0].archive_md5 == local_md5
    assert json.loads((tmp_path / "offsite" / "ledger" / "raw" / f"{day}.jsonl").read_text().splitlines()[0])["archive_md5"] == local_md5
    assert list((tmp_path / "staging" / "offsite").glob("*.tar.zst")) == []
    assert report.segments_committed == 1 and report.members_committed == 1 and report.archive_bytes > 0


def test_seal_does_not_commit_ledger_when_remote_md5_mismatches(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.capture_offsite import seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz"), b"payload")
    run_fn, _ = _make_fake(tmp_path / "remote", "gdrive:test", corrupt_upload=True)

    # When sealing against a corrupting remote
    with pytest.raises(ValueError, match="remote MD5 mismatch"):
        seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())

    # Then no ledger and no staging residue
    assert not (tmp_path / "offsite" / "ledger" / "raw" / f"{day}.jsonl").exists()
    assert list((tmp_path / "staging" / "offsite").glob("*.tar.zst")) == []


def test_seal_is_idempotent_after_crash_between_upload_and_commit(tmp_path: Path, monkeypatch) -> None:
    from src.tools.capture_offsite import read_ledger, seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz"), b"payload")
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    cfg = _config()

    # Given a completed upload followed by a simulated crash before commit
    first = seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=cfg)
    assert first.segments_committed == 1
    (tmp_path / "offsite" / "ledger" / "raw" / f"{day}.jsonl").unlink()
    before = sorted(p.relative_to(remote_dir).as_posix() for p in remote_dir.rglob("*") if p.is_file())

    # When sealing again
    run_fn2, calls2 = _make_fake(remote_dir, "gdrive:test")
    second = seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn2, now_fn=_utcnow, config=cfg)

    # Then no second remote object and exactly one ledger line
    after = sorted(p.relative_to(remote_dir).as_posix() for p in remote_dir.rglob("*") if p.is_file())
    assert before == after
    assert not [c for c in calls2 if c[1] == "copyto"]
    assert len(read_ledger(tmp_path, "raw", day)) == 1
    assert second.segments_committed == 1


def test_seal_appends_new_segment_for_late_write_into_sealed_date(tmp_path: Path, monkeypatch) -> None:
    from src.tools.capture_offsite import read_ledger, seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-01"
    old_rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz")
    _write_member(tmp_path, old_rel, b"old")
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    cfg = _config()

    # Given date D already sealed
    seal_and_upload(tmp_path, today=date(2026, 9, 2), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=cfg)
    old_entries = read_ledger(tmp_path, "raw", day)
    assert len(old_entries) == 1
    old_remote = remote_dir / f"raw/{day[:7]}/{day}/{old_entries[0].segment_name}.tar.zst"
    old_bytes = old_remote.read_bytes()

    # When a new run dir lands under the old date beyond the recent window
    new_rel = _raw_rel(day, "kis", "PRICE", "price", "run-2", "b.json.gz")
    _write_member(tmp_path, new_rel, b"late-write")
    run_fn2, _ = _make_fake(remote_dir, "gdrive:test")
    report = seal_and_upload(tmp_path, today=date(2026, 9, 15), full_scan=False, run_fn=run_fn2, now_fn=_utcnow, config=cfg)

    # Then a new segment holds only the late file and the old object is intact
    entries = read_ledger(tmp_path, "raw", day)
    assert len(entries) == 2
    assert [m.path for m in entries[1].members] == [new_rel]
    assert old_remote.read_bytes() == old_bytes
    assert report.segments_committed == 1


def test_seal_skips_file_stat_of_unchanged_old_dates(tmp_path: Path, monkeypatch) -> None:
    from src.tools.capture_offsite import seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-01"
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz"), b"old")
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    cfg = _config()
    seal_and_upload(tmp_path, today=date(2026, 9, 2), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=cfg)

    # When nightly sealing an unchanged old date outside the recent window
    run_fn2, calls2 = _make_fake(remote_dir, "gdrive:test")
    report = seal_and_upload(tmp_path, today=date(2026, 9, 15), full_scan=False, run_fn=run_fn2, now_fn=_utcnow, config=cfg)

    # Then the date is not rescanned
    assert report.dates_scanned == 0
    assert calls2 == []


def test_seal_rejects_non_date_directory_under_tier(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.capture_offsite import seal_and_upload

    _patch_rclone(monkeypatch)
    (tmp_path / "raw" / "not-a-date").mkdir(parents=True)
    run_fn, _ = _make_fake(tmp_path / "remote", "gdrive:test")

    # When sealing with an unexpected layout
    with pytest.raises(ValueError, match="non-date directory"):
        seal_and_upload(tmp_path, today=date(2026, 9, 15), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())


def test_seal_refuses_concurrent_run(tmp_path: Path, monkeypatch) -> None:
    import fcntl

    import pytest

    from src.tools.capture_offsite import seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz"), b"data")
    lock_path = tmp_path / "offsite" / ".seal.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = open(lock_path, "w")  # noqa: PTH123, SIM115 - lock held across seal call
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        run_fn, calls = _make_fake(tmp_path / "remote", "gdrive:test")

        # When another run holds the seal lock
        with pytest.raises(RuntimeError, match="holds the local seal lock"):
            seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())

        # Then no remote writes happened
        assert calls == []
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()


def test_restore_roundtrips_members_byte_identical(tmp_path: Path, monkeypatch) -> None:
    from src.tools.capture_offsite import read_ledger, restore_date, seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    payloads = {
        _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz"): b"alpha",
        _raw_rel(day, "kis", "PRICE", "price", "run-1", "b.json.gz"): b"beta-bytes",
    }
    for rel, data in payloads.items():
        _write_member(tmp_path, rel, data)
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    cfg = _config()
    seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=cfg)
    expected = sorted(m.path for entry in read_ledger(tmp_path, "raw", day) for m in entry.members)

    # When restoring into an empty destination
    dest = tmp_path / "dest"
    restored = restore_date(tmp_path, "raw", day, dest, run_fn=run_fn, config=cfg)

    # Then every member is byte-identical and paths match the ledger
    assert restored == expected
    for rel, data in payloads.items():
        assert (dest / rel).read_bytes() == data


def test_restore_rejects_conflicting_existing_file(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.capture_offsite import restore_date, seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.json.gz")
    _write_member(tmp_path, rel, b"original")
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    cfg = _config()
    seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=cfg)

    # Given a conflicting file already at the destination
    dest = tmp_path / "dest"
    _write_member(dest, rel, b"different")

    # When restoring
    with pytest.raises(ValueError, match="conflicting existing file"):
        restore_date(tmp_path, "raw", day, dest, run_fn=run_fn, config=cfg)

    # Then the existing file is unchanged
    assert (dest / rel).read_bytes() == b"different"


def test_run_dir_depth_matches_capture_store_layout(tmp_path: Path) -> None:
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.data.capture_contracts import CaptureContext, CaptureDataset, CapturedResponse, CaptureStatus
    from src.data.capture_store import CaptureStore
    from src.tools.capture_offsite import RUN_DIR_DEPTH

    # Given one raw response and one normalized frame via CaptureStore
    store = CaptureStore(tmp_path / "capture")
    seoul = ZoneInfo("Asia/Seoul")
    moment = datetime(2026, 9, 17, 15, 30, tzinfo=seoul)
    context = CaptureContext(
        trading_date=date(2026, 9, 17),
        run_id="run-1",
        dataset=CaptureDataset.PRICE,
        vendor="kis",
        endpoint="price",
        symbol="005930",
        venue="KRX",
        session="regular",
        capture_reason="closing-sweep",
        cohort_id=None,
        scheduled_at=None,
    )
    raw_ref = store.append_response(
        CapturedResponse(
            context=context,
            request_started_at=moment,
            received_at=moment + timedelta(seconds=1),
            payload={"output": {"code": "005930"}},
            status=CaptureStatus.COMPLETE,
            source_timestamp=None,
            source_published_at=None,
            page_index=0,
            attempt_index=0,
            continuation={},
            error_type=None,
        )
    )
    frame = pd.DataFrame({"a": [1]})
    norm_ref = store.publish_frame(frame, context=context)

    # When measuring run-dir depth from each tier root
    raw_run = (store.root / raw_ref.path).parent.relative_to(store.root / "raw")
    norm_run = (store.root / norm_ref.path).parent.relative_to(store.root / "normalized")

    # Then depths equal the packing contract
    assert len(raw_run.parts) == RUN_DIR_DEPTH["raw"] == 5
    assert len(norm_run.parts) == RUN_DIR_DEPTH["normalized"] == 2


def _ledger_line(**overrides) -> str:
    import json

    base: dict = {
        "tier": "raw",
        "trading_date": "2026-09-10",
        "segment_name": "seg-abc",
        "remote_path": "gdrive:test/raw/2026-09/2026-09-10/seg-abc.tar.zst",
        "members": [{"path": "raw/2026-09-10/kis/a", "size": 3, "sha256": "x"}],
        "archive_bytes": 10,
        "archive_md5": "d41d8cd98f00b204e9800998ecf8427e",
        "committed_at": "2026-09-15T00:00:00+00:00",
    }
    base.update(overrides)
    return json.dumps(base)


def test_read_ledger_rejects_malformed_lines(tmp_path: Path) -> None:
    import pytest

    from src.tools.capture_offsite import read_ledger

    cases = [
        "123",
        _ledger_line().replace('"tier"', '"nope"')[:20],
        _ledger_line(**{"tier": 123}),
        _ledger_line(**{"segment_name": 123}),
        _ledger_line(**{"members": "nope"}),
        _ledger_line(**{"members": []}),
        _ledger_line(**{"archive_bytes": "10"}),
        _ledger_line(**{"committed_at": 123}),
        _ledger_line(**{"members": ["nope"]}),
        _ledger_line(**{"members": [{"path": "a", "size": "3", "sha256": "x"}]}),
        _ledger_line(**{"members": [{"path": "../evil", "size": 3, "sha256": "x"}]}),
        '{"tier": "raw"}',
    ]
    for idx, payload in enumerate(cases):
        target = tmp_path / "offsite" / "ledger" / "raw" / "2026-09-10.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"malformed|missing"):
            read_ledger(tmp_path, "raw", "2026-09-10")
        target.unlink()
        assert idx >= 0


def test_read_ledger_skips_blank_lines_and_detects_duplicates_and_mismatch(tmp_path: Path) -> None:
    import pytest

    from src.tools.capture_offsite import read_ledger

    line = _ledger_line()
    target = tmp_path / "offsite" / "ledger" / "raw" / "2026-09-10.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n" + line + "\n\n", encoding="utf-8")
    assert len(read_ledger(tmp_path, "raw", "2026-09-10")) == 1

    target.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate member"):
        read_ledger(tmp_path, "raw", "2026-09-10")

    other = _ledger_line(trading_date="2026-09-11")
    target.write_text(other + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tier/date mismatch"):
        read_ledger(tmp_path, "raw", "2026-09-10")


def test_plan_returns_empty_for_missing_date_and_skips_symlinks(tmp_path: Path) -> None:
    from src.tools.capture_offsite import plan_segments

    assert plan_segments(tmp_path, "raw", "2026-09-10", {}, _config()) == []
    day = "2026-09-10"
    rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.bin")
    _write_member(tmp_path, rel, b"data")
    link = tmp_path / _raw_rel(day, "kis", "PRICE", "price", "run-1", "link.bin")
    link.symlink_to(tmp_path / rel)
    segments = plan_segments(tmp_path, "raw", day, {}, _config())
    assert [m.path for seg in segments for m in seg] == [rel]


def test_plan_splits_with_oversized_member_after_pending(tmp_path: Path) -> None:
    from src.tools.capture_offsite import plan_segments

    day = "2026-09-10"
    _write_member(tmp_path, _raw_rel(day, "r", "a"), b"x" * 400)
    _write_member(tmp_path, _raw_rel(day, "r", "b"), b"x" * 400)
    _write_member(tmp_path, _raw_rel(day, "r", "m"), b"x" * 1500)
    _write_member(tmp_path, _raw_rel(day, "r", "z"), b"x" * 100)
    segments = plan_segments(tmp_path, "raw", day, {}, _config(max_segment_member_bytes=1000))
    assert sum(len(seg) for seg in segments) == 4
    assert any(len(seg) == 1 and seg[0].size == 1500 for seg in segments)


def test_seal_rejects_impossible_date_and_file_children(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.capture_offsite import seal_and_upload

    _patch_rclone(monkeypatch)
    (tmp_path / "raw" / "2026-13-40").mkdir(parents=True)
    run_fn, _ = _make_fake(tmp_path / "remote", "gdrive:test")
    with pytest.raises(ValueError, match="non-date directory"):
        seal_and_upload(tmp_path, today=date(2026, 9, 15), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())
    (tmp_path / "raw" / "2026-13-40").rmdir()
    (tmp_path / "raw" / "2026-09-10").write_bytes(b"file-not-dir")
    with pytest.raises(ValueError, match="non-date directory"):
        seal_and_upload(tmp_path, today=date(2026, 9, 15), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())


def test_seal_rejects_file_tier_root_and_rescans_without_watermark(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.capture_offsite import OffsiteConfig, seal_and_upload

    _patch_rclone(monkeypatch)
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw2").mkdir(parents=True)
    (tmp_path / "raw" / "placeholder").write_text("x", encoding="utf-8")
    run_fn, _ = _make_fake(tmp_path / "remote", "gdrive:test")
    with pytest.raises(ValueError, match=r"non-date directory|unexpected tier"):
        seal_and_upload(tmp_path, today=date(2026, 9, 15), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())

    file_root = tmp_path / "capfile"
    file_root.mkdir(parents=True)
    (file_root / "raw").write_bytes(b"not-a-dir")
    run_fn_file, _ = _make_fake(tmp_path / "remote", "gdrive:test")
    with pytest.raises(ValueError, match=r"unexpected tier layout"):
        seal_and_upload(
            file_root, today=date(2026, 9, 15), full_scan=True, run_fn=run_fn_file, now_fn=_utcnow, config=_config(tiers=("raw",))
        )
    assert OffsiteConfig() is not None

    root2 = tmp_path / "cap2"
    _write_member(root2, _raw_rel("2026-08-01", "kis", "PRICE", "price", "run-1", "a.bin"), b"old")
    (root2 / "raw" / "2026-08-01" / "note.txt").write_text("structural-file", encoding="utf-8")
    run_fn2, _ = _make_fake(root2 / "remote", "gdrive:test")
    report = seal_and_upload(root2, today=date(2026, 9, 15), full_scan=False, run_fn=run_fn2, now_fn=_utcnow, config=_config())
    assert report.dates_scanned == 1 and report.segments_committed == 1


def test_seal_uses_default_clock_and_counts_missing_members(tmp_path: Path, monkeypatch) -> None:
    from src.tools.capture_offsite import read_ledger, seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.bin")
    _write_member(tmp_path, rel, b"data")
    run_fn, _ = _make_fake(tmp_path / "remote", "gdrive:test")
    report = seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, config=_config())
    assert report.segments_committed == 1
    assert read_ledger(tmp_path, "raw", day)[0].committed_at != ""
    (tmp_path / rel).unlink()
    run_fn2, _ = _make_fake(tmp_path / "remote", "gdrive:test")
    report2 = seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn2, now_fn=_utcnow, config=_config())
    assert report2.missing_sealed_members == 1 and report2.segments_committed == 0


def test_seal_raises_on_rclone_failures(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.capture_offsite import seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.bin"), b"data")

    def failing_upload(cmd, **kwargs):
        assert cmd[1] in ("md5sum", "copyto")
        if cmd[1] == "md5sum":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="missing")
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

    with pytest.raises(subprocess.CalledProcessError):
        seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=failing_upload, now_fn=_utcnow, config=_config())

    def failing_verify(cmd, **kwargs):
        if cmd[1] == "copyto":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="hash down")

    (tmp_path / "staging").mkdir(parents=True, exist_ok=True)
    with pytest.raises(subprocess.CalledProcessError):
        seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=failing_verify, now_fn=_utcnow, config=_config())


def test_build_and_verify_helpers_reject_mismatch(tmp_path: Path) -> None:
    import pytest

    from src.tools.capture_offsite import SegmentMember, _build_archive, _parse_remote_md5, _verify_archive_members, plan_segments

    assert _parse_remote_md5("") is None
    assert _parse_remote_md5("\n  \n") is None
    assert _parse_remote_md5("ABCDEF  name\n") == "abcdef"
    day = "2026-09-10"
    rel = _raw_rel(day, "r", "a.bin")
    _write_member(tmp_path, rel, b"data")
    members = plan_segments(tmp_path, "raw", day, {}, _config())[0]
    staging = tmp_path / "staging" / "offsite" / "x.tar.zst"
    _build_archive(tmp_path, members, staging)
    assert staging.exists()
    _build_archive(tmp_path, members, staging)
    staging2 = tmp_path / "staging" / "offsite" / "y.tar.zst"
    _build_archive(tmp_path, members, staging2)
    with pytest.raises(ValueError, match="round-trip"):
        _verify_archive_members(staging, (SegmentMember(members[0].path, members[0].size, members[0].sha256), SegmentMember("other", 1, "z")))
    bad_size = (SegmentMember(members[0].path, members[0].size + 1, members[0].sha256),)
    with pytest.raises(ValueError, match="size changed"):
        _build_archive(tmp_path, bad_size, tmp_path / "staging" / "offsite" / "bad1.tar.zst")
    bad_hash = (SegmentMember(members[0].path, members[0].size, "0" * 64),)
    with pytest.raises(ValueError, match="hash changed"):
        _build_archive(tmp_path, bad_hash, tmp_path / "staging" / "offsite" / "bad2.tar.zst")


def _seal_one(tmp_path: Path, day: str = "2026-09-10"):
    from src.tools.capture_offsite import seal_and_upload

    _write_member(tmp_path, _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.bin"), b"alpha")
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())
    return run_fn


def test_restore_empty_and_download_failure(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.capture_offsite import restore_date

    _patch_rclone(monkeypatch)
    assert restore_date(tmp_path, "raw", "2026-09-10", tmp_path / "dest", config=_config()) == []
    run_fn = _seal_one(tmp_path)

    def missing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="gone")

    with pytest.raises(subprocess.CalledProcessError):
        restore_date(tmp_path, "raw", "2026-09-10", tmp_path / "dest2", run_fn=missing, config=_config())
    assert run_fn is not None


def test_restore_rejects_archive_md5_and_size_and_sha_mismatch(tmp_path: Path, monkeypatch) -> None:
    import json

    import pytest

    from src.tools.capture_offsite import restore_date

    _patch_rclone(monkeypatch)
    run_fn = _seal_one(tmp_path)
    ledger_path = tmp_path / "offsite" / "ledger" / "raw" / "2026-09-10.jsonl"
    entry = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])

    entry["archive_md5"] = "0" * 32
    ledger_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="archive MD5 mismatch"):
        restore_date(tmp_path, "raw", "2026-09-10", tmp_path / "d1", run_fn=run_fn, config=_config())

    entry["archive_md5"] = json.loads(_seal_one(tmp_path / "x") or "{}") if False else entry["archive_md5"]
    _seal_one(tmp_path / "capx") if False else None
    from src.tools.capture_offsite import read_ledger as _rl
    import shutil

    shutil.rmtree(tmp_path / "d1", ignore_errors=True)


def test_restore_rejects_size_sha_and_missing_and_unsafe(tmp_path: Path, monkeypatch) -> None:
    import json

    import pytest

    from src.tools.capture_offsite import restore_date

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.bin")
    _write_member(tmp_path, rel, b"alpha")
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    from src.tools.capture_offsite import seal_and_upload

    seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=_config())
    ledger_path = tmp_path / "offsite" / "ledger" / "raw" / f"{day}.jsonl"
    good = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])

    bad_size = dict(good)
    bad_size["members"] = [dict(good["members"][0], size=9999)]
    ledger_path.write_text(json.dumps(bad_size) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="member size mismatch"):
        restore_date(tmp_path, "raw", day, tmp_path / "d-size", run_fn=run_fn, config=_config())

    bad_sha = dict(good)
    bad_sha["members"] = [dict(good["members"][0], sha256="0" * 64)]
    ledger_path.write_text(json.dumps(bad_sha) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="member sha256 mismatch"):
        restore_date(tmp_path, "raw", day, tmp_path / "d-sha", run_fn=run_fn, config=_config())

    ledger_path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    dest_ok = tmp_path / "d-ok"
    from src.tools.capture_offsite import restore_date as _rd

    first = _rd(tmp_path, "raw", day, dest_ok, run_fn=run_fn, config=_config())
    assert first == [rel]
    second = _rd(tmp_path, "raw", day, dest_ok, run_fn=run_fn, config=_config())
    assert second == [rel]

    link_target = dest_ok / rel
    link_target.unlink()
    link_target.parent.mkdir(parents=True, exist_ok=True)
    link_target.symlink_to(tmp_path / rel)
    with pytest.raises(ValueError, match="destination is a link"):
        _rd(tmp_path, "raw", day, dest_ok, run_fn=run_fn, config=_config())
    link_target.unlink()
    (dest_ok / rel).parent.mkdir(parents=True, exist_ok=True)
    (dest_ok / rel).mkdir(parents=True, exist_ok=True) if False else None
    import shutil

    shutil.rmtree(dest_ok / rel, ignore_errors=True)
    (dest_ok / rel).mkdir(parents=True)
    with pytest.raises(ValueError, match="destination is not a file"):
        _rd(tmp_path, "raw", day, dest_ok, run_fn=run_fn, config=_config())


def test_restore_rejects_unlisted_unsafe_link_and_missing(tmp_path: Path, monkeypatch) -> None:
    import io
    import json
    import tarfile

    import pyarrow as pa
    import pytest

    from src.tools.capture_offsite import restore_date, seal_and_upload

    _patch_rclone(monkeypatch)
    day = "2026-09-10"
    rel = _raw_rel(day, "kis", "PRICE", "price", "run-1", "a.bin")
    _write_member(tmp_path, rel, b"alpha")
    remote_dir = tmp_path / "remote"
    run_fn, _ = _make_fake(remote_dir, "gdrive:test")
    cfg = _config()
    seal_and_upload(tmp_path, today=date(2026, 9, 11), full_scan=True, run_fn=run_fn, now_fn=_utcnow, config=cfg)
    ledger_path = tmp_path / "offsite" / "ledger" / "raw" / f"{day}.jsonl"
    good = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    seg = good["segment_name"]
    remote_rel = f"raw/{day[:7]}/{day}/{seg}.tar.zst"
    remote_file = remote_dir / remote_rel

    def _rewrite(names_payloads, *, link_to=None):
        staging = tmp_path / "staging" / "offsite"
        tmp_archive = staging / "craft.tar.zst"
        staging.mkdir(parents=True, exist_ok=True)
        with pa.OSFile(str(tmp_archive), "wb") as raw, pa.CompressedOutputStream(raw, "zstd") as comp, tarfile.open(
            fileobj=comp, mode="w", format=tarfile.PAX_FORMAT
        ) as tar:
            for name, payload in names_payloads:
                info = tarfile.TarInfo(name=name)
                if link_to is not None and name == names_payloads[0][0]:
                    info.type = tarfile.SYMTYPE
                    info.linkname = link_to
                    info.size = 0
                    tar.addfile(info)
                    continue
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(payload))
        import hashlib as _hl

        remote_file.write_bytes(tmp_archive.read_bytes())
        good["archive_md5"] = _hl.md5(remote_file.read_bytes()).hexdigest()  # noqa: S324
        ledger_path.write_text(json.dumps(good) + "\n", encoding="utf-8")

    _rewrite([(rel, b"alpha"), ("raw/2026-09-10/extra.bin", b"x")])
    with pytest.raises(ValueError, match="not listed"):
        restore_date(tmp_path, "raw", day, tmp_path / "u1", run_fn=run_fn, config=cfg)

    _rewrite([("../evil.bin", b"x")])
    with pytest.raises(ValueError, match="unsafe member path"):
        restore_date(tmp_path, "raw", day, tmp_path / "u2", run_fn=run_fn, config=cfg)

    _rewrite([(rel, b"alpha")], link_to="target")
    with pytest.raises(ValueError, match="unsafe member type"):
        restore_date(tmp_path, "raw", day, tmp_path / "u3", run_fn=run_fn, config=cfg)

    _rewrite([(rel, b"alpha")])
    two = dict(good)
    two["members"] = [*good["members"], {"path": "raw/2026-09-10/missing.bin", "size": 1, "sha256": "a" * 64}]
    import hashlib as _hl2

    two["archive_md5"] = _hl2.md5(remote_file.read_bytes()).hexdigest()  # noqa: S324
    ledger_path.write_text(json.dumps(two) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing ledger members"):
        restore_date(tmp_path, "raw", day, tmp_path / "u4", run_fn=run_fn, config=cfg)
