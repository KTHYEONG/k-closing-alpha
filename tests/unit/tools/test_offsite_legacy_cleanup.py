from __future__ import annotations

import fcntl
import json
import subprocess
from pathlib import Path

from src.tools.capture_offsite import LedgerEntry, SegmentMember


def _member(path: str, size: int) -> SegmentMember:
    return SegmentMember(path=path, size=size, sha256="ab" * 32)


def _entry(
    tier: str,
    trading_date: str,
    seg: str = "seg-abc",
    members: tuple[SegmentMember, ...] | None = None,
    md5: str = "d41d8cd98f00b204e9800998ecf8427e",
) -> LedgerEntry:
    own = members if members is not None else (_member(f"{tier}/{trading_date}/a.json.gz", 10),)
    return LedgerEntry(
        tier=tier,
        trading_date=trading_date,
        segment_name=seg,
        remote_path=f"gdrive:test/{tier}/{trading_date}/{seg}.tar.zst",
        members=own,
        archive_bytes=100,
        archive_md5=md5,
        committed_at="2026-09-23T00:00:00+00:00",
    )


def _ledgers(*entries: LedgerEntry) -> dict[tuple[str, str], list[LedgerEntry]]:
    grouped: dict[tuple[str, str], list[LedgerEntry]] = {}
    for item in entries:
        grouped.setdefault((item.tier, item.trading_date), []).append(item)
    return grouped


def test_build_legacy_plan_verified_sealed_member_is_deletable() -> None:
    from src.tools.offsite_legacy_cleanup import build_legacy_plan

    # Given a loose object matching a verified segment member with equal size
    member = _member("raw/2026-09-21/a/b.json.gz", 10)
    item = _entry("raw", "2026-09-21", members=(member,))

    # When classifying / Then sealed_loose_member candidate with segment evidence
    plan = build_legacy_plan([("raw/2026-09-21/a/b.json.gz", 10)], _ledgers(item), frozenset({item.remote_path}))
    assert len(plan.candidates) == 1
    assert plan.candidates[0].rule == "sealed_loose_member"
    assert item.remote_path in plan.candidates[0].evidence
    assert item.archive_md5 in plan.candidates[0].evidence
    assert plan.kept == ()


def test_build_legacy_plan_size_mismatch_keeps_loose_copy() -> None:
    from src.tools.offsite_legacy_cleanup import build_legacy_plan

    # Given member size differs from the loose object
    member = _member("raw/2026-09-21/a/b.json.gz", 11)
    item = _entry("raw", "2026-09-21", members=(member,))

    # When classifying / Then kept as size_mismatch
    plan = build_legacy_plan([("raw/2026-09-21/a/b.json.gz", 10)], _ledgers(item), frozenset({item.remote_path}))
    assert plan.candidates == ()
    assert plan.kept == (("raw/2026-09-21/a/b.json.gz", "size_mismatch"),)


def test_build_legacy_plan_unverified_segment_keeps_members() -> None:
    from src.tools.offsite_legacy_cleanup import build_legacy_plan

    # Given the segment remote path is absent from verified_segments
    member = _member("raw/2026-09-21/a/b.json.gz", 10)
    item = _entry("raw", "2026-09-21", members=(member,))

    # When classifying / Then kept as segment_unverified
    plan = build_legacy_plan([("raw/2026-09-21/a/b.json.gz", 10)], _ledgers(item), frozenset())
    assert plan.candidates == ()
    assert plan.kept == (("raw/2026-09-21/a/b.json.gz", "segment_unverified"),)


def test_build_legacy_plan_unsealed_date_is_kept() -> None:
    from src.tools.offsite_legacy_cleanup import build_legacy_plan

    # Given no ledger for that (tier, date), plus an empty ledger variant
    plan = build_legacy_plan([("raw/2026-09-21/a.json.gz", 10)], {}, frozenset())
    assert plan.candidates == ()
    assert plan.kept == (("raw/2026-09-21/a.json.gz", "unsealed"),)

    other = _entry("raw", "2026-09-20")
    plan2 = build_legacy_plan(
        [("raw/2026-09-21/a.json.gz", 10), ("raw/2026-09-20/b.json.gz", 5)],
        {(other.tier, other.trading_date): []},
        frozenset(),
    )
    assert plan2.candidates == ()
    assert ("raw/2026-09-21/a.json.gz", "unsealed") in plan2.kept

    # And a loose path with no committed member match on a sealed date
    sealed = _entry("raw", "2026-09-21", members=(_member("raw/2026-09-21/other.json.gz", 10),))
    plan3 = build_legacy_plan(
        [("raw/2026-09-21/a.json.gz", 10)], _ledgers(sealed), frozenset({sealed.remote_path})
    )
    assert plan3.candidates == ()
    assert plan3.kept == (("raw/2026-09-21/a.json.gz", "unsealed"),)


def test_build_legacy_plan_rollback_snapshots_always_removed() -> None:
    from src.tools.offsite_legacy_cleanup import build_legacy_plan

    # Given rollback snapshot copies regardless of ledgers
    plan = build_legacy_plan([("backups/intraday/regular/2026-09-22/x.parquet", 7)], {}, frozenset())
    assert len(plan.candidates) == 1
    assert plan.candidates[0].rule == "rollback_snapshot_copy"
    assert plan.kept == ()


def test_build_legacy_plan_out_of_scope_subtrees_immune() -> None:
    from src.tools.offsite_legacy_cleanup import build_legacy_plan

    # Given non-tier subtrees and short tier paths
    loose = [
        ("manifests/2026-09-21/run/manifest-complete.json", 3),
        ("decision/2026-09-21/run/input.parquet", 4),
        ("staging/offsite/x.tmp", 5),
        ("raw/2026-09-21", 6),
    ]
    plan = build_legacy_plan(loose, {}, frozenset())
    assert plan.candidates == ()
    assert plan.kept == tuple((rel, "out_of_scope") for rel, _ in loose)


def test_verify_segments_fail_closed() -> None:
    from src.tools.offsite_legacy_cleanup import verify_segments

    # Given md5sum exits 1, returns another hash, raises, or prints nothing
    good = _entry("raw", "2026-09-21", seg="seg-good", md5="aa" * 16)
    bad_exit = _entry("raw", "2026-09-21", seg="seg-bad", md5="bb" * 16)
    bad_hash = _entry("normalized", "2026-09-21", seg="seg-hash", md5="cc" * 16)
    boom = _entry("normalized", "2026-09-21", seg="seg-boom", md5="dd" * 16)
    empty = _entry("raw", "2026-09-22", seg="seg-empty", md5="ee" * 16)
    ledgers = _ledgers(good, bad_exit, bad_hash, boom, empty)

    def run_fn(cmd, **kwargs):
        remote = cmd[2]
        if remote == good.remote_path:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{good.archive_md5}  {remote}\n", stderr="")
        if remote == bad_hash.remote_path:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{'ff' * 16}  {remote}\n", stderr="")
        if remote == boom.remote_path:
            raise OSError("drive down")
        if remote == empty.remote_path:
            return subprocess.CompletedProcess(cmd, 0, stdout="\n", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")

    # When verifying / Then only the matching segment is verified
    assert verify_segments(ledgers, run_fn=run_fn) == frozenset({good.remote_path})


def _write_ledger(root: Path, tier: str, trading_date: str, entries: list[LedgerEntry]) -> None:
    path = root / "offsite" / "ledger" / tier / f"{trading_date}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "tier": item.tier,
                "trading_date": item.trading_date,
                "segment_name": item.segment_name,
                "remote_path": item.remote_path,
                "members": [{"path": m.path, "size": m.size, "sha256": m.sha256} for m in item.members],
                "archive_bytes": item.archive_bytes,
                "archive_md5": item.archive_md5,
                "committed_at": item.committed_at,
            },
            sort_keys=True,
        )
        for item in entries
    ]
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def _patch_common(monkeypatch, tmp_path: Path, lock_wait: float = 30.0):
    import src.tools.offsite_legacy_cleanup as cleanup

    monkeypatch.setattr(cleanup, "_capture_root", lambda: tmp_path / "capture")
    monkeypatch.setattr(cleanup, "_resolve_rclone_bin", lambda: "rclone")
    monkeypatch.setattr(cleanup, "_drive_lock_path", lambda: tmp_path / "quant-gdrive.lock")
    monkeypatch.setattr(cleanup, "DRIVE_LOCK_WAIT_SEC", lock_wait)


def _fake_run(calls: list[list[str]], listing: list[dict[str, object]], md5: dict[str, str], *, fail_ls: bool = False):
    def run_fn(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "lsjson":
            if fail_ls:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ls failed")
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(listing), stderr="")
        if cmd[1] == "md5sum":
            digest = md5.get(cmd[2])
            if digest is None:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{digest}  {cmd[2]}\n", stderr="")
        if cmd[1] == "delete":
            # 일괄 삭제 목록은 호출 직후 임시 디렉터리와 함께 사라지므로 내용을 기록해 둔다
            files_from = Path(cmd[cmd.index("--files-from") + 1])
            calls.append(["files-from-content", *files_from.read_text(encoding="utf-8").splitlines()])
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "rmdirs":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected rclone call: {cmd}")

    return run_fn


def test_main_dry_run_is_non_mutating(tmp_path: Path, monkeypatch) -> None:
    import subprocess as _subprocess

    import src.tools.offsite_legacy_cleanup as cleanup

    # Given one sealed member plus one rollback copy on the remote
    _patch_common(monkeypatch, tmp_path)
    member = _member("raw/2026-09-21/a.json.gz", 10)
    item = _entry("raw", "2026-09-21", members=(member,))
    _write_ledger(tmp_path / "capture", "raw", "2026-09-21", [item])
    listing = [
        {"Path": "raw/2026-09-21/a.json.gz", "Size": 10},
        {"Path": "backups/x.parquet", "Size": 5},
    ]
    calls: list[list[str]] = []
    monkeypatch.setattr(
        _subprocess, "run", _fake_run(calls, listing, {item.remote_path: item.archive_md5})
    )

    # When dry-running / Then exit 0 with only listing and md5sum calls
    assert cleanup.main([]) == 0
    ops = [cmd[1] for cmd in calls]
    assert "lsjson" in ops and "md5sum" in ops
    assert "delete" not in ops and "rmdirs" not in ops


def test_main_restore_drill_failure_aborts_apply(tmp_path: Path, monkeypatch) -> None:
    import subprocess as _subprocess

    import src.tools.offsite_legacy_cleanup as cleanup

    # Given apply where the restore drill raises
    _patch_common(monkeypatch, tmp_path)
    member = _member("raw/2026-09-21/a.json.gz", 10)
    item = _entry("raw", "2026-09-21", members=(member,))
    _write_ledger(tmp_path / "capture", "raw", "2026-09-21", [item])
    listing = [{"Path": "raw/2026-09-21/a.json.gz", "Size": 10}]
    calls: list[list[str]] = []
    monkeypatch.setattr(
        _subprocess, "run", _fake_run(calls, listing, {item.remote_path: item.archive_md5})
    )

    def _boom(capture_root, tier, trading_date, dest_root, **kwargs):
        raise ValueError("archive MD5 mismatch")

    monkeypatch.setattr(cleanup, "restore_date", _boom)

    # When applying / Then exit 1 with no deletefile call
    assert cleanup.main(["--apply"]) == 1
    assert [cmd[1] for cmd in calls] == ["lsjson", "md5sum"]


def test_main_apply_holds_shared_drive_lock(tmp_path: Path, monkeypatch) -> None:
    import subprocess as _subprocess

    import src.tools.offsite_legacy_cleanup as cleanup

    # Given the lock is held elsewhere and only a short wait is allowed
    _patch_common(monkeypatch, tmp_path, lock_wait=0.2)
    lock_path = tmp_path / "quant-gdrive.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = open(lock_path, "w")  # noqa: SIM115
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        calls: list[list[str]] = []
        monkeypatch.setattr(_subprocess, "run", _fake_run(calls, [], {}))

        # When applying / Then non-zero exit with no rclone mutation
        assert cleanup.main(["--apply"]) != 0
        assert calls == []
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_main_apply_deletes_candidates_and_cleans_dirs(tmp_path: Path, monkeypatch) -> None:
    import subprocess as _subprocess

    import src.tools.offsite_legacy_cleanup as cleanup

    # Given apply with a passing drill over two tiers
    _patch_common(monkeypatch, tmp_path)
    raw_member = _member("raw/2026-09-22/a.json.gz", 10)
    norm_member = _member("normalized/2026-09-21/b.parquet", 20)
    raw_item = _entry("raw", "2026-09-22", seg="seg-raw", members=(raw_member,))
    norm_item = _entry("normalized", "2026-09-21", seg="seg-norm", members=(norm_member,))
    _write_ledger(tmp_path / "capture", "raw", "2026-09-22", [raw_item])
    _write_ledger(tmp_path / "capture", "normalized", "2026-09-21", [norm_item])
    listing = [
        {"Path": "raw/2026-09-22/a.json.gz", "Size": 10},
        {"Path": "normalized/2026-09-21/b.parquet", "Size": 20},
        {"Path": "backups/x.parquet", "Size": 5},
    ]
    calls: list[list[str]] = []
    monkeypatch.setattr(
        _subprocess,
        "run",
        _fake_run(calls, listing, {raw_item.remote_path: raw_item.archive_md5, norm_item.remote_path: norm_item.archive_md5}),
    )
    drilled: list[tuple[str, str]] = []

    def _ok(capture_root, tier, trading_date, dest_root, **kwargs):
        drilled.append((tier, trading_date))
        return [f"{tier}/{trading_date}/a"]

    monkeypatch.setattr(cleanup, "restore_date", _ok)

    # When applying / Then drill covers the latest date per tier and every candidate is deleted
    assert cleanup.main(["--apply"]) == 0
    assert sorted(drilled) == [("normalized", "2026-09-21"), ("raw", "2026-09-22")]
    batch = [cmd for cmd in calls if cmd[1] == "delete"]
    assert len(batch) == 1 and batch[0][2] == cleanup.LOOSE_REMOTE_ROOT
    listed = next(cmd[1:] for cmd in calls if cmd[0] == "files-from-content")
    assert sorted(listed) == ["backups/x.parquet", "normalized/2026-09-21/b.parquet", "raw/2026-09-22/a.json.gz"]
    assert any(cmd[1] == "rmdirs" and "--leave-root" in cmd for cmd in calls)


def test_main_listing_failure_exits_nonzero_without_deletion(tmp_path: Path, monkeypatch) -> None:
    import subprocess as _subprocess

    import src.tools.offsite_legacy_cleanup as cleanup

    # Given the remote listing fails (dry-run and apply variants)
    _patch_common(monkeypatch, tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(_subprocess, "run", _fake_run(calls, [], {}, fail_ls=True))

    # When running / Then non-zero exit with no deletion
    assert cleanup.main([]) == 1
    assert cleanup.main(["--apply"]) == 1
    assert "delete" not in [cmd[1] for cmd in calls]


def test_build_legacy_plan_verified_size_mismatch_beats_unverified_copy() -> None:
    from src.tools.offsite_legacy_cleanup import build_legacy_plan

    # Given the same path claimed by a verified segment with a wrong size and an unverified one
    rel = "raw/2026-09-21/a.json.gz"
    verified_wrong = _entry("raw", "2026-09-21", seg="seg-v", members=(_member(rel, 11),))
    unverified = _entry("raw", "2026-09-21", seg="seg-u", members=(_member(rel, 10),))
    ledgers = {(verified_wrong.tier, verified_wrong.trading_date): [verified_wrong, unverified]}

    # When classifying / Then kept as size_mismatch, never deleted
    plan = build_legacy_plan([(rel, 10)], ledgers, frozenset({verified_wrong.remote_path}))
    assert plan.candidates == ()
    assert plan.kept == ((rel, "size_mismatch"),)


def _payload_run(payload: str) -> tuple[list[list[str]], object]:
    calls: list[list[str]] = []

    def run_fn(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")

    return calls, run_fn


def test_drive_lock_path_uses_runtime_dir(tmp_path: Path, monkeypatch) -> None:
    import os as _os

    import src.tools.offsite_legacy_cleanup as cleanup

    # Given XDG_RUNTIME_DIR set / Then the shared lock lives under it
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert cleanup._drive_lock_path() == tmp_path / "quant-gdrive.lock"

    # And without it / Then the /run/user/<uid> fallback applies
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert cleanup._drive_lock_path() == Path(f"/run/user/{_os.getuid()}") / "quant-gdrive.lock"


def test_main_malformed_listing_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    import subprocess as _subprocess

    import src.tools.offsite_legacy_cleanup as cleanup

    # Given unreadable listing payloads (bad JSON, non-list, malformed items)
    _patch_common(monkeypatch, tmp_path)
    for payload in ("not json", '{"a": 1}', "[1, 2]", json.dumps([{"Path": "a"}])):
        calls, run_fn = _payload_run(payload)
        monkeypatch.setattr(_subprocess, "run", run_fn)

        # When dry-running / Then non-zero exit with no deletion
        assert cleanup.main([]) == 1
        assert "delete" not in [cmd[1] for cmd in calls]
