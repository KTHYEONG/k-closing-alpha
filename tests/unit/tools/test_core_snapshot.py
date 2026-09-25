from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path


def _write_parquet(path: Path, rows: int, max_date: str) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"date": [max_date] * rows, "value": range(rows)})
    frame.to_parquet(path, index=False)


def _stat(relpath: str, rows: int, max_date: str, sha: str = "a" * 64) -> object:
    from src.tools.core_snapshot import CorePanelStat

    return CorePanelStat(relpath=relpath, sha256=sha, bytes=10, rows=rows, max_date=max_date)


def _manifest_json(entries: list) -> str:
    import dataclasses

    return json.dumps([dataclasses.asdict(e) for e in entries])


def test_core_snapshot_blocks_shrunk_panel() -> None:
    import tempfile

    from src.tools.core_snapshot import collect_core_stats, validate_core_panels

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_parquet(root / "data/history/price_history.parquet", 900, "2026-09-23")
        current = [s for s in collect_core_stats(root) if s.relpath == "data/history/price_history.parquet"]
        assert len(current) == 1
        previous = [_stat("data/history/price_history.parquet", 1000, "2026-09-23")]  # type: ignore[list-item]
        issues = validate_core_panels(current, previous)  # type: ignore[arg-type]
        assert any("rows_shrank" in issue and "price_history" in issue for issue in issues)


def test_core_snapshot_shrunk_blocks_upload_without_copy(tmp_path: Path, monkeypatch) -> None:
    from src.tools.core_snapshot import run_core_snapshot

    monkeypatch.setattr("src.tools.core_snapshot._resolve_rclone_bin", lambda: "rclone")
    _write_parquet(tmp_path / "data/history/price_history.parquet", 900, "2026-09-23")
    prev = _stat("data/history/price_history.parquet", 1000, "2026-09-23")
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="2026-09-01/\n", stderr="")
        if cmd[1] == "cat":
            return subprocess.CompletedProcess(cmd, 0, stdout=_manifest_json([prev]), stderr="")
        raise AssertionError(f"unexpected call: {cmd}")

    import pytest

    with pytest.raises(RuntimeError, match="rows_shrank"):
        run_core_snapshot(tmp_path, today=date(2026, 9, 28), run_fn=_run)
    assert not any(c[1] == "copy" for c in calls)


def test_core_snapshot_blocks_regressed_max_date(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.core_snapshot import run_core_snapshot

    monkeypatch.setattr("src.tools.core_snapshot._resolve_rclone_bin", lambda: "rclone")
    _write_parquet(tmp_path / "data/history/price_history.parquet", 100, "2026-09-18")
    prev = _stat("data/history/price_history.parquet", 100, "2026-09-23")

    def _run(cmd, **kwargs):
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="2026-09-01/\n", stderr="")
        if cmd[1] == "cat":
            return subprocess.CompletedProcess(cmd, 0, stdout=_manifest_json([prev]), stderr="")
        raise AssertionError(f"unexpected call: {cmd}")

    with pytest.raises(RuntimeError, match="max_date_regressed"):
        run_core_snapshot(tmp_path, today=date(2026, 9, 28), run_fn=_run)


def test_core_snapshot_blocks_unreadable_parquet(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.core_snapshot import collect_core_stats, run_core_snapshot, validate_core_panels

    monkeypatch.setattr("src.tools.core_snapshot._resolve_rclone_bin", lambda: "rclone")
    target = tmp_path / "data/history/price_history.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"truncated-parquet-bytes")
    current = collect_core_stats(tmp_path)
    matched = [s for s in current if s.relpath == "data/history/price_history.parquet"]
    assert len(matched) == 1
    assert validate_core_panels(matched, []) == ["core_panel:data/history/price_history.parquet:unreadable"]

    def _run(cmd, **kwargs):
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected call: {cmd}")

    with pytest.raises(RuntimeError, match="unreadable"):
        run_core_snapshot(tmp_path, today=date(2026, 9, 28), run_fn=_run)


def test_core_snapshot_uploads_immutably_with_manifest(tmp_path: Path, monkeypatch) -> None:
    from src.tools.core_snapshot import CORE_SNAPSHOT_REMOTE_ROOT, run_core_snapshot

    monkeypatch.setattr("src.tools.core_snapshot._resolve_rclone_bin", lambda: "rclone")
    _write_parquet(tmp_path / "data/history/price_history.parquet", 10, "2026-09-23")
    captured: dict[str, object] = {}
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "copy":
            assert "--immutable" in cmd
            assert "snapshots/2026-09-28/" in cmd[3]
            staging = Path(cmd[2])
            manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
            captured["manifest"] = manifest
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "purge":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected call: {cmd}")

    pruned = run_core_snapshot(tmp_path, today=date(2026, 9, 28), run_fn=_run)
    assert pruned == []
    copy_calls = [c for c in calls if c[1] == "copy"]
    assert len(copy_calls) == 1
    assert copy_calls[0][3] == f"{CORE_SNAPSHOT_REMOTE_ROOT}/2026-09-28/"
    manifest = captured["manifest"]
    assert isinstance(manifest, list)
    by_rel = {item["relpath"]: item for item in manifest}
    assert "data/history/price_history.parquet" in by_rel
    assert len(by_rel["data/history/price_history.parquet"]["sha256"]) == 64


def test_core_snapshot_retention_keeps_weekly_and_monthly_anchors() -> None:
    from datetime import timedelta

    from src.tools.core_snapshot import select_snapshots_to_prune

    start = date(2026, 5, 4)
    names = [(start + timedelta(weeks=i)).isoformat() for i in range(20)]
    pruned = select_snapshots_to_prune(names)
    newest_8 = set(names[-8:])
    months = sorted({name[:7] for name in names})
    firsts = {min(n for n in names if n.startswith(month)) for month in months}
    keep = newest_8 | firsts
    assert all(name not in pruned for name in keep)
    assert pruned
    assert set(pruned) == set(names) - keep


def test_core_snapshot_never_prunes_when_too_few() -> None:
    from src.tools.core_snapshot import select_snapshots_to_prune

    names = ["2026-09-01", "2026-09-08", "2026-09-15", "2026-09-22", "2026-09-28"]
    assert select_snapshots_to_prune(names) == []


def test_core_snapshot_collects_paper_and_bundle_panels(tmp_path: Path) -> None:
    from src.tools.core_snapshot import collect_core_stats, core_panel_paths

    _write_parquet(tmp_path / "data/paper/fills.parquet", 3, "2026-09-20")
    bundle_file = tmp_path / "artifacts/models/topk_ranker/bundle.json"
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    bundle_file.write_text("{}", encoding="utf-8")
    paths = core_panel_paths(tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in paths]
    assert "data/paper/fills.parquet" in rels
    assert "artifacts/models/topk_ranker/bundle.json" in rels
    stats = {s.relpath: s for s in collect_core_stats(tmp_path)}
    assert stats["data/paper/fills.parquet"].rows == 3
    assert stats["artifacts/models/topk_ranker/bundle.json"].rows is None


def test_core_snapshot_reads_decision_date_and_empty_frame(tmp_path: Path) -> None:
    import pandas as pd

    from src.tools.core_snapshot import collect_core_stats

    frame = pd.DataFrame({"decision_date": ["2026-09-18", "2026-09-20"], "v": [1, 2]})
    target = tmp_path / "data/parquet/topk_decisions.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    stats = {s.relpath: s for s in collect_core_stats(tmp_path)}
    assert stats["data/parquet/topk_decisions.parquet"].max_date == "2026-09-20"
    empty = tmp_path / "data/parquet/rank_pool_predictions.parquet"
    pd.DataFrame({"date": pd.to_datetime([])}).to_parquet(empty, index=False)
    stats2 = {s.relpath: s for s in collect_core_stats(tmp_path)}
    assert stats2["data/parquet/rank_pool_predictions.parquet"].max_date == ""


def test_core_snapshot_missing_file_reports_missing() -> None:
    from src.tools.core_snapshot import CorePanelStat, validate_core_panels

    prev = [CorePanelStat(relpath="data/history/archive.parquet", sha256="a" * 64, bytes=1, rows=5, max_date="2026-09-20")]
    assert validate_core_panels([], prev) == ["core_panel:data/history/archive.parquet:missing"]


def test_core_snapshot_resolve_and_listing_branches(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    import src.tools.core_snapshot as module

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/rclone")
    assert module._resolve_rclone_bin() == "/usr/bin/rclone"
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    assert module._resolve_rclone_bin().endswith("rclone")

    def _missing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="directory not found")

    assert module._list_remote_snapshot_dirs("rclone", _missing) == []

    def _boom(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    import pytest

    with pytest.raises(subprocess.CalledProcessError):
        module._list_remote_snapshot_dirs("rclone", _boom)

    def _cat_bad(cmd, **kwargs):
        if cmd[1] == "cat":
            return subprocess.CompletedProcess(cmd, 0, stdout="not-json", stderr="")
        raise AssertionError

    assert module._read_remote_manifest("rclone", "2026-09-01", _cat_bad) == []

    def _cat_dict(cmd, **kwargs):
        if cmd[1] == "cat":
            return subprocess.CompletedProcess(cmd, 0, stdout='{"a": 1}', stderr="")
        raise AssertionError

    assert module._read_remote_manifest("rclone", "2026-09-01", _cat_dict) == []

    def _cat_mixed(cmd, **kwargs):
        if cmd[1] == "cat":
            import json as _json

            payload = _json.dumps([{"relpath": "a", "sha256": "x", "bytes": 1, "rows": 1}, "bad", {"relpath": "b"}])
            return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")
        raise AssertionError

    got = module._read_remote_manifest("rclone", "2026-09-01", _cat_mixed)
    assert [e.relpath for e in got] == ["a"]

    def _cat_missing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")

    assert module._read_remote_manifest("rclone", "2026-09-01", _cat_missing) == []


def test_core_snapshot_upload_and_prune_failures(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    from datetime import date

    import pytest

    import src.tools.core_snapshot as module

    monkeypatch.setattr(module, "_resolve_rclone_bin", lambda: "rclone")
    _write_parquet(tmp_path / "data/history/price_history.parquet", 2, "2026-09-23")

    def _fail_upload(cmd, **kwargs):
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "copy":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="drive down")
        raise AssertionError

    with pytest.raises(RuntimeError, match="upload failed"):
        module.run_core_snapshot(tmp_path, today=date(2026, 9, 28), run_fn=_fail_upload)

    names = ["2026-05-04", "2026-05-11", "2026-05-18", "2026-05-25", "2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22", "2026-06-29", "2026-09-28"]

    def _fail_prune(cmd, **kwargs):
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(n + "/" for n in names[:-1]) + "\n", stderr="")
        if cmd[1] == "cat":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")
        if cmd[1] == "copy":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "purge":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        raise AssertionError

    with pytest.raises(RuntimeError, match="prune failed"):
        module.run_core_snapshot(tmp_path, today=date(2026, 9, 28), run_fn=_fail_prune)


def test_core_snapshot_prunes_old_snapshots_after_upload(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    from datetime import date, timedelta

    import src.tools.core_snapshot as module

    monkeypatch.setattr(module, "_resolve_rclone_bin", lambda: "rclone")
    _write_parquet(tmp_path / "data/history/price_history.parquet", 2, "2026-09-23")
    start = date(2026, 5, 4)
    old = [(start + timedelta(weeks=i)).isoformat() for i in range(12)]
    purges: list[str] = []

    def _run(cmd, **kwargs):
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(n + "/" for n in old) + "\n", stderr="")
        if cmd[1] == "cat":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")
        if cmd[1] == "copy":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "purge":
            purges.append(cmd[2])
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    pruned = module.run_core_snapshot(tmp_path, today=date(2026, 9, 28), run_fn=_run)
    assert pruned
    assert all(p.startswith(module.CORE_SNAPSHOT_REMOTE_ROOT) for p in pruned)
    assert purges == sorted(purges)


def test_core_snapshot_skips_symlinked_members(tmp_path: Path) -> None:
    from src.tools.core_snapshot import core_panel_paths

    paper_dir = tmp_path / "data/paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    real = paper_dir / "real.parquet"
    _write_parquet(real, 1, "2026-09-20")
    (paper_dir / "link.parquet").symlink_to(real)
    bundle_dir = tmp_path / "artifacts/models/topk_ranker"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_real = bundle_dir / "bundle.json"
    bundle_real.write_text("{}", encoding="utf-8")
    (bundle_dir / "bundle_link.json").symlink_to(bundle_real)
    rels = [p.relative_to(tmp_path).as_posix() for p in core_panel_paths(tmp_path)]
    assert "data/paper/real.parquet" in rels
    assert "data/paper/link.parquet" not in rels
    assert "artifacts/models/topk_ranker/bundle_link.json" not in rels
