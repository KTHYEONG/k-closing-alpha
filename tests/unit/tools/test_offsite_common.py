"""Offsite shared-constant invariant guards."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


def test_rclone_resolution_prefers_path(monkeypatch) -> None:
    from src.tools import offsite_common

    monkeypatch.setattr(offsite_common.shutil, "which", lambda name: "/usr/bin/rclone")

    assert offsite_common.resolve_rclone_bin() == "/usr/bin/rclone"


def test_rclone_resolution_falls_back_to_user_bin(monkeypatch) -> None:
    from src.tools import offsite_common

    monkeypatch.setattr(offsite_common.shutil, "which", lambda name: None)

    assert offsite_common.resolve_rclone_bin() == str(Path.home() / ".local" / "bin" / "rclone")


def test_derived_remote_roots_unchanged() -> None:
    from src.tools import backup_prune, core_snapshot, offsite_backup
    from src.tools.capture_offsite import OffsiteConfig
    from src.tools.offsite_common import OFFSITE_REMOTE_BASE

    assert OFFSITE_REMOTE_BASE == "gdrive:quant-lake/live/k-closing-alpha"
    assert backup_prune.BACKUP_REMOTE_ROOT == OFFSITE_REMOTE_BASE + "/_deleted"
    assert core_snapshot.CORE_SNAPSHOT_REMOTE_ROOT == OFFSITE_REMOTE_BASE + "/snapshots"
    assert OffsiteConfig().remote_root == OFFSITE_REMOTE_BASE + "/capture_sealed"
    assert offsite_backup.BACKUP_REMOTE_BASE == OFFSITE_REMOTE_BASE


def test_dated_dir_regex_strictness() -> None:
    from src.tools.offsite_common import DATED_DIR_RE

    assert DATED_DIR_RE.match("2026-09-25")
    assert not DATED_DIR_RE.match("2026-9-25")
    assert not DATED_DIR_RE.match("2026-09-25x")
    assert not DATED_DIR_RE.match("latest")


def test_streaming_digest_equals_hashlib(tmp_path: Path) -> None:
    from src.tools.offsite_common import sha256_file

    payload = bytes((index * 7) % 251 for index in range(2 * 1024 * 1024 + 13))
    target = tmp_path / "segment.bin"
    target.write_bytes(payload)

    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_no_backup_import_cycle() -> None:
    for name in ("src/tools/capture_offsite.py", "src/tools/offsite_common.py"):
        tree = ast.parse(Path(name).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "src.tools.backup_prune" not in imported, name
    tree = ast.parse(Path("src/tools/offsite_common.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
            raise AssertionError(f"offsite_common imports {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("src."), alias.name


def test_consumer_aliases_share_one_resolver() -> None:
    from src.tools import backup_prune, capture_offsite, core_snapshot, offsite_backup
    from src.tools.offsite_common import resolve_rclone_bin

    assert backup_prune._resolve_rclone_bin is resolve_rclone_bin
    assert capture_offsite._resolve_rclone_bin is resolve_rclone_bin
    assert core_snapshot._resolve_rclone_bin is resolve_rclone_bin
    assert offsite_backup._resolve_rclone_bin is resolve_rclone_bin
