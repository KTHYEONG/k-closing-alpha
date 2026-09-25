"""호스트 git 체크아웃과 systemd 유닛을 GHA가 검증한 커밋으로 수렴시킨다.

CI(.github/workflows/deploy.yml)의 test job이 이미 pytest 전체스위트를
통과시킨 커밋만 build-and-push/deploy로 넘어오므로, 여기서 다시 테스트를
돌리지 않는다. 운영 시크릿(EnvironmentFile)을 물고 호스트에서 pytest를
재실행하는 방식은 시크릿이 새어들어 격리 결함에 취약했고(실측:
2026-09-16~21 COLLECTION_* 환경변수 유출로 배포가 며칠간 조용히 막힘),
가동 중인 트레이딩 파이프라인과 CPU를 다투는 낭비이기도 했다. 이 모듈은
정확한 커밋으로 체크아웃을 맞추고 deploy/systemd 를 수렴시키는 것만 한다.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

DEFAULT_REMOTE: str = "origin"
GIT_TIMEOUT_SEC: int = 60
UV_SYNC_TIMEOUT_SEC: int = 600
SYSTEMD_UNIT_GLOBS: tuple[str, ...] = ("kca-*.service", "kca-*.timer")
SYSTEMCTL_TIMEOUT_SEC: int = 60
OPTIONAL_MANUAL_TIMERS: frozenset[str] = frozenset(
    {"kca-auction-close.timer", "kca-auction-open.timer", "kca-altdata-capture.timer"}
)
# 브로커 시크릿·Gmail 앱 비밀번호가 담긴 .env 는 소유자만 읽을 수 있어야 한다
SECRET_FILE_MODE: int = 0o600


@dataclass(frozen=True)
class UnitInstallResult:
    """Outcome of converging installed systemd user units onto the repo copies.

    Attributes:
        changed: Unit file names copied because they were new or differed.
        removed: kca-* unit file names deleted because the repo no longer has them.
        enabled: Newly installed timer names enabled and started.
    """

    changed: tuple[str, ...]
    removed: tuple[str, ...]
    enabled: tuple[str, ...]


def _resolve_uv_bin() -> str:
    """Resolve the uv executable's absolute path for subprocess calls.

    `systemctl --user show-environment` and a GHA SSH session both carry a
    PATH that excludes ~/.local/bin, so a bare "uv" argv element fails with
    FileNotFoundError even though uv is installed. shutil.which is checked
    first so a CI/dev shell with uv already on PATH is unaffected; the fixed
    ~/.local/bin/uv path is the fallback every deployment context assumes.

    Returns:
        Absolute path to the uv executable.
    """
    return shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


def _git(args: list[str], cwd: str) -> str:
    """Run git and return trimmed stdout; raise CalledProcessError on a non-zero exit."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=GIT_TIMEOUT_SEC, check=True)  # noqa: S603, S607
    return result.stdout.strip()


def _uv_sync(repo_dir: str) -> None:
    """Refresh the venv from the freshly checked-out lockfile."""
    subprocess.run([_resolve_uv_bin(), "sync"], cwd=repo_dir, check=True, capture_output=True, text=True, timeout=UV_SYNC_TIMEOUT_SEC)  # noqa: S603, S607


def install_systemd_units(
    repo_dir: str,
    *,
    dest_dir: Path | None = None,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> UnitInstallResult:
    """Converge the installed kca-* systemd user units onto deploy/systemd in the repo.

    systemd reads its own copies under ~/.config/systemd/user, so a git checkout
    update alone never changes a schedule. Only differing or new files are
    copied, kca-* units deleted from the repo are disabled (timers) and removed,
    daemon-reload runs only when something changed, and only newly installed
    timers are enabled -- existing timers keep whatever enable state an
    operator set. Services are never enabled directly: they start from their
    timer or from OnSuccess=/OnFailure= chaining.

    Args:
        repo_dir: Repository checkout containing deploy/systemd.
        dest_dir: Installed unit directory; None selects ~/.config/systemd/user.
        run_fn: subprocess.run-compatible runner (tests substitute a fake).

    Returns:
        UnitInstallResult listing changed, removed and enabled unit names.

    Raises:
        subprocess.CalledProcessError: A systemctl call failed; propagated so
            the deploy step fails loudly rather than leaving units stale.
    """
    src_dir = Path(repo_dir) / "deploy" / "systemd"
    dest = dest_dir if dest_dir is not None else Path.home() / ".config" / "systemd" / "user"
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {path.name: path for pattern in SYSTEMD_UNIT_GLOBS for path in src_dir.glob(pattern)}
    installed = {path.name for pattern in SYSTEMD_UNIT_GLOBS for path in dest.glob(pattern)}
    changed = sorted(
        name for name, path in wanted.items()
        if not (dest / name).exists() or (dest / name).read_bytes() != path.read_bytes()
    )
    removed = sorted(installed - set(wanted))
    # Exclude the three declared optional timer names from newly installed auto-enable candidates.
    enabled = sorted(name for name in changed if name.endswith(".timer") and name not in installed and name not in OPTIONAL_MANUAL_TIMERS)

    def _systemctl(*args: str) -> None:
        run_fn(["systemctl", "--user", *args], capture_output=True, text=True, timeout=SYSTEMCTL_TIMEOUT_SEC, check=True)

    for name in removed:
        if name.endswith(".timer"):
            _systemctl("disable", "--now", name)
        (dest / name).unlink()
    for name in changed:
        shutil.copyfile(wanted[name], dest / name)
    if changed or removed:
        _systemctl("daemon-reload")
    for name in enabled:
        _systemctl("enable", "--now", name)
    return UnitInstallResult(changed=tuple(changed), removed=tuple(removed), enabled=tuple(enabled))


def ensure_secret_permissions(repo_dir: str) -> bool:
    """Tighten the repo .env to owner-only (0600) when it is broader.

    Args:
        repo_dir: Repository checkout that may contain .env.

    Returns:
        True when permissions were tightened, False when already 0600 or absent.
    """
    path = Path(repo_dir) / ".env"
    if not path.exists() or (path.stat().st_mode & 0o777) == SECRET_FILE_MODE:
        return False
    path.chmod(SECRET_FILE_MODE)
    return True


def sync_units(
    repo_dir: str,
    *,
    sha: str,
    remote: str = DEFAULT_REMOTE,
    git_fn: Callable[[list[str], str], str] = _git,
    uv_sync_fn: Callable[[str], None] = _uv_sync,
    install_units_fn: Callable[[str], UnitInstallResult] = install_systemd_units,
    secure_fn: Callable[[str], bool] = ensure_secret_permissions,
) -> UnitInstallResult:
    """Check out an exact, CI-validated commit and converge systemd units onto it.

    Args:
        repo_dir: Host git checkout to converge (a clone of the production repo).
        sha: Exact commit the CI pipeline already tested and built the running image from.
        remote: Remote name to fetch the commit from.
        git_fn: Injected git runner (tests substitute a fake).
        uv_sync_fn: Injected dependency refresh.
        install_units_fn: Injected systemd unit convergence.
        secure_fn: Injected .env permission tightening.

    Returns:
        UnitInstallResult listing changed, removed and enabled unit names.

    Raises:
        subprocess.CalledProcessError: git/uv/systemctl failure; propagated so
            the deploy step fails loudly rather than leaving the host stale.
    """
    git_fn(["fetch", remote, sha, "--quiet"], repo_dir)
    git_fn(["reset", "--hard", sha], repo_dir)
    uv_sync_fn(repo_dir)
    secure_fn(repo_dir)
    return install_units_fn(repo_dir)


def main(argv: list[str] | None = None) -> None:
    """CI 진입점: settings.BASE_DIR 체크아웃을 --sha 로 수렴시킨다."""
    from src import settings

    parser = argparse.ArgumentParser(description="Converge the host checkout and systemd units onto a CI-tested commit")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    args = parser.parse_args(argv)
    units = sync_units(str(settings.BASE_DIR), sha=args.sha, remote=args.remote)
    logger.info(
        "[SYS] code_sync sha=%s units_changed=%s units_removed=%s timers_enabled=%s",
        args.sha[:8], list(units.changed), list(units.removed), list(units.enabled),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
