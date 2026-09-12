"""git 코드 동기화: 원격 main 이 fast-forward 가능할 때만 반영, pytest 전체스위트 게이트 통과 시에만 유지.

당겨받은 커밋이 테스트를 깨면 즉시 이전 커밋으로 롤백하고 얼러트를 보낸다.
브랜치가 갈라졌거나(비-fast-forward, 예: 강제푸시) 사람 개입이 필요한 경우는
자동 반영하지 않고 얼러트만 보낸다. 대상 systemd 서비스들은 매 실행마다
새 프로세스로 뜨는 oneshot 이므로, 이 모듈은 체크아웃 상태만 갱신하면
충분하고 재기동/데몬 재시작을 스스로 트리거할 필요가 없다.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_REMOTE: str = "origin"
DEFAULT_BRANCH: str = "main"
GIT_TIMEOUT_SEC: int = 60
FAST_FORWARD_CHECK_TIMEOUT_SEC: int = 30
TEST_GATE_TIMEOUT_SEC: int = 1800
UV_SYNC_TIMEOUT_SEC: int = 600
ALERT_DETAIL_TAIL_CHARS: int = 2000


@dataclass(frozen=True)
class SyncResult:
    """One code-sync attempt outcome.

    Attributes:
        updated: True only when the working tree now sits at to_sha.
        from_sha: HEAD before this sync attempt.
        to_sha: origin/<branch>'s HEAD at fetch time (FETCH_HEAD).
        reason: "up_to_date" | "fast_forwarded" | "not_fast_forward" | "test_gate_failed".
    """

    updated: bool
    from_sha: str
    to_sha: str
    reason: str


def _git(args: list[str], cwd: str) -> str:
    """Run git and return trimmed stdout; raise CalledProcessError on a non-zero exit."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=GIT_TIMEOUT_SEC, check=True)  # noqa: S603, S607
    return result.stdout.strip()


def _is_fast_forward(repo_dir: str, from_sha: str, to_sha: str) -> bool:
    """True when from_sha is an ancestor of to_sha (a clean fast-forward)."""
    result = subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", from_sha, to_sha],  # noqa: S607
        cwd=repo_dir, capture_output=True, text=True, timeout=FAST_FORWARD_CHECK_TIMEOUT_SEC,
    )
    return result.returncode == 0


def _run_test_gate(repo_dir: str) -> tuple[bool, str]:
    """Run the full pytest suite at the working tree's current HEAD.

    Returns:
        (passed, tail_of_combined_output) - output capped to the last
        ALERT_DETAIL_TAIL_CHARS characters so a failing gate's alert stays bounded.
    """
    result = subprocess.run(  # noqa: S607
        ["uv", "run", "pytest", "-q"], cwd=repo_dir, capture_output=True, text=True, timeout=TEST_GATE_TIMEOUT_SEC,  # noqa: S607
    )
    combined = result.stdout + result.stderr
    return result.returncode == 0, combined[-ALERT_DETAIL_TAIL_CHARS:]


def _uv_sync(repo_dir: str) -> None:
    """Refresh the venv from the freshly fast-forwarded lockfile."""
    subprocess.run(["uv", "sync"], cwd=repo_dir, check=True, capture_output=True, text=True, timeout=UV_SYNC_TIMEOUT_SEC)  # noqa: S607


def sync_repo(
    repo_dir: str,
    *,
    remote: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
    git_fn: Callable[[list[str], str], str] = _git,
    fast_forward_fn: Callable[[str, str, str], bool] = _is_fast_forward,
    test_gate_fn: Callable[[str], tuple[bool, str]] = _run_test_gate,
    uv_sync_fn: Callable[[str], None] = _uv_sync,
) -> SyncResult:
    """Fast-forward-only, test-gated git sync; roll back and alert on anything short of a clean pass.

    Args:
        repo_dir: Working tree to sync (a git checkout of the production repo).
        remote: Remote name to fetch from.
        branch: Branch name to track.
        git_fn: Injected git runner (tests substitute a fake).
        fast_forward_fn: Injected ancestor check.
        test_gate_fn: Injected full-suite test runner.
        uv_sync_fn: Injected dependency refresh.

    Returns:
        SyncResult describing what happened.

    Raises:
        subprocess.CalledProcessError: Propagated from git_fn/uv_sync_fn on
            failures outside the expected control flow (e.g. fetch network
            failure) -- never silently swallowed.
    """
    from_sha = git_fn(["rev-parse", "HEAD"], repo_dir)
    git_fn(["fetch", remote, branch, "--quiet"], repo_dir)
    to_sha = git_fn(["rev-parse", "FETCH_HEAD"], repo_dir)

    if from_sha == to_sha:
        return SyncResult(updated=False, from_sha=from_sha, to_sha=to_sha, reason="up_to_date")

    if not fast_forward_fn(repo_dir, from_sha, to_sha):
        from src.tools.alerts import dispatch_failure_alert

        dispatch_failure_alert(
            "kca-code-sync.service",
            detail=f"origin/{branch} is not a fast-forward of local HEAD ({from_sha[:8]} -> {to_sha[:8]}); manual merge required",
        )
        return SyncResult(updated=False, from_sha=from_sha, to_sha=to_sha, reason="not_fast_forward")

    git_fn(["reset", "--hard", to_sha], repo_dir)
    passed, tail = test_gate_fn(repo_dir)
    if not passed:
        git_fn(["reset", "--hard", from_sha], repo_dir)
        from src.tools.alerts import dispatch_failure_alert

        dispatch_failure_alert(
            "kca-code-sync.service",
            detail=f"test gate failed at {to_sha[:8]}, rolled back to {from_sha[:8]}: {tail}",
        )
        return SyncResult(updated=False, from_sha=from_sha, to_sha=to_sha, reason="test_gate_failed")

    uv_sync_fn(repo_dir)
    return SyncResult(updated=True, from_sha=from_sha, to_sha=to_sha, reason="fast_forwarded")


def main(argv: list[str] | None = None) -> None:
    """systemd 진입점: settings.BASE_DIR 를 동기화 대상 저장소로 사용한다."""
    from src import settings

    parser = argparse.ArgumentParser(description="Fast-forward-only, test-gated git code sync")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    args = parser.parse_args(argv)
    result = sync_repo(str(settings.BASE_DIR), remote=args.remote, branch=args.branch)
    logger.info(
        "[SYS] code_sync updated=%s from=%s to=%s reason=%s",
        result.updated, result.from_sha[:8], result.to_sha[:8], result.reason,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
