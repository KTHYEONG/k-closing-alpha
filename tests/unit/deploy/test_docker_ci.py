def test_dockerfile_installs_libgomp_and_ripgrep_and_syncs_frozen() -> None:
    from pathlib import Path

    text = Path("Dockerfile").read_text(encoding="utf-8")

    assert "libgomp1" in text
    assert "ripgrep" in text
    assert "uv sync --frozen" in text


def test_dockerignore_excludes_secrets_and_runtime_data() -> None:
    from pathlib import Path

    text = Path(".dockerignore").read_text(encoding="utf-8")

    for entry in (".env", ".git", "data/", "artifacts/"):
        assert entry in text, entry


def test_deploy_workflow_gates_build_on_tests_and_targets_arm64() -> None:
    from pathlib import Path

    text = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "uv run pytest -q" in text
    assert "needs: test" in text
    assert "needs: build-and-push" in text
    assert "platforms: linux/arm64" in text


def test_dockerfile_and_dockerignore_and_workflow_content() -> None:
    test_dockerfile_installs_libgomp_and_ripgrep_and_syncs_frozen()
    test_dockerignore_excludes_secrets_and_runtime_data()
    test_deploy_workflow_gates_build_on_tests_and_targets_arm64()



def test_image_bakes_code_commit_from_ci_build_arg() -> None:
    from pathlib import Path

    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")

    # Then: 컨테이너에는 .git이 없으므로 커밋은 빌드 인자로만 들어온다
    assert "ARG KCA_CODE_COMMIT=UNKNOWN" in dockerfile
    assert "ENV KCA_CODE_COMMIT=${KCA_CODE_COMMIT}" in dockerfile
    # 의존성 레이어 캐시를 깨지 않도록 마지막 uv sync 뒤에 선언
    assert dockerfile.rindex("uv sync --frozen --no-dev") < dockerfile.index("ARG KCA_CODE_COMMIT")
    assert "KCA_CODE_COMMIT=${{ github.sha }}" in workflow


def test_dockerfile_keeps_uv_cache_out_of_image() -> None:
    from pathlib import Path

    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.splitlines()[0].startswith("# syntax=docker/dockerfile:1")
    assert "UV_CACHE_DIR=/root/.cache/uv" in dockerfile
    assert dockerfile.count("--mount=type=cache,target=/root/.cache/uv,sharing=locked") == 2
    assert "UV_NO_SYNC=1" in dockerfile
    assert "libgomp1" in dockerfile
    assert "ripgrep" in dockerfile


def test_deploy_validates_commit_image_before_moving_latest() -> None:
    from pathlib import Path

    workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")

    pull = "docker pull ghcr.io/kthyeong/k-closing-alpha:sha-$COMMIT_SHA"
    run = "docker run --rm -v ~/k-closing-alpha/artifacts:/app/artifacts:ro ghcr.io/kthyeong/k-closing-alpha:sha-$COMMIT_SHA uv run python -m src.tools.deploy_preflight"
    tag = "docker tag ghcr.io/kthyeong/k-closing-alpha:sha-$COMMIT_SHA ghcr.io/kthyeong/k-closing-alpha:latest"
    reset = "git reset --hard $COMMIT_SHA"

    positions = [workflow.index(line) for line in (pull, run, tag, reset)]
    assert positions == sorted(positions)
    assert "docker pull ghcr.io/kthyeong/k-closing-alpha:latest" not in workflow


def test_deploy_waits_for_blackout_before_moving_latest() -> None:
    from pathlib import Path

    workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")

    preflight = "src.tools.deploy_preflight"
    wait = "src.tools.deploy_window --wait"
    tag = "docker tag ghcr.io/kthyeong/k-closing-alpha:sha-$COMMIT_SHA ghcr.io/kthyeong/k-closing-alpha:latest"
    reset = "git reset --hard $COMMIT_SHA"

    positions = [workflow.index(line) for line in (preflight, wait, tag, reset)]
    assert positions == sorted(positions)


def test_deploy_job_serialized_and_bounded() -> None:
    from pathlib import Path

    workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
    deploy_job = workflow.split("build-and-push", 1)[1]

    assert "concurrency:" in deploy_job
    assert "cancel-in-progress: false" in deploy_job
    assert "timeout-minutes:" in deploy_job


def test_deploy_skips_superseded_commit() -> None:
    from pathlib import Path

    workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
    tag = "docker tag ghcr.io/kthyeong/k-closing-alpha:sha-$COMMIT_SHA ghcr.io/kthyeong/k-closing-alpha:latest"

    assert "LATEST_MAIN_SHA" in workflow
    assert "git ls-remote" in workflow
    assert workflow.index("LATEST_MAIN_SHA") < workflow.index(tag)
    assert "SKIP: superseded by" in workflow


def test_watchdog_workflow_schedule_and_probe() -> None:
    from pathlib import Path

    workflow = Path(".github/workflows/watchdog.yml").read_text(encoding="utf-8")

    assert 'cron: "40 22 * * 1-5"' in workflow
    assert "src.tools.watchdog_probe" in workflow
    assert "uv run --no-sync" in workflow
    assert "tailscale/github-action" in workflow
    assert "timeout-minutes:" in workflow


def test_watchdog_checks_ghcr_pat_expiry() -> None:
    from pathlib import Path

    workflow = Path(".github/workflows/watchdog.yml").read_text(encoding="utf-8")

    assert "github-authentication-token-expiration" in workflow


def test_deploy_workflow_builds_native_arm64_and_tags_commit_sha() -> None:
    from pathlib import Path

    workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04-arm" in workflow
    assert "platforms: linux/arm64" in workflow
    assert "linux/amd64" not in workflow
    assert "setup-qemu-action" not in workflow
    assert "sha-${{ github.sha }}" in workflow
    assert "${{ env.IMAGE }}:latest" in workflow
    assert "needs: test" in workflow
    assert "needs: build-and-push" in workflow
    assert "uv run pytest -q" in workflow
    assert "KCA_CODE_COMMIT=${{ github.sha }}" in workflow
