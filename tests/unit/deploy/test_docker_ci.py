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

