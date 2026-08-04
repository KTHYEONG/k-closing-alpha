"""lean_check import-index 캐싱 회귀 가드 테스트."""

from __future__ import annotations

from pathlib import Path

from tools.agent_skills import lean_check


def test_import_index_includes_semantic_reference(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_candidate_promotion_cli.py"
    test_file.write_text(
        "from src.candidate_promotion.cli import run_candidate_promotion\n",
        encoding="utf-8",
    )
    lean_check._imported_source_modules.cache_clear()
    lean_check._load_test_ast.cache_clear()
    modules = lean_check._imported_source_modules(str(test_file))
    assert "src.candidate_promotion.cli" in modules
    assert lean_check._test_references_source(str(test_file), "src/candidate_promotion/cli.py")


def test_test_file_parsed_exactly_once_across_pair_checks(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    test_file.write_text("import src.foo\n", encoding="utf-8")
    lean_check._load_test_ast.cache_clear()
    lean_check._imported_source_modules.cache_clear()
    for _ in range(5):
        lean_check._imported_source_modules(str(test_file))
    assert lean_check._load_test_ast.cache_info().misses == 1
    assert lean_check._imported_source_modules.cache_info().misses == 1
    assert lean_check._imported_source_modules.cache_info().hits == 4
