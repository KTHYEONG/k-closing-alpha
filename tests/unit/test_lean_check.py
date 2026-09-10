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


def test_spec_compliance_enum_member_and_class_attribute(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    models_py = src_dir / "models.py"
    models_py.write_text(
        "from enum import StrEnum\n\n"
        "class EvidenceKind(StrEnum):\n"
        "    LIFECYCLE_EVENTS = 'lifecycle_events'\n"
        "    OTHER_EVENT: str = 'other'\n",
        encoding="utf-8",
    )

    caller_py = src_dir / "caller.py"
    caller_py.write_text(
        "from src.models import EvidenceKind\n\ndef run():\n    return EvidenceKind.LIFECYCLE_EVENTS\n",
        encoding="utf-8",
    )

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        """{
            "changes": [
                {"target_file": "src/models.py", "kind": "enum_member", "symbol": "EvidenceKind.LIFECYCLE_EVENTS"},
                {"target_file": "src/models.py", "kind": "class_attribute", "symbol": "EvidenceKind.OTHER_EVENT"}
            ],
            "wiring": [
                {
                    "target_file": "src/caller.py",
                    "anchor": "run",
                    "import_symbol": "EvidenceKind",
                    "invocation_expression": "EvidenceKind.LIFECYCLE_EVENTS"
                }
            ]
        }""",
        encoding="utf-8",
    )

    code, diags = lean_check._check_spec_compliance(str(spec_file), pre_impl=False)
    assert code == 0, f"Expected 0 diagnostics, got: {diags}"


def test_spec_compliance_wiring_multiline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pipeline_py = src_dir / "pipeline.py"
    pipeline_py.write_text(
        "from src.service import process_items\n\n"
        "def main_flow():\n"
        "    # anchor: start processing\n"
        "    result = process_items(\n"
        "        arg1='foo',\n"
        "        arg2='bar',\n"
        "    )\n"
        "    return result\n",
        encoding="utf-8",
    )

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        """{
            "wiring": [
                {
                    "target_file": "src/pipeline.py",
                    "anchor": "start processing",
                    "import_symbol": "process_items",
                    "invocation_expression": "process_items(arg1='foo', arg2='bar')"
                }
            ]
        }""",
        encoding="utf-8",
    )

    code, diags = lean_check._check_spec_compliance(str(spec_file), pre_impl=False)
    assert code == 0, f"Expected 0 diagnostics, got: {diags}"


def test_spec_compliance_skeleton_with_dummy_callback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    foo_py = src_dir / "foo.py"
    foo_py.write_text("def run(): pass\n", encoding="utf-8")

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        """{
            "scenarios": [
                {
                    "scenario_id": "test_feature_with_callback",
                    "test_skeleton": "def test_feature_with_callback():\\n    def dummy_cb(x):\\n        pass\\n    assert dummy_cb(1) is None"
                }
            ],
            "wiring": [
                {
                    "target_file": "src/foo.py",
                    "anchor": "run"
                }
            ]
        }""",
        encoding="utf-8",
    )

    code, diags = lean_check._check_spec_compliance(str(spec_file), pre_impl=True)
    assert code == 0, f"Expected 0 diagnostics, got: {diags}"


def test_spec_compliance_deleted_enum_member(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    models_py = src_dir / "models.py"
    models_py.write_text(
        "from enum import Enum\n\nclass Status(Enum):\n    ACTIVE = 1\n",
        encoding="utf-8",
    )

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        """{
            "changes": [
                {"target_file": "src/models.py", "kind": "deleted_enum_member", "symbol": "Status.DEPRECATED"}
            ],
            "wiring": [
                {
                    "target_file": "src/models.py",
                    "anchor": "Status"
                }
            ]
        }""",
        encoding="utf-8",
    )

    code, diags = lean_check._check_spec_compliance(str(spec_file), pre_impl=False)
    assert code == 0, f"Expected 0 diagnostics, got: {diags}"


def test_spec_compliance_deleted_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    existing_file = src_dir / "to_delete.py"
    existing_file.write_text("def old(): pass\n", encoding="utf-8")

    # When file still exists, deleted_file should fail
    dummy_file = src_dir / "dummy.py"
    dummy_file.write_text("# anchor\n", encoding="utf-8")
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        """{
            "changes": [
                {"target_file": "src/to_delete.py", "kind": "deleted_file", "name": "to_delete"}
            ],
            "wiring": [
                {
                    "target_file": "src/dummy.py",
                    "anchor": "anchor"
                }
            ]
        }""",
        encoding="utf-8",
    )
    code, diags = lean_check._check_spec_compliance(str(spec_file), pre_impl=False)
    assert code == 1
    assert any("still exists" in d["error"] for d in diags)

    # When file is removed, deleted_file should pass
    existing_file.unlink()
    code, diags = lean_check._check_spec_compliance(str(spec_file), pre_impl=False)
    assert code == 0, f"Expected 0 diagnostics, got: {diags}"
