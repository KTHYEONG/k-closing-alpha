#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def _repository_test_files() -> list[str]:
    """Collect every repository test file, repo-relative and sorted."""
    found: list[str] = []
    for root, dirs, files in os.walk("tests"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        found.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.endswith(".py")
        )
    return sorted(found)


def _test_references_source(test_path: str, source_file: str) -> bool:
    """Check whether a test file names a source module by dotted or file path."""
    dotted = source_file[:-3].replace("/", ".") if source_file.endswith(".py") else source_file
    try:
        text = pathlib.Path(test_path).read_text(encoding="utf-8")
    except OSError:
        return False
    return dotted in text or source_file in text


def _matching_tests(source_file: str, test_files: list[str]) -> list[str]:
    """Return every repository test that covers ``source_file``.

    Exact mirrored ``tests/<category>/<dir>/test_<module>.py`` paths are the fast
    path; otherwise a local dotted-name reference matcher is used so
    feature-named CLI/workflow tests remain linked.
    """
    parts = source_file.split("/")
    module_name = parts[-1]
    test_name = f"test_{module_name}"
    exact = {
        f"tests/{category}/{'/'.join(parts[1:-1])}/{test_name}" if parts[1:-1]
        else f"tests/{category}/{test_name}"
        for category in ("unit", "integration", "e2e")
    }
    matched = [tp for tp in test_files if tp in exact]
    if matched:
        return matched
    return [tp for tp in test_files if _test_references_source(tp, source_file)]


def main() -> None:
    py_files: list[str] = []
    for root, dirs, files in os.walk("src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        py_files.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.endswith(".py")
        )
    py_files = sorted(py_files)
    test_files = _repository_test_files()

    code_map: dict[str, object] = {}
    for source_file in py_files:
        if source_file.endswith("__init__.py"):
            continue
        matched = _matching_tests(source_file, test_files)
        entry: dict[str, object] = {}
        if matched:
            entry["testing"] = matched[0] if len(matched) == 1 else matched
        code_map[source_file] = entry

    # Tolerate absent active code_map.json; do not recreate archived records under docs/
    # Only active src files are mapped; legacy sources remain in legacy/docs/code_map.json
    import contextlib
    docs_path = pathlib.Path("docs/code_map.json")
    # If active docs/code_map.json is absent, still generate active-only map without archived entries
    with contextlib.suppress(FileNotFoundError):
        if not docs_path.parent.exists():
            docs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(docs_path, "w", encoding="utf-8") as handle:
            json.dump(code_map, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"regenerated docs/code_map.json with {len(code_map)} canonical sources")
    except FileNotFoundError:
        print("active docs/code_map.json absent, skipped regeneration (archived map remains in legacy/docs)")


if __name__ == "__main__":
    main()
