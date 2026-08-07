#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import functools
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from typing import Any

JsonDiag = dict[str, Any]

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def _emit_json(
    status: str,
    phase: str,
    diagnostics: list[JsonDiag],
    coverage: int | None = None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "phase": phase,
            "exit_code": 0 if status == "PASS" else 1,
            "coverage": coverage,
            "diagnostics": diagnostics,
        }
    )


def _fail_exit_many(phase: str, header: str, diags: list[JsonDiag]) -> None:
    """Like _fail_exit but reports every diagnostic in one pass instead of
    just the first, so a single run surfaces the full remediation list."""
    print(header)
    for d in diags:
        print(f"FAIL | {d.get('error', '')}")
    print(_emit_json("FAIL", phase, diags), file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    # Strip unnecessary 'uv run' prefix when already running inside virtualenv to avoid double env setup overhead
    if len(cmd) >= 3 and cmd[0] == "uv" and cmd[1] == "run" and os.environ.get("VIRTUAL_ENV"):
        cmd = cmd[2:]
    try:
        return subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, shell=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout="",
            stderr=f"Error: timed out after {timeout}s.",
        )


def _fail_exit(phase: str, msg: str, diag: JsonDiag) -> None:
    print(msg)
    print(_emit_json("FAIL", phase, [diag]), file=sys.stderr)
    sys.exit(1)


def _is_test_file(path: str) -> bool:
    """A file is a test file if it lives under tests/ or its basename starts with test_.

    Substring matching (``"test_" in path``) would misclassify source files such as
    ``src/ml/backtest_evaluator.py`` whose name contains ``test_`` mid-word.
    """
    return path.startswith("tests/") or path.rsplit("/", 1)[-1].startswith("test_")


def _find_test_files(py_files: list[str]) -> list[str]:
    test_files = [f for f in py_files if _is_test_file(f)]
    source_files = [f for f in py_files if not _is_test_file(f)]
    repository_files = _repository_test_files()
    for sf in source_files:
        if sf.startswith("src/") and not sf.endswith("__init__.py"):
            parts = sf.split("/")
            module_name = parts[-1]
            test_name = f"test_{module_name}"
            for category in ["unit", "integration", "e2e"]:
                sub_path = "/".join(parts[1:-1])
                td = f"tests/{category}/{sub_path}" if sub_path else f"tests/{category}"
                tp = f"{td}/{test_name}"
                if os.path.exists(tp) and tp not in test_files:
                    test_files.append(tp)
                    break
            for tp in repository_files:
                if tp not in test_files and _test_references_source(tp, sf):
                    test_files.append(tp)
    return test_files


@functools.cache
def _repository_test_files() -> tuple[str, ...]:
    """Return test modules in deterministic order for semantic source matching.

    Cached: the walk spans the whole ``tests/`` tree and is unchanged within a
    single run, so repeated callers (``_find_test_files``, ``gen_code_map``)
    reuse one result instead of re-walking the tree per source file.
    """
    test_files: list[str] = []
    for root, _dirs, files in os.walk("tests"):
        test_files.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.startswith("test_") and filename.endswith(".py")
        )
    return tuple(sorted(test_files))


@functools.cache
def _load_test_ast(test_file: str) -> ast.Module | None:
    """Parse a test file exactly once; repeated (source, test) checks reuse the tree."""
    try:
        with open(test_file, encoding="utf-8") as handle:
            return ast.parse(handle.read(), filename=test_file)
    except (OSError, SyntaxError):
        return None


@functools.cache
def _imported_source_modules(test_file: str) -> frozenset[str]:
    """Every source module path the test imports, built once from the cached AST.

    Mirrors the import patterns the per-pair walk previously scanned:
    ``import a.b`` contributes ``a.b``; ``from a.b import c`` contributes both
    ``a.b`` and ``a.b.c``. Pair checks then collapse to O(1) set membership.
    """
    tree = _load_test_ast(test_file)
    if tree is None:
        return frozenset()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return frozenset(modules)


@functools.cache
def _test_references_source(test_file: str, source_file: str) -> bool:
    """Match tests by imported source symbol when filenames are feature-oriented.

    Exact mirrored ``test_<module>.py`` paths remain the fast path. AST matching
    covers valid names such as ``test_candidate_promotion_cli.py`` without
    relying on brittle substring searches.
    """
    source_module = source_file[:-3].replace("/", ".")
    return source_module in _imported_source_modules(test_file)


def _source_has_matching_test(source_file: str, test_files: list[str]) -> bool:
    """Check exact mirrored paths first, then semantic AST references."""
    parts = source_file.split("/")
    module_name = parts[-1]
    test_name = f"test_{module_name}"
    exact = {
        f"tests/{category}/{'/'.join(parts[1:-1])}/{test_name}" if parts[1:-1]
        else f"tests/{category}/{test_name}"
        for category in ("unit", "integration", "e2e")
    }
    return any(test in exact or _test_references_source(test, source_file) for test in test_files)


def _get_source_files(py_files: list[str]) -> list[str]:
    return [
        f for f in py_files
        if not _is_test_file(f)
    ]


def _get_target_coverage(file_path: str) -> int:
    # Tier 1 (Core): ML preprocessing pipeline
    if any(k in file_path for k in ("processing/",)):
        return 85
    # Tier 2 (Adapter/Repository/IO): data loaders, api, utils, sync
    if any(k in file_path for k in ("data/", "api/", "utils/", "sync/")):
        return 65
    return 50  # Fallback target for other source paths


@functools.cache
def _parse_coverage_table(stdout: str) -> dict[str, tuple[int | None, str]]:
    """Parse ``--cov-report=term-missing`` output into a single token table.

    Maps each reported module path (or ``TOTAL``) to ``(coverage %, missing
    ranges)`` so every per-file check is a dict lookup instead of a fresh
    full-output rescan.
    """
    table: dict[str, tuple[int | None, str]] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        for i, part in enumerate(parts):
            if "%" not in part:
                continue
            try:
                cov: int | None = int(part.replace("%", ""))
            except ValueError:
                cov = None
            missing = "".join(parts[i + 1 :])
            table.setdefault(parts[0], (cov, missing))
            break
    return table


def _coverage_entry(
    table: dict[str, tuple[int | None, str]], file_path: str
) -> tuple[int | None, str] | None:
    norm_path = file_path.replace("\\", "/")
    mkey = norm_path.replace(".py", "")
    module_key = mkey.replace("/", ".")
    for key in (norm_path, file_path, mkey, module_key, norm_path.replace("/", "\\")):
        if key in table:
            return table[key]
    return None



def _coverage_lookup(
    table: dict[str, tuple[int | None, str]], file_path: str
) -> int | None:
    entry = _coverage_entry(table, file_path)
    return entry[0] if entry is not None else None


def _coverage_missing_lines(
    table: dict[str, tuple[int | None, str]], file_path: str
) -> set[int]:
    """Parse the Missing column for ``file_path`` from the cached table."""
    entry = _coverage_entry(table, file_path)
    if entry is None or not entry[1]:
        return set()
    missing: set[int] = set()
    for token in entry[1].split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                a, b = token.split("-", 1)
                for i in range(int(a.strip()), int(b.strip()) + 1):
                    missing.add(i)
            else:
                missing.add(int(token))
        except ValueError:
            continue
    return missing


def _get_changed_lines(file_path: str) -> set[int]:
    """Return line numbers added/modified by git diff (uncommitted changes)."""
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "diff", "--unified=0", "HEAD", "--", file_path],  # noqa: S607
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
    except Exception:
        return set()
    changed: set[int] = set()
    if not res or not res.stdout:
        return changed
    for line in res.stdout.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) else 1
                for i in range(start, start + count):
                    changed.add(i)
    return changed



def _is_new_file(file_path: str) -> bool:
    """Return True if the file is new (does not exist in Git HEAD)."""
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "cat-file", "-e", f"HEAD:{file_path}"],  # noqa: S607
            capture_output=True, text=True, timeout=10
        )
        return res.returncode != 0
    except Exception:
        return False


def _is_stub_node(node: ast.AST) -> bool:
    """Check if an AST node (function/method body) is a stub implementation."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = node.body
    # Filter out docstrings and logger calls
    filtered_body = [
        stmt for stmt in body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str))
        and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and getattr(getattr(stmt.value, "func", None), "attr", "") in ("debug", "info", "warning", "error", "critical"))
    ]
    if not filtered_body:
        return True
    if len(filtered_body) == 1:
        single = filtered_body[0]
        if isinstance(single, ast.Pass):
            return True
        if isinstance(single, ast.Expr) and isinstance(single.value, ast.Constant) and single.value.value == Ellipsis:
            return True
        if isinstance(single, ast.Raise):
            if isinstance(single.exc, ast.Call) and getattr(single.exc.func, "id", None) == "NotImplementedError":
                return True
            if isinstance(single.exc, ast.Name) and single.exc.id == "NotImplementedError":
                return True
        # Check dummy single return (e.g. return None, return {}, return [], return "", return True, return False)
        if isinstance(single, ast.Return):
            if single.value is None:
                return True
            if isinstance(single.value, ast.Constant) and (single.value.value in (None, "", 0, False, True) or isinstance(single.value.value, (int, float, str))):
                return True
            if isinstance(single.value, (ast.List, ast.Dict, ast.Tuple, ast.Set)) and not getattr(single.value, "elts", getattr(single.value, "keys", None)):
                return True
    return False


def _is_json_primitive(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, list):
        return all(_is_json_primitive(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json_primitive(x) for k, x in v.items())
    return False


def _looks_like_prose(v: Any) -> bool:
    """Spec authors write two kinds of string values: real short scalar
    arguments ("xs", "rsi") and descriptive placeholders for fixtures/objects
    ("300-bar synthetic fixture", "(8,3) finite"). Real arguments never
    contain whitespace; descriptions almost always do. This single heuristic
    is what actually distinguishes them in practice."""
    if isinstance(v, str):
        return " " in v.strip() or len(v) > 40
    if isinstance(v, list):
        return any(_looks_like_prose(x) for x in v)
    if isinstance(v, dict):
        return any(_looks_like_prose(x) for x in v.values())
    return False


def _is_concretely_callable(fn: Any, inp: dict[str, Any]) -> bool:
    """True only when every input key is a real parameter of fn, every value
    is a JSON primitive with no prose-like strings, and binding those kwargs
    against fn's signature would not raise (i.e. no missing required args)."""
    if not all(_is_json_primitive(v) and not _looks_like_prose(v) for v in inp.values()):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    valid_params = set(sig.parameters.keys())
    if not set(inp.keys()) <= valid_params:
        return False
    try:
        sig.bind(**inp)
    except TypeError:
        return False
    return True


def _values_match(actual: Any, expected: Any) -> bool:
    try:
        import numpy as np
        if isinstance(actual, np.ndarray):
            actual = actual.tolist()
        if isinstance(expected, bool) or isinstance(actual, bool):
            return bool(actual) == bool(expected)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(actual) - float(expected)) < 1e-6
        if isinstance(expected, list) and isinstance(actual, list):
            return len(expected) == len(actual) and all(
                _values_match(a, e) for a, e in zip(actual, expected, strict=True)
            )
        return bool(actual == expected)
    except Exception:
        return False


def _coerce_list_args(inp: dict[str, Any]) -> dict[str, Any]:
    """This codebase's contract functions overwhelmingly take NDArray
    parameters. A JSON list value ([[1.0],[2.0]]) is coerced to a numpy array
    (float64, or bool if every leaf is a bool) so real array-taking functions
    become executable instead of only ever matching scalar-only signatures."""
    import numpy as np

    def is_bool_only(v: Any) -> bool:
        if isinstance(v, bool):
            return True
        if isinstance(v, list):
            return all(is_bool_only(x) for x in v)
        return False

    out: dict[str, Any] = {}
    for k, v in inp.items():
        if isinstance(v, list):
            out[k] = np.array(v, dtype=np.bool_ if is_bool_only(v) else np.float64)
        else:
            out[k] = v
    return out


def _file_to_module(file_hint: str) -> str:
    path = file_hint[:-3] if file_hint.endswith(".py") else file_hint
    return path.replace("/", ".").replace("\\", ".")


def _repo_relative(path: str) -> str:
    """Normalize a contract file path (often absolute) to the repo root.

    Contracts declare absolute ``target_file`` / ``caller_file`` paths; module
    imports and ``os.path.exists`` checks need repo-relative paths so the tool
    works regardless of how the contract was authored.
    """
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, os.getcwd())
    except ValueError:
        return path


def _execute_assertions(
    fh: str, kind: str, name: str, assertions: list[Any],
) -> list[JsonDiag]:
    """Best-effort dynamic verification: actually calls the contract's target
    function with each assertion's 'input' and compares to 'output'/'exception'.

    Only attempts execution when kind=='function' (no dotted owner -- methods
    need an instance the contract doesn't describe) and every input/output
    value is a JSON primitive (numbers/strings/bools/lists/dicts thereof).
    Descriptive assertions ("output": "tuple of 4 LegBook in registry order")
    or fixture-based inputs ("input": {"close": "300-bar synthetic fixture"})
    are silently skipped, not failed -- they still require a human/implement
    step, but a skip must never be reported as a pass.
    """
    diags: list[JsonDiag] = []
    if kind != "function" or "." in name:
        return diags
    module_name = _file_to_module(fh)
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        diags.append({
            "file": fh, "line": 0,
            "error": f"Spec: could not import {module_name} to verify assertions for '{name}': {type(e).__name__}: {e}",
            "fix_hint": f"Fix the ImportError in {fh} so assertion verification can run",
        })
        return diags
    fn = getattr(module, name, None)
    if fn is None or not callable(fn):
        return diags

    for idx, assertion in enumerate(assertions):
        inp = assertion.get("input")
        if not isinstance(inp, dict) or not _is_concretely_callable(fn, inp):
            continue  # not machine-callable (fixture/object/prose input) -- skip, don't fail

        expects_exception = "exception" in assertion
        expects_output = (
            "output" in assertion
            and _is_json_primitive(assertion["output"])
            and not _looks_like_prose(assertion["output"])
        )
        if not expects_exception and not expects_output:
            continue  # descriptive-only assertion -- skip, don't fail

        try:
            result = fn(**_coerce_list_args(inp))
        except Exception as e:  # noqa: BLE001
            if expects_exception:
                expected_exc = assertion["exception"]
                if type(e).__name__ != expected_exc:
                    diags.append({
                        "file": fh, "line": 0,
                        "error": (
                            f"Spec assertion #{idx} for '{name}': expected exception "
                            f"{expected_exc!r} but got {type(e).__name__}: {e}"
                        ),
                        "fix_hint": f"Fix {name} in {fh} to raise {expected_exc} for input {inp}",
                    })
            else:
                diags.append({
                    "file": fh, "line": 0,
                    "error": f"Spec assertion #{idx} for '{name}': unexpected {type(e).__name__} for input {inp}: {e}",
                    "fix_hint": f"Fix {name} in {fh} so it does not raise for input {inp}",
                })
            continue

        if expects_exception:
            diags.append({
                "file": fh, "line": 0,
                "error": (
                    f"Spec assertion #{idx} for '{name}': expected exception "
                    f"{assertion['exception']!r} but got return value {result!r} for input {inp}"
                ),
                "fix_hint": f"Fix {name} in {fh} to raise {assertion['exception']} for input {inp}",
            })
        elif expects_output and not _values_match(result, assertion["output"]):
            diags.append({
                "file": fh, "line": 0,
                "error": (
                    f"Spec assertion #{idx} for '{name}': input {inp} produced {result!r}, "
                    f"expected {assertion['output']!r}"
                ),
                "fix_hint": f"Fix {name} in {fh} so input {inp} returns {assertion['output']!r}",
            })
    return diags


@functools.lru_cache(maxsize=1)
def _get_src_files_contents() -> tuple[tuple[str, str], ...]:
    """Cache all python files under src/ and their contents once per process.

    Eliminates redundant disk I/O when checking for orphaned implementations.
    """
    results: list[tuple[str, str]] = []
    if os.path.exists("src"):
        for root, dirs, files in os.walk("src"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn_name in files:
                if fn_name.endswith(".py"):
                    fp = os.path.join(root, fn_name)
                    try:
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            results.append((fp, f.read()))
                    except OSError:
                        continue
    return tuple(results)


def _check_orphaned_implementations(fh: str, kind: str, name: str) -> list[JsonDiag]:
    """A function/class that only appears in its own def line and in tests/
    is dead in production -- it compiles, may even be unit-tested, and does
    nothing when the pipeline runs. Grep src/ (excluding tests/) for any
    reference outside the definition itself."""
    if kind == "field":
        return []
    # Dev utilities under tools/ are spec-phase helpers, not runtime: they are
    # wired by their own caller inside tools/ (or invoked from a tool CLI), so
    # a src/-only grep can never see a caller. The orphan gate is scoped to the
    # production tree it is designed to police.
    if not fh.startswith("src"):
        return []
    leaf = name.rpartition(".")[2] if "." in name else name
    if not leaf:
        return []
    ref_pat = re.compile(rf"\b{re.escape(leaf)}\b")
    def_pat = re.compile(rf"^\s*(?:def|class)\s+{re.escape(leaf)}\b")
    found_caller = False
    for _fp, content in _get_src_files_contents():
        for line in content.splitlines():
            if ref_pat.search(line) and not def_pat.match(line):
                found_caller = True
                break
        if found_caller:
            break
    if found_caller:
        return []
    return [{
        "file": fh, "line": 0,
        "error": f"Spec: {kind} '{name}' has no callers in src/ outside its own definition (orphaned implementation)",
        "fix_hint": f"Wire {name} into its caller per the spec's wiring plan -- it currently does nothing in production",
    }]


@functools.lru_cache(maxsize=1)
def _get_tests_files_contents() -> tuple[tuple[str, str], ...]:
    """Cache all test files and their contents once per process."""
    results: list[tuple[str, str]] = []
    if os.path.exists("tests"):
        for root, _dirs, fnames in os.walk("tests"):
            for fn in fnames:
                if fn.endswith(".py"):
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            results.append((fp, f.read()))
                    except OSError:
                        continue
    return tuple(results)


def _execute_python_assertion(fh: str, name: str, assertion_code: str) -> list[JsonDiag]:
    diags: list[JsonDiag] = []
    if not assertion_code:
        return diags
    module_name = _file_to_module(fh)
    exec_globals: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        module = importlib.import_module(module_name)
        exec_globals.update(vars(module))
    except Exception as e:
        diags.append({
            "file": fh, "line": 0,
            "error": f"Spec: could not import {module_name} to verify python_assertion for '{name}': {type(e).__name__}: {e}",
            "fix_hint": f"Fix the ImportError in {fh} so assertion verification can run",
        })
        return diags

    try:
        exec(assertion_code, exec_globals)  # noqa: S102
    except Exception as e:
        diags.append({
            "file": fh, "line": 0,
            "error": f"Spec python_assertion for '{name}' failed: {type(e).__name__}: {e}",
            "fix_hint": f"Fix {name} in {fh} or python_assertion in contract.json to pass assertion",
        })
    return diags


def _iter_contract_entries(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize legacy 'contracts', 'changes', and top-level contract schemas.

    Two external schemas are supported:
    * generic ``contracts`` with ``file_hint`` / ``name``
    * ``changes`` with ``target_file`` / ``symbol`` / ``python_assertion``

    This repo's own spec contracts also declare their surface at the top level
    (``target_file`` / ``symbol`` / ``python_assertion`` with no section wrapper),
    so those are reduced to a single entry as well. All three are normalized to
    one list so symbol, stub, and assertion checks stay shared.
    """
    entries: list[dict[str, Any]] = list(contract.get("contracts", []))
    for change in contract.get("changes", []):
        symbol = change.get("symbol", "")
        # An explicit ``kind`` (e.g. ``"field"`` for an uppercase module-level
        # constant like a profile-id) overrides the case-based default, matching
        # the ``contracts`` schema which already passes ``kind`` through.
        entries.append({
            "file_hint": _repo_relative(change.get("target_file", "")),
            "kind": change.get("kind") or ("class" if symbol and symbol[0].isupper() else "function"),
            "name": symbol,
            "assertions": [],
            "python_assertion": change.get("python_assertion", ""),
        })
    if contract.get("target_file") and contract.get("symbol"):
        symbol = contract["symbol"]
        entries.append({
            "file_hint": _repo_relative(contract.get("target_file", "")),
            "kind": contract.get("kind") or ("class" if symbol and symbol[0].isupper() else "function"),
            "name": symbol,
            "assertions": [],
            "python_assertion": contract.get("python_assertion", ""),
        })
    return entries


def _check_spec_compliance(spec_path: str) -> tuple[int, list[JsonDiag]]:
    diagnostics: list[JsonDiag] = []
    try:
        with open(spec_path, encoding="utf-8") as f:
            contract = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return (1, [{"file": spec_path, "line": 0, "error": f"Spec file error: {e}", "fix_hint": ""}])

    for c in _iter_contract_entries(contract):
        fh: str = c.get("file_hint", "") or c.get("file", "")
        kind: str = c.get("kind", "function")
        raw_name: str = c.get("name", "") or c.get("symbol", "")
        # Clean symbol name if it contains parenthetical hints like 'run_backtest (extended)'
        name: str = raw_name.split()[0] if raw_name else ""
        assertions: list[Any] = c.get("assertions", [])
        python_assertion: str = c.get("python_assertion", "")
        if not fh or not name:
            continue
        if not os.path.exists(fh):
            d = {"file": fh, "line": 0, "error": f"Spec: file not found ({kind} {name})", "fix_hint": f"Create {fh}"}
            diagnostics.append(d)
            continue
        
        # Check assertions requirement
        if not assertions and not python_assertion:
            msg = f"Spec contract '{name}' in {fh} must define at least one exact input/output assertion or 'python_assertion'"
            d = {"file": fh, "line": 0, "error": msg, "fix_hint": "Add 'assertions' or 'python_assertion' in contract.json"}
            diagnostics.append(d)

        with open(fh, encoding="utf-8") as sf:
            sf_content = sf.read()
            if kind == "field":
                field_name = name.split(".")[-1] if "." in name else name
                pat = rf"\b{re.escape(field_name)}[\"']?\s*(?::|=)"
                if not re.search(pat, sf_content, re.MULTILINE):
                    msg = f"Spec: {kind} '{name}' not implemented"
                    d = {"file": fh, "line": 0, "error": msg, "fix_hint": f"Implement {kind} {name} in {fh}"}
                    diagnostics.append(d)
            else:
                owner, _, leaf = name.rpartition(".")
                target_node: ast.AST | None = None
                found_impl = False
                try:
                    tree = ast.parse(sf_content, filename=fh)
                    if owner:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == owner:
                                for member in node.body:
                                    if (
                                        isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                                        and member.name == leaf
                                    ):
                                        target_node = member
                                        found_impl = True
                    else:
                        for node in ast.walk(tree):
                            if (
                                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                and node.name == name
                            ):
                                target_node = node
                                found_impl = True
                except Exception:  # noqa: S110
                    pat = rf"^(?:class|def)\s+{re.escape(name)}\b"
                    found_impl = bool(re.search(pat, sf_content, re.MULTILINE))

                if not found_impl:
                    msg = f"Spec: {kind} '{name}' not implemented"
                    d = {"file": fh, "line": 0, "error": msg, "fix_hint": f"Implement {kind} {name} in {fh}"}
                    diagnostics.append(d)
                elif target_node is not None and _is_stub_node(target_node):
                    msg = f"Spec: {kind} '{name}' is a stub or dummy implementation (pass / Ellipsis / NotImplementedError / logger+dummy return)"
                    d = {"file": fh, "line": getattr(target_node, "lineno", 0), "error": msg, "fix_hint": f"Implement real logic in {name}"}
                    diagnostics.append(d)
                else:
                    if assertions:
                        diagnostics.extend(_execute_assertions(fh, kind, name, assertions))
                    if python_assertion:
                        diagnostics.extend(_execute_python_assertion(fh, name, python_assertion))
                    diagnostics.extend(_check_orphaned_implementations(fh, kind, name))

    for s in contract.get("scenarios", []):
        test_name: str = s.get("name", "") or s.get("scenario_id", "")
        if not test_name:
            continue
        # repo schema: scenario_id like 'ECR-01-CAUSAL-ATTRIBUTION' is referenced
        # by its stable prefix ('ECR-01') in the target test's docstring.
        # Use exact word boundary or full prefix (minimum hyphenated prefix like ECR-01) to avoid weak substring matches (e.g. 'ECR' matching ECR-01~06).
        if s.get("scenario_id"):
            parts = test_name.split("-")
            reference = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
        else:
            reference = test_name

        target_test_file: str = _repo_relative(s.get("target_test_file", ""))
        found = False
        ref_pattern = re.compile(rf"\b{re.escape(reference)}\b")
        # Accept tests named either `test_<scenario>` (repo convention) or the
        # bare scenario id; `\b` treats `_` as a word char so the underscore
        # prefix must be matched explicitly.
        test_def_pattern = re.compile(
            rf"^[ \t]*def\s+(?:{re.escape(test_name)}|test_{re.escape(reference)})\s*\(",
            re.MULTILINE,
        )
        if target_test_file and os.path.exists(target_test_file):
            with open(target_test_file, encoding="utf-8") as tf:
                content = tf.read()
            found = bool(ref_pattern.search(content)) or bool(test_def_pattern.search(content))
        if not found:
            for _fp, content in _get_tests_files_contents():
                if bool(ref_pattern.search(content)) or test_def_pattern.search(content):
                    found = True
                    break
        if not found:
            fix_hint = f"Write a test referencing {test_name} in {target_test_file}" if target_test_file else f"Write {test_name}"
            d = {"file": target_test_file, "line": 0, "error": f"Spec: missing test '{test_name}'", "fix_hint": fix_hint}
            diagnostics.append(d)

    wirings: list[dict[str, Any]] = []
    if "wiring" in contract and isinstance(contract["wiring"], list):
        wirings.extend(contract["wiring"])
    elif "wiring" in contract and isinstance(contract["wiring"], dict):
        # repo schema: a single wiring object keyed by caller_file/anchor
        wirings.append(contract["wiring"])
    wirings.extend(
        c["wiring"] for c in contract.get("contracts", []) if "wiring" in c and isinstance(c["wiring"], dict)
    )

    if not wirings:
        msg = "Spec: contract.json missing mandatory 'wiring' section defining pipeline integration"
        diagnostics.append({"file": spec_path, "line": 0, "error": msg, "fix_hint": "Add 'wiring' array or per-contract wiring object to contract.json"})

    for w in wirings:
        wf: str = _repo_relative(
            w.get("file", "") or w.get("target", "") or w.get("caller_file", "")
        )
        anchor: str = w.get("anchor", "")
        import_symbol: str = w.get("import_symbol", "") or w.get("callee", "")
        invocation_symbol: str = w.get("invocation_symbol", "")
        invocation_regex: str = w.get("invocation_regex", "")
        invocation_expression: str = w.get("invocation_expression", "")
        if not wf:
            continue
        if not os.path.exists(wf):
            d = {"file": wf, "line": 0, "error": f"Spec wiring target file not found: {wf}", "fix_hint": f"Create {wf}"}
            diagnostics.append(d)
            continue
        with open(wf, encoding="utf-8") as f:
            wf_content = f.read()
            normalized_wf_content = re.sub(r"\s+", " ", wf_content)
            
            # Anchor check (check direct substring, regex, or code symbol within anchor)
            if anchor:
                anchor_code_tokens = [t for t in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", anchor) if t not in ("step", "main", "only", "when", "after", "before", "signature")]
                found_anchor = (anchor in wf_content) or (re.sub(r"\s+", " ", anchor) in normalized_wf_content) or any(t in wf_content for t in anchor_code_tokens)
                if not found_anchor:
                    hint = f"Add ref to {anchor} in {wf}"
                    d = {"file": wf, "line": 0, "error": f"Spec wiring: missing anchor '{anchor}'", "fix_hint": hint}
                    diagnostics.append(d)

            if import_symbol and import_symbol not in wf_content:
                hint = f"Import or reference '{import_symbol}' in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec wiring: missing reference to '{import_symbol}' in {wf}", "fix_hint": hint}
                diagnostics.append(d)

            if invocation_symbol and invocation_symbol not in wf_content:
                hint = f"Invoke or instantiate '{invocation_symbol}' in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec wiring: missing invocation of '{invocation_symbol}' in {wf}", "fix_hint": hint}
                diagnostics.append(d)

            if invocation_regex and not re.search(invocation_regex, wf_content, re.MULTILINE):
                hint = f"Invoke matching pattern '{invocation_regex}' in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec wiring: missing pattern match '{invocation_regex}' in {wf}", "fix_hint": hint}
                diagnostics.append(d)

            if invocation_expression:
                normalized_expr = re.sub(r"\s+", " ", invocation_expression)
                found_expr = (invocation_expression in wf_content) or (normalized_expr in normalized_wf_content)
                if not found_expr and import_symbol:
                    found_expr = import_symbol in wf_content
                if not found_expr:
                    hint = f"Invoke expression '{invocation_expression}' in {wf}"
                    d = {"file": wf, "line": 0, "error": f"Spec wiring: missing invocation expression '{invocation_expression}' in {wf}", "fix_hint": hint}
                    diagnostics.append(d)

    return (1 if diagnostics else 0, diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean Check with JSON diagnostics.")
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--spec", default=None, help="Path to spec contract JSON for compliance verification")
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff linting")
    parser.add_argument("--skip-mypy", action="store_true", help="Skip Mypy static check")
    parser.add_argument(
        "--spec-only", action="store_true",
        help="Run ONLY spec-compliance (AST non-dummy check, dynamic assertion execution, "
             "orphaned-implementation gate, wiring text-match) and exit -- no ruff/mypy/pytest/"
             "coverage. Seconds, not minutes. Intended as an /implement inner-loop check: run it "
             "after Phase B (core logic) and again after Phase C (wiring), before ever invoking "
             "the full /check pass. A green --spec-only run is necessary but not sufficient for "
             "/check PASS -- mypy/pytest/coverage still run separately and can still fail.",
    )
    parser.add_argument(
        "--deselect", nargs="*", default=[],
        help="Pytest node ids to deselect (use for pre-existing baseline failures "
             "confirmed via git-stash reproduction, never for failures introduced this session)",
    )
    parser.add_argument(
        "--pytest-timeout",
        type=int,
        default=None,
        help="Seconds to allow the pytest+coverage step (default: auto-scaled by "
        "test-file count, floor 300s, cap 1200s). Explicitly raising this is the "
        "right move for heavy orchestrator test files that exceed the default.",
    )
    args = parser.parse_args()

    if args.spec_only:
        if not args.spec:
            print("FAIL | --spec-only requires --spec")
            sys.exit(2)
        ec, diags = _check_spec_compliance(args.spec)
        if ec != 0:
            for d in diags:
                print(f"FAIL | {d.get('error', '')}")
            print(_emit_json("FAIL", "spec-compliance", diags), file=sys.stderr)
            sys.exit(1)
        print("PASS | Spec compliance verified (assertions executed, orphaned-implementation gate clear)")
        print(_emit_json("PASS", "spec-compliance", []), file=sys.stderr)
        sys.exit(0)

    if not args.files and not args.spec_only:
        # Auto-detect modified and untracked .py files via git
        try:
            diff_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=False
            )
            git_files: list[str] = []
            for line in diff_res.stdout.splitlines():
                status_code = line[:2]
                filepath = line[3:].strip()
                # Ignore deleted files
                if (
                    "D" not in status_code
                    and filepath.endswith(".py")
                    and not filepath.startswith("tools/")
                    and os.path.exists(filepath)
                ):
                    git_files.append(filepath)
            args.files = git_files
        except Exception:
            args.files = []

    # Auto-detect spec file in docs/specs/*_contract.json if not provided
    if not args.spec and os.path.exists("docs/specs"):
        spec_candidates = [
            os.path.join("docs/specs", f)
            for f in os.listdir("docs/specs")
            if f.endswith("_contract.json") or f == "contract.json"
        ]
        if spec_candidates:
            # Pick the most recently modified spec file
            spec_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            args.spec = spec_candidates[0]

    py_files = [f for f in args.files if f.endswith(".py")]
    if not py_files and not args.spec:
        print("ALLCHECKS:PASS | No modified .py files detected")
        sys.exit(0)


    # 0. Spec Compliance (optional, runs first — most fundamental)
    if args.spec:
        ec, diags = _check_spec_compliance(args.spec)
        if ec != 0:
            for d in diags:
                print(f"FAIL | {d.get('error', '')}")
            print(_emit_json("FAIL", "spec-compliance", diags), file=sys.stderr)
            sys.exit(1)
        print("PASS | Spec compliance verified")

    # 1. Co-modification Mapping Verification (tools/ excluded)
    test_files = _find_test_files(py_files)
    for pf in py_files:
        if not pf.startswith("src/") or pf.endswith("__init__.py") or pf.startswith("tools/"):
            continue
        if not _source_has_matching_test(pf, test_files):
            d = {"file": pf, "line": 0, "error": f"No matching test for {pf}", "fix_hint": ""}
            _fail_exit("co-modification", f"FAIL | {pf}: test file missing", d)

    # 2. print() Detection (tools/ excluded)
    if not args.skip_lint:
        print_re = re.compile(r"(?<!#)\bprint\s*\(")
        for pf in py_files:
            if pf.startswith("tools/"):
                continue
            with open(pf, encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if print_re.search(line):
                        d = {"file": pf, "line": idx, "error": "Unsanctioned print()", "fix_hint": ""}
                        _fail_exit("print-check", f"FAIL | {pf}:{idx} print() detected", d)

    # 3. Ruff
    if not args.skip_lint:
        ruff_res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
        if ruff_res.returncode != 0:
            out = (ruff_res.stdout or ruff_res.stderr).strip()
            # Slice error output to max 10 lines for token efficiency
            out_sliced = "\n".join(out.splitlines()[:10])
            d = {"file": py_files[0] if py_files else "", "line": 0, "error": out_sliced, "fix_hint": "Resolve ruff errors"}
            _fail_exit("ruff", "FAIL | Ruff Lint Failed", d)

    # 4. Mypy
    if not args.skip_mypy:
        mypy_res = run_cmd(["uv", "run", "mypy", *py_files, "--ignore-missing-imports"])
        if mypy_res.returncode != 0:
            out = (mypy_res.stdout or mypy_res.stderr).strip()
            # Slice error output to max 10 lines for token efficiency
            out_sliced = "\n".join(out.splitlines()[:10])
            d = {"file": py_files[0] if py_files else "", "line": 0, "error": out_sliced, "fix_hint": "Resolve type errors"}
            _fail_exit("mypy", "FAIL | Mypy Type Check Failed", d)

    # 5. Single pytest with coverage
    source_files = _get_source_files(py_files)

    if not test_files:
        print("PASS | Lint & Type check passed (no tests to run)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
        return

    # Measure coverage only on the changed source files, never the whole src
    # tree: tracing every module roughly doubles pytest time for a single-file
    # change while the per-file gate below only ever inspects changed files.
    # (pytest-cov accumulates repeated --cov sources, so one flag per file is
    # correct; report keys match `_coverage_entry`'s path spellings.)
    cov_args = []
    if source_files:
        # Dotted module names, never ".py"-suffixed paths: coverage treats a
        # value ending in ".py" as a literal module literally named "...backtest.py"
        # (never imported), which silently reports 0%. Modules are importable via
        # the project's pythonpath and coverage reports them under their real path.
        cov_args = [
            (
                f"--cov={sf[:-3].replace('/', '.')}"
                if sf.endswith(".py")
                else f"--cov={sf}"
            )
            for sf in source_files
        ]
    report_args = ["--cov-report=term-missing"] if cov_args else []

    deselect_args = [f"--deselect={node}" for node in args.deselect]
    core_cmd = [
        "uv",
        "run",
        "pytest",
        *cov_args,
        "-m",
        "not slow",
        *test_files,
        *deselect_args,
        "-q",
        "--tb=line",
        *report_args,
    ]
    # The growth evaluator's sealed integration fixtures exercise real rolling
    # windows and reliability gates.  Package-wide coverage tracing can exceed
    # the former three-minute limit even when the test command itself passes.
    pytest_timeout = args.pytest_timeout or max(300, min(1200, 240 * len(test_files)))
    pt_res = run_cmd(core_cmd, timeout=pytest_timeout)

    cov_val: int | None = None
    missing_infos: list[str] = []

    if pt_res.returncode == 0:
        cov_table = _parse_coverage_table(pt_res.stdout)
        cov_val = _coverage_lookup(cov_table, "TOTAL")

        # Check coverage targets per file
        coverage_violations: list[JsonDiag] = []
        for sf in source_files:
            target_cov = _get_target_coverage(sf)
            if target_cov == 0:
                continue

            actual_cov = _coverage_lookup(cov_table, sf)
            if actual_cov is None:
                actual_cov = 0  # Fail-safe: if file not executed at all, count as 0%

            is_new = _is_new_file(sf)
            if is_new:
                # New files must meet the target coverage globally
                if actual_cov < target_cov:
                    d = {
                        "file": sf,
                        "line": 0,
                        "error": f"Coverage target violation (New File): actual {actual_cov}% < target {target_cov}%",
                        "fix_hint": f"Add test cases to cover the new file {sf} up to {target_cov}%",
                    }
                    coverage_violations.append(d)
            else:
                # Existing files: ensure 100% of modified lines are covered AND total coverage is at or above 40% floor (for files <= 1000 lines to prevent legacy token waste)
                num_lines = 0
                try:
                    with open(sf, encoding="utf-8") as f_sf:
                        num_lines = len(f_sf.readlines())
                except Exception:  # noqa: S110
                    pass
                if actual_cov < 40 and num_lines <= 1000:
                    d = {
                        "file": sf,
                        "line": 0,
                        "error": f"Coverage target violation: actual coverage {actual_cov}% is below minimum safety floor (40%)",
                        "fix_hint": f"Add test cases to bring total coverage of {sf} above 40% (current: {actual_cov}%)",
                    }
                    coverage_violations.append(d)
                
                changed = _get_changed_lines(sf)
                if changed:
                    missing = _coverage_missing_lines(cov_table, sf)
                    uncovered_changed = missing & changed
                    if uncovered_changed:
                        d = {
                            "file": sf,
                            "line": 0,
                            "error": f"Coverage target violation (Modified File): changed lines {sorted(uncovered_changed)} are not covered by tests",
                            "fix_hint": (
                                f"Add fast unit tests targeting the modified lines in {sf}: "
                                f"{sorted(uncovered_changed)} (the audit runs `-m 'not slow'`; "
                                f"slow integration tests are excluded by design)"
                            ),
                        }
                        coverage_violations.append(d)

            entry = _coverage_entry(cov_table, sf)
            if entry is not None and entry[1]:
                missing_infos.append(f"{sf.split('/')[-1]}:{entry[1]}")

        if coverage_violations:
            n = len(coverage_violations)
            _fail_exit_many(
                "coverage-target",
                f"FAIL | {n} coverage target violation(s) across {len({d['file'] for d in coverage_violations})} file(s)",
                coverage_violations,
            )

        suffix = f", Missing: [{', '.join(missing_infos)}]" if missing_infos else ""
        cov_s = f"{cov_val}%" if cov_val is not None else "N/A"
        print(f"PASS | All checks passed (Cov {cov_s}{suffix})")
        print(_emit_json("PASS", "all", [], cov_val), file=sys.stderr)
    else:
        last_err = [
            line
            for line in (pt_res.stdout or "").splitlines()
            if any(x in line for x in ("FAIL", "Error", "AssertionError"))
        ]
        # A subprocess timeout (returncode 124) puts its explanation on stderr,
        # not stdout -- surface it so a killed run is never misreported as an
        # assertion failure.
        cause = (
            last_err[-1]
            if last_err
            else (pt_res.stderr or "Check pytest output.").strip()
        )
        d = {"file": "", "line": 0, "error": cause, "fix_hint": "Fix failing assertions in tests"}
        _fail_exit("pytest", f"FAIL | Pytest Failed: {cause}", d)


if __name__ == "__main__":
    main()
