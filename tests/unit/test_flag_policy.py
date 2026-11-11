from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import Settings
from src.tools.provision_env import VPS_SELECTORS

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "docs" / "flags.toml"
RULE_FILE = ".agents/rules/config-flags.md"
TOGGLE_NAME_RE = re.compile(r"(?i)(enable|disable|resend|_on_|allow|auto_adopt|force_)")
KINDS = frozenset({"env_binding", "safety_interlock", "component", "experiment"})
REVIEW_GRACE_DAYS = 14
TODAY: date = date.today()

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def discover_toggles(model: type[BaseModel] = Settings) -> dict[str, bool]:
    return {
        name: bool(field.default)
        for name, field in model.model_fields.items()
        if field.annotation is bool and TOGGLE_NAME_RE.search(name)
    }


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        rows = tomllib.load(handle).get("flag", [])
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["name"]
        if name in registry:
            raise ValueError(f"duplicate flag registry row: {name}")
        registry[name] = row
    return registry


def coverage_violations(toggles: Mapping[str, bool], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    missing = [f"{name}: toggle has no row in docs/flags.toml ({RULE_FILE})" for name in toggles if name not in registry]
    stale = [f"{name}: registry row names no existing toggle ({RULE_FILE})" for name in registry if name not in toggles]
    return [*missing, *stale]


def kind_violations(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: kind {row.get('kind')!r} is not one of {sorted(KINDS)} ({RULE_FILE})"
        for name, row in registry.items()
        if row.get("kind") not in KINDS
    ]


def default_violations(toggles: Mapping[str, bool], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: registry default {row.get('default')!r} differs from code default {toggles[name]!r} ({RULE_FILE})"
        for name, row in registry.items()
        if name in toggles and row.get("default") != toggles[name]
    ]


def component_violations(registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{name}: component must default to True, found {row.get('default')!r} ({RULE_FILE})"
        for name, row in registry.items()
        if row.get("kind") == "component" and row.get("default") is not True
    ]


def experiment_violations(registry: Mapping[str, Mapping[str, Any]], today: date) -> list[str]:
    violations: list[str] = []
    for name, row in registry.items():
        if row.get("kind") != "experiment":
            continue
        if not str(row.get("exit_condition", "")).strip():
            violations.append(f"{name}: experiment needs a non-empty exit_condition ({RULE_FILE})")
        try:
            review_by = date.fromisoformat(str(row.get("review_by", "")))
        except ValueError:
            violations.append(f"{name}: experiment review_by must be YYYY-MM-DD ({RULE_FILE})")
            continue
        if today > review_by + timedelta(days=REVIEW_GRACE_DAYS):
            violations.append(
                f"{name}: experiment review_by {review_by.isoformat()} is more than "
                f"{REVIEW_GRACE_DAYS} days past; delete the flag and its OFF branch ({RULE_FILE})"
            )
    return violations


def _as_bool(value: str) -> bool | None:
    token = value.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def baseline_violations(
    registry: Mapping[str, Mapping[str, Any]], baseline: Iterable[tuple[str, str]]
) -> list[str]:
    violations: list[str] = []
    for key, value in baseline:
        row = registry.get(key)
        if row is None or row.get("kind") == "env_binding":
            continue
        kind = row.get("kind")
        if kind in {"component", "experiment"}:
            violations.append(f"{key}: deploy baseline must not set {kind} toggles ({RULE_FILE})")
        elif kind == "safety_interlock" and _as_bool(value) == row.get("default"):
            violations.append(f"{key}: deploy baseline sets the toggle to its own default ({RULE_FILE})")
    return violations


def test_registry_covers_every_toggle() -> None:
    assert coverage_violations(discover_toggles(), load_registry()) == []


def test_registry_covers_synthetic_stale_row() -> None:
    registry = {"GONE_ENABLED": {"name": "GONE_ENABLED", "kind": "safety_interlock", "default": True}}

    assert coverage_violations({}, registry) == [
        f"GONE_ENABLED: registry row names no existing toggle ({RULE_FILE})"
    ]


def test_registry_kind_is_valid() -> None:
    assert kind_violations(load_registry()) == []
    assert kind_violations({"X_ENABLED": {"kind": "rollout", "default": True}}) == [
        f"X_ENABLED: kind 'rollout' is not one of {sorted(KINDS)} ({RULE_FILE})"
    ]


def test_registry_default_matches_code() -> None:
    assert default_violations(discover_toggles(), load_registry()) == []


def test_component_defaults_on_and_not_overridden() -> None:
    off_component = {"X_ENABLED": {"kind": "component", "default": False}}
    on_component = {"X_ENABLED": {"kind": "component", "default": True}}

    assert component_violations(off_component) != []
    assert component_violations(on_component) == []
    assert baseline_violations(on_component, [("X_ENABLED", "true")]) != []


def test_experiment_requires_deadline_and_exit() -> None:
    base = {"name": "X_ENABLED", "kind": "experiment", "default": True, "reason": "r"}
    missing_exit = {**base, "exit_condition": "", "review_by": TODAY.isoformat()}
    missing_date = {**base, "exit_condition": "metric holds", "review_by": ""}
    stale = {**base, "exit_condition": "metric holds", "review_by": (TODAY - timedelta(days=15)).isoformat()}
    within_grace = {**base, "exit_condition": "metric holds", "review_by": (TODAY - timedelta(days=13)).isoformat()}

    assert experiment_violations({"X_ENABLED": missing_exit}, TODAY) != []
    assert experiment_violations({"X_ENABLED": missing_date}, TODAY) != []
    assert experiment_violations({"X_ENABLED": stale}, TODAY) != []
    assert experiment_violations({"X_ENABLED": within_grace}, TODAY) == []


def test_baseline_cannot_duplicate_defaults() -> None:
    registry = load_registry()

    assert baseline_violations(registry, VPS_SELECTORS) == []
    assert baseline_violations(registry, [("HISTORY_EVICTION_ENABLED", "true")]) != []
    env_binding = {"KIS_MODE": {"kind": "env_binding", "default": "mock"}}
    assert baseline_violations(env_binding, [("KIS_MODE", "mock")]) == []


def test_policy_detects_new_default_off_flag() -> None:
    class _Synthetic(BaseModel):
        NEW_FEATURE_ENABLED: bool = False

    toggles = discover_toggles(_Synthetic)

    assert toggles == {"NEW_FEATURE_ENABLED": False}
    assert coverage_violations(toggles, {}) == [
        f"NEW_FEATURE_ENABLED: toggle has no row in docs/flags.toml ({RULE_FILE})"
    ]
