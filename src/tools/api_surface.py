"""API surface inventory: which broker TRs/paths each project on the host calls, shipped as a manifest.

The VPS holds only kca sources (krx/kse run from images or data-only dirs), so the notice watcher reads this
manifest instead of scanning sibling repos. Regenerate locally with `--sync` after touching a broker client.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_RELPATH: Path = Path("deploy/shared/api_surface.json")
MANIFEST_SCHEMA_VERSION: int = 1
KCA_ROOT: Path = Path(__file__).resolve().parents[2]
SCANNED_SUFFIXES: frozenset[str] = frozenset({".py", ".toml"})
ANY_BROKER = "*"

BROKER_LS = "ls"
BROKER_KIS = "kis"
BROKER_KIWOOM = "kiwoom"
BROKER_DBSEC = "dbsec"
BROKER_TOSS = "toss"

_TR_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    BROKER_LS: (re.compile(r"\bt\d{4}\b"),),
    BROKER_KIWOOM: (re.compile(r"\bk[a-z]\d{5}\b"),),
    BROKER_KIS: (re.compile(r"\b[A-Z]{4,5}\d{5,9}[A-Z0-9]{0,2}\b"), re.compile(r"\bH0[A-Z]{3,5}\d\b")),
}
_PATH_PATTERN = re.compile(r"/(?:uapi|api|oauth2?|v\d)[A-Za-z0-9_/{}\-]*[A-Za-z0-9_}]")
_PATH_PREFIX_BROKER: Mapping[str, str] = {"/uapi/": BROKER_KIS, "/api/dostk/": BROKER_KIWOOM}
_PATH_HINT_ALIASES: Mapping[str, str] = {
    "ls": BROKER_LS,
    "kis": BROKER_KIS,
    "kiwoom": BROKER_KIWOOM,
    "db": BROKER_DBSEC,
    "dbsec": BROKER_DBSEC,
    "toss": BROKER_TOSS,
}


@dataclass(frozen=True)
class ProjectSpec:
    """One project's broker-facing sources, relative to the projects base dir.

    `extra_tokens` lists short realtime codes no regex can safely infer (e.g. LS `S3_`); sync fails if one
    is absent from the scanned sources so stale entries cannot linger.
    """

    name: str
    root: str
    scan: tuple[str, ...]
    extra_tokens: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


PROJECTS: tuple[ProjectSpec, ...] = (
    ProjectSpec("kca", "k-closing-alpha", ("src/api",)),
    ProjectSpec(
        "krx",
        "krx-alpha",
        ("src/brokers", "src/realtime/adapters", "src/marketdata/toss_calendar.py"),
        {BROKER_LS: ("S3_", "K3_", "H1_", "HA_"), BROKER_DBSEC: ("S00", "S01")},
    ),
    ProjectSpec("kse", "k-stock-engine", ("src/integrations", "src/data", "config/providers.toml")),
)


@dataclass(frozen=True)
class TokenEntry:
    brokers: frozenset[str]
    files: tuple[str, ...]


@dataclass(frozen=True)
class Surface:
    """Loaded manifest: project -> token -> usage; Toss paths retain their case."""

    generated_at: str
    projects: Mapping[str, Mapping[str, TokenEntry]]

    def used(self, broker: str) -> dict[str, frozenset[str]]:
        """Tokens per project that can belong to `broker` (broker-attributed plus unattributed ones)."""
        return {
            project: frozenset(t for t, e in tokens.items() if not e.brokers or broker in e.brokers)
            for project, tokens in self.projects.items()
        }

    def attributed(self, broker: str) -> frozenset[str]:
        """Tokens explicitly attributed to `broker` across projects (excludes unattributed ones)."""
        return frozenset(t for tokens in self.projects.values() for t, e in tokens.items() if broker in e.brokers)

    def projects_using(self, broker: str) -> tuple[str, ...]:
        """Projects with at least one token attributed to `broker` (affected by broker-wide notices)."""
        return tuple(sorted(p for p, tokens in self.projects.items() if any(broker in e.brokers for e in tokens.values())))

    def files(self, project: str, token: str) -> tuple[str, ...]:
        entry = self.projects.get(project, {}).get(token)
        return entry.files if entry else ()

    def age_days(self, today: datetime) -> int:
        return (today - datetime.fromisoformat(self.generated_at)).days


def _hint_brokers(rel: Path) -> frozenset[str]:
    parts = {piece for part in rel.parts for piece in re.split(r"[_.]", part.lower())}
    return frozenset(broker for alias, broker in _PATH_HINT_ALIASES.items() if alias in parts)


def _classify(token: str, tr_broker: str | None, hint: frozenset[str]) -> frozenset[str]:
    if tr_broker is not None:
        return frozenset({tr_broker})
    for prefix, broker in _PATH_PREFIX_BROKER.items():
        if token.startswith(prefix):
            return frozenset({broker})
    return hint


def scan_project(spec: ProjectSpec, base: Path) -> dict[str, TokenEntry]:
    """Scan one project's declared sources; raises FileNotFoundError if the root or a scan path is missing."""
    root = base / spec.root
    found: dict[str, tuple[set[str], set[str]]] = {}
    texts: list[str] = []
    for scan_path in spec.scan:
        target = root / scan_path
        if not target.exists():
            raise FileNotFoundError(f"{spec.name}: scan path missing: {target}")
        files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.suffix in SCANNED_SUFFIXES)
        for source_file in files:
            text = source_file.read_text(encoding="utf-8")
            texts.append(text)
            rel = source_file.relative_to(root)
            hint = _hint_brokers(rel)
            hits: list[tuple[str, str | None]] = [
                (m.lower(), broker) for broker, pats in _TR_PATTERNS.items() for p in pats for m in p.findall(text)
            ]
            hits += [
                (m if BROKER_TOSS in hint else m.lower(), None)
                for m in _PATH_PATTERN.findall(text) if m.count("/") >= 2
            ]
            for token, tr_broker in hits:
                brokers, where = found.setdefault(token, (set(), set()))
                brokers.update(_classify(token, tr_broker, hint))
                where.add(rel.as_posix())
    joined = "\n".join(texts)
    for broker, extras in spec.extra_tokens.items():
        for extra in extras:
            if extra not in joined:
                raise ValueError(f"{spec.name}: extra token {extra!r} not found in scanned sources")
            brokers, where = found.setdefault(extra.lower(), (set(), set()))
            brokers.add(broker)
            where.update(rel_path for rel_path in _files_containing(root, spec.scan, extra))
    return {t: TokenEntry(frozenset(b), tuple(sorted(w))) for t, (b, w) in sorted(found.items())}


def _files_containing(root: Path, scan: Sequence[str], needle: str) -> list[str]:
    out: list[str] = []
    for scan_path in scan:
        target = root / scan_path
        files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.suffix in SCANNED_SUFFIXES)
        out.extend(p.relative_to(root).as_posix() for p in files if needle in p.read_text(encoding="utf-8"))
    return out


def build_surface(base: Path, specs: Sequence[ProjectSpec] = PROJECTS, *, now: datetime | None = None) -> Surface:
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    return Surface(stamp, {spec.name: scan_project(spec, base) for spec in specs})


def surface_to_json(surface: Surface) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": surface.generated_at,
        "projects": {
            project: {
                token: {"brokers": sorted(entry.brokers), "files": list(entry.files)} for token, entry in tokens.items()
            }
            for project, tokens in surface.projects.items()
        },
    }


def load_surface(path: Path = KCA_ROOT / MANIFEST_RELPATH) -> Surface:
    """Load the manifest; raises OSError when missing/invalid (never silently treat as 'nothing used')."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError("schema_version")
        projects = {
            project: {
                token: TokenEntry(frozenset(item["brokers"]), tuple(item["files"])) for token, item in tokens.items()
            }
            for project, tokens in raw["projects"].items()
        }
        return Surface(str(raw["generated_at"]), projects)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise OSError(f"Unreadable api surface manifest: {path}") from exc


def _comparable(surface: Surface) -> dict[str, Any]:
    payload = surface_to_json(surface)
    payload.pop("generated_at")
    return payload


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI; logic covered via build/load tests
    parser = argparse.ArgumentParser(description="Regenerate or verify the cross-project API surface manifest")
    parser.add_argument("--base", type=Path, default=Path.home(), help="directory holding the project checkouts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sync", action="store_true", help="rewrite the manifest from the local checkouts")
    group.add_argument("--check", action="store_true", help="exit 1 when the manifest is stale vs the checkouts")
    args = parser.parse_args(argv)
    fresh = build_surface(args.base)
    target = KCA_ROOT / MANIFEST_RELPATH
    if args.sync:
        target.write_text(json.dumps(surface_to_json(fresh), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        sys.stdout.write(f"wrote {target} ({', '.join(f'{p}={len(t)}' for p, t in fresh.projects.items())})\n")
        return 0
    stale = _comparable(load_surface(target)) != _comparable(fresh)
    sys.stdout.write("STALE: run --sync and commit\n" if stale else "OK\n")
    return 1 if stale else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
