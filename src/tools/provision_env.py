"""Local-first runtime secret provisioning (workstation SSOT -> VPS shared env file).

로컬 워크스테이션 소스(기본 ~/.quant.env)에서 RUNTIME_ENV_SPEC 선언 키만
추출해 공유 런타임 fragment 를 빌드하고, SSH stdin 으로만 VPS에 원자 설치한다.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ProvisioningError(RuntimeError):
    """Fail-closed provisioning failure (missing key, duplicate, unreadable source)."""


@dataclass(frozen=True)
class RuntimeEnvKey:
    target: str
    sources: tuple[str, ...]


RUNTIME_ENV_SPEC: tuple[RuntimeEnvKey, ...] = (
    RuntimeEnvKey(target="ALERT_GMAIL_USER", sources=("ALERT_GMAIL_USER", "LIVE_ALERT_GMAIL_USER")),
    RuntimeEnvKey(
        target="ALERT_GMAIL_APP_PASSWORD", sources=("ALERT_GMAIL_APP_PASSWORD", "LIVE_ALERT_GMAIL_APP_PASSWORD")
    ),
    RuntimeEnvKey(target="ALERT_GMAIL_TO", sources=("ALERT_GMAIL_TO",)),
    RuntimeEnvKey(target="KIS_APP_KEY", sources=("KIS_APP_KEY",)),
    RuntimeEnvKey(target="KIS_APP_SECRET", sources=("KIS_APP_SECRET",)),
    RuntimeEnvKey(target="KIS_ACCOUNT_ID", sources=("KIS_ACCOUNT_ID", "KIS_ACCOUNT_NO")),
    RuntimeEnvKey(target="KIS_HTS_ID", sources=("KIS_HTS_ID",)),
    RuntimeEnvKey(target="KIS_DATA_APP_KEY", sources=("KIS_DATA_APP_KEY",)),
    RuntimeEnvKey(target="KIS_DATA_APP_SECRET", sources=("KIS_DATA_APP_SECRET",)),
    RuntimeEnvKey(target="KIS_DATA_HTS_ID", sources=("KIS_DATA_HTS_ID",)),
    RuntimeEnvKey(target="KIWOM_APP_KEY", sources=("KIWOM_APP_KEY",)),
    RuntimeEnvKey(target="KIWOM_SECRET_KEY", sources=("KIWOM_SECRET_KEY",)),
    RuntimeEnvKey(target="LS_APP_KEY", sources=("LS_APP_KEY",)),
    RuntimeEnvKey(target="LS_APP_SECRET", sources=("LS_APP_SECRET",)),
    RuntimeEnvKey(target="KRX_OPENAPI_KEY", sources=("KRX_OPENAPI_KEY",)),
    RuntimeEnvKey(target="TOSS_APP_KEY", sources=("TOSS_APP_KEY",)),
    RuntimeEnvKey(target="TOSS_APP_SECRET", sources=("TOSS_APP_SECRET",)),
    RuntimeEnvKey(target="OPENDART_API_KEY", sources=("OPENDART_API_KEY",)),
)

REMOTE_RUNTIME_ENV_PATH: str = "/home/ubuntu/quant-secrets/k-closing-alpha.env"

REMOTE_RUNTIME_INSTALL_SCRIPT: str = f"""set -euo pipefail
d="$(dirname "{REMOTE_RUNTIME_ENV_PATH}")"
mkdir -p "$d"
chmod 0700 "$d"
tmp="$(mktemp "$d/.k-closing-alpha.env.XXXXXX")"
cat > "$tmp"
chmod 600 "$tmp"
chown ubuntu:ubuntu "$tmp"
mv -f "$tmp" "{REMOTE_RUNTIME_ENV_PATH}"
chmod 600 "{REMOTE_RUNTIME_ENV_PATH}"
chown ubuntu:ubuntu "{REMOTE_RUNTIME_ENV_PATH}"
"""


def parse_workstation_assignments(source_path: Path, accepted_keys: frozenset[str]) -> dict[str, str]:
    """Parse shell-style KEY=VALUE assignments, collecting only accepted keys."""
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProvisioningError(f"cannot read source: {source_path}") from exc
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in accepted_keys:
            continue
        if not value:
            continue
        if key in parsed:
            raise ProvisioningError(f"duplicate assignment: {key}")
        parsed[key] = value
    return parsed


def build_runtime_fragment(source_path: Path) -> str:
    """Build canonical TARGET=VALUE fragment in RUNTIME_ENV_SPEC order."""
    accepted: frozenset[str] = frozenset(source for key in RUNTIME_ENV_SPEC for source in key.sources)
    assignments = parse_workstation_assignments(source_path, accepted)
    lines: list[str] = []
    for key in RUNTIME_ENV_SPEC:
        value: str | None = None
        for source in key.sources:
            candidate = assignments.get(source)
            if candidate:
                value = candidate
                break
        if not value:
            raise ProvisioningError(f"missing required key: {key.target}")
        lines.append(f"{key.target}={value}")
    return "\n".join(lines) + "\n"


def install_runtime_fragment(host: str, fragment: str) -> None:
    """Install fragment to REMOTE_RUNTIME_ENV_PATH over SSH (stdin only, atomic)."""
    subprocess.run(  # noqa: S603
        ["ssh", host, "bash", "-c", REMOTE_RUNTIME_INSTALL_SCRIPT],  # noqa: S607
        input=fragment,
        text=True,
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: build fragment, optionally install to host."""
    parser = argparse.ArgumentParser(description="Provision shared runtime env file to VPS")
    parser.add_argument("--host", default="or-vps")
    parser.add_argument("--source", default=str(Path.home() / ".quant.env"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.source)
    fragment = build_runtime_fragment(source)
    logger.info("provisioned %d keys to %s from %s", len(RUNTIME_ENV_SPEC), args.host, source)
    if args.dry_run:
        return 0
    install_runtime_fragment(args.host, fragment)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
