"""Local-first runtime secret provisioning (workstation SSOT -> VPS shared env file).

로컬 워크스테이션 소스(기본 ~/.quant.env)에서 RUNTIME_ENV_SPEC 선언 키만
추출해 공유 런타임 fragment 를 빌드하고, SSH stdin 으로만 VPS에 원자 설치한다.
"""

from __future__ import annotations

import argparse
import logging
import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ~/.quant.env 는 소싱되는 bash 스크립트라 "KIS_APP_KEY=$KIS_TRADE_APP_KEY" 같은
# 셸 변수 참조가 정상 문법이다(실측: 2026-09-16 프로덕션 장애 — krx-alpha 에서 이
# 리터럴 텍스트를 그대로 배포해 자격증명이 빈 문자열로 주입됨. 동일 SSOT 라인을
# 참조하는 본 저장소도 같은 결함을 갖고 있었다). 값 전체가 단일 참조일 때만 같은
# 파일 내 다른 할당을 조회해 해석한다.
_VAR_REF_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


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
    """Parse shell-style KEY=VALUE assignments, collecting only accepted keys.

    A value that is exactly a bare ``$VAR``/``${VAR}`` reference is resolved
    against other assignments in the same file (mirroring bash ``source``
    semantics); an unresolvable reference resolves to empty and is therefore
    treated as absent, so callers fail closed on the true value being missing
    rather than silently shipping the literal reference text.
    """
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProvisioningError(f"cannot read source: {source_path}") from exc
    raw: dict[str, str] = {}
    accepted_seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key in accepted_keys:
            if key in accepted_seen:
                raise ProvisioningError(f"duplicate assignment: {key}")
            accepted_seen.add(key)
        raw[key] = value  # 비허용 키의 재할당은 정상 bash 문법이라 마지막 값이 우선한다

    def _resolve(key: str, chain: frozenset[str]) -> str:
        value = raw.get(key, "")
        match = _VAR_REF_RE.match(value)
        if not match:
            return value
        ref = match.group(1)
        if ref in chain:
            raise ProvisioningError(f"circular variable reference resolving {key}")
        return _resolve(ref, chain | {ref})

    parsed: dict[str, str] = {}
    for key in accepted_seen:
        resolved = _resolve(key, frozenset({key}))
        if resolved:
            parsed[key] = resolved
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
    """Install fragment to REMOTE_RUNTIME_ENV_PATH over SSH (stdin only, atomic).

    ssh joins argv[2:] with spaces before the remote login shell sees it, so
    passing ("bash", "-c", SCRIPT) as separate elements lets the newline-
    bearing SCRIPT get re-split: -c only receives the first word ("set"),
    and the remaining lines run in the outer login shell. That stray
    ``bash -c set`` dumps the whole shell environment to stdout (measured).
    ``shlex.quote`` collapses the script into one argv element so the
    remote shell parses it as exactly ``bash -c <SCRIPT>``.
    """
    subprocess.run(  # noqa: S603
        ["ssh", host, f"bash -c {shlex.quote(REMOTE_RUNTIME_INSTALL_SCRIPT)}"],  # noqa: S607
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
