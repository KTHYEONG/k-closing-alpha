"""Local-first runtime secret provisioning (workstation SSOT -> VPS shared env file).

로컬 워크스테이션 소스(기본 ~/.quant.env)에서 RUNTIME_ENV_SPEC 선언 키만
추출해 공유 런타임 fragment 를 빌드하고, SSH stdin 으로만 VPS에 원자 설치한다.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import ValidationError

from src.utils.cli_logging import configure_cli_logging

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
    optional: bool = False


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
    RuntimeEnvKey(target="KIWOOM_APP_KEY", sources=("KIWOOM_APP_KEY", "KIWOM_APP_KEY")),
    RuntimeEnvKey(target="KIWOOM_SECRET_KEY", sources=("KIWOOM_SECRET_KEY", "KIWOM_SECRET_KEY")),
    RuntimeEnvKey(target="LS_APP_KEY", sources=("LS_APP_KEY",)),
    RuntimeEnvKey(target="LS_APP_SECRET", sources=("LS_APP_SECRET",)),
    RuntimeEnvKey(target="KRX_OPENAPI_KEY", sources=("KRX_OPENAPI_KEY",)),
    RuntimeEnvKey(target="TOSS_APP_KEY", sources=("TOSS_APP_KEY",)),
    RuntimeEnvKey(target="TOSS_APP_SECRET", sources=("TOSS_APP_SECRET",)),
    RuntimeEnvKey(target="OPENDART_API_KEY", sources=("OPENDART_API_KEY",)),
    RuntimeEnvKey(target="OPENDART_API_KEY_2", sources=("OPENDART_API_KEY_2",), optional=True),
)

# VPS 전용 선택자. 워크스테이션 SSOT(~/.quant.env)에는 없고 호스트 운영 배선에만 의미가 있어
# 코드가 단일 원천이다. VPS 파일에 손으로만 적어 두면 재프로비저닝이 조용히 지워
# 수집 잡이 `disabled`로 침묵 종료한다(실측: 2026-09-24 auction/altdata 비활성화).
# 값은 쉼표 목록으로 쓴다: docker --env-file은 따옴표를 그대로 넘기고 systemd
# EnvironmentFile은 벗기므로 JSON 표기는 로더마다 다르게 읽힌다.
#
# KIS 데이터 슬롯(앱키당 REST 20TPS)은 krx-alpha와 공유한다. 같은 키를 두 프로젝트가 동시에
# 쓰면 합산 TPS가 한도를 넘는다(실측: krx 스냅샷이 DATA_1을 쓰던 09-22/23 15:20에 kca 결정창
# EGW00201 44/77건). kca 프로세스끼리는 호스트 공유 버킷으로 키별 합산이 제한되므로, 배정은
# "krx와 시간대가 겹치는 슬롯을 공유하지 않는다"만 지키면 된다.
#   DATA_1  kca 결정 역할(15:20 수집 선두·15:30 종가확정·페이퍼) 전용 REST
#   DATA_2  krx 스냅샷 REST 08:00~15:39 전용(krx docker-compose가 고정) -> kca는 장중 사용 금지
#   DATA_3  kca 장 개시·마감 연구 캡처(08:39~09:03, 15:21~15:30)
#   DATA_4  kca 배치 역할(가격·아카이브·대체데이터 기본키) + 연구 캡처 보조
#   DATA_5  kca 결정창 샤드 2번(15:20), krx 예비
# krx 애프터마켓 웹소켓 샤드(15:30~20:00)는 REST 유량을 쓰지 않아 위 배정과 충돌하지 않는다.
#   KIS_DECISION_SHARD_SLOTS: 결정 역할 선두(1)로 시작해야 하며 15:20 벌크를 절반으로 줄인다.
#   COLLECTION_RESEARCH_SLOTS: 샤드와 겹치면 안 되고(코드 검증), krx 스냅샷(2)과 경매 시각이 같아 2도 제외.
#   COLLECTION_ALTDATA_EXTRA_SLOTS: 21:35에는 krx가 REST를 쓰지 않으므로 2·3을 빌려 3키로 분산한다.
VPS_SELECTORS: tuple[tuple[str, str], ...] = (
    ("KIS_DECISION_SHARD_SLOTS", "1,5"),
    ("COLLECTION_AUCTION_ENABLED", "true"),
    ("COLLECTION_ALTDATA_ENABLED", "true"),
    ("COLLECTION_RESEARCH_SLOTS", "3,4"),
    ("COLLECTION_ALTDATA_EXTRA_SLOTS", "2,3"),
)

# krx-alpha docker-compose의 KRX_ALPHA_SNAPSHOT_KIS_DATA_SLOT과 같아야 한다(장중 배타 슬롯).
KRX_SNAPSHOT_DATA_SLOT: str = "2"

REMOTE_RUNTIME_ENV_PATH: str = "/home/ubuntu/quant-secrets/k-closing-alpha.env"
# kca 컨테이너는 이 파일을 k-closing-alpha.env 다음에 --env-file로 함께 받는다(데이터 슬롯 풀).
REMOTE_KIS_DATA_ENV_PATH: str = "/home/ubuntu/quant-secrets/kis-data.env"
REMOTE_IMAGE: str = "ghcr.io/kthyeong/k-closing-alpha:latest"
_SECTION_SENTINEL: str = "<<KCA-PROVISION-SECTION>>"

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
            if key.optional:
                continue
            raise ProvisioningError(f"missing required key: {key.target}")
        lines.append(f"{key.target}={value}")
    return "\n".join(lines) + "\n"


# 개명되어 제거된 런타임 키. 관리 대상에서 빠지면 원격 파일의 낡은 라인이
# 영원히 외부 키로 취급되어 보존되므로, 명시적으로 관리 집합에 포함해 삭제한다.
RETIRED_RUNTIME_KEYS: tuple[str, ...] = ("KIWOM_APP_KEY", "KIWOM_SECRET_KEY")


def _managed_keys() -> frozenset[str]:
    return (
        frozenset(key.target for key in RUNTIME_ENV_SPEC)
        | frozenset(name for name, _ in VPS_SELECTORS)
        | frozenset(RETIRED_RUNTIME_KEYS)
    )


def merge_remote_env(fragment: str, remote_text: str) -> tuple[str, tuple[str, ...]]:
    """Compose the VPS env file: managed keys first, then untouched foreign keys.

    Provisioning owns ``RUNTIME_ENV_SPEC`` targets and ``VPS_SELECTORS``; any other
    ``KEY=VALUE`` already on the host is operator state this tool must not erase.
    A managed key that is no longer produced (for example an optional key removed
    from the workstation) is dropped, not preserved.

    Args:
        fragment: Canonical fragment from ``build_runtime_fragment``.
        remote_text: Current contents of the remote file ("" when absent).

    Returns:
        The full file text and the names (never values) of the preserved keys.
    """
    managed = _managed_keys()
    preserved: dict[str, str] = {}
    for line in remote_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name = stripped.partition("=")[0].strip()
        if name in managed:
            continue
        preserved[name] = stripped
    selector_lines = "".join(f"{name}={value}\n" for name, value in VPS_SELECTORS)
    foreign_lines = "".join(f"{line}\n" for line in preserved.values())
    return fragment + selector_lines + foreign_lines, tuple(preserved)


@dataclass(frozen=True)
class RemoteState:
    """Host facts read before any write.

    Attributes:
        env_text: Current runtime env file ("" when absent).
        kis_data_text: Shared KIS data-slot env file ("" when absent).
        image_commit: ``KCA_CODE_COMMIT`` of the image the timers run ("" when unknown).
    """

    env_text: str
    kis_data_text: str
    image_commit: str


REMOTE_READ_SCRIPT: str = f"""set -euo pipefail
emit() {{ if [ -f "$1" ]; then cat "$1"; fi; }}
emit "{REMOTE_RUNTIME_ENV_PATH}"
printf '\\n{_SECTION_SENTINEL}\\n'
emit "{REMOTE_KIS_DATA_ENV_PATH}"
printf '\\n{_SECTION_SENTINEL}\\n'
docker image inspect "{REMOTE_IMAGE}" --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}' | sed -n 's/^KCA_CODE_COMMIT=//p'
"""


def read_remote_state(host: str) -> RemoteState:
    """Read the env files and the deployed image commit in one SSH round trip.

    Args:
        host: SSH host alias.

    Returns:
        Current host state.

    Raises:
        ProvisioningError: The host cannot be read; installing blind is refused.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["ssh", host, f"bash -c {shlex.quote(REMOTE_READ_SCRIPT)}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ProvisioningError(f"cannot read remote state on {host} ({type(exc).__name__}); refusing blind install") from None
    sections = result.stdout.split(f"\n{_SECTION_SENTINEL}\n")
    if len(sections) != 3:
        raise ProvisioningError(f"unexpected remote state layout on {host}")
    return RemoteState(env_text=sections[0], kis_data_text=sections[1], image_commit=sections[2].strip())


def _parse_env(text: str) -> dict[str, str]:
    return {k: v for k, v in dotenv_values(stream=io.StringIO(text)).items() if v is not None}


def _hermetic_collection_settings() -> type[Any]:
    from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

    from src.config.collection import CollectionSettings

    class _HostCollectionSettings(CollectionSettings):
        # 검증은 대상 파일만 반영해야 한다: 워크스테이션 환경변수·.env가 섞이면 결과가 달라진다.
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],  # noqa: ARG003 - fixed pydantic-settings hook signature
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,  # noqa: ARG003
            dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
            file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (init_settings,)

    return _HostCollectionSettings


def validate_runtime_env(env_text: str, kis_data_text: str) -> None:
    """Prove the file about to be installed boots every kca job, using the jobs' own parsers.

    A syntactically valid file can still disable collection silently (a missing
    switch reads as ``False``) or break slot resolution only at 15:20. Each
    check below is the exact resolver a scheduled job calls, fed with the two
    files in the same order docker passes them.

    Args:
        env_text: Candidate ``k-closing-alpha.env`` contents.
        kis_data_text: Current ``kis-data.env`` contents.

    Raises:
        ProvisioningError: Any job would fail or run degraded; the message
            carries field and slot names only, never values.
    """
    from src.api.kis.key_pool import (
        resolve_decision_shard_credentials,
        resolve_host_issued_credentials,
        resolve_research_credentials,
    )
    from src.backfill.altdata.dart_keys import resolve_dart_key_pool

    env: dict[str, str] = {**_parse_env(env_text), **_parse_env(kis_data_text)}
    collection_values = {k: v for k, v in env.items() if k.startswith("COLLECTION_")}
    try:
        profile = _hermetic_collection_settings()(**collection_values)
    except ValidationError as exc:
        fields = ",".join(".".join(str(p) for p in err["loc"]) or err["msg"] for err in exc.errors())
        raise ProvisioningError(f"collection settings invalid: {fields}") from None
    declared = dict(VPS_SELECTORS)
    for flag in ("COLLECTION_AUCTION_ENABLED", "COLLECTION_ALTDATA_ENABLED"):
        if flag in declared and bool(getattr(profile, flag)) != (declared[flag].lower() == "true"):
            raise ProvisioningError(f"{flag} does not resolve to its declared value")
    try:
        resolve_host_issued_credentials(env)
        resolve_decision_shard_credentials(env)
        if profile.COLLECTION_AUCTION_ENABLED:
            resolve_research_credentials(env, slots=tuple(profile.COLLECTION_RESEARCH_SLOTS))
        if profile.COLLECTION_ALTDATA_EXTRA_SLOTS:
            resolve_research_credentials(env, slots=tuple(profile.COLLECTION_ALTDATA_EXTRA_SLOTS))
    except ValueError as exc:
        raise ProvisioningError(f"KIS slot resolution fails: {exc}") from None
    if profile.COLLECTION_ALTDATA_ENABLED and resolve_dart_key_pool(
        primary=env.get("OPENDART_API_KEY", ""), secondary=env.get("OPENDART_API_KEY_2", "")
    ).is_empty():
        raise ProvisioningError("altdata enabled without any OpenDART key")


def diff_env_names(before_text: str, after_text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (added, removed, changed) key names between two env files; values never leave."""
    before = _parse_env(before_text)
    after = _parse_env(after_text)
    added = tuple(sorted(set(after) - set(before)))
    removed = tuple(sorted(set(before) - set(after)))
    changed = tuple(sorted(k for k in set(before) & set(after) if before[k] != after[k]))
    return added, removed, changed


def local_code_commit(repo_dir: Path) -> str:
    """Commit whose parsers validate the file; "" when unknown or ``src`` has uncommitted edits.

    Uncommitted parser edits would validate a spelling the deployed image may
    not read, so a dirty ``src`` never counts as matching any deployed commit.
    """
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_dir), "status", "--porcelain", "--", "src"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""
    return "" if dirty else head


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
    """Build, merge, validate and (unless dry-run) install the VPS runtime env.

    Every run reads the host first. Installation is refused when the merged
    file fails the jobs' own parsers, when a key would disappear without
    ``--allow-remove``, or when the deployed image is not the commit whose
    parsers validated the file (a newer spelling could be unreadable by an
    older image) without ``--allow-version-skew``.
    """
    parser = argparse.ArgumentParser(description="Provision shared runtime env file to VPS")
    parser.add_argument("--host", default="or-vps")
    parser.add_argument("--source", default=str(Path.home() / ".quant.env"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-remove", action="store_true")
    parser.add_argument("--allow-version-skew", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.source)
    fragment = build_runtime_fragment(source)
    state = read_remote_state(args.host)
    merged, preserved = merge_remote_env(fragment, state.env_text)
    validate_runtime_env(merged, state.kis_data_text)
    added, removed, changed = diff_env_names(state.env_text, merged)
    logger.info(
        "[SYS] stage=provision_env host=%s keys=%d added=%s removed=%s changed=%s preserved=%s",
        args.host,
        len(merged.splitlines()),
        ",".join(added) or "-",
        ",".join(removed) or "-",
        ",".join(changed) or "-",
        ",".join(preserved) or "-",
    )
    if removed and not args.allow_remove:
        raise ProvisioningError(f"refusing to remove keys without --allow-remove: {','.join(removed)}")
    local_commit = local_code_commit(Path(__file__).resolve().parents[2])
    if (not local_commit or local_commit != state.image_commit) and not args.allow_version_skew:
        raise ProvisioningError(
            f"deployed image {state.image_commit[:12] or 'unknown'} != validating code {local_commit[:12] or 'unknown'}; "
            "deploy first or pass --allow-version-skew"
        )
    if args.dry_run:
        return 0
    install_runtime_fragment(args.host, merged)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    raise SystemExit(main())
