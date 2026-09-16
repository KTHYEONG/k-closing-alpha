"""KIS data key pool with fixed key-to-host assignment and once-per-day issuance.

Each app key has exactly one host that may issue its token (INV-KEY-HOST).
That host keeps a shared per-key token cache file and a fixed-time warmup
timer issues one token per data key per day (INV-TOKEN-1PD). Jobs only read
the cache and issue under an exclusive file lock as a fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

KIS_DATA_ROLE_DECISION = "decision"
KIS_DATA_ROLE_BATCH = "batch"

_SLOT_RE = re.compile(r"^[1-9][0-9]*$")


@dataclass(frozen=True)
class KisCredential:
    slot: str
    app_key: str
    app_secret: str
    hts_id: str


@dataclass(frozen=True)
class HostKeySpec:
    """A non-pool KIS app key this host must issue once per day.

    Attributes:
        name: Slot label surfaced in warmup results, logs and audit output.
        app_key_var: Env var holding the app key.
        app_secret_var: Env var holding the app secret.
        hts_id_var: Env var holding the HTS id.
    """

    name: str
    app_key_var: str
    app_secret_var: str
    hts_id_var: str


# 데이터 슬롯 풀 밖에서 이 호스트가 발급 책임을 지는 키 선언(단일 원천).
# PRIMARY(KIS_APP_KEY)는 체결 경로와 외부 실시간 세션이 함께 쓰는 계정 키다.
# 여기에 선언되지 않은 키는 아무도 발급하지 않으므로, 읽기 전용 소비자는
# 만료 시점에 fail-closed로 멈춘다 -- 발급 책임은 선언으로만 부여한다.
HOST_ISSUED_KEY_SPECS: tuple[HostKeySpec, ...] = (
    HostKeySpec(
        name="PRIMARY",
        app_key_var="KIS_APP_KEY",
        app_secret_var="KIS_APP_SECRET",  # noqa: S106 - env var name, not a hardcoded secret
        hts_id_var="KIS_HTS_ID",
    ),
)


def kis_key_id(app_key: str) -> str:
    return hashlib.sha256(app_key.encode("utf-8")).hexdigest()[:12]


def token_cache_path(app_key: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"token_{kis_key_id(app_key)}.json"


def read_token_issued_date(token_file: Path) -> str | None:
    """토큰 캐시 파일의 issued_at 날짜(YYYY-MM-DD)를 읽는다. 없거나 손상되면 None.

    조회 전용 헬퍼이므로 읽기 실패를 예외로 올리지 않고 None으로 낮춘다
    (client._read_cache_payload와 동일한 fail-open 정책).
    """
    if not token_file.exists():
        return None
    try:
        payload = json.loads(token_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    issued_at = str(payload.get("issued_at", ""))
    return issued_at[:10] or None


def load_kis_env(env_file: Path) -> dict[str, str]:
    if env_file.is_file():
        file_vals = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
        merged = dict(file_vals)
        merged.update(dict(os.environ))
        return merged
    return dict(os.environ)


def _parse_slot_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    tokens = [t.strip() for t in text.split(",")]
    tokens = [t for t in tokens if t]
    for t in tokens:
        if not _SLOT_RE.match(t):
            raise ValueError(f"invalid data slot token: {t}")
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate data slot")
    return tokens


def parse_host_data_credentials(env: Mapping[str, str]) -> tuple[KisCredential, ...]:
    pool = _parse_slot_list(env.get("KIS_DATA_SLOTS"))
    if pool is None:
        return ()
    host_raw = env.get("KIS_HOST_DATA_SLOTS")
    host = _parse_slot_list(host_raw)
    if host is None:
        raise ValueError("KIS_HOST_DATA_SLOTS is required when KIS_DATA_SLOTS is set")
    pool_set = set(pool)
    for n in host:
        if n not in pool_set:
            raise ValueError(f"host slot DATA_{n} is not in KIS_DATA_SLOTS pool")
    creds: list[KisCredential] = []
    for n in host:
        app_key = (env.get(f"KIS_DATA_{n}_APP_KEY") or "").strip()
        app_secret = (env.get(f"KIS_DATA_{n}_APP_SECRET") or "").strip()
        hts_id = (env.get(f"KIS_DATA_{n}_HTS_ID", "") or "").strip()
        if not app_key or not app_secret:
            raise ValueError(f"missing credentials for slot DATA_{n}")
        creds.append(KisCredential(slot=f"DATA_{n}", app_key=app_key, app_secret=app_secret, hts_id=hts_id))
    seen: set[str] = set()
    for c in creds:
        if c.app_key in seen:
            raise ValueError(f"duplicate app_key in slot {c.slot}")
        seen.add(c.app_key)
    for trade_var in ("KIS_TRADE_APP_KEY", "KIS_APP_KEY"):
        trade_key = (env.get(trade_var) or "").strip()
        if trade_key and trade_key in seen:
            raise ValueError(f"data slot app_key collides with {trade_var}")
    return tuple(creds)


def resolve_host_data_credentials(env: Mapping[str, str]) -> tuple[KisCredential, ...]:
    """Resolve this host's assigned KIS data slot credentials.

    Args:
        env: Credential environment mapping.

    Returns:
        Credentials in KIS_HOST_DATA_SLOTS order.

    Raises:
        ValueError: The key pool is not declared in env. 선언된 풀 밖의 단일 키로
            묵시적으로 대체하면 자격증명 배선 누락이 기동 시점에 드러나지 않고
            결정창(15:20~15:30)의 벤더 인증 실패로 지연 표면화된다.
    """
    parsed = parse_host_data_credentials(env)
    if not parsed:
        raise ValueError(
            "KIS host data slots are not configured; expected KIS_DATA_SLOTS and KIS_HOST_DATA_SLOTS in the credential env"
        )
    return parsed


def parse_host_extra_credentials(env: Mapping[str, str]) -> tuple[KisCredential, ...]:
    """Resolve the declared non-pool keys this host issues daily.

    Args:
        env: Credential environment mapping.

    Returns:
        Credentials in HOST_ISSUED_KEY_SPECS order.

    Raises:
        ValueError: A declared key's app key or secret is missing or blank.
    """
    creds: list[KisCredential] = []
    for spec in HOST_ISSUED_KEY_SPECS:
        app_key = (env.get(spec.app_key_var) or "").strip()
        app_secret = (env.get(spec.app_secret_var) or "").strip()
        if not app_key or not app_secret:
            raise ValueError(f"missing credentials for host key {spec.name} ({spec.app_key_var})")
        creds.append(
            KisCredential(
                slot=spec.name,
                app_key=app_key,
                app_secret=app_secret,
                hts_id=(env.get(spec.hts_id_var) or "").strip(),
            )
        )
    return tuple(creds)


def resolve_host_issued_credentials(env: Mapping[str, str]) -> tuple[KisCredential, ...]:
    """Every credential this host must issue exactly once per day.

    Args:
        env: Credential environment mapping.

    Returns:
        Host-assigned data slot credentials followed by the declared non-pool keys.

    Raises:
        ValueError: 풀 미선언, host 슬롯 자격증명 누락, 또는 결정창 샤드 슬롯 자격증명 누락/중복.
    """
    host = resolve_host_data_credentials(env)
    seen_keys = {c.app_key for c in host}
    shard_extra = tuple(c for c in parse_decision_shard_credentials(env) if c.app_key not in seen_keys)
    return host + shard_extra + parse_host_extra_credentials(env)


def select_data_credential(credentials: tuple[KisCredential, ...], role: str) -> KisCredential:
    if role not in (KIS_DATA_ROLE_DECISION, KIS_DATA_ROLE_BATCH):
        raise ValueError(f"unknown data role: {role}")
    if not credentials:
        raise ValueError("no data credentials available")
    if role == KIS_DATA_ROLE_DECISION:
        return credentials[0]
    return credentials[-1]


def parse_decision_shard_credentials(env: Mapping[str, str]) -> tuple[KisCredential, ...]:
    """15:20 결정창 벌크 수집을 나눌 추가 슬롯을 KIS_DECISION_SHARD_SLOTS에서 읽는다.

    [STEP-BY-STEP RECIPE FOR IMPLEMENTER]:
    Step 1. `KIS_DATA_SLOTS`를 `_parse_slot_list`로 파싱해 pool을 얻는다. pool이 None이면
       빈 튜플 `()`을 반환한다(선언된 풀 자체가 없으면 이 기능은 조용히 꺼진다).
    Step 2. `env.get("KIS_DECISION_SHARD_SLOTS")`를 `_parse_slot_list`로 파싱한다. 결과가
       None이면(미설정) 빈 튜플 `()`을 반환한다 — 이것이 "기능 꺼짐" 기본값이다.
    Step 3. 파싱된 각 슬롯 번호 n이 pool(Step 1의 set)에 없으면
       `ValueError(f"decision shard slot DATA_{n} is not in KIS_DATA_SLOTS pool")`을 던진다.
    Step 4. 각 슬롯 n에 대해 `KIS_DATA_{n}_APP_KEY`/`KIS_DATA_{n}_APP_SECRET`/`KIS_DATA_{n}_HTS_ID`를
       읽어 `.strip()`한다. app_key 또는 app_secret이 비었으면
       `ValueError(f"missing credentials for slot DATA_{n}")`을 던진다(hts_id는 빈 문자열 허용,
       `parse_host_data_credentials`와 동일 관례).
    Step 5. `KisCredential(slot=f"DATA_{n}", app_key=..., app_secret=..., hts_id=...)`를
       Step 2에서 파싱된 순서 그대로 리스트에 쌓는다.
    Step 6. app_key 중복을 검사한다: 이미 본 app_key가 재등장하면
       `ValueError(f"duplicate app_key in decision shard slot {c.slot}")`을 던진다.
    Step 7. 튜플로 변환해 반환한다.
    """
    pool = _parse_slot_list(env.get("KIS_DATA_SLOTS"))
    if pool is None:
        return ()
    shards = _parse_slot_list(env.get("KIS_DECISION_SHARD_SLOTS"))
    if shards is None:
        return ()
    pool_set = set(pool)
    for n in shards:
        if n not in pool_set:
            raise ValueError(f"decision shard slot DATA_{n} is not in KIS_DATA_SLOTS pool")
    creds: list[KisCredential] = []
    for n in shards:
        app_key = (env.get(f"KIS_DATA_{n}_APP_KEY") or "").strip()
        app_secret = (env.get(f"KIS_DATA_{n}_APP_SECRET") or "").strip()
        hts_id = (env.get(f"KIS_DATA_{n}_HTS_ID", "") or "").strip()
        if not app_key or not app_secret:
            raise ValueError(f"missing credentials for slot DATA_{n}")
        creds.append(KisCredential(slot=f"DATA_{n}", app_key=app_key, app_secret=app_secret, hts_id=hts_id))
    seen: set[str] = set()
    for c in creds:
        if c.app_key in seen:
            raise ValueError(f"duplicate app_key in decision shard slot {c.slot}")
        seen.add(c.app_key)
    return tuple(creds)


def resolve_decision_shard_credentials(env: Mapping[str, str]) -> tuple[KisCredential, ...]:
    """collect.py가 실제로 사용할 결정창 샤드 자격증명 목록(1개 또는 N개)을 확정한다.

    [STEP-BY-STEP RECIPE FOR IMPLEMENTER]:
    Step 1. `parsed = parse_decision_shard_credentials(env)`를 호출한다.
    Step 2. `primary = select_data_credential(resolve_host_data_credentials(env), KIS_DATA_ROLE_DECISION)`을
       계산한다(기존 단일 키 경로가 실제로 고르는 자격증명 — 레거시 동작의 단일 진실 원천).
    Step 3. `parsed`가 빈 튜플이면(KIS_DECISION_SHARD_SLOTS 미설정) `(primary,)`를 반환한다.
       이것이 기능 꺼짐 상태의 레거시 폴백이며, `main()`이 이미 발급한 `client`를 그대로
       재사용할 수 있도록 슬롯 1개짜리 튜플이어야 한다.
    Step 4. `parsed`가 비어있지 않으면 fail-closed 불변식을 검사한다: `parsed[0].app_key`가
       `primary.app_key`와 달라야 한다면(즉 다르면) 아래 메시지로 즉시 예외를 던진다 —
       원문 앱키/시크릿은 로그/예외에 절대 노출하지 않고 `kis_key_id()` 지문만 사용한다.
    Step 5. 검사를 통과하면 `parsed`를 그대로 반환한다.
    """
    parsed = parse_decision_shard_credentials(env)
    primary = select_data_credential(resolve_host_data_credentials(env), KIS_DATA_ROLE_DECISION)
    if not parsed:
        return (primary,)
    if parsed[0].app_key != primary.app_key:
        raise ValueError(
            "KIS_DECISION_SHARD_SLOTS must lead with the same credential "
            f"select_data_credential(role=decision) resolves to "
            f"(expected key_id={kis_key_id(primary.app_key)}, "
            f"got slot={parsed[0].slot} key_id={kis_key_id(parsed[0].app_key)})"
        )
    return parsed
