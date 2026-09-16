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

from src import settings

KIS_DATA_ROLE_DECISION = "decision"
KIS_DATA_ROLE_BATCH = "batch"

_SLOT_RE = re.compile(r"^[1-9][0-9]*$")


@dataclass(frozen=True)
class KisCredential:
    slot: str
    app_key: str
    app_secret: str
    hts_id: str


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
    parsed = parse_host_data_credentials(env)
    if parsed:
        return parsed
    cfg = settings.KIS_DATA_API_CONFIG
    return (
        KisCredential(
            slot="DATA",
            app_key=cfg.get("app_key") or "",
            app_secret=cfg.get("app_secret") or "",
            hts_id=cfg.get("hts_id") or "",
        ),
    )


def select_data_credential(credentials: tuple[KisCredential, ...], role: str) -> KisCredential:
    if role not in (KIS_DATA_ROLE_DECISION, KIS_DATA_ROLE_BATCH):
        raise ValueError(f"unknown data role: {role}")
    if not credentials:
        raise ValueError("no data credentials available")
    if role == KIS_DATA_ROLE_DECISION:
        return credentials[0]
    return credentials[-1]
