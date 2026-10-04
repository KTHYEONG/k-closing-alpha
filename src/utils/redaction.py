"""Pure text redaction for operator-facing egress (alerts, digests, outbox).

Credentials reach alert text through tracebacks, vendor error bodies and request
URLs (e.g. OpenDART ``crtfc_key`` query parameters inside ``requests`` exception
messages). Masking is applied at egress rather than at every log call site because
the alert path forwards arbitrary journal text it did not author.
"""

from __future__ import annotations

import re
from collections.abc import Collection

REDACTION_MASK: str = "***"
"""Replacement token for a masked value; key names and quoting are preserved."""

SECRET_MIN_MASK_CHARS: int = 8
"""Configured values shorter than this are not exact-matched (see OD-2)."""

CREDENTIAL_KEY_SUFFIXES: tuple[str, ...] = (
    "appkey",
    "appsecret",
    "secretkey",
    "authkey",
    "apikey",
    "accesstoken",
    "approvalkey",
    "crtfckey",
    "clientsecret",
    "apppassword",
)
"""Normalized (lower-case, ``_``/``-`` removed) key-name suffixes whose values are credentials."""

CREDENTIAL_EXACT_KEYS: tuple[str, ...] = ("token", "authorization")
"""Normalized key names that are credentials only on exact match (not as suffix)."""

_DIGIT_SUFFIX_RE = re.compile(r"\d+$")

# Separator padding is horizontal-only so a key at end of line never captures the next line.
# The backslash-quoted forms cover JSON documents embedded as string values (``"body": "{\\"appkey\\": ...}"``).
_KV_RE = re.compile(
    r"(?P<key_tok>\\\"(?P<key_qe>[A-Za-z0-9_.\-]+)\\\"|\"(?P<key_qd>[A-Za-z0-9_.\-]+)\""
    r"|'(?P<key_qs>[A-Za-z0-9_.\-]+)'|(?P<key_u>[A-Za-z0-9_.\-]+))"
    r"(?P<pre>[ \t]*)(?P<sep>[:=])(?P<post>[ \t]*)"
    r"(?:\\\"(?P<val_de>(?:[^\"\\\n]|\\[^\"\n])*)\\\""
    r"|\"(?P<val_dq>(?:[^\"\\\n]|\\.)*)\"|'(?P<val_sq>(?:[^'\\\n]|\\.)*)'|(?P<val_u>[^\s,;&)}\]'\"]+))"
)

_BEARER_RE = re.compile(r"\b(?P<scheme>Bearer)[ \t]+(?P<tok>[^\s,;&)}\]'\"\"]+)", re.IGNORECASE)


def _normalize_key(key: str) -> str:
    return key.lower().replace("_", "").replace("-", "")


def _is_credential_key(normalized: str) -> bool:
    if normalized in CREDENTIAL_EXACT_KEYS:
        return True
    for suffix in CREDENTIAL_KEY_SUFFIXES:
        if normalized.endswith(suffix):
            return True
    stripped = _DIGIT_SUFFIX_RE.sub("", normalized)
    if stripped != normalized:
        for suffix in CREDENTIAL_KEY_SUFFIXES:
            if stripped.endswith(suffix):
                return True
    return False


def _mask_bearer_match(match: re.Match[str]) -> str:
    token = match.group("tok")
    if token == REDACTION_MASK:
        return match.group(0)
    return f"{match.group('scheme')} {REDACTION_MASK}"


def _pair_key(match: re.Match[str]) -> str:
    return match.group("key_qe") or match.group("key_qd") or match.group("key_qs") or match.group("key_u") or ""


def _mask_kv_match(match: re.Match[str]) -> str:
    """Mask the value of a pair whose key is already known to be a credential."""
    key = _pair_key(match)
    prefix = f"{match.group('key_tok')}{match.group('pre')}{match.group('sep')}{match.group('post')}"
    is_authorization = _normalize_key(key) == "authorization"
    for group, quote in (("val_de", '\\"'), ("val_dq", '"'), ("val_sq", "'")):
        quoted = match.group(group)
        if quoted is None:
            continue
        if is_authorization and quoted.strip().lower().startswith("bearer "):
            return f"{prefix}{quote}{quoted.strip().split(None, 1)[0]} {REDACTION_MASK}{quote}"
        return f"{prefix}{quote}{REDACTION_MASK}{quote}"
    bare = match.group("val_u").strip()
    if is_authorization and bare.lower() == "bearer":
        # ``authorization: Bearer <token>``: the token follows the space and is masked by the scheme pass.
        return f"{prefix}{bare}"
    return f"{prefix}{REDACTION_MASK}"


def _mask_credential_pairs(text: str) -> str:
    # A non-credential pair (``url: https://...``, ``err: "appkey=..."``) may embed a
    # credential pair inside its value, so only its key and separator are consumed and
    # scanning resumes inside the value instead of skipping it.
    out: list[str] = []
    pos = 0
    while True:
        match = _KV_RE.search(text, pos)
        if match is None:
            break
        key = _pair_key(match)
        if not _is_credential_key(_normalize_key(key)):
            resume = match.end("sep")
            out.append(text[pos:resume])
            pos = resume
            continue
        out.append(text[pos : match.start()])
        out.append(_mask_kv_match(match))
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def redact_secrets(text: str, secret_values: Collection[str] = ()) -> str:
    """Mask credential material in free text before it leaves the process.

    Two independent passes, in this order:
    1. Exact-value pass: every distinct member of ``secret_values`` whose stripped
       length is at least SECRET_MIN_MASK_CHARS is replaced by REDACTION_MASK wherever
       it occurs as a literal substring (case-sensitive, longest value first so a
       secret containing another secret is masked whole).
    2. Pattern pass (case-insensitive): the value of any key/value pair whose key
       normalizes to a name ending in CREDENTIAL_KEY_SUFFIXES or equal to a member of
       CREDENTIAL_EXACT_KEYS is replaced by REDACTION_MASK, for the shapes
       ``key=value``, ``key: value``, ``"key": "value"``, ``'key': 'value'`` and
       query-string ``key=value&``; an ``Authorization`` value keeps its scheme
       (``Bearer ***``); and any standalone ``Bearer <token>`` is reduced to
       ``Bearer ***``.

    Args:
        text: Arbitrary operator-facing text (journal tail, exception line, digest body).
        secret_values: Raw configured credential values. Never logged, never returned.

    Returns:
        Text with credential values replaced; key names, separators and quotes are kept
        so the alert still shows *which* credential was involved.
    """
    if not text:
        return text
    seen: set[str] = set()
    candidates: list[str] = []
    for raw in secret_values:
        value = raw.strip() if isinstance(raw, str) else str(raw).strip()
        if len(value) < SECRET_MIN_MASK_CHARS or value in seen:
            continue
        seen.add(value)
        candidates.append(value)
    candidates.sort(key=len, reverse=True)
    for value in candidates:
        if value in text:
            text = text.replace(value, REDACTION_MASK)
    text = _mask_credential_pairs(text)
    text = _BEARER_RE.sub(_mask_bearer_match, text)
    return text


__all__ = [
    "CREDENTIAL_EXACT_KEYS",
    "CREDENTIAL_KEY_SUFFIXES",
    "REDACTION_MASK",
    "SECRET_MIN_MASK_CHARS",
    "redact_secrets",
]
