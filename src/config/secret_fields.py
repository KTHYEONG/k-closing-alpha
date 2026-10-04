"""Registry of credential-bearing settings and env names (names only, never values).

Settings credential fields are plain ``str`` (not ``SecretStr``), so nothing in the
type system marks them; this registry is the single declaration egress redaction
uses to enumerate what must never leave the process.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings

SECRET_SETTING_FIELDS: tuple[str, ...] = (
    "KIS_APP_KEY",
    "KIS_APP_SECRET",
    "KIS_ACCOUNT_ID",
    "LS_APP_KEY",
    "LS_APP_SECRET",
    "TOSS_APP_KEY",
    "TOSS_APP_SECRET",
    "KIWOOM_APP_KEY",
    "KIWOOM_SECRET_KEY",
    "OPENDART_API_KEY",
    "OPENDART_API_KEY_2",
    "DART_API_KEY",
    "KRX_OPENAPI_KEY",
    "ALERT_GMAIL_APP_PASSWORD",
    "ALERT_WEBHOOK_URL",
)
"""Settings model fields whose values are credentials or credential-equivalent
(a Slack/Discord webhook URL embeds its auth token; see OD-1 for KIS_ACCOUNT_ID)."""

SECRET_ENV_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"(?:APP_KEY|APP_SECRET|SECRET_KEY|API_KEY|APP_PASSWORD|AUTH_KEY|ACCESS_TOKEN)(?:_\d+)?$",
    re.IGNORECASE,
)
"""Case-insensitive pattern matching env names that hold credentials but are not
settings fields (numbered pools such as ``KIS_DATA_<n>_APP_SECRET``,
``KIS_TRADE_APP_KEY``, ``LIVE_ALERT_GMAIL_APP_PASSWORD``, legacy ``KIWOM_SECRET_KEY``):
a name ending in APP_KEY, APP_SECRET, SECRET_KEY, API_KEY, APP_PASSWORD, AUTH_KEY or
ACCESS_TOKEN, optionally followed by ``_<digits>``."""


def _credential_forms(raw: object) -> set[str]:
    # HTTP client errors quote only the request path ("Max retries exceeded with url: /services/T/B/<token>"),
    # so a credential-bearing URL is also masked by its path+query, which alone carries the token.
    text = str(raw).strip()
    if not text:
        return set()
    forms = {text}
    parts = urlsplit(text)
    if parts.scheme in ("http", "https") and parts.netloc:
        locator = parts.path + (f"?{parts.query}" if parts.query else "")
        if len(locator.strip("/")) > 1:
            forms.add(locator)
    return forms


def configured_secret_values(cfg: BaseSettings, env: Mapping[str, str]) -> frozenset[str]:
    """Collect the raw credential values configured for this process.

    Args:
        cfg: Settings instance (normally the live singleton); each name in
            SECRET_SETTING_FIELDS is read with a missing attribute treated as absent.
        env: Credential environment mapping (env file merged with ``os.environ``);
            every entry whose name matches SECRET_ENV_NAME_PATTERN contributes its value.

    Returns:
        Distinct non-empty stripped values, plus the path+query of any http(s) URL value. The caller
        must treat the result as secret: never log, persist, or include it in an exception message.
    """
    values: set[str] = set()
    for name in SECRET_SETTING_FIELDS:
        raw: Any = getattr(cfg, name, None)
        if raw is not None:
            values |= _credential_forms(raw)
    for name, raw in env.items():
        if SECRET_ENV_NAME_PATTERN.search(str(name)):
            values |= _credential_forms(raw)
    return frozenset(values)


__all__ = [
    "SECRET_ENV_NAME_PATTERN",
    "SECRET_SETTING_FIELDS",
    "configured_secret_values",
]
