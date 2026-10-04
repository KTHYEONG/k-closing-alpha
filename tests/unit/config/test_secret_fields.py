from __future__ import annotations

import pytest

from src.config import Settings
from src.config.secret_fields import SECRET_ENV_NAME_PATTERN, SECRET_SETTING_FIELDS, configured_secret_values


def test_secret_fields_registry_covers_every_credential_settings_field() -> None:
    fields = set(Settings.model_fields)
    credential_named = {name for name in fields if SECRET_ENV_NAME_PATTERN.search(name)}
    assert credential_named <= set(SECRET_SETTING_FIELDS)
    assert set(SECRET_SETTING_FIELDS) <= fields


@pytest.mark.parametrize(
    "name",
    [
        "KIS_DATA_3_APP_SECRET",
        "KIS_DATA_12_APP_KEY",
        "KIS_TRADE_APP_KEY",
        "KIWOM_SECRET_KEY",
        "LIVE_ALERT_GMAIL_APP_PASSWORD",
        "OPENDART_API_KEY_2",
        "kis_data_1_app_secret",
    ],
)
def test_secret_fields_pattern_matches_pool_and_legacy_names(name: str) -> None:
    assert SECRET_ENV_NAME_PATTERN.search(name)


@pytest.mark.parametrize(
    "name",
    [
        "COLLECTION_CONCURRENCY_PER_KEY",
        "KIS_TOKEN_CACHE_DIR",
        "KIS_HTS_ID",
        "KIS_DATA_1_HTS_ID",
        "KIS_DATA_SLOTS",
        "ALERT_GMAIL_USER",
    ],
)
def test_secret_fields_pattern_rejects_non_secrets(name: str) -> None:
    assert not SECRET_ENV_NAME_PATTERN.search(name)


def test_secret_fields_configured_values_collected_from_settings_and_env() -> None:
    cfg = Settings(KIS_APP_SECRET="kis-secret-value", ALERT_WEBHOOK_URL="https://hooks.example/T/B/xyz", LS_APP_KEY="")
    values = configured_secret_values(
        cfg, {"KIS_DATA_2_APP_SECRET": " pool-secret-2 ", "KIS_DATA_SLOTS": "1,2"}
    )
    assert {"kis-secret-value", "https://hooks.example/T/B/xyz", "pool-secret-2"} <= values
    assert "1,2" not in values
    assert "" not in values


def test_secret_fields_webhook_url_path_masked_without_host() -> None:
    from src.utils.redaction import redact_secrets

    cfg = Settings(ALERT_WEBHOOK_URL="https://hooks.slack.com/services/T0/B0/XXXXSECRET")
    values = configured_secret_values(cfg, {})
    out = redact_secrets("Max retries exceeded with url: /services/T0/B0/XXXXSECRET (Caused by ...)", values)
    assert "XXXXSECRET" not in out
    assert "Max retries exceeded with url:" in out


def test_secret_fields_non_string_values_stringified() -> None:
    values = configured_secret_values(Settings(), {"KIS_DATA_9_APP_KEY": 123456789})  # type: ignore[dict-item]
    assert "123456789" in values
