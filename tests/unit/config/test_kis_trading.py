"""KisTradingSettings invariant guards."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.config.kis_trading import KisTradingSettings


def test_trading_env_defaults_to_off() -> None:
    assert KisTradingSettings(_env_file=None).KIS_TRADING_ENV == "off"


def test_unknown_env_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        KisTradingSettings(KIS_TRADING_ENV="prod", _env_file=None)


def test_ambient_kis_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIS_ENV", "live")
    assert KisTradingSettings(_env_file=None).KIS_TRADING_ENV == "off"


def test_per_order_cap_cannot_exceed_seed() -> None:
    with pytest.raises(ValidationError):
        KisTradingSettings(
            ORDER_MAX_NOTIONAL_PER_ORDER_KRW=11_000_000,
            ORDER_SEED_CAPITAL=10_000_000,
            _env_file=None,
        )


@pytest.mark.parametrize("buffer", [-1.0, 3000.1, float("nan")])
def test_buffer_bounds_are_enforced(buffer: float) -> None:
    with pytest.raises(ValidationError):
        KisTradingSettings(ORDER_ENTRY_LIMIT_BUFFER_BP=buffer, _env_file=None)


def test_combined_settings_exposes_trading_fields() -> None:
    settings = Settings(_env_file=None)
    assert settings.KIS_TRADING_ENV == "off"
    assert settings.ORDER_JOURNAL_DIR == settings.DATA_DIR / "orders"


@pytest.mark.parametrize("model", [KisTradingSettings, Settings])
def test_credentials_are_hidden_in_repr_and_validation(model) -> None:
    credentials = {
        "KIS_MOCK_APP_KEY": "unique-mock-key",
        "KIS_MOCK_APP_SECRET": "unique-mock-secret",
        "KIS_MOCK_ACCOUNT_NO": "unique-mock-account",
        "KIS_APP_KEY": "unique-live-key",
        "KIS_APP_SECRET": "unique-live-secret",
        "KIS_ACCOUNT_ID": "unique-live-account",
    }
    cfg = model(**credentials, _env_file=None)
    assert all(value not in repr(cfg) for value in credentials.values())
    with pytest.raises(ValidationError) as error:
        model(**credentials, ORDER_MAX_NOTIONAL_PER_ORDER_KRW=11_000_000, _env_file=None)
    assert all(value not in str(error.value) for value in credentials.values())
