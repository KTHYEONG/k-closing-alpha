"""AltDataSettings가 KRX Open API 키 4종을 소유하는지 검증합니다."""

from __future__ import annotations


def test_altdata_owns_krx_openapi_keys() -> None:
    from src.config.altdata import AltDataSettings
    from src.settings import Settings

    for field in ("KRX_OPENAPI_KEY", "KRX_OPENAPI_BASE_URL", "KRX_OPENAPI_BASE_URLS", "KRX_OPENAPI_ENDPOINTS"):
        assert field in AltDataSettings.model_fields
        assert hasattr(Settings(), field)

    settings = Settings()
    assert settings.KRX_OPENAPI_BASE_URL == ""
    assert settings.KRX_OPENAPI_BASE_URLS == ""
    assert settings.KRX_OPENAPI_ENDPOINTS == ""
