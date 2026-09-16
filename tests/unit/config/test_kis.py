"""config/kis.py KIS API 설정 도메인 단위 테스트."""

from __future__ import annotations

from src.config.kis import KisSettings


def test_kis_settings_defaults(monkeypatch) -> None:
    for name in ("KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT_ID", "KIS_HTS_ID"):
        monkeypatch.delenv(name, raising=False)
    settings = KisSettings(_env_file=None)
    assert settings.KIS_APP_KEY == ""
    assert settings.KIS_APP_SECRET == ""
    assert settings.KIS_ACCOUNT_ID == ""
    assert settings.KIS_HTS_ID == ""
    assert settings.KIS_BASE_URL == "https://openapi.koreainvestment.com:9443"


def test_kis_settings_env_resolution(monkeypatch) -> None:
    monkeypatch.setenv("KIS_APP_KEY", "env_key")
    monkeypatch.setenv("KIS_APP_SECRET", "env_secret")
    monkeypatch.setenv("KIS_ACCOUNT_ID", "env_account")
    monkeypatch.setenv("KIS_HTS_ID", "env_hts")
    settings = KisSettings(_env_file=None)
    assert settings.KIS_API_CONFIG == {
        "app_key": "env_key",
        "app_secret": "env_secret",
        "account_id": "env_account",
        "hts_id": "env_hts",
    }


def test_kis_settings_has_no_legacy_data_single_key_fields() -> None:
    from src.config.kis import KisSettings

    settings = KisSettings(_env_file=None)

    # Then: 풀 기반(key_pool.py DATA_1~5)로 완전히 대체된 죽은 필드가 재도입되지 않았다
    assert not hasattr(settings, "KIS_DATA_APP_KEY")
    assert not hasattr(settings, "KIS_DATA_APP_SECRET")
    assert not hasattr(settings, "KIS_DATA_HTS_ID")
    assert not hasattr(settings, "KIS_DATA_API_CONFIG")
    # And: 여전히 살아있는 필드는 보존됨
    assert settings.KIS_DATA_ROLE == "batch"
