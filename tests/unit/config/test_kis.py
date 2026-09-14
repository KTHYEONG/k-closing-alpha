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


def test_kis_settings_data_account_fields_default_empty_and_config_shape(monkeypatch) -> None:
    from src.config.kis import KisSettings

    # Given: 실제 개발 셸(~/.quant_env.sh 소싱)의 KIS_DATA_* 값이 새어들지 않도록 격리
    for var in ("KIS_DATA_APP_KEY", "KIS_DATA_APP_SECRET", "KIS_DATA_HTS_ID"):
        monkeypatch.delenv(var, raising=False)
    s = KisSettings(_env_file=None)

    # Then: 기본값은 빈 문자열, account_id는 항상 하드코딩 빈 값
    assert s.KIS_DATA_APP_KEY == ""
    assert s.KIS_DATA_APP_SECRET == ""
    assert s.KIS_DATA_HTS_ID == ""
    assert s.KIS_DATA_API_CONFIG == {"app_key": "", "app_secret": "", "account_id": "", "hts_id": ""}

    # Given: 명시적으로 값을 채운 인스턴스
    s2 = KisSettings(_env_file=None, KIS_DATA_APP_KEY="k", KIS_DATA_APP_SECRET="s", KIS_DATA_HTS_ID="h")

    # Then
    assert s2.KIS_DATA_API_CONFIG == {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h"}
