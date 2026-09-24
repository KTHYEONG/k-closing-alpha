from src import settings


def test_altdata_config_paths_and_dart_key_registered() -> None:
    assert settings.ALTDATA_DIR.name == "altdata"
    assert settings.ALTDATA_DIR.parent.name == "history"
    assert isinstance(settings.DART_API_KEY, str)


def test_altdata_secondary_key_defaults_empty_and_reads_env(monkeypatch) -> None:
    """New setting defaults to empty and reads from env."""
    from src.config.altdata import AltDataSettings

    assert AltDataSettings(_env_file=None).OPENDART_API_KEY_2 == ""
    monkeypatch.setenv("OPENDART_API_KEY_2", "second-key")
    assert AltDataSettings(_env_file=None).OPENDART_API_KEY_2 == "second-key"
