from src import settings


def test_altdata_config_paths_and_dart_key_registered() -> None:
    assert settings.ALTDATA_DIR.name == "altdata"
    assert settings.ALTDATA_DIR.parent.name == "history"
    assert isinstance(settings.DART_API_KEY, str)
