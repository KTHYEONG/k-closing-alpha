"""Auto-generated from contract: kis_key_pool."""

from __future__ import annotations

def test_kis_settings_key_pool_defaults_and_reexport(monkeypatch) -> None:
    from pathlib import Path

    from src.config.kis import KisSettings

    monkeypatch.delenv("KIS_DATA_ROLE", raising=False)
    monkeypatch.delenv("KIS_TOKEN_CACHE_DIR", raising=False)

    s = KisSettings(_env_file=None)

    assert s.KIS_DATA_ROLE == "batch"
    assert s.KIS_TOKEN_CACHE_DIR == Path.home() / ".cache" / "kis"  # noqa: SIM300 - contract skeleton order

    monkeypatch.setenv("KIS_DATA_ROLE", "decision")
    assert KisSettings(_env_file=None).KIS_DATA_ROLE == "decision"

    import src.config as config_pkg
    assert "KIS_DATA_ROLE" in config_pkg.__all__
    assert "KIS_TOKEN_CACHE_DIR" in config_pkg.__all__

