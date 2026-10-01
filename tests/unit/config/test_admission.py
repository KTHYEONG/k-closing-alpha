"""Admission settings invariant guards."""

from __future__ import annotations


def test_admission_defaults() -> None:
    from src.config.admission import AdmissionSettings

    s = AdmissionSettings(_env_file=None)
    assert s.BROKER_ADMISSION_DIR is None
    assert s.BROKER_ADMISSION_CLASS == "bulk"
    assert s.BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS == 1.0
    assert s.BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS == 0.25
    assert s.BROKER_ADMISSION_REQUIRE_SHARED == "auto"
    assert s.BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS == 30.0


def test_admission_rejects_inverted_leads() -> None:
    import pytest

    from src.config.admission import AdmissionSettings

    with pytest.raises(Exception, match="bulk lead"):
        AdmissionSettings(_env_file=None, BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS=2.0, BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS=1.0)


def test_admission_equal_leads_pass() -> None:
    from src.config.admission import AdmissionSettings

    s = AdmissionSettings(
        _env_file=None,
        BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS=1.0,
        BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS=1.0,
    )
    assert s.BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS == 1.0


def test_admission_env_loading(monkeypatch) -> None:
    from src.config.admission import AdmissionSettings

    monkeypatch.setenv("BROKER_ADMISSION_CLASS", "critical")
    monkeypatch.setenv("BROKER_ADMISSION_REQUIRE_SHARED", "never")
    s = AdmissionSettings(_env_file=None)
    assert s.BROKER_ADMISSION_CLASS == "critical"
    assert s.BROKER_ADMISSION_REQUIRE_SHARED == "never"


def test_admission_rejects_nonpositive_lead() -> None:
    import pytest

    from src.config.admission import AdmissionSettings

    with pytest.raises(Exception, match="greater than 0"):
        AdmissionSettings(_env_file=None, BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS=0.0)


def test_admission_wired_into_settings() -> None:
    import src.config as config_mod

    assert config_mod.AdmissionSettings is not None
    assert config_mod.BROKER_ADMISSION_CLASS == config_mod.settings.BROKER_ADMISSION_CLASS
    assert config_mod.BROKER_ADMISSION_DIR == config_mod.settings.BROKER_ADMISSION_DIR
