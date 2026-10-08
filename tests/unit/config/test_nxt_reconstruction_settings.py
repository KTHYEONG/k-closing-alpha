"""NXT reconstruction settings defaults and fail-closed validation."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.config.nxt_reconstruction import NxtReconstructionSettings


def test_reconstruction_defaults() -> None:
    """Declared fit and collection parameters load without error."""
    settings = NxtReconstructionSettings(_env_file=None)
    assert settings.NXT_RECON_ALPHAS == (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    assert tuple(
        (prior, gap) for prior in (2, 3, 5) for gap in (10, 15, 20, 30)
    ) == settings.NXT_RECON_STRUCTURES
    assert settings.NXT_RECON_IDENTITY_TOLERANCE == 0.05
    assert settings.NXT_RECON_HOLDOUT_FRACTION == 0.25
    assert settings.NXT_RECON_MIN_HOLDOUT_DAYS == 30
    assert settings.NXT_RECON_MAX_SELECTION_REL_ERR_P90 == 0.14
    assert settings.NXT_RECON_CALIBRATION_CONCURRENCY == 8


def test_reconstruction_settings_join_the_combined_singleton() -> None:
    """The combined Settings exposes the reconstruction fields."""
    import src.config as config_pkg

    assert any(base.__name__ == "NxtReconstructionSettings" for base in Settings.__mro__)
    assert "NXT_RECON_ALPHAS" in Settings.model_fields
    assert "NxtReconstructionSettings" in config_pkg.__all__


def test_reconstruction_rejects_alpha_outside_unit_interval() -> None:
    """Alphas at or below zero, or above one, raise at construction."""
    with pytest.raises(ValueError, match="NXT_RECON_ALPHAS"):
        NxtReconstructionSettings(NXT_RECON_ALPHAS=(0.0, 0.2, 0.5), _env_file=None)
    with pytest.raises(ValueError, match="NXT_RECON_ALPHAS"):
        NxtReconstructionSettings(NXT_RECON_ALPHAS=(0.2, 0.5, 1.5), _env_file=None)


def test_reconstruction_rejects_too_few_alphas() -> None:
    """Fewer than three distinct alphas raise at construction."""
    with pytest.raises(ValueError, match="NXT_RECON_ALPHAS"):
        NxtReconstructionSettings(NXT_RECON_ALPHAS=(0.5, 0.5, 0.5), _env_file=None)
    with pytest.raises(ValueError, match="NXT_RECON_ALPHAS"):
        NxtReconstructionSettings(NXT_RECON_ALPHAS=(0.3, 0.7), _env_file=None)


def test_reconstruction_rejects_shallow_structures() -> None:
    """min_prior_days below 2 raises at construction."""
    with pytest.raises(ValueError, match="min_prior_days"):
        NxtReconstructionSettings(NXT_RECON_STRUCTURES=((1, 10), (2, 10)), _env_file=None)
    with pytest.raises(ValueError, match="NXT_RECON_STRUCTURES"):
        NxtReconstructionSettings(NXT_RECON_STRUCTURES=(), _env_file=None)


def test_reconstruction_rejects_bad_fractions_and_limits() -> None:
    """Holdout fraction above 0.5 and non-positive concurrency raise."""
    with pytest.raises(ValueError, match="less than or equal to"):
        NxtReconstructionSettings(NXT_RECON_HOLDOUT_FRACTION=0.75, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        NxtReconstructionSettings(NXT_RECON_CALIBRATION_CONCURRENCY=0, _env_file=None)
