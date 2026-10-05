"""NXT consolidated-to-KRX reconstruction fit and collection parameters (NxtReconstructionSettings)."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from src.config._env import EnvSettings

_DEFAULT_ALPHAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
_DEFAULT_STRUCTURES: tuple[tuple[int, int], ...] = tuple(
    (prior, gap) for prior in (2, 3, 5) for gap in (10, 15, 20, 30)
)


class NxtReconstructionSettings(EnvSettings):
    """Declared parameters of the NXT consolidated-to-KRX reconstruction fit and its collection run.

    Attributes:
        NXT_RECON_ALPHAS: Candidate EWMA coefficients in (0, 1]; at least three distinct values.
        NXT_RECON_STRUCTURES: (min_prior_days, max_gap_days) grid; min_prior_days >= 2
            (depth 1 measured p90 15.8%).
        NXT_RECON_IDENTITY_TOLERANCE: Maximum |EOD - (V_krx + A)| / EOD kept.
        NXT_RECON_HOLDOUT_FRACTION: Fraction of distinct dates held out, in (0, 0.5].
        NXT_RECON_MIN_HOLDOUT_DAYS: Minimum distinct holdout dates.
        NXT_RECON_MAX_SELECTION_REL_ERR_P90: Ceiling on a candidate's selection-set p90 error.
        NXT_RECON_CALIBRATION_CONCURRENCY: In-flight Toss symbol-days per date.

    Raises:
        ValueError: Any bound violation (fail closed at construction).
    """

    NXT_RECON_ALPHAS: tuple[float, ...] = Field(default=_DEFAULT_ALPHAS)
    NXT_RECON_STRUCTURES: tuple[tuple[int, int], ...] = Field(default=_DEFAULT_STRUCTURES)
    NXT_RECON_IDENTITY_TOLERANCE: float = Field(default=0.05, gt=0.0, allow_inf_nan=False)
    NXT_RECON_HOLDOUT_FRACTION: float = Field(default=0.25, gt=0.0, le=0.5, allow_inf_nan=False)
    NXT_RECON_MIN_HOLDOUT_DAYS: int = Field(default=30, ge=1)
    NXT_RECON_MAX_SELECTION_REL_ERR_P90: float = Field(default=0.14, ge=0.0, allow_inf_nan=False)
    NXT_RECON_CALIBRATION_CONCURRENCY: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def _check_grids(self) -> Self:
        alphas = [float(a) for a in self.NXT_RECON_ALPHAS]
        if any(isinstance(a, bool) for a in self.NXT_RECON_ALPHAS):
            raise ValueError("NXT_RECON_ALPHAS must hold numbers within (0, 1]")
        if len(set(alphas)) < 3 or any(not 0.0 < a <= 1.0 for a in alphas):
            raise ValueError("NXT_RECON_ALPHAS must hold at least three distinct values within (0, 1]")
        pairs = [tuple(s) for s in self.NXT_RECON_STRUCTURES]
        if not pairs:
            raise ValueError("NXT_RECON_STRUCTURES must hold (min_prior_days, max_gap_days) pairs")
        for item in pairs:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or isinstance(item[0], bool)
                or isinstance(item[1], bool)
                or not isinstance(item[0], int)
                or not isinstance(item[1], int)
            ):
                raise ValueError(
                    "NXT_RECON_STRUCTURES must hold (min_prior_days, max_gap_days) integer pairs"
                )
            if item[0] < 2:
                raise ValueError("NXT_RECON_STRUCTURES min_prior_days must be >= 2")
            if item[1] < 1:
                raise ValueError("NXT_RECON_STRUCTURES max_gap_days must be >= 1")
        return self
