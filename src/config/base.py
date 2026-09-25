"""전역 경로 설정 도메인 (PathSettings).

프로젝트 루트 기준 경로(BASE_DIR, DATA_DIR, MODELS_DIR)와
경로로부터 파생되는 computed 경로들을 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import computed_field

from src.config._env import PROJECT_ROOT, EnvSettings


class PathSettings(EnvSettings):
    """프로젝트 루트 경로 및 파생 경로 설정."""

    # ---------------------------------------------------------
    # [경로 설정]
    # ---------------------------------------------------------
    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # ---------------------------------------------------------
    # [파생 경로]
    # ---------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def MODELS_DIR(self) -> Path:
        return self.BASE_DIR / "artifacts" / "models"
    @computed_field  # type: ignore[prop-decorator]
    @property
    def PARQUET_DIR(self) -> Path:
        return self.DATA_DIR / "parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def TRADE_LOG_PARQUET_PATH(self) -> Path:
        return self.PARQUET_DIR / "trade_log.parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def THEME_PARQUET_PATH(self) -> Path:
        return self.PARQUET_DIR / "theme.parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_PARQUET_PATH(self) -> Path:
        return self.HISTORY_DIR / "archive.parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_DIR(self) -> Path:
        return self.DATA_DIR / "history"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def PAPER_DIR(self) -> Path:
        return self.DATA_DIR / "paper"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ALTDATA_DIR(self) -> Path:
        return self.HISTORY_DIR / "altdata"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def PRICE_HISTORY_PARQUET_PATH(self) -> Path:
        return self.HISTORY_DIR / "price_history.parquet"


# Decision artifact file names (not computed paths): consumers join them with
# `settings.PARQUET_DIR` at call time so PARQUET_DIR overrides keep redirecting them.
TOPK_DECISIONS_PARQUET_NAME: str = "topk_decisions.parquet"
RANK_POOL_PARQUET_NAME: str = "rank_pool_predictions.parquet"
