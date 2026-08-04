"""Google Sheets 연동 설정 도메인 (GSheetSettings).

ServiceAccount 키 경로, 시트/워크시트명, 컬럼명 매핑을 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class GSheetSettings(BaseSettings):
    """Google Sheets 연동 (ServiceAccount / Worksheet) 설정."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GSPREAD_KEY_PATH_ENV: str = Field(default="", validation_alias="GSPREAD_KEY_PATH")
    GSPREAD_SA_JSON: str = Field(default="")
    GOOGLE_SHEET_NAME: str = "Stock"
    TRADE_WORKSHEETS: list[str] = ["Trade", "Trade2"]
    THEME_WORKSHEET_NAME: str = "코드_테마_DB"

    # Google Sheets 컬럼명 설정
    GOTTEN_COLS: dict[str, str] = {
        "DATE": "(매수날짜)",
        "CODE": "(종목코드)",
        "PROGRAM": "(프로그램_순매수)",
        "INST": "(기관_순매수)",
        "FOREIGN": "(외국인_순매수)",
        "V_KOSPI": "(v-kospi)",
        "V_KOSDAQ": "(v-kosdaq)",
    }

    # ---------------------------------------------------------
    # [스펙 하위 호환 별칭] (spec: gspread_key_path, google_sheet_name, ...)
    # ---------------------------------------------------------
    @property
    def gspread_key_path(self) -> str:
        return self.GSPREAD_KEY_PATH_ENV

    @property
    def google_sheet_name(self) -> str:
        return self.GOOGLE_SHEET_NAME
