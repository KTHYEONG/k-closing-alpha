"""config/base.py 경로 설정 도메인 단위 테스트."""

from __future__ import annotations

from pathlib import Path

from src.config.base import PathSettings


def test_path_settings_defaults_point_to_project_root() -> None:
    settings = PathSettings()
    assert Path(__file__).resolve().parent.parent.parent.parent == settings.BASE_DIR
    assert settings.DATA_DIR == settings.BASE_DIR / "data"
    assert settings.MODELS_DIR == settings.BASE_DIR / "artifacts" / "models"


def test_path_settings_derived_paths(tmp_path: Path) -> None:
    settings = PathSettings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", _env_file=None)
    assert tmp_path / "data" / "parquet" == settings.PARQUET_DIR
    assert settings.TRADE_LOG_PARQUET_PATH == settings.PARQUET_DIR / "trade_log.parquet"
    assert settings.THEME_PARQUET_PATH == settings.PARQUET_DIR / "theme.parquet"
    assert settings.HISTORY_PARQUET_PATH == settings.HISTORY_DIR / "archive.parquet"
    assert tmp_path / "data" / "history" == settings.HISTORY_DIR


def test_path_settings_no_longer_defines_stock_db_or_condition_csv(tmp_path: Path) -> None:
    """stock.db 폐기에 따라 STOCK_DB_PATH/CONDITION_PARQUET_PATH computed field가 base 도메인에서 제거되었는지 검증합니다."""
    settings = PathSettings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", _env_file=None)
    assert not hasattr(settings, "STOCK_DB_PATH")
    assert not hasattr(settings, "CONDITION_PARQUET_PATH")


def test_ls_tick_max_pages_moved_to_ls_settings() -> None:
    from src.config.base import PathSettings
    from src.config.kiwoom import KiwoomSettings
    from src.config.ls import LsSettings
    from src.settings import Settings

    # Then: the per-vendor tick budgets are retired; COLLECTION_CHART_MAX_PAGES
    # is the single source for the first-pass chart/tick page budget.
    assert "LS_TICK_MAX_PAGES" not in PathSettings.model_fields
    assert "LS_TICK_MAX_PAGES" not in LsSettings.model_fields
    assert "KIWOM_TICK_MAX_PAGES" not in KiwoomSettings.model_fields

    # And: the surviving budget keeps its default.
    settings = Settings()
    assert settings.COLLECTION_CHART_MAX_PAGES == 30


def test_decision_artifact_names_are_stable() -> None:
    from src.config.base import RANK_POOL_PARQUET_NAME, TOPK_DECISIONS_PARQUET_NAME

    assert TOPK_DECISIONS_PARQUET_NAME == "topk_decisions.parquet"
    assert RANK_POOL_PARQUET_NAME == "rank_pool_predictions.parquet"


def test_predict_call_time_join_honors_parquet_dir_override(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import predict

    monkeypatch.setattr(predict.settings, "PARQUET_DIR", tmp_path)
    frame = pd.DataFrame(
        [
            {
                "symbol": "005930",
                "score": 0.5,
            }
        ]
    )

    assert predict.persist_topk_decision(pd.Timestamp("2026-09-18"), frame) == 1
    assert (tmp_path / "topk_decisions.parquet").exists()


def test_no_duplicated_decision_filename_literal() -> None:
    import ast
    from pathlib import Path

    hits: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if path == Path("src/config/base.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "topk_decisions.parquet" in text or "rank_pool_predictions.parquet" in text:
            hits.append(str(path))
    assert hits == []
