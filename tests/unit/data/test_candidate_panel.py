from __future__ import annotations


def test_load_candidate_universe_symbols_prunes_and_dedups(tmp_path) -> None:
    import pandas as pd

    from src.data import candidate_panel as mod

    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "ABC"],
            "시장구분": ["KOSPI", "KOSPI"],
            "종가": [100.0, 200.0],
        }
    )
    path = tmp_path / "condition_history_cleaned.parquet"
    condition.to_parquet(path, index=False)
    arch = pd.DataFrame({"스냅샷_날짜": ["2026-03-03"], "종목코드": ["005930"]})

    out = mod.load_candidate_universe_symbols(condition_history_path=path, archive_df=arch)

    assert list(out.columns) == ["symbol", "market"]
    assert set(out["symbol"]) == {"005930"}
    # Archive row (no market) wins the duplicate key.
    assert out.set_index("symbol").loc["005930", "market"] == "UNKNOWN"


def test_load_candidate_universe_symbols_empty_sources_returns_empty_frame(tmp_path) -> None:
    import pandas as pd

    from src.data import candidate_panel as mod

    out = mod.load_candidate_universe_symbols(
        condition_history_path=tmp_path / "missing.parquet",
        archive_df=pd.DataFrame(),
    )

    assert out.empty
    assert list(out.columns) == ["symbol", "market"]




def test_candidate_panel_module_no_longer_exposes_restoration_symbols() -> None:
    import src.data.candidate_panel as mod

    removed_names = (
        "build_restored_trade_log",
        "attach_reconstructed_labels",
        "measure_execution_offset_pct",
        "check_price_history_freshness",
        "load_candidate_snapshot_panel",
        "_read_condition_history",
        "CONDITION_HISTORY_COLUMN_ALIAS",
        "_ARCHIVE_COLUMN_ALIAS",
        "PANEL_COLUMNS",
        "_PANEL_FLOAT32_COLUMNS",
        "LABEL_SOURCE_COLUMN",
        "EXECUTED_LABEL_SOURCE",
        "RECONSTRUCTED_LABEL_SOURCE",
        "UNSCORED_SCENARIO_SENTINEL",
        "ARCHIVE_SCENARIO_THEME_AUTHENTIC_SINCE",
        "NO_THEME_SENTINELS",
        "_CLOSE_RTOL",
    )
    for name in removed_names:
        assert not hasattr(mod, name), f"{name} should have been removed"

    assert hasattr(mod, "load_candidate_universe_symbols")
    assert hasattr(mod, "_resolve_archive_df")
    assert hasattr(mod, "_default_condition_history_path")

