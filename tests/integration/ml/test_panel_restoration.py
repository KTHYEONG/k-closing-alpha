import pytest


@pytest.mark.slow
def test_restored_panel_feature_parity_with_legacy_panel() -> None:
    import pandas as pd

    from src.data.candidate_panel import build_restored_trade_log
    from src.ml.dataset import build_ml_dataset

    # Arrange
    trade_log = pd.read_parquet("data/parquet/trade_log.parquet")
    theme = pd.read_parquet("data/parquet/theme.parquet")
    price_history = pd.read_parquet("data/history/price_history.parquet")
    x_legacy, _, cat_legacy, _ = build_ml_dataset(trade_log, theme, feature_set="close_morning61")

    # Act
    restored = build_restored_trade_log(trade_log, price_history, theme_df=theme)
    x_new, _, cat_new, proc_new = build_ml_dataset(restored, theme, feature_set="close_morning61")

    # Assert: identical feature contract, strictly more recent-regime rows
    assert list(x_new.columns) == list(x_legacy.columns)
    assert cat_new == cat_legacy
    assert len(x_new) > len(x_legacy)
    recent = proc_new[proc_new["trade_date"] >= "2026-01-01"]
    per_day = len(recent) / max(recent["trade_date"].nunique(), 1)
    assert per_day > 5.0
    assert restored.attrs["panel_restoration"]["execution_offset_pct"] < 0.0
