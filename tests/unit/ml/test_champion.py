"""champion 오케스트레이터의 feature_set / price_history 배선 계약."""
from __future__ import annotations

import inspect

from src.ml.champion import train_champion_bundle, train_tuned_champion_bundle


def test_champion_entrypoints_accept_feature_set_and_price_history() -> None:
    for fn in (train_champion_bundle, train_tuned_champion_bundle):
        params = inspect.signature(fn).parameters
        assert params["feature_set"].default == "close_morning61"
        assert "price_history_df" in params
        assert params["price_history_df"].default is None
