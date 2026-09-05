"""FetchConfig end-date bound tracks today, not a frozen literal."""

from __future__ import annotations


def test_default_backfill_end_date_returns_today() -> None:
    import pandas as pd

    from src.backfill.price.config import default_backfill_end_date

    assert default_backfill_end_date() == pd.Timestamp.today().normalize()


def test_fetch_config_fixed_end_date_uses_default_factory() -> None:
    import pandas as pd

    from src.backfill.price.config import FetchConfig, default_backfill_end_date

    cfg = FetchConfig()

    assert cfg.fixed_end_date == default_backfill_end_date()
    assert cfg.fixed_end_date > pd.Timestamp("2025-12-31")
