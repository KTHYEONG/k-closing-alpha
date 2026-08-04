from __future__ import annotations

from src.backfill.backfill_condition_history import run_condition_history_backfill
from src.backfill.backfill_sheet import SheetBackfillConfig, run_sheet_backfill


def main() -> None:
    cfg = SheetBackfillConfig(
        sheets=("Trade", "Trade2"),
        fill_price=True,
        fill_ema_volume=True,
        fill_flow=True,
        fill_index_vol=True,
    )
    run_sheet_backfill(cfg)
    run_condition_history_backfill("condition_history_종가매매.xlsx")


if __name__ == "__main__":
    main()
