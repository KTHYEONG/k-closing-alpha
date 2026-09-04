"""Runner re-export gate for lean_check co-modification check."""

from __future__ import annotations


def test_runner_reexports_new_panels() -> None:
    from src.backfill.altdata import runner

    assert callable(runner.collect_credit_balance)
    assert callable(runner.collect_program_trade_daily)
