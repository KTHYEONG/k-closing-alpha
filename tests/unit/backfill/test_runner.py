"""Contract test: the altdata runner re-exports the collectors of the newer panels."""

from __future__ import annotations


def test_runner_reexports_new_panels() -> None:
    from src.backfill.altdata import runner

    assert callable(runner.collect_credit_balance)
    assert callable(runner.collect_program_trade_daily)
