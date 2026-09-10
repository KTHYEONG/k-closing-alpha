from __future__ import annotations


def test_live_entrypoints_import_after_constant_consolidation() -> None:
    import importlib

    entrypoints = [
        "src.daily.collect",
        "src.daily.paper_trade",
        "src.daily.predict",
        "src.daily.archive_intraday",
        "src.ml.retrain",
        "src.ml.costaware_topk",
        "src.ml.topk_ranker_research",
        "src.backfill.backfill_price",
        "src.backfill.backfill_altdata",
        "src.backfill.kis_flow_backfill",
        "src.backfill.intraday.backfill_minute_history",
    ]

    # Then: no import cycle was introduced by the new edges into strategy.contract.
    for name in entrypoints:
        assert importlib.import_module(name) is not None, f"{name} failed to import"
