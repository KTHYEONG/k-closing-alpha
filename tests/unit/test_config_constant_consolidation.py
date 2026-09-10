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
        "src.backfill.backfill_regime",
        "src.backfill.kis_flow_backfill",
        "src.backfill.intraday.backfill_minute_history",
    ]

    # Then: no import cycle was introduced by the new edges into strategy.contract.
    for name in entrypoints:
        assert importlib.import_module(name) is not None, f"{name} failed to import"


def test_this_phase_introduces_no_silent_swallow() -> None:
    import ast
    import inspect
    from pathlib import Path

    from src.backfill import backfill_regime

    # Given: the parser this phase authored.
    tree = ast.parse(inspect.getsource(backfill_regime._split_csv))

    # Then: it catches nothing at all - no handler, bare or otherwise.
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert handlers == [], "_split_csv must not catch exceptions"

    # And: it performs no I/O and consults no environment.
    body = inspect.getsource(backfill_regime._split_csv)
    for forbidden in ("open(", "read_text", "os.getenv", "os.environ", "Path("):
        assert forbidden not in body, f"_split_csv must be pure, found {forbidden}"

    # And: the hand-rolled .env reader and its blanket swallow are gone from the module.
    module_source = Path("src/backfill/backfill_regime.py").read_text(encoding="utf-8")
    assert 'BASE_DIR / ".env"' not in module_source
    assert not hasattr(backfill_regime, "_get_env_value")
    assert not hasattr(backfill_regime, "_get_env_csv")
