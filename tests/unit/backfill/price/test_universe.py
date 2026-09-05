"""Price backfill candidate universe includes screened-but-untraded symbols."""

from __future__ import annotations


def test_load_candidate_universe_unions_condition_history_symbols(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import universe as mod

    monkeypatch.setattr(
        mod,
        "load_or_build_snapshot",
        lambda **kwargs: pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]}),
    )
    monkeypatch.setattr(
        mod,
        "load_candidate_universe_symbols",
        lambda: pd.DataFrame({"symbol": ["000660", "035720"], "market": ["KOSPI", "KOSDAQ"]}),
    )

    out = mod._load_candidate_universe()

    assert set(out["symbol"]) == {"005930", "000660", "035720"}
    assert out["symbol"].is_unique


def test_load_candidate_universe_falls_back_when_candidate_source_errors(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import universe as mod

    monkeypatch.setattr(
        mod,
        "load_or_build_snapshot",
        lambda **kwargs: pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]}),
    )

    def _boom():
        raise RuntimeError("candidate source unavailable")

    monkeypatch.setattr(mod, "load_candidate_universe_symbols", _boom)

    out = mod._load_candidate_universe()

    assert set(out["symbol"]) == {"005930"}
