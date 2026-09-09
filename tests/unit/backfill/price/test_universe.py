"""Price backfill candidate universe includes screened-but-untraded symbols."""

from __future__ import annotations




def test_load_candidate_universe_delegates_to_candidate_universe_symbols(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import universe as mod

    monkeypatch.setattr(
        mod,
        "load_candidate_universe_symbols",
        lambda: pd.DataFrame({"symbol": ["000660", "035720"], "market": ["KOSPI", "KOSDAQ"]}),
    )

    out = mod._load_candidate_universe()

    assert set(out["symbol"]) == {"000660", "035720"}
    assert list(out.columns) == ["symbol", "market"]



def test_load_candidate_universe_returns_empty_frame_when_candidate_source_errors(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import universe as mod

    def _boom():
        raise RuntimeError("candidate source unavailable")

    monkeypatch.setattr(mod, "load_candidate_universe_symbols", _boom)

    out = mod._load_candidate_universe()

    assert out.empty
    assert list(out.columns) == ["symbol", "market"]



def test_universe_module_no_longer_exposes_load_or_build_snapshot() -> None:
    from src.backfill.price import universe as mod

    assert not hasattr(mod, "load_or_build_snapshot")



def test_load_candidate_universe_returns_empty_frame_when_candidate_source_empty(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import universe as mod

    monkeypatch.setattr(
        mod,
        "load_candidate_universe_symbols",
        lambda: pd.DataFrame(columns=["symbol", "market"]),
    )

    out = mod._load_candidate_universe()

    assert out.empty
    assert list(out.columns) == ["symbol", "market"]
