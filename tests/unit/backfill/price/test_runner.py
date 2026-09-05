"""Price runner end-date fallback wiring (unfrozen backfill bound)."""

from __future__ import annotations


def test_run_backfill_end_date_defaults_to_today(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.backfill.price import runner as mod

    captured: dict = {}

    def _fake_windows(universe, *, fetch_cfg, **kwargs):
        captured["fixed_end_date"] = fetch_cfg.fixed_end_date
        return []

    monkeypatch.setattr(
        mod,
        "_load_candidate_universe",
        lambda: pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]}),
    )
    monkeypatch.setattr(mod, "_build_symbol_windows", _fake_windows)

    out = mod.run_backfill(
        lookback_trading_days=10,
        max_workers=1,
        kis_rest_limit_per_sec=20.0,
        kis_rest_safety_ratio=0.6,
        kis_max_parallel_calls=1,
        symbol_limit=None,
        include_symbols=None,
        parquet_out=tmp_path / "price_history.parquet",
    )

    assert out.empty
    assert captured["fixed_end_date"] == pd.Timestamp.today().normalize()
    assert captured["fixed_end_date"] > pd.Timestamp("2025-12-31")
