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


def test_to_parquet_heals_merged_panel_across_incremental_runs(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    from src.backfill.price.runner import _to_parquet
    from src.data.panel_integrity import assert_price_history_units_clean

    def _frame(dates, closes, prevs, chgs):
        n = len(dates)
        return pd.DataFrame({
            "date": pd.to_datetime(dates),
            "symbol": ["005930"] * n,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "prev_close": prevs,
            "market_cap_100m": [900.0] * n,
            "trade_value_100m": [300.0] * n,
            "daily_change_pct": chgs,
            "market": ["KOSPI"] * n,
            "volume": [1000] * n,
        })

    target = tmp_path / "price_history.parquet"

    # Given: a first run persists two bars.
    first = _frame(["2026-03-02", "2026-03-03"], [10000.0, 10500.0], [np.nan, 10000.0], [np.nan, 0.05])
    _to_parquet(first, target)

    # When: a second incremental run brings a slice whose first row lost prev_close.
    second = _frame(["2026-03-04", "2026-03-05"], [10920.0, 11000.0], [np.nan, 10920.0], [np.nan, 11000.0 / 10920.0 - 1.0])
    _to_parquet(second, target)

    # Then: the boundary is healed from stored history and the panel is clean on disk.
    stored = pd.read_parquet(target).sort_values("date").reset_index(drop=True)
    assert len(stored) == 4
    assert assert_price_history_units_clean(stored) is None
    np.testing.assert_allclose(
        stored["daily_change_pct"].to_numpy(dtype=float)[1:],
        [0.05, 0.04, 11000.0 / 10920.0 - 1.0],
        rtol=1e-9,
    )


def test_to_parquet_invokes_guard_on_both_exists_and_new_file_branches(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.backfill.price import corporate_actions, runner as mod
    from src.backfill.price.config import FetchConfig

    calls: list[str] = []

    def _spy(merged, fetch_cfg, market_hint):
        calls.append("called")
        return merged

    # _to_parquet가 함수 내부에서 매 호출마다 다시 임포트하므로 corporate_actions 쪽만 패치하면 된다.
    monkeypatch.setattr(corporate_actions, "heal_corporate_action_breach", _spy)

    # heal_price_history_panel이 요구하는 전체 컬럼(특히 prev_close)을 갖춘 프레임이어야 한다.
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01"]),
        "symbol": ["005930"],
        "open": [70_000], "high": [70_000], "low": [70_000], "close": [70_000],
        "prev_close": [69_500],
        "market_cap_100m": [900.0], "trade_value_100m": [300.0],
        "daily_change_pct": [70_000 / 69_500 - 1.0],
        "market": ["KOSPI"], "volume": [1000],
    })
    target = tmp_path / "price_history.parquet"
    fetch_cfg = FetchConfig()

    # When: 신규 생성 분기(else)에서 가드 인자를 넘긴다
    mod._to_parquet(frame, target, fetch_cfg=fetch_cfg, market_hint={"005930": "KOSPI"})

    # Then: 가드가 실제로 실행됐다
    assert calls == ["called"]

    # When: 기존 파일이 있는 병합 분기에서도 가드 인자를 넘긴다
    mod._to_parquet(frame, target, fetch_cfg=fetch_cfg, market_hint={"005930": "KOSPI"})

    # Then: 두 번째(병합) 호출에서도 실행됐다
    assert calls == ["called", "called"]
