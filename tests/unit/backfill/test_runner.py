"""Runner re-export gate for lean_check co-modification check."""

from __future__ import annotations


def test_runner_reexports_new_panels() -> None:
    from src.backfill.altdata import runner

    assert callable(runner.collect_credit_balance)
    assert callable(runner.collect_program_trade_daily)


def test_run_backfill_passes_fetch_cfg_and_market_hint_to_final_write(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.backfill.price import runner as mod

    monkeypatch.setattr(
        mod, "_load_candidate_universe",
        lambda: pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]}),
    )

    def _one_window(universe, *, fetch_cfg, **kwargs):
        return [("005930", pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"), "KOSPI")]

    monkeypatch.setattr(mod, "_build_symbol_windows", _one_window)

    def _fake_fetch_one_symbol(symbol, start, end, market_hint, fetch_cfg):
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-09-02"]), "symbol": [symbol],
            "close": [70_000], "foreign_netbuy": [0.0], "inst_netbuy": [0.0], "program_netbuy": [0.0],
        })

    monkeypatch.setattr(mod, "fetch_one_symbol", _fake_fetch_one_symbol)
    monkeypatch.setattr(mod, "_merge_index_returns", lambda history, fetch_cfg: history)

    captured: dict = {}

    def _capture_to_parquet(df, parquet_path, fetch_cfg=None, market_hint=None):
        captured["fetch_cfg"] = fetch_cfg
        captured["market_hint"] = market_hint

    monkeypatch.setattr(mod, "_to_parquet", _capture_to_parquet)

    # When: run_backfill을 끝까지 실제로 실행한다(258~259번 라인이 이 호출로 실행됨)
    mod.run_backfill(
        lookback_trading_days=10, max_workers=1,
        kis_rest_limit_per_sec=20.0, kis_rest_safety_ratio=0.6, kis_max_parallel_calls=1,
        symbol_limit=None, include_symbols=None,
        parquet_out=tmp_path / "price_history.parquet",
    )

    # Then: 최종 저장 호출에 fetch_cfg와 market_hint가 실제로 전달됐다
    assert captured["fetch_cfg"] is not None
    assert captured["market_hint"] == {"005930": "KOSPI"}


def test_run_backfill_wires_corporate_action_guard_into_final_write_only() -> None:
    from pathlib import Path

    text = Path("src/backfill/price/runner.py").read_text(encoding="utf-8")

    # Then: 최종 저장 호출부에만 가드 인자가 배선된다
    assert "_to_parquet(history, parquet_path=parquet_out, fetch_cfg=fetch_cfg, market_hint=" in text

    # And: 체크포인트 중간 호출은 인자를 추가하지 않는다(라인에 fetch_cfg=가 없어야 함)
    checkpoint_line = next(
        line for line in text.splitlines() if "_to_parquet(pd.concat(checkpoint_chunks" in line
    )
    assert "fetch_cfg=" not in checkpoint_line
