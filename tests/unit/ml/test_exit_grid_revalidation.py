"""Locked-OOS exit-grid revalidation (synthetic panel)."""
from __future__ import annotations


def _synthetic_panel(n_days: int = 45):
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel

    dates = pd.bdate_range("2023-02-01", periods=n_days)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(12):
        base = 18000.0 if i < 8 else 30000.0
        for d in dates:
            prev = base / 1.05
            rows.append({
                "date": d, "symbol": f"{i:06d}", "open": prev, "high": base * 1.01,
                "low": prev * 0.99, "close": base, "prev_close": prev, "volume": 1e6,
                "market_cap_100m": 3000.0, "trade_value_100m": 500.0,
                "market": "KOSPI" if i % 2 == 0 else "KOSDAQ", "daily_change_pct": 0.05,
                "inst_netbuy": float(rng.integers(-10**8, 10**8)),
                "foreign_netbuy": float(rng.integers(-10**8, 10**8)),
                "program_netbuy": 0.0, "kospi_pct": 0.001, "kosdaq_pct": 0.002,
                "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
    ph, _prov = prepare_price_panel(pd.DataFrame(rows))
    market_dates = np.array(sorted(ph["date"].unique()))
    return ph, market_dates, {d: i for i, d in enumerate(market_dates)}


def test_build_exit_grid_oof_returns_expected_shape_and_cost_ratio() -> None:
    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.research.exit_grid_revalidation import build_exit_grid_oof
    from src.ml.topk_ranker_research import CERT_REGIME_START

    ph, market_dates, d_to_idx = _synthetic_panel(45)
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=1)

    oof_df, cost_ratio = build_exit_grid_oof(
        ph, market_dates, d_to_idx, cv=cv, train_start=CERT_REGIME_START, min_train_rows=10,
    )

    # Then: exactly the columns evaluate_exit_grid expects, and a real (non-zero,
    # non-fabricated) cost_ratio derived from the same population's tick_cost_bp.
    assert list(oof_df.columns) == ["trade_date", "stock_code", "pred", "net_return"]
    assert len(oof_df) == 352
    assert oof_df["trade_date"].nunique() == 44
    assert cost_ratio == 0.0005555555555555557


def test_build_exit_grid_oof_rejects_top_k_below_minimum() -> None:
    import dataclasses

    import pytest

    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.research.exit_grid_revalidation import build_exit_grid_oof
    from src.strategy.contract import KCA_TOPK_COSTAWARE_001

    ph, market_dates, d_to_idx = _synthetic_panel(45)
    bad_spec = dataclasses.replace(KCA_TOPK_COSTAWARE_001, top_k=MIN_TOP_K - 1)

    with pytest.raises(ValueError, match="below the minimum investable K"):
        build_exit_grid_oof(ph, market_dates, d_to_idx, spec=bad_spec, min_train_rows=10)


def test_run_exit_grid_revalidation_runs_end_to_end_with_injected_panel() -> None:
    from src.ml.research.exit_grid_revalidation import run_exit_grid_revalidation
    from src.ml.topk_ranker_research import CERT_REGIME_START

    ph, market_dates, d_to_idx = _synthetic_panel(45)

    summary = run_exit_grid_revalidation(
        ph=ph, market_dates=market_dates, d_to_idx=d_to_idx,
        train_start=CERT_REGIME_START, min_train_rows=10,
    )

    # Then: a real CPCV(8,2) run (matching the certified ranker's own config),
    # clearing evaluate_exit_grid's n_days>=30 floor, with cost_ratio/grid populated.
    assert summary["cv_n_groups"] == 8
    assert summary["cv_k_test"] == 2
    assert summary["n_days"] == 44
    assert summary["cost_ratio"] == 0.0005555555555555557
    assert isinstance(summary["grid"], list) and len(summary["grid"]) == 5


def test_run_exit_grid_revalidation_loads_default_panel_when_not_injected(monkeypatch) -> None:
    import src.ml.research.exit_grid_revalidation as exit_grid_mod
    from src import settings
    from src.ml.topk_ranker_research import CERT_REGIME_START

    ph, market_dates, d_to_idx = _synthetic_panel(45)
    seen: dict[str, object] = {}

    def _fake_load(path):
        seen["path"] = path
        return ph, market_dates, d_to_idx

    monkeypatch.setattr(exit_grid_mod, "load_and_prepare_price_history", _fake_load)
    monkeypatch.setattr(settings, "PRICE_HISTORY_PARQUET_PATH", "sentinel/path.parquet")

    summary = exit_grid_mod.run_exit_grid_revalidation(train_start=CERT_REGIME_START, min_train_rows=10)

    # Then: the default (ph=None) branch loaded via settings.PRICE_HISTORY_PARQUET_PATH,
    # and the loaded synthetic panel flows through to a real result.
    assert seen["path"] == "sentinel/path.parquet"
    assert summary["n_days"] == 44



def test_run_exit_grid_revalidation_writes_export_parquet(tmp_path) -> None:
    import pandas as pd

    from src.ml.research.exit_grid_revalidation import run_exit_grid_revalidation
    from src.ml.topk_ranker_research import CERT_REGIME_START

    ph, market_dates, d_to_idx = _synthetic_panel(45)

    summary = run_exit_grid_revalidation(
        ph=ph, market_dates=market_dates, d_to_idx=d_to_idx,
        train_start=CERT_REGIME_START, min_train_rows=10, export_dir=str(tmp_path),
    )

    out_path = tmp_path / "exit_grid_revalidation_report.parquet"
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert len(written) == len(summary["grid"]) == 5



def test_exit_grid_revalidation_main_calls_run_with_default_export_dir(monkeypatch) -> None:
    import src.ml.research.exit_grid_revalidation as exit_grid_mod

    seen: dict[str, object] = {}

    def _fake_run(*, export_dir=None, **kwargs):
        seen["export_dir"] = export_dir
        return {"n_days": 1, "cost_ratio": 0.0, "incumbent_mean_net": 0.0, "grid": [], "best": None}

    monkeypatch.setattr(exit_grid_mod, "run_exit_grid_revalidation", _fake_run)

    exit_grid_mod.main()

    assert seen["export_dir"] == "artifacts/models"

