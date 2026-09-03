# ruff: noqa: PERF401, B007
import numpy as np
import pandas as pd

from src.ml.exit_policy import attach_next_day_path


def test_attach_next_day_path_keys_next_trading_row() -> None:
    # Given a 3-day price history for one symbol and a 2-row decision frame
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    ph = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["000660"] * 3,
            "open": [100.0, 110.0, 121.0],
            "high": [105.0, 118.0, 130.0],
            "low": [98.0, 108.0, 119.0],
            "close": [102.0, 112.0, 125.0],
            "daily_change_pct": [0.02, 0.098, 0.116],
        }
    )
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "stock_code": ["000660", "000660"],
            "tag": ["first", "last"],
        }
    )

    # When
    out = attach_next_day_path(df, ph)

    # Then
    assert list(out["tag"]) == ["first", "last"]
    row0 = out.iloc[0]
    assert row0["entry_close"] == 102.0
    assert abs(row0["entry_change_ratio"] - 0.02) < 1e-9
    assert row0["nd_open"] == 110.0
    assert row0["nd_high"] == 118.0
    assert row0["nd_close"] == 112.0
    # last date for the symbol -> no subsequent row
    assert np.isnan(out.iloc[1]["nd_open"])


import pandas as pd
import pytest

from src.ml.exit_policy import attach_next_day_path


def test_attach_next_day_path_missing_column_raises() -> None:
    # Given a price history missing 'high'
    ph = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["000660", "000660"],
            "open": [100.0, 110.0],
            "low": [98.0, 108.0],
            "close": [102.0, 112.0],
            "daily_change_pct": [0.02, 0.098],
        }
    )
    df = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-02"]), "stock_code": ["000660"]})

    # When / Then
    with pytest.raises(ValueError, match="high"):
        attach_next_day_path(df, ph)


import numpy as np

from src.ml.exit_policy import simulate_take_profit_exit


def test_simulate_take_profit_gap_through_fills_at_open() -> None:
    # Given entry 100, tp 5% -> tp price 105, next open 108 (gapped through)
    entry = np.array([100.0])
    nd_open = np.array([108.0])
    nd_high = np.array([112.0])
    nd_close = np.array([101.0])

    # When
    r = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fallback="moc")

    # Then exit at the open, not the 105 limit
    assert abs(r[0] - 0.08) < 1e-9


import numpy as np

from src.ml.exit_policy import simulate_take_profit_exit


def test_simulate_take_profit_intraday_touch_fills_at_limit() -> None:
    # Given entry 100, tp 5%, open 101 (< 105), high 106 (>= 105), close 99
    entry = np.array([100.0, 100.0])
    nd_open = np.array([101.0, 101.0])
    nd_high = np.array([106.0, 106.0])
    nd_close = np.array([99.0, 99.0])

    # When (deterministic, full fill, no haircut)
    r = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fallback="moc")
    # Then
    assert np.allclose(r, 0.05)

    # And a 0.3% haircut shaves the limit fill price
    r_hc = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fallback="moc", fill_haircut=0.003)
    assert np.all(r_hc < 0.05)
    assert np.allclose(r_hc, (100.0 * 1.05 * (1 - 0.003)) / 100.0 - 1.0)


import numpy as np

from src.ml.exit_policy import simulate_take_profit_exit


def test_simulate_take_profit_no_touch_falls_back_to_moc() -> None:
    # Given entry 100, tp 5%, high 103 (< 105) -> no fill
    entry = np.array([100.0])
    nd_open = np.array([100.5])
    nd_high = np.array([103.0])
    nd_close = np.array([98.0])

    r_moc = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fallback="moc")
    r_open = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fallback="next_open")

    assert abs(r_moc[0] - (-0.02)) < 1e-9
    assert abs(r_open[0] - 0.005) < 1e-9


import numpy as np
import pytest

from src.ml.exit_policy import simulate_take_profit_exit


def test_simulate_take_profit_invalid_params_raise() -> None:
    entry = np.array([100.0])
    nd_open = np.array([101.0])
    nd_high = np.array([106.0])
    nd_close = np.array([99.0])

    with pytest.raises(ValueError, match="take_profit_pct"):
        simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.0)
    with pytest.raises(ValueError, match="take_profit_pct"):
        simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.6)
    with pytest.raises(ValueError, match="fallback"):
        simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fallback="trailing")
    with pytest.raises(ValueError, match="fill_probability"):
        simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fill_probability=1.5)
    with pytest.raises(ValueError, match="positive"):
        simulate_take_profit_exit(np.array([0.0]), nd_open, nd_high, nd_close, take_profit_pct=0.05)


import numpy as np

from src.ml.exit_policy import simulate_take_profit_exit


def test_simulate_take_profit_fill_probability_is_seed_deterministic() -> None:
    rng = np.random.default_rng(1)
    n = 400
    entry = np.full(n, 100.0)
    nd_open = np.full(n, 101.0)
    nd_high = np.full(n, 106.0)  # every day touches 105
    nd_close = np.full(n, 97.0)

    a = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fill_probability=0.7, seed=42)
    b = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fill_probability=0.7, seed=42)
    full = simulate_take_profit_exit(entry, nd_open, nd_high, nd_close, take_profit_pct=0.05, fill_probability=1.0)

    assert np.array_equal(a, b)
    # ~70% of days fill at +0.05, the rest at moc (-0.03)
    assert full.mean() > a.mean() > -0.03
    assert 0.55 < np.mean(np.isclose(a, 0.05)) < 0.85


import numpy as np
import pandas as pd

from src.ml.exit_policy import evaluate_exit_grid
from src.ml.robust_eval import CombinatorialPurgedCV


def _grid_synth(n_days: int = 60, per_day: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    all_dates = pd.bdate_range("2023-01-02", periods=n_days + 2)
    oof_rows = []
    for d in dates:
        for j in range(per_day):
            pred = rng.normal()
            oof_rows.append(
                {"trade_date": d, "stock_code": f"{j:06d}", "pred": pred, "net_return": 100 * (0.004 + 0.01 * pred)}
            )
    oof = pd.DataFrame(oof_rows)
    # Every (date, code): entry close 100 on every entry date; next day opens +0.6%,
    # spikes through +5% intraday (high ~ +6%), then closes back near flat.
    ph_rows = []
    for code in [f"{j:06d}" for j in range(per_day)]:
        for d in all_dates:
            ph_rows.append(
                {
                    "date": d,
                    "symbol": code,
                    "open": 100.6 + rng.normal(0, 0.05),
                    "high": 106.0 + rng.normal(0, 0.2),
                    "low": 97.0,
                    "close": 100.0 + rng.normal(0, 0.1),
                    "daily_change_pct": 0.03,
                }
            )
    return oof, pd.DataFrame(ph_rows)


def test_evaluate_exit_grid_promotes_dominant_take_profit() -> None:
    # Given a guard-valid CPCV config (n_groups >= k_test*(1+purge+embargo)+1)
    oof, ph = _grid_synth()
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2)

    # When
    results = evaluate_exit_grid(
        oof, ph, cost_ratio=0.002, take_profit_grid=(0.03, 0.05, 0.07), cv=cv, alpha=0.10
    )

    # Then the 5% rule beats 'sell at next open' with a significant, unanimous-path edge
    r5 = {round(r.take_profit_pct, 2): r for r in results}[0.05]
    assert r5.delta_vs_incumbent > 0
    assert r5.p_value < 0.10
    assert len(r5.cpcv_path_deltas) == len(list(cv.split(np.repeat(np.arange(8), 10))))
    assert min(r5.cpcv_path_deltas) > 0
    assert r5.promoted is True
    assert r5.n_days == 60


import numpy as np
import pandas as pd

from src.ml.exit_policy import evaluate_exit_grid


def test_evaluate_exit_grid_rejects_when_moc_fallback_bleeds() -> None:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=50)
    all_dates = pd.bdate_range("2023-01-02", periods=52)
    oof_rows, ph_rows = [], []
    for d in dates:
        for j in range(3):
            oof_rows.append(
                {"trade_date": d, "stock_code": f"{j:06d}", "pred": rng.normal(), "net_return": rng.normal()}
            )
    for code in [f"{j:06d}" for j in range(3)]:
        for d in all_dates:
            # entry close 100 every day; next open only +0.6%, high never reaches +5%,
            # and the fallback close (== 100) leaves the MOC exit below the incumbent open.
            ph_rows.append(
                {
                    "date": d, "symbol": code,
                    "open": 100.6 + rng.normal(0, 0.05),
                    "high": 101.5 + rng.normal(0, 0.1),
                    "low": 95.0,
                    "close": 100.0 + rng.normal(0, 0.05),
                    "daily_change_pct": 0.02,
                }
            )
    oof = pd.DataFrame(oof_rows)
    ph = pd.DataFrame(ph_rows)

    results = evaluate_exit_grid(oof, ph, cost_ratio=0.002, take_profit_grid=(0.05,), cv=None, alpha=0.10)

    assert results[0].delta_vs_incumbent < 0
    assert results[0].promoted is False
    assert results[0].cpcv_path_deltas == ()


import numpy as np
import pandas as pd

from src.ml.exit_policy import evaluate_exit_grid


def test_evaluate_exit_grid_excludes_limit_up_entries() -> None:
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2023-01-02", periods=45)
    all_dates = pd.bdate_range("2023-01-02", periods=47)
    oof_rows, ph_rows = [], []
    # exactly one candidate per day so the top-1 is forced
    for i, d in enumerate(dates):
        oof_rows.append({"trade_date": d, "stock_code": "000000", "pred": 1.0, "net_return": 0.5})
    for k, d in enumerate(all_dates):
        # first 10 entry days are limit-up locked (0.30), rest normal (0.03)
        chg = 0.30 if k < 10 else 0.03
        ph_rows.append(
            {"date": d, "symbol": "000000", "open": 100.4, "high": 106.0, "low": 97.0,
             "close": 100.0, "daily_change_pct": chg}
        )
    oof = pd.DataFrame(oof_rows)
    ph = pd.DataFrame(ph_rows)

    results = evaluate_exit_grid(oof, ph, cost_ratio=0.002, take_profit_grid=(0.05,), cv=None)

    # 45 decision days minus the 10 limit-up entry days
    assert results[0].n_days == 35


import numpy as np
import pandas as pd
import pytest

from src.ml.exit_policy import evaluate_exit_grid


def test_evaluate_exit_grid_requires_minimum_days() -> None:
    dates = pd.bdate_range("2023-01-02", periods=20)
    all_dates = pd.bdate_range("2023-01-02", periods=22)
    oof = pd.DataFrame(
        {"trade_date": dates, "stock_code": ["000000"] * 20, "pred": [1.0] * 20, "net_return": [0.3] * 20}
    )
    ph = pd.DataFrame(
        {
            "date": all_dates, "symbol": ["000000"] * 22,
            "open": [100.4] * 22, "high": [106.0] * 22, "low": [97.0] * 22,
            "close": [100.0] * 22, "daily_change_pct": [0.03] * 22,
        }
    )

    with pytest.raises(ValueError, match="30"):
        evaluate_exit_grid(oof, ph, cost_ratio=0.002, take_profit_grid=(0.05,), cv=None)


import json

from src.ml.exit_policy import ExitRuleResult, summarize_exit_grid


def _mk(tp: float, mean_net: float, promoted: bool) -> ExitRuleResult:
    return ExitRuleResult(
        take_profit_pct=tp, fallback="moc", n_days=100, candidate_mean_net=mean_net,
        incumbent_mean_net=0.003, candidate_sharpe=0.2, candidate_win_rate=0.6,
        delta_vs_incumbent=mean_net - 0.003, p_value=0.01 if promoted else 0.4,
        ci_low=0.0, ci_high=0.02, cpcv_path_deltas=(0.001, 0.002), promoted=promoted,
    )


def test_summarize_exit_grid_selects_best_promoted() -> None:
    results = (_mk(0.03, 0.010, True), _mk(0.05, 0.014, True), _mk(0.07, 0.016, False))
    summary = summarize_exit_grid(results)

    assert summary["incumbent_mean_net"] == 0.003
    assert summary["n_days"] == 100
    assert summary["best"]["take_profit_pct"] == 0.05
    assert isinstance(summary["grid"][0]["cpcv_path_deltas"], list)
    json.dumps(summary)  # must be serialisable for provenance

    none_promoted = (_mk(0.03, 0.010, False), _mk(0.05, 0.014, False))
    assert summarize_exit_grid(none_promoted)["best"] is None


