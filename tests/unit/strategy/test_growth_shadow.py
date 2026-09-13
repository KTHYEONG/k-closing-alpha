def _ph(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["date", "symbol", "open", "close"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def _decisions(date, preds, tick=5.0):
    import pandas as pd
    return pd.DataFrame(
        {"decision_date": date, "symbol": [s for s, _ in preds], "pred": [p for _, p in preds], "tick_cost_bp": tick}
    )


def _realized(n_days, nets_by_day):
    import pandas as pd
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rows = []
    for i, d in enumerate(dates):
        for r, net in enumerate(nets_by_day(i), start=1):
            rows.append({"decision_date": d, "symbol": f"{r:06d}", "pred": 1.0 / r, "rank": r,
                         "gross_return": net, "net_return": net, "status": "REALIZED"})
    return pd.DataFrame(rows)


def test_realize_decision_returns_net_matches_cost_model():
    import numpy as np
    import pytest
    from src.execution.cost_model import statutory_bp_asof
    from src.strategy.growth_shadow import STATUS_REALIZED, realize_decision_returns

    # Given: 3 picks decided 2026-09-01, next trading day 2026-09-02 opens known
    decisions = _decisions("2026-09-01", [("000002", 0.02), ("000001", 0.03), ("000003", 0.01)])
    ph = _ph([
        ("2026-09-01", "000001", 990, 1000), ("2026-09-01", "000002", 1990, 2000), ("2026-09-01", "000003", 3990, 4000),
        ("2026-09-02", "000001", 1010, 1020), ("2026-09-02", "000002", 1980, 1990), ("2026-09-02", "000003", 4000, 4010),
    ])
    # When
    out = realize_decision_returns(decisions, ph).set_index("symbol")
    # Then
    stat = float(statutory_bp_asof(np.array(["2026-09-01"], dtype="datetime64[ns]"))[0])
    assert out.loc["000001", "rank"] == 1
    assert out.loc["000003", "rank"] == 3
    assert (out["status"] == STATUS_REALIZED).all()
    assert out.loc["000001", "gross_return"] == pytest.approx(0.01)
    assert out.loc["000001", "net_return"] == pytest.approx(0.01 - (stat + 2.0 * 5.0) / 1e4)
    assert out.loc["000002", "net_return"] == pytest.approx(-0.01 - (stat + 10.0) / 1e4)


def test_realize_decision_returns_marks_pending_when_next_day_not_ingested():
    import numpy as np
    from src.strategy.growth_shadow import STATUS_PENDING, realize_decision_returns

    # Given: price_history ends on the decision date
    decisions = _decisions("2026-09-01", [("000001", 0.03), ("000002", 0.02), ("000003", 0.01)])
    ph = _ph([("2026-09-01", s, 1000, 1000) for s in ("000001", "000002", "000003")])
    # When
    out = realize_decision_returns(decisions, ph)
    # Then
    assert (out["status"] == STATUS_PENDING).all()
    assert np.isnan(out["net_return"]).all()


def test_realize_decision_returns_marks_exit_unavailable_when_no_next_open():
    import numpy as np
    from src.strategy.growth_shadow import STATUS_EXIT_UNAVAILABLE, STATUS_REALIZED, realize_decision_returns

    # Given: 000002 halted (no row) and 000003 open=0 on D+1
    decisions = _decisions("2026-09-01", [("000001", 0.03), ("000002", 0.02), ("000003", 0.01)])
    ph = _ph([
        ("2026-09-01", "000001", 1000, 1000), ("2026-09-01", "000002", 1000, 1000), ("2026-09-01", "000003", 1000, 1000),
        ("2026-09-02", "000001", 1010, 1010), ("2026-09-02", "000003", 0, 1000),
    ])
    # When
    out = realize_decision_returns(decisions, ph).set_index("symbol")
    # Then
    assert out.loc["000001", "status"] == STATUS_REALIZED
    assert out.loc["000002", "status"] == STATUS_EXIT_UNAVAILABLE
    assert out.loc["000003", "status"] == STATUS_EXIT_UNAVAILABLE
    assert np.isnan(out.loc[["000002", "000003"], "net_return"]).all()


def test_realize_decision_returns_normalizes_symbol_and_categorical_dtypes():
    import pytest
    from src.strategy.growth_shadow import STATUS_REALIZED, realize_decision_returns

    # Given: production-like dtypes
    decisions = _decisions("2026-09-01", [("1", 0.03), ("2", 0.02), ("3", 0.01)])
    ph = _ph([
        ("2026-09-01", "000001", 1000, 1000), ("2026-09-01", "000002", 1000, 1000), ("2026-09-01", "000003", 1000, 1000),
        ("2026-09-02", "000001", 1100, 1000), ("2026-09-02", "000002", 1000, 1000), ("2026-09-02", "000003", 1000, 1000),
    ])
    ph["symbol"] = ph["symbol"].astype("category")
    ph["open"] = ph["open"].astype("Int32")
    ph["close"] = ph["close"].astype("Int32")
    # When
    out = realize_decision_returns(decisions, ph).set_index("symbol")
    # Then
    assert (out["status"] == STATUS_REALIZED).all()
    assert out.loc["000001", "gross_return"] == pytest.approx(0.10)


def test_realize_decision_returns_rejects_missing_columns():
    import pytest
    from src.strategy.growth_shadow import realize_decision_returns

    decisions = _decisions("2026-09-01", [("000001", 0.03)]).drop(columns=["tick_cost_bp"])
    ph = _ph([("2026-09-01", "000001", 1000, 1000)])
    with pytest.raises(ValueError, match="missing columns"):
        realize_decision_returns(decisions, ph)
    with pytest.raises(ValueError, match="missing columns"):
        realize_decision_returns(_decisions("2026-09-01", [("000001", 0.03)]), ph.drop(columns=["open"]))


def test_build_shadow_ledger_k2_is_mean_of_top2_by_rank():
    import pytest
    from src.strategy.growth_shadow import build_shadow_ledger

    # Given: one realized day with rank nets 0.03, 0.01, -0.04
    realized = _realized(1, lambda i: [0.03, 0.01, -0.04])
    # When
    ledger = build_shadow_ledger(realized)
    # Then
    row = ledger.iloc[0]
    assert row["n_picks"] == 3
    assert row["arm_k3_net"] == pytest.approx(0.0)
    assert row["arm_k2_net"] == pytest.approx(0.02)
    assert bool(row["trail_gate_open"]) is True
    assert row["arm_k3_trail_net"] == pytest.approx(0.0)


def test_build_shadow_ledger_trail_gate_is_causal_and_closes_on_negative_trailing_mean():
    import numpy as np
    import pytest
    from src.strategy.growth_shadow import TRAIL_GATE_WINDOW_DAYS, build_shadow_ledger

    n = TRAIL_GATE_WINDOW_DAYS + 1
    # Given: first 120 days lose 1%, last day earns 5%
    realized = _realized(n, lambda i: [0.05] * 3 if i == n - 1 else [-0.01] * 3)
    # When
    ledger = build_shadow_ledger(realized).reset_index(drop=True)
    # Then: warm-up rows are open and book the K3 arm
    assert np.isnan(ledger.loc[n - 2, "trail_mean_k3"])
    assert bool(ledger.loc[n - 2, "trail_gate_open"]) is True
    assert ledger.loc[n - 2, "arm_k3_trail_net"] == pytest.approx(-0.01)
    # Then: the last day sees a negative trailing mean -> gate closed -> cash
    assert ledger.loc[n - 1, "trail_mean_k3"] == pytest.approx(-0.01)
    assert bool(ledger.loc[n - 1, "trail_gate_open"]) is False
    assert ledger.loc[n - 1, "arm_k3_trail_net"] == 0.0
    # Then: causality - changing the last day's own outcome leaves its gate input unchanged
    realized2 = _realized(n, lambda i: [-0.5] * 3 if i == n - 1 else [-0.01] * 3)
    ledger2 = build_shadow_ledger(realized2).reset_index(drop=True)
    assert ledger2.loc[n - 1, "trail_mean_k3"] == pytest.approx(ledger.loc[n - 1, "trail_mean_k3"])


def test_build_shadow_ledger_exit_unavailable_yields_nan_arms():
    import numpy as np
    from src.strategy.growth_shadow import STATUS_EXIT_UNAVAILABLE, build_shadow_ledger

    realized = _realized(1, lambda i: [0.03, 0.01, -0.04])
    realized.loc[realized["rank"] == 3, ["net_return", "gross_return"]] = np.nan
    realized.loc[realized["rank"] == 3, "status"] = STATUS_EXIT_UNAVAILABLE
    row = build_shadow_ledger(realized).iloc[0]
    assert np.isnan(row["arm_k3_net"])
    assert np.isnan(row["arm_k2_net"])
    assert np.isnan(row["arm_k3_trail_net"])


def test_build_shadow_ledger_excludes_pending_dates():
    import numpy as np
    from src.strategy.growth_shadow import STATUS_PENDING, build_shadow_ledger

    realized = _realized(2, lambda i: [0.01, 0.01, 0.01])
    last = realized["decision_date"] == realized["decision_date"].max()
    realized.loc[last, "status"] = STATUS_PENDING
    realized.loc[last, ["net_return", "gross_return"]] = np.nan
    ledger = build_shadow_ledger(realized)
    assert len(ledger) == 1
    assert ledger["decision_date"].iloc[0] == realized["decision_date"].min()


def test_build_shadow_ledger_rejects_wrong_pick_count():
    import pytest
    from src.strategy.growth_shadow import build_shadow_ledger

    realized = _realized(1, lambda i: [0.01, 0.02])
    with pytest.raises(ValueError, match="MIN_TOP_K"):
        build_shadow_ledger(realized)


def test_run_growth_shadow_writes_ledger(tmp_path):
    import pandas as pd
    import pytest
    from src.strategy.growth_shadow import run_growth_shadow

    # Given
    dec_path = tmp_path / "topk_decisions.parquet"
    ph_path = tmp_path / "price_history.parquet"
    out_path = tmp_path / "growth_shadow.parquet"
    _decisions("2026-09-01", [("000001", 0.03), ("000002", 0.02), ("000003", 0.01)]).to_parquet(dec_path)
    _ph([
        ("2026-09-01", "000001", 1000, 1000), ("2026-09-01", "000002", 1000, 1000), ("2026-09-01", "000003", 1000, 1000),
        ("2026-09-02", "000001", 1030, 1000), ("2026-09-02", "000002", 1010, 1000), ("2026-09-02", "000003", 960, 1000),
    ]).to_parquet(ph_path)
    # When
    n = run_growth_shadow(decisions_path=dec_path, price_history_path=ph_path, out_path=out_path)
    # Then
    assert n == 1
    ledger = pd.read_parquet(out_path)
    assert list(ledger.columns) == [
        "decision_date", "n_picks", "arm_k3_net", "arm_k2_net", "trail_mean_k3", "trail_gate_open", "arm_k3_trail_net"
    ]
    assert ledger["arm_k2_net"].iloc[0] - ledger["arm_k3_net"].iloc[0] == pytest.approx((0.03 + 0.01) / 2 - 0.0)


def test_run_growth_shadow_missing_decisions_returns_zero(tmp_path):
    from src.strategy.growth_shadow import run_growth_shadow

    out_path = tmp_path / "growth_shadow.parquet"
    n = run_growth_shadow(
        decisions_path=tmp_path / "absent.parquet", price_history_path=tmp_path / "ph.parquet", out_path=out_path
    )
    assert n == 0
    assert not out_path.exists()
