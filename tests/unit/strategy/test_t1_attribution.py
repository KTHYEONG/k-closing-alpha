from __future__ import annotations


def test_realize_pool_exit_arms_scores_open_and_take_profit_arms() -> None:
    import numpy as np

    import pandas as pd

    # Given: 결정일 09-10 5종목(종가 10,000원 동일, 틱비용 0), 09-11 익일 OHLC, 09-11 결정분은 익일 시세 없음
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    pool = pd.DataFrame({
        "decision_date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "pred": [0.05, 0.04, 0.03, 0.02, 0.01] * 2,
        "tick_cost_bp": [0.0] * 10,
        "admitted": [True, True, True, True, False] * 2,
        "selected": [True, True, True, False, False] * 2,
        "model_version": ["KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"] * 10,
    })
    price_history = pd.DataFrame({
        "date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "open": [10_000.0] * 5 + [10_300.0, 10_200.0, 10_100.0, 10_000.0, 9_900.0],
        "high": [10_000.0] * 5 + [10_600.0, 10_300.0, 10_200.0, 10_100.0, 10_000.0],
        "close": [10_000.0] * 5 + [10_100.0, 9_900.0, 10_000.0, 9_800.0, 9_700.0],
    })

    from src.strategy.t1_attribution import REALIZED_COLUMNS, realize_pool_exit_arms

    # When
    out = realize_pool_exit_arms(pool, price_history)

    # Then
    assert tuple(out.columns) == REALIZED_COLUMNS
    day1 = out[out["decision_date"] == pd.Timestamp("2026-09-10")].set_index("symbol")
    assert (day1["status"] == "REALIZED").all()
    # 시가 청산: 시가/전일종가 - 1, 2026년 법정비용 20bp 차감(틱비용 0)
    np.testing.assert_allclose(day1["open_gross"].to_numpy(), [0.03, 0.02, 0.01, 0.0, -0.01], atol=1e-12)
    np.testing.assert_allclose(day1["open_net"].to_numpy(), [0.028, 0.018, 0.008, -0.002, -0.012], atol=1e-12)
    # TP5%+MOC: 000001은 고가 10,600 >= 10,500 익절, 나머지는 종가(MOC)
    np.testing.assert_allclose(day1["tp_gross"].to_numpy(), [0.05, -0.01, 0.0, -0.02, -0.03], atol=1e-12)
    np.testing.assert_allclose(day1["tp_net"].to_numpy(), [0.048, -0.012, -0.002, -0.022, -0.032], atol=1e-12)
    assert day1["rank"].tolist() == [1, 2, 3, 4, 5]
    day2 = out[out["decision_date"] == pd.Timestamp("2026-09-11")]
    assert (day2["status"] == "PENDING").all()
    assert day2[["open_gross", "open_net", "tp_gross", "tp_net"]].isna().all().all()


def test_realize_pool_exit_arms_flags_price_discontinuity_and_missing_bars() -> None:

    import pandas as pd

    # Given: 결정일 09-10 5종목(종가 10,000원 동일, 틱비용 0), 09-11 익일 OHLC, 09-11 결정분은 익일 시세 없음
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    pool = pd.DataFrame({
        "decision_date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "pred": [0.05, 0.04, 0.03, 0.02, 0.01] * 2,
        "tick_cost_bp": [0.0] * 10,
        "admitted": [True, True, True, True, False] * 2,
        "selected": [True, True, True, False, False] * 2,
        "model_version": ["KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"] * 10,
    })
    price_history = pd.DataFrame({
        "date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "open": [10_000.0] * 5 + [10_300.0, 10_200.0, 10_100.0, 10_000.0, 9_900.0],
        "high": [10_000.0] * 5 + [10_600.0, 10_300.0, 10_200.0, 10_100.0, 10_000.0],
        "close": [10_000.0] * 5 + [10_100.0, 9_900.0, 10_000.0, 9_800.0, 9_700.0],
    })

    from src.strategy.t1_attribution import STATUS_PRICE_DISCONTINUITY, realize_pool_exit_arms

    # Given: 000004 익일 시가 +50%(액면분할 등 비수정 가격), 000005 익일 봉 누락
    ph = price_history.copy()
    ph.loc[(ph["date"] == "2026-09-11") & (ph["symbol"] == "000004"), ["open", "high", "close"]] = [15_000.0, 15_000.0, 15_000.0]
    ph = ph[~((ph["date"] == "2026-09-11") & (ph["symbol"] == "000005"))]

    # When
    out = realize_pool_exit_arms(pool[pool["decision_date"] == "2026-09-10"], ph).set_index("symbol")

    # Then
    assert out.loc["000004", "status"] == STATUS_PRICE_DISCONTINUITY
    assert out.loc[["000004"], ["open_gross", "open_net", "tp_gross", "tp_net"]].isna().all().all()
    assert out.loc["000005", "status"] == "EXIT_UNAVAILABLE"
    assert out.loc[["000005"], ["open_net", "tp_net"]].isna().all().all()
    assert out.loc["000001", "status"] == "REALIZED"


def test_realize_pool_exit_arms_fails_closed_on_schema_and_duplicates() -> None:
    import pytest

    import pandas as pd

    # Given: 결정일 09-10 5종목(종가 10,000원 동일, 틱비용 0), 09-11 익일 OHLC, 09-11 결정분은 익일 시세 없음
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    pool = pd.DataFrame({
        "decision_date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "pred": [0.05, 0.04, 0.03, 0.02, 0.01] * 2,
        "tick_cost_bp": [0.0] * 10,
        "admitted": [True, True, True, True, False] * 2,
        "selected": [True, True, True, False, False] * 2,
        "model_version": ["KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"] * 10,
    })
    price_history = pd.DataFrame({
        "date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "open": [10_000.0] * 5 + [10_300.0, 10_200.0, 10_100.0, 10_000.0, 9_900.0],
        "high": [10_000.0] * 5 + [10_600.0, 10_300.0, 10_200.0, 10_100.0, 10_000.0],
        "close": [10_000.0] * 5 + [10_100.0, 9_900.0, 10_000.0, 9_800.0, 9_700.0],
    })

    from src.strategy.t1_attribution import realize_pool_exit_arms

    # When / Then
    with pytest.raises(ValueError, match="pool missing columns"):
        realize_pool_exit_arms(pool.drop(columns=["admitted"]), price_history)
    with pytest.raises(ValueError, match="price_history missing columns"):
        realize_pool_exit_arms(pool, price_history.drop(columns=["high"]))
    with pytest.raises(ValueError, match="duplicate"):
        realize_pool_exit_arms(pd.concat([pool, pool.iloc[[0]]], ignore_index=True), price_history)


def test_build_attribution_ledger_computes_daily_rank_ic_and_basket_arms() -> None:
    import numpy as np
    import pytest

    import pandas as pd

    # Given: 결정일 09-10 5종목(종가 10,000원 동일, 틱비용 0), 09-11 익일 OHLC, 09-11 결정분은 익일 시세 없음
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    pool = pd.DataFrame({
        "decision_date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "pred": [0.05, 0.04, 0.03, 0.02, 0.01] * 2,
        "tick_cost_bp": [0.0] * 10,
        "admitted": [True, True, True, True, False] * 2,
        "selected": [True, True, True, False, False] * 2,
        "model_version": ["KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"] * 10,
    })
    price_history = pd.DataFrame({
        "date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "open": [10_000.0] * 5 + [10_300.0, 10_200.0, 10_100.0, 10_000.0, 9_900.0],
        "high": [10_000.0] * 5 + [10_600.0, 10_300.0, 10_200.0, 10_100.0, 10_000.0],
        "close": [10_000.0] * 5 + [10_100.0, 9_900.0, 10_000.0, 9_800.0, 9_700.0],
    })

    from src.strategy.t1_attribution import T1_LEDGER_COLUMNS, build_attribution_ledger, realize_pool_exit_arms

    # When
    ledger = build_attribution_ledger(realize_pool_exit_arms(pool, price_history))

    # Then
    assert tuple(ledger.columns) == T1_LEDGER_COLUMNS
    assert ledger["day_status"].tolist() == ["SETTLED", "PENDING"]
    d1 = ledger.iloc[0]
    assert (d1["n_pool"], d1["n_pool_realized"], d1["n_admitted"], d1["n_admitted_realized"], d1["n_selected"], d1["n_selected_realized"]) == (5, 5, 4, 4, 3, 3)
    assert d1["ic_pool_net"] == pytest.approx(1.0)
    assert d1["ic_admitted_net"] == pytest.approx(1.0)
    assert d1["selected_open_net"] == pytest.approx(0.018)
    assert d1["selected_tp_net"] == pytest.approx((0.048 - 0.012 - 0.002) / 3)
    assert d1["admitted_open_net"] == pytest.approx(0.013)
    assert d1["selection_edge_net"] == pytest.approx(0.005)
    assert d1["model_version"] == "KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"
    d2 = ledger.iloc[1]
    assert d2["n_pool"] == 5 and d2["n_pool_realized"] == 0
    assert np.isnan(d2[["ic_pool_net", "ic_admitted_net", "selected_open_net", "selected_tp_net", "admitted_open_net", "selection_edge_net"]].to_numpy(dtype=float)).all()


def test_build_attribution_ledger_guards_small_pools_partial_baskets_and_mixed_versions() -> None:
    import numpy as np
    import pytest

    import pandas as pd

    # Given: 결정일 09-10 5종목(종가 10,000원 동일, 틱비용 0), 09-11 익일 OHLC, 09-11 결정분은 익일 시세 없음
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    pool = pd.DataFrame({
        "decision_date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "pred": [0.05, 0.04, 0.03, 0.02, 0.01] * 2,
        "tick_cost_bp": [0.0] * 10,
        "admitted": [True, True, True, True, False] * 2,
        "selected": [True, True, True, False, False] * 2,
        "model_version": ["KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"] * 10,
    })
    price_history = pd.DataFrame({
        "date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "open": [10_000.0] * 5 + [10_300.0, 10_200.0, 10_100.0, 10_000.0, 9_900.0],
        "high": [10_000.0] * 5 + [10_600.0, 10_300.0, 10_200.0, 10_100.0, 10_000.0],
        "close": [10_000.0] * 5 + [10_100.0, 9_900.0, 10_000.0, 9_800.0, 9_700.0],
    })

    from src.strategy.t1_attribution import build_attribution_ledger, realize_pool_exit_arms

    # Given: 선택 종목 000002 익일 봉 누락 + 000004/000005 누락 -> 실현 3종목(< 4)
    ph = price_history[~((price_history["date"] == "2026-09-11") & price_history["symbol"].isin(["000002", "000004", "000005"]))]
    day1 = pool[pool["decision_date"] == "2026-09-10"]

    # When
    ledger = build_attribution_ledger(realize_pool_exit_arms(day1, ph))

    # Then
    row = ledger.iloc[0]
    assert row["n_pool_realized"] == 2
    assert np.isnan(row["ic_pool_net"]) and np.isnan(row["ic_admitted_net"])
    assert np.isnan(row["selected_open_net"]) and np.isnan(row["selected_tp_net"])
    assert row["admitted_open_net"] == pytest.approx((0.028 + 0.008) / 2)

    # And: 한 결정일에 모델 버전이 섞이면 귀속 불가 -> fail-closed
    mixed = day1.copy()
    mixed.loc[mixed.index[0], "model_version"] = "OTHER@x@y"
    with pytest.raises(ValueError, match="mixes model versions"):
        build_attribution_ledger(realize_pool_exit_arms(mixed, price_history))


def test_summarize_attribution_reports_mean_sd_t_and_exit_gap() -> None:
    import math

    import numpy as np
    import pandas as pd
    import pytest

    from src.strategy.t1_attribution import summarize_attribution

    # Given: IC 0.1/0.3/NaN, 청산 비교 가능일 2일
    ledger = pd.DataFrame({
        "ic_pool_net": [0.1, 0.3, np.nan],
        "ic_admitted_net": [0.2, np.nan, np.nan],
        "selected_open_net": [0.010, 0.020, np.nan],
        "selected_tp_net": [0.004, 0.012, 0.050],
    })

    # When
    s = summarize_attribution(ledger)

    # Then
    assert s["ic_pool_net_days"] == 2
    assert s["ic_pool_net_mean"] == pytest.approx(0.2)
    assert s["ic_pool_net_sd"] == pytest.approx(math.sqrt(0.02))
    assert s["ic_pool_net_t"] == pytest.approx(0.2 / math.sqrt(0.02) * math.sqrt(2))
    assert s["ic_admitted_net_days"] == 1
    assert math.isnan(s["ic_admitted_net_sd"]) and math.isnan(s["ic_admitted_net_t"])
    assert s["exit_compare_days"] == 2
    assert s["selected_open_net_mean_bp"] == pytest.approx(150.0)
    assert s["selected_tp_net_mean_bp"] == pytest.approx(80.0)
    assert s["tp_minus_open_mean_bp"] == pytest.approx(-70.0)
    assert s["tp_minus_open_t"] == pytest.approx(-0.007 / math.sqrt(0.000002) * math.sqrt(2))


def test_run_t1_attribution_writes_ledger_and_skips_without_pool(tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    # Given: 결정일 09-10 5종목(종가 10,000원 동일, 틱비용 0), 09-11 익일 OHLC, 09-11 결정분은 익일 시세 없음
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    pool = pd.DataFrame({
        "decision_date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "pred": [0.05, 0.04, 0.03, 0.02, 0.01] * 2,
        "tick_cost_bp": [0.0] * 10,
        "admitted": [True, True, True, True, False] * 2,
        "selected": [True, True, True, False, False] * 2,
        "model_version": ["KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"] * 10,
    })
    price_history = pd.DataFrame({
        "date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "open": [10_000.0] * 5 + [10_300.0, 10_200.0, 10_100.0, 10_000.0, 9_900.0],
        "high": [10_000.0] * 5 + [10_600.0, 10_300.0, 10_200.0, 10_100.0, 10_000.0],
        "close": [10_000.0] * 5 + [10_100.0, 9_900.0, 10_000.0, 9_800.0, 9_700.0],
    })

    from src.strategy.t1_attribution import T1_LEDGER_COLUMNS, run_t1_attribution

    pool_path = tmp_path / "rank_pool_predictions.parquet"
    ph_path = tmp_path / "price_history.parquet"
    out_path = tmp_path / "t1_attribution.parquet"
    price_history.to_parquet(ph_path)

    # When: 풀 저장소 없음
    with caplog.at_level(logging.WARNING, logger="src.strategy.t1_attribution"):
        assert run_t1_attribution(pool_path, ph_path, out_path) == 0
    # Then
    assert not out_path.exists()
    assert "NO_POOL" in caplog.text

    # When: 풀 저장(추가 컬럼 포함)
    pool.assign(rank=1, name="x").to_parquet(pool_path)
    n = run_t1_attribution(pool_path, ph_path, out_path)

    # Then: 결정일 2개 -> 원장 2행, 재실행해도 동일(멱등)
    assert n == 2
    first = pd.read_parquet(out_path)
    assert tuple(first.columns) == T1_LEDGER_COLUMNS
    assert run_t1_attribution(pool_path, ph_path, out_path) == 2
    pd.testing.assert_frame_equal(pd.read_parquet(out_path), first)


def test_attribution_handles_no_pick_days_and_empty_inputs() -> None:
    import math

    import numpy as np
    import pytest

    import pandas as pd

    # Given: 결정일 09-10 5종목(종가 10,000원 동일, 틱비용 0), 09-11 익일 OHLC, 09-11 결정분은 익일 시세 없음
    symbols = ["000001", "000002", "000003", "000004", "000005"]
    pool = pd.DataFrame({
        "decision_date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "pred": [0.05, 0.04, 0.03, 0.02, 0.01] * 2,
        "tick_cost_bp": [0.0] * 10,
        "admitted": [True, True, True, True, False] * 2,
        "selected": [True, True, True, False, False] * 2,
        "model_version": ["KCA-TOPK-COSTAWARE-001@2026-09-04 00:00:00@UNKNOWN"] * 10,
    })
    price_history = pd.DataFrame({
        "date": ["2026-09-10"] * 5 + ["2026-09-11"] * 5,
        "symbol": symbols * 2,
        "open": [10_000.0] * 5 + [10_300.0, 10_200.0, 10_100.0, 10_000.0, 9_900.0],
        "high": [10_000.0] * 5 + [10_600.0, 10_300.0, 10_200.0, 10_100.0, 10_000.0],
        "close": [10_000.0] * 5 + [10_100.0, 9_900.0, 10_000.0, 9_800.0, 9_700.0],
    })

    from src.strategy.t1_attribution import REALIZED_COLUMNS, T1_LEDGER_COLUMNS, build_attribution_ledger, realize_pool_exit_arms, summarize_attribution

    # Given: admitted < top_k 로 선정 없이 풀만 저장된 날
    no_pick = pool[pool["decision_date"] == "2026-09-10"].assign(selected=False)

    # When
    row = build_attribution_ledger(realize_pool_exit_arms(no_pick, price_history)).iloc[0]

    # Then: 풀 IC와 벤치마크는 산출, 바스켓 지표는 선정이 없어 NaN
    assert row["day_status"] == "SETTLED"
    assert row["n_selected"] == 0
    assert row["ic_pool_net"] == pytest.approx(1.0)
    assert row["admitted_open_net"] == pytest.approx(0.013)
    assert np.isnan(row["selected_open_net"]) and np.isnan(row["selected_tp_net"]) and np.isnan(row["selection_edge_net"])

    # When: 풀 저장소가 비어 있는 경우
    empty = build_attribution_ledger(pd.DataFrame(columns=list(REALIZED_COLUMNS)))
    summary = summarize_attribution(empty)

    # Then
    assert empty.empty
    assert tuple(empty.columns) == T1_LEDGER_COLUMNS
    assert summary["ic_pool_net_days"] == 0 and summary["exit_compare_days"] == 0
    assert math.isnan(summary["ic_pool_net_mean"]) and math.isnan(summary["ic_pool_net_sd"]) and math.isnan(summary["ic_pool_net_t"])
    assert math.isnan(summary["tp_minus_open_mean_bp"]) and math.isnan(summary["tp_minus_open_t"])
