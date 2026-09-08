from __future__ import annotations


def test_cost_spec_rejects_zero_tick_round_trip() -> None:
    # Given / When / Then
    import pytest

    from src.strategy.contract import CostSpec, ExecutionMode

    with pytest.raises(ValueError, match="round_trip_ticks"):
        CostSpec(mode=ExecutionMode.AA, statutory_bp=20.0, round_trip_ticks=0.0)

    with pytest.raises(ValueError, match="round_trip_ticks"):
        CostSpec(mode=ExecutionMode.AA, statutory_bp=20.0, round_trip_ticks=1.0)

    with pytest.raises(ValueError, match="round_trip_ticks"):
        CostSpec(mode=ExecutionMode.PA, statutory_bp=20.0, round_trip_ticks=0.0)

    with pytest.raises(ValueError, match="statutory_bp"):
        CostSpec(mode=ExecutionMode.AA, statutory_bp=-1.0, round_trip_ticks=2.0)

    # PA legitimately crosses only the exit leg
    pa = CostSpec(mode=ExecutionMode.PA, statutory_bp=20.0, round_trip_ticks=1.0)
    assert pa.round_trip_ticks == 1.0


def test_round_trip_cost_bp_reproduces_measured_u0_level() -> None:
    # Given
    import numpy as np

    from src.strategy.contract import AA_COST, PA_COST, round_trip_cost_bp

    price = np.array([10000.0, 1500.0, 30000.0], dtype=np.float64)

    # When
    aa = round_trip_cost_bp(price, AA_COST)
    pa = round_trip_cost_bp(price, PA_COST)

    # Then: tick ladder is 10 / 1 / 50 for these prices
    np.testing.assert_allclose(aa, [20.0 + 20.0, 20.0 + 2.0 * 1.0 / 1500.0 * 1e4, 20.0 + 2.0 * 50.0 / 30000.0 * 1e4])
    np.testing.assert_allclose(pa, [20.0 + 10.0, 20.0 + 1.0 * 1.0 / 1500.0 * 1e4, 20.0 + 1.0 * 50.0 / 30000.0 * 1e4])
    assert np.all(aa > pa)


def test_round_trip_cost_bp_propagates_nan_for_invalid_price() -> None:
    # Given
    import numpy as np

    from src.strategy.contract import AA_COST, round_trip_cost_bp

    price = np.array([0.0, -100.0, np.nan, np.inf, 10000.0], dtype=np.float64)

    # When
    cost = round_trip_cost_bp(price, AA_COST)

    # Then
    assert np.isnan(cost[:4]).all()
    assert np.isfinite(cost[4])
    assert not np.any(cost[:4] == 0.0)


def test_select_universe_applies_half_open_bounds_and_liquidity_floors() -> None:
    # Given
    import numpy as np
    import pandas as pd

    from src.strategy.contract import select_universe

    df = pd.DataFrame(
        {
            "chg_ratio": [0.02, 0.0999, 0.10, 0.0199, 0.05, 0.05, 0.05, 0.05, 0.05, np.nan],
            "tv_clean": [100.0, 500.0, 500.0, 500.0, 99.9, 500.0, 500.0, 500.0, 500.0, 500.0],
            "mc_clean": [500.0, 800.0, 800.0, 800.0, 800.0, 499.9, 800.0, 800.0, 800.0, 800.0],
            "close": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.0, 1000.0, 1000.0, 1000.0],
            "volume": [10, 10, 10, 10, 10, 10, 10, 0, 10, 10],
            "is_ceiling": [False, False, False, False, False, False, False, False, True, False],
        }
    )

    # When
    mask = select_universe(df)

    # Then
    assert mask.dtype == bool
    assert mask.tolist() == [True, True, False, False, False, False, False, False, False, False]


def test_select_universe_raises_on_missing_required_columns() -> None:
    # Given
    import pandas as pd
    import pytest

    from src.strategy.contract import UniverseSpec, select_universe

    df = pd.DataFrame({"chg_ratio": [0.05], "close": [1000.0], "volume": [10]})

    # When / Then
    with pytest.raises(ValueError) as exc:  # noqa: PT011 - contract skeleton asserts fail-closed columns
        select_universe(df)
    message = str(exc.value)
    assert "tv_clean" in message
    assert "mc_clean" in message
    assert "is_ceiling" in message

    # exclude_ceiling=False drops only the is_ceiling requirement
    with pytest.raises(ValueError) as exc2:  # noqa: PT011 - contract skeleton asserts fail-closed columns
        select_universe(df, UniverseSpec(exclude_ceiling=False))
    assert "is_ceiling" not in str(exc2.value)


def test_mark_ceiling_flags_only_limit_up_closes() -> None:
    # Given
    import pandas as pd
    import pytest

    from src.strategy.contract import mark_ceiling

    df = pd.DataFrame(
        {
            "chg_ratio": [0.30, 0.29, 0.30, 0.28, 0.05],
            "close": [1300.0, 1290.0, 1250.0, 1280.0, 1050.0],
            "high": [1300.0, 1290.0, 1300.0, 1280.0, 1060.0],
        }
    )

    # When
    flags = mark_ceiling(df)

    # Then
    assert flags.tolist() == [True, True, False, False, False]

    with pytest.raises(ValueError, match="high"):
        mark_ceiling(df.drop(columns=["high"]))


def test_classify_rank_feature_is_mismatch_despite_perfect_correlation() -> None:
    # Given / When
    from src.strategy.contract import classify_feature_contract

    unproven = classify_feature_contract(
        "chg_rank",
        "groupby(date).rank(pct=True) over U0",
        "change_rate_pct_rank",
        spearman=1.0,
        is_cross_sectional_rank=True,
        populations_match=None,
    )
    disjoint = classify_feature_contract(
        "tv_rank",
        "groupby(date).rank(pct=True) over U0",
        "trade_value_pct_rank",
        spearman=1.0,
        is_cross_sectional_rank=True,
        populations_match=False,
    )
    aligned = classify_feature_contract(
        "inst_rank",
        "groupby(date).rank(pct=True) over U0",
        "inst_net_buy_pct_rank",
        spearman=1.0,
        is_cross_sectional_rank=True,
        populations_match=True,
    )

    # Then
    assert unproven.status == "MISMATCH"
    assert unproven.action == "REMOVE_OR_REPLACE"
    assert disjoint.status == "MISMATCH"
    assert aligned.status == "MATCH"


def test_classify_feature_contract_status_ladder() -> None:
    # Given / When
    from src.strategy.contract import classify_feature_contract

    exact = classify_feature_contract(
        "upper_shadow_ratio", "(high-max(open,close))/range", "upper_shadow_ratio", spearman=1.0
    )
    approx = classify_feature_contract(
        "intraday_range", "(high-low)/close", "intraday_range", spearman=0.9985
    )
    broken = classify_feature_contract(
        "body_ratio", "(close-open)/range (signed)", "body_ratio", spearman=0.9335
    )
    missing = classify_feature_contract("v_kosdaq_live", "v_kosdaq", None)

    # Then
    assert (exact.status, exact.action) == ("MATCH", "KEEP")
    assert (approx.status, approx.action) == ("APPROXIMATE", "JUSTIFY_OR_ALIGN")
    assert (broken.status, broken.action) == ("MISMATCH", "REMOVE_OR_REPLACE")
    assert (missing.status, missing.action) == ("UNAVAILABLE_LIVE", "REMOVE_OR_REPLACE")
    assert missing.spearman is None


def test_classify_feature_contract_requires_measurement() -> None:
    # Given / When / Then
    import pytest

    from src.strategy.contract import classify_feature_contract

    with pytest.raises(ValueError, match="spearman"):
        classify_feature_contract("log_tv", "log1p(tv_clean)", "log_trade_value_100m")

    with pytest.raises(ValueError, match="spearman"):
        classify_feature_contract(
            "chg_rank", "rank(pct)", "change_rate_pct_rank", is_cross_sectional_rank=True, populations_match=True
        )


def test_assert_production_feature_set_rejects_unjustified_rows() -> None:
    # Given
    import pytest

    from src.strategy.contract import assert_production_feature_set, classify_feature_contract

    ok = classify_feature_contract("upper_shadow_ratio", "h", "upper_shadow_ratio", spearman=1.0)
    approx = classify_feature_contract("intraday_range", "h", "intraday_range", spearman=0.9985)
    bad = classify_feature_contract("body_ratio", "h", "body_ratio", spearman=0.9335)

    # When / Then
    assert assert_production_feature_set([ok]) is None
    assert assert_production_feature_set([ok, approx], justified_approximates=frozenset({"intraday_range"})) is None

    with pytest.raises(ValueError, match="intraday_range"):
        assert_production_feature_set([ok, approx])

    with pytest.raises(ValueError, match="body_ratio"):
        assert_production_feature_set([ok, bad], justified_approximates=frozenset({"body_ratio"}))


def test_strategy_spec_fingerprint_detects_any_parameter_change() -> None:
    # Given
    import dataclasses

    from src.strategy.contract import KCA_TOP3_SHADOW_001, UniverseSpec

    base = KCA_TOP3_SHADOW_001.fingerprint()

    # When
    changed_k = dataclasses.replace(KCA_TOP3_SHADOW_001, top_k=5).fingerprint()
    changed_universe = dataclasses.replace(
        KCA_TOP3_SHADOW_001, universe=UniverseSpec(chg_max=0.15)
    ).fingerprint()

    # Then
    assert KCA_TOP3_SHADOW_001.strategy_id == "KCA-TOP3-SHADOW-001"
    assert KCA_TOP3_SHADOW_001.top_k == 3
    assert len(base) == 64
    assert base == KCA_TOP3_SHADOW_001.fingerprint()
    assert changed_k != base
    assert changed_universe != base

def test_derive_chg_ratio_is_exact_and_independent_of_vendor_column() -> None:
    # Given: 실제 오염 사례(2022-11-02 025530) — 벤더 컬럼은 1.054018 (percent)
    import numpy as np

    from src.strategy.contract import derive_chg_ratio

    close = np.array([3835.0, 1100.0, 900.0], dtype=np.float64)
    prev_close = np.array([3795.0, 1000.0, 1000.0], dtype=np.float64)

    # When
    ratio = derive_chg_ratio(close, prev_close)

    # Then
    np.testing.assert_allclose(ratio, [40.0 / 3795.0, 0.10, -0.10])
    assert abs(float(ratio[0]) - 0.01054018445) < 1e-9
    # percent 인코딩 벤더값(1.054018)과 100배 차이임을 명시적으로 고정
    assert abs(float(ratio[0]) * 100.0 - 1.054018445) < 1e-6


def test_derive_chg_ratio_nans_out_bad_prev_close_and_limit_violations() -> None:
    # Given
    import numpy as np

    from src.strategy.contract import KRX_DAILY_LIMIT_RATIO, derive_chg_ratio

    close = np.array([1000.0, 1000.0, 1000.0, 1400.0, 600.0, 1290.0], dtype=np.float64)
    prev_close = np.array([0.0, -50.0, np.nan, 1000.0, 1000.0, 1000.0], dtype=np.float64)

    # When
    ratio = derive_chg_ratio(close, prev_close)

    # Then
    assert KRX_DAILY_LIMIT_RATIO == 0.31
    assert np.isnan(ratio[:3]).all()
    assert not np.any(ratio[:3] == 0.0)
    assert np.isnan(ratio[3])  # +40% 는 제도 한계 초과
    assert np.isnan(ratio[4])  # -40% 도 초과
    np.testing.assert_allclose(ratio[5], 0.29)  # 상한가는 유효

    # limit 을 넓히면 통과한다
    loose = derive_chg_ratio(close, prev_close, limit=1.0)
    np.testing.assert_allclose(loose[3], 0.40)


def test_detect_mixed_unit_rows_flags_percent_encoded_only() -> None:
    # Given
    import numpy as np

    from src.strategy.contract import detect_mixed_unit_rows

    close = np.array([3835.0, 3835.0, 1100.0, 1000.0, 1000.0], dtype=np.float64)
    prev_close = np.array([3795.0, 3795.0, 1000.0, 0.0, 1000.0], dtype=np.float64)
    true_ratio = 40.0 / 3795.0
    vendor = np.array(
        [true_ratio * 100.0, true_ratio, 0.10, 5.0, np.nan],
        dtype=np.float64,
    )

    # When
    flagged = detect_mixed_unit_rows(close, prev_close, vendor)

    # Then
    assert flagged.dtype == bool
    assert flagged.tolist() == [True, False, False, False, False]
    # prev_close<=0 인 행은 판정 불가이므로 오염으로 단정하지 않는다
    assert bool(flagged[3]) is False


def test_select_universe_applies_tick_cost_cap_when_configured() -> None:
    # Given
    import numpy as np
    import pandas as pd

    from src.strategy.contract import COST_AWARE_UNIVERSE, DEFAULT_UNIVERSE, select_universe

    df = pd.DataFrame(
        {
            "chg_ratio": [0.05, 0.05, 0.05, 0.05],
            "tv_clean": [500.0, 500.0, 500.0, 500.0],
            "mc_clean": [800.0, 800.0, 800.0, 800.0],
            "close": [15000.0, 3000.0, 15000.0, 15000.0],
            "volume": [10, 10, 10, 10],
            "is_ceiling": [False, False, False, False],
            "tick_cost_bp": [6.67, 16.67, 7.5, np.nan],
        }
    )

    # When
    capped = select_universe(df, COST_AWARE_UNIVERSE)
    uncapped = select_universe(df, DEFAULT_UNIVERSE)

    # Then: 7.5bp 상한은 포함(<=), 16.67bp 초과 배제, NaN 배제
    assert COST_AWARE_UNIVERSE.max_tick_cost_bp == 7.5
    assert capped.tolist() == [True, False, True, False]
    # 기본 스펙은 비용축을 적용하지 않는다 (하위호환)
    assert DEFAULT_UNIVERSE.max_tick_cost_bp is None
    assert uncapped.tolist() == [True, True, True, True]


def test_select_universe_requires_tick_cost_column_only_when_capped() -> None:
    # Given
    import pandas as pd
    import pytest

    from src.strategy.contract import COST_AWARE_UNIVERSE, DEFAULT_UNIVERSE, select_universe

    df = pd.DataFrame(
        {
            "chg_ratio": [0.05],
            "tv_clean": [500.0],
            "mc_clean": [800.0],
            "close": [15000.0],
            "volume": [10],
            "is_ceiling": [False],
        }
    )

    # When / Then
    with pytest.raises(ValueError) as exc:  # noqa: PT011 - fail-closed column contract
        select_universe(df, COST_AWARE_UNIVERSE)
    assert "tick_cost_bp" in str(exc.value)

    # 기본 스펙은 동일 프레임으로 정상 동작 (기존 호출부 회귀 방지)
    mask = select_universe(df, DEFAULT_UNIVERSE)
    assert mask.tolist() == [True]


def test_cost_aware_strategy_spec_uses_topk_three() -> None:
    # Given / When
    from src.strategy.contract import (
        AA_COST,
        COST_AWARE_UNIVERSE,
        DEFAULT_UNIVERSE,
        KCA_TOP3_SHADOW_001,
        KCA_TOPK_COSTAWARE_001,
    )

    # Then: 신규 스펙 계약
    assert KCA_TOPK_COSTAWARE_001.strategy_id == "KCA-TOPK-COSTAWARE-001"
    assert KCA_TOPK_COSTAWARE_001.top_k == 3
    assert KCA_TOPK_COSTAWARE_001.top_k >= 2
    assert KCA_TOPK_COSTAWARE_001.universe is COST_AWARE_UNIVERSE
    assert KCA_TOPK_COSTAWARE_001.cost is AA_COST
    assert len(KCA_TOPK_COSTAWARE_001.fingerprint()) == 64
    assert KCA_TOPK_COSTAWARE_001.fingerprint() != KCA_TOP3_SHADOW_001.fingerprint()

    # 기존 기본값 불변 (하위호환 회귀 방지)
    assert DEFAULT_UNIVERSE.max_tick_cost_bp is None
    assert DEFAULT_UNIVERSE.min_market_cap_100m == 500.0
    assert DEFAULT_UNIVERSE.chg_min == 0.02
    assert DEFAULT_UNIVERSE.chg_max == 0.10
    assert KCA_TOP3_SHADOW_001.universe is DEFAULT_UNIVERSE

    # 신규 풀은 레거시 >=10% 가 아니라 2~10% 밴드를 유지하고 비용축만 추가한다
    assert COST_AWARE_UNIVERSE.chg_min == 0.02
    assert COST_AWARE_UNIVERSE.chg_max == 0.10
    assert COST_AWARE_UNIVERSE.min_trade_value_100m == 100.0
    assert COST_AWARE_UNIVERSE.exclude_ceiling is True
