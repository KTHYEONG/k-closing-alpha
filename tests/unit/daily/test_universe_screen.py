from __future__ import annotations


def test_build_screen_frame_derives_screen_inputs_from_snapshot_columns() -> None:
    import numpy as np
    import pandas as pd

    from src.daily.universe_screen import build_screen_frame

    # Given: 억 단위 스냅샷 2행 (정상 +5%, 상한가 +29.9% 고가마감)
    df = pd.DataFrame({
        "종가": [10500.0, 12990.0],
        "전일종가": [10000.0, 10000.0],
        "고가": [10600.0, 12990.0],
        "거래량": [100000.0, 50000.0],
        "거래대금": [150.0, 80.0],
        "시가총액": [900.0, 700.0],
        "시장구분": ["KOSPI", "KOSDAQ"],
    })

    # When
    out = build_screen_frame(df, decision_date=pd.Timestamp("2026-09-14"))

    # Then
    assert list(out.columns) == ["chg_ratio", "is_ceiling", "tick_cost_bp", "tv_clean", "mc_clean", "close", "volume"]
    assert np.isclose(out["chg_ratio"].iloc[0], 0.05)
    assert out["is_ceiling"].tolist() == [False, True]
    assert np.isclose(out["tick_cost_bp"].iloc[0], 10.0 / 10500.0 * 1e4)
    assert out["tv_clean"].tolist() == [150.0, 80.0]
    assert out["mc_clean"].tolist() == [900.0, 700.0]



def test_rank_pool_mask_uses_training_screen_without_tick_cap() -> None:
    import pandas as pd

    from src.daily.collect import flag_cost_aware_admission
    from src.daily.universe_screen import rank_pool_mask

    # Given: A=밴드내 유동성 충분하나 틱비용(5원/2010원=24.9bp)>12bp, B=음수 등락(Toss union형),
    #        C=거래대금 50억(<100억), D=상한가
    df = pd.DataFrame({
        "종가": [2010.0, 9500.0, 10500.0, 12990.0],
        "전일종가": [1900.0, 10000.0, 10000.0, 10000.0],
        "고가": [2020.0, 9800.0, 10600.0, 12990.0],
        "거래량": [5_000_000.0, 100000.0, 100000.0, 100000.0],
        "거래대금": [1000.0, 9000.0, 50.0, 900.0],
        "시가총액": [5000.0, 90000.0, 900.0, 900.0],
        "시장구분": ["KOSDAQ", "KOSPI", "KOSPI", "KOSDAQ"],
        "is_screenable": [True, True, True, True],
    })
    decision = pd.Timestamp("2026-09-14")

    # When
    mask = rank_pool_mask(df, decision_date=decision)
    admitted = flag_cost_aware_admission(df, decision_date=decision)["admitted"].to_numpy()

    # Then: 학습 풀은 틱캡 없이 A만 포함, admission(틱캡)은 A도 거부해 풀에 중첩된다
    assert mask.tolist() == [True, False, False, False]
    assert admitted.tolist() == [False, False, False, False]
    assert not (admitted & ~mask).any()


def _screen_snapshot():
    import pandas as pd

    return pd.DataFrame({
        "종가": [10500.0, 12990.0],
        "전일종가": [10000.0, 10000.0],
        "고가": [10600.0, 12990.0],
        "거래량": [100000.0, 50000.0],
        "거래대금": [150.0, 80.0],
        "시가총액": [900.0, 700.0],
        "시장구분": ["KOSPI", "KOSDAQ"],
    })


def test_build_screen_frame_passes_class_verdict_through() -> None:
    import pandas as pd

    from src.daily.universe_screen import build_screen_frame

    df = _screen_snapshot()
    df["is_screenable"] = [True, False]
    out = build_screen_frame(df, decision_date=pd.Timestamp("2026-09-14"))
    assert list(out.columns) == ["chg_ratio", "is_ceiling", "tick_cost_bp", "tv_clean", "mc_clean", "close", "volume", "is_screenable"]
    assert out["is_screenable"].tolist() == [True, False]


def test_build_screen_frame_without_verdict_is_unchanged() -> None:
    import pandas as pd

    from src.daily.universe_screen import build_screen_frame

    out = build_screen_frame(_screen_snapshot(), decision_date=pd.Timestamp("2026-09-14"))
    assert list(out.columns) == ["chg_ratio", "is_ceiling", "tick_cost_bp", "tv_clean", "mc_clean", "close", "volume"]


def test_build_screen_frame_rejects_null_verdict() -> None:
    import pandas as pd
    import pytest

    from src.daily.universe_screen import build_screen_frame

    df = _screen_snapshot()
    df["is_screenable"] = pd.Series([True, None], dtype=object)
    with pytest.raises(ValueError, match="is_screenable"):
        build_screen_frame(df, decision_date=pd.Timestamp("2026-09-14"))


def test_class_filtered_screen_fails_closed_without_verdict() -> None:
    import pandas as pd
    import pytest

    from src.daily.universe_screen import rank_pool_mask
    from src.strategy.contract import UniverseSpec

    spec = UniverseSpec(exclude_non_screenable_class=True)
    with pytest.raises(ValueError, match="is_screenable"):
        rank_pool_mask(_screen_snapshot(), decision_date=pd.Timestamp("2026-09-14"), screen=spec)


def test_rank_pool_default_requires_the_verdict() -> None:
    import pandas as pd
    import pytest

    from src.daily.universe_screen import rank_pool_mask

    df = pd.DataFrame({
        "종가": [2010.0, 9500.0, 10500.0, 12990.0],
        "전일종가": [1900.0, 10000.0, 10000.0, 10000.0],
        "고가": [2020.0, 9800.0, 10600.0, 12990.0],
        "거래량": [5_000_000.0, 100000.0, 100000.0, 100000.0],
        "거래대금": [1000.0, 9000.0, 50.0, 900.0],
        "시가총액": [5000.0, 90000.0, 900.0, 900.0],
        "시장구분": ["KOSDAQ", "KOSPI", "KOSPI", "KOSDAQ"],
    })
    with pytest.raises(ValueError, match="is_screenable"):
        rank_pool_mask(df, decision_date=pd.Timestamp("2026-09-14"))


def test_rank_pool_drops_a_non_screenable_row() -> None:
    import pandas as pd

    from src.daily.universe_screen import rank_pool_mask

    df = pd.DataFrame({
        "종가": [2010.0, 9500.0, 10500.0, 12990.0],
        "전일종가": [1900.0, 10000.0, 10000.0, 10000.0],
        "고가": [2020.0, 9800.0, 10600.0, 12990.0],
        "거래량": [5_000_000.0, 100000.0, 100000.0, 100000.0],
        "거래대금": [1000.0, 9000.0, 50.0, 900.0],
        "시가총액": [5000.0, 90000.0, 900.0, 900.0],
        "시장구분": ["KOSDAQ", "KOSPI", "KOSPI", "KOSDAQ"],
        "is_screenable": [False, True, True, True],
    })
    assert rank_pool_mask(df, decision_date=pd.Timestamp("2026-09-14")).tolist() == [False, False, False, False]

