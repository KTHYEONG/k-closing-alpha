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
    })
    decision = pd.Timestamp("2026-09-14")

    # When
    mask = rank_pool_mask(df, decision_date=decision)
    admitted = flag_cost_aware_admission(df, decision_date=decision)["admitted"].to_numpy()

    # Then: 학습 풀은 틱캡 없이 A만 포함, admission(틱캡)은 A도 거부해 풀에 중첩된다
    assert mask.tolist() == [True, False, False, False]
    assert admitted.tolist() == [False, False, False, False]
    assert not (admitted & ~mask).any()

