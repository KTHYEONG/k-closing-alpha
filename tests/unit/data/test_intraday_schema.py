from __future__ import annotations



def test_normalize_bar_frame_ls_value_is_scaled_to_krw_per_bar() -> None:
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame

    # Given: 실측 LS t8412 응답 행 (2026-09-04, 009900)
    raw = pd.DataFrame(
        {
            "time": ["090300", "090400"],
            "open": [9100, 9210],
            "high": [9100, 9210],
            "low": [9100, 9210],
            "close": [9100, 9210],
            "jdiff_vol": [55226, 108719],
            "value": [498, 998],
        }
    )

    # When
    out = normalize_bar_frame(raw, "ls", "2026-09-04", "009900")

    # Then
    assert out["value_krw"].tolist() == [498_000_000, 998_000_000]
    notional = out["close"].iloc[0] * out["volume"].iloc[0]
    assert abs(out["value_krw"].iloc[0] - notional) / notional < 0.01
    assert out["symbol"].tolist() == ["009900", "009900"]
    assert out["snapshot_date"].tolist() == ["2026-09-04", "2026-09-04"]


def test_normalize_bar_frame_kis_cumulative_value_becomes_per_bar_diff() -> None:
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame

    # Given: 실측 KIS FHKST03010200 응답 행 (2026-09-03, 004710)
    raw = pd.DataFrame(
        {
            "stck_cntg_hour": ["090100", "090000", "090300"],
            "stck_oprc": ["8400", "8100", "8450"],
            "stck_hgpr": ["8420", "8130", "8500"],
            "stck_lwpr": ["8390", "8090", "8440"],
            "stck_prpr": ["8410", "8120", "8480"],
            "cntg_vol": ["89523", "90667", "135971"],
            "acml_tr_pbmn": ["1481173175", "738988315", "2629666630"],
        }
    )

    # When
    out = normalize_bar_frame(raw, "kis", "2026-09-03", "004710")

    # Then: 시각 오름차순 정렬 후 1차 차분, 첫 봉은 누적값 그대로
    assert out["ts_hms"].tolist() == [90000, 90100, 90300]
    assert out["value_krw"].tolist() == [
        738_988_315,
        1_481_173_175 - 738_988_315,
        2_629_666_630 - 1_481_173_175,
    ]
    assert out["vendor"].tolist() == ["kis", "kis", "kis"]


def test_normalize_bar_frame_flags_ls_synthetic_zero_volume_bar() -> None:
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame

    # Given: 실측 009900 09:01-09:03 -- 09:01/09:02는 무거래 합성봉(직전가 carry)
    raw = pd.DataFrame(
        {
            "time": ["090100", "090200", "090300"],
            "open": [7770, 7770, 9100],
            "high": [7770, 7770, 9100],
            "low": [7770, 7770, 9100],
            "close": [7770, 7770, 9100],
            "jdiff_vol": [0, 0, 55226],
            "value": [6, 0, 498],
        }
    )

    # When
    out = normalize_bar_frame(raw, "ls", "2026-09-04", "009900")

    # Then: 행은 보존하되 합성봉임을 플래그로 구분한다
    assert len(out) == 3
    assert out["has_trade"].tolist() == [False, False, True]
    assert out["has_trade"].dtype == bool


def test_normalize_bar_frame_rejects_unknown_vendor() -> None:
    import pandas as pd
    import pytest

    from src.data.intraday_schema import normalize_bar_frame

    raw = pd.DataFrame({"time": ["090100"], "open": [1], "high": [1], "low": [1], "close": [1], "jdiff_vol": [1], "value": [1]})

    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts fail-closed vendor
        normalize_bar_frame(raw, "kiwoom", "2026-09-04", "005930")

    incomplete = pd.DataFrame({"time": ["090100"], "open": [1]})
    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts fail-closed columns
        normalize_bar_frame(incomplete, "ls", "2026-09-04", "005930")


def test_normalize_tick_frame_carries_truncated_flag_and_optional_kis_fields() -> None:
    import pandas as pd

    from src.data.intraday_schema import CANONICAL_TICK_COLUMNS, normalize_tick_frame

    ls_raw = pd.DataFrame({"time": ["090247", "090248"], "close": [8990, 9000], "jdiff_vol": [30301, 45]})
    ls_out = normalize_tick_frame(ls_raw, "ls", "2026-09-04", "009900", truncated=True)

    assert list(ls_out.columns) == list(CANONICAL_TICK_COLUMNS)
    assert ls_out["price"].tolist() == [8990, 9000]
    assert ls_out["truncated"].all()
    assert ls_out["trade_strength"].isna().all()

    kis_raw = pd.DataFrame(
        {"stck_cntg_hour": ["090247"], "stck_prpr": ["8990"], "cnqn": ["30301"], "tday_rltv": ["120.5"], "askp": ["9000"], "bidp": ["8990"]}
    )
    kis_out = normalize_tick_frame(kis_raw, "kis", "2026-09-04", "009900")

    assert kis_out["truncated"].tolist() == [False]
    assert float(kis_out["trade_strength"].iloc[0]) == 120.5
    assert int(kis_out["ask1"].iloc[0]) == 9000
    assert int(kis_out["bid1"].iloc[0]) == 8990


def test_normalize_bar_frame_empty_input_returns_canonical_empty_frame() -> None:
    import pandas as pd

    from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, normalize_bar_frame

    out = normalize_bar_frame(pd.DataFrame(), "ls", "2026-09-04", "005930")

    assert list(out.columns) == list(CANONICAL_BAR_COLUMNS)
    assert len(out) == 0


def test_normalize_tick_frame_empty_input_returns_canonical_empty_frame() -> None:
    import pandas as pd

    from src.data.intraday_schema import CANONICAL_TICK_COLUMNS, normalize_tick_frame

    out = normalize_tick_frame(pd.DataFrame(), "kis", "2026-09-04", "005930")

    assert list(out.columns) == list(CANONICAL_TICK_COLUMNS)
    assert len(out) == 0


def test_normalize_bar_frame_clamps_negative_kis_cumulative_diff_to_zero() -> None:
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame

    # Given: 두 번째 봉의 누적 거래대금이 첫 번째보다 역행(데이터 이상)
    raw = pd.DataFrame(
        {
            "stck_cntg_hour": ["090000", "090100"],
            "stck_oprc": ["8000", "8000"],
            "stck_hgpr": ["8000", "8000"],
            "stck_lwpr": ["8000", "8000"],
            "stck_prpr": ["8000", "8000"],
            "cntg_vol": ["100", "100"],
            "acml_tr_pbmn": ["1000000", "500000"],
        }
    )

    out = normalize_bar_frame(raw, "kis", "2026-09-03", "005930")

    assert out["value_krw"].tolist() == [1_000_000, 0]


def test_normalize_tick_frame_kis_falls_back_to_cntg_vol_and_missing_optional_fields() -> None:
    import pandas as pd

    from src.data.intraday_schema import normalize_tick_frame

    raw = pd.DataFrame({"stck_cntg_hour": ["090247"], "stck_prpr": ["8990"], "cntg_vol": ["100"]})

    out = normalize_tick_frame(raw, "kis", "2026-09-04", "009900")

    assert out["volume"].iloc[0] == 100
    assert out["trade_strength"].isna().all()
    assert out["ask1"].isna().all()
    assert out["bid1"].isna().all()


def test_normalize_tick_frame_kis_missing_volume_column_raises() -> None:
    import pandas as pd
    import pytest

    from src.data.intraday_schema import normalize_tick_frame

    raw = pd.DataFrame({"stck_cntg_hour": ["090247"], "stck_prpr": ["8990"]})

    with pytest.raises(ValueError, match="volume"):
        normalize_tick_frame(raw, "kis", "2026-09-04", "009900")


def test_assert_canonical_bars_and_ticks_reject_mismatched_columns() -> None:
    import pandas as pd
    import pytest

    from src.data.intraday_schema import assert_canonical_bars, assert_canonical_ticks

    with pytest.raises(ValueError, match="Non-canonical bar frame"):
        assert_canonical_bars(pd.DataFrame({"symbol": ["005930"]}))

    with pytest.raises(ValueError, match="Non-canonical tick frame"):
        assert_canonical_ticks(pd.DataFrame({"symbol": ["005930"]}))


def test_assert_canonical_bars_and_ticks_accept_matching_columns() -> None:
    import pandas as pd

    from src.data.intraday_schema import (
        CANONICAL_BAR_COLUMNS,
        CANONICAL_TICK_COLUMNS,
        assert_canonical_bars,
        assert_canonical_ticks,
    )

    assert_canonical_bars(pd.DataFrame({c: [] for c in CANONICAL_BAR_COLUMNS}))
    assert_canonical_ticks(pd.DataFrame({c: [] for c in CANONICAL_TICK_COLUMNS}))

