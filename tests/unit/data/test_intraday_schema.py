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
        normalize_bar_frame(raw, "unknown_vendor", "2026-09-04", "005930")

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

def test_normalize_tick_frame_kiwoom_vendor_maps_fields() -> None:
    import pandas as pd

    from src.data.intraday_schema import CANONICAL_TICK_COLUMNS, normalize_tick_frame

    df = pd.DataFrame([{"cur_prc": "270000", "trde_qty": "150", "cntr_tm": "20260904153000"}])

    out = normalize_tick_frame(df, "kiwoom", "2026-09-04", "005930")

    assert list(out.columns) == list(CANONICAL_TICK_COLUMNS)
    assert int(out.iloc[0]["ts_hms"]) == 153000
    assert int(out.iloc[0]["price"]) == 270000
    assert int(out.iloc[0]["volume"]) == 150
    assert out.iloc[0]["vendor"] == "kiwoom"
    assert pd.isna(out.iloc[0]["trade_strength"])
    assert pd.isna(out.iloc[0]["ask1"])
    assert pd.isna(out.iloc[0]["bid1"])


def test_normalize_tick_frame_kiwoom_missing_required_column_raises() -> None:
    import pandas as pd
    import pytest

    from src.data.intraday_schema import normalize_tick_frame

    df = pd.DataFrame([{"cur_prc": "270000", "cntr_tm": "20260904153000"}])

    with pytest.raises(ValueError, match="Missing required kiwoom source columns"):
        normalize_tick_frame(df, "kiwoom", "2026-09-04", "005930")


def test_check_vendor_accepts_kiwoom() -> None:
    from src.data.intraday_schema import _check_vendor

    assert _check_vendor("kiwoom") == "kiwoom"


def test_check_vendor_rejects_unknown_vendor() -> None:
    import pytest

    from src.data.intraday_schema import _check_vendor

    with pytest.raises(ValueError, match="Unknown intraday vendor"):
        _check_vendor("unknown_vendor")


def test_normalize_tick_frame_rejects_unknown_vendor() -> None:
    import pandas as pd
    import pytest

    from src.data.intraday_schema import normalize_tick_frame

    raw = pd.DataFrame({"time": ["090100"], "close": [1000], "jdiff_vol": [10]})

    with pytest.raises(ValueError, match="Unknown intraday vendor"):
        normalize_tick_frame(raw, "unknown_vendor", "2026-09-04", "005930")


def test_normalize_bar_frame_kiwoom_vendor() -> None:
    import pandas as pd
    from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, normalize_bar_frame

    raw = pd.DataFrame([
        {
            "cntr_tm": "20260904195900",
            "cur_prc": "+257000",
            "open_pric": "+256500",
            "high_pric": "+257000",
            "low_pric": "+256500",
            "trde_qty": "100",
        }
    ])
    df = normalize_bar_frame(raw, "kiwoom", "2026-09-04", "005930")
    assert list(df.columns) == list(CANONICAL_BAR_COLUMNS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "005930"
    assert row["ts_hms"] == 195900
    assert row["close"] == 257000
    assert row["open"] == 256500
    assert row["high"] == 257000
    assert row["low"] == 256500
    assert row["volume"] == 100
    assert row["value_krw"] == 257000 * 100
    assert row["vendor"] == "kiwoom"
    assert row["has_trade"] is True


def test_normalize_bar_and_tick_kiwoom_strips_negative_signs() -> None:
    import pandas as pd
    from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame

    bar_raw = pd.DataFrame([
        {
            "cntr_tm": "20260904084500",
            "cur_prc": "-257000",
            "open_pric": "-256500",
            "high_pric": "-257000",
            "low_pric": "-256500",
            "trde_qty": "100",
        }
    ])
    df_bar = normalize_bar_frame(bar_raw, "kiwoom", "2026-09-04", "005930")
    assert df_bar["close"].iloc[0] == 257000
    assert df_bar["open"].iloc[0] == 256500
    assert df_bar["value_krw"].iloc[0] == 257000 * 100

    tick_raw = pd.DataFrame([
        {"cur_prc": "-257000", "trde_qty": "50", "cntr_tm": "20260904153000"}
    ])
    df_tick = normalize_tick_frame(tick_raw, "kiwoom", "2026-09-04", "005930")
    assert df_tick["price"].iloc[0] == 257000
    assert df_tick["volume"].iloc[0] == 50



def test_extract_vendor_business_dates_per_vendor_and_missing_field() -> None:
    import pandas as pd

    from src.data.intraday_schema import extract_vendor_business_dates

    kis = pd.DataFrame({"stck_bsop_date": ["20260501", "20260430"], "stck_prpr": ["1", "2"]})
    assert extract_vendor_business_dates(kis, "kis").tolist() == ["20260501", "20260430"]

    ls = pd.DataFrame({"date": ["20260501"], "close": [1]})
    assert extract_vendor_business_dates(ls, "ls").tolist() == ["20260501"]

    kiwoom = pd.DataFrame({"cntr_tm": ["20260501160100"], "cur_prc": ["1"]})
    assert extract_vendor_business_dates(kiwoom, "kiwoom").tolist() == ["20260501"]

    # 필드 부재 -> 검증 불가 신호로 None
    assert extract_vendor_business_dates(pd.DataFrame({"stck_prpr": ["1"]}), "kis") is None


def test_extract_vendor_business_dates_rejects_unknown_vendor() -> None:
    import pandas as pd
    import pytest

    from src.data.intraday_schema import extract_vendor_business_dates

    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts fail-closed vendor
        extract_vendor_business_dates(pd.DataFrame({"a": [1]}), "bloomberg")


def test_filter_to_business_date_drops_stale_rows_and_warns_when_unverifiable(caplog) -> None:
    import logging

    import pandas as pd

    from src.data.intraday_schema import filter_to_business_date

    # Given: 요청일(20260501) 2행 + 타 영업일(20250829) 1행
    df = pd.DataFrame({
        "stck_bsop_date": ["20260501", "20250829", "20260501"],
        "stck_cntg_hour": ["090100", "090100", "090200"],
    })
    out = filter_to_business_date(df, "kis", "2026-05-01", "005930")
    assert len(out) == 2
    assert set(out["stck_bsop_date"]) == {"20260501"}

    # When: 영업일 필드가 없는 응답 -> 통과시키되 경고
    with caplog.at_level(logging.WARNING):
        nofield = pd.DataFrame({"stck_cntg_hour": ["090100"]})
        passed = filter_to_business_date(nofield, "kis", "2026-05-01", "005930")
    assert len(passed) == 1
    assert any("date_verified=false" in r.getMessage() for r in caplog.records)


def test_normalize_bar_frame_gates_stale_business_date_before_cumulative_diff() -> None:
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame

    # Given: 요청일 2봉 사이에 타 영업일 1봉(누적 거래대금 계열이 다름)이 섞여 있다
    df = pd.DataFrame({
        "stck_bsop_date": ["20260501", "20250829", "20260501"],
        "stck_cntg_hour": ["090100", "090150", "090200"],
        "stck_oprc": ["1000", "9999", "1010"],
        "stck_hgpr": ["1010", "9999", "1020"],
        "stck_lwpr": ["995", "9999", "1005"],
        "stck_prpr": ["1005", "9999", "1015"],
        "cntg_vol": ["100", "777", "200"],
        "acml_tr_pbmn": ["100000", "50000000", "300000"],
    })

    out = normalize_bar_frame(df, "kis", "2026-05-01", "005930")

    assert len(out) == 2
    assert out["ts_hms"].tolist() == [90100, 90200]
    # 차분이 요청일 행만으로 계산됨: 100000, 300000-100000
    assert out["value_krw"].tolist() == [100000, 200000]
    assert out["close"].tolist() == [1005, 1015]


def test_normalize_frames_return_empty_when_all_rows_are_stale() -> None:
    import pandas as pd

    from src.data.intraday_schema import (
        CANONICAL_BAR_COLUMNS,
        CANONICAL_TICK_COLUMNS,
        normalize_bar_frame,
        normalize_tick_frame,
    )

    bars = pd.DataFrame({
        "stck_bsop_date": ["20250829"],
        "stck_cntg_hour": ["090100"],
        "stck_oprc": ["1000"], "stck_hgpr": ["1010"], "stck_lwpr": ["995"], "stck_prpr": ["1005"],
        "cntg_vol": ["100"], "acml_tr_pbmn": ["100000"],
    })
    out_bars = normalize_bar_frame(bars, "kis", "2026-05-01", "005930")
    assert len(out_bars) == 0
    assert list(out_bars.columns) == list(CANONICAL_BAR_COLUMNS)

    ticks = pd.DataFrame({
        "stck_bsop_date": ["20250829"],
        "stck_cntg_hour": ["090100"],
        "stck_prpr": ["1005"],
        "cnqn": ["10"],
    })
    out_ticks = normalize_tick_frame(ticks, "kis", "2026-05-01", "005930")
    assert len(out_ticks) == 0
    assert list(out_ticks.columns) == list(CANONICAL_TICK_COLUMNS)
