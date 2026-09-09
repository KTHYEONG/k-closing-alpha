"""test_observe_closing_auction_price_behavior.py 단위 테스트.

라이브 KIS API를 호출하는 main()/_poll_once()는 net I/O 이므로 범위 밖이며,
순수 로직인 _find_antc_fields/_hhmmss만 검증한다.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.tools.observe_closing_auction_price_behavior import (
    _find_antc_fields,
    _hhmmss,
    _now_kst,
)


def test_find_antc_fields_locates_keys_in_output2_not_output1() -> None:
    # Given: KIS's real FHKST01010200 shape -- antc_* fields live in output2, not output1
    res = {
        "rt_cd": "0",
        "output1": {"askp1": "70000", "bidp1": "69900"},
        "output2": {
            "antc_mkop_cls_code": "112",
            "antc_cnpr": "269500",
            "antc_cntg_vrss": "0",
            "antc_vol": "1258219",
            "stck_prpr": "267500",
        },
    }

    found = _find_antc_fields(res)

    assert found == {
        "output2.antc_mkop_cls_code": "112",
        "output2.antc_cnpr": "269500",
        "output2.antc_cntg_vrss": "0",
        "output2.antc_vol": "1258219",
    }


def test_find_antc_fields_returns_empty_when_no_antc_keys_anywhere() -> None:
    res = {"rt_cd": "0", "output1": {"askp1": "70000"}, "output2": {"stck_prpr": "70000"}}

    assert _find_antc_fields(res) == {}


def test_find_antc_fields_walks_nested_lists_and_dicts() -> None:
    res = {"output2": [{"antc_cnpr": "1"}, {"nested": {"antc_vol": "2"}}]}

    found = _find_antc_fields(res)

    assert found == {
        "output2[0].antc_cnpr": "1",
        "output2[1].nested.antc_vol": "2",
    }


def test_hhmmss_formats_kst_datetime() -> None:
    dt = datetime(2026, 9, 10, 15, 22, 5, tzinfo=ZoneInfo("Asia/Seoul"))

    assert _hhmmss(dt) == "152205"


def test_now_kst_returns_asia_seoul_tz_aware_datetime() -> None:
    now = _now_kst()

    assert now.tzinfo is not None
    assert now.utcoffset() is not None
    assert now.utcoffset().total_seconds() == 9 * 3600
