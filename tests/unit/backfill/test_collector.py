from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from src.backfill.intraday.collector import collect_intraday_bars


def _kis_bar(hour: str, prpr: str, vol: str, cum: str) -> dict:
    return {
        "stck_cntg_hour": hour,
        "stck_oprc": prpr,
        "stck_hgpr": prpr,
        "stck_lwpr": prpr,
        "stck_prpr": prpr,
        "cntg_vol": vol,
        "acml_tr_pbmn": cum,
    }


def test_collect_intraday_bars_tags_rows_with_code_and_date() -> None:
    client = AsyncMock()
    client.get_intraday_minute_chart = AsyncMock(
        return_value={"rt_cd": "0", "output2": [_kis_bar("153000", "70000", "1000", "70000000")]}
    )

    result = asyncio.run(
        collect_intraday_bars(client, session=None, stock_codes=["005930", "000660"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["symbol"]) == {"005930", "000660"}
    assert (result["snapshot_date"] == "2026-09-03").all()
    assert (result["has_trade"]).all()
    assert (result["vendor"] == "kis").all()


def test_collect_intraday_bars_empty_universe_returns_empty_df() -> None:
    client = AsyncMock()
    result = asyncio.run(collect_intraday_bars(client, session=None, stock_codes=[], snapshot_date="2026-09-03"))
    assert result.empty


def test_collect_nxt_aftermarket_bars_is_continuous_series_and_skips_unlisted() -> None:
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    client = AsyncMock()

    async def _fake_chart(session, code, bar_interval_minutes=1, end_hour=None, floor_hour=None, market_div_code=None):
        assert market_div_code == "NX"
        if code == "005930":
            return {"rt_cd": "0", "output2": [
                _kis_bar("154000", "71000", "100", "7100000"),
                _kis_bar("154100", "71100", "200", "21320000"),
            ]}
        return {"rt_cd": "9", "msg1": "NXT 미상장"}

    client.get_intraday_minute_chart = _fake_chart

    result = asyncio.run(
        collect_nxt_aftermarket_bars(client, session=None, stock_codes=["005930", "000660"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["symbol"]) == {"005930"}


def test_collect_intraday_trade_ticks_tags_rows_with_code_and_date() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"stck_cntg_hour": "093000", "cnqn": "1000", "stck_prpr": "70000"}]}
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930", "000660"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["symbol"]) == {"005930", "000660"}
    assert (result["snapshot_date"] == "2026-09-03").all()


def test_collect_intraday_bars_normalize_failure_yields_empty_frame() -> None:
    """KIS 원천 행에 필수 컬럼이 없으면 정규화가 실패하고 빈 프레임으로 안전하게 스킵한다."""
    client = AsyncMock()
    client.get_intraday_minute_chart = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"stck_cntg_hour": "153000"}]}
    )

    result = asyncio.run(
        collect_intraday_bars(client, session=None, stock_codes=["005930"], snapshot_date="2026-09-03")
    )

    assert result.empty


def test_collect_intraday_trade_ticks_synthesizes_cnqn_from_cumulative_acml_vol() -> None:
    """cnqn/cntg_vol 없이 acml_vol만 있는 KIS 틱 응답은 누적값 차분으로 봉당 체결량을 합성한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(
        return_value={
            "rt_cd": "0",
            "output2": [
                {"stck_cntg_hour": "090001", "acml_vol": "150", "stck_prpr": "70000"},
                {"stck_cntg_hour": "090000", "acml_vol": "100", "stck_prpr": "69900"},
            ],
        }
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["volume"]) == {100, 50}


def test_collect_intraday_trade_ticks_clamps_unparseable_acml_vol_to_zero() -> None:
    """acml_vol이 숫자로 파싱되지 않는 행은 0으로 클램프하고 나머지는 정상 합성한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(
        return_value={
            "rt_cd": "0",
            "output2": [
                {"stck_cntg_hour": "090000", "acml_vol": "not-a-number", "stck_prpr": "70000"},
                {"stck_cntg_hour": "090001", "acml_vol": "150", "stck_prpr": "70100"},
            ],
        }
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["volume"]) == {0, 150}


def test_collect_intraday_trade_ticks_normalize_failure_yields_empty_frame() -> None:
    """필수 컬럼이 없는 KIS 틱 응답은 정규화가 실패하고 빈 프레임으로 안전하게 스킵한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"cnqn": "100"}]}
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930"], snapshot_date="2026-09-03")
    )

    assert result.empty


def test_collect_intraday_bars_routes_to_ls_with_fallback() -> None:
    """collect_intraday_bars에 ls_client가 주어지면 LS 우선 라우팅 경로를 탄다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_bars

    mock_ls = AsyncMock()
    mock_ls.get_minute_chart = AsyncMock(
        return_value={
            "rt_cd": "0",
            "vendor": "ls",
            "output2": [{"time": "090300", "open": 9100, "high": 9100, "low": 9100, "close": 9100, "jdiff_vol": 100, "value": 1}],
        }
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "005930"
    assert result.iloc[0]["vendor"] == "ls"
    mock_ls.get_minute_chart.assert_awaited_once()
    mock_kis.get_intraday_minute_chart.assert_not_awaited()


def test_collect_intraday_bars_ls_client_empty_universe_returns_empty_df() -> None:
    """ls_client 경로에서도 빈 유니버스는 빈 프레임을 반환한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_bars

    result = asyncio.run(
        collect_intraday_bars(AsyncMock(), session=None, stock_codes=[], snapshot_date="2026-09-04", ls_client=AsyncMock())
    )

    assert result.empty


def test_collect_intraday_bars_ls_exception_falls_back_to_kis() -> None:
    """LS 클라이언트 호출 자체가 예외를 던져도 KIS fallback으로 계속 수집한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_bars

    mock_ls = AsyncMock()
    mock_ls.get_minute_chart = AsyncMock(side_effect=RuntimeError("LS unreachable"))
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(
        return_value={"rt_cd": "0", "output2": [_kis_bar("153000", "70000", "1000", "70000000")]}
    )

    result = asyncio.run(
        collect_intraday_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "005930"
    assert result.iloc[0]["vendor"] == "kis"


def test_collect_intraday_bars_ls_normalize_failure_yields_empty_frame() -> None:
    """LS 응답이 필수 컬럼을 결여하면 정규화 실패로 빈 프레임을 반환한다 (KIS로 재시도하지 않음)."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_bars

    mock_ls = AsyncMock()
    mock_ls.get_minute_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "ls", "output2": [{"time": "090300"}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert result.empty
    mock_kis.get_intraday_minute_chart.assert_not_awaited()


def test_collect_intraday_bars_kis_fallback_non_zero_rt_cd_yields_empty_frame() -> None:
    """LS가 빈 응답을 주고 KIS fallback이 rt_cd != '0'을 반환하면 빈 프레임을 반환한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_bars

    mock_ls = AsyncMock()
    mock_ls.get_minute_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(return_value={"rt_cd": "9", "msg1": "error"})

    result = asyncio.run(
        collect_intraday_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert result.empty


def test_collect_intraday_bars_kis_fallback_empty_rows_yields_empty_frame() -> None:
    """LS가 빈 응답을 주고 KIS fallback이 성공하지만 행이 없으면 빈 프레임을 반환한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_bars

    mock_ls = AsyncMock()
    mock_ls.get_minute_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})

    result = asyncio.run(
        collect_intraday_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert result.empty


def test_collect_intraday_bars_kis_fallback_exception_yields_empty_frame() -> None:
    """LS가 빈 응답을 주고 KIS fallback 호출마저 예외를 던지면 빈 프레임으로 안전하게 스킵한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_bars

    mock_ls = AsyncMock()
    mock_ls.get_minute_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(side_effect=RuntimeError("KIS unreachable"))

    result = asyncio.run(
        collect_intraday_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert result.empty


def test_collect_intraday_trade_ticks_ls_exception_falls_back_to_kis() -> None:
    """LS 틱 조회 자체가 예외를 던져도 KIS fallback으로 계속 수집한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(side_effect=RuntimeError("LS unreachable"))
    mock_kis = AsyncMock()
    mock_kis.get_intraday_trade_ticks = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"stck_cntg_hour": "093000", "cnqn": "1000", "stck_prpr": "70000"}]}
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "005930"
    assert result.iloc[0]["vendor"] == "kis"


def test_collect_intraday_trade_ticks_ls_normalize_failure_yields_empty() -> None:
    """LS 틱 응답이 필수 컬럼을 결여하면 정규화 실패로 빈 결과를 반환한다 (KIS 재시도 없음)."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(return_value={"rt_cd": "0", "vendor": "ls", "output2": [{"time": "090300"}]})
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert result.empty
    mock_kis.get_intraday_trade_ticks.assert_not_awaited()


def test_collect_intraday_trade_ticks_ls_empty_output_falls_back_to_kis() -> None:
    """R2 fail-closed: LS rt_cd=0 with empty output2 falls back to KIS (supersedes skip behavior)."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_trade_ticks = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"stck_cntg_hour": "093000", "cnqn": "1000", "stck_prpr": "70000"}]}
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls)
    )

    assert len(result) == 1
    assert result.iloc[0]["vendor"] == "kis"
    mock_kis.get_intraday_trade_ticks.assert_awaited_once()


def test_collect_intraday_trade_ticks_no_ls_client_kis_exception_yields_empty() -> None:
    """ls_client가 없을 때 KIS 호출 자체가 예외를 던지면 빈 결과를 반환한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(side_effect=RuntimeError("KIS unreachable"))

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930"], snapshot_date="2026-09-03")
    )

    assert result.empty


def test_collect_intraday_trade_ticks_no_ls_client_non_zero_rt_cd_yields_empty() -> None:
    """ls_client가 없을 때 KIS가 rt_cd != '0'을 반환하면 빈 결과를 반환한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(return_value={"rt_cd": "9", "msg1": "error"})

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930"], snapshot_date="2026-09-03")
    )

    assert result.empty


def test_collect_intraday_trade_ticks_no_ls_client_empty_rows_yields_empty() -> None:
    """ls_client가 없을 때 KIS가 성공하지만 빈 행이면 빈 결과를 반환한다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(return_value={"rt_cd": "0", "output2": []})

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930"], snapshot_date="2026-09-03")
    )

    assert result.empty


def test_collector_routes_to_ls_with_fallback() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_intraday_trade_ticks
    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(return_value={"rt_cd": "0", "vendor": "ls", "truncated": False, "output2": [{"time": "153000", "close": 1000, "jdiff_vol": 10}]})
    mock_kis = AsyncMock()
    res = asyncio.run(collect_intraday_trade_ticks(mock_kis, None, ["005930"], "2026-09-04", ls_client=mock_ls))
    assert len(res) == 1
    assert res.iloc[0]["symbol"] == "005930"
    mock_ls.get_tick_chart.assert_awaited_once()
    mock_kis.get_intraday_trade_ticks.assert_not_awaited()

def test_collect_intraday_trade_ticks_kiwoom_success_skips_ls_and_kis() -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    today_str = datetime.now().strftime("%Y-%m-%d")
    mock_kiwoom = AsyncMock()
    mock_kiwoom.get_tick_chart = AsyncMock(
        return_value={
            "rt_cd": "0",
            "vendor": "kiwoom",
            "truncated": False,
            "output2": [{"cur_prc": "270000", "trde_qty": "150", "cntr_tm": f"{today_str.replace('-', '')}153000"}],
        }
    )
    mock_ls = AsyncMock()
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date=today_str,
            ls_client=mock_ls, kiwoom_client=mock_kiwoom,
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "005930"
    assert result.iloc[0]["vendor"] == "kiwoom"
    mock_ls.get_tick_chart.assert_not_awaited()
    mock_kis.get_intraday_trade_ticks.assert_not_awaited()


def test_collect_intraday_trade_ticks_kiwoom_exception_falls_back_to_ls() -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    today_str = datetime.now().strftime("%Y-%m-%d")
    mock_kiwoom = AsyncMock()
    mock_kiwoom.get_tick_chart = AsyncMock(side_effect=RuntimeError("kiwoom unreachable"))
    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "ls", "truncated": False, "output2": [{"time": "153000", "close": 1000, "jdiff_vol": 10}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date=today_str,
            ls_client=mock_ls, kiwoom_client=mock_kiwoom,
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["vendor"] == "ls"
    mock_kis.get_intraday_trade_ticks.assert_not_awaited()


def test_collect_intraday_trade_ticks_kiwoom_empty_success_skips_ls_and_kis() -> None:
    """Superseded by fail-closed R2: kiwoom empty now falls back (see ..._kiwoom_empty_falls_back_to_ls)."""
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    today_str = datetime.now().strftime("%Y-%m-%d")
    mock_kiwoom = AsyncMock()
    mock_kiwoom.get_tick_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "ls", "truncated": False, "output2": [{"time": "153000", "close": 1000, "jdiff_vol": 10}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date=today_str,
            ls_client=mock_ls, kiwoom_client=mock_kiwoom,
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["vendor"] == "ls"
    mock_ls.get_tick_chart.assert_awaited_once()


def test_collect_intraday_trade_ticks_kiwoom_normalize_failure_yields_empty_no_fallback() -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    today_str = datetime.now().strftime("%Y-%m-%d")
    mock_kiwoom = AsyncMock()
    mock_kiwoom.get_tick_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "kiwoom", "output2": [{"cur_prc": "270000"}]}
    )
    mock_ls = AsyncMock()
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date=today_str,
            ls_client=mock_ls, kiwoom_client=mock_kiwoom,
        )
    )

    assert result.empty
    mock_ls.get_tick_chart.assert_not_awaited()
    mock_kis.get_intraday_trade_ticks.assert_not_awaited()


def test_collect_intraday_trade_ticks_no_kiwoom_client_falls_back_to_ls_unchanged() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "ls", "truncated": False, "output2": [{"time": "153000", "close": 1000, "jdiff_vol": 10}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", ls_client=mock_ls,
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["vendor"] == "ls"


def test_collect_nxt_aftermarket_bars_prefers_kiwoom() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart.return_value = {
        "rt_cd": "0",
        "output2": [{"cntr_tm": "20260904195900", "cur_prc": "+257000", "open_pric": "+256500", "high_pric": "+257000", "low_pric": "+256500", "trde_qty": "100"}],
        "vendor": "kiwoom",
    }
    kis_client = AsyncMock()

    df = asyncio.run(collect_nxt_aftermarket_bars(kis_client, object(), ["005930"], "2026-09-04", kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["vendor"] == "kiwoom"
    assert kw_client.get_nxt_minute_chart.call_count == 1
    assert kis_client.get_intraday_minute_chart.call_count == 0


def test_collect_nxt_aftermarket_bars_falls_back_to_kis_when_kiwoom_fails() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart.return_value = {"rt_cd": "1", "output2": []}
    kis_client = AsyncMock()
    kis_client.get_intraday_minute_chart.return_value = {
        "rt_cd": "0",
        "output2": [{"stck_bsop_date": "20260904", "stck_cntg_hour": "195900", "stck_prpr": "257000", "stck_oprc": "256500", "stck_hgpr": "257000", "stck_lwpr": "256500", "cntg_vol": "100", "acml_tr_pbmn": "25700000"}],
    }

    df = asyncio.run(collect_nxt_aftermarket_bars(kis_client, object(), ["005930"], "2026-09-04", kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["vendor"] == "kis"
    assert kw_client.get_nxt_minute_chart.call_count == 1
    assert kis_client.get_intraday_minute_chart.call_count == 1


def test_collect_nxt_aftermarket_bars_zero_regression_when_kiwoom_none() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kis_client = AsyncMock()
    kis_client.get_intraday_minute_chart.return_value = {
        "rt_cd": "0",
        "output2": [{"stck_bsop_date": "20260904", "stck_cntg_hour": "195900", "stck_prpr": "257000", "stck_oprc": "256500", "stck_hgpr": "257000", "stck_lwpr": "256500", "cntg_vol": "100", "acml_tr_pbmn": "25700000"}],
    }

    df = asyncio.run(collect_nxt_aftermarket_bars(kis_client, object(), ["005930"], "2026-09-04", kiwoom_client=None))

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["vendor"] == "kis"
    assert kis_client.get_intraday_minute_chart.call_count == 1


def test_collect_nxt_aftermarket_bars_empty_universe_with_kiwoom_returns_empty() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    result = asyncio.run(
        collect_nxt_aftermarket_bars(AsyncMock(), session=None, stock_codes=[], snapshot_date="2026-09-04", kiwoom_client=AsyncMock())
    )

    assert result.empty


def test_collect_nxt_aftermarket_bars_kiwoom_exception_falls_back_to_kis() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart = AsyncMock(side_effect=RuntimeError("kiwoom unreachable"))
    kis_client = AsyncMock()
    kis_client.get_intraday_minute_chart = AsyncMock(
        return_value={"rt_cd": "0", "output2": [_kis_bar("154000", "71000", "100", "7100000")]}
    )

    df = asyncio.run(collect_nxt_aftermarket_bars(kis_client, object(), ["005930"], "2026-09-04", kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["vendor"] == "kis"


def test_collect_nxt_aftermarket_bars_kiwoom_normalize_failure_yields_empty_no_fallback() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "kiwoom", "output2": [{"cntr_tm": "20260904195900"}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_nxt_aftermarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty
    mock_kis.get_intraday_minute_chart.assert_not_awaited()


def test_collect_nxt_aftermarket_bars_kiwoom_empty_success_skips_kis() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_nxt_aftermarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty
    mock_kis.get_intraday_minute_chart.assert_not_awaited()


def test_collect_nxt_aftermarket_bars_kis_fallback_exception_yields_empty() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(side_effect=RuntimeError("KIS unreachable"))

    result = asyncio.run(
        collect_nxt_aftermarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty


def test_collect_nxt_aftermarket_bars_kis_fallback_non_zero_rt_cd_yields_empty() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(return_value={"rt_cd": "9", "msg1": "error"})

    result = asyncio.run(
        collect_nxt_aftermarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty


def test_collect_nxt_aftermarket_bars_kis_fallback_empty_rows_yields_empty() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_minute_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})

    result = asyncio.run(
        collect_nxt_aftermarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty



def test_collect_nxt_premarket_bars_prefers_kiwoom() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_premarket_chart.return_value = {
        "rt_cd": "0",
        "output2": [{"cntr_tm": "20260904080000", "cur_prc": "+251500", "open_pric": "+251000", "high_pric": "+253500", "low_pric": "+251000", "trde_qty": "100"}],
        "vendor": "kiwoom",
    }
    kis_client = AsyncMock()

    df = asyncio.run(collect_nxt_premarket_bars(kis_client, object(), ["005930"], "2026-09-04", kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["vendor"] == "kiwoom"
    assert kw_client.get_nxt_premarket_chart.call_count == 1
    assert kis_client.get_intraday_minute_chart.call_count == 0


def test_collect_nxt_premarket_bars_falls_back_to_kis_when_kiwoom_fails() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_premarket_chart.return_value = {"rt_cd": "1", "output2": []}
    kis_client = AsyncMock()
    kis_client.get_intraday_minute_chart.return_value = {
        "rt_cd": "0",
        "output2": [{"stck_bsop_date": "20260904", "stck_cntg_hour": "080000", "stck_prpr": "251500", "stck_oprc": "251000", "stck_hgpr": "253500", "stck_lwpr": "251000", "cntg_vol": "100", "acml_tr_pbmn": "25150000"}],
    }

    df = asyncio.run(collect_nxt_premarket_bars(kis_client, object(), ["005930"], "2026-09-04", kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["vendor"] == "kis"
    assert kw_client.get_nxt_premarket_chart.call_count == 1
    assert kis_client.get_intraday_minute_chart.call_count == 1


def test_collect_nxt_premarket_bars_zero_regression_when_kiwoom_none() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    kis_client = AsyncMock()
    kis_client.get_intraday_minute_chart.return_value = {
        "rt_cd": "0",
        "output2": [{"stck_bsop_date": "20260904", "stck_cntg_hour": "080000", "stck_prpr": "251500", "stck_oprc": "251000", "stck_hgpr": "253500", "stck_lwpr": "251000", "cntg_vol": "100", "acml_tr_pbmn": "25150000"}],
    }

    df = asyncio.run(collect_nxt_premarket_bars(kis_client, object(), ["005930"], "2026-09-04", kiwoom_client=None))

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "005930"
    assert df.iloc[0]["vendor"] == "kis"
    assert kis_client.get_intraday_minute_chart.call_count == 1


def test_collect_nxt_premarket_bars_empty_universe_with_kiwoom_returns_empty() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    result = asyncio.run(
        collect_nxt_premarket_bars(AsyncMock(), session=None, stock_codes=[], snapshot_date="2026-09-04", kiwoom_client=AsyncMock())
    )

    assert result.empty


def test_collect_nxt_premarket_bars_kiwoom_normalize_failure_yields_empty_no_fallback() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_premarket_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "kiwoom", "output2": [{"cntr_tm": "20260904080000"}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_nxt_premarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty
    mock_kis.get_intraday_minute_chart.assert_not_awaited()


def test_collect_nxt_premarket_bars_kiwoom_empty_success_skips_kis() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_premarket_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_nxt_premarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty
    mock_kis.get_intraday_minute_chart.assert_not_awaited()


def test_collect_nxt_premarket_bars_kis_fallback_non_zero_rt_cd_yields_empty() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_premarket_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(return_value={"rt_cd": "9", "msg1": "error"})

    result = asyncio.run(
        collect_nxt_premarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty


def test_collect_nxt_premarket_bars_kis_fallback_empty_rows_yields_empty() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.backfill.intraday.collector import collect_nxt_premarket_bars

    kw_client = AsyncMock()
    kw_client.get_nxt_premarket_chart = AsyncMock(return_value={"rt_cd": "1", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_minute_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})

    result = asyncio.run(
        collect_nxt_premarket_bars(mock_kis, session=None, stock_codes=["005930"], snapshot_date="2026-09-04", kiwoom_client=kw_client)
    )

    assert result.empty


def test_collect_intraday_trade_ticks_kiwoom_empty_falls_back_to_ls() -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    today_str = datetime.now().strftime("%Y-%m-%d")
    mock_kiwoom = AsyncMock()
    mock_kiwoom.get_tick_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "ls", "truncated": False, "output2": [{"time": "153000", "close": 1000, "jdiff_vol": 10}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date=today_str,
            ls_client=mock_ls, kiwoom_client=mock_kiwoom,
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["vendor"] == "ls"
    mock_kiwoom.get_tick_chart.assert_awaited_once()
    mock_ls.get_tick_chart.assert_awaited_once()
    mock_kis.get_intraday_trade_ticks.assert_not_awaited()


def test_collect_intraday_trade_ticks_past_date_routes_to_ls_first() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    past_date = "2020-01-02"
    mock_kiwoom = AsyncMock()
    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "ls", "truncated": False, "output2": [{"time": "153000", "close": 2000, "jdiff_vol": 20}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date=past_date,
            ls_client=mock_ls, kiwoom_client=mock_kiwoom,
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["vendor"] == "ls"
    mock_kiwoom.get_tick_chart.assert_not_awaited()
    mock_ls.get_tick_chart.assert_awaited_once()


def test_collect_intraday_trade_ticks_caps_ls_page_budget() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import DEFAULT_LS_TICK_MAX_PAGES, collect_intraday_trade_ticks

    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(
        return_value={"rt_cd": "0", "vendor": "ls", "truncated": False, "output2": [{"time": "153000", "close": 3000, "jdiff_vol": 30}]}
    )
    mock_kis = AsyncMock()

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date="2020-01-02",
            ls_client=mock_ls, kiwoom_client=None,
        )
    )

    assert len(result) == 1
    _, kwargs = mock_ls.get_tick_chart.call_args
    assert kwargs.get("max_pages") == DEFAULT_LS_TICK_MAX_PAGES


def test_collect_intraday_trade_ticks_ls_empty_falls_back_to_kis() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    mock_ls = AsyncMock()
    mock_ls.get_tick_chart = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    mock_kis = AsyncMock()
    mock_kis.get_intraday_trade_ticks = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"stck_cntg_hour": "153000", "cnqn": "100", "stck_prpr": "40000"}]}
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(
            mock_kis, session=None, stock_codes=["005930"], snapshot_date="2020-01-02",
            ls_client=mock_ls, kiwoom_client=None,
        )
    )

    assert len(result) == 1
    assert result.iloc[0]["vendor"] == "kis"
    mock_ls.get_tick_chart.assert_awaited_once()
    mock_kis.get_intraday_trade_ticks.assert_awaited_once()


def test_collect_krx_aftermarket_bars_skips_dates_before_launch() -> None:
    import asyncio

    import pandas as pd

    from src.backfill.intraday.collector import collect_krx_aftermarket_bars

    calls: list[dict] = []

    class _Client:
        async def get_intraday_minute_chart(self, session, code, **kwargs):
            calls.append({"code": code, **kwargs})
            return {
                "rt_cd": "0",
                "output2": [{
                    "stck_bsop_date": "20260914",
                    "stck_cntg_hour": "160100",
                    "stck_oprc": "1000", "stck_hgpr": "1010",
                    "stck_lwpr": "995", "stck_prpr": "1005",
                    "cntg_vol": "100", "acml_tr_pbmn": "100000",
                }],
            }

    client = _Client()

    # Given: 시행일 이전 -> API 호출 없음
    before = asyncio.run(collect_krx_aftermarket_bars(client, object(), ["005930"], "2026-09-11"))
    assert isinstance(before, pd.DataFrame) and before.empty
    assert calls == []

    # When: 시행일 당일 -> KRX 애프터 구간으로 조회
    after = asyncio.run(collect_krx_aftermarket_bars(client, object(), ["005930"], "2026-09-14"))
    assert len(calls) == 1
    assert calls[0]["floor_hour"] == "160000"
    assert calls[0]["end_hour"] == "200000"
    assert calls[0]["market_div_code"] == "J"
    assert after["ts_hms"].tolist() == [160100]


def test_backfill_krx_aftermarket_bars_uses_historical_tr_after_launch() -> None:
    import asyncio

    from src.backfill.intraday.collector import backfill_krx_aftermarket_bars

    calls: list[tuple] = []

    class _Client:
        async def get_historical_minute_chart(self, session, code, target_date, **kwargs):
            calls.append((code, target_date, kwargs.get("floor_hour"), kwargs.get("end_hour")))
            return {"rt_cd": "0", "output2": []}

    client = _Client()

    empty = asyncio.run(backfill_krx_aftermarket_bars(client, object(), ["005930"], "2026-09-01"))
    assert empty.empty
    assert calls == []

    asyncio.run(backfill_krx_aftermarket_bars(client, object(), ["005930"], "2026-09-15"))
    assert calls == [("005930", "20260915", "160000", "200000")]
