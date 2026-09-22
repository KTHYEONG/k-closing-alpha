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


def _ls_bar_row(t, close=70000, vol=100, day="20260904"):
    return {"date": day, "time": t, "open": close, "high": close, "low": close, "close": close, "jdiff_vol": vol, "value": 10}


def _kis_bar_row(h, close="70000", vol="100", cum="70000000"):
    return {
        "stck_cntg_hour": h, "stck_oprc": close, "stck_hgpr": close, "stck_lwpr": close,
        "stck_prpr": close, "cntg_vol": vol, "acml_tr_pbmn": cum,
    }


def _ls_tick_row(t, close=70000, vol=10, day="20260904"):
    return {"date": day, "time": t, "close": close, "jdiff_vol": vol}


def _kw_tick_row(dt="20260904153000", prc="+70000", qty="10"):
    return {"cntr_tm": dt, "cur_prc": prc, "trde_qty": qty}


def _kis_tick_row(h="093000", vol="1000", prpr="70000"):
    return {"stck_cntg_hour": h, "cnqn": vol, "stck_prpr": prpr}


def _aware_page_clocks():
    import datetime as _dt

    from src.data.capture_contracts import SEOUL as _SEOUL

    start = _dt.datetime(2026, 9, 4, 15, 30, tzinfo=_SEOUL)
    return start, start + _dt.timedelta(seconds=1)


def _capture_profile(tmp_path, routes=None):
    from src.config.collection import CollectionSettings

    return CollectionSettings(COLLECTION_ROOT=tmp_path / "capture", COLLECTION_VERIFIED_CHART_ROUTES=dict(routes or {}))


def _capture_store(tmp_path):
    from src.data.capture_store import CaptureStore

    return CaptureStore(tmp_path / "capture")


class _LsBars:
    def __init__(self, rows, terminal="exhausted", rt="0", pages=2):
        self._rows = rows
        self._terminal = terminal
        self._rt = rt
        self._pages = pages
        self.calls = []

    async def get_minute_chart(self, session, code, target_date, budget=None, on_page=None):
        self.calls.append({"code": code, "budget": budget})
        if on_page is not None:
            start, received = _aware_page_clocks()
            for idx in range(self._pages):
                on_page({"t8412OutBlock1": self._rows}, {"tr_cont": "N"}, start, received, idx, 0)
        done = self._terminal in ("exhausted", "crossed_target_date")
        return {
            "rt_cd": self._rt, "output2": self._rows if self._rt == "0" else [], "vendor": "ls",
            "truncated": not done, "termination_reason": self._terminal,
            "pages_fetched": self._pages, "continuation": {},
        }


class _KisBars:
    def __init__(self, rows, rt="0", fail=False):
        self._rows = rows
        self._rt = rt
        self._fail = fail
        self.historical_calls = []
        self.intraday_calls = []

    async def get_historical_minute_chart(self, session, code, target_date, **kwargs):
        self.historical_calls.append((code, target_date))
        if self._fail:
            raise RuntimeError("KIS unreachable")
        return {"rt_cd": self._rt, "output2": self._rows if self._rt == "0" else []}

    async def get_intraday_minute_chart(self, session, code, **kwargs):
        self.intraday_calls.append(code)
        if self._fail:
            raise RuntimeError("KIS unreachable")
        return {"rt_cd": self._rt, "output2": self._rows if self._rt == "0" else []}

    async def get_intraday_trade_ticks(self, session, code, **kwargs):
        self.intraday_calls.append(code)
        if self._fail:
            raise RuntimeError("KIS unreachable")
        return {"rt_cd": self._rt, "output2": self._rows if self._rt == "0" else []}


class _LsTicks:
    def __init__(self, rows, terminal="exhausted", truncated=False, fail=False):
        self._rows = rows
        self._terminal = terminal
        self._truncated = truncated
        self._fail = fail
        self.calls = []

    async def get_tick_chart(self, session, code, target_date, max_pages=None, budget=None, on_page=None):
        self.calls.append({"code": code, "max_pages": max_pages, "budget": budget})
        if self._fail:
            raise RuntimeError("LS unreachable")
        if on_page is not None:
            start, received = _aware_page_clocks()
            on_page({"t8411OutBlock1": self._rows}, {"tr_cont": "N"}, start, received, 0, 0)
        return {
            "rt_cd": "0", "output2": self._rows, "vendor": "ls", "truncated": self._truncated,
            "termination_reason": self._terminal, "pages_fetched": 1, "continuation": {},
        }


class _KiwoomTicks:
    def __init__(self, batches, fail=False):
        self._batches = list(batches)
        self._fail = fail
        self.calls = []

    async def get_tick_chart(self, session, code, target_date, max_pages=None, budget=None, on_page=None):
        self.calls.append({"max_pages": max_pages, "budget": budget})
        if self._fail:
            raise RuntimeError("kiwoom unreachable")
        payload = self._batches.pop(0) if self._batches else {"rows": [], "truncated": False, "terminal": "exhausted"}
        if on_page is not None:
            start, received = _aware_page_clocks()
            on_page({"stk_tic_chart_qry": payload["rows"]}, {"cont-yn": "N"}, start, received, 0, 0)
        return {
            "rt_cd": "0", "output2": payload["rows"], "vendor": "kiwoom",
            "truncated": payload["truncated"], "termination_reason": payload["terminal"],
            "pages_fetched": 1, "continuation": {},
        }


def test_collect_bars_restores_morning_and_excludes_extended(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8412": "KRX"})
    ls_client = _LsBars([_ls_bar_row("090100"), _ls_bar_row("160000")])
    delivered = {}

    result = asyncio.run(
        collect_intraday_bars(
            _KisBars([]), None, ["005930"], "2026-09-04", 1, ls_client=ls_client,
            profile=profile, capture_store=store, run_id="run-morning",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert result.empty
    frame, entry = delivered["005930"]
    assert entry.status.value == "COMPLETE"
    assert frame["ts_hms"].tolist() == [90100]
    assert (store.root / "raw").exists()
    assert list((store.root / "normalized").rglob("*.parquet"))
    manifests = store.read_manifests("2026-09-04")
    assert any(item.status.value == "PENDING" for item in manifests)


def test_collect_bars_unknown_venue_preserves_raw(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {})
    ls_client = _LsBars([_ls_bar_row("090100")])
    delivered = {}

    asyncio.run(
        collect_intraday_bars(
            _KisBars([], rt="1"), None, ["005930"], "2026-09-04", 1, ls_client=ls_client,
            profile=profile, capture_store=store, run_id="run-unknown",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    frame, entry = delivered["005930"]
    assert entry.status.value in ("UNKNOWN", "PARTIAL")
    assert frame.empty
    assert list((store.root / "raw").rglob("*.json.gz"))


def test_collect_ticks_truncated_triggers_bounded_repair(tmp_path) -> None:
    import asyncio
    import datetime as _dt

    from src.backfill.intraday.collector import collect_intraday_trade_ticks
    from src.data.capture_contracts import SEOUL as _SEOUL

    today = _dt.datetime.now(_SEOUL).date().isoformat()
    ymd = today.replace("-", "")
    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8411": "KRX"})
    kiwoom = _KiwoomTicks([
        {"rows": [_kw_tick_row(f"{ymd}093000")], "truncated": True, "terminal": "page_budget"},
        {"rows": [_kw_tick_row(f"{ymd}093000")], "truncated": True, "terminal": "page_budget"},
    ])
    ls_client = _LsTicks([_ls_tick_row("093000", close=71000, day=ymd)])
    delivered = {}

    asyncio.run(
        collect_intraday_trade_ticks(
            _KisBars([]), None, ["005930"], today, ls_client=ls_client, kiwoom_client=kiwoom,
            profile=profile, capture_store=store, run_id="run-repair",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    frame, entry = delivered["005930"]
    assert entry.status.value == "COMPLETE"
    assert frame["price"].tolist() == [71000]
    assert len(kiwoom.calls) == 2
    assert kiwoom.calls[1]["budget"] is not None
    assert kiwoom.calls[1]["budget"].max_pages == profile.COLLECTION_TICK_REPAIR_MAX_PAGES


def test_collect_ticks_whole_source_fallback_avoids_union(tmp_path) -> None:
    import asyncio
    import datetime as _dt

    from src.backfill.intraday.collector import collect_intraday_trade_ticks
    from src.data.capture_contracts import SEOUL as _SEOUL

    today = _dt.datetime.now(_SEOUL).date().isoformat()
    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {})
    ls_client = _LsTicks([_ls_tick_row("093000", close=70000)], terminal="page_budget", truncated=True)
    kis = _KisBars([_kis_tick_row("093000", vol="5", prpr="71000")])
    delivered = {}

    asyncio.run(
        collect_intraday_trade_ticks(
            kis, None, ["005930"], today, ls_client=ls_client,
            profile=profile, capture_store=store, run_id="run-union",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    frame, entry = delivered["005930"]
    assert entry.status.value == "COMPLETE"
    assert entry.venue == "KRX"
    assert frame["price"].tolist() == [71000]
    assert frame["vendor"].tolist() == ["kis"]


def test_collect_bars_sparse_trading_not_synthetic_loss(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8412": "KRX"})
    ls_client = _LsBars([_ls_bar_row("090000"), _ls_bar_row("150000")])
    delivered = {}

    asyncio.run(
        collect_intraday_bars(
            _KisBars([]), None, ["005930"], "2026-09-04", 1, ls_client=ls_client,
            profile=profile, capture_store=store, run_id="run-sparse",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    frame, entry = delivered["005930"]
    assert entry.status.value == "COMPLETE"
    assert len(frame) == 2


def test_collect_ticks_observer_mode_stays_bounded(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8411": "KRX"})
    ls_client = _LsTicks([_ls_tick_row("093000")])
    delivered = {}
    real_to_dict = pd.DataFrame.to_dict

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("observer mode must not accumulate global records")

    pd.DataFrame.to_dict = _forbidden
    try:
        result = asyncio.run(
            collect_intraday_trade_ticks(
                _KisBars([]), None, ["005930", "000660", "035420"], "2020-01-02",
                ls_client=ls_client, profile=profile, capture_store=store,
                run_id="run-bounded",
                on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
            )
        )
    finally:
        pd.DataFrame.to_dict = real_to_dict

    assert set(delivered) == {"005930", "000660", "035420"}
    assert "symbol" in result.columns
    assert result.empty


def test_collect_bars_historical_repair_uses_historical_date(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8412": "KRX"})
    ls_client = _LsBars([_ls_bar_row("090100", day="20200102")], terminal="page_budget")
    kis = _KisBars([_kis_bar_row("090100")])
    delivered = {}

    asyncio.run(
        collect_intraday_bars(
            kis, None, ["005930"], "2020-01-02", 1, ls_client=ls_client,
            profile=profile, capture_store=store, run_id="run-hist",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    frame, entry = delivered["005930"]
    assert entry.status.value == "COMPLETE"
    assert kis.historical_calls == [("005930", "20200102")]
    assert kis.intraday_calls == []


def test_collect_ticks_failed_and_empty_symbols_stay_visible(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8411": "KRX"})

    class _MixedLs:
        async def get_tick_chart(self, session, code, target_date, max_pages=None, budget=None, on_page=None):
            if code == "005930":
                start, received = _aware_page_clocks()
                if on_page is not None:
                    on_page({"t8411OutBlock1": [_ls_tick_row("093000", day="20200102")]}, {"tr_cont": "N"}, start, received, 0, 0)
                return {"rt_cd": "0", "output2": [_ls_tick_row("093000", day="20200102")], "vendor": "ls",
                        "truncated": False, "termination_reason": "exhausted", "pages_fetched": 1, "continuation": {}}
            if code == "000660":
                raise RuntimeError("LS transport boom")
            start, received = _aware_page_clocks()
            if on_page is not None:
                on_page({"t8411OutBlock1": []}, {"tr_cont": "N"}, start, received, 0, 0)
            return {"rt_cd": "0", "output2": [], "vendor": "ls",
                    "truncated": False, "termination_reason": "exhausted", "pages_fetched": 1, "continuation": {}}

    delivered = {}
    asyncio.run(
        collect_intraday_trade_ticks(
            _KisBars([]), None, ["005930", "000660", "035420"], "2020-01-02",
            ls_client=_MixedLs(), profile=profile, capture_store=store, run_id="run-visible",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert delivered["005930"][1].status.value == "COMPLETE"
    assert delivered["000660"][1].status.value == "FAILED"
    assert delivered["035420"][1].status.value == "UNKNOWN"


def test_collect_bars_nxt_kiwoom_complete_and_empty_unknown(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"kiwoom:ka10080": "NXT"})

    class _KwNxt:
        async def get_nxt_minute_chart(self, session, code, target_date):
            if code == "005930":
                return {"rt_cd": "0", "vendor": "kiwoom", "output2": [
                    {"cntr_tm": "20260904160000", "cur_prc": "+70000", "open_pric": "+70000",
                     "high_pric": "+70100", "low_pric": "+69900", "trde_qty": "10"},
                ]}
            return {"rt_cd": "0", "vendor": "kiwoom", "output2": []}

    delivered = {}
    asyncio.run(
        collect_nxt_aftermarket_bars(
            _KisBars([]), None, ["005930", "000660"], "2026-09-04", 1, kiwoom_client=_KwNxt(),
            profile=profile, capture_store=store, run_id="run-nxt",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert delivered["005930"][1].status.value == "COMPLETE"
    assert len(delivered["005930"][0]) == 1
    assert delivered["000660"][1].status.value == "UNKNOWN"


def test_collect_krx_aftermarket_capture_before_launch_and_kis_complete(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_krx_aftermarket_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {})
    delivered = {}
    result = asyncio.run(
        collect_krx_aftermarket_bars(
            _KisBars([]), None, ["005930"], "2026-09-01", 1,
            profile=profile, capture_store=store, run_id="run-krx-na",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert result.empty
    assert delivered["005930"][1].status.value == "UNKNOWN"

    kis = _KisBars([_kis_bar_row("160100")])
    delivered.clear()
    asyncio.run(
        collect_krx_aftermarket_bars(
            kis, None, ["005930"], "2026-09-14", 1,
            profile=profile, capture_store=store, run_id="run-krx",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert delivered["005930"][1].status.value == "COMPLETE"
    assert delivered["005930"][1].venue == "KRX"


def test_collect_capture_boundaries_and_helpers(tmp_path) -> None:
    import asyncio

    import pytest

    from src.backfill.intraday import collector as collector_mod
    from src.backfill.intraday.collector import collect_intraday_bars, collect_intraday_trade_ticks

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8412": "KRX"})
    assert collector_mod._redacted_error(RuntimeError("x")) == "RuntimeError"
    assert collector_mod._venue_for(vendor="kis", endpoint="x", market_div_code="J", profile=profile) == "KRX"
    assert collector_mod._venue_for(vendor="zzz", endpoint="x", market_div_code="Q", profile=profile) == "UNKNOWN"
    assert collector_mod._event_key({"cntr_tm": "xx"}, "kiwoom") == ("", "")
    assert collector_mod._is_past_date("2020-01-02") is True
    with pytest.raises(ValueError, match="snapshot_date"):
        asyncio.run(collect_intraday_bars(_KisBars([]), None, ["005930"], "not-a-date", 1, profile=profile, capture_store=store))
    with pytest.raises(ValueError, match="bar_interval"):
        asyncio.run(collect_intraday_bars(_KisBars([]), None, ["005930"], "2026-09-04", 0, profile=profile, capture_store=store))
    with pytest.raises(ValueError, match="ls_max_pages"):
        asyncio.run(collect_intraday_trade_ticks(_KisBars([]), None, ["005930"], "2026-09-04", ls_max_pages=0, profile=profile, capture_store=store))
    with pytest.raises(ValueError, match="run_id"):
        asyncio.run(collect_intraday_bars(_KisBars([]), None, ["005930"], "2026-09-04", 1, profile=profile, capture_store=store, run_id="  "))
    combined = asyncio.run(
        collect_intraday_bars(
            _KisBars([_kis_bar_row("090100")]), None, ["005930"], "2026-09-04", 1,
            profile=profile, capture_store=store, run_id="run-combined",
        )
    )
    assert len(combined) == 1
    assert set(combined["symbol"]) == {"005930"}
    empty = asyncio.run(
        collect_intraday_bars(
            _KisBars([]), None, [], "2026-09-04", 1,
            profile=profile, capture_store=store, run_id="run-emptycodes",
        )
    )
    assert empty.empty
    failed_ls = _LsBars([{"time": "090100"}], terminal="exhausted")
    delivered = {}
    asyncio.run(
        collect_intraday_bars(
            _KisBars([], rt="1"), None, ["005930"], "2026-09-04", 1, ls_client=failed_ls,
            profile=profile, capture_store=store, run_id="run-failednorm",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value in ("PARTIAL", "FAILED", "UNKNOWN")


def test_collect_capture_store_resolution_and_run_identity(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday import collector as collector_mod
    from src.backfill.intraday.collector import collect_intraday_bars
    from src.config.collection import CollectionSettings

    profile = CollectionSettings(COLLECTION_ROOT=tmp_path / "cap")
    kis = _KisBars([_kis_bar_row("090100")])
    result = asyncio.run(
        collect_intraday_bars(kis, None, ["005930"], "2020-01-02", 1, profile=profile, run_id=None)
    )
    assert len(result) == 1
    from src.data.capture_store import CaptureStore

    manifests = CaptureStore(tmp_path / "cap").read_manifests("2020-01-02")
    assert any(item.context.run_id.startswith("intraday-2020-01-02-minute_bars-") for item in manifests)
    assert collector_mod._capture_root(CollectionSettings(COLLECTION_ROOT=tmp_path / "cap2")) == tmp_path / "cap2"


def test_collect_capture_default_root_and_split_helpers(tmp_path, monkeypatch) -> None:
    import asyncio

    from src import settings as _settings
    from src.backfill.intraday import collector as collector_mod
    from src.backfill.intraday.collector import collect_intraday_bars
    from src.config.collection import CollectionSettings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    profile = CollectionSettings(COLLECTION_ROOT=None)
    assert collector_mod._capture_root(profile) == tmp_path / "capture"
    regular, other = collector_mod._split_session_window(
        [{"date": "20200102", "time": "090100"}, None, {"date": "20200102", "time": "160000"}],
        "20200102", "090000", "153000", "ls",
    )
    assert regular == [{"date": "20200102", "time": "090100"}]
    assert other == [{"date": "20200102", "time": "160000"}]
    kis = _KisBars([_kis_bar_row("090100")])
    result = asyncio.run(collect_intraday_bars(kis, None, ["005930"], "2020-01-02", 1, profile=profile))
    assert len(result) == 1


def test_collect_bars_kis_transport_and_classification_branches(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {})
    delivered = {}

    async def _run(rows=None, rt="0", fail=False, snap="2026-09-04", codes=("005930",), run="r"):
        delivered.clear()
        kis = _KisBars(rows or [], rt=rt, fail=fail)
        await collect_intraday_bars(
            kis, None, list(codes), snap, 1, profile=profile, capture_store=store, run_id=run,
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
        return delivered

    today = __import__("datetime").datetime.now(__import__("src.data.capture_contracts", fromlist=["SEOUL"]).SEOUL).date().isoformat()
    awaitable = asyncio.run(_run(fail=True, snap=today, run="r-fail"))
    assert awaitable["005930"][1].status.value == "FAILED"
    extended = asyncio.run(_run(rows=[_kis_bar_row("090100"), _kis_bar_row("160000")], snap=today, run="r-ext"))
    assert extended["005930"][1].status.value == "COMPLETE"
    assert extended["005930"][0]["ts_hms"].tolist() == [90100]
    norm_fail = asyncio.run(_run(rows=[{"stck_cntg_hour": "090100"}], snap=today, run="r-norm"))
    assert norm_fail["005930"][1].status.value == "FAILED"
    empty = asyncio.run(_run(rows=[], snap=today, run="r-empty"))
    assert empty["005930"][1].status.value == "UNKNOWN"


def test_collect_bars_ls_failure_branches(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {"ls:t8412": "KRX"})
    delivered = {}

    class _BoomLs:
        async def get_minute_chart(self, *args, **kwargs):
            raise RuntimeError("LS down")

    asyncio.run(
        collect_intraday_bars(
            _KisBars([], fail=True), None, ["005930"], "2026-09-04", 1, ls_client=_BoomLs(),
            profile=profile, capture_store=store, run_id="r-lsboom",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value == "PARTIAL"

    delivered.clear()
    asyncio.run(
        collect_intraday_bars(
            _KisBars([_kis_bar_row("090100")]), None, ["005930"], "2026-09-04", 1,
            ls_client=_LsBars([], rt="1"), profile=profile, capture_store=store, run_id="r-lsvfail",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value == "COMPLETE"
    assert len(delivered["005930"][1].raw_refs) >= 2

    delivered.clear()
    asyncio.run(
        collect_intraday_bars(
            _KisBars([], rt="1"), None, ["005930"], "2026-09-04", 1,
            ls_client=_LsBars([{"time": "090100"}]), profile=profile, capture_store=store, run_id="r-lsnorm",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value in ("PARTIAL", "FAILED", "UNKNOWN")

    delivered.clear()
    asyncio.run(
        collect_intraday_bars(
            _KisBars([], rt="1"), None, ["005930"], "2026-09-04", 1,
            ls_client=_LsBars([{"time": "160000"}]), profile=profile, capture_store=store, run_id="r-lsext",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value in ("PARTIAL", "UNKNOWN")


def test_collect_ticks_uncertified_and_repair_branches(tmp_path) -> None:
    import asyncio
    import datetime as _dt

    from src.backfill.intraday.collector import collect_intraday_trade_ticks
    from src.data.capture_contracts import SEOUL as _SEOUL

    today = _dt.datetime.now(_SEOUL).date().isoformat()
    ymd = today.replace("-", "")
    delivered = {}

    async def _run(kiwoom, ls, kis, snap, run, **kwargs):
        delivered.clear()
        store = _capture_store(tmp_path)
        profile = kwargs.pop("profile", _capture_profile(tmp_path, {}))
        await collect_intraday_trade_ticks(
            kis, None, ["005930"], snap, ls_client=ls, kiwoom_client=kiwoom,
            profile=profile, capture_store=store, run_id=run,
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
        return delivered["005930"], store

    frame_entry, _ = asyncio.run(_run(None, _LsTicks([_ls_tick_row("093000", day=ymd)]), _KisBars([]), today, "r-tick-unk"))
    assert frame_entry[1].status.value == "UNKNOWN"
    assert frame_entry[0].empty

    # Kiwoom transport failure falls through to LS COMPLETE
    class _BoomKw:
        async def get_tick_chart(self, *args, **kwargs):
            raise RuntimeError("kw down")

    frame_entry, _ = asyncio.run(_run(_BoomKw(), _LsTicks([_ls_tick_row("093000", day=ymd)]), _KisBars([]), today, "r-tick-kwboom",
                                      profile=_capture_profile(tmp_path, {"ls:t8411": "KRX"})))
    assert frame_entry[1].status.value == "COMPLETE"

    # Kiwoom truncated, repair transport failure -> FAILED
    class _KwRepairBoom:
        def __init__(self):
            self.calls = 0

        async def get_tick_chart(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"rt_cd": "0", "output2": [_kw_tick_row(f"{ymd}093000")], "vendor": "kiwoom",
                        "truncated": True, "termination_reason": "page_budget", "pages_fetched": 1, "continuation": {}}
            raise RuntimeError("repair down")

    frame_entry, _ = asyncio.run(_run(_KwRepairBoom(), None, _KisBars([], fail=True), today, "r-tick-repboom"))
    assert frame_entry[1].status.value == "FAILED"

    # Kiwoom truncated then repair COMPLETE (certified venue)
    kw = _KiwoomTicks([
        {"rows": [_kw_tick_row(f"{ymd}093000")], "truncated": True, "terminal": "page_budget"},
        {"rows": [_kw_tick_row(f"{ymd}093100")], "truncated": False, "terminal": "exhausted"},
    ])
    frame_entry, _ = asyncio.run(_run(kw, None, _KisBars([]), today, "r-tick-repok",
                                      profile=_capture_profile(tmp_path, {"kiwoom:ka10079": "NXT"})))
    assert frame_entry[1].status.value == "COMPLETE"
    assert frame_entry[0]["ts_hms"].tolist() == [93100]

    # KIS transport / vendor failure / empty on today date
    frame_entry, _ = asyncio.run(_run(None, _LsTicks([_ls_tick_row("093000", day=ymd)], terminal="page_budget", truncated=True), _KisBars([], fail=True), today, "r-tick-kisfail"))
    assert frame_entry[1].status.value == "FAILED"
    frame_entry, _ = asyncio.run(_run(None, _LsTicks([_ls_tick_row("093000", day=ymd)], terminal="page_budget", truncated=True), _KisBars([], rt="9"), today, "r-tick-kisrt"))
    assert frame_entry[1].status.value == "FAILED"
    frame_entry, _ = asyncio.run(_run(None, _LsTicks([_ls_tick_row("093000", day=ymd)], terminal="page_budget", truncated=True), _KisBars([]), today, "r-tick-kisempty"))
    assert frame_entry[1].status.value == "UNKNOWN"

    # Past date, truncated everywhere -> PARTIAL; no clients at all -> UNKNOWN
    frame_entry, _ = asyncio.run(_run(None, _LsTicks([_ls_tick_row("093000", day="20200102")], terminal="page_budget", truncated=True), _KisBars([]), "2020-01-02", "r-tick-partial"))
    assert frame_entry[1].status.value == "PARTIAL"
    frame_entry, _ = asyncio.run(_run(None, None, _KisBars([]), "2020-01-02", "r-tick-nosrc"))
    assert frame_entry[1].status.value == "UNKNOWN"

    # Truncated LS rows that fail normalization still stage safely -> PARTIAL on past date
    frame_entry, _ = asyncio.run(_run(None, _LsTicks([{"close": 1}], terminal="page_budget", truncated=True), _KisBars([]), "2020-01-02", "r-tick-badnorm"))
    assert frame_entry[1].status.value == "PARTIAL"


def test_collect_extended_branches_and_premarket(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_nxt_aftermarket_bars, collect_nxt_premarket_bars

    delivered = {}

    class _BoomKw:
        async def get_nxt_minute_chart(self, *args, **kwargs):
            raise RuntimeError("kw down")

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {})
    kis = _KisBars([_kis_bar_row("160000")])
    asyncio.run(
        collect_nxt_aftermarket_bars(
            kis, None, ["005930"], "2026-09-04", 1, kiwoom_client=_BoomKw(),
            profile=profile, capture_store=store, run_id="r-nxt-boom",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value == "UNKNOWN"

    class _BadKw:
        async def get_nxt_minute_chart(self, *args, **kwargs):
            return {"rt_cd": "0", "vendor": "kiwoom", "output2": [{"cntr_tm": "20260904160000"}]}

    delivered.clear()
    asyncio.run(
        collect_nxt_aftermarket_bars(
            _KisBars([]), None, ["005930"], "2026-09-04", 1, kiwoom_client=_BadKw(),
            profile=_capture_profile(tmp_path, {"kiwoom:ka10080": "NXT"}), capture_store=store, run_id="r-nxt-bad",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value == "FAILED"

    class _UncertKw:
        async def get_nxt_minute_chart(self, *args, **kwargs):
            return {"rt_cd": "0", "vendor": "kiwoom", "output2": [
                {"cntr_tm": "20260904160000", "cur_prc": "+70000", "open_pric": "+70000",
                 "high_pric": "+70100", "low_pric": "+69900", "trde_qty": "10"},
            ]}

    delivered.clear()
    asyncio.run(
        collect_nxt_aftermarket_bars(
            _KisBars([]), None, ["005930"], "2026-09-04", 1, kiwoom_client=_UncertKw(),
            profile=profile, capture_store=store, run_id="r-nxt-uncert",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value == "UNKNOWN"

    class _PreKw:
        async def get_nxt_premarket_chart(self, *args, **kwargs):
            return {"rt_cd": "0", "vendor": "kiwoom", "output2": [
                {"cntr_tm": "20260904083000", "cur_prc": "+70000", "open_pric": "+70000",
                 "high_pric": "+70100", "low_pric": "+69900", "trde_qty": "10"},
            ]}

    delivered.clear()
    asyncio.run(
        collect_nxt_premarket_bars(
            _KisBars([]), None, ["005930"], "2026-09-04", 1, kiwoom_client=_PreKw(),
            profile=_capture_profile(tmp_path, {"kiwoom:ka10080": "NXT"}), capture_store=store, run_id="r-pre",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value == "COMPLETE"
    assert delivered["005930"][0]["ts_hms"].tolist() == [83000]


def test_collect_ticks_primary_and_vendor_failure_branches(tmp_path) -> None:
    import asyncio
    import datetime as _dt

    from src.backfill.intraday.collector import collect_intraday_trade_ticks
    from src.data.capture_contracts import SEOUL as _SEOUL

    today = _dt.datetime.now(_SEOUL).date().isoformat()
    ymd = today.replace("-", "")
    delivered = {}
    store = _capture_store(tmp_path)

    async def _run(kiwoom, ls, kis, snap, run, routes=None):
        delivered.clear()
        await collect_intraday_trade_ticks(
            kis, None, ["005930"], snap, ls_client=ls, kiwoom_client=kiwoom,
            profile=_capture_profile(tmp_path, routes or {}), capture_store=store, run_id=run,
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
        return delivered["005930"]

    # Primary Kiwoom COMPLETE short-circuits repair and fallback sources.
    kw = _KiwoomTicks([{"rows": [_kw_tick_row(f"{ymd}093000")], "truncated": False, "terminal": "exhausted"}])
    ls_client = _LsTicks([_ls_tick_row("093000", day=ymd)])
    kis = _KisBars([_kis_tick_row("093000")])
    frame_entry = asyncio.run(_run(kw, ls_client, kis, today, "r-tick-kwfull", {"kiwoom:ka10079": "NXT"}))
    assert frame_entry[1].status.value == "COMPLETE"
    assert frame_entry[1].venue == "NXT"
    assert len(ls_client.calls) == 0
    assert kis.intraday_calls == []

    # Dict vendor failure without transport exception classifies FAILED directly.
    class _RtFailLs:
        async def get_tick_chart(self, *args, **kwargs):
            return {"rt_cd": "9", "msg1": "denied", "output2": []}

    frame_entry = asyncio.run(_run(None, _RtFailLs(), _KisBars([]), "2020-01-02", "r-tick-rtfail"))
    assert frame_entry[1].status.value == "FAILED"


def test_collect_bars_kis_filtered_empty_unknown(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars

    store = _capture_store(tmp_path)
    profile = _capture_profile(tmp_path, {})
    delivered = {}
    rows = [dict(_kis_bar_row("090100"), stck_bsop_date="20200101")]
    asyncio.run(
        collect_intraday_bars(
            _KisBars(rows), None, ["005930"], "2026-09-04", 1,
            profile=profile, capture_store=store, run_id="r-kis-filtered",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )
    assert delivered["005930"][1].status.value == "UNKNOWN"
    assert delivered["005930"][0].empty


def test_publish_pending_manifest_retry_does_not_raise(tmp_path) -> None:
    """동일 run_id로 재실행 시 completed_at 차이로 인한 immutable identity 충돌을 흡수한다.

    실측: 2026-09-18 archive-intraday 재시도가 이 충돌(ValueError)로 전체 아카이브를
    실패시킴 -- PENDING 매니페스트는 진행 중 체크포인트일 뿐이라 발행 실패해도
    실제 수집은 계속돼야 한다.
    """
    from datetime import date

    from src.backfill.intraday.collector import _publish_pending_manifest
    from src.data.capture_contracts import CaptureDataset

    store = _capture_store(tmp_path)
    kwargs = {
        "store": store,
        "trading_day": date(2026, 9, 18),
        "run_id": "archive-2026-09-18-regular-bars",
        "dataset": CaptureDataset.MINUTE_BARS,
        "vendor": "owner-local",
        "endpoint": "pending",
        "session": "regular",
        "symbols": ["005930"],
    }
    _publish_pending_manifest(**kwargs)
    # 재실행: completed_at이 달라져 동일 경로에 다른 바이트를 쓰려는 충돌이 발생하지만
    # 예외가 밖으로 전파되지 않아야 한다.
    _publish_pending_manifest(**kwargs)


def test_collect_with_observer_default_runs_sequentially(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import _collect_with_observer
    from src.data.capture_contracts import CaptureDataset

    store = _capture_store(tmp_path)
    codes = ["005930", "000660", "035420"]
    entered: list[str] = []

    async def _acquire(code: str):
        entered.append(code)
        return __import__("pandas").DataFrame(), None

    asyncio.run(
        _collect_with_observer(
            codes=codes, snapshot_date="2026-09-04", dataset=CaptureDataset.MINUTE_BARS,
            session_tag="regular", store=store, run_id="run-seq-default",
            acquire=_acquire, on_symbol=None,
        )
    )

    assert entered == codes


def test_collect_with_observer_default_preserves_observer_and_concat_contracts(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.backfill.intraday.collector import _collect_with_observer
    from src.data.capture_contracts import CaptureDataset

    store = _capture_store(tmp_path)

    async def _acquire(code: str):
        return pd.DataFrame([{"symbol": code}]), None

    fired: list[str] = []

    def _on_symbol(symbol, frame, entry) -> None:
        fired.append(symbol)

    observed = asyncio.run(
        _collect_with_observer(
            codes=["005930", "000660", "035420"], snapshot_date="2026-09-04",
            dataset=CaptureDataset.MINUTE_BARS, session_tag="regular", store=store,
            run_id="run-seq-observed", acquire=_acquire, on_symbol=_on_symbol,
        )
    )

    assert fired == ["005930", "000660", "035420"]
    assert observed.empty

    combined = asyncio.run(
        _collect_with_observer(
            codes=["005930", "000660", "035420"], snapshot_date="2026-09-04",
            dataset=CaptureDataset.MINUTE_BARS, session_tag="regular", store=store,
            run_id="run-seq-combined", acquire=_acquire, on_symbol=None,
        )
    )

    assert combined["symbol"].tolist() == ["005930", "000660", "035420"]


def test_collect_with_observer_bounds_concurrent_acquire(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import _collect_with_observer
    from src.data.capture_contracts import CaptureDataset

    store = _capture_store(tmp_path)
    inflight = 0
    peak = 0

    async def _acquire(code: str):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await asyncio.sleep(0)
            return __import__("pandas").DataFrame(), None
        finally:
            inflight -= 1

    delivered: dict[str, int] = {}

    def _on_symbol(symbol, frame, entry) -> None:
        delivered[symbol] = delivered.get(symbol, 0) + 1

    asyncio.run(
        _collect_with_observer(
            codes=["005930", "000660", "035420", "035720", "051910"], snapshot_date="2026-09-04",
            dataset=CaptureDataset.MINUTE_BARS, session_tag="regular", store=store,
            run_id="run-bound", acquire=_acquire, on_symbol=_on_symbol,
            max_concurrency=2,
        )
    )

    assert peak <= 2
    assert peak == 2
    assert len(delivered) == 5


def test_collect_with_observer_delivers_each_code_once(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import _collect_with_observer
    from src.data.capture_contracts import CaptureDataset

    store = _capture_store(tmp_path)
    codes = ["005930", "000660", "035420", "035720", "051910"]
    delivered: dict[str, int] = {}

    async def _acquire(code: str):
        await asyncio.sleep(0)
        return __import__("pandas").DataFrame(), None

    def _on_symbol(symbol, frame, entry) -> None:
        delivered[symbol] = delivered.get(symbol, 0) + 1

    asyncio.run(
        _collect_with_observer(
            codes=codes, snapshot_date="2026-09-04", dataset=CaptureDataset.MINUTE_BARS,
            session_tag="regular", store=store, run_id="run-once",
            acquire=_acquire, on_symbol=_on_symbol, max_concurrency=3,
        )
    )

    assert set(delivered) == set(codes)
    assert all(count == 1 for count in delivered.values())


def test_collect_with_observer_rejects_nonpositive_concurrency(tmp_path) -> None:
    import asyncio

    import pytest

    from src.backfill.intraday.collector import _collect_with_observer
    from src.data.capture_contracts import CaptureDataset

    store = _capture_store(tmp_path)

    async def _acquire(code: str):
        raise AssertionError("must fail before acquiring")

    with pytest.raises(ValueError, match="0"):
        asyncio.run(
            _collect_with_observer(
                codes=["005930"], snapshot_date="2026-09-04",
                dataset=CaptureDataset.MINUTE_BARS, session_tag="regular",
                store=store, run_id="run-invalid", acquire=_acquire,
                on_symbol=None, max_concurrency=0,
            )
        )


def test_collect_with_observer_concat_covers_all_rows_regardless_of_order(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.backfill.intraday.collector import _collect_with_observer
    from src.data.capture_contracts import CaptureDataset

    store = _capture_store(tmp_path)
    codes = ["005930", "000660", "035420"]

    async def _acquire(code: str):
        await asyncio.sleep((len(codes) - 1 - codes.index(code)) * 0.02)
        return pd.DataFrame([{"symbol": code}]), None

    result = asyncio.run(
        _collect_with_observer(
            codes=codes, snapshot_date="2026-09-04", dataset=CaptureDataset.MINUTE_BARS,
            session_tag="regular", store=store, run_id="run-concat",
            acquire=_acquire, on_symbol=None, max_concurrency=3,
        )
    )

    assert set(result["symbol"]) == set(codes)
    assert len(result) == len(codes)


def test_collect_intraday_trade_ticks_bounds_inflight_by_profile(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_trade_ticks
    from src.config.collection import CollectionSettings

    inflight = 0
    peak = 0

    class _TrackingLs:
        async def get_tick_chart(self, session, code, target_date, max_pages=None, on_page=None):
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            try:
                await asyncio.sleep(0.01)
                return {
                    "rt_cd": "0", "output2": [_ls_tick_row("093000", day="20200102")],
                    "vendor": "ls", "truncated": False, "termination_reason": "exhausted",
                    "pages_fetched": 1, "continuation": {},
                }
            finally:
                inflight -= 1

    store = _capture_store(tmp_path)
    profile = CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture",
        COLLECTION_VERIFIED_CHART_ROUTES={"ls:t8411": "KRX"},
        COLLECTION_CONCURRENCY_PER_KEY=4,
    )
    delivered = {}
    codes = ["005930", "000660", "035420", "035720", "051910", "068270"]

    asyncio.run(
        collect_intraday_trade_ticks(
            _KisBars([]), None, codes, "2020-01-02", ls_client=_TrackingLs(),
            profile=profile, capture_store=store, run_id="run-tick-conc",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert peak <= 4
    assert set(delivered) == set(codes)


def test_collect_intraday_bars_bounds_inflight_by_profile(tmp_path) -> None:
    import asyncio

    from src.backfill.intraday.collector import collect_intraday_bars
    from src.config.collection import CollectionSettings

    inflight = 0
    peak = 0

    class _TrackingLsBars:
        async def get_minute_chart(self, session, code, target_date, budget=None, on_page=None):
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            try:
                await asyncio.sleep(0.01)
                return {
                    "rt_cd": "0", "output2": [_ls_bar_row("090100")], "vendor": "ls",
                    "truncated": False, "termination_reason": "exhausted",
                    "pages_fetched": 1, "continuation": {},
                }
            finally:
                inflight -= 1

    store = _capture_store(tmp_path)
    profile = CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture",
        COLLECTION_VERIFIED_CHART_ROUTES={"ls:t8412": "KRX"},
        COLLECTION_CONCURRENCY_PER_KEY=2,
    )
    delivered = {}
    codes = ["005930", "000660", "035420", "035720", "051910"]

    asyncio.run(
        collect_intraday_bars(
            _KisBars([]), None, codes, "2026-09-04", 1, ls_client=_TrackingLsBars(),
            profile=profile, capture_store=store, run_id="run-bar-conc",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert peak <= 2
    assert set(delivered) == set(codes)


def test_collect_krx_aftermarket_bars_bounds_inflight_on_both_branches(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.backfill.intraday import collector as collector_mod
    from src.backfill.intraday.collector import collect_krx_aftermarket_bars
    from src.config.collection import CollectionSettings

    real_sleep = asyncio.sleep
    state = {"inflight": 0, "peak": 0}

    async def _tracking_sleep(delay, *args, **kwargs):
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        try:
            return await real_sleep(delay, *args, **kwargs)
        finally:
            state["inflight"] -= 1

    monkeypatch.setattr(asyncio, "sleep", _tracking_sleep)
    store = _capture_store(tmp_path)
    profile = CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture",
        COLLECTION_VERIFIED_CHART_ROUTES={},
        COLLECTION_CONCURRENCY_PER_KEY=2,
    )
    delivered = {}
    codes = ["005930", "000660", "035420", "035720", "051910", "068270"]

    asyncio.run(
        collect_krx_aftermarket_bars(
            _KisBars([]), None, codes, "2026-09-01", 1,
            profile=profile, capture_store=store, run_id="run-krx-na-conc",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert state["peak"] <= 2
    assert set(delivered) == set(codes)
    assert collector_mod.KRX_AFTERMARKET_START_DATE == "2026-09-14"

    inflight = 0
    peak = 0

    class _TrackingKis:
        async def get_intraday_minute_chart(self, session, code, **kwargs):
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            try:
                await real_sleep(0.01)
                return {"rt_cd": "0", "output2": [_kis_bar_row("160100")]}
            finally:
                inflight -= 1

    delivered.clear()
    asyncio.run(
        collect_krx_aftermarket_bars(
            _TrackingKis(), None, codes[:5], "2026-09-14", 1,
            profile=profile, capture_store=store, run_id="run-krx-conc",
            on_symbol=lambda symbol, frame, entry: delivered.update({symbol: (frame, entry)}),
        )
    )

    assert peak <= 2
    assert set(delivered) == set(codes[:5])
