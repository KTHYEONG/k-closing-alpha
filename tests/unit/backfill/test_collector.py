from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from src.backfill.intraday.collector import collect_intraday_bars


def test_collect_intraday_bars_tags_rows_with_code_and_date() -> None:
    client = AsyncMock()
    client.get_intraday_minute_chart = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"stck_cntg_hour": "153000", "stck_prpr": "70000"}]}
    )

    result = asyncio.run(
        collect_intraday_bars(client, session=None, stock_codes=["005930", "000660"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["종목코드"]) == {"005930", "000660"}
    assert (result["스냅샷_날짜"] == "2026-09-03").all()


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
                {"stck_cntg_hour": "154000", "stck_prpr": "71000"},
                {"stck_cntg_hour": "154100", "stck_prpr": "71100"},
            ]}
        return {"rt_cd": "9", "msg1": "NXT 미상장"}

    client.get_intraday_minute_chart = _fake_chart

    result = asyncio.run(
        collect_nxt_aftermarket_bars(client, session=None, stock_codes=["005930", "000660"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["종목코드"]) == {"005930"}


def test_collect_intraday_trade_ticks_tags_rows_with_code_and_date() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.backfill.intraday.collector import collect_intraday_trade_ticks

    client = AsyncMock()
    client.get_intraday_trade_ticks = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"stck_cntg_hour": "093000", "acml_vol": "1000", "stck_prpr": "70000"}]}
    )

    result = asyncio.run(
        collect_intraday_trade_ticks(client, session=None, stock_codes=["005930", "000660"], snapshot_date="2026-09-03")
    )

    assert len(result) == 2
    assert set(result["종목코드"]) == {"005930", "000660"}
    assert (result["스냅샷_날짜"] == "2026-09-03").all()
