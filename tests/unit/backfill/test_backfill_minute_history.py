from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from pathlib import Path

import pandas as pd
import pytest

from src.backfill.intraday import backfill_minute_history
from src.backfill.intraday.collector import backfill_regular_bars
from src.daily import archive
from src.data import intraday_store


def test_enumerate_backfill_targets_filters_lookback_and_sorts_oldest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({
        "스냅샷_날짜": ["2026-09-01", "2026-08-01", "2024-01-01"],
        "종목코드": ["5930", "660", "5380"],
    })
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: df)

    result = backfill_minute_history.enumerate_backfill_targets(as_of="2026-09-04", lookback_days=365)

    assert ("2024-01-01", "005380") not in result
    assert result == sorted(result)
    assert result[0][0] <= result[-1][0]
    assert ("2026-08-01", "000660") in result


def test_backfill_regular_bars_uses_historical_chart_with_market_div_j() -> None:
    client = AsyncMock()

    async def _fake_chart(session, code, target_date, bar_interval_minutes=1, end_hour=None, floor_hour=None, market_div_code=None):
        assert market_div_code == "J"
        assert target_date == "20260815"
        return {"rt_cd": "0", "output2": [{"stck_cntg_hour": "093000", "stck_prpr": "70000"}]}

    client.get_historical_minute_chart = _fake_chart

    result = asyncio.run(
        backfill_regular_bars(client, session=None, stock_codes=["005930"], snapshot_date="2026-08-15")
    )

    assert len(result) == 1
    assert result.iloc[0]["종목코드"] == "005930"
    assert result.iloc[0]["스냅샷_날짜"] == "2026-08-15"


def test_merge_and_write_partition_preserves_existing_codes_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(backfill_minute_history, "intraday_partition_path", intraday_store.intraday_partition_path)
    monkeypatch.setattr(backfill_minute_history, "write_intraday_partition", intraday_store.write_intraday_partition)

    first = pd.DataFrame({"종목코드": ["005930"], "stck_cntg_hour": ["093000"], "스냅샷_날짜": ["2026-08-15"]})
    backfill_minute_history._merge_and_write_partition(first, 1, "2026-08-15", "regular")

    second = pd.DataFrame({"종목코드": ["000660"], "stck_cntg_hour": ["093000"], "스냅샷_날짜": ["2026-08-15"]})
    backfill_minute_history._merge_and_write_partition(second, 1, "2026-08-15", "regular")

    stored = pd.read_parquet(intraday_store.intraday_partition_path(1, "2026-08-15", "regular"))

    assert set(stored["종목코드"]) == {"005930", "000660"}
