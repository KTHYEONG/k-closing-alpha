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
from src.data.intraday_schema import normalize_bar_frame


def _canon_bar(symbol: str, snapshot_date: str = "2026-08-15", close: int = 70000) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "time": ["093000"],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "jdiff_vol": [1000],
            "value": [70],
        }
    )
    return normalize_bar_frame(raw, "ls", snapshot_date, symbol)


def test_enumerate_backfill_targets_filters_lookback_and_sorts_oldest_first(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 레거시 condition_history_cleaned.parquet 병합 경로가 실제 프로덕션 파일을 읽지 않도록 격리한다.
    monkeypatch.setattr(backfill_minute_history.settings, "DATA_DIR", tmp_path)

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


def test_enumerate_backfill_targets_merges_legacy_condition_history_parquet(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backfill_minute_history.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: pd.DataFrame({
        "스냅샷_날짜": ["2026-08-10"], "종목코드": ["005930"],
    }))

    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    legacy_df = pd.DataFrame({
        "스냅샷_날짜": ["2025-12-29", "2026-01-05"],
        "종목코드": ["000660", "005380"],
    })
    legacy_df.to_parquet(history_dir / "condition_history_cleaned.parquet", index=False)

    result = backfill_minute_history.enumerate_backfill_targets(as_of="2026-09-04", lookback_days=365)

    assert ("2025-12-29", "000660") in result
    assert ("2026-01-05", "005380") in result
    assert ("2026-08-10", "005930") in result
    assert result[0] == ("2025-12-29", "000660")


def test_backfill_regular_bars_uses_historical_chart_with_market_div_j() -> None:
    client = AsyncMock()

    async def _fake_chart(session, code, target_date, bar_interval_minutes=1, end_hour=None, floor_hour=None, market_div_code=None):
        assert market_div_code == "J"
        assert target_date == "20260815"
        return {"rt_cd": "0", "output2": [{
            "stck_cntg_hour": "093000", "stck_oprc": "70000", "stck_hgpr": "70000",
            "stck_lwpr": "70000", "stck_prpr": "70000", "cntg_vol": "1000", "acml_tr_pbmn": "70000000",
        }]}

    client.get_historical_minute_chart = _fake_chart

    result = asyncio.run(
        backfill_regular_bars(client, session=None, stock_codes=["005930"], snapshot_date="2026-08-15")
    )

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "005930"
    assert result.iloc[0]["snapshot_date"] == "2026-08-15"


def test_merge_and_write_partition_preserves_existing_codes_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(backfill_minute_history, "intraday_partition_path", intraday_store.intraday_partition_path)
    monkeypatch.setattr(backfill_minute_history, "write_intraday_partition", intraday_store.write_intraday_partition)

    first = _canon_bar("005930")
    backfill_minute_history._merge_and_write_partition(first, 1, "2026-08-15", "regular")

    second = _canon_bar("000660")
    backfill_minute_history._merge_and_write_partition(second, 1, "2026-08-15", "regular")

    stored = pd.read_parquet(intraday_store.intraday_partition_path(1, "2026-08-15", "regular"))

    assert set(stored["symbol"]) == {"005930", "000660"}


def test_enumerate_backfill_targets_merges_trade_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backfill_minute_history.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: pd.DataFrame())

    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True)
    trade_log_df = pd.DataFrame({
        "매수날짜": ["2026-02-10", "2026-03-01"],
        "종목코드": ["035420", "000270"],
    })
    trade_log_df.to_parquet(parquet_dir / "trade_log.parquet", index=False)

    result = backfill_minute_history.enumerate_backfill_targets(as_of="2026-09-04", lookback_days=365)

    assert ("2026-02-10", "035420") in result
    assert ("2026-03-01", "000270") in result


def test_already_collected_codes_reflects_existing_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(backfill_minute_history, "intraday_partition_path", intraday_store.intraday_partition_path)
    monkeypatch.setattr(backfill_minute_history, "write_intraday_partition", intraday_store.write_intraday_partition)

    assert backfill_minute_history._already_collected_codes(1, "2026-08-15", "regular") == set()

    df = pd.concat([_canon_bar("005930"), _canon_bar("000660")], ignore_index=True)
    backfill_minute_history._merge_and_write_partition(df, 1, "2026-08-15", "regular")

    assert backfill_minute_history._already_collected_codes(1, "2026-08-15", "regular") == {"005930", "000660"}


def test_already_collected_codes_returns_empty_on_missing_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbol/종목코드 둘 다 없는 파일은 컬럼 조회 예외로 빈 집합을 반환한다 (line 97-99)."""
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(backfill_minute_history, "intraday_partition_path", intraday_store.intraday_partition_path)

    target = intraday_store.intraday_partition_path(1, "2026-08-16", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"foo": [1, 2]}).to_parquet(target, index=False)

    assert backfill_minute_history._already_collected_codes(1, "2026-08-16", "regular") == set()


def test_already_collected_codes_returns_empty_when_columns_present_but_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbol/종목코드 컬럼은 존재하나 0행인 파일은 루프를 모두 소진하고 빈 집합을 반환한다 (line 102)."""
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(backfill_minute_history, "intraday_partition_path", intraday_store.intraday_partition_path)

    target = intraday_store.intraday_partition_path(1, "2026-08-17", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": pd.Series(dtype="object"), "종목코드": pd.Series(dtype="object")}).to_parquet(
        target, index=False
    )

    assert backfill_minute_history._already_collected_codes(1, "2026-08-17", "regular") == set()


def test_run_minute_history_backfill_skips_already_collected_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backfill_minute_history.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(
        backfill_minute_history,
        "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365: [("2026-08-15", "005930"), ("2026-08-15", "000660")],
    )

    # 005930은 정규세션에 이미 저장돼 있다고 가정 -- 이번 호출에서 재조회되면 안 된다.
    pre_existing = _canon_bar("005930")
    backfill_minute_history.write_intraday_partition(pre_existing, 1, "2026-08-15", "regular")

    requested_codes: list[list[str]] = []

    async def _fake_regular(client, session, stock_codes, snapshot_date, bar_interval_minutes=1):
        requested_codes.append(list(stock_codes))
        frames = [_canon_bar(code, snapshot_date) for code in stock_codes]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    async def _fake_nxt(client, session, stock_codes, snapshot_date, bar_interval_minutes=1):
        return pd.DataFrame()

    from unittest.mock import AsyncMock, patch

    with (
        patch.object(backfill_minute_history, "backfill_regular_bars", _fake_regular),
        patch.object(backfill_minute_history, "backfill_nxt_aftermarket_bars", _fake_nxt),
        patch.object(backfill_minute_history.KisApiClient, "create_session") as mock_create_session,
        patch.object(backfill_minute_history.KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_create_session.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_create_session.return_value.__aexit__ = AsyncMock(return_value=False)

        backfill_minute_history.run_minute_history_backfill()

    assert requested_codes == [["000660"]]

def test_enumerate_backfill_targets_adds_next_trading_day_from_calendar(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.intraday import backfill_minute_history as mod

    # Arrange: Friday 2026-03-06 -> Monday 2026-03-09; 2026-03-09 is the calendar end.
    frames = [pd.DataFrame({"스냅샷_날짜": ["2026-03-06", "2026-03-09"], "종목코드": ["005930", "000660"]})]
    monkeypatch.setattr(mod, "_load_condition_history_sources", lambda: frames)
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-03-05", "2026-03-06", "2026-03-09"]))

    # Act
    targets = mod.enumerate_backfill_targets(
        as_of="2026-03-10", lookback_days=365, include_exit_day=True, trading_calendar=calendar
    )

    # Assert
    assert ("2026-03-06", "005930") in targets
    assert ("2026-03-09", "005930") in targets
    assert ("2026-03-09", "000660") in targets
    assert all(code != "000660" or date <= "2026-03-09" for date, code in targets)
    assert len(targets) == len(set(targets))
    assert targets == sorted(targets)

    without = mod.enumerate_backfill_targets(
        as_of="2026-03-10", lookback_days=365, include_exit_day=False, trading_calendar=calendar
    )
    assert ("2026-03-09", "005930") not in without

def test_enumerate_backfill_targets_resolves_exit_day_from_price_history_file(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.intraday import backfill_minute_history as mod

    # Arrange: calendar lives in the price-history parquet, not injected.
    frames = [pd.DataFrame({"스냅샷_날짜": ["2026-03-06"], "종목코드": ["005930"]})]
    monkeypatch.setattr(mod, "_load_condition_history_sources", lambda: frames)
    cal_path = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-03-05", "2026-03-06", "2026-03-09"])}).to_parquet(cal_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", cal_path)

    # Act
    targets = mod.enumerate_backfill_targets(as_of="2026-03-10", lookback_days=365)

    # Assert
    assert ("2026-03-06", "005930") in targets
    assert ("2026-03-09", "005930") in targets


def test_enumerate_backfill_targets_skips_exit_when_calendar_missing(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.intraday import backfill_minute_history as mod

    # Arrange: no calendar file and no injected calendar.
    frames = [pd.DataFrame({"스냅샷_날짜": ["2026-03-06"], "종목코드": ["005930"]})]
    monkeypatch.setattr(mod, "_load_condition_history_sources", lambda: frames)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "missing.parquet")

    # Act
    targets = mod.enumerate_backfill_targets(as_of="2026-03-10", lookback_days=365)

    # Assert: entry target survives, exit target is skipped and counted, never guessed.
    assert targets == [("2026-03-06", "005930")]


def test_enumerate_backfill_targets_supports_tz_aware_calendar(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.intraday import backfill_minute_history as mod

    # Arrange
    frames = [pd.DataFrame({"스냅샷_날짜": ["2026-03-06"], "종목코드": ["005930"]})]
    monkeypatch.setattr(mod, "_load_condition_history_sources", lambda: frames)
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-03-06", "2026-03-09"]).tz_localize("Asia/Seoul"))

    # Act
    targets = mod.enumerate_backfill_targets(
        as_of="2026-03-10", lookback_days=365, trading_calendar=calendar
    )

    # Assert
    assert ("2026-03-09", "005930") in targets


def test_enumerate_backfill_targets_skips_unparseable_snapshot_date(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.intraday import backfill_minute_history as mod

    # Arrange: "2026-13-40" passes the string window filter but has no calendar successor.
    frames = [pd.DataFrame({"스냅샷_날짜": ["2026-13-40", "2026-03-06"], "종목코드": ["005930", "000660"]})]
    monkeypatch.setattr(mod, "_load_condition_history_sources", lambda: frames)
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-03-05", "2026-03-06", "2026-03-09"]))

    # Act
    targets = mod.enumerate_backfill_targets(
        as_of="2027-01-01", lookback_days=365, trading_calendar=calendar
    )

    # Assert: the entry survives but no exit date is guessed from it.
    assert ("2026-03-09", "000660") in targets
    assert ("2026-13-40", "005930") in targets
    assert [t for t in targets if t[1] == "005930"] == [("2026-13-40", "005930")]

def test_enumerate_backfill_targets_treats_calendar_without_date_column_as_missing(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.intraday import backfill_minute_history as mod

    # Arrange: calendar parquet exists but holds no dates.
    frames = [pd.DataFrame({"스냅샷_날짜": ["2026-03-06"], "종목코드": ["005930"]})]
    monkeypatch.setattr(mod, "_load_condition_history_sources", lambda: frames)
    cal_path = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}).to_parquet(cal_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", cal_path)

    # Act
    targets = mod.enumerate_backfill_targets(as_of="2026-03-10", lookback_days=365)

    # Assert
    assert targets == [("2026-03-06", "005930")]
