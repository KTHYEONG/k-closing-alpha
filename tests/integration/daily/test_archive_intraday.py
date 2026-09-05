from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.daily import archive_intraday
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame


def _bars(symbol: str, snapshot_date: str, close: int) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "time": ["090300"],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "jdiff_vol": [1000],
            "value": [70],
        }
    )
    return normalize_bar_frame(raw, "ls", snapshot_date, symbol)


def _ticks(symbol: str, snapshot_date: str) -> pd.DataFrame:
    raw = pd.DataFrame({"time": ["090300"], "close": [70000], "jdiff_vol": [1000]})
    return normalize_tick_frame(raw, "ls", snapshot_date, symbol)


def test_run_intraday_archive_writes_both_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_intraday.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(
        archive_intraday, "_archive_target_codes", lambda snapshot_date: ["005930"]
    )

    bars_df = _bars("005930", "2026-09-03", 70000)
    nxt_df = _bars("005930", "2026-09-03", 70500)
    ticks_df = _ticks("005930", "2026-09-03")

    with (
        patch.object(archive_intraday, "collect_intraday_bars", AsyncMock(return_value=bars_df)),
        patch.object(archive_intraday, "collect_nxt_aftermarket_bars", AsyncMock(return_value=nxt_df)),
        patch.object(archive_intraday, "collect_intraday_trade_ticks", AsyncMock(return_value=ticks_df)),
        patch.object(archive_intraday.KisApiClient, "create_session") as mock_create_session,
        patch.object(archive_intraday.KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_create_session.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_create_session.return_value.__aexit__ = AsyncMock(return_value=False)

        bars_rows, nxt_rows, tick_rows = archive_intraday.run_intraday_archive(snapshot_date="2026-09-03", bar_interval_minutes=1)

    assert bars_rows == 1
    assert nxt_rows == 1
    assert tick_rows == 1
    assert (tmp_path / "intraday" / "1m" / "regular" / "2026-09" / "2026-09-03.parquet").exists()
    assert (tmp_path / "intraday" / "1m" / "nxt_aftermarket" / "2026-09" / "2026-09-03.parquet").exists()
    assert (tmp_path / "intraday" / "ticks" / "regular" / "2026-09" / "2026-09-03.parquet").exists()


def test_run_intraday_archive_stages_bars_before_ticks(tmp_path) -> None:
    from pathlib import Path
    from unittest.mock import AsyncMock, patch
    import pandas as pd
    from src.daily import archive_intraday
    execution_order = []
    async def fake_bars(*args, **kwargs):
        execution_order.append("bars")
        return _bars("005930", "2026-09-04", 70000)
    async def fake_nxt(*args, **kwargs):
        execution_order.append("nxt")
        return _bars("005930", "2026-09-04", 70500)
    async def fake_ticks(*args, **kwargs):
        execution_order.append("ticks")
        return _ticks("005930", "2026-09-04")
    with (
        patch.object(archive_intraday.settings, "HISTORY_DIR", tmp_path),
        patch.object(archive_intraday, "_archive_target_codes", lambda d: ["005930"]),
        patch.object(archive_intraday, "collect_intraday_bars", fake_bars),
        patch.object(archive_intraday, "collect_nxt_aftermarket_bars", fake_nxt),
        patch.object(archive_intraday, "collect_intraday_trade_ticks", fake_ticks),
        patch.object(archive_intraday.KisApiClient, "create_session") as mock_sess,
        patch.object(archive_intraday.KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        bars_n, nxt_n, ticks_n = archive_intraday.run_intraday_archive("2026-09-04")
    assert bars_n == 1
    assert nxt_n == 1
    assert ticks_n == 1
    assert execution_order.index("bars") < execution_order.index("ticks")
    assert execution_order.index("nxt") < execution_order.index("ticks")
