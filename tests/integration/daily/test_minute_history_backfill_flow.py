from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.backfill.intraday import backfill_minute_history


def test_run_minute_history_backfill_processes_dates_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backfill_minute_history.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(
        backfill_minute_history,
        "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365: [("2026-08-01", "005930"), ("2026-08-15", "005930")],
    )

    call_order: list[str] = []

    async def _fake_regular(client, session, stock_codes, snapshot_date, bar_interval_minutes=1):
        call_order.append(snapshot_date)
        return pd.DataFrame({"종목코드": stock_codes, "스냅샷_날짜": [snapshot_date] * len(stock_codes)})

    async def _fake_nxt(client, session, stock_codes, snapshot_date, bar_interval_minutes=1):
        return pd.DataFrame()

    with (
        patch.object(backfill_minute_history, "backfill_regular_bars", _fake_regular),
        patch.object(backfill_minute_history, "backfill_nxt_aftermarket_bars", _fake_nxt),
        patch.object(backfill_minute_history.KisApiClient, "create_session") as mock_create_session,
        patch.object(backfill_minute_history.KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_create_session.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_create_session.return_value.__aexit__ = AsyncMock(return_value=False)

        result = backfill_minute_history.run_minute_history_backfill()

    assert call_order == ["2026-08-01", "2026-08-15"]
    assert result["dates"] == 2
    assert result["regular_rows"] == 2
