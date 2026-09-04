from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.daily import archive_intraday


def test_run_intraday_archive_writes_both_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_intraday.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(
        archive_intraday, "_today_watchlist_codes", lambda snapshot_date: ["005930"]
    )

    bars_df = pd.DataFrame({"종목코드": ["005930"], "스냅샷_날짜": ["2026-09-03"], "stck_prpr": [70000]})
    nxt_df = pd.DataFrame({"종목코드": ["005930"], "스냅샷_날짜": ["2026-09-03"], "stck_prpr": [70500]})
    ticks_df = pd.DataFrame({"종목코드": ["005930"], "스냅샷_날짜": ["2026-09-03"], "acml_vol": [1000]})

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
