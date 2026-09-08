from __future__ import annotations

import pytest


def test_map_ranking_rows_to_archive_frame_maps_confirmed_fields_and_tags_scenario() -> None:
    import pandas as pd

    from src.daily.universe_scan import UNIVERSE_SCAN_SCENARIO_TAG, map_ranking_rows_to_archive_frame

    rows = [
        {
            "stck_shrn_iscd": "5930",
            "hts_kor_isnm": "삼성전자",
            "stck_prpr": "70000",
            "stck_sdpr": "66500",
            "prdy_ctrt": "5.26",
            "acml_vol": "1234567",
        }
    ]
    ts = pd.Timestamp("2026-09-07 15:18:00")

    df = map_ranking_rows_to_archive_frame(rows, "2026-09-07", ts)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["종목코드"] == "005930"
    assert row["종목명"] == "삼성전자"
    assert row["종가"] == pytest.approx(70000.0)
    assert row["전일종가"] == pytest.approx(66500.0)
    assert row["등락률"] == pytest.approx(5.26)
    assert row["거래량"] == pytest.approx(1234567.0)
    assert row["스냅샷_날짜"] == "2026-09-07"
    assert row["시나리오"] == UNIVERSE_SCAN_SCENARIO_TAG
    assert row["snapshot_timestamp"] == ts


def test_map_ranking_rows_to_archive_frame_never_fabricates_missing_market_cap() -> None:
    import math

    import pandas as pd

    from src.daily.universe_scan import map_ranking_rows_to_archive_frame

    rows = [{"stck_shrn_iscd": "000660", "prdy_ctrt": "3.1"}]
    df = map_ranking_rows_to_archive_frame(rows, "2026-09-07", pd.Timestamp("2026-09-07 15:18:00"))

    assert math.isnan(df.iloc[0]["시가총액"])


def test_map_ranking_rows_to_archive_frame_empty_input_returns_typed_empty_frame() -> None:
    import pandas as pd

    from src.daily.universe_scan import map_ranking_rows_to_archive_frame

    df = map_ranking_rows_to_archive_frame([], "2026-09-07", pd.Timestamp("2026-09-07 15:18:00"))

    assert len(df) == 0
    assert set(df.columns) == {
        "스냅샷_날짜", "종목코드", "종목명", "종가", "전일종가", "등락률",
        "거래량", "거래대금", "시가총액", "시나리오", "snapshot_timestamp",
    }


def test_collect_universe_scan_uses_default_universe_bounds_as_percent() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import pandas as pd

    from src.daily.universe_scan import collect_universe_scan

    client = AsyncMock()
    client.get_fluctuation_ranking = AsyncMock(return_value={"rt_cd": "0", "output": []})
    ts = pd.Timestamp("2026-09-07 15:18:00")

    df = asyncio.run(collect_universe_scan(client, object(), "2026-09-07", ts))

    client.get_fluctuation_ranking.assert_awaited_once()
    kwargs = client.get_fluctuation_ranking.await_args.kwargs
    assert kwargs["rate_min_pct"] == pytest.approx(2.0)
    assert kwargs["rate_max_pct"] == pytest.approx(10.0)
    assert len(df) == 0


def test_collect_universe_scan_fails_soft_on_error_response() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import pandas as pd

    from src.daily.universe_scan import collect_universe_scan

    client = AsyncMock()
    client.get_fluctuation_ranking = AsyncMock(return_value={"rt_cd": "1", "msg1": "실패"})
    ts = pd.Timestamp("2026-09-07 15:18:00")

    df = asyncio.run(collect_universe_scan(client, object(), "2026-09-07", ts))

    assert len(df) == 0
    assert "종목코드" in df.columns


def test_archive_universe_snapshot_calls_upsert_directly_without_csv(monkeypatch) -> None:
    import pandas as pd

    from src.daily import universe_scan

    captured = {}

    def fake_upsert(df, snapshot_date=None):
        captured["df"] = df
        captured["snapshot_date"] = snapshot_date
        return len(df)

    monkeypatch.setattr(universe_scan.archive, "upsert_archive_snapshot", fake_upsert)
    df = pd.DataFrame({"종목코드": ["005930"]})

    result = universe_scan.archive_universe_snapshot(df, "2026-09-07")

    assert result == 1
    assert captured["snapshot_date"] == "2026-09-07"
    assert captured["df"] is df


def test_map_kiwoom_ranking_rows_to_archive_frame() -> None:
    import math
    import pandas as pd
    from src.daily.universe_scan import UNIVERSE_SCAN_SCENARIO_TAG, map_kiwoom_ranking_rows_to_archive_frame

    rows = [
        {
            "stk_cd": "004490_AL",
            "stk_nm": "세방전지",
            "cur_prc": "+69700",
            "pred_pre": "+5400",
            "flu_rt": "+8.40",
            "now_trde_qty": "143616",
        }
    ]
    ts = pd.Timestamp("2026-09-07 15:18:00")
    df = map_kiwoom_ranking_rows_to_archive_frame(rows, "2026-09-07", ts)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["종목코드"] == "004490"
    assert row["종목명"] == "세방전지"
    assert row["종가"] == 69700.0
    assert row["전일종가"] == 64300.0
    assert row["등락률"] == 8.40
    assert row["거래량"] == 143616.0
    assert row["거래대금"] == round(69700.0 * 143616.0 / 1e8, 4)
    assert math.isnan(row["시가총액"])
    assert row["시나리오"] == UNIVERSE_SCAN_SCENARIO_TAG
    assert row["snapshot_timestamp"] == ts


def test_collect_universe_scan_prefers_kiwoom() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    import pandas as pd
    from src.daily.universe_scan import collect_universe_scan

    kw_client = AsyncMock()
    kw_client.get_fluctuation_ranking.return_value = {
        "rt_cd": "0",
        "output": [{"stk_cd": "004490_AL", "stk_nm": "세방전지", "cur_prc": "+69700", "pred_pre": "+5400", "flu_rt": "+8.40", "now_trde_qty": "143616"}],
        "vendor": "kiwoom",
    }
    kis_client = AsyncMock()
    ts = pd.Timestamp("2026-09-07 15:18:00")

    df = asyncio.run(collect_universe_scan(kis_client, object(), "2026-09-07", ts, kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["종목코드"] == "004490"
    assert kis_client.get_fluctuation_ranking.call_count == 0
    assert kw_client.get_fluctuation_ranking.call_count == 1


def test_collect_universe_scan_falls_back_to_kis_when_kiwoom_fails() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    import pandas as pd
    from src.daily.universe_scan import collect_universe_scan

    kw_client = AsyncMock()
    kw_client.get_fluctuation_ranking.return_value = {"rt_cd": "1", "output": []}
    kis_client = AsyncMock()
    kis_client.get_fluctuation_ranking.return_value = {
        "rt_cd": "0",
        "output": [{"stck_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자", "stck_prpr": "70000", "stck_sdpr": "66500", "prdy_ctrt": "5.26", "acml_vol": "1000"}],
    }
    ts = pd.Timestamp("2026-09-07 15:18:00")

    df = asyncio.run(collect_universe_scan(kis_client, object(), "2026-09-07", ts, kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["종목코드"] == "005930"
    assert kw_client.get_fluctuation_ranking.call_count == 1
    assert kis_client.get_fluctuation_ranking.call_count == 1


def test_collect_universe_scan_zero_regression_when_kiwoom_none() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    import pandas as pd
    from src.daily.universe_scan import collect_universe_scan

    kis_client = AsyncMock()
    kis_client.get_fluctuation_ranking.return_value = {
        "rt_cd": "0",
        "output": [{"stck_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자", "stck_prpr": "70000", "stck_sdpr": "66500", "prdy_ctrt": "5.26", "acml_vol": "1000"}],
    }
    ts = pd.Timestamp("2026-09-07 15:18:00")

    df = asyncio.run(collect_universe_scan(kis_client, object(), "2026-09-07", ts, kiwoom_client=None))

    assert len(df) == 1
    assert df.iloc[0]["종목코드"] == "005930"
    assert kis_client.get_fluctuation_ranking.call_count == 1


def test_map_kiwoom_ranking_rows_empty_input_returns_typed_empty_frame() -> None:
    import pandas as pd

    from src.daily.universe_scan import map_kiwoom_ranking_rows_to_archive_frame

    df = map_kiwoom_ranking_rows_to_archive_frame([], "2026-09-07", pd.Timestamp("2026-09-07 15:18:00"))

    assert len(df) == 0
    assert set(df.columns) == {
        "스냅샷_날짜", "종목코드", "종목명", "종가", "전일종가", "등락률",
        "거래량", "거래대금", "시가총액", "시나리오", "snapshot_timestamp",
    }


def test_collect_universe_scan_falls_back_to_kis_when_kiwoom_raises() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    import pandas as pd
    from src.daily.universe_scan import collect_universe_scan

    kw_client = AsyncMock()
    kw_client.get_fluctuation_ranking.side_effect = RuntimeError("kiwoom unreachable")
    kis_client = AsyncMock()
    kis_client.get_fluctuation_ranking.return_value = {
        "rt_cd": "0",
        "output": [{"stck_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자", "stck_prpr": "70000", "stck_sdpr": "66500", "prdy_ctrt": "5.26", "acml_vol": "1000"}],
    }
    ts = pd.Timestamp("2026-09-07 15:18:00")

    df = asyncio.run(collect_universe_scan(kis_client, object(), "2026-09-07", ts, kiwoom_client=kw_client))

    assert len(df) == 1
    assert df.iloc[0]["종목코드"] == "005930"
    assert kw_client.get_fluctuation_ranking.call_count == 1
    assert kis_client.get_fluctuation_ranking.call_count == 1
