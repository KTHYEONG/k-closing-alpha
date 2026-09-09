"""Unit tests for daily data collection.

SCENARIO_DAILY_COLLECT_REFACTORING_01:
Verifies that collect.py saves collected condition data directly
without chart_pass_cache.json or parenthesis column renaming.

SCENARIO_COLLECT_NO_SLEEP:
 fetch_single_stock 내부에 API_SLEEP_INTERVAL 참조 없음 검증 (perf_v2).

SCENARIO_REGRESSION:
 기존 17개 테스트 회귀 검증.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd



















def test_fetch_all_stock_data_persists_orderbook_and_survives_persist_failure(monkeypatch) -> None:
    """호가 스냅샷을 일괄 영속화하고, 영속화 실패는 로깅만 하고 수집 결과에 영향 없다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(
        return_value={"rt_cd": "0", "output": {"stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "70500", "stck_lwpr": "68900", "acml_vol": "1000", "prdy_ctrt": "1.5", "lstn_stcn": "100", "hts_avls": "1000", "acml_tr_pbmn": "100000000", "rprs_mrkt_kor_name": "KOSPI"}}
    )
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{"whol_smtn_ntby_tr_pbmn": "100"}]})
    ladder = {f"askp{i}": str(70000 + i * 100) for i in range(1, 11)}
    ladder.update({"bidp1": "69900", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})

    def _raise(rows, snapshot_date):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(collect, "append_orderbook_snapshots", _raise)

    stock_list = [{"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "1.5"}]
    results, failed_info = asyncio.run(collect.fetch_all_stock_data(stock_list, client, object()))

    assert len(results) == 1
    assert failed_info == []


















def test_flag_cost_aware_admission_marks_rows_without_dropping() -> None:
    import pandas as pd

    from src.daily.collect import flag_cost_aware_admission

    # Given: the same 5-candidate snapshot the filtering variant is specified on
    # (S1 admitted; S2 tick-cost, S3 chg-band, S4 liquidity, S5 ceiling excluded)
    df = pd.DataFrame({
        "종목코드": ["S1", "S2", "S3", "S4", "S5"],
        "종가": [18000.0, 30000.0, 23000.0, 18000.0, 13000.0],
        "전일종가": [17142.86, 28571.43, 20000.0, 17142.86, 10000.0],
        "고가": [18100.0, 30100.0, 23100.0, 18100.0, 13000.0],
        "거래량": [1_000_000] * 5,
        "거래대금": [500.0, 500.0, 500.0, 10.0, 500.0],
        "시가총액": [3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
        "시장구분": ["KOSPI"] * 5,
    })

    # When
    out = flag_cost_aware_admission(df, decision_date=pd.Timestamp("2026-09-09"))

    # Then: every row survives, carrying the admission verdict as a flag
    assert len(out) == 5
    assert out["admitted"].tolist() == [True, False, False, False, False]
    assert out["종목코드"].tolist() == ["S1", "S2", "S3", "S4", "S5"]






def test_fetch_single_stock_calls_only_the_four_required_apis() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    # Given: a client exposing every legacy endpoint, so the test proves the
    # dropped ones are not merely unavailable but deliberately not called
    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{"whol_smtn_ntby_tr_pbmn": "100"}]})
    ladder = {"askp1": "18010", "bidp1": "17990", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"}
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})

    # When
    sem = asyncio.Semaphore(1)
    asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}, 1, sem, client, object()
        )
    )

    # Then: KRX current price once (no NXT twin), both orderbook venues kept
    assert client.get_current_price.await_count == 1
    assert client.get_orderbook_snapshot.await_count == 2
    assert client.get_investor_trend_estimate.await_count == 1
    client.get_trade_strength.assert_not_awaited()
    client.get_program_net_buy.assert_not_awaited()



def test_fetch_single_stock_returns_minimal_row_schema() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    ladder = {"askp1": "18010", "bidp1": "17990", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"}
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})

    # When
    sem = asyncio.Semaphore(1)
    row, failed, orderbook_rows = asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}, 1, sem, client, object()
        )
    )

    # Then: exactly the reranker-required fields, nothing champion-era
    assert set(row) == {
        "종목명", "종목코드", "시장구분", "시가", "고가", "저가", "종가", "전일종가",
        "거래량", "거래대금", "시가총액", "기관_순매수", "외국인_순매수", "등락률",
    }
    assert failed == []
    # Then: the ladder partition source is still produced for the cost research store
    assert orderbook_rows
    assert {ob["capture_reason"] for ob in orderbook_rows} == {"decision"}



def test_fetch_single_stock_reports_failed_apis_for_kept_endpoints() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    # Given: every retained endpoint fails
    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "1", "msg1": "fail"})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "1"})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "1"})

    # When
    sem = asyncio.Semaphore(1)
    row, failed, orderbook_rows = asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}, 1, sem, client, object()
        )
    )

    # Then: failures are surfaced, not swallowed, and the row still returns
    assert set(failed) == {"현재가", "투자자추정", "호가"}
    assert row["종목코드"] == "005930"
    assert orderbook_rows == []



def test_fetch_all_stock_data_does_not_prefetch_sma120(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    client.get_orderbook_snapshot = AsyncMock(
        return_value={"rt_cd": "0", "output1": {"askp1": "18010", "bidp1": "17990"}}
    )

    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: None)

    # Then: the bulk OHLCV prefetch stage is gone entirely -- not merely
    # unreached, but structurally absent from the module
    assert not hasattr(collect, "prefetch_ohlcv_for_sma120")

    # When
    stock_list = [{"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}]
    results, failed_info = asyncio.run(collect.fetch_all_stock_data(stock_list, client, object()))

    assert len(results) == 1
    assert failed_info == []



def test_resolve_daily_candidates_never_touches_condition_search(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod

    # Given
    client = AsyncMock()
    scan_rows = [{"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}]

    async def _fake_scan(client_arg, session_arg, **kwargs):
        return scan_rows

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _fake_scan)

    # When
    out = asyncio.run(collect_mod.resolve_daily_candidates(client, object()))

    # Then: a plain list, and the manual HTS path is structurally gone
    assert out == scan_rows
    client.get_condition_list.assert_not_awaited()
    client.get_condition_result.assert_not_awaited()



def test_resolve_daily_candidates_returns_empty_list_when_scan_yields_nothing(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod

    # Given: the ranking scan finds nothing inside the 2-10% band today
    async def _empty_scan(client_arg, session_arg, **kwargs):
        rows: list[dict] = []
        return rows

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _empty_scan)

    # When
    out = asyncio.run(collect_mod.resolve_daily_candidates(AsyncMock(), object()))

    # Then: an empty list keeps the caller contract total (no None sentinel)
    assert isinstance(out, list)
    assert len(out) == 0



def test_persist_daily_snapshot_delegates_to_archive_upsert(monkeypatch) -> None:
    import pandas as pd

    import src.daily.collect as collect_mod

    # Given
    captured = {}

    def _fake_upsert(df, snapshot_date=None):
        captured["df"] = df
        captured["snapshot_date"] = snapshot_date
        return len(df)

    monkeypatch.setattr(collect_mod.archive, "upsert_archive_snapshot", _fake_upsert)
    df = pd.DataFrame({"종목코드": ["005930", "000660"], "admitted": [True, False]})

    # When
    stored = collect_mod.persist_daily_snapshot(df, "2026-09-09")

    # Then: single storage path, wide rows preserved (admitted=False kept)
    assert stored == 2
    assert captured["snapshot_date"] == "2026-09-09"
    assert captured["df"]["admitted"].tolist() == [True, False]



def test_persist_daily_snapshot_skips_upsert_on_empty_frame(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.collect as collect_mod

    upsert_mock = Mock()
    monkeypatch.setattr(collect_mod.archive, "upsert_archive_snapshot", upsert_mock)

    # When
    stored = collect_mod.persist_daily_snapshot(pd.DataFrame(), "2026-09-09")

    # Then
    assert stored == 0
    upsert_mock.assert_not_called()
