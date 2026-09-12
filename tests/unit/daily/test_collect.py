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
import pytest



















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






def test_fetch_single_stock_calls_only_the_three_required_apis() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    # Given: a client exposing every legacy endpoint, so the test proves the
    # dropped ones (including NXT orderbook) are not merely unavailable but
    # deliberately not called
    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{"whol_smtn_ntby_tr_pbmn": "100"}]})
    ladder = {"askp1": "18010", "bidp1": "17990", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"}
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder, "output2": {"antc_cnpr": "18020"}})

    # When
    sem = asyncio.Semaphore(1)
    asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}, 1, sem, client, object()
        )
    )

    # Then: KRX current price once, KRX orderbook once (NXT twin removed)
    assert client.get_current_price.await_count == 1
    assert client.get_orderbook_snapshot.await_count == 1
    assert client.get_investor_trend_estimate.await_count == 1
    client.get_trade_strength.assert_not_awaited()
    client.get_program_net_buy.assert_not_awaited()

    call_kwargs = client.get_orderbook_snapshot.call_args.kwargs
    assert call_kwargs.get("market_div_code") == "J"



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

    # Then: exactly the reranker-required fields plus the new failure flag
    assert set(row) == {
        "종목명", "종목코드", "시장구분", "시가", "고가", "저가", "종가", "전일종가",
        "거래량", "거래대금", "시가총액", "기관_순매수", "외국인_순매수", "등락률", "수급_실패",
        "현재가_실패", "결정_종가", "종가_확정",
    }
    assert row["수급_실패"] is False
    assert row["현재가_실패"] is False
    assert failed == []
    # Then: a single KRX-venue partition row is still produced for the cost research store
    assert len(orderbook_rows) == 1
    assert {ob["capture_reason"] for ob in orderbook_rows} == {"decision"}
    assert {ob["venue"] for ob in orderbook_rows} == {"J"}



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
    assert row["수급_실패"] is True
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


def test_parse_market_index_rate_returns_none_on_failure() -> None:
    from src.daily.collect import parse_market_index_rate

    # Given/When/Then: failure cases return None, not 0.0
    assert parse_market_index_rate(None) is None
    assert parse_market_index_rate({"rt_cd": "1"}) is None
    assert parse_market_index_rate({"rt_cd": "0"}) is None
    assert parse_market_index_rate({"rt_cd": "0", "output1": {}}) is None

    # A genuine non-zero rate still parses normally
    ok = parse_market_index_rate({"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "1.23"}})
    assert ok == pytest.approx(1.23)

    # A genuine zero computed from price/change (not the failure fallback) still returns 0.0
    zero = parse_market_index_rate(
        {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "0.00", "bstp_nmix_prpr": "2500.0", "bstp_nmix_prdy_vrss": "0.0"}}
    )
    assert zero == pytest.approx(0.0)


def test_validate_decision_window_raises_outside_window_and_force_bypasses() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.daily.collect import _validate_decision_window

    kst = ZoneInfo("Asia/Seoul")

    # Outside the window: raises
    with pytest.raises(RuntimeError, match="결정 창"):
        _validate_decision_window(datetime(2026, 9, 10, 19, 10, 0, tzinfo=kst))

    # Inside the window: passes silently
    _validate_decision_window(datetime(2026, 9, 10, 15, 25, 0, tzinfo=kst))
    _validate_decision_window(datetime(2026, 9, 10, 15, 20, 0, tzinfo=kst))
    _validate_decision_window(datetime(2026, 9, 10, 15, 30, 0, tzinfo=kst))

    # force=True bypasses regardless of time
    _validate_decision_window(datetime(2026, 9, 10, 19, 10, 0, tzinfo=kst), force=True)


def test_fetch_single_stock_marks_supply_flow_failure_and_nans_the_fields() -> None:
    import asyncio
    import math
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "1", "msg1": "fail"})
    ladder = {"askp1": "18010", "bidp1": "17990"}
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})

    sem = asyncio.Semaphore(1)
    row, failed, _orderbook_rows = asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}, 1, sem, client, object()
        )
    )

    assert row["수급_실패"] is True
    assert math.isnan(row["기관_순매수"])
    assert math.isnan(row["외국인_순매수"])
    assert "투자자추정" in failed
    assert row["종목코드"] == "005930"


def test_main_raises_outside_decision_window_without_force(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock
    from zoneinfo import ZoneInfo

    import pytest

    from src.daily import collect

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 10, 19, 10, 0, tzinfo=tz)

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(collect, "_validate_hts_id", lambda: None)
    never_called = AsyncMock()
    monkeypatch.setattr(collect, "resolve_daily_candidates", never_called)

    with pytest.raises(RuntimeError, match="결정 창"):
        asyncio.run(collect.main(force=False))

    never_called.assert_not_awaited()


def test_parse_market_index_rate_fallthrough_returns_none() -> None:
    from src.daily.collect import parse_market_index_rate

    # rate_str reads exactly zero and price/change are also both zero -> prev_close == 0,
    # cannot be recomputed -> falls through to the terminal `return None`
    unresolvable = parse_market_index_rate(
        {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "0.00", "bstp_nmix_prpr": "0", "bstp_nmix_prdy_vrss": "0"}}
    )
    assert unresolvable is None

    # a non-numeric field raises inside the try block -> except: pass -> same terminal None
    malformed = parse_market_index_rate(
        {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "0.00", "bstp_nmix_prpr": "not-a-number", "bstp_nmix_prdy_vrss": "0"}}
    )
    assert malformed is None
def test_fetch_single_stock_records_decision_close_and_unconfirmed_flag() -> None:
    import asyncio
    from unittest.mock import AsyncMock


    from src.daily import collect
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    client = AsyncMock()
    client.get_current_price = AsyncMock(
        return_value={
            "rt_cd": "0",
            "output": {
                "stck_prpr": "269250",
                "stck_oprc": "270000",
                "stck_hgpr": "272000",
                "stck_lwpr": "268000",
                "acml_vol": "19525671",
                "prdy_ctrt": "-0.09",
                "lstn_stcn": "5969782550",
                "hts_avls": "1600000",
                "acml_tr_pbmn": "5220657837500",
                "rprs_mrkt_kor_name": "KOSPI",
            },
        }
    )
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": {"askp1": "269500", "bidp1": "269250"}})

    async def _run():
        return await collect.fetch_single_stock(
            0,
            {"code": "005930", "name": "삼성전자", "price": "269250", "chgrate": "-0.09"},
            1,
            asyncio.Semaphore(1),
            client,
            object(),
        )

    row, failed_apis, _orderbook_rows = asyncio.run(_run())

    assert failed_apis == []
    assert row["종가"] == 269250
    assert row[DECISION_CLOSE_COL] == 269250
    assert row[CLOSE_CONFIRMED_COL] is False




def test_fetch_single_stock_takes_prev_close_from_vendor_field() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    base_detail = {
        "stck_prpr": "1100",
        "stck_oprc": "1010",
        "stck_hgpr": "1120",
        "stck_lwpr": "1000",
        "acml_vol": "5000",
        "prdy_ctrt": "10.00",
        "lstn_stcn": "1000000",
        "hts_avls": "1000",
        "acml_tr_pbmn": "5500000",
        "rprs_mrkt_kor_name": "KOSDAQ",
    }

    def _client(detail):
        c = AsyncMock()
        c.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": detail})
        c.get_investor_trend_estimate = AsyncMock(
            return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
        )
        c.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": {}})
        return c

    stock = {"code": "005930", "name": "테스트", "price": "1100", "chgrate": "10.00"}
    sem = asyncio.Semaphore(1)

    # Given: 벤더가 전일종가(stck_sdpr)를 제공
    row, failed, _ob = asyncio.run(
        collect.fetch_single_stock(0, stock, 1, sem, _client({**base_detail, "stck_sdpr": "1000"}), object())
    )

    # Then: 역산(int(1100/1.10)=999)이 아니라 원본 1000을 그대로 쓴다
    assert row["전일종가"] == 1000
    assert failed == []

    # And: 필드가 없을 때만 등락률 역산으로 폴백
    row2, _f2, _o2 = asyncio.run(
        collect.fetch_single_stock(0, stock, 1, sem, _client(dict(base_detail)), object())
    )
    assert row2["전일종가"] == int(1100 / 1.10)


def test_fetch_single_stock_flags_quote_failure_without_zero_fill() -> None:
    import asyncio
    import math
    from unittest.mock import AsyncMock

    from src.daily import collect
    from src.processing.schema import ARCHIVE_COLUMN_ORDER, QUOTE_FAILED_COL

    client = AsyncMock()
    # Given: 현재가 TR 이 실패
    client.get_current_price = AsyncMock(return_value={"rt_cd": "9", "msg1": "네트워크 연결 실패"})
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": {}})

    stock = {"code": "005930", "name": "테스트", "price": "1100", "chgrate": "10.00"}

    # When
    row, failed, _ob = asyncio.run(
        collect.fetch_single_stock(0, stock, 1, asyncio.Semaphore(1), client, object())
    )

    # Then: 실패가 행에 표식되고 0으로 위조되지 않는다
    assert row[QUOTE_FAILED_COL] is True
    assert "현재가" in failed
    for col in ("시가", "고가", "저가", "거래량", "거래대금", "시가총액"):
        assert math.isnan(float(row[col])), f"{col} must be NaN, not a synthetic zero"

    # And: 플래그가 아카이브 스키마에 포함되어 왕복에서 살아남는다
    assert QUOTE_FAILED_COL in ARCHIVE_COLUMN_ORDER

    # And: 정상 응답에서는 플래그가 False
    client.get_current_price = AsyncMock(
        return_value={
            "rt_cd": "0",
            "output": {
                "stck_prpr": "1100", "stck_sdpr": "1000", "stck_oprc": "1010",
                "stck_hgpr": "1120", "stck_lwpr": "1000", "acml_vol": "5000",
                "prdy_ctrt": "10.00", "lstn_stcn": "1000000", "hts_avls": "1000",
                "acml_tr_pbmn": "5500000", "rprs_mrkt_kor_name": "KOSDAQ",
            },
        }
    )
    ok_row, ok_failed, _o = asyncio.run(
        collect.fetch_single_stock(0, stock, 1, asyncio.Semaphore(1), client, object())
    )
    assert ok_row[QUOTE_FAILED_COL] is False
    assert ok_failed == []


def test_validate_trading_day_blocks_non_trading_day_and_honours_force(monkeypatch) -> None:
    import asyncio

    import pytest

    from src.daily import collect

    calls = {"n": 0}

    def _oracle(result):
        async def _fn(_client, _session, _date):
            calls["n"] += 1
            return result

        return _fn

    # Given: 비거래일
    monkeypatch.setattr(collect, "is_kis_trading_day", _oracle(False))
    with pytest.raises(collect.NonTradingDayError):
        asyncio.run(collect._validate_trading_day(object(), object(), "2026-09-05"))
    assert calls["n"] == 1

    # And: force 는 오라클 호출 없이 통과 (운영 수동 우회 경로)
    asyncio.run(collect._validate_trading_day(object(), object(), "2026-09-05", force=True))
    assert calls["n"] == 1

    # And: 거래일은 통과
    monkeypatch.setattr(collect, "is_kis_trading_day", _oracle(True))
    asyncio.run(collect._validate_trading_day(object(), object(), "2026-09-10"))
    assert calls["n"] == 2



def test_fetch_single_stock_prev_close_equals_close_when_rate_is_zero() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(
        return_value={'rt_cd': '0', 'output': {'stck_prpr': '1100', 'stck_oprc': '1090', 'stck_hgpr': '1110', 'stck_lwpr': '1080', 'acml_vol': '100', 'prdy_ctrt': '0.00', 'lstn_stcn': '1000', 'hts_avls': '500', 'acml_tr_pbmn': '110000', 'rprs_mrkt_kor_name': 'KOSPI'}}
    )
    client.get_investor_trend_estimate = AsyncMock(
        return_value={'rt_cd': '0', 'output2': [{'frgn_fake_ntby_qty': '1', 'orgn_fake_ntby_qty': '2'}]}
    )
    client.get_orderbook_snapshot = AsyncMock(return_value={'rt_cd': '0', 'output1': {}})

    stock = {'code': '005930', 'name': '테스트', 'price': '1100', 'chgrate': '0.00'}
    row, failed, _ob = asyncio.run(
        collect.fetch_single_stock(0, stock, 1, asyncio.Semaphore(1), client, object())
    )

    assert row['전일종가'] == 1100
    assert failed == []


def test_build_toss_scan_client_returns_none_without_app_key(monkeypatch) -> None:
    import src.api.toss.client as toss_mod
    import src.daily.collect as collect_mod

    class _NoKeyToss:
        def __init__(self, *a, **k):
            self.app_key = ""

    monkeypatch.setattr(toss_mod, "TossApiClient", _NoKeyToss)

    assert collect_mod.build_toss_scan_client() is None


def test_build_toss_scan_client_returns_client_with_app_key(monkeypatch) -> None:
    import src.api.toss.client as toss_mod
    import src.daily.collect as collect_mod

    class _KeyedToss:
        def __init__(self, *a, **k):
            self.app_key = "real-key"

    monkeypatch.setattr(toss_mod, "TossApiClient", _KeyedToss)

    result = collect_mod.build_toss_scan_client()

    assert result is not None and result.app_key == "real-key"


def test_resolve_daily_candidates_passes_toss_client_through(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod

    seen_kwargs = {}

    async def _fake_scan(client_arg, session_arg, **kwargs):
        seen_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _fake_scan)
    sentinel_toss = object()

    asyncio.run(collect_mod.resolve_daily_candidates(AsyncMock(), object(), kiwoom_client=None, toss_client=sentinel_toss))

    assert seen_kwargs.get("toss_client") is sentinel_toss


def test_flag_price_anomaly_flags_zero_price_success_row() -> None:
    import pandas as pd

    from src.daily.collect import flag_price_anomaly

    # Given: a vendor 'success' row that is degenerate (all-zero OHLCV), the
    # exact pattern /probe found for inverse-leveraged ETN codes
    df = pd.DataFrame({
        "종가": [0.0],
        "고가": [0.0],
        "저가": [0.0],
        "거래량": [0.0],
    })

    # When
    out = flag_price_anomaly(df)

    # Then
    assert out.tolist() == [True]
    assert out.dtype == bool


def test_flag_price_anomaly_does_not_flag_nan_quote_failed_row() -> None:
    import pandas as pd

    from src.daily.collect import flag_price_anomaly

    # Given: the existing quote_failed convention (NaN OHLCV, '0 위조 금지')
    df = pd.DataFrame({
        "종가": [float("nan")],
        "고가": [float("nan")],
        "저가": [float("nan")],
        "거래량": [float("nan")],
    })

    # When
    out = flag_price_anomaly(df)

    # Then: NaN rows are the quote_failed path's responsibility, not this one's
    assert out.tolist() == [False]
    assert out.dtype == bool


def test_flag_price_anomaly_flags_inconsistent_ohlc_range() -> None:
    import pandas as pd

    from src.daily.collect import flag_price_anomaly

    # Given: close (100) is above high (90) -- internally inconsistent OHLC
    df = pd.DataFrame({
        "종가": [100.0],
        "고가": [90.0],
        "저가": [80.0],
        "거래량": [1_000.0],
    })

    # When
    out = flag_price_anomaly(df)

    # Then
    assert out.tolist() == [True]


def test_flag_price_anomaly_leaves_healthy_row_unflagged() -> None:
    import pandas as pd

    from src.daily.collect import flag_price_anomaly

    # Given: a healthy row (mirrors S1 from test_flag_cost_aware_admission_marks_rows_without_dropping)
    df = pd.DataFrame({
        "종가": [18000.0],
        "고가": [18100.0],
        "저가": [17800.0],
        "거래량": [1_000_000.0],
    })

    # When
    out = flag_price_anomaly(df)

    # Then
    assert out.tolist() == [False]


def test_check_realtime_collection_coverage_returns_report_when_within_threshold() -> None:
    import pandas as pd
    import pytest

    from src.daily.collect import check_realtime_collection_coverage

    # Given: 1 degraded row out of 200 (0.5% degraded, well within a 10% test threshold)
    df = pd.DataFrame({
        "현재가_실패": [True] + [False] * 199,
        "가격_비정상": [False] * 200,
    })

    # When
    report = check_realtime_collection_coverage(df, min_coverage=0.9)

    # Then
    assert report["n_raw"] == 200
    assert report["n_degraded"] == 1
    assert report["coverage"] == pytest.approx(0.995, abs=1e-6)


def test_check_realtime_collection_coverage_raises_when_below_threshold() -> None:
    import pandas as pd
    import pytest

    from src.daily.collect import check_realtime_collection_coverage

    # Given: 1 degraded row out of 2 (50% degraded), using the production default threshold
    df = pd.DataFrame({
        "현재가_실패": [True, False],
        "가격_비정상": [False, False],
    })

    # When / Then
    with pytest.raises(ValueError, match="real-time collection coverage"):
        check_realtime_collection_coverage(df)


def test_check_realtime_collection_coverage_raises_on_empty_snapshot() -> None:
    import pandas as pd
    import pytest

    from src.daily.collect import check_realtime_collection_coverage

    # Given: an empty snapshot (should never happen in production -- main() returns
    # early on an empty scan -- but the utility must still fail closed, not divide by zero)
    df = pd.DataFrame({"현재가_실패": [], "가격_비정상": []})

    # When / Then
    with pytest.raises(ValueError, match="empty snapshot"):
        check_realtime_collection_coverage(df)
