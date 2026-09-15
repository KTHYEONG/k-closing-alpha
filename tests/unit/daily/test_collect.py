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
        return_value={"rt_cd": "0", "output": {"stck_shrn_iscd": "005930", "stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "70500", "stck_lwpr": "68900", "acml_vol": "1000", "prdy_ctrt": "1.5", "lstn_stcn": "100", "hts_avls": "1000", "acml_tr_pbmn": "100000000", "rprs_mrkt_kor_name": "KOSPI"}}
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
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_shrn_iscd": "005930", "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
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
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_shrn_iscd": "005930", "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
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
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_shrn_iscd": "005930", "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
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
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_shrn_iscd": "005930", "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
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
                "stck_shrn_iscd": "005930", "stck_prpr": "269250",
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
        "stck_shrn_iscd": "005930", "stck_prpr": "1100",
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
                "stck_shrn_iscd": "005930", "stck_prpr": "1100", "stck_sdpr": "1000", "stck_oprc": "1010",
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
        return_value={'rt_cd': '0', 'output': {'stck_shrn_iscd': '005930', 'stck_prpr': '1100', 'stck_oprc': '1090', 'stck_hgpr': '1110', 'stck_lwpr': '1080', 'acml_vol': '100', 'prdy_ctrt': '0.00', 'lstn_stcn': '1000', 'hts_avls': '500', 'acml_tr_pbmn': '110000', 'rprs_mrkt_kor_name': 'KOSPI'}}
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


def test_main_skips_cleanly_on_non_trading_day(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    from src.daily import collect

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 24, 15, 22, 0, tzinfo=tz)

    class _FakeKis:
        def __init__(self, *args, **kwargs):
            pass

        async def ensure_token(self, session):
            return None

        async def get_market_index_rate(self, session, code):
            raise AssertionError("holiday must not query market indices")

    async def _holiday(_client, _session, _date):
        return False

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(collect, "_validate_hts_id", lambda: None)
    monkeypatch.setattr(collect, "KisApiClient", _FakeKis)
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", lambda: None)
    monkeypatch.setattr(collect, "build_toss_scan_client", lambda: None)
    monkeypatch.setattr(collect, "is_kis_trading_day", _holiday)
    never_called = AsyncMock()
    monkeypatch.setattr(collect, "resolve_daily_candidates", never_called)

    # When: 추석 휴장일 결정창 안에서 실행
    result = asyncio.run(collect.main(force=False))

    # Then: 예외 없이 정상 종료, 후보 수집 미호출
    assert result is None
    never_called.assert_not_awaited()


def test_resolve_daily_candidates_unions_trade_value_leaders_with_dedup(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod

    # Given: primary scan returns one stock, trade-value union returns an overlapping
    # code (must be deduped, primary row wins) plus a genuinely new one
    primary_rows = [{"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "5.0"}]
    union_rows = [
        {"code": "005930", "name": None, "price": "70001", "chgrate": "0.01"},
        {"code": "000660", "name": None, "price": "180000", "chgrate": "0.01"},
    ]

    async def _fake_scan(client_arg, session_arg, **kwargs):
        return primary_rows

    async def _fake_union(session_arg, **kwargs):
        return union_rows

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _fake_scan)
    monkeypatch.setattr(collect_mod, "fetch_trade_value_union", _fake_union)

    # When
    out = asyncio.run(collect_mod.resolve_daily_candidates(AsyncMock(), object(), toss_client=AsyncMock()))

    # Then: primary row kept as-is (not overwritten by union's stale price), new code appended
    assert out == [
        {"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "5.0"},
        {"code": "000660", "name": None, "price": "180000", "chgrate": "0.01"},
    ]


def test_resolve_daily_candidates_zero_regression_without_toss_client(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod

    scan_rows = [{"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}]

    async def _fake_scan(client_arg, session_arg, **kwargs):
        return scan_rows

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _fake_scan)

    # When: no toss_client passed at all (mirrors existing regression test call shape)
    out = asyncio.run(collect_mod.resolve_daily_candidates(AsyncMock(), object()))

    # Then: union contributes nothing, output identical to primary-only behavior
    assert out == scan_rows


def test_fetch_single_stock_treats_vendor_unresolved_code_as_quote_failure() -> None:
    import asyncio
    import math
    from unittest.mock import AsyncMock

    from src.daily import collect
    from src.processing.schema import QUOTE_FAILED_COL

    # Given: Q 접두어 없는 ETN 코드 -- KIS 실측처럼 rt_cd=0이지만 종목코드 공란, 시세 전부 0
    client = AsyncMock()
    client.get_current_price = AsyncMock(
        return_value={"rt_cd": "0", "output": {"stck_prpr": "0", "stck_hgpr": "0", "stck_lwpr": "0", "acml_vol": "0", "rprs_mrkt_kor_name": "KOSPI"}}
    )
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": {"askp1": "0"}})
    stock = {"code": "500041", "name": "신한 인버스 2X 구리 선물 ETN", "price": "+35300", "chgrate": "+8.47"}

    # When
    row, failed, orderbook_rows = asyncio.run(
        collect.fetch_single_stock(0, stock, 1, asyncio.Semaphore(1), client, object())
    )

    # Then: 0을 시세로 쓰지 않고 현재가 실패 + 미해석 태그로 표식하며 호가 행도 남기지 않는다
    assert row[QUOTE_FAILED_COL] is True
    assert "현재가" in failed and collect.QUOTE_UNRESOLVED_API in failed
    assert math.isnan(float(row["고가"])) and math.isnan(float(row["거래량"]))
    assert row["종가"] == 35300
    assert orderbook_rows == []


def test_load_eligible_codes_returns_symbols_listed_on_previous_trading_day(tmp_path) -> None:
    import pandas as pd

    from src.daily.collect import load_eligible_codes

    # Given: 금요일(9/11) 상장 구성에 영문코드 보통주/우선주 포함, 목요일 행은 과거
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-10", "2026-09-11", "2026-09-11", "2026-09-11"]),
        "symbol": ["000001", "005930", "0220W0", "00088K"],
        "close": [1000.0, 70000.0, 6790.0, 20000.0],
    })
    path = tmp_path / "price_history.parquet"
    panel.to_parquet(path, index=False)

    # When: 직전 거래일(9/11)을 명시 전달
    out = load_eligible_codes(pd.Timestamp("2026-09-14"), prev_trading_day=pd.Timestamp("2026-09-11"), path=path)

    # Then
    assert out == frozenset({"005930", "0220W0", "00088K"})


def test_load_eligible_codes_fails_closed_when_panel_is_stale_or_missing(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.daily.collect import load_eligible_codes

    # Given: 직전 거래일(9/11) 수집이 누락된 패널
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-09-10"]), "symbol": ["005930"], "close": [70000.0]})
    path = tmp_path / "price_history.parquet"
    panel.to_parquet(path, index=False)

    # When / Then: 신선도 미달은 fail-closed
    with pytest.raises(ValueError, match="stale price_history"):
        load_eligible_codes(pd.Timestamp("2026-09-14"), prev_trading_day=pd.Timestamp("2026-09-11"), path=path)

    # And: 파일 부재도 fail-closed
    with pytest.raises(FileNotFoundError):
        load_eligible_codes(pd.Timestamp("2026-09-14"), prev_trading_day=pd.Timestamp("2026-09-11"), path=tmp_path / "absent.parquet")


def test_filter_eligible_candidates_drops_instruments_outside_research_panel(caplog) -> None:
    import logging

    from src.daily.collect import filter_eligible_candidates

    # Given: 보통주, 영문코드 보통주, Q접두어 없는 ETN, ETF
    stock_list = [
        {"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "3.0"},
        {"code": "0220W0", "name": None, "price": "6790", "chgrate": "0.0847"},
        {"code": "500041", "name": "신한 인버스 2X 구리 선물 ETN", "price": "+35300", "chgrate": "+8.47"},
        {"code": "114800", "name": "KODEX 인버스", "price": "1038", "chgrate": "3.59"},
    ]
    eligible = frozenset({"005930", "0220W0", "000660"})

    # When
    with caplog.at_level(logging.INFO, logger="src.daily.collect"):
        out = filter_eligible_candidates(stock_list, eligible)

    # Then
    assert [row["code"] for row in out] == ["005930", "0220W0"]
    assert "stage=instrument_eligibility" in caplog.text
    assert "n_dropped=2" in caplog.text
    assert filter_eligible_candidates([], eligible) == []


def test_filter_eligible_candidates_raises_when_no_scanned_row_is_eligible() -> None:
    import pytest

    from src.daily.collect import filter_eligible_candidates

    # Given: 스캔 전부가 패널 밖 종목 (패널/스캔 코드 체계 불일치 신호)
    stock_list = [{"code": "500041", "name": "ETN", "price": "1000", "chgrate": "5.0"}]

    # When / Then
    with pytest.raises(ValueError, match="eligible"):
        filter_eligible_candidates(stock_list, frozenset({"005930"}))


def test_main_filters_candidates_by_eligibility_before_quoting(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    import pandas as pd
    import pytest

    from src.daily import collect

    class _Stop(Exception):  # noqa: N818 - contract skeleton name
        pass

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, 15, 20, 5, tzinfo=tz)

    class _FakeKis:
        def __init__(self, *args, **kwargs):
            pass

        async def ensure_token(self, session):
            return None

        async def get_market_index_rate(self, session, code):
            return {"rt_cd": "1"}

    async def _trading_day(_client, _session, _date):
        return True

    scanned = [
        {"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "3.0"},
        {"code": "500041", "name": "ETN", "price": "+35300", "chgrate": "+8.47"},
    ]
    eligibility_dates = []
    quoted = []

    def _eligible(decision_date, **_kwargs):
        eligibility_dates.append(decision_date)
        return frozenset({"005930"})

    async def _fetch_all(stock_list, _client, _session):
        quoted.append([row["code"] for row in stock_list])
        raise _Stop

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(collect, "_validate_hts_id", lambda: None)
    monkeypatch.setattr(collect, "KisApiClient", _FakeKis)
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", lambda: None)
    monkeypatch.setattr(collect, "build_toss_scan_client", lambda: None)
    monkeypatch.setattr(collect, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(collect, "resolve_daily_candidates", AsyncMock(return_value=scanned))
    monkeypatch.setattr(collect, "load_eligible_codes", _eligible)
    monkeypatch.setattr(collect, "fetch_all_stock_data", _fetch_all)

    # When
    with pytest.raises(_Stop):
        asyncio.run(collect.main(force=False))

    # Then: 결정일 기준 적격성으로 거른 뒤에만 시세를 조회한다
    assert eligibility_dates == [pd.Timestamp("2026-09-14")]
    assert quoted == [["005930"]]


def test_main_fails_closed_when_eligibility_panel_is_stale(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    import pytest

    from src.daily import collect

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, 15, 20, 5, tzinfo=tz)

    class _FakeKis:
        def __init__(self, *args, **kwargs):
            pass

        async def ensure_token(self, session):
            return None

        async def get_market_index_rate(self, session, code):
            return {"rt_cd": "1"}

    async def _trading_day(_client, _session, _date):
        return True

    def _stale(_decision_date, **_kwargs):
        raise ValueError("stale price_history: no rows on prev_trading_day=2026-09-11")

    fetch_all = AsyncMock()
    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(collect, "_validate_hts_id", lambda: None)
    monkeypatch.setattr(collect, "KisApiClient", _FakeKis)
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", lambda: None)
    monkeypatch.setattr(collect, "build_toss_scan_client", lambda: None)
    monkeypatch.setattr(collect, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(
        collect,
        "resolve_daily_candidates",
        AsyncMock(return_value=[{"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "3.0"}]),
    )
    monkeypatch.setattr(collect, "load_eligible_codes", _stale)
    monkeypatch.setattr(collect, "fetch_all_stock_data", fetch_all)

    # When / Then
    with pytest.raises(ValueError, match="stale price_history"):
        asyncio.run(collect.main(force=False))
    fetch_all.assert_not_awaited()


def test_fetch_single_stock_requests_krx_quote_without_venue_fallback_and_logs_failures(caplog) -> None:
    import asyncio
    import logging
    from unittest.mock import AsyncMock

    from src.daily import collect

    # Given: TPS 초과로 현재가 실패
    client = AsyncMock()
    client.get_current_price = AsyncMock(
        return_value={"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다."}
    )
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": {}})
    stock = {"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "3.0"}

    # When
    with caplog.at_level(logging.WARNING, logger="src.daily.collect"):
        asyncio.run(collect.fetch_single_stock(0, stock, 1, asyncio.Semaphore(1), client, object()))

    # Then: KRX 단일 시장 조회 + 실패 사유 구조화 로그
    assert client.get_current_price.await_args.kwargs == {"market_div_code": "J", "allow_market_div_fallback": False}
    assert "stage=realtime_quote" in caplog.text
    assert "code=005930" in caplog.text
    assert "unresolved=False" in caplog.text
    assert "rt_cd=1" in caplog.text
    assert "msg_cd=EGW00201" in caplog.text

    # And: 미해석(rt_cd=0, 종목코드 공란) 응답도 사유와 함께 남는다
    caplog.clear()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "msg_cd": "MCA00000", "output": {"stck_prpr": "0"}})
    with caplog.at_level(logging.WARNING, logger="src.daily.collect"):
        asyncio.run(collect.fetch_single_stock(0, stock, 1, asyncio.Semaphore(1), client, object()))
    assert "unresolved=True" in caplog.text
    assert "rt_cd=0" in caplog.text


def test_superseded_unresolved_instrument_helpers_are_removed() -> None:
    from src.daily import collect

    # Then: 시세 조회 후 제거 방식은 조회 전 적격성 필터로 대체되어 삭제된다
    assert not hasattr(collect, "listed_codes_before")
    assert not hasattr(collect, "drop_unresolved_unlisted_instruments")
    assert collect.QUOTE_UNRESOLVED_API == "현재가_미해석"


def test_collect_module_constants_sourced_from_data_account_kwargs() -> None:
    from pathlib import Path

    from src.daily import collect

    kwargs = collect.kis_data_client_kwargs()

    assert kwargs["app_key"] == collect.APP_KEY
    assert kwargs["app_secret"] == collect.APP_SECRET
    assert kwargs["account_id"] == collect.ACCOUNT_ID
    assert kwargs["hts_id"] == collect.HTS_ID
    assert Path(kwargs["token_file"]).name == Path(collect.TOKEN_FILE).name


def test_resolve_prev_trading_day_kis_skips_weekend_and_kis_holiday(monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import collect

    asked = []

    async def _kis(_client, _session, date):
        asked.append(pd.Timestamp(date))
        return pd.Timestamp(date) != pd.Timestamp("2026-09-11")

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis)

    # When: 월요일 결정, 금요일(9/11) 휴장
    prev = asyncio.run(collect.resolve_prev_trading_day_kis(object(), object(), pd.Timestamp("2026-09-14")))

    # Then: 주말은 호출 없이 건너뛰고 목요일 반환
    assert prev == pd.Timestamp("2026-09-10")
    assert asked == [pd.Timestamp("2026-09-11"), pd.Timestamp("2026-09-10")]


def test_resolve_prev_trading_day_kis_falls_back_to_krx_when_kis_oracle_fails(monkeypatch, caplog) -> None:
    import asyncio
    import logging

    import pandas as pd

    from src.daily import collect

    async def _kis_down(_client, _session, _date):
        raise RuntimeError("KIS trading-day oracle failed rt_cd=9")

    krx_asked = []

    def _krx(date):
        krx_asked.append(pd.Timestamp(date))
        return True

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis_down)

    # When
    with caplog.at_level(logging.WARNING, logger="src.daily.collect"):
        prev = asyncio.run(
            collect.resolve_prev_trading_day_kis(object(), object(), pd.Timestamp("2026-09-14"), krx_is_trading_day=_krx)
        )

    # Then
    assert prev == pd.Timestamp("2026-09-11")
    assert krx_asked == [pd.Timestamp("2026-09-11")]
    assert "stage=prev_trading_day" in caplog.text
    assert "fallback=krx" in caplog.text


def test_resolve_prev_trading_day_kis_fails_closed_when_both_oracles_fail_or_no_day(monkeypatch) -> None:
    import asyncio

    import pandas as pd
    import pytest

    from src.daily import collect

    async def _kis_down(_client, _session, _date):
        raise RuntimeError("KIS trading-day oracle failed rt_cd=9")

    def _krx_down(_date):
        raise RuntimeError("krx down")

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis_down)

    # When / Then: 두 오라클 모두 실패
    with pytest.raises(RuntimeError, match="krx down"):
        asyncio.run(
            collect.resolve_prev_trading_day_kis(object(), object(), pd.Timestamp("2026-09-14"), krx_is_trading_day=_krx_down)
        )

    async def _always_closed(_client, _session, _date):
        return False

    monkeypatch.setattr(collect, "is_kis_trading_day", _always_closed)

    # When / Then: 조회 한도 내 거래일 없음
    with pytest.raises(ValueError, match="no trading day"):
        asyncio.run(
            collect.resolve_prev_trading_day_kis(object(), object(), pd.Timestamp("2026-09-14"), max_lookback_days=3)
        )


def test_load_eligible_codes_rejects_prev_trading_day_not_before_decision_date(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.daily.collect import load_eligible_codes

    path = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-09-14"]), "symbol": ["005930"], "close": [1.0]}).to_parquet(path, index=False)

    # When / Then: 결정일 당일 구성은 룩어헤드
    with pytest.raises(ValueError, match="prev_trading_day"):
        load_eligible_codes(pd.Timestamp("2026-09-14"), prev_trading_day=pd.Timestamp("2026-09-14"), path=path)


def test_resolve_eligible_codes_passes_kis_resolved_prev_day_to_panel_lookup(monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import collect

    async def _kis(_client, _session, _date):
        return True

    calls = []

    def _load(decision_date, *, prev_trading_day, path=None):
        calls.append((decision_date, prev_trading_day))
        return frozenset({"005930"})

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis)
    monkeypatch.setattr(collect, "load_eligible_codes", _load)

    # When
    out = asyncio.run(collect.resolve_eligible_codes(object(), object(), pd.Timestamp("2026-09-14")))

    # Then
    assert out == frozenset({"005930"})
    assert calls == [(pd.Timestamp("2026-09-14"), pd.Timestamp("2026-09-11"))]


def test_main_resolves_previous_trading_day_through_kis_before_eligibility(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock

    import pandas as pd
    import pytest

    from src.daily import collect

    class _StopError(Exception):
        pass

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, 15, 20, 5, tzinfo=tz)

    class _FakeKis:
        def __init__(self, *args, **kwargs):
            pass

        async def ensure_token(self, session):
            return None

        async def get_market_index_rate(self, session, code):
            return {"rt_cd": "1"}

    oracle_dates = []

    async def _trading_day(_client, _session, date):
        oracle_dates.append(pd.Timestamp(date).normalize())
        return True

    eligibility_calls = []

    def _eligible(decision_date, *, prev_trading_day, path=None):
        eligibility_calls.append((decision_date, prev_trading_day))
        return frozenset({"005930"})

    async def _fetch_all(stock_list, _client, _session):
        raise _StopError

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(collect, "_validate_hts_id", lambda: None)
    monkeypatch.setattr(collect, "KisApiClient", _FakeKis)
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", lambda: None)
    monkeypatch.setattr(collect, "build_toss_scan_client", lambda: None)
    monkeypatch.setattr(collect, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(
        collect,
        "resolve_daily_candidates",
        AsyncMock(return_value=[{"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "3.0"}]),
    )
    monkeypatch.setattr(collect, "load_eligible_codes", _eligible)
    monkeypatch.setattr(collect, "fetch_all_stock_data", _fetch_all)

    # When
    with pytest.raises(_StopError):
        asyncio.run(collect.main(force=False))

    # Then: 당일 확인(9/14) 후 직전 거래일(9/11)을 KIS로 확정해 적격성 조회에 전달
    assert eligibility_calls == [(pd.Timestamp("2026-09-14"), pd.Timestamp("2026-09-11"))]
    assert oracle_dates == [pd.Timestamp("2026-09-14"), pd.Timestamp("2026-09-11")]


def test_requote_failed_quotes_recovers_transient_failure_but_not_unresolved(caplog) -> None:
    import asyncio
    import logging
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily import collect
    from src.processing.schema import QUOTE_FAILED_COL

    class _Client:
        def __init__(self, fail_first_for=(), unresolved=()):
            self.calls = {}
            self.fail_first_for = set(fail_first_for)
            self.unresolved = set(unresolved)

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            n = self.calls.get(code, 0) + 1
            self.calls[code] = n
            if code in self.unresolved:
                return {"rt_cd": "0", "msg_cd": "MCA00000", "output": {"stck_prpr": "0"}}
            if code in self.fail_first_for and n == 1:
                return {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다."}
            return {"rt_cd": "0", "output": {
                "stck_shrn_iscd": code, "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100",
                "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100",
                "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI",
            }}

        async def get_investor_trend_estimate(self, session, code):
            return {"rt_cd": "0", "output2": []}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output1": {}}

    client = _Client(fail_first_for={"000660"}, unresolved={"500041"})
    stock_list = [
        {"code": "005930", "name": "A", "price": "18000", "chgrate": "5.0"},
        {"code": "000660", "name": "B", "price": "18000", "chgrate": "5.0"},
        {"code": "500041", "name": "C", "price": "35300", "chgrate": "8.47"},
    ]

    async def _run():
        sem = asyncio.Semaphore(4)
        first = await asyncio.gather(*[
            collect.fetch_single_stock(i, s, len(stock_list), sem, client, object()) for i, s in enumerate(stock_list)
        ])
        return first, await collect.requote_failed_quotes(
            stock_list, list(first), client, object(), sem,
            now_fn=lambda: datetime(2026, 9, 14, 15, 21, 30, tzinfo=ZoneInfo("Asia/Seoul")),
        )

    # When
    with caplog.at_level(logging.INFO, logger="src.daily.collect"):
        first, out = asyncio.run(_run())

    # Then: 일시 실패 행만 1회 재조회되어 복구, 미해석 행은 재시도하지 않음
    assert first[1][0][QUOTE_FAILED_COL] is True
    assert out[1][0][QUOTE_FAILED_COL] is False
    assert out[2][0][QUOTE_FAILED_COL] is True
    assert client.calls == {"005930": 1, "000660": 2, "500041": 1}
    assert "stage=realtime_requote" in caplog.text
    assert "n_retry=1" in caplog.text
    assert "n_recovered=1" in caplog.text


def test_requote_failed_quotes_skips_at_or_after_deadline(caplog) -> None:
    import asyncio
    import logging
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.config.market_session import REALTIME_REQUOTE_DEADLINE_HHMMSS
    from src.daily import collect

    class _Client:
        def __init__(self, fail_first_for=(), unresolved=()):
            self.calls = {}
            self.fail_first_for = set(fail_first_for)
            self.unresolved = set(unresolved)

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            n = self.calls.get(code, 0) + 1
            self.calls[code] = n
            if code in self.unresolved:
                return {"rt_cd": "0", "msg_cd": "MCA00000", "output": {"stck_prpr": "0"}}
            if code in self.fail_first_for and n == 1:
                return {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다."}
            return {"rt_cd": "0", "output": {
                "stck_shrn_iscd": code, "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100",
                "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100",
                "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI",
            }}

        async def get_investor_trend_estimate(self, session, code):
            return {"rt_cd": "0", "output2": []}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output1": {}}

    client = _Client(fail_first_for={"000660"})
    stock_list = [{"code": "000660", "name": "B", "price": "18000", "chgrate": "5.0"}]
    hh, mm, ss = int(REALTIME_REQUOTE_DEADLINE_HHMMSS[:2]), int(REALTIME_REQUOTE_DEADLINE_HHMMSS[2:4]), int(REALTIME_REQUOTE_DEADLINE_HHMMSS[4:])

    async def _run():
        sem = asyncio.Semaphore(1)
        first = [await collect.fetch_single_stock(0, stock_list[0], 1, sem, client, object())]
        return await collect.requote_failed_quotes(
            stock_list, first, client, object(), sem,
            now_fn=lambda: datetime(2026, 9, 14, hh, mm, ss, tzinfo=ZoneInfo("Asia/Seoul")),
        )

    # When
    with caplog.at_level(logging.WARNING, logger="src.daily.collect"):
        out = asyncio.run(_run())

    # Then
    assert client.calls == {"000660": 1}
    assert len(out) == 1
    assert "status=SKIPPED" in caplog.text
    assert "reason=deadline" in caplog.text


def test_fetch_all_stock_data_requotes_and_emits_no_carriage_return_off_tty(monkeypatch, capsys, caplog) -> None:
    import asyncio
    import logging
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily import collect
    from src.processing.schema import QUOTE_FAILED_COL

    class _Client:
        def __init__(self, fail_first_for=(), unresolved=()):
            self.calls = {}
            self.fail_first_for = set(fail_first_for)
            self.unresolved = set(unresolved)

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            n = self.calls.get(code, 0) + 1
            self.calls[code] = n
            if code in self.unresolved:
                return {"rt_cd": "0", "msg_cd": "MCA00000", "output": {"stck_prpr": "0"}}
            if code in self.fail_first_for and n == 1:
                return {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다."}
            return {"rt_cd": "0", "output": {
                "stck_shrn_iscd": code, "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100",
                "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100",
                "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI",
            }}

        async def get_investor_trend_estimate(self, session, code):
            return {"rt_cd": "0", "output2": []}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output1": {}}

    client = _Client(fail_first_for={"000660"})
    stock_list = [
        {"code": "005930", "name": "A", "price": "18000", "chgrate": "5.0"},
        {"code": "000660", "name": "B", "price": "18000", "chgrate": "5.0"},
    ]
    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: 0)

    # When
    with caplog.at_level(logging.INFO, logger="src.daily.collect"):
        results, failed_info = asyncio.run(
            collect.fetch_all_stock_data(
                stock_list, client, object(),
                now_fn=lambda: datetime(2026, 9, 14, 15, 21, 0, tzinfo=ZoneInfo("Asia/Seoul")),
            )
        )

    # Then
    assert [row[QUOTE_FAILED_COL] for row in results] == [False, False]
    assert failed_info == []
    assert "\r" not in capsys.readouterr().out
    assert "stage=realtime_quote_batch n_rows=2 n_quote_failed=0" in caplog.text


def test_resolve_prev_trading_day_kis_uses_default_krx_oracle_on_kis_failure(monkeypatch) -> None:
    # Diff-coverage supplement: contract-mandated default-oracle branch
    # (`if oracle is None: from ... import is_krx_trading_day`) has no skeleton.
    import asyncio

    import pandas as pd

    import src.data.trading_calendar as trading_calendar
    from src.daily import collect

    async def _kis_down(_client, _session, _date):
        raise RuntimeError("KIS trading-day oracle failed rt_cd=9")

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis_down)
    monkeypatch.setattr(trading_calendar, "is_krx_trading_day", lambda _date: True)

    # When: krx_is_trading_day 미지정
    prev = asyncio.run(collect.resolve_prev_trading_day_kis(object(), object(), pd.Timestamp("2026-09-14")))

    # Then: 기본 KRX 오라클로 9/11 확정
    assert prev == pd.Timestamp("2026-09-11")


def test_fetch_all_stock_data_writes_progress_bar_on_tty(monkeypatch) -> None:
    # Diff-coverage supplement: contract-mandated TTY branch
    # (`if sys.stdout.isatty():` write/flush) has no skeleton.
    import asyncio
    import io
    import sys
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_shrn_iscd": "005930", "stck_prpr": "18000", "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800", "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100", "hts_avls": "3000", "acml_tr_pbmn": "50000000000", "rprs_mrkt_kor_name": "KOSPI"}})
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    client.get_orderbook_snapshot = AsyncMock(
        return_value={"rt_cd": "0", "output1": {"askp1": "18010", "bidp1": "17990"}}
    )
    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: 0)

    buf = io.StringIO()
    buf.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdout", buf)

    # When
    stock_list = [{"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"}]
    results, failed_info = asyncio.run(collect.fetch_all_stock_data(stock_list, client, object()))

    # Then: TTY에서는 \r 진행바 출력
    assert len(results) == 1
    assert failed_info == []
    assert "\r" in buf.getvalue()


def test_resolve_daily_candidates_enables_kis_band_fallback(monkeypatch) -> None:
    import asyncio

    import src.daily.collect as collect_mod

    captured: dict = {}
    kis_client = object()

    async def _fake_scan(client, session, **kwargs):
        captured["client"] = client
        captured.update(kwargs)
        return [{"code": "000001", "name": "A", "price": "1000", "chgrate": "3.0"}]

    async def _no_union(session, *, toss_client=None, count=100):
        return []

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _fake_scan)
    monkeypatch.setattr(collect_mod, "fetch_trade_value_union", _no_union)

    # When
    out = asyncio.run(collect_mod.resolve_daily_candidates(kis_client, object(), kiwoom_client=object(), toss_client=None))

    # Then
    assert [r["code"] for r in out] == ["000001"]
    assert captured["client"] is kis_client
    assert captured["kis_band_fallback"] is True


def test_resolve_eligible_codes_returns_all_listed_codes_without_history_filter(monkeypatch) -> None:
    """PIT 재구축 이후 price_history는 전종목 이력을 보유하므로, 별도 history-complete 교집합 없이
    직전 거래일 상장 종목이 곧 적격 종목이다 (0220W0처럼 예전엔 배제되던 코드도 그대로 통과)."""
    import asyncio

    import pandas as pd

    from src.daily import collect

    async def _kis(_client, _session, _date):
        return True

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis)
    monkeypatch.setattr(
        collect, "load_eligible_codes", lambda decision_date, *, prev_trading_day, path=None: frozenset({"005930", "138930", "0220W0"})
    )

    # When
    out = asyncio.run(collect.resolve_eligible_codes(object(), object(), pd.Timestamp("2026-09-14")))

    # Then
    assert out == frozenset({"005930", "138930", "0220W0"})


def test_legacy_universe_guard_symbols_are_fully_deleted() -> None:
    from src.daily import collect

    # Then: 가드 상수와 함수가 완전히 사라졌다
    assert not hasattr(collect, "PANEL_LEGACY_UNIVERSE_LAST_DATE")
    assert not hasattr(collect, "load_history_complete_codes")
