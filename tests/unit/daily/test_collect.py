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


@pytest.fixture(autouse=True)
def _standard_session(monkeypatch) -> None:
    """Pre-gate scenarios run under a STANDARD session; gate scenarios inject their own resolver."""
    from src.daily import collect
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay, SessionKind

    def _resolve(trading_day, **_kwargs):
        return SessionDay(
            trading_date=trading_day,
            kind=SessionKind.STANDARD,
            clock=SessionClock.standard(trading_day),
            provenance="standard",
        )

    monkeypatch.setattr(collect, "resolve_session_day", _resolve)


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
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
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
    from zoneinfo import ZoneInfo

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
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
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


def test_collect_main_skips_shifted_and_unknown_before_vendor_calls(monkeypatch) -> None:
    import asyncio
    from datetime import date, datetime

    from src.daily import collect
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay, SessionKind

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 6, 15, 22, 0, tzinfo=tz)

    def _raising_client(*args, **kwargs):
        raise AssertionError("session-gated SKIP must not construct clients")

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(collect, "KisApiClient", _raising_client)

    for kind in (SessionKind.SHIFTED, SessionKind.UNKNOWN):
        target = date(2026, 10, 6)
        clock = None if kind is SessionKind.UNKNOWN else SessionClock.standard(target)
        day = SessionDay(trading_date=target, kind=kind, clock=clock, provenance="test")
        monkeypatch.setattr(collect, "resolve_session_day", lambda _d, _day=day, **_k: _day)
        assert asyncio.run(collect.main(force=False)) is None


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

    async def _fetch_all(stock_list, _client, _session, **_kwargs):
        quoted.append([row["code"] for row in stock_list])
        raise _Stop

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
    monkeypatch.setattr(collect, "KisApiClient", _FakeKis)
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", lambda: None)
    monkeypatch.setattr(collect, "build_toss_scan_client", lambda: None)
    monkeypatch.setattr(collect, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(collect, "resolve_daily_candidates", AsyncMock(return_value=scanned))
    monkeypatch.setattr(collect, "load_eligible_codes", _eligible)
    monkeypatch.setattr(collect, "load_security_classification", lambda *a, **k: frozenset({"005930", "138930", "0220W0", "000660", "500041"}))
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
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
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
    monkeypatch.setattr(collect, "load_security_classification", lambda *a, **k: frozenset({"005930"}))
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


def test_collect_module_has_no_import_time_kis_credential_globals() -> None:
    """collect.py는 safe_float 등 순수 헬퍼 재사용을 위해 finalize_close/paper_trade에서도
    임포트되므로, KIS 자격증명 해석은 main() 호출 시점까지 지연되어야 한다(임포트만으로
    호스트 KIS 슬롯 설정을 요구하면 안 된다)."""
    from src.daily import collect

    for stale_global in ("APP_KEY", "APP_SECRET", "ACCOUNT_ID", "HTS_ID", "TOKEN_FILE", "_KIS_DATA_KWARGS"):
        assert not hasattr(collect, stale_global), stale_global


def test_collect_main_resolves_kis_credentials_lazily_at_call_time(monkeypatch) -> None:
    import asyncio

    import pytest

    from src.daily import collect

    calls: list[str] = []

    def _fake_kwargs():
        calls.append("resolved")
        return {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"}

    def _stop(*_a, **_k):
        raise RuntimeError("stop")

    monkeypatch.setattr(collect, "kis_data_client_kwargs", _fake_kwargs)
    monkeypatch.setattr(collect, "_validate_hts_id", lambda hts_id: calls.append(f"validated:{hts_id}"))
    monkeypatch.setattr(collect, "_validate_decision_window", _stop)

    with pytest.raises(RuntimeError, match="stop"):
        asyncio.run(collect.main(force=False))

    assert calls == ["resolved", "validated:h"]


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
    monkeypatch.setattr(collect, "load_security_classification", lambda *a, **k: frozenset({"005930", "138930", "0220W0", "000660", "500041"}))

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

    async def _fetch_all(stock_list, _client, _session, **_kwargs):
        raise _StopError

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
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
    monkeypatch.setattr(collect, "load_security_classification", lambda *a, **k: frozenset({"005930", "138930", "0220W0", "000660", "500041"}))
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
    monkeypatch.setattr(
        collect, "load_security_classification", lambda decision_date, *, prev_trading_day, path=None: frozenset({"005930", "138930", "0220W0"})
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


def test_fetch_all_stock_data_sharded_single_client_passthrough(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(
        return_value={
            "rt_cd": "0",
            "output": {
                "stck_shrn_iscd": "005930", "stck_prpr": "70000", "stck_oprc": "69000",
                "stck_hgpr": "70500", "stck_lwpr": "68900", "acml_vol": "1000",
                "prdy_ctrt": "1.5", "lstn_stcn": "100", "hts_avls": "1000",
                "acml_tr_pbmn": "100000000", "rprs_mrkt_kor_name": "KOSPI",
            },
        }
    )
    client.get_investor_trend_estimate = AsyncMock(
        return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]}
    )
    ladder = {f"askp{i}": str(70000 + i * 100) for i in range(1, 11)}
    ladder.update({"bidp1": "69900", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})
    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: None)

    stock_list = [{"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "1.5"}]
    results, failed_info = asyncio.run(
        collect.fetch_all_stock_data_sharded(stock_list, [client], object())
    )

    assert len(results) == 1
    assert failed_info == []
    assert client.get_current_price.await_count == 1


def test_fetch_all_stock_data_sharded_splits_across_clients_and_preserves_order(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    def make_client():
        c = AsyncMock()

        async def _price(session, code, market_div_code=None, allow_market_div_fallback=True):
            return {
                "rt_cd": "0",
                "output": {
                    "stck_shrn_iscd": code, "stck_prpr": "1000", "stck_oprc": "1000",
                    "stck_hgpr": "1000", "stck_lwpr": "1000", "acml_vol": "1",
                    "prdy_ctrt": "0.0", "lstn_stcn": "1", "hts_avls": "1",
                    "acml_tr_pbmn": "1", "rprs_mrkt_kor_name": "KOSPI",
                },
            }

        c.get_current_price = AsyncMock(side_effect=_price)
        c.get_investor_trend_estimate = AsyncMock(
            return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "0", "orgn_fake_ntby_qty": "0"}]}
        )
        ladder = {f"askp{i}": "1000" for i in range(1, 11)}
        ladder.update({"bidp1": "999", "total_askp_rsqn": "1", "total_bidp_rsqn": "1"})
        c.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})
        return c

    client_a = make_client()
    client_b = make_client()
    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: None)

    stock_list = [
        {"code": "AAA1", "name": "a1", "price": "1000", "chgrate": "0.0"},
        {"code": "AAA2", "name": "a2", "price": "1000", "chgrate": "0.0"},
        {"code": "AAA3", "name": "a3", "price": "1000", "chgrate": "0.0"},
    ]
    results, failed_info = asyncio.run(
        collect.fetch_all_stock_data_sharded(stock_list, [client_a, client_b], object())
    )

    assert [r["종목코드"] for r in results] == ["AAA1", "AAA2", "AAA3"]
    assert failed_info == []
    assert client_a.get_current_price.await_count == 2
    assert client_b.get_current_price.await_count == 1


def test_fetch_all_stock_data_sharded_skips_empty_chunk_when_fewer_stocks_than_clients(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    def make_client():
        c = AsyncMock()

        async def _price(session, code, market_div_code=None, allow_market_div_fallback=True):
            return {
                "rt_cd": "0",
                "output": {
                    "stck_shrn_iscd": code, "stck_prpr": "1000", "stck_oprc": "1000",
                    "stck_hgpr": "1000", "stck_lwpr": "1000", "acml_vol": "1",
                    "prdy_ctrt": "0.0", "lstn_stcn": "1", "hts_avls": "1",
                    "acml_tr_pbmn": "1", "rprs_mrkt_kor_name": "KOSPI",
                },
            }

        c.get_current_price = AsyncMock(side_effect=_price)
        c.get_investor_trend_estimate = AsyncMock(
            return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "0", "orgn_fake_ntby_qty": "0"}]}
        )
        ladder = {f"askp{i}": "1000" for i in range(1, 11)}
        ladder.update({"bidp1": "999", "total_askp_rsqn": "1", "total_bidp_rsqn": "1"})
        c.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})
        return c

    client_a = make_client()
    client_b = make_client()
    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: None)

    stock_list = [{"code": "AAA1", "name": "a1", "price": "1000", "chgrate": "0.0"}]
    results, failed_info = asyncio.run(
        collect.fetch_all_stock_data_sharded(stock_list, [client_a, client_b], object())
    )

    assert [r["종목코드"] for r in results] == ["AAA1"]
    assert failed_info == []
    assert client_a.get_current_price.await_count == 1
    assert client_b.get_current_price.await_count == 0


def test_fetch_all_stock_data_sharded_returns_empty_when_no_pairs() -> None:
    import asyncio

    from src.daily import collect

    results, failed_info = asyncio.run(
        collect.fetch_all_stock_data_sharded([], [object(), object()], object())
    )

    assert results == []
    assert failed_info == []


def test_main_issues_shard_token_and_fans_out_to_sharded_collect(monkeypatch) -> None:
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

    issued = []

    class _FakeKis:
        def __init__(self, *args, **kwargs):
            issued.append(args)

        async def ensure_token(self, session):
            return None

        async def get_market_index_rate(self, session, code):
            return {"rt_cd": "1"}

    async def _trading_day(_client, _session, _date):
        return True

    def _eligible(decision_date, **_kwargs):
        return frozenset({"005930"})

    captured = {}

    async def _sharded(stock_list, clients, _session, **_kwargs):
        captured["n_clients"] = len(clients)
        raise _Stop

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(
        collect,
        "kis_decision_shard_client_kwargs",
        lambda: [
            {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
            {"app_key": "k5", "app_secret": "s5", "account_id": "", "hts_id": "h5", "token_file": "t5"},
        ],
    )
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
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
    monkeypatch.setattr(collect, "load_security_classification", lambda *a, **k: frozenset({"005930", "138930", "0220W0", "000660", "500041"}))
    monkeypatch.setattr(collect, "fetch_all_stock_data_sharded", _sharded)

    # When
    with pytest.raises(_Stop):
        asyncio.run(collect.main(force=False))

    # Then: 샤드 키 토큰을 발급하고 2개 클라이언트로 분할 수집한다
    assert captured["n_clients"] == 2
    assert len(issued) == 2
    assert issued[1][0] == "k5"


# ---------------------------------------------------------------------------
# closing_capture_06 decision-input invariant guards
# ---------------------------------------------------------------------------

def _capture_quote_payload(code="005930", price="18000"):
    return {
        "rt_cd": "0",
        "output": {
            "stck_shrn_iscd": code, "stck_prpr": price, "stck_sdpr": "17142",
            "stck_oprc": "17900", "stck_hgpr": "18100", "stck_lwpr": "17800",
            "acml_vol": "1000000", "prdy_ctrt": "5.0", "lstn_stcn": "100",
            "hts_avls": "3000", "acml_tr_pbmn": "50000000000",
            "rprs_mrkt_kor_name": "KOSPI",
        },
    }


def _capture_investor_payload():
    return {"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "10", "orgn_fake_ntby_qty": "20"}]}


def _capture_book_payload():
    return {"rt_cd": "0", "output1": {"askp1": "18010", "bidp1": "17990"}, "output2": {"antc_cnpr": "18020"}}


def _capture_cohort(codes=("005930",), trading_day="2026-09-14"):
    from datetime import date

    from src.data.capture_contracts import build_cohort

    scanned = [str(c) for c in codes]
    return build_cohort(
        date.fromisoformat(trading_day), scanned, scanned, {},
        eligibility_rule_version="price_history_panel@v1",
    )


def _raw_envelopes(root):
    import gzip
    import json
    from pathlib import Path

    return [
        (str(path), json.loads(gzip.decompress(path.read_bytes()).decode("utf-8")))
        for path in sorted(Path(root).rglob("*.json.gz"))
    ]


class _DelayedCaptureClient:
    def __init__(self, price=None, investor=None, book=None, delays=(0.05, 0.0, 0.02)):
        import asyncio as _asyncio

        self._asyncio = _asyncio
        self.price = price if price is not None else _capture_quote_payload()
        self.investor = investor if investor is not None else _capture_investor_payload()
        self.book = book if book is not None else _capture_book_payload()
        self.delays = delays

    async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
        await self._asyncio.sleep(self.delays[0])
        return self.price

    async def get_investor_trend_estimate(self, session, code):
        await self._asyncio.sleep(self.delays[1])
        return self.investor

    async def get_orderbook_snapshot(self, session, code, market_div_code=None):
        await self._asyncio.sleep(self.delays[2])
        return self.book


def test_fetch_single_stock_preserves_per_response_clocks(tmp_path, monkeypatch) -> None:
    """Raw observations keep individual clocks and row capture is their maximum."""
    import asyncio
    from datetime import datetime

    from src.daily import collect
    from src.data.capture_store import CaptureStore

    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: 0)
    store = CaptureStore(tmp_path / "capture")
    cohort = _capture_cohort()
    client = _DelayedCaptureClient()

    row, failed, _ob = asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"},
            1, asyncio.Semaphore(1), client, object(),
            capture_store=store, cohort=cohort, run_id="run-clocks",
        )
    )

    assert failed == []
    envelopes = _raw_envelopes(tmp_path / "capture")
    assert len(envelopes) == 3
    received = sorted(e["received_at"] for _, e in envelopes)
    assert len(set(received)) >= 2
    import pandas as _pd
    assert _pd.Timestamp(row["snapshot_timestamp"]) == _pd.Timestamp(max(received))


def test_fetch_single_stock_uses_scoped_observer_receipts(tmp_path, monkeypatch) -> None:
    """Observer receipts, not the outer gather clock, certify per-response times."""
    import asyncio
    import contextlib
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from src.daily import collect
    from src.data.capture_store import CaptureStore

    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: 0)
    kst = ZoneInfo("Asia/Seoul")
    base = datetime(2026, 9, 14, 15, 20, 0, tzinfo=kst)

    class _Scoped(_DelayedCaptureClient):
        def __init__(self):
            super().__init__(delays=(0.0, 0.0, 0.0))
            self.calls = 0

        @contextlib.contextmanager
        def observe_market_responses(self, on_page):
            self.calls += 1
            started = base + timedelta(seconds=self.calls * 10)
            yield
            on_page({"rt_cd": "0"}, {"tr_id": "X"}, started, started + timedelta(seconds=self.calls), 0, 0)

    store = CaptureStore(tmp_path / "capture")
    row, failed, _ob = asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"},
            1, asyncio.Semaphore(1), _Scoped(), object(),
            capture_store=store, cohort=_capture_cohort(), run_id="run-scoped",
        )
    )

    assert failed == []
    import pandas as _pd2
    assert _pd2.Timestamp(row["snapshot_timestamp"]) == _pd2.Timestamp(base + timedelta(seconds=33))


def test_scoped_observer_without_event_uses_call_clock_and_failure_metadata(tmp_path, monkeypatch) -> None:
    """A silent observer falls back to the call clock, while observed failures retain their error class."""
    import asyncio
    import contextlib
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily import collect
    from src.data.capture_store import CaptureStore

    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: 0)

    class _SilentScoped(_DelayedCaptureClient):
        @contextlib.contextmanager
        def observe_market_responses(self, _on_page):
            yield

    store = CaptureStore(tmp_path / "capture")
    row, failed, _ = asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "삼성전자", "price": "18000", "chgrate": "5.0"},
            1, asyncio.Semaphore(1), _SilentScoped(delays=(0.0, 0.0, 0.0)), object(),
            capture_store=store, cohort=_capture_cohort(), run_id="run-silent",
        )
    )
    assert failed == []
    assert row["snapshot_timestamp"].tzinfo is not None

    now = datetime(2026, 9, 14, 15, 20, tzinfo=ZoneInfo("Asia/Seoul"))
    context = collect._capture_context_for(
        _capture_cohort().trading_date, "run-error", _capture_cohort().cohort_id,
        collect.CaptureDataset.PRICE, "005930", "inquire-price",
    )
    collect._persist_observed_market_page(
        store, context, None, {"error_type": "TimeoutError"}, now, now, 0, 0
    )
    failure = [
        envelope for _, envelope in _raw_envelopes(tmp_path / "capture")
        if envelope["context"]["run_id"] == "run-error"
    ]
    assert failure[0]["error_type"] == "TimeoutError"


def test_fetch_single_stock_rejects_incomplete_capture_context() -> None:
    """Capture context is all-or-nothing."""
    import asyncio

    import pytest

    from src.daily import collect
    from src.data.capture_store import CaptureStore

    store = CaptureStore.__new__(CaptureStore)
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        asyncio.run(
            collect.fetch_single_stock(
                0, {"code": "005930", "name": "X", "price": "1", "chgrate": "0"},
                1, asyncio.Semaphore(1), object(), object(),
                capture_store=store, cohort=None, run_id=None,
            )
        )


def test_fetch_single_stock_propagates_capture_persistence_failure(monkeypatch) -> None:
    """Capture-store persistence failure prevents a new audited decision input."""
    import asyncio

    import pytest

    from src.daily import collect

    class _FailingStore:
        def append_response(self, response):
            raise OSError("disk unavailable")

    client = _DelayedCaptureClient(delays=(0.0, 0.0, 0.0))
    with pytest.raises(OSError, match="disk unavailable"):
        asyncio.run(
            collect.fetch_single_stock(
                0, {"code": "005930", "name": "X", "price": "18000", "chgrate": "5.0"},
                1, asyncio.Semaphore(1), client, object(),
                capture_store=_FailingStore(), cohort=_capture_cohort(), run_id="run-fail",
            )
        )


def test_persist_market_response_retries_conflicting_identity(tmp_path) -> None:
    """Retry attempts keep separate identities instead of discarding the first."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily import collect
    from src.data.capture_store import CaptureStore

    store = CaptureStore(tmp_path / "capture")
    cohort = _capture_cohort()
    client = _DelayedCaptureClient(delays=(0.0, 0.0, 0.0))
    now_fn = lambda: datetime(2026, 9, 14, 15, 21, 0, tzinfo=ZoneInfo("Asia/Seoul"))  # noqa: E731

    async def _run():
        first = await collect.fetch_single_stock(
            0, {"code": "005930", "name": "X", "price": "18000", "chgrate": "5.0"},
            1, asyncio.Semaphore(1), client, object(),
            capture_store=store, cohort=cohort, run_id="run-retry",
        )
        client.price = {"rt_cd": "1", "msg1": "tps"}
        failed_first = await collect.fetch_single_stock(
            0, {"code": "005930", "name": "X", "price": "18000", "chgrate": "5.0"},
            1, asyncio.Semaphore(1), client, object(),
            capture_store=store, cohort=cohort, run_id="run-retry",
        )
        return first, failed_first

    first, failed_first = asyncio.run(_run())
    assert first[1] == []
    assert "현재가" in failed_first[1]
    price_files = [p for p, e in _raw_envelopes(tmp_path / "capture") if e["context"]["dataset"] == "PRICE"]
    assert len(price_files) == 2
    _ = now_fn


def test_requote_keeps_failed_first_attempt_evidence(tmp_path, monkeypatch) -> None:
    """Both raw attempts survive with distinct identities when requote succeeds."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily import collect
    from src.data.capture_store import CaptureStore
    from src.processing.schema import QUOTE_FAILED_COL

    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: 0)
    store = CaptureStore(tmp_path / "capture")
    cohort = _capture_cohort()

    class _Flaky(_DelayedCaptureClient):
        def __init__(self):
            super().__init__(delays=(0.0, 0.0, 0.0))
            self.n = 0

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            self.n += 1
            if self.n == 1:
                return {"rt_cd": "1", "msg1": "tps"}
            return _capture_quote_payload()

    stock_list = [{"code": "005930", "name": "X", "price": "18000", "chgrate": "5.0"}]
    results, _failed = asyncio.run(
        collect.fetch_all_stock_data(
            stock_list, _Flaky(), object(),
            now_fn=lambda: datetime(2026, 9, 14, 15, 21, 0, tzinfo=ZoneInfo("Asia/Seoul")),
            capture_store=store, cohort=cohort, run_id="run-requote",
        )
    )
    assert results[0][QUOTE_FAILED_COL] is False
    price_files = [p for p, e in _raw_envelopes(tmp_path / "capture") if e["context"]["dataset"] == "PRICE"]
    assert len(price_files) == 2


def test_missing_investor_estimate_is_unknown_not_zero(tmp_path, monkeypatch) -> None:
    """Empty investor output2 is an unknown estimate, never certified zero flow."""
    import asyncio
    import math

    from src.daily import collect
    from src.data.capture_store import CaptureStore

    monkeypatch.setattr(collect, "append_orderbook_snapshots", lambda rows, snapshot_date: 0)
    store = CaptureStore(tmp_path / "capture")
    client = _DelayedCaptureClient(investor={"rt_cd": "0", "output2": []}, delays=(0.0, 0.0, 0.0))
    row, _failed, _ob = asyncio.run(
        collect.fetch_single_stock(
            0, {"code": "005930", "name": "X", "price": "18000", "chgrate": "5.0"},
            1, asyncio.Semaphore(1), client, object(),
            capture_store=store, cohort=_capture_cohort(), run_id="run-unknown",
        )
    )
    assert row["수급_실패"] is True
    assert math.isnan(float(row["기관_순매수"]))
    assert math.isnan(float(row["외국인_순매수"]))


def test_capture_root_follows_configured_override(tmp_path, monkeypatch) -> None:
    """Capture root honors the configured override directory."""
    from pathlib import Path

    from src.daily import collect

    monkeypatch.setattr(collect.settings, "COLLECTION_ROOT", tmp_path / "custom")
    assert collect._capture_root() == Path(tmp_path / "custom")


def test_persist_conflicting_identity_eventually_fails() -> None:
    """Repeated identity conflicts surface as publication failure."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.daily import collect

    class _AlwaysConflict:
        def append_response(self, response):
            raise ValueError("conflicting immutable artifact identity: 'x'")

    cohort = _capture_cohort()
    now = datetime(2026, 9, 14, 15, 20, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    context = collect._capture_context_for(cohort.trading_date, "run-x", cohort.cohort_id, collect.CaptureDataset.PRICE, "005930", "inquire-price")
    with pytest.raises(OSError, match="cannot be published"):
        collect._persist_market_response(_AlwaysConflict(), context, {"rt_cd": "0", "a": 2}, now, now)


def test_persist_non_conflicting_value_error_propagates(tmp_path) -> None:
    """Non-identity validation errors are not retried as new attempts."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.daily import collect

    class _BadStore:
        def append_response(self, response):
            raise ValueError("bad payload")

    now = datetime(2026, 9, 14, 15, 20, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    context = collect._capture_context_for(_capture_cohort().trading_date, "run-x", "cohort-y", collect.CaptureDataset.PRICE, "005930", "inquire-price")
    with pytest.raises(ValueError, match="bad payload"):
        collect._persist_market_response(_BadStore(), context, {"rt_cd": "0"}, now, now)


def _healthy_wide_row(code="005930", quote_failed=False):
    return {
        "종목명": code, "종목코드": code, "시장구분": "KOSPI",
        "시가": 17900.0, "고가": 18100.0, "저가": 17800.0, "종가": 18000.0,
        "전일종가": 17142.86, "거래량": 1_000_000.0, "거래대금": 500.0,
        "시가총액": 3000.0, "기관_순매수": 10.0, "외국인_순매수": 5.0,
        "등락률": 5.0, "수급_실패": False, "현재가_실패": quote_failed,
        "결정_종가": 18000.0, "종가_확정": False,
    }


def _run_main_with_mocks(monkeypatch, tmp_path, rows, scanned, eligible, *, with_snapshot=False, call_observer=False):
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock
    from zoneinfo import ZoneInfo

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

    async def _fake_scan(client_arg, session_arg, **kwargs):
        observer = kwargs.get("on_page")
        if call_observer and observer is not None:
            observer({"rt_cd": "0", "output": []}, {"vendor": "kiwoom"}, _FrozenDatetime.now(ZoneInfo("Asia/Seoul")), _FrozenDatetime.now(ZoneInfo("Asia/Seoul")), 0, 0)
        return scanned

    async def _fake_sharded(stock_list, clients, session, **kwargs):
        out_rows = []
        for row in rows:
            duplicated = dict(row)
            if with_snapshot:
                import pandas as pd

                duplicated["snapshot_timestamp"] = pd.Timestamp("2026-09-14 15:20:01", tz="Asia/Seoul")
            out_rows.append(duplicated)
        return out_rows, []

    monkeypatch.setattr(collect, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        collect,
        "kis_data_client_kwargs",
        lambda: {"app_key": "k", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "t"},
    )
    monkeypatch.setattr(collect, "_validate_hts_id", lambda _hts_id: None)
    monkeypatch.setattr(collect, "KisApiClient", _FakeKis)
    monkeypatch.setattr(collect, "kis_decision_shard_client_kwargs", lambda: [{}])
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", lambda: None)
    monkeypatch.setattr(collect, "build_toss_scan_client", lambda: None)
    monkeypatch.setattr(collect, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(collect, "resolve_daily_candidates", _fake_scan)
    monkeypatch.setattr(collect, "resolve_eligible_codes", AsyncMock(return_value=eligible))
    monkeypatch.setattr(collect, "fetch_all_stock_data_sharded", _fake_sharded)
    monkeypatch.setattr(collect, "persist_daily_snapshot", lambda df, snapshot_date=None: len(df))
    monkeypatch.setattr(collect, "_capture_root", lambda: tmp_path / "capture")
    import src.api.kis.indicators as indicators_mod

    async def _no_vol(*args, **kwargs):
        raise RuntimeError("no vol")

    monkeypatch.setattr(indicators_mod, "fetch_index_and_calculate_volatility", _no_vol)
    import src.data.panel_integrity as panel_mod

    def _no_panel(*args, **kwargs):
        raise RuntimeError("no panel")

    monkeypatch.setattr(panel_mod, "load_price_panel", _no_panel)
    return asyncio.run(collect.main(force=False))


def test_main_publishes_partial_inputs_before_coverage_failure(tmp_path, monkeypatch) -> None:
    """Coverage rejection does not erase stored decision inputs."""
    from pathlib import Path

    import pytest

    rows = [_healthy_wide_row("005930", quote_failed=True), _healthy_wide_row("000660", quote_failed=True)]
    scanned = [
        {"code": "005930", "name": "A", "price": "18000", "chgrate": "5.0"},
        {"code": "000660", "name": "B", "price": "18000", "chgrate": "5.0"},
    ]
    with pytest.raises(ValueError, match="coverage"):
        _run_main_with_mocks(
            monkeypatch, tmp_path, rows, scanned, frozenset({"005930", "000660"}),
            with_snapshot=True,
            call_observer=True,
        )
    assert list(Path(tmp_path / "capture" / "decision").rglob("input.parquet")) != []


def test_main_publishes_certified_inputs_before_legacy_finalization(tmp_path, monkeypatch) -> None:
    """Qualified decision publication precedes coverage checks and archive writes."""
    from pathlib import Path

    import pandas as pd

    rows = [_healthy_wide_row("005930"), _healthy_wide_row("000660")]
    scanned = [
        {"code": "005930", "name": "A", "price": "18000", "chgrate": "5.0"},
        {"code": "000660", "name": "B", "price": "18000", "chgrate": "5.0"},
    ]
    _run_main_with_mocks(
        monkeypatch, tmp_path, rows, scanned, frozenset({"005930", "000660"}),
        with_snapshot=True,
    )
    stored = list(Path(tmp_path / "capture" / "decision").rglob("input.parquet"))
    assert stored != []
    frame = pd.read_parquet(stored[0])
    assert set(frame["종목코드"].astype(str)) == {"005930", "000660"}
    assert "feature_available_timestamp" in frame.columns


def test_main_legacy_mode_skips_certified_capture(tmp_path, monkeypatch) -> None:
    """Explicit legacy mode retains existing behavior without certified captures."""
    from src.daily import collect

    monkeypatch.setattr(collect.settings, "COLLECTION_RAW_ENABLED", False)
    rows = [_healthy_wide_row("005930"), _healthy_wide_row("000660")]
    scanned = [
        {"code": "005930", "name": "A", "price": "18000", "chgrate": "5.0"},
        {"code": "000660", "name": "B", "price": "18000", "chgrate": "5.0"},
    ]
    _run_main_with_mocks(monkeypatch, tmp_path, rows, scanned, frozenset({"005930", "000660"}))
    assert not (tmp_path / "capture").exists()


def test_decision_publication_retains_admission_failures(tmp_path) -> None:
    """Admission failure stays in research with values and admitted=False."""
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.data.capture_contracts import CoverageEntry, CaptureDataset, CaptureStatus, build_cohort
    from src.data.capture_store import CaptureStore

    trading_day = date(2026, 9, 14)
    cohort = build_cohort(
        trading_day, ["000001", "000002"], ["000001", "000002"], {},
        eligibility_rule_version="price_history_panel@v1",
    )
    completed_at = datetime(2026, 9, 14, 15, 20, 30, tzinfo=ZoneInfo("Asia/Seoul"))
    frame = pd.DataFrame([
        {"종목코드": "000001", "종가": 18000.0, "admitted": True,
         "snapshot_timestamp": pd.Timestamp("2026-09-14 15:20:01", tz="Asia/Seoul"),
         "feature_available_timestamp": completed_at},
        {"종목코드": "000002", "종가": 30000.0, "admitted": False,
         "snapshot_timestamp": pd.Timestamp("2026-09-14 15:20:02", tz="Asia/Seoul"),
         "feature_available_timestamp": completed_at},
    ])
    store = CaptureStore(tmp_path / "capture")
    entries = (CoverageEntry(
        symbol=None, dataset=CaptureDataset.PRICE, venue="KRX", session="regular",
        scheduled_at=None, status=CaptureStatus.COMPLETE, rows=2,
        first_event_time=None, last_event_time=None, reason="decision-input", raw_refs=(),
    ),)
    store.publish_decision(frame, cohort=cohort, run_id="run-admit", completed_at=completed_at, entries=entries)
    replayed = store.read_decision("2026-09-14", available_by=datetime(2026, 9, 14, 15, 21, 0, tzinfo=ZoneInfo("Asia/Seoul")))
    rejected = replayed[replayed["종목코드"] == "000002"].iloc[0]
    assert bool(rejected["admitted"]) is False
    assert float(rejected["종가"]) == 30000.0


def test_scan_prefilter_evidence_survives_without_widening() -> None:
    """Raw prefilter evidence survives without widening trading picks."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily.universe_scan import fetch_candidate_stock_list

    seen = []

    class _Kiwoom:
        async def get_fluctuation_ranking(self, session, *, rate_min_pct, rate_max_pct, on_page=None):
            raw = [
                {"stk_cd": "000001", "stk_nm": "IN", "cur_prc": "18000", "flu_rt": "5.0"},
                {"stk_cd": "999999", "stk_nm": "OUT", "cur_prc": "1000", "flu_rt": "50.0"},
            ]
            if on_page is not None:
                from datetime import datetime
                from zoneinfo import ZoneInfo

                now = datetime.now(ZoneInfo("Asia/Seoul"))
                on_page({"output": raw}, {"vendor": "kiwoom"}, now, now, 0, 0)
            filtered = [r for r in raw if float(r["flu_rt"]) < 10.0]
            return {"rt_cd": "0", "output": filtered}

    def _observer(payload, metadata, started, received, page, attempt):
        seen.append(payload)

    out = asyncio.run(fetch_candidate_stock_list(AsyncMock(), object(), kiwoom_client=_Kiwoom(), on_page=_observer))
    assert [row["code"] for row in out] == ["000001"]
    assert seen and len(seen[0]["output"]) == 2


def test_narrow_toss_fallback_scope_is_explicit() -> None:
    """Toss fallback records its narrower top-100 source scope."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily.universe_scan import fetch_candidate_stock_list

    scopes = []

    class _FailingKiwoom:
        async def get_fluctuation_ranking(self, session, **kwargs):
            return {"rt_cd": "1", "msg1": "down", "output": []}

    class _Toss:
        async def get_rankings(self, session, **kwargs):
            return {"result": {"rankings": [
                {"symbol": "000001", "price": {"lastPrice": "18000", "changeRate": "0.05"}},
            ]}}

    def _observer(payload, metadata, started, received, page, attempt):
        scopes.append(dict(metadata))

    out = asyncio.run(
        fetch_candidate_stock_list(AsyncMock(), object(), kiwoom_client=_FailingKiwoom(), toss_client=_Toss(), on_page=_observer)
    )
    assert [row["code"] for row in out] == ["000001"]
    assert any(scope.get("scope") == "top100" for scope in scopes)


def test_kis_band_subquery_evidence_uses_actual_clocks() -> None:
    """KIS band sub-query responses are recorded with actual receive times."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily.universe_scan import fetch_kis_band_ranking

    stamps = []

    class _Kis:
        async def get_fluctuation_ranking(self, session, *, rate_min_pct, rate_max_pct, market_div_code=None):
            return {"rt_cd": "0", "output": [
                {"stck_shrn_iscd": "000001", "hts_kor_isnm": "A", "stck_prpr": "18000",
                 "stck_sdpr": "17142", "prdy_ctrt": "5.0", "acml_vol": "100", "acml_tr_pbmn": "100000000"},
            ]}

    def _observer(payload, metadata, started, received, page, attempt):
        stamps.append((started, received, dict(metadata)))

    rows = asyncio.run(
        fetch_kis_band_ranking(_Kis(), AsyncMock(), rate_min_pct=2.0, rate_max_pct=2.01, on_page=_observer)
    )
    assert len(rows) == 1
    assert stamps and stamps[0][1] >= stamps[0][0]
    assert stamps[0][2]["vendor"] == "kis"


def test_resolve_daily_candidates_forwards_scan_observer(monkeypatch) -> None:
    """Scan observer reaches both the primary scan and the trade-value union."""
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod

    seen: dict = {}
    union_seen: dict = {}

    async def _fake_scan(client_arg, session_arg, **kwargs):
        seen.update(kwargs)
        return [{"code": "005930", "name": "A", "price": "18000", "chgrate": "5.0"}]

    async def _fake_union(session_arg, **kwargs):
        union_seen.update(kwargs)
        return [{"code": "000660", "name": None, "price": "180000", "chgrate": "0.01"}]

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _fake_scan)
    monkeypatch.setattr(collect_mod, "fetch_trade_value_union", _fake_union)
    sentinel = object()

    out = asyncio.run(collect_mod.resolve_daily_candidates(AsyncMock(), object(), toss_client=AsyncMock(), on_page=sentinel))

    assert [row["code"] for row in out] == ["005930", "000660"]
    assert seen.get("on_page") is sentinel
    assert union_seen.get("on_page") is sentinel


def test_resolve_eligible_codes_narrows_to_screenable_subset(monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import collect

    async def _kis(_client, _session, _date):
        return True

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis)
    monkeypatch.setattr(
        collect, "load_eligible_codes", lambda decision_date, *, prev_trading_day, path=None: frozenset({"A", "B", "C"})
    )
    monkeypatch.setattr(
        collect, "load_security_classification", lambda decision_date, *, prev_trading_day, path=None: frozenset({"A", "C"})
    )
    out = asyncio.run(collect.resolve_eligible_codes(object(), object(), pd.Timestamp("2026-09-14")))
    assert out == frozenset({"A", "C"})


def test_resolve_eligible_codes_fails_on_classification_coverage_gap(monkeypatch) -> None:
    import asyncio

    import pandas as pd
    import pytest

    from src.daily import collect

    async def _kis(_client, _session, _date):
        return True

    def _stale(decision_date, *, prev_trading_day, path=None):
        raise ValueError("stale security_classification: no rows on prev_trading_day=2026-09-11")

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis)
    monkeypatch.setattr(
        collect, "load_eligible_codes", lambda decision_date, *, prev_trading_day, path=None: frozenset({"A"})
    )
    monkeypatch.setattr(collect, "load_security_classification", _stale)
    with pytest.raises(ValueError, match="no rows"):
        asyncio.run(collect.resolve_eligible_codes(object(), object(), pd.Timestamp("2026-09-14")))


def test_resolve_eligible_codes_fails_on_missing_classification_panel(monkeypatch) -> None:
    import asyncio

    import pandas as pd
    import pytest

    from src.daily import collect

    async def _kis(_client, _session, _date):
        return True

    def _missing(decision_date, *, prev_trading_day, path=None):
        raise FileNotFoundError("security_classification not found")

    monkeypatch.setattr(collect, "is_kis_trading_day", _kis)
    monkeypatch.setattr(
        collect, "load_eligible_codes", lambda decision_date, *, prev_trading_day, path=None: frozenset({"A"})
    )
    monkeypatch.setattr(collect, "load_security_classification", _missing)
    with pytest.raises(FileNotFoundError, match="security_classification"):
        asyncio.run(collect.resolve_eligible_codes(object(), object(), pd.Timestamp("2026-09-14")))


def _coverage_frame(n_raw: int, n_degraded: int):
    import pandas as pd

    failed = [True] * n_degraded + [False] * (n_raw - n_degraded)
    return pd.DataFrame({"현재가_실패": failed, "가격_비정상": [False] * n_raw})


def test_evaluate_realtime_coverage_reports_partial_on_shortfall() -> None:
    from src.daily.collect import evaluate_realtime_coverage
    from src.data.capture_contracts import CaptureStatus

    report, status, reason = evaluate_realtime_coverage(_coverage_frame(100, 2))

    assert report == {"n_raw": 100, "n_degraded": 2, "coverage": 0.98}
    assert status is CaptureStatus.PARTIAL
    assert reason == "coverage_below_threshold:0.9800"


def test_evaluate_realtime_coverage_boundary() -> None:
    import pandas as pd

    from src.daily.collect import evaluate_realtime_coverage
    from src.data.capture_contracts import CaptureStatus

    n_raw = 10000
    exact = pd.DataFrame({
        "현재가_실패": [True] * 100 + [False] * (n_raw - 100),
        "가격_비정상": [False] * n_raw,
    })
    _report, status, _reason = evaluate_realtime_coverage(exact)
    assert status is CaptureStatus.COMPLETE

    just_below = pd.DataFrame({
        "현재가_실패": [True] * 101 + [False] * (n_raw - 101),
        "가격_비정상": [False] * n_raw,
    })
    _report2, status2, reason2 = evaluate_realtime_coverage(just_below)
    assert status2 is CaptureStatus.PARTIAL
    assert reason2.startswith("coverage_below_threshold:")


def test_evaluate_realtime_coverage_rejects_empty_snapshot() -> None:
    import pandas as pd
    import pytest

    from src.daily.collect import evaluate_realtime_coverage

    with pytest.raises(ValueError, match="empty snapshot"):
        evaluate_realtime_coverage(pd.DataFrame({"현재가_실패": [], "가격_비정상": []}))


def test_check_realtime_collection_coverage_contract_preserved() -> None:
    import pytest

    from src.daily.collect import check_realtime_collection_coverage

    with pytest.raises(ValueError, match="real-time collection coverage"):
        check_realtime_collection_coverage(_coverage_frame(2, 1))


def _run_main_with_quote_failures(monkeypatch, tmp_path, n_degraded: int):
    import asyncio
    from unittest.mock import AsyncMock

    import pandas as pd

    from src.daily import collect
    from src.processing.schema import QUOTE_FAILED_COL

    n_raw = 100
    codes = [f"{i:06d}" for i in range(1, n_raw + 1)]
    stock_list = [
        {"code": c, "name": f"N{c}", "price": "18000", "chgrate": "1.0"} for c in codes
    ]

    class _FakeSession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def ensure_token(self, _s):
            return None

        async def get_market_index_rate(self, _s, _code):
            return {"rt_cd": "0", "output1": {"prdy_ctrt": "0.5"}}

    import aiohttp as _aio

    monkeypatch.setattr(_aio, "ClientSession", lambda *a, **k: _FakeSession())
    monkeypatch.setattr(collect, "KisApiClient", _FakeClient)
    monkeypatch.setattr(collect, "kis_data_client_kwargs", lambda: {
        "app_key": "k", "app_secret": "s", "account_id": "a", "hts_id": "h", "token_file": str(tmp_path / "t.json"),
    })
    monkeypatch.setattr(collect, "kis_decision_shard_client_kwargs", lambda: [{}])
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", lambda: None)
    monkeypatch.setattr(collect, "build_toss_scan_client", lambda: None)
    monkeypatch.setattr(collect, "_validate_trading_day", AsyncMock(return_value=None))
    monkeypatch.setattr(collect, "resolve_daily_candidates", AsyncMock(return_value=list(stock_list)))
    monkeypatch.setattr(
        collect, "resolve_eligible_codes", AsyncMock(return_value=frozenset(codes))
    )

    async def _fake_fetch(stock_list_arg, *a, **k):
        rows = []
        for i, s in enumerate(stock_list_arg):
            rows.append({
                "종목명": s["name"],
                "종목코드": s["code"],
                "시장구분": "KOSPI",
                "시가": 17900,
                "고가": 18100,
                "저가": 17800,
                "종가": 18000,
                "전일종가": 17800,
                "거래량": 1000000,
                "거래대금": 500.0,
                "시가총액": 3000.0,
                "기관_순매수": 10.0,
                "외국인_순매수": 5.0,
                "등락률": 1.0,
                "수급_실패": False,
                QUOTE_FAILED_COL: i < n_degraded,
            })
        return rows, []

    monkeypatch.setattr(collect, "fetch_all_stock_data_sharded", _fake_fetch)
    monkeypatch.setattr(
        collect, "flag_cost_aware_admission", lambda df, **k: df.assign(admitted=True)
    )
    monkeypatch.setattr(collect.settings, "COLLECTION_RAW_ENABLED", True)
    monkeypatch.setattr(collect.settings, "COLLECTION_ROOT", tmp_path / "capture")
    persisted = {"called": False}

    def _fake_persist(df, snapshot_date):
        persisted["called"] = True
        return len(df)

    monkeypatch.setattr(collect, "persist_daily_snapshot", _fake_persist)
    return persisted


def test_collect_main_publishes_partial_and_raises_without_archive(monkeypatch, tmp_path) -> None:
    import asyncio

    import pytest

    from src.daily import collect
    from src.data.capture_contracts import CaptureStatus
    from src.data.capture_store import CaptureStore

    persisted = _run_main_with_quote_failures(monkeypatch, tmp_path, 2)

    with pytest.raises(ValueError, match="coverage_below_threshold"):
        asyncio.run(collect.main(force=True))

    assert persisted["called"] is False
    manifests = CaptureStore(tmp_path / "capture").read_manifests(
        __import__("datetime").datetime.now(__import__("zoneinfo").ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    )
    assert len(manifests) == 1
    assert manifests[0].status is CaptureStatus.PARTIAL
    assert len(manifests[0].entries) == 1
    assert manifests[0].entries[0].status is CaptureStatus.PARTIAL
    assert manifests[0].entries[0].reason.startswith("coverage_below_threshold")


def test_collect_main_publishes_complete_then_persists(monkeypatch, tmp_path) -> None:
    import asyncio

    from src.daily import collect
    from src.data.capture_contracts import CaptureStatus
    from src.data.capture_store import CaptureStore

    persisted = _run_main_with_quote_failures(monkeypatch, tmp_path, 0)

    asyncio.run(collect.main(force=True))

    assert persisted["called"] is True
    manifests = CaptureStore(tmp_path / "capture").read_manifests(
        __import__("datetime").datetime.now(__import__("zoneinfo").ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    )
    assert len(manifests) == 1
    assert manifests[0].status is CaptureStatus.COMPLETE


def test_full_coverage_publishes_complete_entry() -> None:
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry
    from src.daily import collect

    df = _coverage_frame(100, 0)
    report, status, reason = collect.evaluate_realtime_coverage(df)
    assert status is CaptureStatus.COMPLETE
    entry = CoverageEntry(
        symbol=None,
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=status,
        rows=len(df),
        first_event_time=None,
        last_event_time=None,
        reason="decision-input" if status is CaptureStatus.COMPLETE else reason,
        raw_refs=(),
    )
    assert entry.status is CaptureStatus.COMPLETE
    assert entry.reason == "decision-input"
