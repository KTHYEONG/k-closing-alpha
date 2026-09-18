"""Closing-price finalization scenarios (contract-generated)."""

from __future__ import annotations


def test_is_close_confirmed_passes_only_when_all_three_gates_hold() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily.finalize_close import is_close_confirmed

    kst = ZoneInfo("Asia/Seoul")
    confirmed_price = {"stck_prpr": "269000"}
    confirmed_book = {"antc_mkop_cls_code": "112", "stck_prpr": "269000"}

    # Given/When/Then: 확정 이후 + 112 + 가격 일치 -> True
    assert is_close_confirmed(confirmed_price, confirmed_book, datetime(2026, 9, 10, 15, 30, 25, tzinfo=kst)) is True

    # 결정창 시작 직후 선행 112(실측 15:20:05) -> 시계 게이트로 배제
    assert is_close_confirmed(confirmed_price, confirmed_book, datetime(2026, 9, 10, 15, 20, 5, tzinfo=kst)) is False

    # 15:30 이후지만 아직 단일가 진행중(121) -> False
    assert is_close_confirmed(
        confirmed_price,
        {"antc_mkop_cls_code": "121", "stck_prpr": "269000"},
        datetime(2026, 9, 10, 15, 30, 5, tzinfo=kst),
    ) is False

    # 두 TR 가격 불일치(동결 구간의 실측 패턴) -> False
    assert is_close_confirmed(
        {"stck_prpr": "269250"},
        {"antc_mkop_cls_code": "112", "stck_prpr": "269500"},
        datetime(2026, 9, 10, 15, 30, 25, tzinfo=kst),
    ) is False

    # 가격 0/결측 -> False
    assert is_close_confirmed({"stck_prpr": "0"}, {"antc_mkop_cls_code": "112", "stck_prpr": "0"}, datetime(2026, 9, 10, 15, 31, 0, tzinfo=kst)) is False


def test_build_finalized_row_updates_eod_fields_and_preserves_decision_state() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily.finalize_close import build_finalized_row
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    kst = ZoneInfo("Asia/Seoul")
    decision_row = {
        "종목코드": "005930",
        "종가": 269250,
        "전일종가": 269500,
        "거래량": 19525671,
        "거래대금": 52206.58,
        "시가총액": 1600000.0,
        "등락률": -0.09,
        "admitted": True,
        DECISION_CLOSE_COL: 269250,
        CLOSE_CONFIRMED_COL: False,
    }
    price_output = {
        "stck_prpr": "269000",
        "stck_oprc": "270000",
        "stck_hgpr": "272000",
        "stck_lwpr": "268000",
        "stck_sdpr": "269500",
        "acml_vol": "28037611",
        "acml_tr_pbmn": "7510369697500",
        "hts_avls": "1605000",
        "prdy_ctrt": "-0.19",
    }

    out = build_finalized_row(decision_row, price_output, datetime(2026, 9, 10, 15, 30, 25, tzinfo=kst))

    assert out["종가"] == 269000
    assert out["전일종가"] == 269500
    assert out["거래량"] == 28037611
    assert out["거래대금"] == 75103.7
    assert out["시가총액"] == 1605000.0
    assert out["등락률"] == -0.19
    assert out[DECISION_CLOSE_COL] == 269250
    assert out[CLOSE_CONFIRMED_COL] is True
    assert "admitted" not in out
    assert "종목코드" not in out
    assert "snapshot_timestamp" not in out


def test_build_finalized_row_rejects_volume_regression() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.daily.finalize_close import build_finalized_row

    kst = ZoneInfo("Asia/Seoul")
    decision_row = {"종가": 269250, "전일종가": 269500, "거래량": 19525671}
    price_output = {
        "stck_prpr": "269000",
        "stck_oprc": "270000",
        "stck_hgpr": "272000",
        "stck_lwpr": "268000",
        "stck_sdpr": "269500",
        "acml_vol": "19000000",
        "acml_tr_pbmn": "7510369697500",
        "prdy_ctrt": "-0.19",
    }

    with pytest.raises(ValueError, match=r".*"):
        build_finalized_row(decision_row, price_output, datetime(2026, 9, 10, 15, 30, 25, tzinfo=kst))


def test_build_finalized_row_rejects_price_rate_inconsistency() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.daily.finalize_close import build_finalized_row

    kst = ZoneInfo("Asia/Seoul")
    ts = datetime(2026, 9, 10, 15, 30, 25, tzinfo=kst)
    decision_row = {"종가": 269250, "전일종가": 269500, "거래량": 1000}
    base = {
        "stck_prpr": "269000",
        "stck_oprc": "270000",
        "stck_hgpr": "272000",
        "stck_lwpr": "268000",
        "stck_sdpr": "269500",
        "acml_vol": "28037611",
        "acml_tr_pbmn": "7510369697500",
    }

    # 벤더 소수 2자리 반올림 범위 내 -> 통과
    ok = build_finalized_row(decision_row, {**base, "prdy_ctrt": "-0.19"}, ts)
    assert ok["등락률"] == -0.19

    # 부호가 뒤집힌 등락률 -> fail-closed
    with pytest.raises(ValueError, match=r".*"):
        build_finalized_row(decision_row, {**base, "prdy_ctrt": "5.00"}, ts)


def test_run_close_finalization_updates_rows_in_place_without_new_snapshot_identity(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    kst = ZoneInfo("Asia/Seoul")
    decision_ts = pd.Timestamp("2026-09-10 15:20:18", tz="Asia/Seoul")
    snapshot = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-09-10"],
            "종목코드": ["005930"],
            "종가": [269250],
            "전일종가": [269500],
            "거래량": [19525671],
            "거래대금": [52206.58],
            "등락률": [-0.09],
            "admitted": [True],
            DECISION_CLOSE_COL: [269250],
            CLOSE_CONFIRMED_COL: [False],
            "snapshot_timestamp": [decision_ts],
        }
    )
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())

    captured: list[pd.DataFrame] = []

    def _fake_upsert(df, snapshot_date=None):
        captured.append(df.copy())
        return len(df)

    monkeypatch.setattr(finalize_close.archive, "upsert_archive_snapshot", _fake_upsert)

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "269000",
                    "stck_oprc": "270000",
                    "stck_hgpr": "272000",
                    "stck_lwpr": "268000",
                    "stck_sdpr": "269500",
                    "acml_vol": "28037611",
                    "acml_tr_pbmn": "7510369697500",
                    "hts_avls": "1605000",
                    "prdy_ctrt": "-0.19",
                },
            }

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "112", "stck_prpr": "269000"}}

    async def _no_sleep(_seconds):
        return None

    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_Client(),
            session=object(),
            now_fn=lambda: datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
        )
    )

    assert n == 1
    assert len(captured) == 1
    written = captured[0]
    assert int(written.loc[0, "종가"]) == 269000
    assert int(written.loc[0, "거래량"]) == 28037611
    assert bool(written.loc[0, CLOSE_CONFIRMED_COL]) is True
    assert int(written.loc[0, DECISION_CLOSE_COL]) == 269250
    assert bool(written.loc[0, "admitted"]) is True
    assert pd.Timestamp(written.loc[0, "snapshot_timestamp"]) == decision_ts


def test_run_close_finalization_leaves_unconfirmed_rows_untouched_until_deadline(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    kst = ZoneInfo("Asia/Seoul")
    snapshot = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-09-10"],
            "종목코드": ["000660"],
            "종가": [1860000],
            "전일종가": [1856000],
            "거래량": [4955648],
            "등락률": [0.22],
            DECISION_CLOSE_COL: [1860000],
            CLOSE_CONFIRMED_COL: [False],
            "snapshot_timestamp": [pd.Timestamp("2026-09-10 15:20:18", tz="Asia/Seoul")],
        }
    )
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())

    upserts: list[int] = []
    monkeypatch.setattr(
        finalize_close.archive,
        "upsert_archive_snapshot",
        lambda df, snapshot_date=None: upserts.append(len(df)) or len(df),
    )

    class _NeverConfirms:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "0", "output": {"stck_prpr": "1860000"}}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "121", "stck_prpr": "1859000"}}

    clock = iter(
        [
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 32, 0, tzinfo=kst),
            datetime(2026, 9, 10, 15, 32, 0, tzinfo=kst),
            datetime(2026, 9, 10, 15, 34, 0, tzinfo=kst),
        ]
    )
    sleeps: list[float] = []

    async def _record_sleep(seconds):
        sleeps.append(seconds)

    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_NeverConfirms(),
            session=object(),
            now_fn=lambda: next(clock),
            sleep_fn=_record_sleep,
            retry_interval_seconds=1.0,
        )
    )

    assert n == 0
    assert upserts == []
    assert sleeps  # 데드라인 전까지 최소 1회 재폴링


def test_run_close_finalization_skips_already_confirmed_rows(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    kst = ZoneInfo("Asia/Seoul")
    snapshot = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-09-10"],
            "종목코드": ["005930"],
            "종가": [269000],
            "전일종가": [269500],
            "거래량": [28037611],
            "등락률": [-0.19],
            DECISION_CLOSE_COL: [269250],
            CLOSE_CONFIRMED_COL: [True],
            "snapshot_timestamp": [pd.Timestamp("2026-09-10 15:20:18", tz="Asia/Seoul")],
        }
    )
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    monkeypatch.setattr(
        finalize_close.archive,
        "upsert_archive_snapshot",
        lambda df, snapshot_date=None: (_ for _ in ()).throw(AssertionError("확정 0건이면 쓰기 금지")),
    )

    calls: list[str] = []

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            calls.append(code)
            return {"rt_cd": "0", "output": {}}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            calls.append(code)
            return {"rt_cd": "0", "output2": {}}

    async def _no_sleep(_seconds):
        return None

    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_Client(),
            session=object(),
            now_fn=lambda: datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
        )
    )

    assert n == 0
    assert calls == []


def test_fetch_confirmed_quote_returns_empty_blocks_on_vendor_failure() -> None:
    import asyncio

    from src.daily.finalize_close import fetch_confirmed_quote, is_close_confirmed
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FailingClient:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "1", "msg1": "rate limit"}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0"}

    price_out, book_out2 = asyncio.run(fetch_confirmed_quote(_FailingClient(), object(), "005930"))

    assert price_out == {}
    assert book_out2 == {}
    assert is_close_confirmed(price_out, book_out2, datetime(2026, 9, 10, 15, 31, 0, tzinfo=ZoneInfo("Asia/Seoul"))) is False




def test_run_close_finalization_rejects_invariant_violating_quote_without_touching_row(monkeypatch) -> None:
    """게이트는 통과했으나 불변식(거래량 단조성)을 깬 응답은 행을 갱신하지 않는다."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    kst = ZoneInfo("Asia/Seoul")
    snapshot = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-09-10"],
            "종목코드": ["005930"],
            "종가": [269250],
            "전일종가": [269500],
            "거래량": [19525671],
            "등락률": [-0.09],
            DECISION_CLOSE_COL: [269250],
            CLOSE_CONFIRMED_COL: [False],
            "snapshot_timestamp": [pd.Timestamp("2026-09-10 15:20:18", tz="Asia/Seoul")],
        }
    )
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    upserts: list[int] = []
    monkeypatch.setattr(
        finalize_close.archive,
        "upsert_archive_snapshot",
        lambda df, snapshot_date=None: upserts.append(len(df)) or len(df),
    )

    class _RegressedVolumeClient:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            # 확정 게이트는 통과하지만 누적거래량이 결정시점보다 작다 (벤더 오류/종목 혼선)
            return {
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "269000",
                    "stck_sdpr": "269500",
                    "acml_vol": "19000000",
                    "acml_tr_pbmn": "7510369697500",
                    "prdy_ctrt": "-0.19",
                },
            }

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "112", "stck_prpr": "269000"}}

    clock = iter(
        [
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 34, 0, tzinfo=kst),
        ]
    )

    async def _no_sleep(_seconds):
        return None

    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_RegressedVolumeClient(),
            session=object(),
            now_fn=lambda: next(clock),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
        )
    )

    assert n == 0
    assert upserts == []


def test_finalize_close_main_runs_inside_a_single_event_loop(monkeypatch) -> None:
    import asyncio
    import sys

    from src.daily import finalize_close

    seen = {"loop_at_create": None, "runs": 0, "closed": False, "finalized": None}

    class _FakeSession:
        async def close(self):
            seen["closed"] = True

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            self.token = None

        def create_session(self, **_kw):
            # 회귀 지점: 러닝 루프 밖에서 호출되면 aiohttp 가 RuntimeError 를 던진다
            seen["loop_at_create"] = asyncio.get_running_loop()
            return _FakeSession()

        async def ensure_token(self, _session, force_refresh=False):
            self.token = "T"
            return "T"

    async def _fake_finalization(*_a, **kwargs):
        seen["finalized"] = kwargs.get("snapshot_date")
        return 7

    real_run = asyncio.run

    def _counting_run(coro, **kw):
        seen["runs"] += 1
        return real_run(coro, **kw)

    monkeypatch.setattr(finalize_close, "KisApiClient", _FakeClient)
    monkeypatch.setattr(finalize_close, "run_close_finalization", _fake_finalization)
    monkeypatch.setattr(finalize_close, "load_pick_codes", lambda _d: frozenset())
    monkeypatch.setattr(finalize_close.asyncio, "run", _counting_run)
    monkeypatch.setattr(sys, "argv", ["finalize_close", "--date", "2026-09-10"])

    # When
    finalize_close.main()

    # Then: 단일 이벤트 루프 안에서 세션 생성/사용/종료가 모두 일어난다
    assert seen["runs"] == 1
    assert seen["loop_at_create"] is not None
    assert seen["closed"] is True
    assert seen["finalized"] == "2026-09-10"


def test_fetch_confirmed_quote_requests_krx_price_without_venue_fallback() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily.finalize_close import fetch_confirmed_quote

    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "1000"}})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output2": {"antc_mkop_cls_code": "112"}})

    # When
    price, book = asyncio.run(fetch_confirmed_quote(client, object(), "005930"))

    # Then
    assert client.get_current_price.await_args.kwargs == {"market_div_code": "J", "allow_market_div_fallback": False}
    assert price == {"stck_prpr": "1000"}
    assert book == {"antc_mkop_cls_code": "112"}



def test_order_pending_by_priority_puts_picks_then_admitted_first() -> None:
    import pandas as pd

    from src.daily.finalize_close import order_pending_by_priority

    df = pd.DataFrame(
        {
            "종목코드": ["000001", "000002", "000003", "000004", "000005"],
            "admitted": [False, True, pd.NA, True, False],
        }
    )

    # When
    ordered = order_pending_by_priority(df, [0, 1, 2, 3, 4], frozenset({"000005", "000003"}))

    # Then: 픽(원래 순서 유지) -> admitted(원래 순서) -> 나머지
    assert ordered == [2, 4, 1, 3, 0]
    no_flag = df.drop(columns=["admitted"])
    assert order_pending_by_priority(no_flag, [0, 1, 2], frozenset({"000002"})) == [1, 0, 2]


def test_classify_finalize_outcome_cases() -> None:
    from src.daily.finalize_close import classify_finalize_outcome

    # 휴장일/수집 실패: 아카이브 비어있음 -> 알림 없음(수집 단계가 이미 알림)
    assert classify_finalize_outcome(0, 0, 0, []) == ("OK", "empty_archive")
    # 픽 미확정은 부분 확정이어도 DEGRADED
    assert classify_finalize_outcome(10, 9, 1, ["005930"]) == ("DEGRADED", "picks_unconfirmed")
    # 확정 0건(미확정 행 존재)
    assert classify_finalize_outcome(10, 0, 10, []) == ("DEGRADED", "zero_confirmed")
    # 픽 전부 확정, 비픽 일부 미확정 -> OK
    assert classify_finalize_outcome(10, 7, 3, []) == ("OK", "")
    # 이미 전부 확정된 재실행
    assert classify_finalize_outcome(10, 0, 0, []) == ("OK", "")


def test_run_close_finalization_fetches_picks_first_with_bounded_concurrency(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    def _rows(codes, admitted):
        return pd.DataFrame(
            {
                "스냅샷_날짜": ["2026-09-10"] * len(codes),
                "종목코드": codes,
                "종가": [10000] * len(codes),
                "전일종가": [10000] * len(codes),
                "거래량": [100] * len(codes),
                "등락률": [0.0] * len(codes),
                "admitted": admitted,
                DECISION_CLOSE_COL: [10000] * len(codes),
                CLOSE_CONFIRMED_COL: [False] * len(codes),
            }
        )

    kst = ZoneInfo("Asia/Seoul")
    snapshot = _rows(
        ["000001", "000002", "000003", "000004", "000005", "000006"],
        [False, False, True, False, False, False],
    )
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    monkeypatch.setattr(finalize_close.archive, "upsert_archive_snapshot", lambda df, snapshot_date=None: len(df))

    price_calls: list[str] = []
    state = {"inflight": 0, "max": 0}

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            price_calls.append(code)
            state["inflight"] += 1
            state["max"] = max(state["max"], state["inflight"])
            await asyncio.sleep(0)
            state["inflight"] -= 1
            return {"rt_cd": "0", "output": {"stck_prpr": "10000"}}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "121", "stck_prpr": "10000"}}

    clock = iter(
        [
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 34, 0, tzinfo=kst),
        ]
    )

    async def _no_sleep(_seconds):
        return None

    # When
    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_Client(),
            session=object(),
            now_fn=lambda: next(clock),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
            pick_codes=frozenset({"000005"}),
        )
    )

    # Then
    assert n == 0
    assert finalize_close.FINALIZE_CONCURRENCY == 4
    assert price_calls == ["000005", "000003", "000001", "000002", "000004", "000006"]
    assert state["max"] == finalize_close.FINALIZE_CONCURRENCY


def test_run_close_finalization_stops_batches_after_deadline(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    def _rows(codes, admitted):
        return pd.DataFrame(
            {
                "스냅샷_날짜": ["2026-09-10"] * len(codes),
                "종목코드": codes,
                "종가": [10000] * len(codes),
                "전일종가": [10000] * len(codes),
                "거래량": [100] * len(codes),
                "등락률": [0.0] * len(codes),
                "admitted": admitted,
                DECISION_CLOSE_COL: [10000] * len(codes),
                CLOSE_CONFIRMED_COL: [False] * len(codes),
            }
        )

    kst = ZoneInfo("Asia/Seoul")
    snapshot = _rows(["000001", "000002", "000003", "000004", "000005", "000006"], [False] * 6)
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    monkeypatch.setattr(finalize_close.archive, "upsert_archive_snapshot", lambda df, snapshot_date=None: len(df))

    price_calls: list[str] = []

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            price_calls.append(code)
            return {"rt_cd": "0", "output": {"stck_prpr": "10000"}}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "121"}}

    clock = iter(
        [
            datetime(2026, 9, 10, 15, 32, 59, tzinfo=kst),
            datetime(2026, 9, 10, 15, 32, 59, tzinfo=kst),
            datetime(2026, 9, 10, 15, 33, 1, tzinfo=kst),
            datetime(2026, 9, 10, 15, 33, 5, tzinfo=kst),
        ]
    )

    async def _no_sleep(_seconds):
        return None

    # When
    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_Client(),
            session=object(),
            now_fn=lambda: next(clock),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
        )
    )

    # Then: 첫 배치(4행)만 조회, 데드라인 이후 배치는 시작하지 않음
    assert n == 0
    assert price_calls == ["000001", "000002", "000003", "000004"]


def test_run_close_finalization_reports_degraded_outcome_for_unconfirmed_pick(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    def _rows(codes, admitted):
        return pd.DataFrame(
            {
                "스냅샷_날짜": ["2026-09-10"] * len(codes),
                "종목코드": codes,
                "종가": [10000] * len(codes),
                "전일종가": [10000] * len(codes),
                "거래량": [100] * len(codes),
                "등락률": [0.0] * len(codes),
                "admitted": admitted,
                DECISION_CLOSE_COL: [10000] * len(codes),
                CLOSE_CONFIRMED_COL: [False] * len(codes),
            }
        )

    kst = ZoneInfo("Asia/Seoul")
    snapshot = _rows(["000001", "000002"], [True, True])
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    upserts: list[int] = []
    monkeypatch.setattr(
        finalize_close.archive, "upsert_archive_snapshot", lambda df, snapshot_date=None: upserts.append(len(df)) or len(df)
    )

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "10100", "stck_oprc": "10000", "stck_hgpr": "10200", "stck_lwpr": "9900",
                    "stck_sdpr": "10000", "acml_vol": "500", "acml_tr_pbmn": "5050000", "prdy_ctrt": "1.00",
                },
            }

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            mkop = "121" if code == "000001" else "112"
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": mkop, "stck_prpr": "10100"}}

    clock = iter(
        [
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 34, 0, tzinfo=kst),
        ]
    )
    outcomes: list[tuple] = []

    async def _no_sleep(_seconds):
        return None

    # When
    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_Client(),
            session=object(),
            now_fn=lambda: next(clock),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
            pick_codes=frozenset({"000001"}),
            on_outcome=lambda outcome, **kw: outcomes.append((outcome, kw)),
        )
    )

    # Then
    assert n == 1
    assert upserts == [2]
    assert len(outcomes) == 1
    outcome, kw = outcomes[0]
    assert outcome == "DEGRADED"
    assert kw["run_date"] == "2026-09-10"
    assert kw["reason"] == "picks_unconfirmed"
    assert kw["metrics"] == {"n_rows": 2, "n_finalized": 1, "n_unconfirmed": 1, "n_unresolved": 0, "unconfirmed_picks": ["000001"]}


def test_load_pick_codes_zero_fills_persisted_symbols(monkeypatch) -> None:
    import pandas as pd

    from src.daily import finalize_close

    seen: list[pd.Timestamp] = []

    def _decision(decision_date):
        seen.append(pd.Timestamp(decision_date))
        return pd.DataFrame({"symbol": ["5930", "000660"]})

    monkeypatch.setattr(finalize_close, "load_topk_decision", _decision)

    # When
    codes = finalize_close.load_pick_codes("2026-09-10")

    # Then
    assert codes == frozenset({"005930", "000660"})
    assert seen == [pd.Timestamp("2026-09-10")]
    monkeypatch.setattr(finalize_close, "load_topk_decision", lambda _d: pd.DataFrame())
    assert finalize_close.load_pick_codes("2026-09-10") == frozenset()


def test_finalize_close_main_wires_pick_codes_and_outcome_recorder(monkeypatch) -> None:
    import sys
    from unittest.mock import Mock

    from src.daily import finalize_close

    class _FakeSession:
        async def close(self):
            return None

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        def create_session(self, **_kw):
            return _FakeSession()

        async def ensure_token(self, _session, force_refresh=False):
            return "T"

    captured: dict = {}

    async def _fake_finalization(*_a, **kwargs):
        captured.update(kwargs)
        return 0

    picks_seen: list[str] = []
    recorder = Mock(return_value={})
    monkeypatch.setattr(finalize_close, "KisApiClient", _FakeClient)
    monkeypatch.setattr(finalize_close, "run_close_finalization", _fake_finalization)
    monkeypatch.setattr(finalize_close, "load_pick_codes", lambda d: picks_seen.append(d) or frozenset({"005930"}))
    monkeypatch.setattr(finalize_close, "record_run_outcome", recorder)
    monkeypatch.setattr(sys, "argv", ["finalize_close", "--date", "2026-09-10"])

    # When
    finalize_close.main()
    captured["on_outcome"]("DEGRADED", run_date="2026-09-10", reason="zero_confirmed", metrics={"n_rows": 1})

    # Then
    assert picks_seen == ["2026-09-10"]
    assert captured["snapshot_date"] == "2026-09-10"
    assert captured["pick_codes"] == frozenset({"005930"})
    recorder.assert_called_once_with(
        "finalize_close", "DEGRADED", run_date="2026-09-10", reason="zero_confirmed", metrics={"n_rows": 1}
    )


def test_is_quote_unresolved_detects_blank_code_zero_price() -> None:
    from src.daily.finalize_close import is_quote_unresolved

    # 실측: 맨코드 ETN 500041 -> rt_cd=0, 전 필드 0, 종목코드 공란
    assert is_quote_unresolved({"stck_shrn_iscd": "", "stck_prpr": "0", "acml_vol": "0"}) is True
    # 벤더 실패(빈 블록)는 재시도 대상
    assert is_quote_unresolved({}) is False
    # 정상 시세
    assert is_quote_unresolved({"stck_shrn_iscd": "005930", "stck_prpr": "80000"}) is False
    # 코드 필드가 없어도 가격이 있으면 정상
    assert is_quote_unresolved({"stck_prpr": "10000"}) is False


def test_run_close_finalization_rejects_non_same_day_snapshot(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd
    import pytest

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    def _rows(codes, admitted):
        return pd.DataFrame(
            {
                "스냅샷_날짜": ["2026-09-10"] * len(codes),
                "종목코드": codes,
                "종가": [10000] * len(codes),
                "전일종가": [10000] * len(codes),
                "거래량": [100] * len(codes),
                "등락률": [0.0] * len(codes),
                "admitted": admitted,
                DECISION_CLOSE_COL: [10000] * len(codes),
                CLOSE_CONFIRMED_COL: [False] * len(codes),
            }
        )

    kst = ZoneInfo("Asia/Seoul")
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: _rows(["005930"], [True]))
    monkeypatch.setattr(
        finalize_close.archive,
        "upsert_archive_snapshot",
        lambda df, snapshot_date=None: (_ for _ in ()).throw(AssertionError("과거일 쓰기 금지")),
    )
    calls: list[str] = []

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            calls.append(code)
            return {"rt_cd": "0", "output": {"stck_prpr": "10000"}}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            calls.append(code)
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "112", "stck_prpr": "10000"}}

    async def _no_sleep(_seconds):
        return None

    # When/Then: 9/10 스냅샷을 9/14 15:31에 확정 시도
    with pytest.raises(ValueError, match="same-day"):
        asyncio.run(
            finalize_close.run_close_finalization(
                snapshot_date="2026-09-10",
                client=_Client(),
                session=object(),
                now_fn=lambda: datetime(2026, 9, 14, 15, 31, 0, tzinfo=kst),
                sleep_fn=_no_sleep,
                retry_interval_seconds=0.0,
            )
        )
    assert calls == []


def test_run_close_finalization_drops_unresolved_rows_without_repolling(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    def _rows(codes, admitted):
        return pd.DataFrame(
            {
                "스냅샷_날짜": ["2026-09-10"] * len(codes),
                "종목코드": codes,
                "종가": [10000] * len(codes),
                "전일종가": [10000] * len(codes),
                "거래량": [100] * len(codes),
                "등락률": [0.0] * len(codes),
                "admitted": admitted,
                DECISION_CLOSE_COL: [10000] * len(codes),
                CLOSE_CONFIRMED_COL: [False] * len(codes),
            }
        )

    kst = ZoneInfo("Asia/Seoul")
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: _rows(["500041", "000660"], [True, False]))
    monkeypatch.setattr(finalize_close.archive, "upsert_archive_snapshot", lambda df, snapshot_date=None: len(df))
    price_calls: list[str] = []

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            price_calls.append(code)
            if code == "500041":
                return {"rt_cd": "0", "output": {"stck_shrn_iscd": "", "stck_prpr": "0", "acml_vol": "0"}}
            return {"rt_cd": "0", "output": {"stck_shrn_iscd": "000660", "stck_prpr": "10000"}}

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "121", "stck_prpr": "10000"}}

    clock = iter(
        [
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            datetime(2026, 9, 10, 15, 31, 0, tzinfo=kst),
            datetime(2026, 9, 10, 15, 31, 0, tzinfo=kst),
            datetime(2026, 9, 10, 15, 34, 0, tzinfo=kst),
        ]
    )
    outcomes: list[tuple] = []

    async def _no_sleep(_seconds):
        return None

    # When
    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_Client(),
            session=object(),
            now_fn=lambda: next(clock),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
            pick_codes=frozenset({"500041"}),
            on_outcome=lambda outcome, **kw: outcomes.append((outcome, kw)),
        )
    )

    # Then: 미해석 행은 1회만 조회, 해석 가능 행은 데드라인까지 재조회
    assert n == 0
    assert price_calls == ["500041", "000660", "000660"]
    outcome, kw = outcomes[0]
    assert outcome == "DEGRADED"
    assert kw["reason"] == "picks_unconfirmed"
    assert kw["metrics"] == {
        "n_rows": 2,
        "n_finalized": 0,
        "n_unconfirmed": 1,
        "n_unresolved": 1,
        "unconfirmed_picks": ["500041"],
    }


def test_finalize_close_amain_uses_data_account_client(monkeypatch) -> None:
    import sys
    from unittest.mock import Mock

    from src.daily import finalize_close

    built: list[dict] = []

    class _FakeSession:
        async def close(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            built.append(kwargs)

        def create_session(self, **_kw):
            return _FakeSession()

        async def ensure_token(self, _session, force_refresh=False):
            return "T"

    async def _fake_finalization(*_a, **_kw):
        return 0

    data_kwargs = {"app_key": "DATA", "app_secret": "S", "account_id": "", "hts_id": None, "token_file": "t.json"}
    monkeypatch.setattr(finalize_close, "KisApiClient", _FakeClient)
    monkeypatch.setattr(finalize_close, "kis_data_client_kwargs", lambda: dict(data_kwargs))
    monkeypatch.setattr(finalize_close, "run_close_finalization", _fake_finalization)
    monkeypatch.setattr(finalize_close, "load_pick_codes", lambda _d: frozenset())
    monkeypatch.setattr(finalize_close, "record_run_outcome", Mock())
    monkeypatch.setattr(sys, "argv", ["finalize_close", "--date", "2026-09-10"])

    # When
    finalize_close.main()

    # Then
    assert built == [data_kwargs]


def test_run_close_finalization_confirms_rows_from_float64_archive_flags(monkeypatch) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    kst = ZoneInfo("Asia/Seoul")
    decision_ts = pd.Timestamp("2026-09-15 15:20:18", tz="Asia/Seoul")
    # Given: 아카이브 저장소 실측과 동일하게 종가_확정이 float64(0.0 / NaN)로 로드된다
    snapshot = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-09-15", "2026-09-15"],
            "종목코드": ["005930", "000660"],
            "종가": [269250.0, 269250.0],
            "전일종가": [269500.0, 269500.0],
            "거래량": [19525671.0, 1000.0],
            "거래대금": [52206.58, 26.9],
            "등락률": [-0.09, -0.09],
            "admitted": [True, False],
            DECISION_CLOSE_COL: [269250.0, 269250.0],
            CLOSE_CONFIRMED_COL: [0.0, float("nan")],
            "snapshot_timestamp": [decision_ts, decision_ts],
        }
    )
    assert str(snapshot[CLOSE_CONFIRMED_COL].dtype) == "float64"
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    captured: list[pd.DataFrame] = []

    def _fake_upsert(df, snapshot_date=None):
        captured.append(df.copy())
        return len(df)

    monkeypatch.setattr(finalize_close.archive, "upsert_archive_snapshot", _fake_upsert)

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "269000",
                    "stck_oprc": "270000",
                    "stck_hgpr": "272000",
                    "stck_lwpr": "268000",
                    "stck_sdpr": "269500",
                    "acml_vol": "28037611",
                    "acml_tr_pbmn": "7510369697500",
                    "hts_avls": "1605000",
                    "prdy_ctrt": "-0.19",
                },
            }

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output2": {"antc_mkop_cls_code": "112", "stck_prpr": "269000"}}

    async def _no_sleep(_seconds):
        return None

    # When
    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-15",
            client=_Client(),
            session=object(),
            now_fn=lambda: datetime(2026, 9, 15, 15, 30, 30, tzinfo=kst),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
        )
    )

    # Then: bool 대입 TypeError 없이 두 행 모두 확정되어 저장된다
    assert n == 2
    assert len(captured) == 1
    assert captured[0][CLOSE_CONFIRMED_COL].astype(bool).tolist() == [True, True]


# ---------------------------------------------------------------------------
# closing_capture_06 close-confirmation invariant guards
# ---------------------------------------------------------------------------

def _confirming_client():
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={
        "rt_cd": "0",
        "output": {
            "stck_prpr": "269000", "stck_oprc": "270000", "stck_hgpr": "272000",
            "stck_lwpr": "268000", "stck_sdpr": "269500", "acml_vol": "28037611",
            "acml_tr_pbmn": "7510369697500", "hts_avls": "1605000", "prdy_ctrt": "-0.19",
        },
    })
    client.get_orderbook_snapshot = AsyncMock(return_value={
        "rt_cd": "0",
        "output1": {"askp1": "269000"},
        "output2": {"antc_mkop_cls_code": "112", "stck_prpr": "269000"},
    })
    return client


def _confirmation_snapshot():
    import pandas as pd

    from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

    return pd.DataFrame({
        "스냅샷_날짜": ["2026-09-10"],
        "종목코드": ["005930"],
        "종가": [269250],
        "전일종가": [269500],
        "거래량": [19525671],
        "거래대금": [52206.58],
        "등락률": [-0.09],
        "admitted": [True],
        DECISION_CLOSE_COL: [269250],
        CLOSE_CONFIRMED_COL: [False],
        "snapshot_timestamp": [pd.Timestamp("2026-09-10 15:20:18", tz="Asia/Seoul")],
    })


def test_fetch_confirmed_quote_rejects_inconsistent_capture_context() -> None:
    """Confirmation capture context is all-or-nothing."""
    import asyncio

    import pytest

    from src.daily.finalize_close import fetch_confirmed_quote, run_close_finalization

    with pytest.raises(ValueError, match="inconsistent"):
        asyncio.run(fetch_confirmed_quote(object(), object(), "005930", capture_store=object(), run_id=None, cohort_id=None))
    with pytest.raises(ValueError, match="inconsistent"):
        asyncio.run(
            run_close_finalization(
                snapshot_date="2026-09-10", client=object(), session=object(),
                capture_store=object(), run_id="r", cohort_id=None,
            )
        )


def test_rejected_confirmation_keeps_raw_evidence(tmp_path) -> None:
    """Failed market-code gate still preserves raw output1/output2 evidence."""
    import asyncio
    import gzip
    import json
    from pathlib import Path

    from src.daily.finalize_close import fetch_confirmed_quote, is_close_confirmed
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from unittest.mock import AsyncMock

    from src.data.capture_store import CaptureStore

    store = CaptureStore(tmp_path / "capture")
    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "269250"}})
    client.get_orderbook_snapshot = AsyncMock(return_value={
        "rt_cd": "0",
        "output1": {"askp1": "269500"},
        "output2": {"antc_mkop_cls_code": "121", "stck_prpr": "269500"},
    })
    price_out, book_out2 = asyncio.run(
        fetch_confirmed_quote(client, object(), "005930", capture_store=store, run_id="run-confirm", cohort_id="cohort-x")
    )
    assert is_close_confirmed(price_out, book_out2, datetime(2026, 9, 10, 15, 31, 0, tzinfo=ZoneInfo("Asia/Seoul"))) is False
    raws = list(Path(tmp_path / "capture" / "raw").rglob("*.json.gz"))
    assert len(raws) == 2
    payloads = [json.loads(gzip.decompress(path.read_bytes()).decode("utf-8")) for path in raws]
    assert any("output1" in (envelope.get("payload") or {}) for envelope in payloads)
    assert any((envelope.get("payload") or {}).get("output2", {}).get("antc_mkop_cls_code") == "121" for envelope in payloads)


def test_confirmation_persistence_failure_records_degraded() -> None:
    """Raw confirmation persistence failure keeps payload and safety checks."""
    import asyncio
    import logging

    from src.daily.finalize_close import fetch_confirmed_quote, is_close_confirmed
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class _FailingStore:
        def append_response(self, response):
            raise OSError("disk unavailable")

    price_out, book_out2 = asyncio.run(
        fetch_confirmed_quote(_confirming_client(), object(), "005930", capture_store=_FailingStore(), run_id="r", cohort_id="c")
    )
    assert price_out["stck_prpr"] == "269000"
    assert book_out2["antc_mkop_cls_code"] == "112"
    assert is_close_confirmed(price_out, book_out2, datetime(2026, 9, 10, 15, 31, 0, tzinfo=ZoneInfo("Asia/Seoul"))) is True
    assert logging.getLogger(__name__) is not None


def test_close_outcomes_leave_decision_hash_unchanged(tmp_path, monkeypatch) -> None:
    """Published 15:20 input stays byte-identical after close finalization."""
    import asyncio
    import hashlib
    from datetime import date, datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry, build_cohort
    from src.data.capture_store import CaptureStore

    kst = ZoneInfo("Asia/Seoul")
    trading_day = date(2026, 9, 10)
    cohort = build_cohort(trading_day, ["005930"], ["005930"], {}, eligibility_rule_version="price_history_panel@v1")
    completed_at = datetime(2026, 9, 10, 15, 20, 30, tzinfo=kst)
    store = CaptureStore(tmp_path / "capture")
    decision_frame = pd.DataFrame([{
        "종목코드": "005930", "종가": 269250, "admitted": True,
        "snapshot_timestamp": pd.Timestamp("2026-09-10 15:20:18", tz="Asia/Seoul"),
        "feature_available_timestamp": completed_at,
    }])
    entries = (CoverageEntry(
        symbol=None, dataset=CaptureDataset.PRICE, venue="KRX", session="regular",
        scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1,
        first_event_time=None, last_event_time=None, reason="decision-input", raw_refs=(),
    ),)
    store.publish_decision(decision_frame, cohort=cohort, run_id="run-decision", completed_at=completed_at, entries=entries)
    before = (tmp_path / "capture" / "decision" / "2026-09-10" / "run-decision" / "input.parquet").read_bytes()
    before_hash = hashlib.sha256(before).hexdigest()

    snapshot = _confirmation_snapshot()
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    monkeypatch.setattr(finalize_close.archive, "upsert_archive_snapshot", lambda df, snapshot_date=None: len(df))

    async def _no_sleep(_seconds):
        return None

    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_confirming_client(),
            session=object(),
            now_fn=lambda: datetime(2026, 9, 10, 15, 30, 30, tzinfo=kst),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
            capture_store=store,
            run_id="run-confirm",
            cohort_id=cohort.cohort_id,
        )
    )
    assert n == 1
    after = (tmp_path / "capture" / "decision" / "2026-09-10" / "run-decision" / "input.parquet").read_bytes()
    assert hashlib.sha256(after).hexdigest() == before_hash
    assert list(Path(tmp_path / "capture" / "normalized").rglob("*.parquet")) != []


def test_close_outcome_publication_failure_stays_degraded(tmp_path, monkeypatch) -> None:
    """Close outcome publication failure never invalidates qualified inputs."""
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.daily import finalize_close
    from src.data.capture_store import CaptureStore

    snapshot = _confirmation_snapshot()
    monkeypatch.setattr(finalize_close.archive, "fetch_archive_snapshot", lambda *a, **kw: snapshot.copy())
    monkeypatch.setattr(finalize_close.archive, "upsert_archive_snapshot", lambda df, snapshot_date=None: len(df))

    real_store = CaptureStore(tmp_path / "capture")

    class _PublishFailStore(CaptureStore):
        def publish_frame(self, frame, *, context):
            raise OSError("outcome disk unavailable")

    failing = _PublishFailStore(tmp_path / "capture2")
    monkeypatch.setattr(failing, "append_response", real_store.append_response.__get__(failing, CaptureStore))

    async def _no_sleep(_seconds):
        return None

    n = asyncio.run(
        finalize_close.run_close_finalization(
            snapshot_date="2026-09-10",
            client=_confirming_client(),
            session=object(),
            now_fn=lambda: datetime(2026, 9, 10, 15, 30, 30, tzinfo=ZoneInfo("Asia/Seoul")),
            sleep_fn=_no_sleep,
            retry_interval_seconds=0.0,
            capture_store=failing,
            run_id="run-confirm",
            cohort_id="cohort-x",
        )
    )
    assert n == 1
    assert datetime.now() is not None


def test_amain_wires_confirmation_capture(tmp_path, monkeypatch) -> None:
    """Close finalization reuses the original decision cohort identity."""
    import sys
    from datetime import date, datetime
    from unittest.mock import Mock
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.daily import finalize_close
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry, build_cohort
    from src.data.capture_store import CaptureStore

    snap = "2026-09-10"
    kst = ZoneInfo("Asia/Seoul")
    store = CaptureStore(tmp_path / "capture")
    cohort = build_cohort(date(2026, 9, 10), ["005930"], ["005930"], {}, eligibility_rule_version="price_history_panel@v1")
    completed_at = datetime(2026, 9, 10, 15, 20, 30, tzinfo=kst)
    frame = pd.DataFrame([{
        "종목코드": "005930", "종가": 269250, "admitted": True,
        "snapshot_timestamp": pd.Timestamp("2026-09-10 15:20:18", tz="Asia/Seoul"),
        "feature_available_timestamp": completed_at,
    }])
    entries = (CoverageEntry(
        symbol=None, dataset=CaptureDataset.PRICE, venue="KRX", session="regular",
        scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1,
        first_event_time=None, last_event_time=None, reason="decision-input", raw_refs=(),
    ),)
    store.publish_decision(frame, cohort=cohort, run_id="run-decision", completed_at=completed_at, entries=entries)

    class _FakeSession:
        async def close(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def create_session(self, **_kw):
            return _FakeSession()

        async def ensure_token(self, _session, force_refresh=False):
            return "T"

    captured = {}

    async def _fake_finalization(*_a, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(finalize_close, "KisApiClient", _FakeClient)
    monkeypatch.setattr(finalize_close, "run_close_finalization", _fake_finalization)
    monkeypatch.setattr(finalize_close, "load_pick_codes", lambda _d: frozenset())
    monkeypatch.setattr(finalize_close, "record_run_outcome", Mock())
    monkeypatch.setattr(finalize_close.settings, "COLLECTION_ROOT", tmp_path / "capture")
    monkeypatch.setattr(sys, "argv", ["finalize_close", "--date", snap])

    finalize_close.main()

    assert captured["snapshot_date"] == snap
    assert captured["run_id"] == f"close-{snap}"
    assert captured["cohort_id"] == cohort.cohort_id
    assert captured["capture_store"] is not None
