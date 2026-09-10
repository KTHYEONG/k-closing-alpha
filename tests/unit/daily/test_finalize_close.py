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
        async def get_current_price(self, session, code, market_div_code=None):
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
        async def get_current_price(self, session, code, market_div_code=None):
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
        async def get_current_price(self, session, code, market_div_code=None):
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
        async def get_current_price(self, session, code, market_div_code=None):
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
        async def get_current_price(self, session, code, market_div_code=None):
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
