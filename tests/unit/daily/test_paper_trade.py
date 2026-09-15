from __future__ import annotations


def test_build_entry_orders_sizes_and_skips_zero_qty() -> None:
    import pandas as pd

    from src.daily.paper_trade import build_entry_orders

    # Given: 3픽 등가중, 마지막 종목은 시드 배분으로 0주가 되는 초고가주
    picks = pd.DataFrame(
        {
            "symbol": ["005930", "000660", "999999"],
            "allocation": [1 / 3, 1 / 3, 1 / 3],
            "price": [70_000, 200_000, 9_000_000],
        }
    )
    placed = pd.Timestamp("2026-09-10 15:19:00", tz="Asia/Seoul")

    # When
    orders = build_entry_orders(picks, "2026-09-10", seed_capital=1_000_000, placed_at=placed)

    # Then: 0주 종목은 주문이 생성되지 않는다
    symbols = [o.symbol for o in orders]
    assert "999999" not in symbols
    assert set(symbols) == {"005930", "000660"}
    assert all(o.side == "buy" and o.qty > 0 and o.limit_price is None for o in orders)

    # And: 빈 픽은 빈 주문
    assert build_entry_orders(picks.iloc[0:0], "2026-09-10", 1_000_000, placed) == []


def test_build_exit_orders_take_profit_then_moc() -> None:
    import math

    import pandas as pd

    from src.daily.paper_trade import build_exit_orders
    from src.execution.paper_broker import PAPER_TAKE_PROFIT_RATIO

    positions = pd.DataFrame(
        {"symbol": ["005930"], "qty": [10], "entry_price": [70_000], "decision_date": ["2026-09-10"]}
    )
    placed = pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul")

    # When: 익절 지정가 모드
    tp_orders = build_exit_orders(positions, "2026-09-11", placed, moc=False)

    # Then
    assert len(tp_orders) == 1
    assert tp_orders[0].side == "sell"
    assert tp_orders[0].limit_price == math.ceil(70_000 * (1 + PAPER_TAKE_PROFIT_RATIO))

    # And: MOC 대체 청산은 지정가 없음
    moc_orders = build_exit_orders(positions, "2026-09-11", placed, moc=True)
    assert moc_orders[0].limit_price is None

    # And: 미청산 포지션이 없으면 빈 주문
    assert build_exit_orders(positions.iloc[0:0], "2026-09-11", placed, moc=False) == []


def test_run_paper_session_records_no_decision_when_sleeve_empty(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 오늘 슬리브가 비어 있다(admitted < top_k 상황)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision", lambda _d: pd.DataFrame()
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, ws_client=None, session=None
        )
    )

    # Then: 체결 0건이지만 침묵하지 않는다
    assert n == 0
    df = pd.read_parquet(tmp_path / "decisions.parquet")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == ""
    assert df.iloc[0]["reason"] == "no_persisted_decision"


def test_run_paper_session_rejects_unknown_phase(tmp_path) -> None:
    import asyncio

    import pandas as pd
    import pytest

    from src.daily.paper_trade import run_paper_session
    from src.execution.paper_broker import PaperLedger

    # When / Then
    with pytest.raises(ValueError, match="phase"):
        asyncio.run(
            run_paper_session(
                pd.Timestamp("2026-09-10"),
                phase="rollover",
                ledger=PaperLedger(root=tmp_path),
            )
        )


def test_run_paper_session_exit_uses_open_positions_and_take_profit(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 전일 진입 체결이 원장에 남아 있다
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-10"}],
        kind="fills",
    )

    class _FakeWs:
        async def stream(self, _session, codes):
            assert codes == ["005930"]
            # 익절선(70,000 * 1.05 = 73,500) 미만 -> 미체결
            yield ("005930", "093000", 73_400)
            # 익절선 이상 -> 체결
            yield ("005930", "093100", 73_600)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger,
            ws_client=_FakeWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then: 익절 체결 1건이 원장에 남고, 포지션은 청산된다
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    sells = fills[fills["side"] == "sell"]
    assert len(sells) == 1
    assert int(sells.iloc[0]["fill_price"]) == 73_600
    assert len(ledger.load_open_positions()) == 0


def test_run_paper_session_exit_escalates_to_moc_after_cutoff(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.config.market_session import PAPER_EXIT_MOC_HHMMSS
    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 전일 70,000원 진입 포지션(익절선 73,500)
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-10"}],
        kind="fills",
    )

    class _FakeWs:
        async def stream(self, _session, codes):
            # 종일 익절선 미달 -> 미체결
            yield ("005930", "093000", 71_000)
            yield ("005930", "140000", 72_800)
            # MOC 컷오프 이후 첫 프린트 -> 시장가 등가로 청산
            yield ("005930", PAPER_EXIT_MOC_HHMMSS, 71_500)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger,
            ws_client=_FakeWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then: 익절가가 아니라 컷오프 시점 시장가로 체결되고 포지션이 청산된다
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    sells = fills[fills["side"] == "sell"]
    assert len(sells) == 1
    assert int(sells.iloc[0]["fill_price"]) == 71_500
    assert sells.iloc[0]["trigger"] != "take_profit"
    assert len(ledger.load_open_positions()) == 0


def test_run_paper_session_entry_fills_from_confirmed_close_no_websocket(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    monkeypatch.setattr(
        paper_trade,
        "load_topk_decision",
        lambda _d: pd.DataFrame({"symbol": ["005930"], "allocation": [1.0], "price": [70_000]}),
    )
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930"], "종가": [70_500], "종가_확정": [True],
            "execution_timestamp": [pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul")],
        }),
    )

    class _ExplodingWs:
        async def stream(self, _session, _codes):
            raise AssertionError("entry must not open a websocket stream")
            yield  # pragma: no cover

    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, ws_client=_ExplodingWs(), session=object()
        )
    )

    # Then
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    assert int(fills.iloc[0]["fill_price"]) == 70_500
    assert fills.iloc[0]["trigger"] == "auction_close"


def test_run_paper_session_entry_skips_unconfirmed_symbol_with_warning(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    monkeypatch.setattr(
        paper_trade,
        "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5], "price": [70_000, 200_000],
        }),
    )
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930", "000660"],
            "종가": [70_500, 205_000],
            "종가_확정": [False, True],
            "execution_timestamp": [pd.NaT, pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul")],
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    with caplog.at_level(logging.WARNING):
        n = asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, ws_client=None, session=None
            )
        )

    # Then: 확정된 종목 1건만 체결, 미확정 종목코드가 로그에 남는다
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    assert list(fills["symbol"]) == ["000660"]
    assert any("005930" in r.message and "UNCONFIRMED" in r.message for r in caplog.records)


def test_run_paper_session_entry_skips_symbol_missing_from_snapshot(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    monkeypatch.setattr(
        paper_trade,
        "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5], "price": [70_000, 200_000],
        }),
    )
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["000660"],
            "종가": [205_000],
            "종가_확정": [True],
            "execution_timestamp": [pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul")],
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    with caplog.at_level(logging.WARNING):
        n = asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, ws_client=None, session=None
            )
        )

    # Then: 스냅샷에 없는 종목은 NO_SNAPSHOT_ROW 경고 후 스킵, 확정 종목은 체결
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    assert list(fills["symbol"]) == ["000660"]
    assert any("005930" in r.message and "NO_SNAPSHOT_ROW" in r.message for r in caplog.records)


def test_run_paper_session_exit_still_uses_websocket_stream(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-10"}],
        kind="fills",
    )

    class _FakeWs:
        async def stream(self, _session, codes):
            assert codes == ["005930"]
            yield ("005930", "093000", 73_600)

    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, ws_client=_FakeWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    sells = fills[fills["side"] == "sell"]
    assert int(sells.iloc[0]["fill_price"]) == 73_600


def test_run_paper_session_exit_skips_without_open_positions(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    class _ExplodingWs:
        async def stream(self, _session, _codes):
            raise AssertionError("exit without positions must not open a websocket stream")
            yield  # pragma: no cover

    # Given: 원장에 미청산 매수 체결이 하나도 없다
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-14"), phase="exit", ledger=ledger, ws_client=_ExplodingWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-14 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()


def test_run_paper_session_exit_stops_consuming_after_all_orders_filled(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-10"}],
        kind="fills",
    )

    class _NeverEndingWs:
        async def stream(self, _session, codes):
            yield ("005930", "093000", 73_600)
            raise AssertionError("stream consumed after every order was filled")

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, ws_client=_NeverEndingWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 1
    assert len(ledger.load_open_positions()) == 0


def test_run_paper_session_exit_stops_at_session_end_print_leaving_unprinted_symbol_open(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.config.market_session import PAPER_EXIT_SESSION_END_HHMMSS
    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 두 종목 포지션, 000660은 종일 프린트가 없다(거래정지)
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [
            {"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000, "decision_date": "2026-09-10"},
            {"order_id": "b2", "symbol": "000660", "side": "buy", "qty": 1, "fill_price": 200_000, "decision_date": "2026-09-10"},
        ],
        kind="fills",
    )

    class _HaltedPeerWs:
        async def stream(self, _session, codes):
            yield ("005930", PAPER_EXIT_SESSION_END_HHMMSS, 71_200)
            raise AssertionError("stream consumed after the session-end print")

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, ws_client=_HaltedPeerWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then: 프린트가 온 종목만 MOC 청산되고 거래정지 종목은 미청산으로 남는다
    assert n == 1
    open_positions = ledger.load_open_positions()
    assert list(open_positions["symbol"]) == ["000660"]


def test_run_paper_session_exit_wall_clock_deadline_ends_quiet_stream(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-10"}],
        kind="fills",
    )

    class _QuietWs:
        async def stream(self, _session, codes):
            await asyncio.Event().wait()
            yield ("005930", "093000", 73_600)  # pragma: no cover

    # Given: 장마감 50ms 전에 세션이 시작된 상황(프린트는 영영 오지 않는다)
    near_close = pd.Timestamp("2026-09-11 15:30:00", tz="Asia/Seoul") - pd.Timedelta(milliseconds=50)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, ws_client=_QuietWs(), session=object(),
            now_fn=lambda: near_close,
        )
    )

    # Then: 예외 없이 종료, 체결 0건, 포지션 유지
    assert n == 0
    assert list(ledger.load_open_positions()["symbol"]) == ["005930"]


def test_run_paper_session_exit_skips_when_started_after_session_end(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-10"}],
        kind="fills",
    )

    class _ExplodingWs:
        async def stream(self, _session, _codes):
            raise AssertionError("late start must not open a websocket stream")
            yield  # pragma: no cover

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, ws_client=_ExplodingWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 15:31:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 0
    assert list(ledger.load_open_positions()["symbol"]) == ["005930"]


def test_run_paper_session_entry_consumes_persisted_decision(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 15:21 영속 결정 3종목, archive는 아직 미확정
    picks = pd.DataFrame({
        "decision_date": ["2026-09-14"] * 3,
        "symbol": ["000001", "000002", "000003"],
        "allocation": [1.0 / 3.0] * 3,
        "name": ["AAA", "BBB", "CCC"],
    })
    requested = []

    def _load(decision_date):
        requested.append(decision_date)
        return picks

    snap = pd.DataFrame({
        "종목코드": ["000001", "000002", "000003"],
        "종가": [10000.0, 20000.0, 30000.0],
        "종가_확정": [False, False, False],
    })
    monkeypatch.setattr(paper_trade, "load_topk_decision", _load)
    monkeypatch.setattr(paper_trade, "fetch_archive_snapshot", lambda _d: snap)
    ledger = PaperLedger(root=tmp_path)

    # When
    with caplog.at_level(logging.WARNING, logger="src.daily.paper_trade"):
        n = asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-14"), phase="entry", ledger=ledger, ws_client=None, session=None
            )
        )

    # Then: 재랭킹 경로 부재 + 영속 결정 3건으로 주문, 미확정이라 체결 0
    assert n == 0
    assert requested == [pd.Timestamp("2026-09-14")]
    assert not hasattr(paper_trade, "run_topk_ranker_sleeve")
    assert caplog.text.count("status=UNCONFIRMED") == 3


def test_run_paper_session_exit_issues_approval_key_with_data_account(tmp_path, monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-10"}],
        kind="fills",
    )
    issued = AsyncMock(return_value="APPROVAL")
    seen_keys: list[str] = []

    class _FakeWsClient:
        def __init__(self, approval_key):
            seen_keys.append(approval_key)

        async def stream(self, _session, codes):
            assert codes == ["005930"]
            yield ("005930", "093000", 73_600)

    session = object()
    monkeypatch.setattr(paper_trade, "issue_approval_key", issued)
    monkeypatch.setattr(paper_trade, "KisWebSocketClient", _FakeWsClient)
    monkeypatch.setattr(
        paper_trade,
        "kis_data_client_kwargs",
        lambda: {"app_key": "DATA_KEY", "app_secret": "DATA_SECRET", "account_id": "", "hts_id": None, "token_file": "t"},
    )

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, ws_client=None, session=session,
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 1
    issued.assert_awaited_once_with(session, "DATA_KEY", "DATA_SECRET")
    assert seen_keys == ["APPROVAL"]



def test_run_paper_session_entry_records_terminal_order_status_per_pick(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 4픽 = 확정 체결 / 미확정 / 0주 / 스냅샷 누락
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade,
        "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660", "999999", "111111"],
            "allocation": [0.25, 0.25, 0.25, 0.25],
            "price": [70_000, 200_000, 9_000_000, 50_000],
        }),
    )
    ts = pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul")
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930", "000660", "999999"],
            "종가": [70_500, 205_000, 9_000_000],
            "종가_확정": [1.0, 0.0, 1.0],
            "execution_timestamp": [ts, ts, ts],
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, ws_client=None, session=None
        )
    )

    # Then: 픽마다 정확히 1개의 종결 상태
    assert n == 1
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert len(orders) == 4
    status = dict(zip(orders["symbol"], orders["status"], strict=True))
    assert status == {
        "005930": "FILLED",
        "000660": "UNCONFIRMED",
        "999999": "ZERO_QTY",
        "111111": "NO_SNAPSHOT_ROW",
    }
    assert int(orders.loc[orders["symbol"] == "999999", "qty"].iloc[0]) == 0
    assert set(orders["order_id"]) == {f"2026-09-10:{s}:entry" for s in status}

    # And: NAV는 진입 체결을 반영(35주 x 70,500 + 수수료 89원)
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert len(nav) == 1
    assert nav.iloc[0]["as_of_date"] == "2026-09-10"
    assert int(nav.iloc[0]["cash"]) == 7_532_411
    assert int(nav.iloc[0]["n_open_positions"]) == 1
    assert not (tmp_path / "trades.parquet").exists()


def test_run_paper_session_exit_records_orders_trades_and_nav(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 전일 70,000원 10주 진입 체결
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul"), "decision_date": "2026-09-10",
          "trigger": "auction_close"}],
        kind="fills",
    )

    class _FakeWs:
        async def stream(self, _session, codes):
            yield ("005930", "093000", 73_400)
            yield ("005930", "093100", 73_600)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger,
            ws_client=_FakeWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 1
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["FILLED"]
    assert orders["reason"].tolist() == ["take_profit"]
    assert int(orders.iloc[0]["limit_price"]) == 73_500
    trades = pd.read_parquet(tmp_path / "trades.parquet")
    assert len(trades) == 1
    assert int(trades.iloc[0]["net_pnl"]) == 34_477
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert nav["as_of_date"].tolist() == ["2026-09-11"]
    assert int(nav.iloc[0]["n_open_positions"]) == 0
    assert int(nav.iloc[0]["nav"]) == 10_034_477


def test_run_paper_session_exit_marks_unfilled_order_when_stream_ends(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 익절선 미달 프린트 1건 후 스트림 종료(MOC 컷오프 미도달)
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul"), "decision_date": "2026-09-10",
          "trigger": "auction_close"}],
        kind="fills",
    )

    class _FakeWs:
        async def stream(self, _session, codes):
            yield ("005930", "093000", 73_400)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger,
            ws_client=_FakeWs(), session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul"),
        )
    )

    # Then: 미체결도 종결 상태로 남고 포지션은 NAV에 미청산으로 잡힌다
    assert n == 0
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["UNFILLED"]
    assert orders["order_id"].tolist() == ["2026-09-11:005930:exit"]
    assert not (tmp_path / "trades.parquet").exists()
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert int(nav.iloc[0]["n_open_positions"]) == 1
    assert int(nav.iloc[0]["nav"]) == 9_999_975
