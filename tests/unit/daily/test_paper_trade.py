from __future__ import annotations

import pytest


async def _open_day(_day: str) -> bool:
    return True


@pytest.fixture(autouse=True)
def _standard_session(monkeypatch) -> None:
    """Pre-gate scenarios run under a STANDARD session; gate scenarios inject their own resolver."""
    from src.daily import paper_trade
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay, SessionKind

    def _resolve(trading_day, **_kwargs):
        return SessionDay(
            trading_date=trading_day,
            kind=SessionKind.STANDARD,
            clock=SessionClock.standard(trading_day),
            provenance="standard",
        )

    monkeypatch.setattr(paper_trade, "resolve_session_day", _resolve)


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
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul"),
            trading_day_fn=_open_day,
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


def test_run_paper_session_entry_fills_from_confirmed_close_no_websocket(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    monkeypatch.setattr(
        paper_trade,
        "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0], "close": [70_500],
            "decided_at": [pd.Timestamp("2026-09-10 15:23:02", tz="Asia/Seoul")],
        }),
    )
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930"], "종가": [70_500], "종가_확정": [True],
            "execution_timestamp": [pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul")],
        }),
    )


    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul"),
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
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5], "close": [70_000, 200_000],
            "decided_at": [pd.Timestamp("2026-09-10 15:23:02", tz="Asia/Seoul")] * 2,
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
                pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None,
                now_fn=lambda: pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul"),
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
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5], "close": [70_000, 200_000],
            "decided_at": [pd.Timestamp("2026-09-10 15:23:02", tz="Asia/Seoul")] * 2,
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
                pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None,
                now_fn=lambda: pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul"),
            )
        )

    # Then: 스냅샷에 없는 종목은 NO_SNAPSHOT_ROW 경고 후 스킵, 확정 종목은 체결
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    assert list(fills["symbol"]) == ["000660"]
    assert any("005930" in r.message and "NO_SNAPSHOT_ROW" in r.message for r in caplog.records)


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
        "close": [10000.0, 20000.0, 30000.0],
        "decided_at": [pd.Timestamp("2026-09-14 15:23:02", tz="Asia/Seoul")] * 3,
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
                pd.Timestamp("2026-09-14"), phase="entry", ledger=ledger, session=None,
                now_fn=lambda: pd.Timestamp("2026-09-14 15:34:00", tz="Asia/Seoul"),
            )
        )

    # Then: 재랭킹 경로 부재 + 영속 결정 3건으로 주문, 미확정이라 체결 0
    assert n == 0
    assert requested == [pd.Timestamp("2026-09-14")]
    assert not hasattr(paper_trade, "run_topk_ranker_sleeve")
    assert caplog.text.count("status=UNCONFIRMED") == 3


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
            "close": [70_000, 200_000, 9_000_000, 50_000],
            "decided_at": [pd.Timestamp("2026-09-10 15:23:02", tz="Asia/Seoul")] * 4,
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
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul"),
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
    trades = pd.read_parquet(tmp_path / "trades.parquet")
    assert trades.empty


def test_run_paper_session_entry_sizes_from_available_cash(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    kst = "Asia/Seoul"
    # Given: 청산되지 않은 9,000,000원 로트가 현금을 묶고 있다
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-09:000660:entry", "symbol": "000660", "side": "buy", "qty": 90, "fill_price": 100_000,
          "filled_at": pd.Timestamp("2026-09-09 15:30:20", tz=kst), "decision_date": "2026-09-09", "trigger": "auction_close"}],
        kind="fills",
    )
    monkeypatch.setattr(
        paper_trade,
        "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0], "close": [70_000],
            "decided_at": [pd.Timestamp("2026-09-10 15:23:02", tz="Asia/Seoul")],
        }),
    )
    ts = pd.Timestamp("2026-09-10 15:30:20", tz=kst)
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({"종목코드": ["005930"], "종가": [70_500], "종가_확정": [1.0], "execution_timestamp": [ts]}),
    )

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then: 시드(141주)가 아니라 가용현금 999,673원 기준 14주만 진입하고 현금은 음수가 되지 않는다
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    buy = fills[fills["order_id"] == "2026-09-10:005930:entry"].iloc[0]
    assert int(buy["qty"]) == 14
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert int(nav.iloc[0]["cash"]) == 12_638
    assert int(nav.iloc[0]["cash"]) >= 0
    assert int(nav.iloc[0]["n_open_positions"]) == 2


def test_run_paper_session_entry_second_trigger_same_day_is_a_noop(tmp_path, monkeypatch) -> None:
    """실측: 2026-09-21 kca-finalize-close의 ExecStopPost 체인과 독립 백스톱
    타이머가 같은 날 entry를 두 번 트리거해, 두 번째 실행이 첫 실행의 지출로
    줄어든 현금을 기준으로 재사이징하며 017900을 301주->50주로 축소시키고
    402340/009150 orders 감사기록을 ZERO_QTY로 오기록했다."""
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    kst = "Asia/Seoul"
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    monkeypatch.setattr(
        paper_trade,
        "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0], "close": [70_500],
            "decided_at": [pd.Timestamp("2026-09-10 15:23:02", tz="Asia/Seoul")],
        }),
    )
    ts = pd.Timestamp("2026-09-10 15:30:20", tz=kst)
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({"종목코드": ["005930"], "종가": [70_500], "종가_확정": [1.0], "execution_timestamp": [ts]}),
    )

    # When: ExecStopPost 체인과 백스톱 타이머가 둘 다 같은 날 entry를 트리거한다
    def _at() -> pd.Timestamp:
        return pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul")

    first = asyncio.run(paper_trade.run_paper_session(pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None, now_fn=_at))
    second = asyncio.run(paper_trade.run_paper_session(pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None, now_fn=_at))

    # Then: 두 번째 트리거는 아무것도 하지 않는다 -- 첫 체결이 그대로 남는다
    assert first == 1
    assert second == 0
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    buy = fills[fills["order_id"] == "2026-09-10:005930:entry"]
    assert len(buy) == 1
    assert int(buy.iloc[0]["qty"]) == 141  # 시드 전액(10,000,000/70,500) 기준 최초 사이징, 재축소되지 않음


def test_build_exit_orders_emits_market_open_exit_orders() -> None:
    import pandas as pd

    from src.daily.paper_trade import build_exit_orders

    positions = pd.DataFrame(
        {"entry_order_id": ["b1"], "symbol": ["005930"], "qty": [10], "entry_price": [70_000], "decision_date": ["2026-09-10"]}
    )
    placed = pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul")

    # When
    orders = build_exit_orders(positions, "2026-09-11", placed)

    # Then
    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].limit_price is None
    assert orders[0].reason == "open_exit"
    assert orders[0].order_id == "b1:exit:2026-09-11"
    assert orders[0].entry_order_id == "b1"
    assert orders[0].qty == 10
    assert orders[0].placed_at == placed

    # And: 미청산 포지션이 없으면 빈 주문
    assert build_exit_orders(positions.iloc[0:0], "2026-09-11", placed) == []


def test_run_paper_session_exit_fills_at_krx_open_quote(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 전일 70,000원 10주 진입, 당일 KRX 시가 71,000원
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul"), "decision_date": "2026-09-10",
          "trigger": "auction_close"}],
        kind="fills",
    )

    asked: list[str] = []

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        asked.append(code)
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-11", open_price=71_000)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-11 09:01:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 1
    assert asked == ["005930"]
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    sells = fills[fills["side"] == "sell"]
    assert int(sells.iloc[0]["fill_price"]) == 71_000
    assert sells.iloc[0]["trigger"] == "auction_open"
    assert pd.Timestamp(sells.iloc[0]["filled_at"]) == pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul")
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["FILLED"]
    assert orders["reason"].tolist() == ["open_exit"]
    trades = pd.read_parquet(tmp_path / "trades.parquet")
    assert int(trades.iloc[0]["net_pnl"]) == 8_530
    assert trades.iloc[0]["exit_trigger"] == "auction_open"
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert int(nav.iloc[0]["nav"]) == 10_008_530
    assert len(ledger.load_open_positions()) == 0


def test_run_paper_session_exit_retries_open_quote_then_leaves_unavailable_lot_open(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    kst = "Asia/Seoul"
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [
            {"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
             "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz=kst), "decision_date": "2026-09-10", "trigger": "auction_close"},
            {"order_id": "2026-09-10:000660:entry", "symbol": "000660", "side": "buy", "qty": 3, "fill_price": 200_000,
             "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz=kst), "decision_date": "2026-09-10", "trigger": "auction_close"},
        ],
        kind="fills",
    )
    calls: dict[str, int] = {"005930": 0, "000660": 0}

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        calls[code] += 1
        # 005930은 두 번째 조회에서 당일 세션으로 입증된 시가 형성, 000660은 거래정지로 끝내 0
        if code == "005930" and calls[code] >= 2:
            return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-11", open_price=71_000)
        return paper_trade.DatedOpenQuote(symbol=code, business_date="", open_price=0)

    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    # When
    with caplog.at_level(logging.WARNING, logger="src.daily.paper_trade"):
        n = asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
                now_fn=lambda: pd.Timestamp("2026-09-11 09:01:00", tz=kst), sleep_fn=_sleep,
            )
        )

    # Then
    assert n == 1
    assert calls == {"005930": 2, "000660": paper_trade.PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS}
    assert sleeps == [paper_trade.PAPER_EXIT_OPEN_QUOTE_RETRY_SECONDS] * (paper_trade.PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS - 1)
    orders = pd.read_parquet(tmp_path / "orders.parquet").set_index("symbol")
    assert orders.loc["005930", "status"] == "FILLED"
    assert orders.loc["000660", "status"] == "UNFILLED"
    open_lots = ledger.load_open_positions()
    assert open_lots["symbol"].tolist() == ["000660"]
    assert "OPEN_UNAVAILABLE" in caplog.text


def test_run_paper_session_exit_waits_until_open_quote_is_trustworthy(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul"), "decision_date": "2026-09-10",
          "trigger": "auction_close"}],
        kind="fills",
    )

    events: list[str] = []

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        events.append("quote")
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-11", open_price=71_000)

    async def _sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    # When: 08:59:00 기동(부팅 캐치업 등) -> 09:00:30까지 대기 후 재평가
    clocks = iter([
        pd.Timestamp("2026-09-11 08:59:00", tz="Asia/Seoul"),
        pd.Timestamp("2026-09-11 09:01:00", tz="Asia/Seoul"),
        pd.Timestamp("2026-09-11 09:01:00", tz="Asia/Seoul"),
    ])
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: next(clocks), sleep_fn=_sleep,
        )
    )

    # Then
    assert n == 1
    assert events == ["sleep:90.0", "quote"]


def test_run_paper_session_exit_rejects_non_same_day_run(tmp_path) -> None:
    import asyncio

    import pandas as pd
    import pytest

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul"), "decision_date": "2026-09-10",
          "trigger": "auction_close"}],
        kind="fills",
    )

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        raise AssertionError("past-date exit must not query today's open")

    # When / Then
    with pytest.raises(ValueError, match="same-day"):
        asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
                now_fn=lambda: pd.Timestamp("2026-09-12 09:01:00", tz="Asia/Seoul"),
            )
        )
    assert not (tmp_path / "orders.parquet").exists()


def test_run_paper_session_exit_skips_without_open_positions(tmp_path) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        raise AssertionError("exit without positions must not query quotes")

    # Given: 원장에 미청산 매수 체결이 하나도 없다
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-14"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-14 09:01:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()


def test_run_paper_session_exit_closes_two_lots_of_same_symbol_at_one_open(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    kst = "Asia/Seoul"
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [
            {"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
             "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz=kst), "decision_date": "2026-09-10", "trigger": "auction_close"},
            {"order_id": "2026-09-11:005930:entry", "symbol": "005930", "side": "buy", "qty": 12, "fill_price": 71_000,
             "filled_at": pd.Timestamp("2026-09-11 15:30:20", tz=kst), "decision_date": "2026-09-11", "trigger": "auction_close"},
        ],
        kind="fills",
    )
    asked: list[str] = []

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        asked.append(code)
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-12", open_price=72_000)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-12"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-12 09:01:00", tz=kst),
        )
    )

    # Then
    assert n == 2
    assert asked == ["005930"]
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    sells = fills[fills["side"] == "sell"]
    assert sorted(sells["order_id"].tolist()) == [
        "2026-09-10:005930:entry:exit:2026-09-12",
        "2026-09-11:005930:entry:exit:2026-09-12",
    ]
    trades = pd.read_parquet(tmp_path / "trades.parquet")
    assert sorted(trades["net_pnl"].astype(int).tolist()) == [10_210, 18_509]
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert int(nav.iloc[0]["nav"]) == 10_028_719
    assert len(ledger.load_open_positions()) == 0


def test_fetch_krx_dated_open_quote_attests_by_latest_dated_row() -> None:
    import asyncio

    from src.daily.paper_trade import fetch_krx_dated_open_quote

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "0", "output": {"stck_oprc": "13410"}}

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            return {"rt_cd": "0", "output2": [
                "junk-row",
                {"stck_bsop_date": "20260922", "stck_oprc": "13000"},
                {"stck_bsop_date": "20260923", "stck_oprc": "13410"},
            ]}

    # When: 휴장일(2026-09-24)에 조회하면 창 끝에는 직전 세션 행만 있다
    quote = asyncio.run(fetch_krx_dated_open_quote(_Client(), object(), "005930", "2026-09-24"))

    # Then: 최신 행 날짜로 귀속되고 가격은 입증되지 않아 0이다
    assert quote.symbol == "005930"
    assert quote.business_date == "2026-09-23"
    assert quote.open_price == 0


def test_fetch_krx_dated_open_quote_returns_open_on_matching_session() -> None:
    import asyncio

    from src.daily.paper_trade import fetch_krx_dated_open_quote

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "0", "output": {"stck_oprc": "1000"}}

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260925", "stck_oprc": "1000"}]}

    # When
    quote = asyncio.run(fetch_krx_dated_open_quote(_Client(), object(), "005930", "2026-09-25"))

    # Then
    assert (quote.symbol, quote.business_date, quote.open_price) == ("005930", "2026-09-25", 1000)


def test_fetch_krx_dated_open_quote_fails_soft_on_vendor_error() -> None:
    import asyncio

    from src.daily.paper_trade import fetch_krx_dated_open_quote

    class _FailingInquire:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "1", "output": {"stck_oprc": "1000"}}

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260925", "stck_oprc": "1000"}]}

    class _RaisingInquire:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            raise RuntimeError("transport down")

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            return {"rt_cd": "0", "output2": []}

    class _RaisingChart:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "0", "output": {"stck_oprc": "1000"}}

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            raise RuntimeError("chart down")

    # When / Then: 벤더 오류는 예외 없이 open 0으로 감쇠한다
    assert asyncio.run(fetch_krx_dated_open_quote(_FailingInquire(), object(), "005930", "2026-09-25")).open_price == 0
    assert asyncio.run(fetch_krx_dated_open_quote(_RaisingInquire(), object(), "005930", "2026-09-25")).open_price == 0
    assert asyncio.run(fetch_krx_dated_open_quote(_RaisingChart(), object(), "005930", "2026-09-25")).open_price == 0


def test_fetch_krx_dated_open_quote_rejects_disagreeing_sources() -> None:
    import asyncio

    from src.daily.paper_trade import fetch_krx_dated_open_quote

    class _Client:
        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "0", "output": {"stck_oprc": "13500"}}

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260925", "stck_oprc": "13400"}]}

    # When: 날짜는 당일이지만 두 시가가 다르면
    quote = asyncio.run(fetch_krx_dated_open_quote(_Client(), object(), "005930", "2026-09-25"))

    # Then: 차트 가격으로 대체하지 않고 0이다
    assert quote.business_date == "2026-09-25"
    assert quote.open_price == 0


def test_run_paper_session_exit_default_quote_uses_data_account_client(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz="Asia/Seoul"), "decision_date": "2026-09-10",
          "trigger": "auction_close"}],
        kind="fills",
    )

    built: list[dict] = []

    class _FakeClient:
        def __init__(self, **kwargs):
            built.append(kwargs)

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "0", "output": {"stck_oprc": "71000"}}

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260911", "stck_oprc": "71000"}]}

    monkeypatch.setattr(paper_trade, "KisApiClient", _FakeClient)
    monkeypatch.setattr(paper_trade, "kis_data_client_kwargs", lambda: {"app_key": "DATA_KEY", "app_secret": "DATA_SECRET"})

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, session=object(), trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-11 09:01:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 1
    assert built == [{"app_key": "DATA_KEY", "app_secret": "DATA_SECRET"}]


def _seed_open_lot(tmp_path):
    import pandas as pd

    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-23:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-23 15:30:20", tz="Asia/Seoul"), "decision_date": "2026-09-23",
          "trigger": "auction_close"}],
        kind="fills",
    )
    return ledger


def test_run_paper_session_exit_on_holiday_keeps_lots_open_without_quoting(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 평일 휴장일(2026-09-24) 아침, 전일 진입 로트 1개. 휴장일에도 시세 API는 직전 시가를 돌려준다.
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)
    asked: list[str] = []

    async def _stale_quote(code: str) -> paper_trade.DatedOpenQuote:
        asked.append(code)
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-23", open_price=69_000)

    async def _holiday(_day: str) -> bool:
        return False

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_stale_quote, trading_day_fn=_holiday,
            now_fn=lambda: pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"),
        )
    )

    # Then: 체결·주문 기록 없음, 로트는 다음 실제 시가까지 유지
    assert n == 0
    assert asked == []
    assert ledger.load("orders").empty
    assert (ledger.load("fills")["side"] == "buy").all()
    assert ledger.load_open_positions()["entry_order_id"].tolist() == ["2026-09-23:005930:entry"]


def test_run_paper_session_exit_fails_closed_when_trading_day_oracle_fails(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd
    import pytest

    from src.daily import paper_trade

    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        raise AssertionError("must not quote when the trading day is unknown")

    async def _oracle_down(_day: str) -> bool:
        raise RuntimeError("KIS trading-day oracle failed rt_cd=1")

    with pytest.raises(RuntimeError, match="oracle failed"):
        asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_oracle_down,
                now_fn=lambda: pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"),
            )
        )

    assert ledger.load("orders").empty
    assert len(ledger.load_open_positions()) == 1


def test_run_paper_session_exit_default_oracle_uses_data_account_index_history(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)
    asked_days: list[str] = []

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def get_market_index_history(self, session, code, start, end):
            asked_days.append(start)
            # 휴장일 조회는 요청일 행 없이 직전 거래일 행만 돌려준다
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260923"}]}

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            raise AssertionError("must not quote on a holiday")

    monkeypatch.setattr(paper_trade, "KisApiClient", _FakeClient)
    monkeypatch.setattr(paper_trade, "kis_data_client_kwargs", lambda: {})

    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, session=object(),
            now_fn=lambda: pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"),
        )
    )

    assert n == 0
    assert asked_days == ["20260924"]
    assert len(ledger.load_open_positions()) == 1


def test_run_paper_session_entry_holds_ledger_lock_across_sizing_and_write(tmp_path, monkeypatch) -> None:
    import asyncio
    import threading

    import pandas as pd
    import pytest

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 다른 프로세스가 원장 락을 쥐고 있다
    holder = PaperLedger(root=tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def _hold() -> None:
        with holder.exclusive():
            entered.set()
            assert release.wait(timeout=15)

    t = threading.Thread(target=_hold, daemon=True)
    t.start()
    assert entered.wait(timeout=15)
    try:
        monkeypatch.setattr(
            paper_trade, "load_topk_decision",
            lambda _d: pd.DataFrame({"symbol": ["005930"], "allocation": [1.0], "price": [70_000]}),
        )
        locked = PaperLedger(root=tmp_path, lock_timeout_seconds=0.5)

        # When / Then: 사이징-기록 전 구간이 락을 요구하므로 TimeoutError, 쓰기 없음
        with pytest.raises(TimeoutError):
            asyncio.run(
                paper_trade.run_paper_session(
                    pd.Timestamp("2026-09-10"), phase="entry", ledger=locked, session=None,
                    now_fn=lambda: pd.Timestamp("2026-09-10 15:34:00", tz="Asia/Seoul"),
                )
            )
        assert locked.load("fills").empty
        assert locked.load("orders").empty
    finally:
        release.set()
        t.join(timeout=15)


def test_run_paper_session_entry_sizes_from_effective_cash(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    kst = "Asia/Seoul"
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    buy = {"order_id": "2026-09-09:005930:entry", "symbol": "005930", "side": "buy", "qty": 10,
           "fill_price": 70_000, "filled_at": pd.Timestamp("2026-09-09 15:30:20", tz=kst),
           "decision_date": "2026-09-09", "trigger": "auction_close", "entry_order_id": None}
    sell = {"order_id": "2026-09-09:005930:entry:exit:2026-09-10", "symbol": "005930", "side": "sell", "qty": 10,
            "fill_price": 60_000, "filled_at": pd.Timestamp("2026-09-10 09:00:00", tz=kst),
            "decision_date": "2026-09-10", "trigger": "auction_open",
            "entry_order_id": "2026-09-09:005930:entry"}
    ledger.record([buy, sell], kind="fills")
    # When: 손실 매도를 void하면 현금은 매수 지출만 반영된 상태로 복원된다
    ledger.void_fill(
        sell["order_id"], reason="holiday_phantom_exit", evidence="e", operator="tester", as_of_date="2026-09-10",
    )
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["000660"], "allocation": [1.0], "close": [200_000],
            "decided_at": [pd.Timestamp("2026-09-11 15:23:02", tz="Asia/Seoul")],
        }),
    )
    ts = pd.Timestamp("2026-09-11 15:30:20", tz=kst)
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({"종목코드": ["000660"], "종가": [200_000], "종가_확정": [1.0], "execution_timestamp": [ts]}),
    )

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-11 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then: void 복원 현금(10,000,000 - 700,025 = 9,299,975) 기준 46주 진입
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    new_buy = fills[fills["order_id"] == "2026-09-11:000660:entry"].iloc[0]
    assert int(new_buy["qty"]) == 46


def test_run_paper_session_exit_rereads_open_lots_inside_lock(tmp_path, monkeypatch) -> None:
    import asyncio
    from unittest.mock import patch

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 락 밖 프로브에서는 로트가 보이지만 락 안 재조회에서는 사라진 경합
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = PaperLedger(root=tmp_path)
    probe = pd.DataFrame({
        "entry_order_id": ["b1"], "symbol": ["005930"], "qty": [10],
        "entry_price": [70_000], "decision_date": ["2026-09-10"],
    })
    empty = probe.iloc[0:0]
    calls: list[str] = []

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        calls.append(code)
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-11", open_price=71_000)

    with patch.object(PaperLedger, "load_open_positions", side_effect=[probe, empty]):
        # When
        n = asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
                now_fn=lambda: pd.Timestamp("2026-09-11 09:01:00", tz="Asia/Seoul"),
            )
        )

    # Then: 락 안 권위 재조회가 비어 시세 조회 없이 종료한다
    assert n == 0
    assert calls == []


def test_run_paper_session_exit_stale_quotes_skip_without_recording_and_emit_degraded(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 달력 오라클이 틀리게 휴장일(2026-09-24)을 개장으로 판단, 시세는 직전 세션가
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)

    async def _wrong_open(_day: str) -> bool:
        return True

    async def _stale(code: str) -> paper_trade.DatedOpenQuote:
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-23", open_price=13410)

    async def _no_sleep(_seconds: float) -> None:
        return None

    outcomes: list[tuple] = []

    def _record(outcome: str, **kwargs) -> None:
        outcomes.append((outcome, kwargs))

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_stale, trading_day_fn=_wrong_open,
            now_fn=lambda: pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"), sleep_fn=_no_sleep,
            record_fn=_record,
        )
    )

    # Then: 체결·주문 기록 없이 로트 유지, DEGRADED session_not_opened 방출
    assert n == 0
    assert ledger.load("fills")["side"].tolist() == ["buy"]
    assert ledger.load("orders").empty
    assert ledger.load_open_positions()["entry_order_id"].tolist() == ["2026-09-23:005930:entry"]
    assert outcomes == [("DEGRADED", {"run_date": "2026-09-24", "reason": "session_not_opened", "metrics": {"n_open": 1}})]


def test_run_paper_session_exit_attested_quote_fills_at_inquire_open(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 당일 세션으로 입증된 시가 13500원
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)

    async def _attested(code: str) -> paper_trade.DatedOpenQuote:
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-24", open_price=13500)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_attested, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"),
        )
    )

    # Then: 조회가 그대로 체결가, filled_at은 09:00:00 경매 시각
    assert n == 1
    sells = ledger.load("fills")
    sells = sells[sells["side"] == "sell"]
    assert int(sells.iloc[0]["fill_price"]) == 13500
    assert sells.iloc[0]["trigger"] == "auction_open"
    assert pd.Timestamp(sells.iloc[0]["filled_at"]) == pd.Timestamp("2026-09-24 09:00:00", tz="Asia/Seoul")


def test_run_paper_session_exit_disagreeing_sources_retry_then_leave_lot_open(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    import pandas as pd

    from src.daily import paper_trade

    # Given: 두 원천이 엇갈려 매번 open 0(날짜는 당일이라 stale 경로가 아니다)
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)
    calls = 0

    async def _disagree(code: str) -> paper_trade.DatedOpenQuote:
        nonlocal calls
        calls += 1
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-24", open_price=0)

    async def _no_sleep(_seconds: float) -> None:
        return None

    outcomes: list[tuple] = []

    # When
    with caplog.at_level(logging.WARNING, logger="src.daily.paper_trade"):
        n = asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_disagree, trading_day_fn=_open_day,
                now_fn=lambda: pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"), sleep_fn=_no_sleep,
                record_fn=lambda *a, **k: outcomes.append((a, k)),
            )
        )

    # Then: 기존 UNFILLED 경로로 로트 유지, DEGRADED 없음
    assert n == 0
    assert calls == paper_trade.PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["UNFILLED"]
    assert ledger.load_open_positions()["entry_order_id"].tolist() == ["2026-09-23:005930:entry"]
    assert "OPEN_UNAVAILABLE" in caplog.text
    assert outcomes == []


def test_run_paper_session_exit_partial_attestation_fills_only_attested(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: A는 입증, B는 직전 세션 날짜
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    kst = "Asia/Seoul"
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [
            {"order_id": "2026-09-23:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
             "filled_at": pd.Timestamp("2026-09-23 15:30:20", tz=kst), "decision_date": "2026-09-23",
             "trigger": "auction_close"},
            {"order_id": "2026-09-23:000660:entry", "symbol": "000660", "side": "buy", "qty": 3, "fill_price": 200_000,
             "filled_at": pd.Timestamp("2026-09-23 15:30:20", tz=kst), "decision_date": "2026-09-23",
             "trigger": "auction_close"},
        ],
        kind="fills",
    )

    async def _mixed(code: str) -> paper_trade.DatedOpenQuote:
        if code == "005930":
            return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-24", open_price=71_000)
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-23", open_price=205_000)

    async def _no_sleep(_seconds: float) -> None:
        return None

    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_mixed, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-24 09:01:00", tz=kst), sleep_fn=_no_sleep,
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then: A만 체결, B는 UNFILLED, DEGRADED 없음
    assert n == 1
    orders = pd.read_parquet(tmp_path / "orders.parquet").set_index("symbol")
    assert orders.loc["005930", "status"] == "FILLED"
    assert orders.loc["000660", "status"] == "UNFILLED"
    assert ledger.load_open_positions()["symbol"].tolist() == ["000660"]
    assert outcomes == []


def test_run_paper_session_exit_default_fetcher_requests_unadjusted_chart_over_lookback(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 기록형 가짜 클라이언트, 2026-09-25 세션
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    kst = "Asia/Seoul"
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-09-24:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-09-24 15:30:20", tz=kst), "decision_date": "2026-09-24",
          "trigger": "auction_close"}],
        kind="fills",
    )
    chart_calls: list[dict] = []

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            return {"rt_cd": "0", "output": {"stck_oprc": "71000"}}

        async def get_stock_ohlcv_history(self, session, code, start_date=None, end_date=None,
                                          period_code=None, adj_price=None, market_div_code=None):
            chart_calls.append({
                "code": code, "start_date": start_date, "end_date": end_date,
                "period_code": period_code, "adj_price": adj_price, "market_div_code": market_div_code,
            })
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260925", "stck_oprc": "71000"}]}

    monkeypatch.setattr(paper_trade, "KisApiClient", _FakeClient)
    monkeypatch.setattr(paper_trade, "kis_data_client_kwargs", lambda: {})

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="exit", ledger=ledger, session=object(), trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-25 09:01:00", tz=kst),
        )
    )

    # Then: 무수정 일봉을 조회 창 전체로 요청하고 입증된 시가에 체결한다
    assert n == 1
    assert chart_calls == [{
        "code": "005930", "start_date": "20260911", "end_date": "20260925",
        "period_code": "D", "adj_price": "1", "market_div_code": "J",
    }]
    sells = ledger.load("fills")
    sells = sells[sells["side"] == "sell"]
    assert int(sells.iloc[0]["fill_price"]) == 71_000


def test_exit_window_state_boundaries() -> None:
    import pandas as pd
    import pytest

    from src.daily.paper_trade import WindowState, exit_window_state

    day = pd.Timestamp("2026-09-25")

    def _at(hms: str) -> pd.Timestamp:
        return pd.Timestamp(f"2026-09-25 {hms[:2]}:{hms[2:4]}:{hms[4:]}", tz="Asia/Seoul")

    # Then: 경계 포함 규칙과 대기 상한
    assert exit_window_state(day, _at("090030")) is WindowState.OPEN
    assert exit_window_state(day, _at("093000")) is WindowState.OPEN
    assert exit_window_state(day, _at("093001")) is WindowState.EXPIRED
    assert exit_window_state(day, _at("090029")) is WindowState.WAIT
    assert exit_window_state(day, _at("085030")) is WindowState.WAIT
    assert exit_window_state(day, _at("085029")) is WindowState.EARLY_SKIP
    # And: naive 시계와 다른 날짜는 거부
    with pytest.raises(ValueError, match="tz-aware"):
        exit_window_state(day, pd.Timestamp("2026-09-25 09:01:00"))
    with pytest.raises(ValueError, match="same-day"):
        exit_window_state(day, pd.Timestamp("2026-09-26 09:01:00", tz="Asia/Seoul"))


def test_resolve_entry_decision_date_targets_previous_weekday() -> None:
    import pandas as pd

    from src.daily.paper_trade import resolve_entry_decision_date

    def _at(date: str, hm: str) -> pd.Timestamp:
        return pd.Timestamp(f"{date} {hm[:2]}:{hm[2:]}", tz="Asia/Seoul")

    # Then: 15:30 이후는 당일, 이전은 직전 평일(월->금)
    assert resolve_entry_decision_date(_at("2026-09-25", "1540")) == pd.Timestamp("2026-09-25")
    assert resolve_entry_decision_date(_at("2026-09-25", "0800")) == pd.Timestamp("2026-09-24")
    assert resolve_entry_decision_date(_at("2026-09-28", "0700")) == pd.Timestamp("2026-09-25")
    assert resolve_entry_decision_date(_at("2026-09-26", "0700")) == pd.Timestamp("2026-09-25")
    assert resolve_entry_decision_date(_at("2026-09-27", "0700")) == pd.Timestamp("2026-09-25")


def test_entry_window_state_bounds() -> None:
    import pandas as pd
    import pytest

    from src.daily.paper_trade import WindowState, entry_window_state

    day = pd.Timestamp("2026-09-22")

    def _at(date: str, hms: str) -> pd.Timestamp:
        return pd.Timestamp(f"{date} {hms[:2]}:{hms[2:4]}:{hms[4:]}", tz="Asia/Seoul")

    # Then: [D 15:30, D+1 08:30), 마감 포함 만료, EARLY_SKIP 없음
    assert entry_window_state(day, _at("2026-09-22", "152959")) is WindowState.WAIT
    assert entry_window_state(day, _at("2026-09-22", "153000")) is WindowState.OPEN
    assert entry_window_state(day, _at("2026-09-23", "082959")) is WindowState.OPEN
    assert entry_window_state(day, _at("2026-09-23", "083000")) is WindowState.EXPIRED
    with pytest.raises(ValueError, match="tz-aware"):
        entry_window_state(day, pd.Timestamp("2026-09-22 15:34:00"))


def test_run_paper_session_exit_after_window_alerts_without_quoting(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 창이 닫힌 14:00 캐치업 기동
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        raise AssertionError("expired exit must not quote")

    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-24 14:00:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then: 조회·기록 없이 DEGRADED 알림
    assert n == 0
    assert ledger.load("orders").empty
    assert (ledger.load("fills")["side"] == "buy").all()
    assert outcomes == [(("DEGRADED",), {"run_date": "2026-09-24", "reason": "exit_window_expired", "metrics": {"n_open": 1}})]


def test_run_paper_session_exit_started_too_early_skips_without_sleeping(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 08:00 기동 — 09:01 정규 발화가 처리하므로 대기하지 않는다
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        raise AssertionError("early exit must not quote")

    async def _sleep(_seconds: float) -> None:
        raise AssertionError("early exit must not sleep")

    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-24 08:00:00", tz="Asia/Seoul"), sleep_fn=_sleep,
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then: 조용히 스킵, outcome 없음
    assert n == 0
    assert ledger.load("orders").empty
    assert outcomes == []


def test_run_paper_session_exit_waits_at_most_ten_minutes_then_fills(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 08:55 기동 — 330초 대기면 창 안이다
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)

    async def _attested(code: str) -> paper_trade.DatedOpenQuote:
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-24", open_price=71_000)

    sleeps: list[float] = []
    clocks = iter([
        pd.Timestamp("2026-09-24 08:55:00", tz="Asia/Seoul"),
        pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"),
        pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"),
    ])

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_attested, trading_day_fn=_open_day,
            now_fn=lambda: next(clocks), sleep_fn=_sleep,
        )
    )

    # Then: 330초 한 번 대기 후 체결 기록
    assert n == 1
    assert sleeps == [330.0]
    sells = ledger.load("fills")
    assert int(sells[sells["side"] == "sell"].iloc[0]["fill_price"]) == 71_000


def test_run_paper_session_exit_suspend_past_window_after_wait_alerts(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 대기 중 시스템 절전 등으로 창을 넘긴 기동
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)
    clocks = iter([
        pd.Timestamp("2026-09-24 08:55:00", tz="Asia/Seoul"),
        pd.Timestamp("2026-09-24 10:00:00", tz="Asia/Seoul"),
    ])

    async def _quote(code: str) -> paper_trade.DatedOpenQuote:
        raise AssertionError("post-window exit must not quote")

    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: next(clocks), sleep_fn=lambda _s: asyncio.sleep(0),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then: 재평가에서 만료로 DEGRADED
    assert n == 0
    assert ledger.load("orders").empty
    assert outcomes == [(("DEGRADED",), {"run_date": "2026-09-24", "reason": "exit_window_expired", "metrics": {"n_open": 1}})]


def test_run_paper_session_exit_late_quotes_recorded_unfilled(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade

    # Given: 조회 루프 중에 창을 넘긴 경우 — 입증된 시가도 늦은 체결로 기록하지 않는다
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    ledger = _seed_open_lot(tmp_path)
    clocks = iter([
        pd.Timestamp("2026-09-24 09:01:00", tz="Asia/Seoul"),
        pd.Timestamp("2026-09-24 09:35:00", tz="Asia/Seoul"),
    ])

    async def _attested(code: str) -> paper_trade.DatedOpenQuote:
        return paper_trade.DatedOpenQuote(symbol=code, business_date="2026-09-24", open_price=71_000)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="exit", ledger=ledger, quote_fn=_attested, trading_day_fn=_open_day,
            now_fn=lambda: next(clocks),
        )
    )

    # Then: 체결 0, 주문은 UNFILLED
    assert n == 0
    assert (ledger.load("fills")["side"] == "buy").all()
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["UNFILLED"]


def test_run_paper_session_entry_catch_up_books_yesterday_never_today(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 2026-09-22 픽, 다음날 07:00 캐치업 기동 — main과 같은 방식으로 날짜 해소
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    now = pd.Timestamp("2026-09-23 07:00:00", tz="Asia/Seoul")
    decision_date = paper_trade.resolve_entry_decision_date(now)
    assert decision_date == pd.Timestamp("2026-09-22")
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0], "close": [70_500],
            "decided_at": [pd.Timestamp("2026-09-22 15:23:02", tz="Asia/Seoul")],
        })
        if d == pd.Timestamp("2026-09-22") else pd.DataFrame(),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930"], "종가": [70_500], "종가_확정": [True],
            "execution_timestamp": [pd.Timestamp("2026-09-22 15:30:20", tz="Asia/Seoul")],
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            decision_date, phase="entry", ledger=ledger, session=None, now_fn=lambda: now,
        )
    )

    # Then: 체결의 decision_date는 어제, 오늘자 decisions 행은 없다
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    assert fills["decision_date"].tolist() == ["2026-09-22"]
    assert fills["order_id"].tolist() == ["2026-09-22:005930:entry"]
    assert not (tmp_path / "decisions.parquet").exists()


def test_run_paper_session_entry_after_deadline_alerts_without_writing(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: D 픽이 있지만 D+1 08:30 마감에 걸린 기동
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({"symbol": ["005930", "000660"], "allocation": [0.5, 0.5], "price": [70_000, 200_000]}),
    )
    ledger = PaperLedger(root=tmp_path)
    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-22"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-23 08:30:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then: 아무것도 쓰지 않고 NO_DECISION 알림
    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()
    assert not (tmp_path / "orders.parquet").exists()
    assert not (tmp_path / "decisions.parquet").exists()
    assert outcomes == [(("NO_DECISION",), {"run_date": "2026-09-22", "reason": "entry_window_expired", "metrics": {"n_picks": 2}})]


def test_run_paper_session_entry_after_deadline_without_picks_stays_quiet(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 마감 후 기동인데 픽도 없다 — 알림도 쓰기도 없다
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(paper_trade, "load_topk_decision", lambda _d: pd.DataFrame())
    ledger = PaperLedger(root=tmp_path)
    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-22"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-23 08:30:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then
    assert n == 0
    assert outcomes == []
    assert not (tmp_path / "fills.parquet").exists()


def test_run_paper_session_entry_before_window_skips_without_reading(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 당일자 명시 + 15:10 기동(결정창 종료 전)
    def _load(_d):
        raise AssertionError("before-window entry must not read decisions")

    monkeypatch.setattr(paper_trade, "load_topk_decision", _load)
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-22"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-22 15:10:00", tz="Asia/Seoul"),
        )
    )

    # Then: 대기 없이 스킵, 쓰기 없음
    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()
    assert not (tmp_path / "orders.parquet").exists()


def test_run_paper_session_entry_on_holiday_writes_no_no_decision_row(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 휴장일 15:34, 픽 없음
    monkeypatch.setattr(paper_trade, "load_topk_decision", lambda _d: pd.DataFrame())

    async def _holiday(_day: str) -> bool:
        return False

    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-24 15:34:00", tz="Asia/Seoul"),
            trading_day_fn=_holiday,
        )
    )

    # Then: decisions 원장 그대로(침묵의 무기록 금지 행을 휴장에 남기지 않는다)
    assert n == 0
    assert not (tmp_path / "decisions.parquet").exists()


def test_run_paper_session_entry_default_oracle_skips_holiday_quietly(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 기본 KIS 오라클 경로로 휴장 판정
    monkeypatch.setattr(paper_trade, "load_topk_decision", lambda _d: pd.DataFrame())

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def get_market_index_history(self, session, code, start, end):
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260923"}]}

    monkeypatch.setattr(paper_trade, "KisApiClient", _FakeClient)
    monkeypatch.setattr(paper_trade, "kis_data_client_kwargs", lambda: {})
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-24"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-24 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 0
    assert not (tmp_path / "decisions.parquet").exists()


def test_run_paper_session_entry_sizes_quantity_from_decision_price(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 결정가 10000원, 확정 종가 11000원 — 실계좌는 결정 전에 수량을 확정해야 한다
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0 / 3.0], "close": [10000],
            "pred": [0.5],
            "decided_at": [pd.Timestamp("2026-09-25 15:23:02", tz="Asia/Seoul")],
        }),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930"], "종가": [11000], "종가_확정": [True],
            "execution_timestamp": [pd.Timestamp("2026-09-25 15:30:20", tz="Asia/Seoul")],
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then: 수량은 결정가 기준 floor(3,333,212/10000)=333, 체결가는 확정 종가 11000
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    assert int(fills.iloc[0]["qty"]) == 333
    assert int(fills.iloc[0]["fill_price"]) == 11000


def test_run_paper_session_entry_sizing_buffer_shrinks_quantity(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger, sizing_price

    # Given: 100bp 버퍼 — 사이징가 10100원
    assert sizing_price(10000, 100.0) == 10100
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(paper_trade.settings, "PAPER_ENTRY_SIZING_BUFFER_BP", 100.0)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0 / 3.0], "close": [10000],
            "pred": [0.5],
            "decided_at": [pd.Timestamp("2026-09-25 15:23:02", tz="Asia/Seoul")],
        }),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930"], "종가": [10100], "종가_확정": [True],
            "execution_timestamp": [pd.Timestamp("2026-09-25 15:30:20", tz="Asia/Seoul")],
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then: floor(3,333,212/10100)=330주 — 버퍼 없이 333주보다 결정적으로 적다
    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    assert int(fills.iloc[0]["qty"]) == 330


def test_run_paper_session_entry_decided_after_auction_close_is_never_filled(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 경매 종료(15:30:00) 이후에 끝난 예측
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5], "close": [70_000, 200_000],
            "pred": [0.9, 0.1],
            "decided_at": [pd.Timestamp("2026-09-25 15:30:05", tz="Asia/Seoul")] * 2,
        }),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930", "000660"], "종가": [70_500, 205_000], "종가_확정": [True, True],
            "execution_timestamp": [pd.Timestamp("2026-09-25 15:30:20", tz="Asia/Seoul")] * 2,
        }),
    )
    ledger = PaperLedger(root=tmp_path)
    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then: 전량 MISSED_AUCTION, 체결 없음, NO_DECISION 알림
    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert len(orders) == 2
    assert set(orders["status"]) == {"MISSED_AUCTION"}
    assert outcomes == [(("NO_DECISION",), {"run_date": "2026-09-25", "reason": "decided_after_auction_close"})]


def test_run_paper_session_entry_missing_decided_at_fails_closed(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: decided_at 컬럼 자체가 없는 픽
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0], "close": [70_000], "pred": [0.5],
        }),
    )
    ledger = PaperLedger(root=tmp_path)
    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then: fail-closed — MISSED_AUCTION 기록 후 종료
    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["MISSED_AUCTION"]
    assert outcomes == [(("NO_DECISION",), {"run_date": "2026-09-25", "reason": "decided_after_auction_close"})]


def test_run_paper_session_entry_cash_guard_leaves_unfundable_lot_open(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 결정가 기준 딱 3개 로트분의 현금, 최상위 픽의 확정 종가는 5% 상승
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["AAA111", "BBB222", "CCC333"],
            "allocation": [1.0 / 3.0] * 3,
            "close": [10000, 10000, 10000],
            "pred": [0.9, 0.5, 0.1],
            "decided_at": [pd.Timestamp("2026-09-25 15:23:02", tz="Asia/Seoul")] * 3,
        }),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["AAA111", "BBB222", "CCC333"],
            "종가": [10500, 10000, 10000],
            "종가_확정": [True, True, True],
            "execution_timestamp": [pd.Timestamp("2026-09-25 15:30:20", tz="Asia/Seoul")] * 3,
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When: pred 내림차순으로 체결 시도
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then: 최하위 로트는 자금 부족으로 미체결, NAV 현금은 음수가 아니다
    assert n == 2
    orders = pd.read_parquet(tmp_path / "orders.parquet").set_index("symbol")
    assert orders.loc["AAA111", "status"] == "FILLED"
    assert orders.loc["BBB222", "status"] == "FILLED"
    assert orders.loc["CCC333", "status"] == "INSUFFICIENT_CASH"
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert int(nav.iloc[0]["cash"]) == 3_173_252
    assert int(nav.iloc[0]["cash"]) >= 0


def test_run_paper_session_entry_missing_decision_price_records_zero_qty(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 한 픽의 close가 NaN
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5], "close": [float("nan"), 200_000],
            "pred": [0.9, 0.1],
            "decided_at": [pd.Timestamp("2026-09-25 15:23:02", tz="Asia/Seoul")] * 2,
        }),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930", "000660"], "종가": [70_500, 205_000], "종가_확정": [True, True],
            "execution_timestamp": [pd.Timestamp("2026-09-25 15:30:20", tz="Asia/Seoul")] * 2,
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    with caplog.at_level(logging.WARNING, logger="src.daily.paper_trade"):
        n = asyncio.run(
            paper_trade.run_paper_session(
                pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
                now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
            )
        )

    # Then: NaN 픽은 ZERO_QTY, 정상 픽은 체결
    assert n == 1
    orders = pd.read_parquet(tmp_path / "orders.parquet").set_index("symbol")
    assert orders.loc["005930", "status"] == "ZERO_QTY"
    assert int(orders.loc["005930", "qty"]) == 0
    assert orders.loc["000660", "status"] == "FILLED"
    assert "reason=no_decision_price" in caplog.text


def test_run_paper_session_entry_unparseable_fields_fail_closed(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 숫자로 바꿀 수 없는 close 문자열 — 사이징 0으로 ZERO_QTY
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0], "close": ["not-a-price"],
            "pred": [0.5],
            "decided_at": [pd.Timestamp("2026-09-25 15:23:02", tz="Asia/Seoul")],
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then
    assert n == 0
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["ZERO_QTY"]


def test_run_paper_session_entry_garbage_decided_at_fails_closed(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 파싱 불가 decided_at — 시각 입증 불가이므로 경매를 놓친 것으로 간주
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930"], "allocation": [1.0], "close": [70_000],
            "pred": [0.5], "decided_at": ["garbage!!"],
        }),
    )
    ledger = PaperLedger(root=tmp_path)
    outcomes: list[tuple] = []

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
        )
    )

    # Then
    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert orders["status"].tolist() == ["MISSED_AUCTION"]
    assert outcomes == [(("NO_DECISION",), {"run_date": "2026-09-25", "reason": "decided_after_auction_close"})]


def test_run_paper_session_entry_missed_auction_records_zero_size_pick_as_missed(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: 경매 종료 후 결정 + 결정가 없는 픽 혼합 — 게이트가 우선한다
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5],
            "close": [70_000, float("nan")],
            "pred": [0.9, 0.1],
            "decided_at": [pd.Timestamp("2026-09-25 15:30:05", tz="Asia/Seoul")] * 2,
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then: 0주 픽도 MISSED_AUCTION으로 기록된다
    assert n == 0
    orders = pd.read_parquet(tmp_path / "orders.parquet").set_index("symbol")
    assert orders.loc["005930", "status"] == "MISSED_AUCTION"
    assert orders.loc["000660", "status"] == "MISSED_AUCTION"
    assert int(orders.loc["000660", "qty"]) == 0


def test_run_paper_session_entry_non_numeric_pred_sorts_last(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.execution.paper_broker import PaperLedger

    # Given: pred가 문자열·NaN인 픽 — 순위 계산 불가이므로 꼴찌로 처리하고 체결은 진행
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660"], "allocation": [0.5, 0.5],
            "close": [70_000, 200_000],
            "pred": ["strong", float("nan")],
            "decided_at": [pd.Timestamp("2026-09-25 15:23:02", tz="Asia/Seoul")] * 2,
        }),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930", "000660"], "종가": [70_500, 205_000], "종가_확정": [True, True],
            "execution_timestamp": [pd.Timestamp("2026-09-25 15:30:20", tz="Asia/Seoul")] * 2,
        }),
    )
    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-25"), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp("2026-09-25 15:34:00", tz="Asia/Seoul"),
        )
    )

    # Then: 자금이 충분하므로 둘 다 체결된다
    assert n == 2
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert set(orders["status"]) == {"FILLED"}


def _session_day_for(kind, trading_day):
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay

    clock = None if kind.value in ("CLOSED", "UNKNOWN") else SessionClock.standard(trading_day)
    return SessionDay(trading_date=trading_day, kind=kind, clock=clock, provenance="test")


def _entry_picks(monkeypatch, paper_trade, pd, date_str):
    monkeypatch.setattr(paper_trade.settings, "PAPER_SEED_CAPITAL", 10_000_000)
    monkeypatch.setattr(
        paper_trade, "load_topk_decision",
        lambda _d: pd.DataFrame({
            "symbol": ["005930", "000660"],
            "allocation": [0.5, 0.5],
            "close": [70000, 200000],
            "pred": [0.9, 0.5],
            "decided_at": [pd.Timestamp(f"{date_str} 15:23:02", tz="Asia/Seoul")] * 2,
        }),
    )
    monkeypatch.setattr(
        paper_trade, "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({
            "종목코드": ["005930", "000660"], "종가": [70_500, 205_000], "종가_확정": [True, True],
            "execution_timestamp": [pd.Timestamp(f"{date_str} 15:30:20", tz="Asia/Seoul")] * 2,
        }),
    )


def test_run_paper_session_exit_holds_lots_on_shifted_session(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.data.session_calendar import SessionKind
    from src.execution.paper_broker import PaperLedger

    kst = "Asia/Seoul"
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-11-18:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-11-18 15:30:20", tz=kst), "decision_date": "2026-11-18", "trigger": "auction_close"}],
        kind="fills",
    )

    async def _raising_quote(code: str):
        raise AssertionError("shifted hold must not quote")

    outcomes: list[tuple] = []
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-11-19"), phase="exit", ledger=ledger, quote_fn=_raising_quote,
            trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-11-19 09:01:00", tz=kst),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
            session_day_fn=lambda d: _session_day_for(SessionKind.SHIFTED, d),
        )
    )

    assert n == 0
    assert len(ledger.load_open_positions()) == 1
    assert not (tmp_path / "orders.parquet").exists()
    assert outcomes == [(("DEGRADED",), {"run_date": "2026-11-19", "reason": "shifted_session_hold", "metrics": {"n_open": 1}})]


def test_run_paper_session_exit_blocks_fills_on_calendar_disagreement(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.data.session_calendar import SessionKind
    from src.execution.paper_broker import PaperLedger

    kst = "Asia/Seoul"
    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": "2026-10-05:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp("2026-10-05 15:30:20", tz=kst), "decision_date": "2026-10-05", "trigger": "auction_close"}],
        kind="fills",
    )

    async def _raising_quote(code: str):
        raise AssertionError("disagreement must not quote")

    async def _closed(_day: str) -> bool:
        return False

    outcomes: list[tuple] = []
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-10-06"), phase="exit", ledger=ledger, quote_fn=_raising_quote,
            trading_day_fn=_closed,
            now_fn=lambda: pd.Timestamp("2026-10-06 09:01:00", tz=kst),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
            session_day_fn=lambda d: _session_day_for(SessionKind.STANDARD, d),
        )
    )

    assert n == 0
    assert len(ledger.load_open_positions()) == 1
    assert outcomes == [(("DEGRADED",), {"run_date": "2026-10-06", "reason": "calendar_disagreement", "metrics": {"n_open": 1}})]


def test_run_paper_session_entry_marks_missed_auction_on_shifted_session(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.data.session_calendar import SessionKind
    from src.execution.paper_broker import PaperLedger

    date_str = "2026-11-19"
    _entry_picks(monkeypatch, paper_trade, pd, date_str)
    ledger = PaperLedger(root=tmp_path)
    outcomes: list[tuple] = []
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp(date_str), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp(f"{date_str} 15:34:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
            session_day_fn=lambda d: _session_day_for(SessionKind.SHIFTED, d),
        )
    )

    assert n == 0
    assert not (tmp_path / "fills.parquet").exists()
    orders = pd.read_parquet(tmp_path / "orders.parquet")
    assert (orders["status"] == "MISSED_AUCTION").all()
    assert outcomes == [(("NO_DECISION",), {"run_date": date_str, "reason": "session_shifted"})]


def test_run_paper_session_entry_writes_nothing_on_closed_day(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.data.session_calendar import SessionKind
    from src.execution.paper_broker import PaperLedger

    date_str = "2026-10-09"
    _entry_picks(monkeypatch, paper_trade, pd, date_str)
    ledger = PaperLedger(root=tmp_path)
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp(date_str), phase="entry", ledger=ledger, session=None,
            now_fn=lambda: pd.Timestamp(f"{date_str} 15:34:00", tz="Asia/Seoul"),
            session_day_fn=lambda d: _session_day_for(SessionKind.CLOSED, d),
        )
    )

    assert n == 0
    assert not (tmp_path / "orders.parquet").exists()
    assert not (tmp_path / "fills.parquet").exists()


def _open_lot_ledger(tmp_path, pd, decision_date: str, fill_date: str):
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)
    ledger.record(
        [{"order_id": f"{fill_date}:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
          "filled_at": pd.Timestamp(f"{fill_date} 15:30:20", tz="Asia/Seoul"), "decision_date": fill_date,
          "trigger": "auction_close"}],
        kind="fills",
    )
    return ledger


def test_run_paper_session_exit_blocks_fills_on_closed_calendar_with_kis_trading(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.data.session_calendar import SessionKind

    ledger = _open_lot_ledger(tmp_path, pd, "2026-10-09", "2026-10-08")

    async def _raising_quote(code: str):
        raise AssertionError("disagreement must not quote")

    outcomes: list[tuple] = []
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-10-09"), phase="exit", ledger=ledger, quote_fn=_raising_quote,
            trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-10-09 09:01:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
            session_day_fn=lambda d: _session_day_for(SessionKind.CLOSED, d),
        )
    )

    assert n == 0
    assert len(ledger.load_open_positions()) == 1
    assert outcomes == [(("DEGRADED",), {"run_date": "2026-10-09", "reason": "calendar_disagreement", "metrics": {"n_open": 1}})]


def test_run_paper_session_exit_silently_skips_confirmed_holiday(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd

    from src.daily import paper_trade
    from src.data.session_calendar import SessionKind

    ledger = _open_lot_ledger(tmp_path, pd, "2026-10-09", "2026-10-08")

    async def _closed(_day: str) -> bool:
        return False

    outcomes: list[tuple] = []
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-10-09"), phase="exit", ledger=ledger, quote_fn=None,
            trading_day_fn=_closed,
            now_fn=lambda: pd.Timestamp("2026-10-09 09:01:00", tz="Asia/Seoul"),
            record_fn=lambda *a, **k: outcomes.append((a, k)),
            session_day_fn=lambda d: _session_day_for(SessionKind.CLOSED, d),
        )
    )

    assert n == 0
    assert len(ledger.load_open_positions()) == 1
    assert outcomes == []
