from __future__ import annotations


async def _open_day(_day: str) -> bool:
    return True


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
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None
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


    ledger = PaperLedger(root=tmp_path)

    # When
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=object()
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
                pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None
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
                pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None
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
                pd.Timestamp("2026-09-14"), phase="entry", ledger=ledger, session=None
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
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None
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
        lambda _d: pd.DataFrame({"symbol": ["005930"], "allocation": [1.0], "price": [70_000]}),
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
            pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None
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
        lambda _d: pd.DataFrame({"symbol": ["005930"], "allocation": [1.0], "price": [70_000]}),
    )
    ts = pd.Timestamp("2026-09-10 15:30:20", tz=kst)
    monkeypatch.setattr(
        paper_trade,
        "fetch_archive_snapshot",
        lambda _d: pd.DataFrame({"종목코드": ["005930"], "종가": [70_500], "종가_확정": [1.0], "execution_timestamp": [ts]}),
    )

    # When: ExecStopPost 체인과 백스톱 타이머가 둘 다 같은 날 entry를 트리거한다
    first = asyncio.run(paper_trade.run_paper_session(pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None))
    second = asyncio.run(paper_trade.run_paper_session(pd.Timestamp("2026-09-10"), phase="entry", ledger=ledger, session=None))

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

    async def _quote(code: str) -> int:
        asked.append(code)
        return 71_000

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

    async def _quote(code: str) -> int:
        calls[code] += 1
        # 005930은 두 번째 조회에서 시가 형성, 000660은 거래정지로 끝내 0
        if code == "005930" and calls[code] >= 2:
            return 71_000
        return 0

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

    async def _quote(code: str) -> int:
        events.append("quote")
        return 71_000

    async def _sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    # When: 08:59:00 기동(부팅 캐치업 등) -> 09:00:30까지 대기
    n = asyncio.run(
        paper_trade.run_paper_session(
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, quote_fn=_quote, trading_day_fn=_open_day,
            now_fn=lambda: pd.Timestamp("2026-09-11 08:59:00", tz="Asia/Seoul"), sleep_fn=_sleep,
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

    async def _quote(code: str) -> int:
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

    async def _quote(code: str) -> int:
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

    async def _quote(code: str) -> int:
        asked.append(code)
        return 72_000

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


def test_fetch_krx_open_quote_reads_stck_oprc_and_fails_soft() -> None:
    import asyncio

    from src.daily.paper_trade import fetch_krx_open_quote

    class _Client:
        def __init__(self, payload):
            self.payload = payload
            self.calls: list[tuple] = []

        async def get_current_price(self, session, code, market_div_code=None, allow_market_div_fallback=True):
            self.calls.append((session, code, market_div_code, allow_market_div_fallback))
            return self.payload

    session = object()
    ok = _Client({"rt_cd": "0", "output": {"stck_oprc": "71000"}})

    # When
    price = asyncio.run(fetch_krx_open_quote(ok, session, "005930"))

    # Then: NXT 통합시가가 섞이지 않도록 KRX(J) 고정, 폴백 금지
    assert price == 71_000
    assert ok.calls == [(session, "005930", "J", False)]

    # And: 실패 응답/비정상 payload/미형성 시가는 0
    assert asyncio.run(fetch_krx_open_quote(_Client({"rt_cd": "1", "output": {"stck_oprc": "71000"}}), session, "005930")) == 0
    assert asyncio.run(fetch_krx_open_quote(_Client({"rt_cd": "0", "output": None}), session, "005930")) == 0
    assert asyncio.run(fetch_krx_open_quote(_Client({"rt_cd": "0", "output": {}}), session, "005930")) == 0


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

    async def _stale_quote(code: str) -> int:
        asked.append(code)
        return 69_000

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

    async def _quote(code: str) -> int:
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
