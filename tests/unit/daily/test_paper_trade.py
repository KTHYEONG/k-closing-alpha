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
        paper_trade, "run_topk_ranker_sleeve", lambda _d: pd.DataFrame()
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
        "run_topk_ranker_sleeve",
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
        "run_topk_ranker_sleeve",
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
        "run_topk_ranker_sleeve",
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
            pd.Timestamp("2026-09-11"), phase="exit", ledger=ledger, ws_client=_FakeWs(), session=object()
        )
    )

    assert n == 1
    fills = pd.read_parquet(tmp_path / "fills.parquet")
    sells = fills[fills["side"] == "sell"]
    assert int(sells.iloc[0]["fill_price"]) == 73_600
