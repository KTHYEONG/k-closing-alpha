from __future__ import annotations


def test_size_order_qty_floors_to_whole_shares() -> None:
    import pytest

    from src.execution.paper_broker import size_order_qty

    # Given / When / Then: 10,000,000 * 1/3 / 33,000 = 101.01... -> 101주
    assert size_order_qty(10_000_000, 1 / 3, 33_000) == 101
    # And: 시드가 1주 값에도 못 미치면 0주
    assert size_order_qty(10_000, 1.0, 33_000) == 0
    # And: 비정상 입력은 fail-closed
    with pytest.raises(ValueError, match="price"):
        size_order_qty(10_000_000, 0.5, 0)
    with pytest.raises(ValueError, match="allocation"):
        size_order_qty(10_000_000, 0.0, 1_000)


def test_decide_fill_buy_limit_and_market_paths() -> None:
    import pandas as pd

    from src.execution.paper_broker import PaperOrder, decide_fill

    placed = pd.Timestamp("2026-09-10 15:19:00", tz="Asia/Seoul")
    limit_order = PaperOrder(
        order_id="o1", decision_date="2026-09-10", symbol="005930", side="buy",
        qty=10, limit_price=70_000, placed_at=placed, reason="entry",
    )

    # When: 지정가보다 비싼 프린트 -> 미체결
    assert decide_fill(limit_order, 70_100, placed + pd.Timedelta(seconds=1)) is None

    # When: 지정가 이하 프린트 -> 체결, 체결가는 원시 프린트 그대로(수수료 미가산)
    fill = decide_fill(limit_order, 69_900, placed + pd.Timedelta(seconds=2))
    assert fill is not None
    assert fill.fill_price == 69_900
    assert fill.order_id == "o1"
    assert fill.qty == 10

    # And: 시장가 등가(limit_price=None)는 첫 프린트에 즉시 체결
    mkt_order = PaperOrder(
        order_id="o2", decision_date="2026-09-10", symbol="005930", side="buy",
        qty=10, limit_price=None, placed_at=placed, reason="entry",
    )
    mkt_fill = decide_fill(mkt_order, 71_500, placed + pd.Timedelta(seconds=1))
    assert mkt_fill is not None
    assert mkt_fill.fill_price == 71_500


def test_decide_fill_sell_take_profit_threshold() -> None:
    import pandas as pd

    from src.execution.paper_broker import PaperOrder, decide_fill

    placed = pd.Timestamp("2026-09-11 09:00:00", tz="Asia/Seoul")
    order = PaperOrder(
        order_id="s1", decision_date="2026-09-10", symbol="005930", side="sell",
        qty=10, limit_price=73_500, placed_at=placed, reason="take_profit",
    )

    # When: 지정가 미만 -> 미체결
    assert decide_fill(order, 73_400, placed + pd.Timedelta(seconds=1)) is None

    # When: 지정가 이상 -> 체결
    fill = decide_fill(order, 73_600, placed + pd.Timedelta(seconds=5))
    assert fill is not None
    assert fill.fill_price == 73_600
    assert fill.side == "sell"


def test_decide_fill_rejects_print_before_order_placement() -> None:
    import pandas as pd
    import pytest

    from src.execution.paper_broker import PaperOrder, decide_fill

    placed = pd.Timestamp("2026-09-10 15:19:00", tz="Asia/Seoul")
    order = PaperOrder(
        order_id="o1", decision_date="2026-09-10", symbol="005930", side="buy",
        qty=10, limit_price=None, placed_at=placed, reason="entry",
    )

    # When / Then: 주문보다 이른 프린트는 fail-closed
    with pytest.raises(ValueError, match="placed_at"):
        decide_fill(order, 70_000, placed - pd.Timedelta(seconds=1))


def test_paper_ledger_records_and_rejects_unknown_kind(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)

    # When: fills 기록
    n = ledger.record(
        [{"order_id": "o1", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000}],
        kind="fills",
    )
    assert n == 1
    assert (tmp_path / "fills.parquet").exists()

    # And: 같은 order_id 재기록은 중복되지 않는다
    ledger.record(
        [{"order_id": "o1", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000}],
        kind="fills",
    )
    assert len(pd.read_parquet(tmp_path / "fills.parquet")) == 1

    # And: 알 수 없는 kind는 fail-closed
    with pytest.raises(ValueError, match="kind"):
        ledger.record([{"order_id": "x"}], kind="trades")


def test_paper_ledger_record_no_decision_is_explicit(tmp_path) -> None:
    import pandas as pd

    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)

    # When
    n = ledger.record_no_decision("2026-09-10", reason="admitted_below_top_k")

    # Then: 침묵이 아니라 행으로 남는다
    assert n == 1
    df = pd.read_parquet(tmp_path / "decisions.parquet")
    assert len(df) == 1
    assert df.iloc[0]["decision_date"] == "2026-09-10"
    assert df.iloc[0]["symbol"] == ""
    assert df.iloc[0]["reason"] == "admitted_below_top_k"


def test_load_open_positions_excludes_closed_and_returns_schema(tmp_path) -> None:
    from src.execution.paper_broker import PaperLedger

    # Given: 원장 파일이 아직 없다
    ledger = PaperLedger(root=tmp_path)
    empty = ledger.load_open_positions()

    # Then: 스키마는 고정이고 비어 있다
    assert list(empty.columns) == ["symbol", "qty", "entry_price", "decision_date"]
    assert len(empty) == 0

    # When: 두 종목 매수 후 한 종목만 매도 체결
    ledger.record(
        [
            {"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
             "fill_price": 70_000, "decision_date": "2026-09-10"},
            {"order_id": "b2", "symbol": "000660", "side": "buy", "qty": 5,
             "fill_price": 200_000, "decision_date": "2026-09-10"},
            {"order_id": "s1", "symbol": "000660", "side": "sell", "qty": 5,
             "fill_price": 210_000, "decision_date": "2026-09-10"},
        ],
        kind="fills",
    )
    open_pos = ledger.load_open_positions()

    # Then: 청산된 종목은 빠지고 미청산만 남는다
    assert list(open_pos["symbol"]) == ["005930"]
    assert int(open_pos.iloc[0]["qty"]) == 10
    assert int(open_pos.iloc[0]["entry_price"]) == 70_000


def test_load_open_positions_survives_symbol_reentry_after_close(tmp_path) -> None:
    from src.execution.paper_broker import PaperLedger

    ledger = PaperLedger(root=tmp_path)

    # Given: 진입 후 익일 청산 완료
    ledger.record(
        [{"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10,
          "fill_price": 70_000, "decision_date": "2026-09-01"}],
        kind="fills",
    )
    ledger.record(
        [{"order_id": "s1", "symbol": "005930", "side": "sell", "qty": 10,
          "fill_price": 73_500, "decision_date": "2026-09-02"}],
        kind="fills",
    )
    assert len(ledger.load_open_positions()) == 0

    # When: 같은 종목을 나중에 재진입한다(측정된 종목 중복률이 높아 흔한 경로)
    ledger.record(
        [{"order_id": "b2", "symbol": "005930", "side": "buy", "qty": 8,
          "fill_price": 71_000, "decision_date": "2026-09-08"}],
        kind="fills",
    )
    open_pos = ledger.load_open_positions()

    # Then: 과거 청산 이력에 가려 사라지지 않고 신규 포지션이 남는다
    assert len(open_pos) == 1
    assert open_pos.iloc[0]["symbol"] == "005930"
    assert int(open_pos.iloc[0]["qty"]) == 8
    assert int(open_pos.iloc[0]["entry_price"]) == 71_000
    assert open_pos.iloc[0]["decision_date"] == "2026-09-08"
