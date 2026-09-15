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
        ledger.record([{"order_id": "x"}], kind="positions")


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


def test_build_auction_fill_confirms_only_when_close_confirmed() -> None:
    import pandas as pd

    from src.execution.paper_broker import PaperOrder, build_auction_fill

    placed_at = pd.Timestamp("2026-09-10 15:20:00", tz="Asia/Seoul")
    order = PaperOrder(
        order_id="2026-09-10:005930:entry", decision_date="2026-09-10", symbol="005930",
        side="buy", qty=10, limit_price=None, placed_at=placed_at, reason="entry",
    )

    # Given: 미확정 행
    unconfirmed = {"종가_확정": False, "종가": 70_000, "execution_timestamp": placed_at + pd.Timedelta(minutes=11)}
    assert build_auction_fill(order, unconfirmed) is None

    # When: 확정 행
    confirmed = {"종가_확정": True, "종가": 70_500, "execution_timestamp": placed_at + pd.Timedelta(minutes=11)}
    fill = build_auction_fill(order, confirmed)

    # Then: 확정 종가 그대로 체결
    assert fill is not None
    assert fill.fill_price == 70_500
    assert fill.trigger == "auction_close"
    assert fill.symbol == "005930"


def test_build_auction_fill_treats_nan_confirmation_as_unconfirmed() -> None:
    import math

    import pandas as pd

    from src.execution.paper_broker import PaperOrder, build_auction_fill

    placed_at = pd.Timestamp("2026-09-10 15:20:00", tz="Asia/Seoul")
    order = PaperOrder(
        order_id="2026-09-10:005930:entry", decision_date="2026-09-10", symbol="005930",
        side="buy", qty=10, limit_price=None, placed_at=placed_at, reason="entry",
    )

    # Given: fetch_archive_snapshot이 실제로 반환하는 float64 컬럼 형태
    # (bool True/False가 NaN 혼재 컬럼에서 1.0/0.0/NaN으로 업캐스트됨).
    # not float('nan')은 False라 NaN이 확정으로 오판될 수 있는 함정 케이스.
    row = {"종가_확정": math.nan, "종가": 70_500, "execution_timestamp": placed_at + pd.Timedelta(minutes=11)}

    # When / Then: 확정 여부 불명은 미확정과 동일하게 취급되어 체결하지 않는다
    assert build_auction_fill(order, row) is None


def test_build_auction_fill_rejects_lookahead_and_nonpositive_price() -> None:
    import pandas as pd
    import pytest

    from src.execution.paper_broker import PaperOrder, build_auction_fill

    placed_at = pd.Timestamp("2026-09-10 15:20:00", tz="Asia/Seoul")
    order = PaperOrder(
        order_id="o1", decision_date="2026-09-10", symbol="005930",
        side="buy", qty=10, limit_price=None, placed_at=placed_at, reason="entry",
    )

    # When / Then: 확정시각이 주문시각보다 이르면 룩어헤드 금지 위반
    stale = {"종가_확정": True, "종가": 70_000, "execution_timestamp": placed_at - pd.Timedelta(seconds=1)}
    with pytest.raises(ValueError, match="placed_at"):
        build_auction_fill(order, stale)

    # And: 종가<=0은 거부
    zero_price = {"종가_확정": True, "종가": 0, "execution_timestamp": placed_at + pd.Timedelta(minutes=10)}
    with pytest.raises(ValueError, match="종가"):
        build_auction_fill(order, zero_price)


def test_order_record_carries_terminal_status_and_rejects_unknown() -> None:
    import pandas as pd
    import pytest

    from src.execution.paper_broker import ORDER_STATUS_FILLED, ORDER_STATUSES, PaperOrder, order_record

    # Given
    placed = pd.Timestamp("2026-09-15 15:20:00", tz="Asia/Seoul")
    order = PaperOrder(
        order_id="2026-09-15:005930:entry", decision_date="2026-09-15", symbol="005930",
        side="buy", qty=35, limit_price=None, placed_at=placed, reason="entry",
    )

    # When
    row = order_record(order, ORDER_STATUS_FILLED)

    # Then
    assert row["order_id"] == "2026-09-15:005930:entry"
    assert row["decision_date"] == "2026-09-15"
    assert row["symbol"] == "005930"
    assert row["side"] == "buy"
    assert row["qty"] == 35
    assert row["limit_price"] is None
    assert row["placed_at"] == placed
    assert row["reason"] == "entry"
    assert row["status"] == "FILLED"
    assert pd.Timestamp(row["recorded_at"]).tzinfo is not None
    assert set(ORDER_STATUSES) == {"FILLED", "UNCONFIRMED", "NO_SNAPSHOT_ROW", "ZERO_QTY", "UNFILLED"}

    # And: 미정의 상태는 거부
    with pytest.raises(ValueError, match="status"):
        order_record(order, "PARTIAL")


def test_build_round_trips_pairs_fifo_and_applies_explicit_costs() -> None:
    import pandas as pd
    import pytest

    from src.execution.paper_broker import ROUND_TRIP_COLUMNS, build_round_trips

    kst = "Asia/Seoul"
    # Given: 동일 종목 진입-청산이 두 번 반복된 원장(기록순)
    fills = pd.DataFrame([
        {"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
         "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz=kst), "decision_date": "2026-09-10", "trigger": "auction_close"},
        {"order_id": "2026-09-11:005930:exit", "symbol": "005930", "side": "sell", "qty": 10, "fill_price": 73_600,
         "filled_at": pd.Timestamp("2026-09-11 09:31:00", tz=kst), "decision_date": "2026-09-11", "trigger": "limit"},
        {"order_id": "2026-09-11:005930:entry", "symbol": "005930", "side": "buy", "qty": 5, "fill_price": 72_000,
         "filled_at": pd.Timestamp("2026-09-11 15:30:20", tz=kst), "decision_date": "2026-09-11", "trigger": "auction_close"},
        {"order_id": "2026-09-14:005930:exit", "symbol": "005930", "side": "sell", "qty": 5, "fill_price": 71_000,
         "filled_at": pd.Timestamp("2026-09-14 15:19:05", tz=kst), "decision_date": "2026-09-14", "trigger": "market"},
    ])

    # When
    trips = build_round_trips(fills)

    # Then: FIFO 페어링
    assert list(trips.columns) == list(ROUND_TRIP_COLUMNS)
    assert trips["entry_order_id"].tolist() == ["2026-09-10:005930:entry", "2026-09-11:005930:entry"]
    assert trips["exit_order_id"].tolist() == ["2026-09-11:005930:exit", "2026-09-14:005930:exit"]

    # And: 수수료 편도 0.36396bp 원미만 절사, 2026 매도세 20bp
    first = trips.iloc[0]
    assert first["decision_date"] == "2026-09-10"
    assert int(first["qty"]) == 10
    assert int(first["entry_price"]) == 70_000 and int(first["exit_price"]) == 73_600
    assert first["exit_trigger"] == "limit"
    assert int(first["gross_pnl"]) == 36_000
    assert int(first["buy_fee"]) == 25
    assert int(first["sell_fee"]) == 26
    assert int(first["sell_tax"]) == 1_472
    assert int(first["cost"]) == 1_523
    assert int(first["net_pnl"]) == 34_477
    assert first["gross_ret"] == pytest.approx(73_600 / 70_000 - 1.0)
    assert first["net_ret"] == pytest.approx(34_477 / 700_000)

    second = trips.iloc[1]
    assert int(second["buy_fee"]) == 13
    assert int(second["sell_fee"]) == 12
    assert int(second["sell_tax"]) == 710
    assert int(second["net_pnl"]) == -5_000 - 735


def test_build_round_trips_fails_closed_on_inconsistent_ledger() -> None:
    import pandas as pd
    import pytest

    from src.execution.paper_broker import ROUND_TRIP_COLUMNS, build_round_trips

    kst = "Asia/Seoul"
    buy = {"order_id": "b1", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
           "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz=kst), "decision_date": "2026-09-10", "trigger": "auction_close"}
    sell = {"order_id": "s1", "symbol": "005930", "side": "sell", "qty": 10, "fill_price": 73_600,
            "filled_at": pd.Timestamp("2026-09-11 09:31:00", tz=kst), "decision_date": "2026-09-11", "trigger": "limit"}

    # Then: 매수 없는 매도
    with pytest.raises(ValueError, match="without an open buy"):
        build_round_trips(pd.DataFrame([sell]))
    # Then: 수량 불일치(부분청산 미지원)
    with pytest.raises(ValueError, match="qty"):
        build_round_trips(pd.DataFrame([buy, {**sell, "qty": 4}]))
    # Then: 청산 체결시각 없음(세율 기준일 불명)
    with pytest.raises(ValueError, match="filled_at"):
        build_round_trips(pd.DataFrame([buy, {**sell, "filled_at": pd.NaT}]))
    # Then: 미정의 side
    with pytest.raises(ValueError, match="side"):
        build_round_trips(pd.DataFrame([{**buy, "side": "short"}]))

    # And: 빈 원장/미청산만 있는 원장은 빈 프레임(스키마 유지)
    assert list(build_round_trips(pd.DataFrame()).columns) == list(ROUND_TRIP_COLUMNS)
    assert build_round_trips(pd.DataFrame([buy])).empty


def test_build_nav_snapshot_reconciles_accounting_identity() -> None:
    import pandas as pd

    from src.execution.paper_broker import NAV_COLUMNS, build_nav_snapshot

    kst = "Asia/Seoul"
    # Given: 청산 완료 1건 + 미청산 1건
    fills = pd.DataFrame([
        {"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
         "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz=kst), "decision_date": "2026-09-10", "trigger": "auction_close"},
        {"order_id": "2026-09-11:005930:exit", "symbol": "005930", "side": "sell", "qty": 10, "fill_price": 73_600,
         "filled_at": pd.Timestamp("2026-09-11 09:31:00", tz=kst), "decision_date": "2026-09-11", "trigger": "limit"},
        {"order_id": "2026-09-11:000660:entry", "symbol": "000660", "side": "buy", "qty": 5, "fill_price": 100_000,
         "filled_at": pd.Timestamp("2026-09-11 15:30:20", tz=kst), "decision_date": "2026-09-11", "trigger": "auction_close"},
    ])

    # When
    nav = build_nav_snapshot(fills, seed_capital=10_000_000, as_of_date="2026-09-11")

    # Then
    assert list(nav.columns) == list(NAV_COLUMNS)
    assert len(nav) == 1
    row = nav.iloc[0]
    assert row["as_of_date"] == "2026-09-11"
    assert int(row["seed_capital"]) == 10_000_000
    assert int(row["cash"]) == 9_534_459
    assert int(row["open_cost_basis"]) == 500_000
    assert int(row["open_buy_fees"]) == 18
    assert int(row["realized_net_pnl"]) == 34_477
    assert int(row["cumulative_cost"]) == 1_541
    assert int(row["nav"]) == 10_034_459
    assert int(row["n_open_positions"]) == 1
    assert int(row["n_closed_trades"]) == 1
    # And: 회계 항등식(원 단위 오차 0)
    assert int(row["nav"]) == int(row["cash"]) + int(row["open_cost_basis"])
    assert int(row["nav"]) == 10_000_000 + int(row["realized_net_pnl"]) - int(row["open_buy_fees"])

    # And: 체결 없음 -> 현금=NAV=시드
    empty = build_nav_snapshot(pd.DataFrame(), seed_capital=10_000_000, as_of_date="2026-09-11").iloc[0]
    assert int(empty["cash"]) == 10_000_000
    assert int(empty["nav"]) == 10_000_000
    assert int(empty["n_open_positions"]) == 0


def test_refresh_trade_ledgers_writes_trades_and_nav_idempotently(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.execution.paper_broker import PaperLedger, refresh_trade_ledgers

    kst = "Asia/Seoul"
    buy = {"order_id": "2026-09-10:005930:entry", "symbol": "005930", "side": "buy", "qty": 10, "fill_price": 70_000,
           "filled_at": pd.Timestamp("2026-09-10 15:30:20", tz=kst), "decision_date": "2026-09-10", "trigger": "auction_close"}
    sell = {"order_id": "2026-09-11:005930:exit", "symbol": "005930", "side": "sell", "qty": 10, "fill_price": 73_600,
            "filled_at": pd.Timestamp("2026-09-11 09:31:00", tz=kst), "decision_date": "2026-09-11", "trigger": "limit"}
    ledger = PaperLedger(root=tmp_path)

    # Given: 원장 파일이 없으면 load는 빈 프레임, 미정의 kind는 거부
    assert ledger.load("fills").empty
    with pytest.raises(ValueError, match="kind"):
        ledger.load("positions")
    ledger.record([buy, sell], kind="fills")

    # When: 같은 날 두 번 갱신
    n1 = refresh_trade_ledgers(ledger, seed_capital=10_000_000, as_of_date="2026-09-11")
    n2 = refresh_trade_ledgers(ledger, seed_capital=10_000_000, as_of_date="2026-09-11")

    # Then: 멱등(중복 왕복/NAV 행 없음)
    assert n1 == 1 and n2 == 1
    trades = pd.read_parquet(tmp_path / "trades.parquet")
    assert len(trades) == 1
    assert int(trades.iloc[0]["net_pnl"]) == 34_477
    nav = pd.read_parquet(tmp_path / "nav.parquet")
    assert len(nav) == 1
    assert int(nav.iloc[0]["nav"]) == 10_034_477
    assert ledger.load("trades")["exit_order_id"].tolist() == ["2026-09-11:005930:exit"]

    # And: 미청산만 있으면 trades 파일은 만들지 않고 NAV만 기록
    solo_root = tmp_path / "solo"
    solo_root.mkdir()
    solo = PaperLedger(root=solo_root)
    solo.record([buy], kind="fills")
    assert refresh_trade_ledgers(solo, seed_capital=10_000_000, as_of_date="2026-09-10") == 0
    assert not (solo_root / "trades.parquet").exists()
    solo_nav = pd.read_parquet(solo_root / "nav.parquet")
    assert int(solo_nav.iloc[0]["n_open_positions"]) == 1
    assert int(solo_nav.iloc[0]["cash"]) == 10_000_000 - 700_025
