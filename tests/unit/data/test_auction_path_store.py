"""Invariant guards for the closing-auction indicative path archive."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.data import auction_path_store as store
from src.data.auction_path_store import (
    AUCTION_PATH_COLUMNS,
    AuctionPathRowKind,
    auction_path_partition_path,
    build_auction_path_rows,
    write_auction_path_partition,
)

SEOUL = ZoneInfo("Asia/Seoul")
DATE = "2026-10-15"
AT = datetime(2026, 10, 15, 16, 0, tzinfo=SEOUL)


def _row(hour: str, price: str = "70000", buy: str = "0", sell: str = "0") -> dict[str, str]:
    return {"Hour": hour, "Prpr": price, "ShnuCnqn": buy, "SelnCnqn": sell}


def test_auction_path_rows_are_chronological_and_keep_same_second_order() -> None:
    raw = [_row("152959", price="102"), _row("152959", price="101"), _row("152958", price="100")]

    frame = build_auction_path_rows(raw, snapshot_date=DATE, symbol="005930", available_at=AT)

    assert frame["ts_hms"].tolist() == [152958, 152959, 152959]
    assert frame["price"].tolist() == [100, 101, 102]
    assert frame["seq"].tolist() == [0, 1, 2]


def test_auction_path_classifies_indicative_fill_and_unexpected() -> None:
    raw = [
        _row("152600", buy="5"),
        _row("153001", sell="2380168"),
        _row("152500"),
    ]

    frame = build_auction_path_rows(raw, snapshot_date=DATE, symbol="005930", available_at=AT)

    by_ts = {int(ts): kind for ts, kind in zip(frame["ts_hms"], frame["kind"], strict=True)}
    assert by_ts[152500] == AuctionPathRowKind.INDICATIVE.value
    assert by_ts[153001] == AuctionPathRowKind.AUCTION_FILL.value
    assert by_ts[152600] == AuctionPathRowKind.UNEXPECTED_TRADE.value


def test_auction_path_drops_rows_outside_window() -> None:
    raw = [_row("151959"), _row("153100", sell="10"), _row("155953", sell="10")]

    frame = build_auction_path_rows(raw, snapshot_date=DATE, symbol="005930", available_at=AT)

    assert frame.empty


def test_auction_path_drops_zero_qty_post_close_rows() -> None:
    frame = build_auction_path_rows([_row("153000")], snapshot_date=DATE, symbol="005930", available_at=AT)

    assert frame.empty


def test_auction_path_rejects_non_integer_price() -> None:
    with pytest.raises(ValueError, match=r"005930.*Prpr"):
        build_auction_path_rows([_row("152500", price="1.5")], snapshot_date=DATE, symbol="005930", available_at=AT)


def test_auction_path_partition_write_is_atomic_replace(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(store.settings, "HISTORY_DIR", tmp_path / "history")
    first = build_auction_path_rows([_row("152500")], snapshot_date=DATE, symbol="005930", available_at=AT)
    second = build_auction_path_rows(
        [_row("152500"), _row("152501")], snapshot_date=DATE, symbol="005930", available_at=AT
    )

    assert write_auction_path_partition(first, DATE) == 1
    assert write_auction_path_partition(second, DATE) == 2
    stored = pd.read_parquet(auction_path_partition_path(DATE))

    assert len(stored) == 2
    assert list(stored.columns) == list(AUCTION_PATH_COLUMNS)
    assert write_auction_path_partition(pd.DataFrame(), DATE) == 0
    assert len(pd.read_parquet(auction_path_partition_path(DATE))) == 2


def test_auction_path_parses_native_integers_and_rejects_non_integers() -> None:
    frame = build_auction_path_rows(
        [{"Hour": 152500, "Prpr": 70000.0, "ShnuCnqn": 0, "SelnCnqn": 0}],
        snapshot_date=DATE,
        symbol="005930",
        available_at=AT,
    )
    assert len(frame) == 1
    assert frame["kind"].iloc[0] == AuctionPathRowKind.INDICATIVE.value
    with pytest.raises(ValueError, match="ShnuCnqn"):
        build_auction_path_rows([_row("152500", buy=True)], snapshot_date=DATE, symbol="005930", available_at=AT)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Prpr"):
        build_auction_path_rows(
            [{"Hour": "152500", "Prpr": 1.5, "ShnuCnqn": "0", "SelnCnqn": "0"}],
            snapshot_date=DATE,
            symbol="005930",
            available_at=AT,
        )


def test_auction_path_rejects_naive_or_early_available_at() -> None:
    from datetime import datetime as _datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        build_auction_path_rows([_row("152500")], snapshot_date=DATE, symbol="005930", available_at=_datetime(2026, 10, 15, 16, 0))
    with pytest.raises(ValueError, match="precedes the close"):
        build_auction_path_rows(
            [_row("152500")],
            snapshot_date=DATE,
            symbol="005930",
            available_at=_datetime(2026, 10, 15, 15, 0, tzinfo=SEOUL),
        )


def test_auction_path_rejects_malformed_hour_and_negative_quantity() -> None:
    with pytest.raises(ValueError, match="Hour"):
        build_auction_path_rows([_row("1525")], snapshot_date=DATE, symbol="005930", available_at=AT)
    with pytest.raises(ValueError, match="negative quantity"):
        build_auction_path_rows([_row("152500", buy="-5", sell="5")], snapshot_date=DATE, symbol="005930", available_at=AT)
