"""order_contract pure-domain invariant guards."""

from __future__ import annotations

import pytest

from src.daily import paper_trade
from src.execution.order_contract import (
    EntryPick,
    OrderIntent,
    OrderPhase,
    OrderSide,
    OrderType,
    SubmitVerdict,
    classify_order_response,
    entry_intent_id,
    entry_limit_price,
    exit_intent_id,
    normalize_order_no,
    size_entry_orders,
)
from src.execution.paper_broker import side_fee_krw


def _intent(**overrides) -> OrderIntent:
    base: dict = {
        "intent_id": "2026-10-12:005930:entry",
        "trading_date": "2026-10-12",
        "phase": OrderPhase.ENTRY,
        "symbol": "005930",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "qty": 1,
        "limit_price": 1000,
        "decision_price": 1000,
        "parent_intent_id": None,
    }
    base.update(overrides)
    return OrderIntent(**base)


def test_intent_ids_match_paper_ids() -> None:
    entry = entry_intent_id("2026-10-12", "005930")
    assert entry == paper_trade._entry_order_id("2026-10-12", "005930")
    assert exit_intent_id(entry, "2026-10-13") == f"{entry}:exit:2026-10-13"


@pytest.mark.parametrize("kwargs", [{"limit_price": None}, {"limit_price": 0}])
def test_limit_intent_requires_positive_price(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="LIMIT intent"):
        _intent(**kwargs)


def test_market_intent_forbids_price() -> None:
    with pytest.raises(ValueError, match="MARKET intent"):
        _intent(order_type=OrderType.MARKET, limit_price=1000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase": OrderPhase.ENTRY, "side": OrderSide.SELL},
        {"phase": OrderPhase.ENTRY, "parent_intent_id": "x"},
        {"phase": OrderPhase.EXIT, "side": OrderSide.BUY, "parent_intent_id": "p"},
        {"phase": OrderPhase.EXIT, "side": OrderSide.SELL, "parent_intent_id": None},
    ],
)
def test_entry_exit_side_parent_rules(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="intent requires"):
        _intent(**kwargs)


@pytest.mark.parametrize("qty", [0, -1])
def test_zero_or_negative_quantity_rejected(qty: int) -> None:
    with pytest.raises(ValueError, match="qty must be positive"):
        _intent(qty=qty)


def test_throttle_recognized_on_http_500() -> None:
    assert classify_order_response(500, {"rt_cd": "1", "msg_cd": "EGW00201"}) is SubmitVerdict.THROTTLED


def test_business_rejection_on_http_200_is_definite() -> None:
    assert classify_order_response(200, {"rt_cd": "1", "msg_cd": "40100000"}) is SubmitVerdict.REJECTED


def test_rt_cd_2_is_definite_rejection() -> None:
    assert classify_order_response(200, {"rt_cd": "2", "msg_cd": "OPSQ2000"}) is SubmitVerdict.REJECTED


def test_acceptance_requires_order_number() -> None:
    assert (
        classify_order_response(200, {"rt_cd": "0", "output": {"ODNO": "0000012345"}})
        is SubmitVerdict.ACCEPTED
    )
    assert classify_order_response(200, {"rt_cd": "0"}) is SubmitVerdict.AMBIGUOUS
    assert classify_order_response(200, {"rt_cd": "0", "output": None}) is SubmitVerdict.AMBIGUOUS
    assert classify_order_response(200, {"rt_cd": "0", "output": {"ODNO": "  "}}) is SubmitVerdict.AMBIGUOUS


@pytest.mark.parametrize(
    "args",
    [
        (500, {"rt_cd": "1", "msg_cd": "EGW99999"}),
        (502, "<html>"),
        (200, {}),
    ],
)
def test_unlisted_gateway_failure_is_ambiguous(args: tuple) -> None:
    assert classify_order_response(*args) is SubmitVerdict.AMBIGUOUS


def test_wrong_environment_tr_is_definite_rejection() -> None:
    assert classify_order_response(500, {"rt_cd": "1", "msg_cd": "EGW02006"}) is SubmitVerdict.REJECTED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0000012345", "12345"), (" 12345 ", "12345"), (12345, "12345"), ("", ""), (None, "")],
)
def test_order_number_normalization(raw: object, expected: str) -> None:
    assert normalize_order_no(raw) == expected


def test_limit_price_is_tick_aligned_and_bounded() -> None:
    assert entry_limit_price(263_000, 300.0, 349_000) == 270_500


def test_limit_price_capped_at_upper_limit() -> None:
    assert entry_limit_price(340_000, 300.0, 349_000) == 349_000


def test_band_crossing_keeps_lower_bound() -> None:
    result = entry_limit_price(199_900, 300.0, 259_500)
    assert result >= 199_900
    assert result % 500 == 0


def test_zero_buffer_returns_decision_price() -> None:
    assert entry_limit_price(270_500, 0.0, 349_000) == 270_500


def test_upper_limit_below_decision_is_corrupt() -> None:
    with pytest.raises(ValueError, match="upper_limit_price"):
        entry_limit_price(263_000, 300.0, 262_000)


def _picks() -> list[EntryPick]:
    return [
        EntryPick(symbol="000660", allocation=1 / 3, pred=0.1, decision_price=100_000, limit_price=103_000),
        EntryPick(symbol="005930", allocation=1 / 3, pred=0.9, decision_price=70_000, limit_price=72_100),
        EntryPick(symbol="035420", allocation=1 / 3, pred=0.5, decision_price=200_000, limit_price=206_000),
    ]


def test_sizing_never_exceeds_capital_including_fees() -> None:
    picks = _picks()
    capital = 9_950_000
    out = size_entry_orders(picks, capital, 5_000_000)
    total = sum(q * p.limit_price + side_fee_krw(q * p.limit_price) for p in picks for q in [out[p.symbol]])
    assert total <= capital
    for p in picks:
        assert out[p.symbol] * p.limit_price <= capital // 3


def test_sizing_honours_per_order_cap() -> None:
    pick = EntryPick(symbol="005930", allocation=1.0, pred=0.5, decision_price=70_000, limit_price=72_100)
    out = size_entry_orders([pick], 10_000_000, 5_000_000)
    assert out["005930"] * 72_100 <= 5_000_000


def test_unaffordable_pick_sizes_to_zero() -> None:
    pick = EntryPick(symbol="005930", allocation=0.01, pred=0.5, decision_price=70_000, limit_price=72_100)
    out = size_entry_orders([pick], 100_000, 5_000_000)
    assert out["005930"] == 0
    assert "005930" in out


def test_sizing_is_deterministic_and_rank_ordered() -> None:
    picks = _picks()
    first = size_entry_orders(picks, 1_000_000, 5_000_000)
    second = size_entry_orders(list(reversed(picks)), 1_000_000, 5_000_000)
    assert first == second
    binding = [
        EntryPick(symbol="005930", allocation=1.0, pred=0.9, decision_price=70_000, limit_price=100_000),
        EntryPick(symbol="000660", allocation=1.0, pred=0.1, decision_price=70_000, limit_price=100_000),
    ]
    out = size_entry_orders(binding, 150_000, 5_000_000)
    assert out["005930"] == 1
    assert out["000660"] == 0


def test_zero_capital_sizes_everything_to_zero() -> None:
    out = size_entry_orders(_picks(), 0, 5_000_000)
    assert set(out) == {"000660", "005930", "035420"}
    assert all(q == 0 for q in out.values())


@pytest.mark.parametrize("symbol", ["12345", "ABCDEF", "005930 "])
def test_non_six_digit_symbol_rejected(symbol: str) -> None:
    with pytest.raises(ValueError, match="symbol must be 6 digits"):
        _intent(symbol=symbol)


@pytest.mark.parametrize("date", ["not-a-date", "2026-13-40"])
def test_unparseable_trading_date_rejected(date: str) -> None:
    with pytest.raises(ValueError, match="trading_date must be YYYY-MM-DD"):
        _intent(trading_date=date)


def test_non_padded_trading_date_rejected() -> None:
    with pytest.raises(ValueError, match="trading_date must be YYYY-MM-DD"):
        _intent(trading_date="2026-1-1")


def test_entry_without_decision_price_rejected() -> None:
    with pytest.raises(ValueError, match="ENTRY intent requires decision_price"):
        _intent(order_type=OrderType.MARKET, limit_price=None, decision_price=None)


def test_auth_rejected_verdict() -> None:
    assert classify_order_response(500, {"rt_cd": "1", "msg_cd": "EGW00121"}) is SubmitVerdict.AUTH_REJECTED
    assert classify_order_response(500, {"rt_cd": "1", "msg_cd": "EGW00123"}) is SubmitVerdict.AUTH_REJECTED


def test_classify_order_auth_codes_from_registry() -> None:
    from src.api.auth_rejection import AUTH_REJECTION_CODES

    for code in AUTH_REJECTION_CODES["kis"]:
        assert classify_order_response(200, {"rt_cd": "1", "msg_cd": code}) is SubmitVerdict.AUTH_REJECTED
    assert classify_order_response(200, {"rt_cd": "1", "msg_cd": "EGW00201"}) is SubmitVerdict.THROTTLED


def test_classifier_never_raises() -> None:
    class _Evil(dict):  # type: ignore[type-arg]
        def get(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    assert classify_order_response(200, _Evil()) is SubmitVerdict.AMBIGUOUS


@pytest.mark.parametrize(
    ("decision", "buffer", "upper"),
    [(0, 300.0, 349_000), (263_000, -1.0, 349_000), (263_000, float("nan"), 349_000), (263_000, 300.0, 0)],
)
def test_limit_price_invalid_inputs_rejected(decision: int, buffer: float, upper: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        entry_limit_price(decision, buffer, upper)


def test_sizing_invalid_inputs_rejected() -> None:
    good = EntryPick(symbol="005930", allocation=1.0, pred=0.5, decision_price=70_000, limit_price=72_100)
    with pytest.raises(ValueError, match="capital_krw"):
        size_entry_orders([good], -1, 5_000_000)
    with pytest.raises(ValueError, match="duplicate symbol"):
        size_entry_orders([good, good], 10_000_000, 5_000_000)
    bad_alloc = EntryPick(symbol="005930", allocation=0.0, pred=0.5, decision_price=70_000, limit_price=72_100)
    with pytest.raises(ValueError, match="allocation must be positive"):
        size_entry_orders([bad_alloc], 10_000_000, 5_000_000)
    bad_limit = EntryPick(symbol="005930", allocation=1.0, pred=0.5, decision_price=70_000, limit_price=0)
    with pytest.raises(ValueError, match="limit_price must be positive"):
        size_entry_orders([bad_limit], 10_000_000, 5_000_000)


def test_nan_pred_ranks_last() -> None:
    picks = [
        EntryPick(symbol="000660", allocation=1.0, pred=float("nan"), decision_price=70_000, limit_price=100_000),
        EntryPick(symbol="005930", allocation=1.0, pred=-5.0, decision_price=70_000, limit_price=100_000),
    ]
    out = size_entry_orders(picks, 150_000, 5_000_000)
    assert out["005930"] == 1
    assert out["000660"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbol": "005930\n"},
        {"qty": float("nan")},
        {"qty": 1.5},
        {"qty": True},
        {"order_type": "unknown"},
        {"phase": "unknown"},
        {"limit_price": float("nan")},
        {"decision_price": float("nan")},
        {"phase": OrderPhase.EXIT, "side": OrderSide.SELL, "parent_intent_id": " "},
    ],
)
def test_malformed_intents_fail_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"digits|positive|enums|non-empty|price > 0"):
        _intent(**kwargs)


def test_tick_rounding_preserves_exact_boundary() -> None:
    assert entry_limit_price(100_000, 10.0, 130_000) == 100_100
    with pytest.raises(ValueError, match="tick grid"):
        entry_limit_price(199_999, 0.0, 259_500)
    with pytest.raises(ValueError, match="finite"):
        entry_limit_price(100_000, 1e308, 130_000)


@pytest.mark.parametrize("cap", [0, -1])
def test_nonpositive_order_cap_rejected(cap: int) -> None:
    with pytest.raises(ValueError, match="max_notional_per_order_krw"):
        size_entry_orders(_picks(), 100_000, cap)


def test_sizing_is_maximal_including_fees_and_uses_limit() -> None:
    picks = [EntryPick("005930", 1.0, 0.0, 50_000, 100_000)]
    capital = 1_000_000
    qty = size_entry_orders(picks, capital, capital)["005930"]
    assert qty == 9
    assert qty * 100_000 + side_fee_krw(qty * 100_000) <= capital
    assert (qty + 1) * 100_000 + side_fee_krw((qty + 1) * 100_000) > capital


def test_infinite_predictions_keep_numeric_ranking() -> None:
    picks = [
        EntryPick("000660", 1.0, 1.0, 100_000, 100_000),
        EntryPick("005930", 1.0, float("inf"), 100_000, 100_000),
    ]
    assert size_entry_orders(picks, 150_000, 150_000) == {"005930": 1, "000660": 0}
