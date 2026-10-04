"""Invariant guards for the decision-time (15:20) panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.eod_superset import EodSupersetScreen
from src.data.pit1520_panel import (
    PIT1520_PANEL_COLUMNS,
    Pit1520PanelConfig,
    Pit1520PanelResult,
    aggregate_decision_bars,
    build_pit1520_panel,
    default_panel_paths,
    live_input_to_panel_rows,
    load_live_decision_input,
    load_regular_bars,
    panel_to_decision_input,
    write_pit1520_panel,
)
from src.data.pit1520_panel import (
    PanelSourcePolicy as Policy,
)


def _screen() -> EodSupersetScreen:
    return EodSupersetScreen(
        min_change_ratio=0.01,
        max_change_ratio=0.12,
        min_trade_value_100m=100.0,
        min_market_cap_100m=495.0,
        common_stock_only=False,
    )


def _bar(symbol, ts, o, h, l, c, vol, vendor, trade=True):
    return {
        "symbol": symbol,
        "ts_hms": ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol,
        "value_krw": int(c * vol),
        "has_trade": trade,
        "vendor": vendor,
    }


def _ph_row(date, symbol, *, open, close, prev_close, close_raw=None, mc=900.0, inst=0.0, foreign=0.0,
             kospi=0.5, kosdaq=0.4, v_kospi=18.0, v_kosdaq=20.0, volume=1000.0, tv=500.0,
             chg=None, market="KOSPI"):
    raw = close if close_raw is None else close_raw
    return {
        "date": date,
        "symbol": symbol,
        "market": market,
        "open": open,
        "high": close,
        "low": close,
        "close": close,
        "close_raw": raw,
        "prev_close": prev_close,
        "volume": volume,
        "market_cap_100m": mc,
        "trade_value_100m": tv,
        "mc_clean": mc,
        "inst_netbuy": inst,
        "foreign_netbuy": foreign,
        "kospi_pct": kospi / 100.0,
        "kosdaq_pct": kosdaq / 100.0,
        "v_kospi": v_kospi,
        "v_kosdaq": v_kosdaq,
        "chg_ratio": close / prev_close - 1.0 if chg is None else chg,
        "tv_clean": tv,
    }


def _live_row(symbol, *, o, h, l, c, prev, vol=500000.0, tv=250.0, mc=8000.0, market="KOSPI", kospi=0.5):
    return {
        "종목코드": symbol,
        "시가": o,
        "고가": h,
        "저가": l,
        "종가": c,
        "전일종가": prev,
        "거래량": vol,
        "거래대금": tv,
        "시가총액": mc,
        "시장구분": market,
        "기관_순매수": 0.0,
        "외국인_순매수": 0.0,
        "kospi": kospi,
        "kosdaq": 0.4,
        "v_kospi": 18.0,
    }


def _panel_row(date, symbol, **overrides):
    row = {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "market": "KOSPI",
        "open": 10000.0,
        "high": 10100.0,
        "low": 9900.0,
        "close": 10050.0,
        "close_raw": 10050.0,
        "prev_close": 9800.0,
        "volume": 1000.0,
        "trade_value_100m": 500.0,
        "market_cap_100m": 900.0,
        "inst_netbuy": float("nan"),
        "foreign_netbuy": float("nan"),
        "inst_netbuy_prev": 1.0,
        "foreign_netbuy_prev": 2.0,
        "kospi_pct": 0.005,
        "kosdaq_pct": 0.004,
        "v_kospi": 18.0,
        "v_kosdaq": 20.0,
        "index_basis": "eod_fallback",
        "source": "bars",
        "bars_vendor": "kis",
        "n_bars": 10,
        "first_bar_hms": "90000",
        "last_bar_hms": "151900",
        "capture_run_id": "",
    }
    row.update(overrides)
    return row


def test_aggregate_decision_bars_kis_start_stamp_excludes_auction() -> None:
    bars = pd.DataFrame([
        _bar("005930", 90000, 70000, 70100, 69900, 70050, 100, "kis"),
        _bar("005930", 151900, 71000, 71100, 70900, 71050, 200, "kis"),
        _bar("005930", 153000, 72000, 72100, 71900, 72050, 500, "kis"),
    ])
    agg, exc = aggregate_decision_bars(bars, config=Pit1520PanelConfig())
    assert len(agg) == 1 and len(exc) == 0
    assert float(agg.iloc[0]["close"]) == 71050.0
    assert float(agg.iloc[0]["volume"]) == 300.0
    assert str(agg.iloc[0]["last_bar_hms"]) == "151900"


def test_aggregate_decision_bars_ls_end_stamp_keeps_1520_bar() -> None:
    bars = pd.DataFrame([
        _bar("000660", 90100, 100000, 100100, 99900, 100050, 50, "ls"),
        _bar("000660", 152000, 101000, 101100, 100900, 101050, 60, "ls"),
        _bar("000660", 153000, 102000, 102100, 101900, 102050, 70, "ls"),
        _bar("000660", 160100, 103000, 103100, 102900, 103050, 80, "ls"),
    ])
    agg, exc = aggregate_decision_bars(bars, config=Pit1520PanelConfig())
    assert len(agg) == 1 and len(exc) == 0
    assert float(agg.iloc[0]["close"]) == 101050.0
    assert float(agg.iloc[0]["volume"]) == 110.0


def test_aggregate_decision_bars_ignores_non_trade_rows() -> None:
    bars = pd.DataFrame([
        _bar("000660", 90100, 100000, 100100, 99900, 100050, 50, "ls"),
        _bar("000660", 91000, 100000, 109999, 99900, 100050, 0, "ls", trade=False),
        _bar("000660", 152000, 101000, 101100, 100900, 101050, 60, "ls"),
    ])
    agg, _exc = aggregate_decision_bars(bars, config=Pit1520PanelConfig())
    assert float(agg.iloc[0]["high"]) == 101100.0
    assert float(agg.iloc[0]["volume"]) == 110.0


def test_aggregate_decision_bars_unknown_vendor_fails_closed() -> None:
    bars = pd.DataFrame([_bar("005930", 90000, 70000, 70100, 69900, 70050, 100, "toss")])
    agg, exc = aggregate_decision_bars(bars, config=Pit1520PanelConfig())
    assert len(agg) == 0
    assert exc.iloc[0]["reason"] == "unknown_bar_stamp"


def test_aggregate_decision_bars_mixed_vendor_excluded() -> None:
    bars = pd.DataFrame([
        _bar("005930", 90000, 70000, 70100, 69900, 70050, 100, "kis"),
        _bar("005930", 90100, 70000, 70100, 69900, 70050, 100, "ls"),
    ])
    agg, exc = aggregate_decision_bars(bars, config=Pit1520PanelConfig())
    assert len(agg) == 0
    assert exc.iloc[0]["reason"] == "mixed_vendor"


def test_aggregate_decision_bars_requires_pre_cutoff_trade() -> None:
    bars = pd.DataFrame([_bar("005930", 153000, 72000, 72100, 71900, 72050, 500, "kis")])
    agg, exc = aggregate_decision_bars(bars, config=Pit1520PanelConfig())
    assert len(agg) == 0
    assert exc.iloc[0]["reason"] == "no_bars_before_cutoff"


def _two_day_history(symbol="005930", *, t_open=10000.0, t_close=10020.0, t_raw=None, t_prev=9800.0,
                     mc_prev=1000.0, p_date="2026-09-17", t_date="2026-09-18"):
    return pd.DataFrame([
        _ph_row(p_date, symbol, open=9900.0, close=9900.0, prev_close=9700.0, mc=mc_prev,
                inst=5.0, foreign=6.0),
        _ph_row(t_date, symbol, open=t_open, close=t_close, prev_close=t_prev, close_raw=t_raw),
    ])


def _kis_session(symbol, first_open, last_close, *, first_ts=90000, last_ts=151900):
    return pd.DataFrame([
        _bar(symbol, first_ts, first_open, first_open + 50, first_open - 50, first_open + 20, 100, "kis"),
        _bar(symbol, last_ts, last_close - 20, last_close + 30, last_close - 30, last_close, 150, "kis"),
    ])


def test_build_panel_excludes_head_truncated_symbol() -> None:
    ph = _two_day_history(t_open=10000.0, t_close=10020.0, t_prev=9800.0)
    bars = _kis_session("005930", 10030.0, 10040.0, first_ts=125100)
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    assert len(result.panel) == 0
    assert result.exclusions.iloc[0]["reason"] == "head_truncated"


def test_build_panel_excludes_late_start_partition_even_when_open_matches() -> None:
    ph = _two_day_history(t_open=10000.0, t_close=10020.0, t_prev=9800.0)
    bars = pd.DataFrame([
        _bar("005930", 125100, 10000, 10050, 9990, 10010, 100, "ls"),
        _bar("005930", 152000, 10000, 10030, 9990, 10020, 150, "ls"),
    ])

    result = build_pit1520_panel(
        price_history=ph, dates=["2026-09-18"], bars_loader=lambda _d: bars,
        live_loader=lambda _d: None, screen=_screen(),
    )

    assert len(result.panel) == 0
    assert result.exclusions.iloc[0]["reason"] == "head_truncated"


def test_build_panel_keeps_gap_day_with_folded_preopen_print_and_uses_official_open() -> None:
    ph = _two_day_history(t_open=8000.0, t_close=8300.0, t_prev=10000.0)
    bars = pd.DataFrame([
        _bar("005930", 90100, 10000, 10000, 10000, 10000, 5, "ls"),
        _bar("005930", 90300, 8000, 8100, 7950, 8050, 300, "ls"),
        _bar("005930", 152000, 8200, 8320, 8180, 8300, 150, "ls"),
    ])

    result = build_pit1520_panel(
        price_history=ph, dates=["2026-09-18"], bars_loader=lambda _d: bars,
        live_loader=lambda _d: None, screen=_screen(),
    )

    assert len(result.panel) == 1
    assert float(result.panel.iloc[0]["open"]) == 8000.0


def test_build_panel_excludes_symbol_without_official_open() -> None:
    ph = _two_day_history(t_open=float("nan"), t_close=10020.0, t_prev=9800.0)
    bars = _kis_session("005930", 10030.0, 10040.0)

    result = build_pit1520_panel(
        price_history=ph, dates=["2026-09-18"], bars_loader=lambda _d: bars,
        live_loader=lambda _d: None, screen=_screen(),
    )

    assert len(result.panel) == 0
    assert result.exclusions.iloc[0]["reason"] == "head_truncated"


def test_aggregate_rebuilds_quantized_ls_trade_value_from_volume_and_typical_price() -> None:
    from src.data.pit1520_panel import Pit1520PanelConfig, aggregate_decision_bars

    bars = pd.DataFrame([
        {**_bar("005930", 90100, 10000, 10100, 9900, 10000, 1000, "ls"), "value_krw": 9_000_000},
        {**_bar("005930", 152000, 10000, 10300, 10000, 10200, 500, "ls"), "value_krw": 5_000_000},
    ])

    agg, _excl = aggregate_decision_bars(bars, config=Pit1520PanelConfig())

    expected = (1000 * (10100 + 9900 + 10000) / 3 + 500 * (10300 + 10000 + 10200) / 3) / 1e8
    assert float(agg.iloc[0]["trade_value_100m"]) == pytest.approx(expected)


def test_aggregate_keeps_exact_kis_trade_value() -> None:
    from src.data.pit1520_panel import Pit1520PanelConfig, aggregate_decision_bars

    bars = pd.DataFrame([
        {**_bar("005930", 90000, 10000, 10100, 9900, 10000, 1000, "kis"), "value_krw": 10_123_456},
    ])

    agg, _excl = aggregate_decision_bars(bars, config=Pit1520PanelConfig())

    assert float(agg.iloc[0]["trade_value_100m"]) == pytest.approx(10_123_456 / 1e8)


def _dense_kis_day(n_symbols: int, *, drop_minute: int | None = None) -> pd.DataFrame:
    rows = []
    for i in range(n_symbols):
        symbol = f"{i + 1:06d}"
        minute = 90000
        while minute < 152000:
            if minute != drop_minute:
                rows.append(_bar(symbol, minute, 10000, 10010, 9990, 10000, 10, "kis"))
            hh, mm = divmod(minute // 100, 100)
            mm += 1
            if mm == 60:
                hh, mm = hh + 1, 0
            minute = hh * 10000 + mm * 100
    return pd.DataFrame(rows)


def test_vendor_minute_gap_excludes_whole_bar_day() -> None:
    from src.data.pit1520_panel import Pit1520PanelConfig, aggregate_decision_bars, vendor_minute_gaps

    gapped = _dense_kis_day(30, drop_minute=124900)

    assert vendor_minute_gaps(gapped, config=Pit1520PanelConfig()) == [124900]
    agg, excl = aggregate_decision_bars(gapped, config=Pit1520PanelConfig())
    assert len(agg) == 0
    assert set(excl["reason"]) == {"vendor_minute_gap"}


def test_vendor_minute_gap_ignores_complete_day_and_small_samples() -> None:
    from src.data.pit1520_panel import Pit1520PanelConfig, aggregate_decision_bars, vendor_minute_gaps

    assert vendor_minute_gaps(_dense_kis_day(30), config=Pit1520PanelConfig()) == []
    assert vendor_minute_gaps(_dense_kis_day(5, drop_minute=124900), config=Pit1520PanelConfig()) == []
    agg, _excl = aggregate_decision_bars(_dense_kis_day(30), config=Pit1520PanelConfig())
    assert len(agg) == 30


def test_vendor_minute_gap_skips_empty_and_head_truncated_days() -> None:
    from src.data.pit1520_panel import Pit1520PanelConfig, vendor_minute_gaps

    empty = pd.DataFrame(columns=["symbol", "ts_hms", "open", "high", "low", "close", "volume",
                                  "value_krw", "has_trade", "vendor"])
    late_only = _dense_kis_day(30)
    late_only = late_only[pd.to_numeric(late_only["ts_hms"]) >= 125100]

    assert vendor_minute_gaps(empty, config=Pit1520PanelConfig()) == []
    assert vendor_minute_gaps(late_only, config=Pit1520PanelConfig()) == []


def test_build_panel_excludes_symbol_whose_head_range_misses_official_open() -> None:
    ph = _two_day_history(t_open=10000.0, t_close=10350.0, t_prev=9800.0)
    bars = pd.DataFrame([
        _bar("005930", 90000, 10300, 10320, 10290, 10310, 100, "kis"),
        _bar("005930", 151900, 10330, 10360, 10320, 10350, 150, "kis"),
    ])

    result = build_pit1520_panel(
        price_history=ph, dates=["2026-09-18"], bars_loader=lambda _d: bars,
        live_loader=lambda _d: None, screen=_screen(),
    )

    assert len(result.panel) == 0
    assert result.exclusions.iloc[0]["reason"] == "head_truncated"
    assert "outside head range" in result.exclusions.iloc[0]["detail"]


def test_build_panel_rescales_prev_close_to_raw_basis() -> None:
    ph = _two_day_history(t_open=5050.0, t_close=5000.0, t_raw=10000.0, t_prev=4800.0)
    bars = pd.DataFrame([
        _bar("005930", 90000, 10100, 10120, 10090, 10100, 100, "kis"),
        _bar("005930", 151900, 10100, 10120, 10090, 10100, 100, "kis"),
    ])
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    assert len(result.panel) == 1
    row = result.panel.iloc[0]
    assert float(row["prev_close"]) == 9600.0
    assert float(row["close"]) == 10100.0
    assert float(row["close"]) / float(row["prev_close"]) - 1.0 == pytest.approx(10100.0 / 9600.0 - 1.0)


def test_build_panel_market_cap_uses_prior_day() -> None:
    ph = _two_day_history(t_open=10080.0, t_close=10080.0, t_prev=9600.0, mc_prev=1000.0)
    bars = pd.DataFrame([
        _bar("005930", 90000, 10080, 10100, 10060, 10080, 100, "kis"),
        _bar("005930", 151900, 10080, 10100, 10060, 10080, 150, "kis"),
    ])
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    assert float(result.panel.iloc[0]["market_cap_100m"]) == pytest.approx(1050.0, rel=1e-12)


def _three_symbol_panel_fixture():
    symbols = ["005930", "000660", "005380"]
    closes = [10050.0, 50050.0, 20050.0]
    ph_rows = []
    bar_frames = []
    for symbol, close in zip(symbols, closes, strict=True):
        ph_rows.append(_ph_row("2026-09-17", symbol, open=close - 100, close=close - 100, prev_close=close - 300,
                               mc=1000.0, inst=5.0, foreign=6.0))
        ph_rows.append(_ph_row("2026-09-18", symbol, open=close, close=close, prev_close=close - 250))
        ph_rows.append(_ph_row("2026-09-21", symbol, open=close, close=close, prev_close=close - 200))
        bar_frames.append(_kis_session(symbol, close, close))
    return pd.DataFrame(ph_rows), pd.concat(bar_frames, ignore_index=True)


def test_build_panel_is_invariant_to_eod_outcome_perturbation() -> None:
    ph, bars = _three_symbol_panel_fixture()
    kwargs = {
        "dates": ["2026-09-18"],
        "bars_loader": lambda _d: bars,
        "live_loader": lambda _d: None,
        "screen": _screen(),
    }
    before = build_pit1520_panel(price_history=ph, **kwargs)
    rng = np.random.default_rng(1520)
    mutated = ph.copy()
    t_mask = pd.to_datetime(mutated["date"]).dt.strftime("%Y-%m-%d") == "2026-09-18"
    for col in ("high", "low", "volume", "trade_value_100m", "market_cap_100m", "inst_netbuy", "foreign_netbuy",
                "kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq"):
        mutated.loc[t_mask, col] = np.asarray(mutated.loc[t_mask, col], dtype=np.float64) * rng.uniform(0.5, 1.5, int(t_mask.sum()))
    factor = rng.uniform(0.9, 1.1, int(t_mask.sum()))
    mutated.loc[t_mask, "close"] = np.asarray(mutated.loc[t_mask, "close"], dtype=np.float64) * factor
    mutated.loc[t_mask, "close_raw"] = np.asarray(mutated.loc[t_mask, "close_raw"], dtype=np.float64) * factor
    future = pd.DataFrame([
        _ph_row("2026-09-22", s, open=rng.uniform(1, 99999), close=rng.uniform(1, 99999),
                prev_close=rng.uniform(1, 99999))
        for s in ("005930", "000660", "005380")
    ])
    mutated = pd.concat([mutated, future], ignore_index=True)
    after = build_pit1520_panel(price_history=mutated, **kwargs)
    index_cols = {"kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq"}
    left = before.panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    right = after.panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert list(left.columns) == list(right.columns)
    assert (left["symbol"] == right["symbol"]).all()
    for col in left.columns:
        if col in index_cols:
            continue
        lv, rv = left[col].to_numpy(), right[col].to_numpy()
        if left[col].dtype.kind == "f":
            assert np.allclose(lv.astype(np.float64), rv.astype(np.float64), rtol=1e-12, atol=0, equal_nan=True), col
        else:
            assert (lv.astype(str) == rv.astype(str)).all(), col
    assert all(right["index_basis"] == "eod_fallback")


def test_build_panel_ignores_post_cutoff_bars() -> None:
    ph, bars = _three_symbol_panel_fixture()
    rng = np.random.default_rng(7)
    extra = pd.DataFrame([_bar(s, 153000, rng.uniform(1, 99999), rng.uniform(100000, 199999),
                               rng.uniform(1, 99999), rng.uniform(1, 99999),
                               int(rng.integers(1, 99999)), "kis")
                          for s in ("005930", "000660", "005380")])
    plain = build_pit1520_panel(price_history=ph, dates=["2026-09-18"], bars_loader=lambda _d: bars,
                                live_loader=lambda _d: None, screen=_screen())
    noisy = build_pit1520_panel(price_history=ph, dates=["2026-09-18"],
                                bars_loader=lambda _d: pd.concat([bars, extra], ignore_index=True),
                                live_loader=lambda _d: None, screen=_screen())
    pd.testing.assert_frame_equal(plain.panel, noisy.panel)


def test_build_panel_prefers_live_input_for_whole_day() -> None:
    ph = pd.DataFrame([
        _ph_row("2026-09-17", "005930", open=70000, close=70000, prev_close=69000, mc=8000.0, inst=5.0, foreign=6.0),
        _ph_row("2026-09-18", "005930", open=71000, close=71000, prev_close=70000),
        _ph_row("2026-09-17", "000660", open=100000, close=100000, prev_close=99000, mc=9000.0, inst=7.0, foreign=8.0),
        _ph_row("2026-09-18", "000660", open=101000, close=101000, prev_close=100000),
        _ph_row("2026-09-18", "005380", open=20000, close=20000, prev_close=19800),
    ])
    live = pd.DataFrame([
        _live_row("005930", o=71000, h=71100, l=70900, c=71050, prev=70000),
        _live_row("000660", o=101000, h=101100, l=100900, c=101050, prev=100000),
    ])

    def _no_bars(_d):
        raise AssertionError("bars_loader must not be called under day-level live precedence")

    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=_no_bars,
        live_loader=lambda _d: (live, "run-1"),
        screen=_screen(),
    )
    assert set(result.panel["symbol"]) == {"005930", "000660"}
    assert set(result.panel["source"]) == {"live_decision"}
    assert set(result.panel["index_basis"]) == {"live_1520"}


def test_build_panel_bars_only_policy_skips_live() -> None:
    ph, bars = _three_symbol_panel_fixture()

    def _no_live(_d):
        raise AssertionError("live_loader must not be called under BARS_ONLY")

    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=_no_live,
        screen=_screen(),
        source_policy=Policy.BARS_ONLY,
    )
    assert len(result.panel) == 3
    assert set(result.panel["source"]) == {"bars"}


def test_live_input_to_panel_rows_maps_units() -> None:
    live = pd.DataFrame([_live_row("005930", o=71000, h=71100, l=70900, c=71050, prev=70000, mc=14024.0, kospi=0.8)])
    rows = live_input_to_panel_rows(live, pd.Timestamp("2026-09-18"), run_id="run-9")
    assert float(rows.iloc[0]["kospi_pct"]) == pytest.approx(0.008)
    assert float(rows.iloc[0]["market_cap_100m"]) == 14024.0
    assert np.isnan(float(rows.iloc[0]["inst_netbuy"]))
    assert list(rows.columns) == list(PIT1520_PANEL_COLUMNS)


def test_panel_to_decision_input_round_trip() -> None:
    panel = pd.DataFrame([_panel_row("2026-09-18", s) for s in ("005930", "000660")])
    panel["v_kosdaq"] = float("nan")
    snapshot = panel_to_decision_input(panel)
    back = live_input_to_panel_rows(snapshot, pd.Timestamp("2026-09-18"), run_id="run-x")
    for col in ("open", "high", "low", "close", "close_raw", "prev_close", "volume", "trade_value_100m",
                "market_cap_100m", "kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq"):
        assert np.allclose(panel[col].to_numpy(dtype=np.float64), back[col].to_numpy(dtype=np.float64),
                           rtol=1e-12, atol=0, equal_nan=True), col


def test_panel_to_decision_input_rejects_multiple_dates() -> None:
    panel = pd.DataFrame([_panel_row("2026-09-18", "005930"), _panel_row("2026-09-21", "005930")])
    with pytest.raises(ValueError, match="exactly one date"):
        panel_to_decision_input(panel)


def test_build_panel_day_coverage_and_not_fetched_attribution() -> None:
    ph = pd.DataFrame([
        _ph_row("2026-09-17", "005930", open=9900, close=9900, prev_close=9700, mc=1000.0, inst=5.0, foreign=6.0),
        _ph_row("2026-09-18", "005930", open=10050, close=10050, prev_close=9800),
        _ph_row("2026-09-17", "000660", open=49900, close=49900, prev_close=48900, mc=1000.0, inst=5.0, foreign=6.0),
        _ph_row("2026-09-18", "000660", open=50050, close=50050, prev_close=49000, close_raw=50500.0),
        _ph_row("2026-09-17", "005380", open=19900, close=19900, prev_close=19700, mc=1000.0, inst=5.0, foreign=6.0),
        _ph_row("2026-09-18", "005380", open=20050, close=20050, prev_close=19800),
    ])
    bars = _kis_session("005930", 10050.0, 10050.0)
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    day = result.days.iloc[0]
    assert int(day["n_superset"]) == 3
    assert int(day["n_superset_present"]) == 1
    assert float(day["superset_coverage"]) == pytest.approx(1 / 3)
    by_symbol = {r["symbol"]: r["reason"] for _, r in result.exclusions.iterrows()}
    assert by_symbol["000660"] == "price_basis_adjusted"
    assert by_symbol["005380"] == "not_fetched"


def test_build_panel_never_imputes_missing_prior_values() -> None:
    ph = pd.DataFrame([
        _ph_row("2026-09-17", "005930", open=9900, close=9900, prev_close=9700, mc=float("nan")),
        _ph_row("2026-09-18", "005930", open=10050, close=10050, prev_close=9800),
    ])
    bars = _kis_session("005930", 10050.0, 10050.0)
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    assert len(result.panel) == 0
    assert result.exclusions.iloc[0]["reason"] == "prev_market_cap_unavailable"


def test_load_regular_bars_missing_vs_unreadable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    from src.data.intraday_store import intraday_partition_path

    target = intraday_partition_path(1, "2026-09-18", "regular")
    empty = load_regular_bars("2026-09-18")
    assert len(empty) == 0
    assert list(empty.columns) == ["symbol", "ts_hms", "open", "high", "low", "close", "volume",
                                   "value_krw", "has_trade", "vendor"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a parquet file")
    with pytest.raises(OSError, match="Cannot read intraday partition evidence"):
        load_regular_bars("2026-09-18")


def test_load_regular_bars_reads_partition_with_pushdown(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    from src.data.intraday_store import intraday_partition_path

    target = intraday_partition_path(1, "2026-09-18", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([
        _bar("005930", 90000, 70000, 70100, 69900, 70050, 100, "kis"),
        _bar("005930", 151900, 71000, 71100, 70900, 71050, 200, "kis"),
        _bar("005930", 153000, 72000, 72100, 71900, 72050, 500, "kis"),
    ])
    frame.to_parquet(target, index=False)
    loaded = load_regular_bars("2026-09-18")
    assert set(loaded["ts_hms"].tolist()) == {90000, 151900}


class _FakeStore:
    def __init__(self, behavior):
        self.behavior = behavior
        self.seen_cutoff = None

    def read_decision(self, snapshot_date, *, available_by, run_id=None):
        self.seen_cutoff = available_by
        if self.behavior == "missing":
            raise FileNotFoundError("no capture")
        if self.behavior == "broken":
            raise ValueError("hash mismatch")
        frame = pd.DataFrame([_live_row("005930", o=1, h=1, l=1, c=1, prev=1)])
        frame["capture_run_id"] = "run-abc"
        return frame


def test_load_live_decision_input_fail_modes() -> None:
    assert load_live_decision_input(_FakeStore("missing"), "2026-09-18") is None
    with pytest.raises(ValueError, match="hash mismatch"):
        load_live_decision_input(_FakeStore("broken"), "2026-09-18")
    store = _FakeStore("ok")
    frame, run_id = load_live_decision_input(store, "2026-09-18")
    assert run_id == "run-abc"
    assert store.seen_cutoff.tzinfo is not None
    assert (store.seen_cutoff.hour, store.seen_cutoff.minute, store.seen_cutoff.second) == (15, 30, 0)


def test_write_pit1520_panel_replaces_only_requested_dates(tmp_path) -> None:
    paths = (
        tmp_path / "pit1520_panel.parquet",
        tmp_path / "pit1520_panel_exclusions.parquet",
        tmp_path / "pit1520_panel_days.parquet",
    )
    old_panel = pd.DataFrame([
        _panel_row("2026-09-17", "005930"),
        _panel_row("2026-09-18", "005930", close=1.0),
    ])
    old_panel["date"] = pd.to_datetime(old_panel["date"])
    old_excl = pd.DataFrame([{"date": pd.Timestamp("2026-09-17"), "symbol": "000660",
                              "reason": "not_fetched", "detail": "x"}])
    old_days = pd.DataFrame([{"date": pd.Timestamp("2026-09-17"), "source": "bars", "n_rows": 1,
                              "n_excluded": 1, "n_superset": 2, "n_superset_present": 1,
                              "superset_coverage": 0.5, "index_basis": "eod_fallback"}])
    from src.data.io_utils import atomic_write_parquet

    atomic_write_parquet(old_panel, paths[0])
    atomic_write_parquet(old_excl, paths[1])
    atomic_write_parquet(old_days, paths[2])
    new_panel = pd.DataFrame([_panel_row("2026-09-18", "005930", close=777.0)])
    new_panel["date"] = pd.to_datetime(new_panel["date"])
    result = Pit1520PanelResult(panel=new_panel, exclusions=old_excl.iloc[0:0].copy(),
                                days=old_days.iloc[0:0].copy())
    write_pit1520_panel(result, paths=paths, replace_dates=["2026-09-18"])
    got = pd.read_parquet(paths[0]).sort_values(["date", "symbol"]).reset_index(drop=True)
    assert len(got) == 2
    kept = got[got["date"] == pd.Timestamp("2026-09-17")]
    pd.testing.assert_frame_equal(kept.reset_index(drop=True),
                                  old_panel[old_panel["date"] == pd.Timestamp("2026-09-17")].reset_index(drop=True),
                                  check_dtype=False)
    assert float(got[got["date"] == pd.Timestamp("2026-09-18")].iloc[0]["close"]) == 777.0


def test_build_panel_is_deterministic() -> None:
    ph, bars = _three_symbol_panel_fixture()
    kwargs = {
        "price_history": ph,
        "dates": ["2026-09-18"],
        "bars_loader": lambda _d: bars,
        "live_loader": lambda _d: None,
        "screen": _screen(),
    }
    first = build_pit1520_panel(**kwargs)
    second = build_pit1520_panel(**kwargs)
    pd.testing.assert_frame_equal(first.panel, second.panel)
    pd.testing.assert_frame_equal(first.exclusions, second.exclusions)
    pd.testing.assert_frame_equal(first.days, second.days)


def test_default_panel_paths_under_history_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    panel, exclusions, days = default_panel_paths()
    assert panel == Path(str(tmp_path)) / "pit1520_panel.parquet"
    assert exclusions == Path(str(tmp_path)) / "pit1520_panel_exclusions.parquet"
    assert days == Path(str(tmp_path)) / "pit1520_panel_days.parquet"


def test_aggregate_rejects_missing_columns_and_empty_input() -> None:
    with pytest.raises(ValueError, match="ts_hms"):
        aggregate_decision_bars(pd.DataFrame([{"symbol": "005930"}]), config=Pit1520PanelConfig())
    cols = ("symbol", "ts_hms", "open", "high", "low", "close", "volume", "value_krw", "has_trade", "vendor")
    agg, exc = aggregate_decision_bars(
        pd.DataFrame({c: pd.Series(dtype="object") for c in cols}), config=Pit1520PanelConfig()
    )
    assert len(agg) == 0 and len(exc) == 0


def test_aggregate_marks_invalid_price() -> None:
    bars = pd.DataFrame([_bar("005930", 90000, 0, 0, 0, 0, 100, "kis")])
    agg, exc = aggregate_decision_bars(bars, config=Pit1520PanelConfig())
    assert len(agg) == 0
    assert exc.iloc[0]["reason"] == "invalid_price"


def test_aggregate_accepts_int_and_string_trade_flags() -> None:
    int_bars = pd.DataFrame([
        {**_bar("005930", 90000, 70000, 70100, 69900, 70050, 100, "kis"), "has_trade": 1},
        {**_bar("005930", 151900, 71000, 71100, 70900, 71050, 200, "kis"), "has_trade": 0},
    ])
    agg, _exc = aggregate_decision_bars(int_bars, config=Pit1520PanelConfig())
    assert len(agg) == 1 and float(agg.iloc[0]["close"]) == 70050.0
    str_bars = pd.DataFrame([
        {**_bar("005930", 90000, 70000, 70100, 69900, 70050, 100, "kis"), "has_trade": "true"},
        {**_bar("005930", 151900, 71000, 71100, 70900, 71050, 200, "kis"), "has_trade": "false"},
    ])
    agg, _exc = aggregate_decision_bars(str_bars, config=Pit1520PanelConfig())
    assert len(agg) == 1 and float(agg.iloc[0]["close"]) == 70050.0


def test_live_input_rejects_missing_columns_and_duplicates() -> None:
    live = pd.DataFrame([_live_row("005930", o=1, h=1, l=1, c=1, prev=1)])
    with pytest.raises(ValueError, match="kospi"):
        live_input_to_panel_rows(live.drop(columns=["kospi"]), pd.Timestamp("2026-09-18"), run_id="r")
    dup = pd.DataFrame([
        _live_row("005930", o=1, h=1, l=1, c=1, prev=1),
        _live_row("005930", o=2, h=2, l=2, c=2, prev=1),
    ])
    with pytest.raises(ValueError, match="duplicate"):
        live_input_to_panel_rows(dup, pd.Timestamp("2026-09-18"), run_id="r")


def test_panel_to_decision_input_rejects_missing_columns() -> None:
    panel = pd.DataFrame([_panel_row("2026-09-18", "005930")])
    with pytest.raises(ValueError, match="kospi_pct"):
        panel_to_decision_input(panel.drop(columns=["kospi_pct"]))


def test_build_panel_validates_price_history_and_dates() -> None:
    ph, bars = _three_symbol_panel_fixture()
    base_kwargs = {
        "dates": ["2026-09-18"],
        "bars_loader": lambda _d: bars,
        "live_loader": lambda _d: None,
        "screen": _screen(),
    }
    with pytest.raises(ValueError, match="mc_clean"):
        build_pit1520_panel(price_history=ph.drop(columns=["mc_clean"]), **base_kwargs)
    bad_date = ph.copy()
    bad_date.loc[bad_date.index[0], "date"] = "not-a-date"
    with pytest.raises(ValueError, match="unparseable"):
        build_pit1520_panel(price_history=bad_date, **base_kwargs)
    duped = pd.concat([ph, ph.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_pit1520_panel(price_history=duped, **base_kwargs)
    with pytest.raises(ValueError, match="absent from price_history"):
        build_pit1520_panel(price_history=ph, dates=["1900-01-01"], bars_loader=lambda _d: bars,
                            live_loader=lambda _d: None, screen=_screen())
    empty = build_pit1520_panel(price_history=ph, dates=[], bars_loader=lambda _d: bars,
                                live_loader=lambda _d: None, screen=_screen())
    assert len(empty.panel) == 0 and len(empty.exclusions) == 0 and len(empty.days) == 0


def test_build_panel_excludes_symbol_absent_from_price_history() -> None:
    ph = _two_day_history()
    bars = _kis_session("999999", 10050.0, 10050.0)
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    assert len(result.panel) == 0
    by_symbol = {r["symbol"]: r["reason"] for _, r in result.exclusions.iterrows()}
    assert by_symbol["999999"] == "not_in_price_history"


def test_build_panel_marks_prev_close_unavailable() -> None:
    ph = pd.DataFrame([
        _ph_row("2026-09-17", "005930", open=9900, close=9900, prev_close=9700, mc=1000.0, inst=5.0, foreign=6.0),
        _ph_row("2026-09-18", "005930", open=10050, close=0.0, prev_close=9800, close_raw=0.0),
    ])
    bars = _kis_session("005930", 10050.0, 10050.0)
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    assert len(result.panel) == 0
    assert result.exclusions.iloc[0]["reason"] == "prev_close_unavailable"


def test_build_panel_handles_empty_bar_day() -> None:
    ph = _two_day_history()
    cols = ("symbol", "ts_hms", "open", "high", "low", "close", "volume", "value_krw", "has_trade", "vendor")
    empty_bars = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    result = build_pit1520_panel(
        price_history=ph,
        dates=["2026-09-18"],
        bars_loader=lambda _d: empty_bars,
        live_loader=lambda _d: None,
        screen=_screen(),
    )
    assert len(result.panel) == 0
    assert int(result.days.iloc[0]["n_rows"]) == 0


def test_load_regular_bars_returns_empty_when_all_filtered(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    from src.data.intraday_store import intraday_partition_path

    target = intraday_partition_path(1, "2026-09-18", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_bar("005930", 153000, 72000, 72100, 71900, 72050, 500, "kis")]).to_parquet(target, index=False)
    assert len(load_regular_bars("2026-09-18")) == 0


def test_write_pit1520_panel_creates_fresh_files(tmp_path) -> None:
    paths = (
        tmp_path / "pit1520_panel.parquet",
        tmp_path / "pit1520_panel_exclusions.parquet",
        tmp_path / "pit1520_panel_days.parquet",
    )
    panel = pd.DataFrame([_panel_row("2026-09-18", "005930")])
    panel["date"] = pd.to_datetime(panel["date"])
    exclusions = pd.DataFrame([{"date": pd.Timestamp("2026-09-18"), "symbol": "000660",
                                "reason": "not_fetched", "detail": "x"}])
    days = pd.DataFrame([{"date": pd.Timestamp("2026-09-18"), "source": "bars", "n_rows": 1,
                          "n_excluded": 1, "n_superset": 2, "n_superset_present": 1,
                          "superset_coverage": 0.5, "index_basis": "eod_fallback"}])
    write_pit1520_panel(Pit1520PanelResult(panel=panel, exclusions=exclusions, days=days),
                        paths=paths, replace_dates=["2026-09-18"])
    assert len(pd.read_parquet(paths[0])) == 1
    assert len(pd.read_parquet(paths[1])) == 1
    assert len(pd.read_parquet(paths[2])) == 1


def _raw_price_history_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2026-09-17", "symbol": "005930", "market": "KOSPI", "open": 9900.0, "high": 9950.0,
         "low": 9850.0, "close": 9900.0, "close_raw": 9900.0, "prev_close": 9700.0, "volume": 1000.0,
         "market_cap_100m": 1000.0, "trade_value_100m": 500.0, "inst_netbuy": 5.0, "foreign_netbuy": 6.0,
         "kospi_pct": 0.005, "kosdaq_pct": 0.004, "v_kospi": 18.0, "v_kosdaq": 20.0},
        {"date": "2026-09-18", "symbol": "005930", "market": "KOSPI", "open": 10050.0, "high": 10100.0,
         "low": 10000.0, "close": 10050.0, "close_raw": 10050.0, "prev_close": 9800.0, "volume": 1000.0,
         "market_cap_100m": 900.0, "trade_value_100m": 500.0, "inst_netbuy": 0.0, "foreign_netbuy": 0.0,
         "kospi_pct": 0.005, "kosdaq_pct": 0.004, "v_kospi": 18.0, "v_kosdaq": 20.0},
    ])


def test_main_builds_bar_panel_to_out_dir(monkeypatch, tmp_path) -> None:
    from src.data.pit1520_panel import main

    ph_path = tmp_path / "price_history.parquet"
    _raw_price_history_frame().to_parquet(ph_path, index=False)
    monkeypatch.setattr("src.settings.PRICE_HISTORY_PARQUET_PATH", ph_path)
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    part = tmp_path / "intraday" / "1m" / "regular" / "2026-09" / "2026-09-18.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        _bar("005930", 90000, 10050, 10100, 10000, 10050, 100, "kis"),
        _bar("005930", 151900, 10050, 10100, 10000, 10050, 150, "kis"),
    ]).to_parquet(part, index=False)
    out = tmp_path / "out"
    main(["--out-dir", str(out)])
    got = pd.read_parquet(out / "pit1520_panel.parquet")
    assert len(got) == 1 and str(got.iloc[0]["symbol"]) == "005930"
    assert set(pd.read_parquet(out / "pit1520_panel_days.parquet")["source"]) == {"bars"}


def test_main_returns_without_writing_when_no_dates(monkeypatch, tmp_path) -> None:
    from src.data.pit1520_panel import main

    ph_path = tmp_path / "price_history.parquet"
    _raw_price_history_frame().to_parquet(ph_path, index=False)
    monkeypatch.setattr("src.settings.PRICE_HISTORY_PARQUET_PATH", ph_path)
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    out = tmp_path / "out"
    main(["--start", "2026-09-20", "--end", "2026-09-19", "--out-dir", str(out)])
    assert not (out / "pit1520_panel.parquet").exists()


def test_main_falls_back_to_calendar_start_without_partitions(monkeypatch, tmp_path) -> None:
    from src.data.pit1520_panel import main

    ph_path = tmp_path / "price_history.parquet"
    _raw_price_history_frame().to_parquet(ph_path, index=False)
    monkeypatch.setattr("src.settings.PRICE_HISTORY_PARQUET_PATH", ph_path)
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    out = tmp_path / "out"
    main(["--end", "2026-09-18", "--out-dir", str(out)])
    assert len(pd.read_parquet(out / "pit1520_panel.parquet")) == 0
    assert len(pd.read_parquet(out / "pit1520_panel_days.parquet")) == 2


def test_main_rejects_empty_price_history(monkeypatch, tmp_path) -> None:
    from src.data.pit1520_panel import main

    ph_path = tmp_path / "price_history.parquet"
    _raw_price_history_frame().iloc[0:0].to_parquet(ph_path, index=False)
    monkeypatch.setattr("src.settings.PRICE_HISTORY_PARQUET_PATH", ph_path)
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="no dates"):
        main(["--out-dir", str(tmp_path / "out")])


def test_default_panel_paths_under_history_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.settings.HISTORY_DIR", str(tmp_path))
    panel, exclusions, days = default_panel_paths()
    assert panel == Path(str(tmp_path)) / "pit1520_panel.parquet"
    assert exclusions == Path(str(tmp_path)) / "pit1520_panel_exclusions.parquet"
    assert days == Path(str(tmp_path)) / "pit1520_panel_days.parquet"
