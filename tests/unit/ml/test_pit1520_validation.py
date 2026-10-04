"""Invariant guards for decision-time panel live-parity validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.eod_superset import EodSupersetScreen
from src.data.pit1520_panel import PIT1520_PANEL_COLUMNS
from src.ml.research.pit1520_validation import (
    PanelValidationConfig,
    PanelValidationReport,
    compute_feature_parity,
)
from src.ml.topk_contract import RANKER_FEATURE_COLS

SYMBOLS = [f"{100000 + i:06d}" for i in range(8)]
HIST_DAYS = 100
N_OVERLAP = 5


def _screen() -> EodSupersetScreen:
    return EodSupersetScreen(
        min_change_ratio=0.01,
        max_change_ratio=0.12,
        min_trade_value_100m=100.0,
        min_market_cap_100m=495.0,
        common_stock_only=False,
    )


def _history(rng: np.random.Generator, symbols: list[str], days: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for i, symbol in enumerate(symbols):
        prev = 40000.0 + i * 5000.0
        for day in days:
            chg = float(rng.uniform(-0.08, 0.10))
            close = prev * (1.0 + chg)
            open_ = prev * (1.0 + float(rng.uniform(-0.03, 0.03)))
            rows.append({
                "date": day,
                "symbol": symbol,
                "open": open_,
                "close": close,
                "prev_close": prev,
                "volume": float(rng.integers(200000, 2000000)),
                "inst_netbuy": float(rng.integers(-10**8, 10**8)),
                "foreign_netbuy": float(rng.integers(-10**8, 10**8)),
                "chg_ratio": chg,
                "tv_clean": 500.0,
                "mc_clean": 8000.0 + i * 1000.0,
            })
            prev = close
    return pd.DataFrame(rows)


def _snapshot_row(symbol, idx, *, o, h, l, c, prev, market, kospi=0.5, kosdaq=0.4):
    return {
        "종목코드": symbol,
        "시가": o,
        "고가": h,
        "저가": l,
        "종가": c,
        "전일종가": prev,
        "거래량": 800000.0,
        "거래대금": c * 800000.0 / 1e8,
        "시가총액": 8000.0 + idx * 1000.0,
        "시장구분": market,
        "기관_순매수": 1e6,
        "외국인_순매수": -1e6,
        "kospi": kospi,
        "kosdaq": kosdaq,
        "v_kospi": 18.0,
    }


def _eod_rows_from_snapshot(day, live: pd.DataFrame) -> pd.DataFrame:
    """EOD price_history rows for T equal to the snapshot (decision state == close in this fixture)."""
    return pd.DataFrame({
        "date": pd.Timestamp(day),
        "symbol": live["종목코드"].astype(str),
        "open": live["시가"].astype(float),
        "close": live["종가"].astype(float),
        "prev_close": live["전일종가"].astype(float),
        "volume": live["거래량"].astype(float),
        "inst_netbuy": live["기관_순매수"].astype(float),
        "foreign_netbuy": live["외국인_순매수"].astype(float),
        "chg_ratio": live["종가"].astype(float) / live["전일종가"].astype(float) - 1.0,
        "tv_clean": live["거래대금"].astype(float),
        "mc_clean": live["시가총액"].astype(float),
    })


def _panel_row(day, symbol, snap: dict, *, kospi_pct=None, kosdaq_pct=None):
    return {
        "date": pd.Timestamp(day),
        "symbol": symbol,
        "market": snap["시장구분"],
        "open": float(snap["시가"]),
        "high": float(snap["고가"]),
        "low": float(snap["저가"]),
        "close": float(snap["종가"]),
        "close_raw": float(snap["종가"]),
        "prev_close": float(snap["전일종가"]),
        "volume": float(snap["거래량"]),
        "trade_value_100m": float(snap["거래대금"]),
        "market_cap_100m": float(snap["시가총액"]),
        "inst_netbuy": float("nan"),
        "foreign_netbuy": float("nan"),
        "inst_netbuy_prev": 1e6,
        "foreign_netbuy_prev": -1e6,
        "kospi_pct": float(snap["kospi"]) / 100.0 if kospi_pct is None else kospi_pct,
        "kosdaq_pct": float(snap["kosdaq"]) / 100.0 if kosdaq_pct is None else kosdaq_pct,
        "v_kospi": 18.0,
        "v_kosdaq": 20.0,
        "index_basis": "eod_fallback",
        "source": "bars",
        "bars_vendor": "kis",
        "n_bars": 300,
        "first_bar_hms": "90000",
        "last_bar_hms": "151900",
        "capture_run_id": "",
    }


def _fixture(seed: int = 1520, n_overlap: int = N_OVERLAP):
    from src.serving.realtime.features import build_topk_ranker_features

    rng = np.random.default_rng(seed)
    hist_days = pd.bdate_range("2026-01-05", periods=HIST_DAYS)
    eval_days = pd.bdate_range(hist_days[-1] + pd.Timedelta(days=1), periods=n_overlap)
    history = _history(rng, SYMBOLS, hist_days)
    last_close = {s: float(history[history["symbol"] == s].iloc[-1]["close"]) for s in SYMBOLS}
    live_inputs: dict[str, pd.DataFrame] = {}
    panel_rows = []
    eod_rows = []
    for day in eval_days:
        snaps = []
        for i, symbol in enumerate(SYMBOLS):
            prev = last_close[symbol]
            chg = 0.045 + 0.004 * i + float(rng.uniform(-0.004, 0.004))
            close = prev * (1.0 + chg)
            open_ = prev * (1.0 + float(rng.uniform(-0.01, 0.01)))
            high = max(open_, close) * 1.005
            low = min(open_, close) * 0.995
            market = "KOSPI" if i % 2 == 0 else "KOSDAQ"
            snaps.append(_snapshot_row(symbol, i, o=open_, h=high, l=low, c=close, prev=prev, market=market))
            last_close[symbol] = close
        live = pd.DataFrame(snaps)
        day_label = pd.Timestamp(day).strftime("%Y-%m-%d")
        live_inputs[day_label] = live
        panel_rows.extend(_panel_row(day, s, snap) for s, snap in zip(SYMBOLS, snaps, strict=True))
        hist = history[pd.to_datetime(history["date"]) < pd.Timestamp(day)].copy()
        feats = build_topk_ranker_features(live, pd.Timestamp(day), price_history=hist)
        feats["date"] = pd.Timestamp(day)
        eod_rows.append(feats)
        history = pd.concat([history, _eod_rows_from_snapshot(day, live)], ignore_index=True)
    panel = pd.DataFrame(panel_rows, columns=list(PIT1520_PANEL_COLUMNS))
    eod_pool = pd.concat(eod_rows, ignore_index=True)
    return live_inputs, panel, eod_pool, history


def test_feature_parity_identical_sources_pass() -> None:
    live_inputs, panel, eod_pool, history = _fixture()
    report = compute_feature_parity(
        live_inputs=live_inputs, panel=panel, eod_pool=eod_pool, price_history=history, screen=_screen()
    )
    assert report.verdict == "PASS", report.reasons
    assert report.n_overlap_days == N_OVERLAP
    summary = report.frame[report.frame["row_type"] == "feature_summary"]
    gated = summary[(summary["comparison"] == "panel_vs_live") & (~summary["feature"].isin(("kospi_pct", "kosdaq_pct", "v_kospi")))]
    assert len(gated) == len(RANKER_FEATURE_COLS) - 3
    assert np.allclose(gated["spearman"].to_numpy(dtype=np.float64), 1.0, rtol=1e-9, atol=1e-12)


def test_feature_parity_names_failing_feature() -> None:
    live_inputs, panel, eod_pool, history = _fixture()
    rng = np.random.default_rng(99)
    noisy = panel.copy()
    jitter = rng.uniform(-0.03, 0.03, len(noisy))
    noisy["close"] = np.asarray(noisy["close"], dtype=np.float64) * (1.0 + jitter)
    report = compute_feature_parity(
        live_inputs=live_inputs, panel=noisy, eod_pool=eod_pool, price_history=history, screen=_screen()
    )
    assert report.verdict == "FAIL"
    assert any("chg_ratio" in reason for reason in report.reasons)


def test_feature_parity_requires_panel_not_worse_than_eod() -> None:
    live_inputs, panel, eod_pool, history = _fixture()
    rng = np.random.default_rng(99)
    noisy = panel.copy()
    jitter = rng.uniform(-0.03, 0.03, len(noisy))
    noisy["close"] = np.asarray(noisy["close"], dtype=np.float64) * (1.0 + jitter)
    report = compute_feature_parity(
        live_inputs=live_inputs,
        panel=noisy,
        eod_pool=eod_pool,
        price_history=history,
        screen=_screen(),
        config=PanelValidationConfig(min_feature_spearman=0.5, require_not_worse_than_eod=True),
    )
    assert report.verdict == "FAIL"
    assert any("not_worse_than_eod" in reason for reason in report.reasons)


def test_feature_parity_measures_superset_recall_on_price_history_t() -> None:
    live_inputs, panel, eod_pool, history = _fixture()
    first_day = sorted(live_inputs)[0]
    history.loc[
        (pd.to_datetime(history["date"]).dt.strftime("%Y-%m-%d") == first_day)
        & (history["symbol"] == SYMBOLS[0]),
        "chg_ratio",
    ] = 0.005
    report = compute_feature_parity(
        live_inputs=live_inputs, panel=panel, eod_pool=eod_pool, price_history=history, screen=_screen()
    )
    coverage = report.frame[report.frame["row_type"] == "day_coverage"]
    row = coverage[pd.to_datetime(coverage["date"]).dt.strftime("%Y-%m-%d") == first_day].iloc[0]
    assert float(row["superset_recall"]) == pytest.approx((len(SYMBOLS) - 1) / len(SYMBOLS))
    assert report.verdict == "FAIL"


def test_feature_parity_superset_recall_ignores_training_pool_screen() -> None:
    live_inputs, panel, eod_pool, history = _fixture()
    first_day = sorted(live_inputs)[0]
    pool_dates = pd.to_datetime(eod_pool["date"]).dt.strftime("%Y-%m-%d")
    eod_pool = eod_pool[~((pool_dates == first_day) & (eod_pool["symbol"] == SYMBOLS[0]))]
    report = compute_feature_parity(
        live_inputs=live_inputs, panel=panel, eod_pool=eod_pool, price_history=history, screen=_screen()
    )
    coverage = report.frame[report.frame["row_type"] == "day_coverage"]
    row = coverage[pd.to_datetime(coverage["date"]).dt.strftime("%Y-%m-%d") == first_day].iloc[0]
    assert float(row["superset_recall"]) == 1.0


def test_feature_parity_insufficient_overlap() -> None:
    live_inputs, panel, eod_pool, history = _fixture(n_overlap=4)
    report = compute_feature_parity(
        live_inputs=live_inputs, panel=panel, eod_pool=eod_pool, price_history=history, screen=_screen()
    )
    assert report.verdict == "INSUFFICIENT_OVERLAP"
    assert report.n_overlap_days == 4


def test_feature_parity_constant_features_are_reported_only() -> None:
    live_inputs, panel, eod_pool, history = _fixture()
    shifted = panel.copy()
    shifted["kospi_pct"] = np.asarray(shifted["kospi_pct"], dtype=np.float64) + 0.003
    report = compute_feature_parity(
        live_inputs=live_inputs, panel=shifted, eod_pool=eod_pool, price_history=history, screen=_screen()
    )
    assert report.verdict == "PASS", report.reasons
    summary = report.frame[(report.frame["row_type"] == "feature_summary")
                           & (report.frame["feature"] == "kospi_pct")
                           & (report.frame["comparison"] == "panel_vs_live")]
    assert float(summary.iloc[0]["median_abs_diff"]) == pytest.approx(0.003)


def test_feature_parity_history_excludes_t_and_future() -> None:
    live_inputs, panel, eod_pool, history = _fixture()
    base = compute_feature_parity(
        live_inputs=live_inputs, panel=panel, eod_pool=eod_pool, price_history=history, screen=_screen()
    )
    rng = np.random.default_rng(31337)
    last_day = sorted(live_inputs)[-1]
    extra_days = pd.bdate_range(last_day, periods=N_OVERLAP + 1)
    extra = pd.DataFrame([{
        "date": day,
        "symbol": symbol,
        "open": float(rng.uniform(1, 99999)),
        "close": float(rng.uniform(1, 99999)),
        "prev_close": float(rng.uniform(1, 99999)),
        "volume": float(rng.integers(1, 99999)),
        "inst_netbuy": float(rng.integers(-10**8, 10**8)),
        "foreign_netbuy": float(rng.integers(-10**8, 10**8)),
    } for day in extra_days for symbol in SYMBOLS])
    extended = pd.concat([history, extra], ignore_index=True)
    rerun = compute_feature_parity(
        live_inputs=live_inputs, panel=panel, eod_pool=eod_pool, price_history=extended, screen=_screen()
    )
    pd.testing.assert_frame_equal(base.frame, rerun.frame)


def test_pit1520_validation_cli_exits_nonzero_on_fail(monkeypatch, tmp_path) -> None:
    import src.ml.research.pit1520_validation as validation

    frame = pd.DataFrame([{
        "row_type": "feature_summary",
        "date": pd.NaT,
        "feature": "chg_ratio",
        "comparison": "panel_vs_live",
        "spearman": 0.5,
        "median_abs_diff": 0.01,
        "n_symbols": 8,
        "superset_recall": 1.0,
        "rank_pool_coverage": 1.0,
    }])
    failing = PanelValidationReport(verdict="FAIL", reasons=("boom",), n_overlap_days=5, frame=frame)
    passing = PanelValidationReport(verdict="PASS", reasons=(), n_overlap_days=5, frame=frame)
    monkeypatch.setattr(validation, "_load_validation_inputs", lambda _s, _e: ({}, frame, frame, frame, _screen()))
    monkeypatch.setattr(validation, "compute_feature_parity", lambda **_k: failing)
    out = tmp_path / "validation.parquet"
    with pytest.raises(SystemExit) as exc:
        validation.main(["--out", str(out)])
    assert exc.value.code == 1
    assert out.exists()
    monkeypatch.setattr(validation, "compute_feature_parity", lambda **_k: passing)
    validation.main(["--out", str(out)])
    assert out.exists()


def test_default_live_dates_reads_date_from_capture_layout(monkeypatch, tmp_path) -> None:
    import src.data.capture_store as capture_store
    from src.ml.research import pit1520_validation

    for day, run in (("2026-09-18", "decision-2026-09-18-62f2a969"), ("2026-09-21", "decision-2026-09-21-4946bac5")):
        target = tmp_path / "decision" / day / run / "input.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x")
    monkeypatch.setattr(capture_store, "resolve_capture_root", lambda: tmp_path)

    assert pit1520_validation._default_live_dates() == ["2026-09-18", "2026-09-21"]

