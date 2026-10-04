"""Train/serve feature-vector parity for the top-k ranker."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _parity_panel() -> pd.DataFrame:
    from src.data.panel_integrity import prepare_price_panel

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=90)
    rows = []
    for i in range(6):
        market = "KOSPI" if i % 2 == 0 else "KOSDAQ"
        prev = 50000.0 + i * 10000.0
        for d in dates:
            chg = float(rng.uniform(-0.08, 0.10))
            close = prev * (1.0 + chg)
            open_ = prev * (1.0 + float(rng.uniform(-0.03, 0.03)))
            high = max(open_, close) * (1.0 + float(rng.uniform(0.0, 0.02)))
            low = min(open_, close) * (1.0 - float(rng.uniform(0.0, 0.02)))
            volume = float(rng.integers(100000, 2000000))
            trade_value_100m = close * volume / 1e8 * float(rng.uniform(0.9, 1.1))
            market_cap_100m = close * 1e7 / 1e8
            rows.append({
                "date": d,
                "symbol": f"{i:06d}",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "prev_close": prev,
                "volume": volume,
                "market_cap_100m": market_cap_100m,
                "trade_value_100m": trade_value_100m,
                "market": market,
                "daily_change_pct": chg,
                "inst_netbuy": float(rng.integers(-10**8, 10**8)),
                "foreign_netbuy": float(rng.integers(-10**8, 10**8)),
                "program_netbuy": 0.0,
                "kospi_pct": float(rng.normal(0, 0.005)),
                "kosdaq_pct": float(rng.normal(0, 0.006)),
                "v_kospi": 18.0,
                "v_kosdaq": 22.0,
            })
            prev = close
    ph, _prov = prepare_price_panel(pd.DataFrame(rows))
    assert "close_raw" not in ph.columns
    return ph


def _train_day_features(ph: pd.DataFrame) -> pd.DataFrame:
    from src.ml import topk_contract
    from src.ml.topk_history_features import attach_lagged_flow_features, attach_topk_features

    t = pd.Timestamp(pd.to_datetime(ph["date"]).max())
    day = ph[pd.to_datetime(ph["date"]) == t].copy().sort_values("symbol").reset_index(drop=True)
    cands = topk_contract.compute_derived_features(day)
    cands = attach_lagged_flow_features(cands, ph)
    return attach_topk_features(cands, ph).sort_values("symbol").reset_index(drop=True)


def _serve_day_features(ph: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    from src.ml.topk_history_features import HISTORY_REQUIRED_COLUMNS
    from src.serving.realtime.features import build_topk_ranker_features

    t = pd.Timestamp(pd.to_datetime(ph["date"]).max())
    hist = ph[pd.to_datetime(ph["date"]) < t][list(HISTORY_REQUIRED_COLUMNS)]
    return build_topk_ranker_features(snapshot, t, price_history=hist)


def _day_snapshot(day: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "종목코드": day["symbol"].astype(str).to_numpy(),
        "종가": day["close"].to_numpy(dtype=np.float64),
        "전일종가": day["prev_close"].to_numpy(dtype=np.float64),
        "고가": day["high"].to_numpy(dtype=np.float64),
        "저가": day["low"].to_numpy(dtype=np.float64),
        "시가": day["open"].to_numpy(dtype=np.float64),
        "거래량": day["volume"].to_numpy(dtype=np.float64),
        "거래대금": day["tv_clean"].to_numpy(dtype=np.float64),
        "시가총액": day["mc_clean"].to_numpy(dtype=np.float64),
        "기관_순매수": day["inst_netbuy"].to_numpy(dtype=np.float64),
        "외국인_순매수": day["foreign_netbuy"].to_numpy(dtype=np.float64),
        "kospi": day["kospi_pct"].to_numpy(dtype=np.float64) * 100.0,
        "kosdaq": day["kosdaq_pct"].to_numpy(dtype=np.float64) * 100.0,
        "v_kospi": day["v_kospi"].to_numpy(dtype=np.float64),
        "시장구분": day["market"].astype(str).to_numpy(),
    })


def test_train_serve_feature_vectors_agree() -> None:
    from src.ml import topk_contract

    ph = _parity_panel()
    t = pd.Timestamp(pd.to_datetime(ph["date"]).max())
    day = ph[pd.to_datetime(ph["date"]) == t].copy().sort_values("symbol").reset_index(drop=True)
    train = _train_day_features(ph)
    served = _serve_day_features(ph, _day_snapshot(day))
    served = served.set_index(served["symbol"].astype(str)).loc[train["symbol"].astype(str)].reset_index(drop=True)

    assert len(train) == 6
    for name in topk_contract.RANKER_FEATURE_COLS:
        first = train[name].to_numpy(dtype=np.float64)
        second = served[name].to_numpy(dtype=np.float64)
        assert np.allclose(first, second, rtol=1e-9, atol=1e-12, equal_nan=True), name
    assert int(train[topk_contract.RANKER_FEATURE_COLS].isna().to_numpy().sum()) == 0


def test_parity_detects_unit_drift() -> None:
    from src.ml import topk_contract

    ph = _parity_panel()
    t = pd.Timestamp(pd.to_datetime(ph["date"]).max())
    day = ph[pd.to_datetime(ph["date"]) == t].copy().sort_values("symbol").reset_index(drop=True)
    train = _train_day_features(ph)
    snapshot = _day_snapshot(day)
    snapshot["kospi"] = day["kospi_pct"].to_numpy(dtype=np.float64)
    served = _serve_day_features(ph, snapshot)
    served = served.set_index(served["symbol"].astype(str)).loc[train["symbol"].astype(str)].reset_index(drop=True)

    assert not np.allclose(
        train["kospi_pct"].to_numpy(dtype=np.float64),
        served["kospi_pct"].to_numpy(dtype=np.float64),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
    )
