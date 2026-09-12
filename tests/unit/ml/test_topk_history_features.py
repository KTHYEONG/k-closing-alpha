"""Top-K history feature unit tests."""
from __future__ import annotations

def _panel(n_days: int = 90, symbols: tuple[str, ...] = ("000001", "000002"), seed: int = 3):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []
    for s in symbols:
        close = 10000.0
        for d in dates:
            prev = close
            close = float(round(prev * (1.0 + rng.normal(0.003, 0.03)), 0))
            op = float(round(prev * (1.0 + rng.normal(0.0, 0.01)), 0))
            rows.append({
                "date": d, "symbol": s, "open": op, "close": close, "prev_close": prev,
                "volume": float(rng.integers(10_000, 50_000)),
                "inst_netbuy": float(rng.normal(0.0, 1e7)), "foreign_netbuy": float(rng.normal(0.0, 1e7)),
            })
    return pd.DataFrame(rows)


def test_compute_topk_history_features_uses_only_past_rows() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.topk_history_features import TOPK_HISTORY_FEATURE_COLS, compute_topk_history_features

    # Given: a panel and a copy whose rows after the cutoff are scrambled
    panel = _panel()
    cutoff = pd.Timestamp(sorted(panel["date"].unique())[70])
    future = panel["date"] > cutoff
    scrambled = panel.copy()
    scrambled.loc[future, ["open", "close", "prev_close", "volume", "inst_netbuy", "foreign_netbuy"]] *= 3.7

    # When
    a = compute_topk_history_features(panel)
    b = compute_topk_history_features(scrambled)

    # Then: every feature on or before the cutoff is identical (no look-ahead)
    ka = a[a["date"] <= cutoff].set_index(["date", "symbol"])[list(TOPK_HISTORY_FEATURE_COLS)]
    kb = b[b["date"] <= cutoff].set_index(["date", "symbol"])[list(TOPK_HISTORY_FEATURE_COLS)].loc[ka.index]
    assert np.allclose(ka.to_numpy(), kb.to_numpy(), equal_nan=True)
    assert set(a.columns) == {"date", "symbol", *TOPK_HISTORY_FEATURE_COLS}
    assert len(a) == len(panel)


def test_compute_topk_history_features_matches_hand_computed_windows() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import compute_topk_history_features

    # Given: one symbol, 12 days with known closes/opens
    dates = pd.bdate_range("2025-03-03", periods=12)
    closes = [100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 110.0, 109.0, 112.0, 111.0, 115.0]
    prevs = [99.0, *closes[:-1]]
    opens = [p * 1.01 for p in prevs]
    panel = pd.DataFrame({
        "date": dates, "symbol": "000777", "open": opens, "close": closes, "prev_close": prevs,
        "volume": 1000.0, "inst_netbuy": 5000.0, "foreign_netbuy": -2000.0,
    })

    # When
    out = compute_topk_history_features(panel).set_index("date")

    # Then: f_ret5 at day 7 is the log-return sum over days 2..6 (window [t-5, t-1])
    lr = np.log(np.array(closes) / np.array(prevs))
    assert out["f_ret5"].iloc[7] == pytest.approx(lr[2:7].sum())
    # Then: min_periods = max(3, w // 2) -> f_ret5 needs 3 lagged obs, f_ret20 needs 10
    assert np.isnan(out["f_ret5"].iloc[2])
    assert out["f_ret5"].iloc[3] == pytest.approx(lr[0:3].sum())
    assert np.isnan(out["f_ret20"].iloc[9])
    assert out["f_ret20"].iloc[10] == pytest.approx(lr[0:10].sum())
    # Then: f_gap is today's open over prev_close, known at decision time
    assert out["f_gap"].to_numpy() == pytest.approx(np.full(12, 0.01))
    # Then: 5-day flow is net-buy KRW over traded value KRW, including today
    assert out["f_inst_cum5"].iloc[6] == pytest.approx(5 * 5000.0 / sum(c * 1000.0 for c in closes[2:7]))
    # Then: dist_high60 needs 20 observations, so a 12-day history yields NaN
    assert out["f_dist_high60"].isna().all()


def test_compute_topk_history_features_upnext_needs_three_prior_up_days() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import compute_topk_history_features

    # Given: up days (+5%) on days 1, 3, 5; next-day gaps on days 2, 4, 6 are 1%, 2%, 3%
    dates = pd.bdate_range("2025-04-01", periods=8)
    chg = [0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0]
    gaps = [0.0, 0.0, 0.01, 0.0, 0.02, 0.0, 0.03, 0.0]
    rows, prev = [], 1000.0
    for d, c, g in zip(dates, chg, gaps, strict=True):
        close = prev * (1.0 + c)
        rows.append({"date": d, "symbol": "000555", "open": prev * (1.0 + g), "close": close, "prev_close": prev,
                     "volume": 100.0, "inst_netbuy": 0.0, "foreign_netbuy": 0.0})
        prev = close
    out = compute_topk_history_features(pd.DataFrame(rows)).set_index("date")["f_upnext_on60"]

    # Then: until the third observed up->next-gap pair it is NaN; from day 6 (gap of day 6 known) it is the mean
    assert np.isnan(out.iloc[5])
    assert out.iloc[6] == pytest.approx(0.02)
    assert out.iloc[7] == pytest.approx(0.02)


def test_compute_topk_history_features_treats_zero_open_as_missing_gap() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.topk_history_features import compute_topk_history_features

    # Given: a halted day with open 0 (no auction print)
    panel = _panel(n_days=30, symbols=("000009",))
    panel.loc[5, "open"] = 0.0

    # When
    out = compute_topk_history_features(panel)

    # Then: the gap is NaN, never a -100% overnight return
    assert np.isnan(out["f_gap"].iloc[5])
    assert (out["f_gap"].dropna() > -0.5).all()


def test_compute_topk_history_features_rejects_missing_columns_and_duplicates() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import compute_topk_history_features

    panel = _panel(n_days=10, symbols=("000001",))
    with pytest.raises(ValueError, match="prev_close"):
        compute_topk_history_features(panel.drop(columns=["prev_close"]))
    with pytest.raises(ValueError, match="duplicate"):
        compute_topk_history_features(pd.concat([panel, panel.iloc[[0]]]))


def test_compute_topk_history_features_truncated_window_matches_full_history() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.topk_history_features import (
        HISTORY_LOOKBACK_CALENDAR_DAYS,
        TOPK_HISTORY_FEATURE_COLS,
        compute_topk_history_features,
    )

    # Given: 300 business days of history and the serving lookback window
    panel = _panel(n_days=300)
    last = panel["date"].max()
    trunc = panel[panel["date"] >= last - pd.Timedelta(days=HISTORY_LOOKBACK_CALENDAR_DAYS)]

    # When
    full = compute_topk_history_features(panel)
    part = compute_topk_history_features(trunc)

    # Then: decision-day features are identical (serving window is long enough for every rolling window)
    f = full[full["date"] == last].set_index("symbol")[list(TOPK_HISTORY_FEATURE_COLS)].sort_index()
    p = part[part["date"] == last].set_index("symbol")[list(TOPK_HISTORY_FEATURE_COLS)].sort_index()
    assert np.allclose(f.to_numpy(), p.to_numpy(), rtol=0.0, atol=1e-12, equal_nan=True)
    assert f.notna().all().all()


def test_attach_topk_features_preserves_row_order_and_adds_cost_features() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import TOPK_COST_FEATURE_COLS, TOPK_HISTORY_FEATURE_COLS, attach_topk_features

    # Given: candidates listed in reverse symbol order with a non-default index
    panel = _panel(n_days=70)
    last = panel["date"].max()
    today = panel[panel["date"] == last]
    cands = pd.DataFrame({
        "date": [last, last], "symbol": ["000002", "000001"],
        "close": [today.loc[today.symbol == "000002", "close"].iloc[0], today.loc[today.symbol == "000001", "close"].iloc[0]],
        "tick_cost_bp": [5.0, 6.5],
    }, index=[10, 11])

    # When
    out = attach_topk_features(cands, panel)

    # Then: same index and order, cost features derived row-wise, history joined per symbol
    assert out.index.tolist() == [10, 11]
    assert out["symbol"].tolist() == ["000002", "000001"]
    assert out["f_tick_cost"].tolist() == [5.0, 6.5]
    assert out["f_log_close"].to_numpy() == pytest.approx(np.log(cands["close"].to_numpy()))
    for col in (*TOPK_COST_FEATURE_COLS, *TOPK_HISTORY_FEATURE_COLS):
        assert col in out.columns


def test_attach_topk_features_requires_tick_cost_column() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import attach_topk_features

    panel = _panel(n_days=10, symbols=("000001",))
    cands = pd.DataFrame({"date": [panel["date"].max()], "symbol": ["000001"], "close": [10000.0]})
    with pytest.raises(ValueError, match="tick_cost_bp"):
        attach_topk_features(cands, panel)


def test_stitch_live_panel_drops_same_day_history_and_appends_live_rows() -> None:
    import pandas as pd

    from src.ml.topk_history_features import HISTORY_REQUIRED_COLUMNS, stitch_live_panel

    # Given: history that already (wrongly) contains the decision day, plus the live snapshot rows
    panel = _panel(n_days=20, symbols=("000001",))
    decision = panel["date"].max()
    live = pd.DataFrame({"symbol": ["000001", "000003"], "open": [1.0, 2.0], "close": [1.1, 2.2],
                         "prev_close": [1.0, 2.0], "volume": [10.0, 20.0],
                         "inst_netbuy": [0.0, 0.0], "foreign_netbuy": [0.0, 0.0]})

    # When
    out = stitch_live_panel(panel, live, decision)

    # Then: only strictly-past history survives and the live rows carry the decision date
    assert list(out.columns) == list(HISTORY_REQUIRED_COLUMNS)
    today = out[out["date"] == decision]
    assert sorted(today["symbol"].tolist()) == ["000001", "000003"]
    assert today.loc[today["symbol"] == "000001", "close"].iloc[0] == 1.1
    assert (out["date"] <= decision).all()
    assert len(out) == 19 + 2


def test_stitch_live_panel_rejects_empty_or_duplicate_live_rows() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import stitch_live_panel

    panel = _panel(n_days=5, symbols=("000001",))
    cols = {"symbol": [], "open": [], "close": [], "prev_close": [], "volume": [], "inst_netbuy": [], "foreign_netbuy": []}
    with pytest.raises(ValueError, match="empty"):
        stitch_live_panel(panel, pd.DataFrame(cols), pd.Timestamp("2025-02-03"))
    dup = pd.DataFrame({**{k: [1.0, 1.0] for k in cols}, "symbol": ["000001", "000001"]})
    with pytest.raises(ValueError, match="duplicate"):
        stitch_live_panel(panel, dup, pd.Timestamp("2025-02-03"))
    with pytest.raises(ValueError, match="open"):
        stitch_live_panel(panel, dup.drop(columns=["open"]), pd.Timestamp("2025-02-03"))
    valid_live = pd.DataFrame({**{k: [1.0] for k in cols}, "symbol": ["000001"]})
    with pytest.raises(ValueError, match="price_history missing required columns"):
        stitch_live_panel(panel.drop(columns=["volume"]), valid_live, pd.Timestamp("2025-02-03"))


def test_resolve_prev_trading_day_skips_weekends_and_holidays() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import resolve_prev_trading_day

    # Given: decision Monday 2026-10-05; Friday 10-02 is a holiday per the oracle
    asked: list[pd.Timestamp] = []

    def _oracle(d: pd.Timestamp) -> bool:
        asked.append(d)
        return d != pd.Timestamp("2026-10-02")

    # When
    prev = resolve_prev_trading_day(pd.Timestamp("2026-10-05 15:21"), _oracle)

    # Then: Thursday is returned and weekend days never reach the oracle
    assert prev == pd.Timestamp("2026-10-01")
    assert all(d.weekday() < 5 for d in asked)
    with pytest.raises(ValueError, match="no trading day"):
        resolve_prev_trading_day(pd.Timestamp("2026-10-05"), lambda _d: False)


def test_assert_history_fresh_raises_on_stale_or_empty_history() -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import assert_history_fresh

    hist = pd.DataFrame({"date": pd.to_datetime(["2026-09-08", "2026-09-09"])})
    assert assert_history_fresh(hist, pd.Timestamp("2026-09-09")) is None
    with pytest.raises(ValueError, match="stale"):
        assert_history_fresh(hist, pd.Timestamp("2026-09-10"))
    with pytest.raises(ValueError, match="empty"):
        assert_history_fresh(hist.iloc[0:0], pd.Timestamp("2026-09-10"))


def test_load_serving_price_history_reads_pit_window_and_fails_closed(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.ml.topk_history_features import HISTORY_REQUIRED_COLUMNS, load_serving_price_history

    # Given: a parquet whose last row is the decision day itself (must be excluded) and D-1 present
    panel = _panel(n_days=260, symbols=("000001",))
    panel["extra_col"] = 1.0
    path = tmp_path / "ph.parquet"
    panel.to_parquet(path)
    decision = panel["date"].max()
    prev = sorted(panel["date"].unique())[-2]

    # When
    out = load_serving_price_history(decision, path=path, is_trading_day=lambda _d: True)

    # Then: only required columns, strictly before the decision day, within the lookback window
    assert list(out.columns) == list(HISTORY_REQUIRED_COLUMNS)
    assert pd.to_datetime(out["date"]).max() == pd.Timestamp(prev)
    assert pd.to_datetime(out["date"]).min() >= decision - pd.Timedelta(days=200)

    # Then: history ending two trading days back is stale
    stale = panel[panel["date"] < prev]
    stale.to_parquet(path)
    with pytest.raises(ValueError, match="stale"):
        load_serving_price_history(decision, path=path, is_trading_day=lambda _d: True)
    with pytest.raises(FileNotFoundError):
        load_serving_price_history(decision, path=tmp_path / "missing.parquet", is_trading_day=lambda _d: True)
    # Re-save fresh panel to test default is_trading_day (line 286) with a mid-year date.
    # 기본 오라클(is_krx_trading_day)은 실 네트워크/자격증명이 필요하므로, '배선이
    # trading_calendar.is_krx_trading_day 를 가져다 쓰는지'만 결정적으로 검증한다.
    import unittest.mock

    mid_decision = pd.Timestamp("2025-06-16")  # Monday
    panel.to_parquet(path)
    with unittest.mock.patch("src.data.trading_calendar.is_krx_trading_day", return_value=True):
        out_default = load_serving_price_history(mid_decision, path=path)
    assert not out_default.empty

