from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.ml.universe import ScreenConfig
from src.ml.universe_research import UniverseScreenRecord, evaluate_universe_screen

pytestmark = pytest.mark.slow


def _price_history(n_days: int = 130, n_syms: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for s in range(n_syms):
        code = f"{s:06d}"
        px = 10000.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.025, size=n_days))
        for i, d in enumerate(dates):
            prev = px[i - 1] if i else px[0]
            rows.append({
                "date": d, "symbol": code, "open": prev, "high": max(px[i], prev) * 1.02,
                "low": min(px[i], prev) * 0.98, "close": px[i], "prev_close": prev,
                "market_cap_100m": 600.0 + 40 * s, "trade_value_100m": 120.0 + 8 * s,
                "daily_change_pct": px[i] / prev - 1.0, "market": "KOSPI" if s % 2 else "KOSDAQ",
                "volume": 1e5, "foreign_netbuy": rng.normal(0, 25), "inst_netbuy": rng.normal(0, 25),
                "program_netbuy": rng.normal(0, 12), "kospi_pct": rng.normal(0, 0.006),
                "kosdaq_pct": rng.normal(0, 0.007), "v_kospi": 19.0, "v_kosdaq": 23.0,
            })
    return pd.DataFrame(rows)


def test_record_and_oos_filter() -> None:
    ph = _price_history()
    screen = ScreenConfig(change_lower=-1.0, change_upper=None, min_trade_value_100m=100.0, min_market_cap_100m=500.0)
    common = dict(screen_name="liq_only", start_date="2023-01-02", end_date="2023-07-31",  # noqa: C408
                  n_splits=4, cpcv_n_groups=5, cpcv_k_test=2)

    rec_full = evaluate_universe_screen(ph, screen, **common)
    rec_oos = evaluate_universe_screen(ph, screen, oos_reserve_start="2023-05-01", **common)

    assert isinstance(rec_full, UniverseScreenRecord)
    assert rec_full.verdict in {"clears_cost", "below_cost", "insufficient_data"}
    assert rec_full.cost_ratio_bp == pytest.approx(46.0, abs=1.0)
    assert np.isfinite(rec_full.model_free_net_bp)
    assert rec_oos.n_days < rec_full.n_days
    if rec_full.verdict == "clears_cost":
        assert rec_full.ranked_top1_net_bp > 0.0 and rec_full.cpcv_top1_path_win_rate >= 0.60


import numpy as np
import pandas as pd
import pytest

from src.ml.universe import ScreenConfig
from src.ml.universe_research import run_universe_screen_grid
from tests.integration.ml.test_universe_research_harness import _price_history  # reuse S4 fixture

pytestmark = pytest.mark.slow


def test_grid_sorted() -> None:
    ph = _price_history()
    screens = {
        "liq_wide": ScreenConfig(change_lower=-1.0, change_upper=None, min_trade_value_100m=100.0, min_market_cap_100m=500.0),
        "modest_up": ScreenConfig(change_lower=0.01, change_upper=0.12, min_trade_value_100m=100.0, min_market_cap_100m=500.0),
    }

    df = run_universe_screen_grid(
        ph, screens, start_date="2023-01-02", end_date="2023-07-31",
        n_splits=4, cpcv_n_groups=5, cpcv_k_test=2,
    )

    assert list(df["screen_name"]) and len(df) == 2
    net = df["ranked_top1_net_bp"].to_numpy(dtype=float)
    finite = net[np.isfinite(net)]
    assert (np.diff(finite) <= 1e-9).all()  # descending

    with pytest.raises(ValueError, match="non-empty"):
        run_universe_screen_grid(ph, {}, start_date="2023-01-02", end_date="2023-07-31")


import pandas as pd
import pytest

from src.ml.universe_research import main
from tests.integration.ml.test_universe_research_harness import _price_history

pytestmark = pytest.mark.slow


def test_universe_research_main_writes_grid_parquet(tmp_path) -> None:
    ph = _price_history(n_days=90, n_syms=20)
    ph_path = tmp_path / "price_history.parquet"
    ph.to_parquet(ph_path)
    out_path = tmp_path / "grid.parquet"

    main(["--price-history", str(ph_path), "--theme", str(tmp_path / "no_theme.parquet"),
          "--out", str(out_path), "--cpcv-n-groups", "5", "--cpcv-k-test", "2"])

    df = pd.read_parquet(out_path)
    assert len(df) == 6 and "screen_name" in df.columns and "ranked_top1_net_bp" in df.columns

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--price-history", str(tmp_path / "missing.parquet"), "--out", str(out_path)])
