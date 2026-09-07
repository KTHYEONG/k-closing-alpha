from __future__ import annotations
import numpy as np
import pandas as pd

from src.ml.decision_labels import DECISION_LABEL_COLUMNS, assert_no_label_leakage
from src.ml.universe import ScreenConfig
from src.ml.universe_research import build_universe_training_panel


def _price_history(n_days: int = 90, n_syms: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for s in range(n_syms):
        code = f"{s:06d}"
        px = 10000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.02, size=n_days))
        for i, d in enumerate(dates):
            prev = px[i - 1] if i else px[0]
            chg = px[i] / prev - 1.0
            rows.append({
                "date": d, "symbol": code, "open": prev, "high": max(px[i], prev) * 1.01,
                "low": min(px[i], prev) * 0.99, "close": px[i], "prev_close": prev,
                "market_cap_100m": 800.0 + 50 * s, "trade_value_100m": 150.0 + 10 * s,
                "daily_change_pct": chg, "market": "KOSPI" if s % 2 else "KOSDAQ",
                "volume": 1e5, "foreign_netbuy": rng.normal(0, 20), "inst_netbuy": rng.normal(0, 20),
                "program_netbuy": rng.normal(0, 10), "kospi_pct": 0.003, "kosdaq_pct": 0.001,
                "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
    return pd.DataFrame(rows)


def test_build_universe_training_panel_schema_and_scale() -> None:
    # Arrange
    ph = _price_history()
    screen = ScreenConfig(change_lower=0.02, change_upper=0.10, min_trade_value_100m=100.0, min_market_cap_100m=500.0)

    # Act
    x, targets, feature_cols, processed, prov = build_universe_training_panel(
        ph, screen, None, feature_set="close_morning61", start_date="2023-01-02", end_date="2023-05-31",
    )

    # Assert
    assert len(processed) > 0
    for col in ("trade_date", "stock_code", "target_return", "mechanical_gross", "chart_analysis"):
        assert col in processed.columns
    assert processed["chart_analysis"].isna().sum() == 0
    assert feature_cols and "mechanical_gross" not in feature_cols and "target_return" not in feature_cols
    assert not (set(DECISION_LABEL_COLUMNS) & set(feature_cols))
    assert_no_label_leakage(feature_cols)
    # percent scale: a 2%-10% screen has change_rate well above 1.0
    assert float(processed["change_rate"].median()) > 1.0
    assert int(prov["n_screened"]) >= len(processed)


import pandas as pd
import pytest

from src.ml.universe import ScreenConfig
from src.ml.universe_research import build_universe_training_panel


def test_build_universe_training_panel_requires_price_history() -> None:
    screen = ScreenConfig(change_lower=0.0, min_trade_value_100m=100.0, min_market_cap_100m=500.0)
    with pytest.raises(ValueError, match="price_history"):
        build_universe_training_panel(None, screen, None, start_date="2023-01-02", end_date="2023-06-30")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="price_history"):
        build_universe_training_panel(pd.DataFrame(), screen, None, start_date="2023-01-02", end_date="2023-06-30")


import numpy as np
import pandas as pd

from src.ml.dataset import build_ml_dataset
from src.ml.universe import ScreenConfig
from src.ml.universe_research import build_universe_training_panel
from tests.unit.ml.test_universe_research import _price_history  # reuse S1 fixture


def _trade_log(codes: list[str], n_days: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for d in pd.bdate_range("2023-01-03", periods=n_days):
        for j, code in enumerate(codes):
            e = rng.normal()
            rows.append({
                "매수날짜": d.strftime("%Y-%m-%d"), "종목코드": code,
                "(시가)": "10000", "(고가)": "10400", "(저가)": "9800", "(종가)": "10200",
                "(전일종가)": "10000", "(시가총액, 억)": "5000", "(거래대금, 억)": "300",
                "(등락률)": f"{11 + e:.2f}", "(선정 순위)": str(j + 1),
                "(기관_순매수)": "10", "(외국인_순매수)": "5", "(프로그램_순매수)": "2",
                "(체결강도)": "120", "(시장구분)": "KOSPI", "(총 종목 수)": str(len(codes)),
                "(평균 거래대금)": "250", "(kospi, %)": "0.3", "(kosdaq, %)": "0.1",
                "v_kospi": "18", "v_kosdaq": "22", "(거래량)": "100000",
                "(테마/섹터)": "반도체", "(차트분석)": "거래량 폭증",
                "(매수 가격)": "10200", "(매도 가격)": f"{10200*(1+0.01*e):.0f}", "(수익률, %)": f"{e:.2f}",
            })
    return pd.DataFrame(rows)


def test_universe_panel_feature_parity_with_trade_log() -> None:
    ph = _price_history()
    codes = sorted(ph["symbol"].unique().tolist())
    screen = ScreenConfig(change_lower=-1.0, change_upper=None, min_trade_value_100m=100.0, min_market_cap_100m=500.0)

    _xu, _tu, feat_universe, _pu, _prov = build_universe_training_panel(
        ph, screen, None, feature_set="close_morning61", start_date="2023-01-02", end_date="2023-05-31",
    )
    x_tl, _tt, cat_tl, _pt = build_ml_dataset(
        _trade_log(codes), None, feature_set="close_morning61", panel_mode="scenario_action",
        price_history_df=ph, scenario_source="auto",
    )
    feat_tl = [c for c in x_tl.columns if c not in cat_tl]

    assert sorted(feat_universe) == sorted(feat_tl)


import math

import pandas as pd

from src.ml.universe import ScreenConfig
from src.ml.universe_research import evaluate_universe_screen
from tests.unit.ml.test_universe_research import _price_history


def test_evaluate_universe_screen_empty_screen_degrades() -> None:
    ph = _price_history()
    # market_cap floor no symbol can meet -> 0 rows
    dead = ScreenConfig(change_lower=0.0, min_trade_value_100m=100.0, min_market_cap_100m=1e9)

    rec = evaluate_universe_screen(
        ph, dead, screen_name="dead", start_date="2023-01-02", end_date="2023-05-31",
        n_splits=4, cpcv_n_groups=5, cpcv_k_test=2,
    )

    assert rec.verdict == "insufficient_data"
    assert rec.n_rows == 0
    assert math.isnan(rec.ranked_top1_net_bp)




import numpy as np
import pandas as pd

from src.ml.universe import ScreenConfig
from src.ml.universe_research import (
    UniverseScreenRecord,
    _daily_top1_net_bp,
    evaluate_universe_screen,
    main,
    run_universe_screen_grid,
)
from tests.unit.ml.test_universe_research import _price_history


def test_to_row_roundtrips_record_fields() -> None:
    rec = UniverseScreenRecord(
        screen_name="liq", screen={"change_lower": -1.0}, n_rows=10, n_days=5, per_day=2.0,
        model_free_gross_bp=1.0, model_free_net_bp=-45.0, model_free_t_stat=0.3,
        ranked_rank_ic=0.02, ranked_top1_gross_bp=12.0, ranked_top1_net_bp=-34.0,
        cpcv_top1_path_win_rate=0.4, cpcv_ic_path_win_rate=0.5, cpcv_n_paths=4,
        breakeven_cost_bp=12.0, cost_ratio_bp=46.0, verdict="below_cost",
    )
    row = rec.to_row()
    assert row["screen_name"] == "liq" and row["verdict"] == "below_cost"
    assert row["cpcv_n_paths"] == 4 and row["ranked_top1_net_bp"] == -34.0
    assert set(row) == {f.name for f in __import__("dataclasses").fields(UniverseScreenRecord)}


def test_daily_top1_net_bp_argmax_and_cost() -> None:
    oof = pd.DataFrame({
        "pred": [0.1, 0.9, 0.5, 0.2],
        "mechanical_gross": [0.01, 0.05, -0.02, 0.03],
        "trade_date": pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03"]),
    })
    gross_bp, net_bp = _daily_top1_net_bp(oof, "trade_date", "mechanical_gross", 0.0046)
    # day1 pick=0.9 -> +0.05 ; day2 pick=0.5 -> -0.02 ; mean = 0.015
    assert gross_bp == pytest.approx(150.0)
    assert net_bp == pytest.approx((0.015 - 0.0046) * 1e4)

    empty = _daily_top1_net_bp(oof.iloc[:0], "trade_date", "mechanical_gross", 0.0046)
    assert np.isnan(empty[0]) and np.isnan(empty[1])


def test_evaluate_universe_screen_minimal_smoke() -> None:
    ph = _price_history(n_days=45, n_syms=10)
    screen = ScreenConfig(change_lower=-1.0, change_upper=None, min_trade_value_100m=100.0, min_market_cap_100m=500.0)

    rec = evaluate_universe_screen(
        ph, screen, screen_name="liq", start_date="2023-01-02", end_date="2023-04-30",
        n_splits=4, cpcv_n_groups=5, cpcv_k_test=2, model_params={"n_estimators": 20, "num_leaves": 7},
    )

    assert isinstance(rec, UniverseScreenRecord)
    assert rec.verdict in {"clears_cost", "below_cost"}
    assert rec.n_days >= 5 and rec.cpcv_n_paths >= 4
    assert rec.cost_ratio_bp == pytest.approx(46.0, abs=1.0)
    assert np.isfinite(rec.model_free_net_bp)


def test_run_universe_screen_grid_minimal_smoke() -> None:
    ph = _price_history(n_days=45, n_syms=10)
    screens = {
        "liq_wide": ScreenConfig(change_lower=-1.0, change_upper=None, min_trade_value_100m=100.0, min_market_cap_100m=500.0),
        "modest_up": ScreenConfig(change_lower=0.0, change_upper=0.15, min_trade_value_100m=100.0, min_market_cap_100m=500.0),
    }

    df = run_universe_screen_grid(
        ph, screens, start_date="2023-01-02", end_date="2023-04-30",
        n_splits=4, cpcv_n_groups=5, cpcv_k_test=2, model_params={"n_estimators": 20, "num_leaves": 7},
    )

    assert len(df) == 2 and list(df["screen_name"])
    net = df["ranked_top1_net_bp"].to_numpy(dtype=float)
    finite = net[np.isfinite(net)]
    assert (np.diff(finite) <= 1e-9).all()

    with pytest.raises(ValueError, match="non-empty"):
        run_universe_screen_grid(ph, {}, start_date="2023-01-02", end_date="2023-04-30")


def test_universe_research_main_smoke(tmp_path) -> None:
    ph = _price_history(n_days=60, n_syms=10)
    ph_path = tmp_path / "ph.parquet"
    ph.to_parquet(ph_path)
    out_path = tmp_path / "grid.parquet"

    main(["--price-history", str(ph_path), "--theme", str(tmp_path / "none.parquet"),
          "--out", str(out_path), "--cpcv-n-groups", "5", "--cpcv-k-test", "2"])

    df = pd.read_parquet(out_path)
    assert len(df) == 6 and "screen_name" in df.columns and "verdict" in df.columns

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--price-history", str(tmp_path / "missing.parquet"), "--out", str(out_path)])
