import numpy as np
import pandas as pd

from legacy.ml_research.training.purged_cv import PurgedGroupTimeSeriesSplit as LegacySplit
from src.ml.purged_cv import PurgedGroupTimeSeriesSplit


def test_purged_cv_matches_legacy_split_indices() -> None:
    groups = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=40), 5))
    x = pd.DataFrame({"f": np.arange(len(groups))})
    got = list(PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=1).split(x, groups=groups))
    exp = list(LegacySplit(n_splits=4, purge_gap=1).split(x, groups=groups))
    assert len(got) == len(exp)
    for (gtr, gva), (etr, eva) in zip(got, exp, strict=True):
        np.testing.assert_array_equal(gtr, etr)
        np.testing.assert_array_equal(gva, eva)

import numpy as np
import pandas as pd

from legacy.ml_research.features.dataset import build_ml_dataset as legacy_build
from src.ml.dataset import build_ml_dataset


def _raw_trade_log(n_dates: int = 30, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=n_dates):
        for j in range(per_day):
            e = rng.normal()
            rows.append({
                "\ub9e4\uc218\ub0a0\uc9dc": d.strftime("%Y-%m-%d"), "\uc885\ubaa9\ucf54\ub4dc": f"{j:06d}",
                "(\uc2dc\uac00)": "10000", "(\uace0\uac00)": "10400", "(\uc800\uac00)": "9800", "(\uc885\uac00)": "10200", "(\uc804\uc77c\uc885\uac00)": "10000",
                "(\uc2dc\uac00\ucd1d\uc561, \uc5b5)": "5000", "(\uac70\ub798\ub300\uae08, \uc5b5)": "300", "(\ub4f1\ub77d\ub960)": f"{2 + e:.2f}",
                "(\uc120\uc815 \uc21c\uc704)": str(j + 1), "(\uae30\uad00_\uc21c\ub9e4\uc218)": f"{e*100:.0f}", "(\uc678\uad6d\uc778_\uc21c\ub9e4\uc218)": f"{e*80:.0f}",
                "(\ud504\ub85c\uadf8\ub7a8_\uc21c\ub9e4\uc218)": f"{e*50:.0f}", "(\uccb4\uacb0\uac15\ub3c4)": "120", "(\uc2dc\uc7a5\uad6c\ubd84)": "KOSPI",
                "(\ucd1d \uc885\ubaa9 \uc218)": str(per_day), "(\ud3c9\uade0 \uac70\ub798\ub300\uae08)": "250", "(kospi, %)": "0.3", "(kosdaq, %)": "0.1",
                "v_kospi": "18", "v_kosdaq": "20", "(\uac70\ub798\ub7c9)": "100000", "(\ud14c\ub9c8/\uc139\ud130)": "\ubc18\ub3c4\uccb4",
                "(\ucc28\ud2b8\ubd84\uc11d)": "\uac70\ub798\ub7c9 \ud3ed\uc99d", "(\ub9e4\uc218 \uac00\uaca9)": "10200",
                "(\ub9e4\ub3c4 \uac00\uaca9)": f"{10200*(1+0.01*e):.0f}", "(\uc218\uc775\ub960, %)": f"{e:.2f}",
            })
    return pd.DataFrame(rows)


def test_build_ml_dataset_matches_legacy_champion_columns() -> None:
    raw = _raw_trade_log()
    gx, _gt, gcat, gproc = build_ml_dataset(raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action")
    lx, _lt, lcat, lproc = legacy_build(raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action")
    assert sorted(gx.columns) == sorted(lx.columns)
    assert sorted(gcat) == sorted(lcat)
    np.testing.assert_allclose(
        gproc.sort_index()["target_return"].to_numpy(), lproc.sort_index()["target_return"].to_numpy(), rtol=1e-9, atol=1e-12
    )

import numpy as np
import pandas as pd

from legacy.ml_research.evaluation.single_stock_policy import (
    default_policy_candidates as legacy_candidates,
    evaluate_single_stock_policy_oof as legacy_eval,
)
from src.ml.policy_eval import default_policy_candidates, evaluate_single_stock_policy_oof


def _oof(n_dates: int = 320, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    dates = np.repeat(pd.bdate_range("2022-01-03", periods=n_dates), per_day)
    m = len(dates)
    pred = rng.normal(size=m)
    return pd.DataFrame({
        "trade_date": dates,
        "stock_code": [f"{i % 40:06d}" for i in range(m)],
        "chart_analysis": "volume_surge",
        "market_type": "KOSPI",
        "rank_score": pred,
        "target_return": 0.01 * pred + rng.normal(scale=0.02, size=m),
    })


def test_evaluate_single_stock_policy_oof_matches_legacy_selection() -> None:
    oof = _oof()
    cutoff = str(oof["trade_date"].max())
    got = evaluate_single_stock_policy_oof(
        oof.copy(), "target_return", "trade_date", "stock_code",
        default_policy_candidates(cutoff), 252, scenario_col="chart_analysis", score_col="rank_score",
    )
    exp = legacy_eval(
        oof.copy(), "target_return", "trade_date", "stock_code",
        legacy_candidates(cutoff), 252, scenario_col="chart_analysis", score_col="rank_score",
    )
    assert got.selected_policy.candidate == exp.selected_policy.candidate
    assert np.isclose(got.metrics["scheduled_mean_return"], exp.metrics["scheduled_mean_return"], atol=1e-9)

import pathlib


def test_src_ml_has_no_legacy_imports() -> None:
    offenders = []
    for path in pathlib.Path("src/ml").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import legacy" in text or "from legacy" in text:
            offenders.append(str(path))
    assert offenders == [], f"src/ml must not import legacy: {offenders}"
