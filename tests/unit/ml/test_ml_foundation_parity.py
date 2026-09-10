"""Parity of the src ML foundation against the frozen legacy fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _raw_trade_log(n_dates: int = 30, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=n_dates):
        for j in range(per_day):
            e = rng.normal()
            rows.append({
                "매수날짜": d.strftime("%Y-%m-%d"), "종목코드": f"{j:06d}",
                "(시가)": "10000", "(고가)": "10400", "(저가)": "9800", "(종가)": "10200", "(전일종가)": "10000",
                "(시가총액, 억)": "5000", "(거래대금, 억)": "300", "(등락률)": f"{2 + e:.2f}",
                "(선정 순위)": str(j + 1), "(기관_순매수)": f"{e*100:.0f}", "(외국인_순매수)": f"{e*80:.0f}",
                "(프로그램_순매수)": f"{e*50:.0f}", "(체결강도)": "120", "(시장구분)": "KOSPI",
                "(총 종목 수)": str(per_day), "(평균 거래대금)": "250", "(kospi, %)": "0.3", "(kosdaq, %)": "0.1",
                "v_kospi": "18", "v_kosdaq": "20", "(거래량)": "100000", "(테마/섹터)": "반도체",
                "(차트분석)": "거래량 폭증", "(매수 가격)": "10200",
                "(매도 가격)": f"{10200*(1+0.01*e):.0f}", "(수익률, %)": f"{e:.2f}",
            })
    return pd.DataFrame(rows)


def test_purged_cv_matches_frozen_legacy_split_indices() -> None:
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from src.ml.purged_cv import PurgedGroupTimeSeriesSplit

    # Given: the split indices frozen from legacy before that tree was retired.
    fixture = json.loads(
        Path("tests/fixtures/legacy_parity.json").read_text(encoding="utf-8")
    )
    expected = fixture["purged_cv_splits"]
    groups = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=40), 5))
    x = pd.DataFrame({"f": np.arange(len(groups))})

    # When
    got = list(PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=1).split(x, groups=groups))

    # Then: fold count and every train/test index array match the frozen record.
    assert len(got) == len(expected) == 4
    for (train_idx, test_idx), exp in zip(got, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(train_idx), np.asarray(exp["train"]))
        np.testing.assert_array_equal(np.asarray(test_idx), np.asarray(exp["test"]))


def test_build_ml_dataset_matches_frozen_legacy_parity() -> None:
    import json
    from pathlib import Path

    import numpy as np

    from src.ml.dataset import build_ml_dataset

    # Given: the column/categorical/target record frozen from legacy.
    fixture = json.loads(
        Path("tests/fixtures/legacy_parity.json").read_text(encoding="utf-8")
    )
    expected = fixture["build_ml_dataset"]

    # When: the same seeded synthetic trade log the fixture was generated from.
    gx, _gt, gcat, gproc = build_ml_dataset(
        _raw_trade_log(), None, feature_set="close_morning61", panel_mode="scenario_action"
    )

    # Then: the three original parity assertions, at the original tolerance.
    assert sorted(map(str, gx.columns)) == expected["columns_sorted"]
    assert sorted(map(str, gcat)) == expected["cat_features_sorted"]
    np.testing.assert_allclose(
        gproc.sort_index()["target_return"].to_numpy(dtype=float),
        np.asarray(expected["target_return"], dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )


def test_legacy_tree_is_absent_and_unimportable() -> None:
    import importlib
    import pathlib

    import pytest

    # Given / Then: the tree itself is gone.
    assert not pathlib.Path("legacy").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("legacy")

    # And: the guard now covers all of src/, not just src/ml.
    offenders = [
        str(path)
        for path in pathlib.Path("src").rglob("*.py")
        if "import legacy" in path.read_text(encoding="utf-8")
        or "from legacy" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"src must not import legacy: {offenders}"
