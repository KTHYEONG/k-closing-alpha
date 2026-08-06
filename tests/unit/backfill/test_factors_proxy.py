from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backfill.price import factors
from src.backfill.price.config import FetchConfig
from src.backfill.price.factors import compute_vkospi_proxy


def test_compute_proxy_handles_empty_and_kosdaq_output() -> None:
    assert list(compute_vkospi_proxy(pd.DataFrame(), output_col="v_kosdaq").columns) == ["date", "v_kosdaq"]
    dates = pd.date_range("2020-01-01", periods=25, freq="B")
    out = compute_vkospi_proxy(pd.DataFrame({"date": dates, "close": np.arange(100, 125)}), output_col="v_kosdaq")
    assert out.columns.tolist() == ["date", "v_kosdaq"]
    assert out["v_kosdaq"].notna().sum() == 5
    assert compute_vkospi_proxy(pd.DataFrame({"date": dates}), output_col="v_kospi").empty
    assert compute_vkospi_proxy(pd.DataFrame({"date": [None], "close": [None]}), output_col="v_kospi").empty


def test_merge_index_returns_includes_both_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_index(start, end, code, out_col, fetch_cfg):
        return pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), out_col: [1.0]})

    def fake_proxy(start, end, fetch_cfg, index_code, output_col):
        return pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), output_col: [2.0]})

    monkeypatch.setattr(factors, "_fetch_index_returns", fake_index)
    monkeypatch.setattr(factors, "_fetch_vkospi_proxy", fake_proxy)
    out = factors._merge_index_returns(
        pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "symbol": ["000001"]}), FetchConfig()
    )
    assert out.loc[0, "v_kospi"] == 2.0
    assert out.loc[0, "v_kosdaq"] == 2.0
