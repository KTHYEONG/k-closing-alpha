from __future__ import annotations

import pandas as pd
import requests
import pytest

from src.backfill.price import sources
from src.backfill.price.config import FetchConfig


def test_index_returns_falls_back_to_kis_after_pykrx_failure(monkeypatch) -> None:
    def fail_pykrx(*args, **kwargs):
        raise KeyError("지수명")

    monkeypatch.setattr(sources.stock, "get_index_ohlcv_by_date", fail_pykrx)
    monkeypatch.setattr(
        sources,
        "_fetch_kis_index_close",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "close": [100.0, 102.0],
            }
        ),
    )

    out = sources._fetch_index_returns(
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        "1001",
        "kospi_pct",
        FetchConfig(retries=1),
    )

    assert out["kospi_pct"].isna().iloc[0]
    assert out["kospi_pct"].iloc[1] == pytest.approx(0.02)


def test_kis_index_close_normalizes_rows(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "output2": [
                    {"stck_bsop_date": "20250103", "bstp_nmix_prpr": "102"},
                    {"stck_bsop_date": "20250102", "bstp_nmix_prpr": "100"},
                ]
            }

    class Client:
        base_url = "https://example.test"

        @staticmethod
        def _get_headers(_tr_id):
            return {}

    monkeypatch.setattr(sources, "_kis_sync_client", lambda: Client())
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())

    out = sources._fetch_kis_index_close(
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        "1001",
        FetchConfig(retries=1, request_sleep_sec=0),
    )

    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2025-01-02", "2025-01-03"]
    assert out["close"].tolist() == [100.0, 102.0]


def test_vkospi_proxy_uses_kis_index_fallback(monkeypatch) -> None:
    monkeypatch.setattr(sources.stock, "get_index_ohlcv_by_date", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("지수명")))
    monkeypatch.setattr(
        sources,
        "_fetch_kis_index_close",
        lambda *args, **kwargs: pd.DataFrame({"date": pd.date_range("2025-01-01", periods=21, freq="B"), "close": range(100, 121)}),
    )
    out = sources._fetch_vkospi_proxy(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01"), FetchConfig(retries=1, request_sleep_sec=0), index_code="1001", output_col="v_kospi")
    assert "v_kospi" in out.columns
    assert out["v_kospi"].notna().any()


def test_vkospi_proxy_returns_empty_when_all_sources_fail(monkeypatch) -> None:
    monkeypatch.setattr(sources.stock, "get_index_ohlcv_by_date", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(sources, "_fetch_kis_index_close", lambda *args, **kwargs: pd.DataFrame())
    out = sources._fetch_vkospi_proxy(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01"), FetchConfig(retries=1, request_sleep_sec=0), index_code="2001", output_col="v_kosdaq")
    assert out.columns.tolist() == ["date", "v_kosdaq"]
