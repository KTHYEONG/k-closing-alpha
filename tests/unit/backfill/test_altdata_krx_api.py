from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backfill.altdata import derivatives, krx_api
from src.backfill.altdata.config import AltDataFetchConfig


def _cfg(**kw: object) -> AltDataFetchConfig:
    base: dict[str, object] = {
        "start": pd.Timestamp("2025-06-01"),
        "end": pd.Timestamp("2025-06-10"),
        "out_dir": Path("x"),
        "retries": 1,
        "retry_sleep_sec": 0.0,
        "krx_api_key": "dummy",
    }
    base.update(kw)
    return AltDataFetchConfig(**base)  # type: ignore[arg-type]


class _Resp:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_fetch_krx_openapi_day_returns_outblock_rows(monkeypatch) -> None:
    payload = {"OutBlock_1": [{"BAS_DD": "20250602", "TDD_CLSPRC": "359.14"}]}
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _Resp(200, payload))
    out = krx_api.fetch_krx_openapi_day("/svc/apis/drv/fut_bydd_trd", "20250602", _cfg())
    assert list(out.columns) == ["BAS_DD", "TDD_CLSPRC"]
    assert out.iloc[0]["TDD_CLSPRC"] == "359.14"


def test_fetch_krx_openapi_day_unavailable_is_fail_soft(monkeypatch) -> None:
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _Resp(401, {"respMsg": "Unauthorized API Call"}))
    out = krx_api.fetch_krx_openapi_day("/svc/apis/sto/nope", "20250602", _cfg())
    assert out.empty
    # no key -> also empty, no request attempted
    assert krx_api.fetch_krx_openapi_day("/svc/apis/drv/fut_bydd_trd", "20250602", _cfg(krx_api_key="")).empty


def test_collect_derivatives_basis_uses_krx_front_month(monkeypatch) -> None:
    raw = pd.DataFrame(
        [
            {"PROD_NM": "코스피200 선물", "MKT_NM": "정규", "ISU_NM": "코스피200 F 202506", "SPOT_PRC": "359.69", "TDD_CLSPRC": "359.15", "ACC_TRDVOL": "300000", "ACC_OPNINT_QTY": "120000"},
            {"PROD_NM": "코스피200 선물", "MKT_NM": "정규", "ISU_NM": "코스피200 F 202509", "SPOT_PRC": "359.69", "TDD_CLSPRC": "360.10", "ACC_TRDVOL": "50", "ACC_OPNINT_QTY": "160"},
            {"PROD_NM": "미니코스피200 선물", "MKT_NM": "정규", "ISU_NM": "미니코스피 F 202506", "SPOT_PRC": "359.69", "TDD_CLSPRC": "359.14", "ACC_TRDVOL": "100133", "ACC_OPNINT_QTY": "49397"},
        ]
    )
    monkeypatch.setattr(derivatives, "fetch_krx_openapi_day", lambda *a, **k: raw)
    out = derivatives.collect_derivatives_basis(_cfg(), [pd.Timestamp("2025-06-02")])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["k200_future_close"] == 359.15
    assert row["kospi200_close"] == 359.69
    assert abs(row["basis"] - (359.15 - 359.69)) < 1e-9
    assert row["future_volume"] == 300000.0
