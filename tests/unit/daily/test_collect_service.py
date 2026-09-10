"""일일 수집(collect) 서비스 단위 테스트: 스칼라 파싱·검증·시나리오 선택·집계."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.daily import collect


def test_safe_float_converts_values() -> None:
    assert collect.safe_float(None) == 0.0
    assert collect.safe_float("1,234.5") == 1234.5
    assert collect.safe_float("abc") == 0.0
    assert collect.safe_float(7) == 7.0


def test_parse_market_index_rate_returns_zero_on_missing() -> None:
    assert collect.parse_market_index_rate(None) is None
    assert collect.parse_market_index_rate({"rt_cd": "1"}) is None
    assert collect.parse_market_index_rate({"rt_cd": "0", "output1": None}) is None


def test_parse_market_index_rate_uses_rate_and_fallback() -> None:
    assert (
        collect.parse_market_index_rate(
            {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "1.25"}}
        )
        == 1.25
    )
    fallback = collect.parse_market_index_rate(
        {"rt_cd": "0", "output1": {"bstp_nmix_prpr": "100", "bstp_nmix_prdy_vrss": "2"}}
    )
    assert fallback == pytest.approx(2.04)


def test_validate_hts_id_raises_on_placeholder() -> None:
    with (
        patch.object(collect, "HTS_ID", "여기에 HTS ID를 입력"),
        pytest.raises(RuntimeError),
    ):
        collect._validate_hts_id()
    with patch.object(collect, "HTS_ID", "real-hts"):
        collect._validate_hts_id()  # should not raise


def _fake_client() -> SimpleNamespace:
    return SimpleNamespace(
        get_current_price=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "10500",
                    "stck_oprc": "10000",
                    "stck_hgpr": "11000",
                    "stck_lwpr": "9000",
                    "acml_vol": "100000",
                    "prdy_ctrt": "5.00",
                    "lstn_stcn": "1000000",
                    "rprs_mrkt_kor_name": "KOSPI",
                    "hts_avls": "100000",
                    "acml_tr_pbmn": "12000000000",
                },
            }
        ),
        get_trade_strength=AsyncMock(
            return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]}
        ),
        get_investor_trend_estimate=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output2": [{"frgn_fake_ntby_qty": "10000", "orgn_fake_ntby_qty": "5000"}],
            }
        ),
        get_program_net_buy=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output": [{"whol_smtn_ntby_tr_pbmn": "500000000"}],
            }
        ),
    )


async def _run_fetch_single_stock(client, **scenario_sets) -> tuple[dict, list[str], list[dict]]:
    sem = asyncio.Semaphore(2)
    stock = {"code": "005930", "name": "삼성전자", "price": "10000", "chgrate": "1.0"}
    with patch(
        "src.api.kis.indicators.calculate_all_moving_averages",
        new=AsyncMock(
            return_value=(
                {5: 10000, 10: 10000, 20: 10000},
                (10000.0, True, 300),
                (10000.0, True),
                (10000.0, True),
            )
        ),
    ):
        return await collect.fetch_single_stock(
            0, stock, 1, sem, client, None, **scenario_sets
        )








def test_collect_main_wires_kiwoom_scan_and_trading_day_gate(monkeypatch) -> None:
    import asyncio

    from src.daily import collect

    seen = {"gate": 0, "kiwoom_built": 0, "scan_kwargs": None, "force": None}

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            self.token = None

        async def ensure_token(self, _session, force_refresh=False):
            self.token = "T"
            return "T"

        async def get_market_index_rate(self, _session, _code):
            return {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "0.5"}}

    sentinel = object()

    def _build_kiwoom():
        seen["kiwoom_built"] += 1
        return sentinel

    async def _gate(_client, _session, _date, *, force=False):
        seen["gate"] += 1
        seen["force"] = force

    async def _resolve(_client, _session, *, kiwoom_client=None):
        seen["scan_kwargs"] = kiwoom_client
        return []

    monkeypatch.setattr(collect, "_validate_hts_id", lambda: None)
    monkeypatch.setattr(collect, "_validate_decision_window", lambda *_a, **_k: None)
    monkeypatch.setattr(collect, "KisApiClient", _FakeClient)
    monkeypatch.setattr(collect.aiohttp, "ClientSession", lambda *_a, **_kw: _FakeSession())
    monkeypatch.setattr(collect, "build_kiwoom_scan_client", _build_kiwoom)
    monkeypatch.setattr(collect, "_validate_trading_day", _gate)
    monkeypatch.setattr(collect, "resolve_daily_candidates", _resolve)

    # When: 후보가 비어 조기 반환하는 최단 경로로 main 을 구동
    asyncio.run(collect.main())

    # Then: 게이트가 스캔보다 먼저 1회 수행되고, Kiwoom 클라이언트가 스캔으로 전달된다
    assert seen["gate"] == 1
    assert seen["force"] is False
    assert seen["kiwoom_built"] == 1
    assert seen["scan_kwargs"] is sentinel



def test_build_kiwoom_scan_client_returns_none_without_credentials(monkeypatch) -> None:
    from types import SimpleNamespace

    import src.api.kiwoom.client as kiwoom_module
    from src.daily import collect

    monkeypatch.setattr(kiwoom_module, 'KiwoomApiClient', lambda *a, **k: SimpleNamespace(app_key=''))
    assert collect.build_kiwoom_scan_client() is None
