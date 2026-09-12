from __future__ import annotations


def test_is_krx_trading_day_distinguishes_holiday_from_outage(monkeypatch) -> None:
    import pandas as pd
    import pytest

    from src.data import trading_calendar

    # Given: 휴장일 응답(0행)
    monkeypatch.setattr(
        trading_calendar, "fetch_krx_openapi_day_strict", lambda *a, **k: pd.DataFrame()
    )
    trading_calendar._TRADING_DAY_CACHE.clear()
    assert trading_calendar.is_krx_trading_day("2026-01-01") is False

    # And: 영업일 응답(행 존재)
    monkeypatch.setattr(
        trading_calendar,
        "fetch_krx_openapi_day_strict",
        lambda *a, **k: pd.DataFrame([{"IDX_NM": "코스피", "CLSPRC_IDX": "3000"}]),
    )
    trading_calendar._TRADING_DAY_CACHE.clear()
    assert trading_calendar.is_krx_trading_day(pd.Timestamp("2026-09-09")) is True

    # And: 장애는 '휴장일'로 삼켜지지 않는다
    def _boom(*a, **k):
        raise RuntimeError("krx http_status=500")

    monkeypatch.setattr(trading_calendar, "fetch_krx_openapi_day_strict", _boom)
    trading_calendar._TRADING_DAY_CACHE.clear()
    with pytest.raises(RuntimeError, match="500"):
        trading_calendar.is_krx_trading_day("2026-09-10")


def test_is_krx_trading_day_caches_repeat_lookups(monkeypatch) -> None:
    import pandas as pd

    from src.data import trading_calendar

    calls = {"n": 0}

    def _counting(*a, **k) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame([{"IDX_NM": "코스피"}])

    monkeypatch.setattr(trading_calendar, "fetch_krx_openapi_day_strict", _counting)
    trading_calendar._TRADING_DAY_CACHE.clear()

    # When: 같은 날짜를 세 번 묻는다
    assert trading_calendar.is_krx_trading_day("2026-09-09") is True
    assert trading_calendar.is_krx_trading_day("2026-09-09") is True
    assert trading_calendar.is_krx_trading_day(pd.Timestamp("2026-09-09")) is True

    # Then: 실제 호출은 1회
    assert calls["n"] == 1


def test_default_cfg_carries_api_key_from_settings(monkeypatch) -> None:
    """기본 cfg가 키 없이 만들어지면 strict 페처가 ValueError로 죽는다(부팅 감사 전체 실패).

    페처를 목킹하면 이 배선 결함이 숨으므로, 실제로 전달된 cfg를 캡처해 검증한다.
    실 .env 자격증명이 없는 환경(CI 등)에서도 배선 자체를 검증할 수 있도록
    settings.KRX_OPENAPI_KEY 를 가짜 비공백 값으로 주입한다 -- 이 테스트의
    목적은 실제 키의 유효성이 아니라 '주입 경로'이므로 대체 가능하다.
    """
    import pandas as pd

    from src.data import trading_calendar

    monkeypatch.setattr(trading_calendar.settings, "KRX_OPENAPI_KEY", "test-krx-key", raising=False)

    captured: dict[str, object] = {}

    def _capture(_endpoint: str, _date: str, cfg) -> pd.DataFrame:
        captured["cfg"] = cfg
        return pd.DataFrame([{"IDX_NM": "코스피"}])

    monkeypatch.setattr(trading_calendar, "fetch_krx_openapi_day_strict", _capture)
    trading_calendar._TRADING_DAY_CACHE.clear()

    # When: cfg를 명시하지 않은 기본 호출(= daily_audit이 쓰는 경로)
    assert trading_calendar.is_krx_trading_day("2026-09-09") is True

    # Then: 키가 settings에서 주입돼 있어야 한다(빈 문자열이면 실경로에서 ValueError)
    cfg = captured["cfg"]
    assert cfg.krx_api_key == trading_calendar.settings.KRX_OPENAPI_KEY
    assert str(cfg.krx_api_key).strip() != ""


def test_is_kis_trading_day_true_only_when_requested_date_is_returned() -> None:
    import asyncio

    import pytest

    from src.data.trading_calendar import is_kis_trading_day

    class _Client:
        def __init__(self, res):
            self._res = res
            self.calls = []

        async def get_market_index_history(self, _session, market_code, start, end, *a, **k):
            self.calls.append((market_code, start, end))
            return self._res

    # Given: 거래일 (요청일과 동일한 stck_bsop_date 1행)
    ok = _Client({"rt_cd": "0", "output2": [{"stck_bsop_date": "20260910", "bstp_nmix_prpr": "7033.92"}]})
    assert asyncio.run(is_kis_trading_day(ok, object(), "2026-09-10")) is True
    assert ok.calls == [("0001", "20260910", "20260910")]

    # And: 휴장일/장전 (0행)
    empty = _Client({"rt_cd": "0", "output2": []})
    assert asyncio.run(is_kis_trading_day(empty, object(), "2026-09-05")) is False

    # And: 다른 날짜만 돌아오면 거래일로 인정하지 않는다
    mismatch = _Client({"rt_cd": "0", "output2": [{"stck_bsop_date": "20260909"}]})
    assert asyncio.run(is_kis_trading_day(mismatch, object(), "2026-09-10")) is False

    # And: 장애(rt_cd != "0")는 휴장으로 삼키지 않고 전파
    outage = _Client({"rt_cd": "9", "msg1": "network"})
    with pytest.raises(RuntimeError):
        asyncio.run(is_kis_trading_day(outage, object(), "2026-09-10"))
