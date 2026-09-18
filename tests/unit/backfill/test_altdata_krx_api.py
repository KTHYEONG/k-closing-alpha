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


def test_fetch_krx_openapi_day_strict_raises_on_unauthorized(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.backfill.altdata import krx_api
    from src.backfill.altdata.config import AltDataFetchConfig

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2026-09-01"), end=pd.Timestamp("2026-09-30"),
        out_dir=tmp_path, krx_api_key="dummy-key",
    )

    class _Resp:
        def __init__(self, status: int) -> None:
            self.status_code = status

        def json(self) -> dict:
            return {}

    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api, "retry_call", lambda fn, _cfg, label="": fn())

    # Given: 미구독 엔드포인트(401)
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _Resp(401))

    # When / Then: 폴백 없는 설계이므로 조용한 빈 프레임이 아니라 예외
    with pytest.raises(RuntimeError, match="401"):
        krx_api.fetch_krx_openapi_day_strict("/svc/apis/sto/knx_bydd_trd", "20260909", cfg)

    # And: 기타 비200도 동일하게 fail-closed
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _Resp(500))
    with pytest.raises(RuntimeError, match="500"):
        krx_api.fetch_krx_openapi_day_strict("/svc/apis/sto/stk_bydd_trd", "20260909", cfg)


def test_fetch_krx_openapi_day_strict_requires_key_and_allows_empty_holiday(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.backfill.altdata import krx_api
    from src.backfill.altdata.config import AltDataFetchConfig

    # Given: 키 미설정은 조용히 넘어가지 않는다
    with pytest.raises(ValueError, match="key"):
        krx_api.fetch_krx_openapi_day_strict(
            krx_api.KRX_ENDPOINT_STK_DAILY,
            "20260909",
            AltDataFetchConfig(
                start=pd.Timestamp("2026-09-01"), end=pd.Timestamp("2026-09-30"),
                out_dir=tmp_path, krx_api_key="",
            ),
        )

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2026-09-01"), end=pd.Timestamp("2026-09-30"),
        out_dir=tmp_path, krx_api_key="dummy-key",
    )

    class _Resp:
        status_code = 200

        def json(self) -> dict:
            return {"OutBlock_1": []}

    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api, "retry_call", lambda fn, _cfg, label="": fn())
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _Resp())

    # When: 휴장일 응답(200 + 0행)
    out = krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20260101", cfg)

    # Then: 장애가 아니라 정상 빈 프레임
    assert out.empty


def test_unused_pykrx_collectors_are_deleted_and_unwired(tmp_path) -> None:
    import importlib
    from pathlib import Path

    import pandas as pd
    import pytest

    # Then: 패널이 한 번도 생성된 적 없는 두 수집기는 삭제된다
    for rel, mod in {
        "src/backfill/altdata/fundamental.py": "src.backfill.altdata.fundamental",
        "src/backfill/altdata/investor_detail.py": "src.backfill.altdata.investor_detail",
    }.items():
        assert not Path(rel).exists(), f"{rel} should be deleted"
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)

    # And: 허용 패널 목록과 기본 sources에서도 제거된다
    from src.backfill.altdata.config import _ALTDATA_PANELS, AltDataFetchConfig

    assert "fundamental" not in _ALTDATA_PANELS
    assert "investor_detail" not in _ALTDATA_PANELS
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2026-09-01"), end=pd.Timestamp("2026-09-30"), out_dir=tmp_path
    )
    assert "fundamental" not in cfg.sources
    assert "investor_detail" not in cfg.sources

    # And: 살아있는 패널은 그대로 유지된다
    for kept in ("shorting", "derivatives_basis", "credit_balance", "program_trade_daily", "disclosure"):
        assert kept in _ALTDATA_PANELS

    # And: 러너가 삭제된 수집기를 더 이상 재수출하지 않는다
    runner = importlib.import_module("src.backfill.altdata.runner")
    assert not hasattr(runner, "collect_fundamental")
    assert not hasattr(runner, "collect_investor_detail")


def test_fetch_krx_openapi_day_strict_raises_when_retries_exhausted(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.backfill.altdata import krx_api
    from src.backfill.altdata.config import AltDataFetchConfig

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2026-09-01"), end=pd.Timestamp("2026-09-30"),
        out_dir=tmp_path, krx_api_key="dummy-key",
    )
    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api, "retry_call", lambda fn, _cfg, label="": None)

    # When: 재시도가 전부 실패하면(None) 조용한 None 반환이 아니라 예외
    with pytest.raises(RuntimeError, match="retries"):
        krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20260909", cfg)


def test_collect_derivatives_basis_returns_empty_without_pykrx_fallback(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata import derivatives

    # Given: KRX 주경로가 빈 프레임을 반환 (기존 _cfg() 헬퍼 재사용, 이 파일 상단에 이미 정의됨)
    monkeypatch.setattr(derivatives, "fetch_krx_openapi_day", lambda *a, **k: pd.DataFrame())

    # When
    out = derivatives.collect_derivatives_basis(_cfg(), [pd.Timestamp("2025-06-09")])

    # Then: pykrx 폴백 없이 빈 프레임 그대로(_collect_via_pykrx가 더 이상 존재하지 않는다)
    assert out.empty
    assert not hasattr(derivatives, "_collect_via_pykrx")
    assert not hasattr(derivatives, "stock")


def _strict_cfg(**kw: object):
    base: dict[str, object] = {
        "start": pd.Timestamp("2026-09-01"),
        "end": pd.Timestamp("2026-09-30"),
        "out_dir": Path("x"),
        "retries": 2,
        "retry_sleep_sec": 0.0,
        "krx_api_key": "dummy-key",
    }
    base.update(kw)
    return AltDataFetchConfig(**base)  # type: ignore[arg-type]


class _StrictResp:
    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _observe_sink() -> tuple[list, object]:
    seen: list = []

    def _on_page(payload: object, meta: object, started: object, received: object, page: int, attempt: int) -> None:
        seen.append((payload, meta, started, received, page, attempt))

    return seen, _on_page


def test_fetch_strict_observer_receives_full_response(monkeypatch) -> None:
    from src.backfill.altdata import krx_api

    payload = {
        "OutBlock_1": [{"BAS_DD": "20250602", "TDD_CLSPRC": "359.14"}],
        "extra_top": {"note": "kept"},
        "respCd": "0000",
    }
    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _StrictResp(200, payload))
    seen, on_page = _observe_sink()
    out = krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20250602", _strict_cfg(), on_page=on_page)  # type: ignore[arg-type]
    assert len(seen) == 1
    observed, meta, started, received, page, attempt = seen[0]
    assert observed == payload
    assert observed["extra_top"] == {"note": "kept"}
    assert page == 0 and attempt == 0
    assert getattr(started, "tzinfo", None) is not None and getattr(received, "tzinfo", None) is not None
    assert received >= started
    assert set(meta) <= {"endpoint", "basDd"}
    assert "AUTH_KEY" not in str(meta) and "dummy-key" not in str(meta)
    assert len(out) == 1 and out.iloc[0]["TDD_CLSPRC"] == "359.14"


def test_fetch_strict_observer_precedes_normalization(monkeypatch) -> None:
    from src.backfill.altdata import krx_api

    payload = {"OutBlock_1": [{"BAS_DD": "20250602", "TDD_CLSPRC": "359.14", "ACC_TRDVOL": "300000"}]}
    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _StrictResp(200, payload))
    seen, on_page = _observe_sink()
    out = krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20250602", _strict_cfg(), on_page=on_page)  # type: ignore[arg-type]
    assert isinstance(seen[0][0]["OutBlock_1"][0]["TDD_CLSPRC"], str)
    assert seen[0][0]["OutBlock_1"][0]["TDD_CLSPRC"] == "359.14"
    assert out.iloc[0]["TDD_CLSPRC"] == "359.14"


def test_fetch_strict_receipt_is_not_publication(monkeypatch) -> None:
    from src.backfill.altdata import krx_api

    payload = {"OutBlock_1": [], "basDd": "20260101"}
    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _StrictResp(200, payload))
    seen, on_page = _observe_sink()
    out = krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20260101", _strict_cfg(), on_page=on_page)  # type: ignore[arg-type]
    assert out.empty
    assert len(seen) == 1
    assert set(seen[0][1]) == {"endpoint", "basDd"}
    assert not any("publish" in key for key in seen[0][1])


def test_fetch_strict_empty_source_remains_ambiguous(monkeypatch) -> None:
    from src.backfill.altdata import krx_api

    payload = {"OutBlock_1": []}
    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _StrictResp(200, payload))
    seen, on_page = _observe_sink()
    out = krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20260101", _strict_cfg(), on_page=on_page)  # type: ignore[arg-type]
    assert out.empty
    assert seen[0][0]["OutBlock_1"] == []


def test_fetch_strict_retry_attempts_stay_distinct(monkeypatch) -> None:
    from src.backfill.altdata import krx_api

    payload = {"OutBlock_1": [{"BAS_DD": "20250602", "TDD_CLSPRC": "359.14"}]}
    calls = {"n": 0}

    def _flaky(*a: object, **k: object) -> _StrictResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transport down")
        return _StrictResp(200, payload)

    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", _flaky)
    seen, on_page = _observe_sink()
    out = krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20250602", _strict_cfg(), on_page=on_page)  # type: ignore[arg-type]
    assert len(out) == 1
    assert [attempt for _, _, _, _, _, attempt in seen] == [0, 1]
    assert seen[0][0] is None
    assert seen[0][1]["error_type"] == "ConnectionError"
    assert seen[1][0] == payload


def test_fetch_strict_capture_failure_propagates(monkeypatch) -> None:
    import pytest

    from src.backfill.altdata import krx_api
    from src.data.capture_contracts import RawCaptureError

    payload = {"OutBlock_1": [{"BAS_DD": "20250602"}]}
    calls = {"n": 0}

    def _counted(*a: object, **k: object) -> _StrictResp:
        calls["n"] += 1
        return _StrictResp(200, payload)

    def _failing(*a: object, **k: object) -> None:
        raise RawCaptureError("store full")

    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", _counted)
    with pytest.raises(RawCaptureError):
        krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20250602", _strict_cfg(retries=3), on_page=_failing)  # type: ignore[arg-type]
    assert calls["n"] == 1


def test_fetch_strict_observer_error_wrapped_without_credentials(monkeypatch) -> None:
    import pytest

    from src.backfill.altdata import krx_api
    from src.data.capture_contracts import RawCaptureError

    payload = {"OutBlock_1": [{"BAS_DD": "20250602"}]}

    def _leaky(*a: object, **k: object) -> None:
        raise ValueError("store blew up with SECRET-XYZ")

    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _StrictResp(200, payload))
    with pytest.raises(RawCaptureError) as exc_info:
        krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20250602", _strict_cfg(), on_page=_leaky)  # type: ignore[arg-type]
    assert "SECRET-XYZ" not in str(exc_info.value)


def test_fetch_strict_malformed_response_fails_explicit(monkeypatch, caplog) -> None:
    import logging

    import pytest

    from src.backfill.altdata import krx_api

    monkeypatch.setattr(krx_api, "wait_for_krx_slot", lambda _cfg: None)
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _StrictResp(200, ValueError("not json")))
    with caplog.at_level(logging.WARNING, logger="src.backfill.altdata.ratelimit"), pytest.raises(RuntimeError, match="retries failed"):
        krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20250602", _strict_cfg(retries=1), on_page=None)
    assert any("malformed" in record.message for record in caplog.records)
    caplog.clear()
    monkeypatch.setattr(krx_api.requests, "get", lambda *a, **k: _StrictResp(200, ["not", "a", "dict"]))
    with caplog.at_level(logging.WARNING, logger="src.backfill.altdata.ratelimit"), pytest.raises(RuntimeError, match="retries failed"):
        krx_api.fetch_krx_openapi_day_strict(krx_api.KRX_ENDPOINT_STK_DAILY, "20250602", _strict_cfg(retries=1), on_page=None)
    assert any("malformed" in record.message for record in caplog.records)
