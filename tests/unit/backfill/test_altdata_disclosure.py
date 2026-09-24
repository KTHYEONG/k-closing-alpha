from pathlib import Path

import pandas as pd

from src.backfill.altdata import disclosure
from src.backfill.altdata.config import AltDataFetchConfig


def test_aggregate_rows_categorizes_and_flags_material() -> None:
    rows = [
        {"stock_code": "005930", "report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20240102"},
        {"stock_code": "005930", "report_nm": "전환사채권발행결정", "rcept_dt": "20240102"},
        {"stock_code": "000660", "report_nm": "기업설명회(IR)개최", "rcept_dt": "20240102"},
        {"stock_code": "", "report_nm": "기타경영사항", "rcept_dt": "20240102"},
    ]
    out = disclosure._aggregate_rows(rows)
    row = out[out["symbol"] == "005930"].iloc[0]
    assert row["n_supply_contract"] == 1
    assert row["n_cb_bw"] == 1
    assert row["n_total"] == 2
    assert bool(row["has_material"]) is True
    assert bool(out[out["symbol"] == "000660"].iloc[0]["has_material"]) is False
    assert (out["symbol"] == "").sum() == 0


def test_collect_disclosures_uses_pblntf_ty_and_flushes_per_window(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def _fake_window(cfg, pblntf_ty, start_ymd, end_ymd, corp_to_stock, *, on_page=None, page_offset=0):  # noqa: ANN001, ANN202
        calls.append((pblntf_ty, start_ymd, end_ymd))
        return [{"stock_code": "005930", "report_nm": "유상증자결정", "rcept_dt": f"{start_ymd}"}]

    monkeypatch.setattr(disclosure, "_fetch_disclosure_window", _fake_window)
    flushed: list[pd.DataFrame] = []
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-12-31"),
        out_dir=Path("x"), dart_api_key="k", retries=1, retry_sleep_sec=0.0,
    )
    ret = disclosure.collect_disclosures(
        cfg, pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}), on_window=flushed.append
    )
    assert ret.empty  # on_window 모드 → 누적 안 함
    assert {c[0] for c in calls} == {"B", "I"}  # corp_cls 분할 대신 공시유형 필터
    for _ty, bgn, end in calls:
        assert pd.Timestamp(end) - pd.Timestamp(bgn) <= pd.Timedelta(days=92)
    assert len(flushed) >= 4  # 1년 → 창별 flush
    assert "n_rights_offering" in flushed[0].columns


def test_collect_disclosures_skips_fully_covered_windows(monkeypatch) -> None:
    fetched_windows: list[str] = []

    def _fake_window(cfg, pblntf_ty, start_ymd, end_ymd, corp_to_stock, *, on_page=None, page_offset=0):  # noqa: ANN001, ANN202
        fetched_windows.append(start_ymd)
        return []

    monkeypatch.setattr(disclosure, "_fetch_disclosure_window", _fake_window)
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-12-31"),
        out_dir=Path("x"), dart_api_key="k", retries=1, retry_sleep_sec=0.0,
    )
    covered = {d.normalize() for d in pd.bdate_range("2020-01-01", "2020-03-21")}
    disclosure.collect_disclosures(
        cfg, pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        on_window=lambda _df: None, covered_dates=covered,
    )
    # 첫 80일 창(2020-01-01~)은 전부 covered → 조회 안 함
    assert "20200101" not in fetched_windows
    assert any(w >= "20200322" for w in fetched_windows)


import io
import zipfile

import pytest


def test_download_corp_code_map_parses_zip(monkeypatch) -> None:
    xml = (
        "<result><list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>"
        "<stock_code>005930</stock_code><modify_date>20240101</modify_date></list>"
        "<list><corp_code>00999999</corp_code><corp_name>비상장</corp_name>"
        "<stock_code> </stock_code><modify_date>20240101</modify_date></list></result>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", xml)
    payload = buf.getvalue()

    class _Resp:
        status_code = 200
        content = payload
        headers = {"content-type": "application/x-msdownload"}

    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _Resp())
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-02-01"),
        out_dir=Path("x"), dart_api_key="k",
    )
    df = disclosure.download_corp_code_map(cfg)
    assert list(df["stock_code"]) == ["005930"]
    with pytest.raises(ValueError, match="DART_API_KEY"):
        disclosure.download_corp_code_map(
            AltDataFetchConfig(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-02-01"), out_dir=Path("x"))
        )


def _dart_cfg(**kw: object) -> AltDataFetchConfig:
    base: dict[str, object] = {
        "start": pd.Timestamp("2024-01-01"),
        "end": pd.Timestamp("2024-02-20"),
        "out_dir": Path("x"),
        "dart_api_key": "k",
        "retries": 1,
        "retry_sleep_sec": 0.0,
    }
    base.update(kw)
    return AltDataFetchConfig(**base)  # type: ignore[arg-type]


class _DartResp:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> object:
        return self._payload


def _dart_item(rcept_no: str, report_nm: str = "유상증자결정") -> dict[str, str]:
    return {
        "corp_cls": "Y",
        "corp_code": "00126380",
        "stock_code": "005930",
        "report_nm": report_nm,
        "rcept_dt": "20240115",
        "rcept_no": rcept_no,
    }


def _dart_page(items: list[dict[str, str]], total_page: int = 1, status: str = "000") -> dict[str, object]:
    return {"status": status, "message": "정상" if status == "000" else "조회된 데이터가 없습니다.", "total_page": total_page, "list": items}


def test_collect_disclosures_retains_receipt_identifiers(monkeypatch) -> None:
    items = [_dart_item("20240115000123"), _dart_item("20240115000124", "전환사채권발행결정")]
    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _DartResp(_dart_page(items)))
    seen: list = []
    out = disclosure.collect_disclosures(
        _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
    )
    assert len(seen) == 2
    observed_nos = {item["rcept_no"] for payload, *_ in seen for item in payload["list"]}
    assert observed_nos == {"20240115000123", "20240115000124"}
    assert "005930" in set(out["symbol"])


def test_collect_disclosures_correction_reobserved(monkeypatch) -> None:
    from src.backfill.altdata import disclosure as disc

    original = {"status": "000", "total_page": 1, "list": [_dart_item("20240115000123", "유상증자결정")]}
    amended = {"status": "000", "total_page": 1, "list": [_dart_item("20240115000123", "유상증자결정(정정)"), dict(_dart_item("20240115000123"), rm="정정")]}
    monkeypatch.setattr(disc, "wait_for_dart_slot", lambda _cfg: None)
    seen: list = []
    cfg = _dart_cfg()
    params: dict[str, object] = {"crtfc_key": "k", "page_no": 1}
    monkeypatch.setattr(disc.requests, "get", lambda *a, **k: _DartResp(original))
    disc._dart_get_json(
        disc._LIST_URL, params, cfg,
        on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
    )
    monkeypatch.setattr(disc.requests, "get", lambda *a, **k: _DartResp(amended))
    disc._dart_get_json(
        disc._LIST_URL, params, cfg,
        on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
    )
    assert len(seen) == 2
    assert seen[0][0]["list"][0]["report_nm"] == "유상증자결정"
    assert seen[1][0]["list"][0]["report_nm"] == "유상증자결정(정정)"
    assert all(entry[0]["list"][0]["rcept_no"] == "20240115000123" for entry in seen)


def test_collect_disclosures_parallel_pages_all_retained(monkeypatch) -> None:
    def _paged(url: str, params: object = None, timeout: object = None) -> _DartResp:
        page = int(dict(params)["page_no"])  # type: ignore[arg-type]
        return _DartResp(_dart_page([_dart_item(f"2024011500000{page}", f"공시{page}")], total_page=4))

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _paged)
    seen: list = []
    disclosure.collect_disclosures(
        _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
    )
    assert len(seen) == 8
    assert {meta["page_no"] for _, meta, *_ in seen} == {"1", "2", "3", "4"}
    # Then: page_index는 (pblntf_ty, 창) 버킷 오프셋 + page_no라 두 유형(B/I)의 동일 page_no가
    # 서로 다른 page_index로 갈려 evidence 경로가 겹치지 않는다.
    seen_indices = {pi for _, _, _, _, pi, _ in seen}
    assert len(seen_indices) == 8
    for _, meta, started, received, page, _attempt in seen:
        assert page % disclosure._MAX_PAGES_PER_BUCKET == int(meta["page_no"])
        assert getattr(started, "tzinfo", None) is not None and getattr(received, "tzinfo", None) is not None


def test_dart_get_json_retains_no_result_status(monkeypatch) -> None:
    payload = {"status": "013", "message": "조회된 데이터가 없습니다.", "list": []}
    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _DartResp(payload))
    seen: list = []
    out = disclosure._dart_get_json(
        disclosure._LIST_URL, {"crtfc_key": "k", "page_no": 1}, _dart_cfg(),
        on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
    )
    assert out["status"] == "013"
    assert len(seen) == 1 and seen[0][0]["status"] == "013"


def test_dart_get_json_rejects_unsupported_status_after_observing(monkeypatch) -> None:
    import pytest

    payload = {"status": "020", "message": "에러", "list": []}
    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _DartResp(payload))
    seen: list = []
    with pytest.raises(RuntimeError, match="020"):
        disclosure._dart_get_json(
            disclosure._LIST_URL, {"crtfc_key": "k", "page_no": 1}, _dart_cfg(),
            on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
        )
    assert len(seen) == 1 and seen[0][0]["status"] == "020"


def test_collect_disclosures_scope_unchanged(monkeypatch) -> None:
    requested: list = []

    def _tracked(url: str, params: object = None, timeout: object = None) -> _DartResp:
        requested.append((url, dict(params)))  # type: ignore[arg-type]
        return _DartResp(_dart_page([_dart_item("20240115000123")]))

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _tracked)
    out = disclosure.collect_disclosures(
        _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
    )
    assert {url for url, _ in requested} == {disclosure._LIST_URL}
    assert {params["pblntf_ty"] for _, params in requested} == {"B", "I"}
    assert list(out.columns) == list(disclosure._OUT_COLS)


def test_collect_disclosures_missing_publication_time_unknown(monkeypatch) -> None:
    items = [_dart_item("20240115000123")]
    assert "publish" not in str(items).lower()
    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _DartResp(_dart_page(items)))
    seen: list = []
    out = disclosure.collect_disclosures(
        _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
    )
    assert len(seen) == 2
    for _, meta, *_ in seen:
        assert set(meta) <= {"endpoint", "page_no"}
        assert not any("publish" in key for key in meta)
    assert out.iloc[0]["date"] == pd.Timestamp("2024-01-15")


def test_dart_list_page_retry_uses_actual_attempt(monkeypatch) -> None:
    calls = {"n": 0}

    def _flaky(url: str, params: object = None, timeout: object = None) -> _DartResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transport down")
        return _DartResp(_dart_page([_dart_item("20240115000123")]))

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _flaky)
    seen: list = []
    rows, total = disclosure._fetch_list_page(
        _dart_cfg(retries=2), {"crtfc_key": "k", "page_no": 1}, 1,
        on_page=lambda p, m, s, r, pi, ai: seen.append((p, m, s, r, pi, ai)),
    )
    assert len(rows) == 1 and total == 1
    assert len(seen) == 1
    assert seen[0][4] == 1 and seen[0][5] == 1


def test_collect_disclosures_capture_failure_propagates(monkeypatch) -> None:
    import pytest

    from src.data.capture_contracts import RawCaptureError

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    calls = {"n": 0}

    def _counted(url: str, params: object = None, timeout: object = None) -> _DartResp:
        calls["n"] += 1
        return _DartResp(_dart_page([_dart_item("20240115000123")]))

    def _failing(*a: object, **k: object) -> None:
        raise RawCaptureError("store full")

    monkeypatch.setattr(disclosure.requests, "get", _counted)
    with pytest.raises(RawCaptureError):
        disclosure.collect_disclosures(
            _dart_cfg(retries=3), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
            on_page=_failing,  # type: ignore[arg-type]
        )
    assert calls["n"] == 1

    def _leaky(*a: object, **k: object) -> None:
        raise OSError("disk SECRET-XYZ")

    with pytest.raises(RawCaptureError) as exc_info:
        disclosure.collect_disclosures(
            _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
            on_page=_leaky,  # type: ignore[arg-type]
        )
    assert "SECRET-XYZ" not in str(exc_info.value)


def test_collect_disclosures_two_types_do_not_collide_in_capture_store(monkeypatch, tmp_path) -> None:
    """실측: 2026-09-21 B/I 두 공시유형이 같은 (page_no=1, attempt=0)로 같은 evidence
    경로에 써져 conflicting immutable artifact identity로 매번 PARTIAL 처리됐다."""
    import uuid

    from src.data.capture_contracts import CaptureContext, CapturedResponse, CaptureDataset, CaptureStatus
    from src.data.capture_store import CaptureStore

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _DartResp(_dart_page([_dart_item("20240115000123")])))

    store = CaptureStore(tmp_path / "capture")
    run_id = f"disclosure-{uuid.uuid4().hex[:8]}"
    refs = []

    def _on_page(payload, meta, started, received, page_index, attempt_index):  # noqa: ANN001
        refs.append(
            store.append_response(
                CapturedResponse(
                    context=CaptureContext(
                        trading_date=pd.Timestamp("2024-02-20").date(), run_id=run_id, dataset=CaptureDataset.DISCLOSURE,
                        vendor="owner-local", endpoint="disclosure-collector", symbol=None, venue="KRX",
                        session="regular", capture_reason="altdata-backfill", cohort_id=None, scheduled_at=None,
                    ),
                    request_started_at=started, received_at=received,
                    payload=dict(payload) if isinstance(payload, dict) else None,
                    source_timestamp=None, source_published_at=None,
                    status=CaptureStatus.COMPLETE if isinstance(payload, dict) else CaptureStatus.FAILED,
                    page_index=int(page_index), attempt_index=int(attempt_index),
                    continuation={k: str(v) for k, v in dict(meta).items()},
                    error_type=None if isinstance(payload, dict) else "transport",
                )
            )
        )

    disclosure.collect_disclosures(
        _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}), on_page=_on_page,
    )

    assert len(refs) == 2  # B, I 유형별 page 1
    assert len({r.path for r in refs}) == 2  # 경로가 서로 다르다 -> 충돌 없음


def _pool_cfg(**kw: object):
    from src.backfill.altdata.dart_keys import DartCredential, DartKeyPool

    pool = kw.pop("pool", None)
    base: dict[str, object] = {
        "start": pd.Timestamp("2024-01-01"),
        "end": pd.Timestamp("2024-01-10"),
        "out_dir": Path("x"),
        "retries": 1,
        "retry_sleep_sec": 0.0,
    }
    base.update(kw)
    if pool is None:
        pool = DartKeyPool([DartCredential(label="KEY_1", key="KEY-A-VALUE"), DartCredential(label="KEY_2", key="KEY-B-VALUE")])
    base["dart_key_pool"] = pool
    return AltDataFetchConfig(**base)  # type: ignore[arg-type]


def test_collect_disclosures_quota_failover_to_next_key(monkeypatch) -> None:
    counts: dict[str, int] = {"KEY-A-VALUE": 0, "KEY-B-VALUE": 0}

    def _fake(url: str, params: object = None, timeout: object = None):
        key = str(dict(params)["crtfc_key"])  # type: ignore[arg-type]
        counts[key] += 1
        if key == "KEY-A-VALUE":
            return _DartResp({"status": "020", "message": "한도초과", "list": []})
        return _DartResp(_dart_page([_dart_item("20240115000123")]))

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)
    out = disclosure.collect_disclosures(
        _pool_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
    )
    assert not out.empty
    assert counts["KEY-A-VALUE"] <= 2
    assert counts["KEY-B-VALUE"] >= 1


def test_collect_disclosures_failover_uses_unique_capture_identities(monkeypatch) -> None:
    def _fake(url: str, params: object = None, timeout: object = None):
        key = str(dict(params)["crtfc_key"])  # type: ignore[arg-type]
        if key == "KEY-A-VALUE":
            return _DartResp({"status": "020", "message": "한도초과", "list": []})
        return _DartResp(_dart_page([_dart_item("20240115000123")]))

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)
    seen: list = []
    disclosure.collect_disclosures(
        _pool_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        on_page=lambda p, m, s, r, pi, ai: seen.append((pi, ai)),
    )
    pairs = [(pi, ai) for pi, ai in seen]
    assert len(pairs) == len(set(pairs)) and len(pairs) >= 1


def test_collect_disclosures_all_exhausted_stops_requesting(monkeypatch) -> None:
    import pytest

    from src.backfill.altdata.ratelimit import DartQuotaExhaustedError

    calls = {"n": 0}

    def _fake(url: str, params: object = None, timeout: object = None):
        calls["n"] += 1
        return _DartResp({"status": "020", "message": "한도초과", "list": []})

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)

    def _boom_sleep(_s: float) -> None:
        raise AssertionError("no retry sleeps on quota exhaustion")

    import time as _time

    monkeypatch.setattr(_time, "sleep", _boom_sleep)
    with pytest.raises(DartQuotaExhaustedError):
        disclosure.collect_disclosures(
            _pool_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        )
    assert calls["n"] == 2


def test_collect_disclosures_bad_new_key_fails_over(monkeypatch, caplog) -> None:
    import logging

    def _fake(url: str, params: object = None, timeout: object = None):
        key = str(dict(params)["crtfc_key"])  # type: ignore[arg-type]
        if key == "KEY-A-VALUE":
            return _DartResp({"status": "010", "message": "미등록키", "list": []})
        return _DartResp(_dart_page([_dart_item("20240115000123")]))

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)
    with caplog.at_level(logging.ERROR, logger="src.backfill.altdata.dart_keys"):
        out = disclosure.collect_disclosures(
            _pool_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        )
    assert not out.empty
    assert "KEY_1" in caplog.text and "010" in caplog.text


def test_collect_disclosures_rejected_plus_exhausted_not_tolerable(monkeypatch) -> None:
    import pytest

    from src.backfill.altdata.ratelimit import DartKeysUnusableError

    def _fake(url: str, params: object = None, timeout: object = None):
        key = str(dict(params)["crtfc_key"])  # type: ignore[arg-type]
        if key == "KEY-A-VALUE":
            return _DartResp({"status": "010", "message": "미등록키", "list": []})
        return _DartResp({"status": "020", "message": "한도초과", "list": []})

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)
    with pytest.raises(DartKeysUnusableError):
        disclosure.collect_disclosures(
            _pool_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        )


def test_collect_disclosures_single_key_behavior_unchanged(monkeypatch) -> None:
    import pytest

    from src.backfill.altdata.ratelimit import DartNonRetryableError

    seen_params: list = []

    def _fake(url: str, params: object = None, timeout: object = None):
        seen_params.append(dict(params))  # type: ignore[arg-type]
        return _DartResp(_dart_page([_dart_item("20240115000123")]))

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)
    out = disclosure.collect_disclosures(
        _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
    )
    assert not out.empty
    assert all("crtfc_key" in p for p in seen_params)

    def _quota(url: str, params: object = None, timeout: object = None):
        return _DartResp({"status": "020", "message": "한도초과", "list": []})

    monkeypatch.setattr(disclosure.requests, "get", _quota)
    with pytest.raises(DartNonRetryableError):
        disclosure.collect_disclosures(
            _dart_cfg(), pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        )


def _assert_no_secret_in_traceback(exc: BaseException, secret: str) -> None:
    """The requests error text embeds the key-bearing URL; it must not survive via chaining either."""
    import traceback

    rendered = "".join(traceback.format_exception(exc))
    assert secret not in rendered
    assert exc.__cause__ is None


def test_collect_disclosures_transport_errors_never_leak_key(monkeypatch) -> None:
    import pytest

    secret = "SUPER-SECRET-KEY-VALUE"

    def _boom(url: str, params: object = None, timeout: object = None):
        raise ConnectionError(f"failed fetching {params} containing {secret}")

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _boom)
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-10"),
        out_dir=Path("x"), dart_api_key=secret, retries=1, retry_sleep_sec=0.0,
    )
    with pytest.raises(RuntimeError) as exc_info:
        disclosure._dart_get_json(
            disclosure._LIST_URL, {"page_no": 1}, cfg,
            credential=__import__("src.backfill.altdata.dart_keys", fromlist=["DartCredential"]).DartCredential(label="DEFAULT", key=secret),
        )
    _assert_no_secret_in_traceback(exc_info.value, secret)


def test_download_corp_code_map_failover(monkeypatch) -> None:
    import io
    import zipfile

    xml = (
        "<result><list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>"
        "<stock_code>005930</stock_code><modify_date>20240101</modify_date></list></result>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", xml)
    payload = buf.getvalue()

    class _Resp:
        def __init__(self, content: bytes, ctype: str = "application/x-msdownload") -> None:
            self.status_code = 200
            self.content = content
            self.headers = {"content-type": ctype}

    def _fake(url: str, params: object = None, timeout: object = None):
        key = str(dict(params)["crtfc_key"])  # type: ignore[arg-type]
        if key == "KEY-A-VALUE":
            return _Resp("<result><status>020</status><message>한도초과</message></result>".encode(), "text/xml")
        return _Resp(payload)

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)
    df = disclosure.download_corp_code_map(_pool_cfg())
    assert list(df["stock_code"]) == ["005930"]

    def _unknown(url: str, params: object = None, timeout: object = None):
        return _Resp(b"not a zip at all", "text/html")

    monkeypatch.setattr(disclosure.requests, "get", _unknown)
    import pytest

    with pytest.raises(RuntimeError):
        disclosure.download_corp_code_map(_pool_cfg())


def test_download_corp_code_map_rejected_key_fails_over(monkeypatch) -> None:
    import io
    import zipfile

    xml = (
        "<result><list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>"
        "<stock_code>005930</stock_code><modify_date>20240101</modify_date></list></result>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", xml)
    payload = buf.getvalue()

    class _Resp:
        def __init__(self, content: bytes) -> None:
            self.status_code = 200
            self.content = content

    def _fake(url: str, params: object = None, timeout: object = None):
        key = str(dict(params)["crtfc_key"])  # type: ignore[arg-type]
        if key == "KEY-A-VALUE":
            return _Resp("<result><status>010</status><message>미등록</message></result>".encode())
        return _Resp(payload)

    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)
    monkeypatch.setattr(disclosure.requests, "get", _fake)
    df = disclosure.download_corp_code_map(_pool_cfg())
    assert list(df["stock_code"]) == ["005930"]


def test_download_corp_code_map_transport_and_http_errors(monkeypatch) -> None:
    import pytest

    secret = "CORP-SECRET-VALUE"
    monkeypatch.setattr(disclosure, "wait_for_dart_slot", lambda _cfg: None)

    def _boom(url: str, params: object = None, timeout: object = None):
        raise ConnectionError(f"down {secret}")

    monkeypatch.setattr(disclosure.requests, "get", _boom)
    cfg = _pool_cfg()
    with pytest.raises(RuntimeError) as exc_info:
        disclosure.download_corp_code_map(cfg)
    _assert_no_secret_in_traceback(exc_info.value, secret)

    class _Bad:
        status_code = 500
        content = b""

    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _Bad())
    with pytest.raises(RuntimeError, match="500"):
        disclosure.download_corp_code_map(cfg)
