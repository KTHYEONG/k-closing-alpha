"""Invariant guards for decision-run extras parsing over raw capture envelopes."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src import settings
from src.data.capture_contracts import (
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    build_cohort,
)
from src.data.capture_store import CaptureStore
from src.data.decision_extras import (
    KIS_PRICE_CODE_FIELDS,
    KIS_PRICE_NUMERIC_FIELDS,
    build_decision_extras,
    decision_run_id,
    investor_series_rows,
    kis_price_rows,
    kiwoom_scan_rows,
)

SEOUL = ZoneInfo("Asia/Seoul")
DAY = date(2026, 11, 4)
DATE = DAY.isoformat()
RUN = "decision-2026-11-04-0976289f"
START = datetime(2026, 11, 4, 15, 20, tzinfo=SEOUL)
RECEIVED = START + timedelta(seconds=1)
COHORT = ("000660", "004490", "035420")


@pytest.fixture(autouse=True)
def _history(monkeypatch: Any, tmp_path: Path) -> Path:
    history = tmp_path / "history"
    monkeypatch.setattr(settings, "HISTORY_DIR", history)
    return history


def _store(tmp_path: Path) -> CaptureStore:
    return CaptureStore(tmp_path / "capture")


def _write(
    store: CaptureStore,
    *,
    vendor: str,
    dataset: CaptureDataset,
    endpoint: str,
    payload: dict[str, Any] | None,
    symbol: str | None = None,
    page: int = 0,
    attempt: int = 0,
    run_id: str = RUN,
    received: datetime = RECEIVED,
) -> None:
    store.append_response(
        CapturedResponse(
            context=CaptureContext(
                trading_date=DAY,
                run_id=run_id,
                dataset=dataset,
                vendor=vendor,
                endpoint=endpoint,
                symbol=symbol,
                venue="KRX",
                session="regular",
                capture_reason="decision-input",
                cohort_id=None,
                scheduled_at=None,
            ),
            request_started_at=received - timedelta(seconds=1),
            received_at=received,
            payload=payload,
            status=CaptureStatus.COMPLETE if payload is not None else CaptureStatus.FAILED,
            source_timestamp=None,
            source_published_at=None,
            page_index=page,
            attempt_index=attempt,
            continuation={},
            error_type=None if payload is not None else "vendor_failure",
        )
    )


def _write_scan(store: CaptureStore, rows: list[Any], *, page: int = 0, run_id: str = RUN) -> None:
    _write(
        store,
        vendor="kiwoom",
        dataset=CaptureDataset.SCAN,
        endpoint="fluctuation-ranking",
        payload={"return_code": 0, "pred_pre_flu_rt_upper": rows},
        page=page,
        run_id=run_id,
    )


def _price_output(**overrides: str) -> dict[str, str]:
    output = dict.fromkeys(KIS_PRICE_NUMERIC_FIELDS, "1.5")
    output.update(dict.fromkeys(KIS_PRICE_CODE_FIELDS, "N"))
    output.update(overrides)
    return output


def _write_price(store: CaptureStore, symbol: str | None, payload: dict[str, Any] | None, attempt: int = 0) -> None:
    _write(
        store,
        vendor="kis",
        dataset=CaptureDataset.PRICE,
        endpoint="inquire-price",
        payload=payload,
        symbol=symbol,
        attempt=attempt,
    )


def _write_investor(store: CaptureStore, symbol: str, output2: Any, attempt: int = 0) -> None:
    _write(
        store,
        vendor="kis",
        dataset=CaptureDataset.INVESTOR_ESTIMATE,
        endpoint="investor-trend-estimate",
        payload={"rt_cd": "0", "output2": output2},
        symbol=symbol,
        attempt=attempt,
    )


def _stream(store: CaptureStore) -> list[tuple[str, dict[str, Any]]]:
    return list(store.iter_run_envelopes(DATE, RUN))


def test_scan_rows_parse_cntr_str_and_codes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_scan(
        store,
        [
            {"stk_cd": "004490_AL", "cntr_str": "123.45", "buy_req": "+1500", "sel_req": "-300", "flu_rt": "+8.40"},
            {"stk_cd": "000660", "cntr_str": "nan", "buy_req": "abc", "sel_req": "", "flu_rt": "-1.2"},
            {"stk_cd": "  ", "cntr_str": "9"},
            "not-a-row",
        ],
    )
    _write_price(store, "004490", {"rt_cd": "0", "output": _price_output()})

    scan = kiwoom_scan_rows(_stream(store)).set_index("symbol")

    assert list(scan.index) == ["004490", "000660"]
    assert scan.loc["004490", "kiwoom_cntr_str"] == pytest.approx(123.45)
    assert scan.loc["004490", "kiwoom_buy_req"] == pytest.approx(1500.0)
    assert scan.loc["004490", "kiwoom_sel_req"] == pytest.approx(-300.0)
    assert scan.loc["004490", "kiwoom_flu_rt"] == pytest.approx(8.4)
    assert math.isnan(scan.loc["000660", "kiwoom_cntr_str"])
    assert math.isnan(scan.loc["000660", "kiwoom_buy_req"])
    assert math.isnan(scan.loc["000660", "kiwoom_sel_req"])
    assert scan.loc["000660", "kiwoom_flu_rt"] == pytest.approx(-1.2)
    assert scan.loc["004490", "kiwoom_received_at"] == RECEIVED


def test_scan_rows_skip_failed_pages_and_foreign_routes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write(
        store,
        vendor="kiwoom",
        dataset=CaptureDataset.SCAN,
        endpoint="fluctuation-ranking",
        payload=None,
        page=0,
    )
    _write_scan(store, [{"stk_cd": "004490_AL", "cntr_str": "2.0"}], page=1)
    _write_price(store, "004490", {"rt_cd": "0", "output": _price_output()})

    scan = kiwoom_scan_rows(_stream(store))

    assert scan["symbol"].tolist() == ["004490"]
    assert scan["kiwoom_cntr_str"].tolist() == [pytest.approx(2.0)]


def test_scan_rows_first_page_wins_on_duplicate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_scan(store, [{"stk_cd": "004490_AL", "cntr_str": "1.5"}, {"stk_cd": "004490", "cntr_str": "7.0"}], page=0)
    _write_scan(store, [{"stk_cd": "004490_AL", "cntr_str": "9.9"}, {"stk_cd": "000660", "cntr_str": "3.0"}], page=1)

    scan = kiwoom_scan_rows(_stream(store)).set_index("symbol")

    assert scan.loc["004490", "kiwoom_cntr_str"] == pytest.approx(1.5)
    assert scan.loc["000660", "kiwoom_cntr_str"] == pytest.approx(3.0)


def test_price_rows_fields_and_last_attempt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_price(store, "004490", {"rt_cd": "1", "msg1": "temporary failure"}, attempt=0)
    _write_price(store, "004490", {"rt_cd": "0", "output": _price_output(vol_tnrt="2.25", vi_cls_code="Y")}, attempt=1)

    price = kis_price_rows(_stream(store))

    assert price["symbol"].tolist() == ["004490"]
    for field in KIS_PRICE_NUMERIC_FIELDS:
        assert f"kis_{field}" in price.columns
    for field in KIS_PRICE_CODE_FIELDS:
        assert f"kis_{field}" in price.columns
    assert price.loc[0, "kis_vol_tnrt"] == pytest.approx(2.25)
    assert price.loc[0, "kis_vi_cls_code"] == "Y"
    assert price.loc[0, "kis_received_at"] == RECEIVED


def test_price_rows_keep_last_successful_attempt_regardless_of_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_price(store, "004490", {"rt_cd": "0", "output": _price_output(vol_tnrt="1.0")}, attempt=0)
    _write_price(store, "004490", {"rt_cd": "0", "output": _price_output(vol_tnrt="2.0")}, attempt=1)

    reordered = list(reversed(_stream(store)))
    price = kis_price_rows(reordered)

    assert price.loc[0, "kis_vol_tnrt"] == pytest.approx(2.0)


def test_price_rows_skip_unkeyed_and_unparseable_responses(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_price(store, None, {"rt_cd": "0", "output": _price_output()})
    _write_price(store, "000660", {"rt_cd": "0", "output": "not-a-dict"})
    _write_price(store, "035420", {"rt_cd": "0", "output": _price_output(vol_tnrt="n/a", temp_stop_yn="")})
    _write_investor(store, "004490", [])

    price = kis_price_rows(_stream(store)).set_index("symbol")

    assert list(price.index) == ["035420"]
    assert math.isnan(price.loc["035420", "kis_vol_tnrt"])
    assert price.loc["035420", "kis_temp_stop_yn"] is None


def test_investor_series_long_shape(tmp_path: Path) -> None:
    store = _store(tmp_path)
    buckets = [
        {"bsop_hour_gb": hour, "frgn_fake_ntby_qty": "+10", "orgn_fake_ntby_qty": "-5", "sum_fake_ntby_qty": "5"}
        for hour in ("1", "2", "3", "4", "5")
    ]
    _write_investor(store, "004490", buckets)

    series = investor_series_rows(_stream(store))

    assert len(series) == 5
    assert set(zip(series["symbol"], series["bsop_hour_gb"], strict=True)) == {("004490", str(h)) for h in range(1, 6)}
    assert series["frgn_fake_ntby_qty"].tolist() == [10] * 5
    assert series["orgn_fake_ntby_qty"].tolist() == [-5] * 5
    assert series["received_at"].tolist() == [RECEIVED] * 5


def test_investor_series_skips_unkeyed_entries_and_keeps_last_attempt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_investor(
        store,
        "004490",
        [
            "stray",
            {"frgn_fake_ntby_qty": "1"},
            {"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "1", "sum_fake_ntby_qty": "2"},
        ],
        attempt=0,
    )
    _write_investor(
        store,
        "004490",
        [{"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "7", "orgn_fake_ntby_qty": "x", "sum_fake_ntby_qty": "8"}],
        attempt=1,
    )
    _write_investor(store, "000660", "not-a-list")
    _write(
        store,
        vendor="kis",
        dataset=CaptureDataset.INVESTOR_ESTIMATE,
        endpoint="investor-trend-estimate",
        payload={"rt_cd": "0", "output2": [{"bsop_hour_gb": "2"}]},
        symbol=None,
    )

    series = investor_series_rows(_stream(store))

    assert series[["symbol", "bsop_hour_gb"]].values.tolist() == [["004490", "1"]]
    assert series["frgn_fake_ntby_qty"].tolist() == [7]
    assert pd.isna(series["orgn_fake_ntby_qty"].iloc[0])


def test_build_keeps_cohort_order_and_nulls(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _write_scan(store, [{"stk_cd": "004490_AL", "cntr_str": "123.45", "flu_rt": "+8.40"}])
    _write_price(store, "004490", {"rt_cd": "0", "output": _price_output()})

    extras = build_decision_extras(COHORT, kiwoom_scan_rows(_stream(store)), kis_price_rows(_stream(store)), RUN)

    assert extras["symbol"].tolist() == list(COHORT)
    missing = extras.set_index("symbol").loc["000660"]
    assert math.isnan(missing["kiwoom_cntr_str"])
    assert pd.isna(missing["kiwoom_received_at"])
    assert pd.isna(missing["kis_received_at"])
    assert extras["run_id"].tolist() == [RUN] * len(COHORT)


def _publish_decision(store: CaptureStore, *, run_id: str, completed: datetime, cohort: tuple[str, ...]) -> None:
    frame = pd.DataFrame(
        {
            "symbol": list(cohort),
            "admitted": [True] * len(cohort),
            "snapshot_timestamp": [START] * len(cohort),
            "feature_available_timestamp": [START] * len(cohort),
        }
    )
    decision_cohort = build_cohort(
        DAY, list(cohort), list(cohort), {}, eligibility_rule_version="v1"
    )
    store.publish_decision(
        frame,
        cohort=decision_cohort,
        run_id=run_id,
        completed_at=completed,
        entries=(),
    )


def test_decision_run_id_ignores_non_decision_manifests(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _publish_decision(store, run_id=RUN, completed=START + timedelta(minutes=1), cohort=("004490",))
    store.publish_manifest(
        CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=DAY,
                run_id="research-roster-later",
                dataset=CaptureDataset.SCAN,
                vendor="kis",
                endpoint="research-roster",
                symbol=None,
                venue="KRX",
                session="regular",
                capture_reason="research-roster",
                cohort_id=None,
                scheduled_at=None,
            ),
            cohort=None,
            completed_at=START + timedelta(minutes=5),
            entries=(),
            artifacts=(),
            status=CaptureStatus.COMPLETE,
        )
    )

    assert decision_run_id(store, DATE, available_by=START + timedelta(hours=1)) == RUN


def test_decision_run_id_respects_cutoff_and_absence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(FileNotFoundError, match="no qualifying decision run"):
        decision_run_id(store, DATE, available_by=START + timedelta(hours=1))

    _publish_decision(store, run_id=RUN, completed=START + timedelta(minutes=1), cohort=("004490",))

    with pytest.raises(FileNotFoundError, match="no qualifying decision run"):
        decision_run_id(store, DATE, available_by=START)
    with pytest.raises(ValueError, match="available_by must be an aware cutoff"):
        decision_run_id(store, DATE, available_by=datetime(2026, 11, 4, 16, 0))
