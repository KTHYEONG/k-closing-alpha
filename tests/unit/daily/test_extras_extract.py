"""Invariant guards for the nightly decision-extras extraction job."""

from __future__ import annotations

import ast
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src import settings
from src.daily import extras_extract
from src.daily.extras_extract import extract_decision_extras, main
from src.data.capture_contracts import (
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureStatus,
    build_cohort,
)
from src.data.capture_store import CaptureStore
from src.data.decision_extras import DECISION_EXTRAS_DATASET, INVESTOR_DECISION_DATASET
from src.data.research_store import read_research_partition, research_partition_path
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_OK, RUN_OUTCOME_SKIPPED

SEOUL = ZoneInfo("Asia/Seoul")
DATE = "2026-11-04"
DAY = date(2026, 11, 4)
RUN = "decision-2026-11-04-0976289f"
NOW = datetime(2026, 11, 4, 16, 30, tzinfo=SEOUL)
START = datetime(2026, 11, 4, 15, 20, tzinfo=SEOUL)
COHORT = ("000660", "004490")
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _history(monkeypatch: Any, tmp_path: Path) -> Path:
    history = tmp_path / "history"
    monkeypatch.setattr(settings, "HISTORY_DIR", history)
    return history


def _raw(
    store: CaptureStore,
    *,
    vendor: str,
    dataset: CaptureDataset,
    endpoint: str,
    payload: dict[str, Any],
    run_id: str = RUN,
    symbol: str | None = None,
) -> None:
    received = START + timedelta(seconds=2)
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
            status=CaptureStatus.COMPLETE,
            source_timestamp=None,
            source_published_at=None,
            page_index=0,
            attempt_index=0,
            continuation={},
            error_type=None,
        )
    )


def _seed_decision_run(store: CaptureStore) -> None:
    _raw(
        store,
        vendor="kiwoom",
        dataset=CaptureDataset.SCAN,
        endpoint="fluctuation-ranking",
        payload={
            "return_code": 0,
            "pred_pre_flu_rt_upper": [
                {"stk_cd": "004490_AL", "cntr_str": "123.45", "flu_rt": "+8.40", "buy_req": "+10", "sel_req": "-5"}
            ],
        },
    )
    _raw(
        store,
        vendor="kis",
        dataset=CaptureDataset.PRICE,
        endpoint="inquire-price",
        payload={"rt_cd": "0", "output": {"vol_tnrt": "2.25", "vi_cls_code": "N", "temp_stop_yn": "N"}},
        symbol="004490",
    )
    _raw(
        store,
        vendor="kis",
        dataset=CaptureDataset.INVESTOR_ESTIMATE,
        endpoint="investor-trend-estimate",
        payload={
            "rt_cd": "0",
            "output2": [
                {"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "+10", "orgn_fake_ntby_qty": "-3", "sum_fake_ntby_qty": "7"},
                {"bsop_hour_gb": "2", "frgn_fake_ntby_qty": "4", "orgn_fake_ntby_qty": "1", "sum_fake_ntby_qty": "5"},
            ],
        },
        symbol="004490",
    )
    frame = pd.DataFrame(
        {
            "symbol": list(COHORT),
            "admitted": [True, True],
            "snapshot_timestamp": [START, START],
            "feature_available_timestamp": [START, START],
        }
    )
    cohort = build_cohort(DAY, list(COHORT), list(COHORT), {}, eligibility_rule_version="v1")
    store.publish_decision(
        frame,
        cohort=cohort,
        run_id=RUN,
        completed_at=START + timedelta(minutes=1),
        entries=(),
    )


def _root(tmp_path: Path) -> Path:
    return tmp_path / "capture"


def _patch_main(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[tuple[str, str, dict[str, Any]]]:
    recorded: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(extras_extract, "resolve_capture_root", lambda: root)
    monkeypatch.setattr(extras_extract, "_now", lambda: NOW)
    monkeypatch.setattr(
        extras_extract,
        "record_run_outcome",
        lambda job, outcome, **kwargs: recorded.append((job, outcome, kwargs)),
    )
    return recorded


def test_extract_writes_partitions_in_cohort_order(tmp_path: Path) -> None:
    store = CaptureStore(_root(tmp_path))
    _seed_decision_run(store)

    counts = extract_decision_extras(DATE, store=store, now=NOW)

    assert counts == (len(COHORT), 2)
    extras = read_research_partition(DECISION_EXTRAS_DATASET, DATE)
    assert extras["symbol"].tolist() == list(COHORT)
    assert extras.set_index("symbol").loc["004490", "kiwoom_cntr_str"] == pytest.approx(123.45)
    assert extras.set_index("symbol").loc["004490", "kis_vol_tnrt"] == pytest.approx(2.25)
    assert pd.isna(extras.set_index("symbol").loc["000660", "kiwoom_cntr_str"])
    assert extras["run_id"].tolist() == [RUN, RUN]
    series = read_research_partition(INVESTOR_DECISION_DATASET, DATE)
    assert sorted(series["bsop_hour_gb"].tolist()) == ["1", "2"]


def test_extract_without_vendor_extras_keeps_cohort_with_nulls(tmp_path: Path) -> None:
    store = CaptureStore(_root(tmp_path))
    _seed_decision_run(store)
    for path in (_root(tmp_path) / "raw" / DATE).rglob("*.json.gz"):
        if "/SCAN/" in path.as_posix() or "/PRICE/" in path.as_posix() or "/INVESTOR_ESTIMATE/" in path.as_posix():
            path.unlink()

    counts = extract_decision_extras(DATE, store=store, now=NOW)

    assert counts == (len(COHORT), 0)
    extras = read_research_partition(DECISION_EXTRAS_DATASET, DATE).set_index("symbol")
    assert list(extras.index) == list(COHORT)
    assert extras["kiwoom_cntr_str"].isna().all()
    assert extras["kis_vol_tnrt"].isna().all()
    assert not research_partition_path(INVESTOR_DECISION_DATASET, DATE).exists()


def test_extract_idempotent(tmp_path: Path) -> None:
    store = CaptureStore(_root(tmp_path))
    _seed_decision_run(store)

    first = extract_decision_extras(DATE, store=store, now=NOW)
    before = {name: read_research_partition(name, DATE) for name in (DECISION_EXTRAS_DATASET, INVESTOR_DECISION_DATASET)}
    second = extract_decision_extras(DATE, store=store, now=NOW)

    assert first == second
    for name, frame in before.items():
        pd.testing.assert_frame_equal(read_research_partition(name, DATE), frame)


def test_extract_without_decision_run_raises(tmp_path: Path) -> None:
    store = CaptureStore(_root(tmp_path))

    with pytest.raises(FileNotFoundError, match="no qualifying decision run"):
        extract_decision_extras(DATE, store=store, now=NOW)
    assert not research_partition_path(DECISION_EXTRAS_DATASET, DATE).exists()


def test_retro_skips_raw_absent_dates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _root(tmp_path)
    _seed_decision_run(CaptureStore(root))
    recorded = _patch_main(monkeypatch, root)

    main(["--from", "2026-11-03", "--to", "2026-11-05"])

    assert [(job, outcome) for job, outcome, _ in recorded] == [("decision_extras", RUN_OUTCOME_OK)]
    metrics = recorded[0][2]["metrics"]
    assert metrics == {
        "ok": 1,
        "raw_absent": 2,
        "no_decision_run": 0,
        "invalid": 0,
        "extras_rows": 2,
        "series_rows": 2,
    }
    assert recorded[0][2]["run_date"] == "2026-11-05"
    assert read_research_partition(DECISION_EXTRAS_DATASET, DATE)["symbol"].tolist() == list(COHORT)
    assert not research_partition_path(DECISION_EXTRAS_DATASET, "2026-11-03").exists()


def test_clock_is_seoul_wall_time() -> None:
    assert extras_extract._now().utcoffset() == timedelta(hours=9)


def test_main_default_date_is_clock_date(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _root(tmp_path)
    _seed_decision_run(CaptureStore(root))
    recorded = _patch_main(monkeypatch, root)

    main([])

    assert [(job, outcome) for job, outcome, _ in recorded] == [("decision_extras", RUN_OUTCOME_OK)]
    assert recorded[0][2]["run_date"] == DATE
    assert research_partition_path(INVESTOR_DECISION_DATASET, DATE).exists()


def test_main_without_decision_input_skips_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = CaptureStore(root)
    _raw(
        store,
        vendor="kis",
        dataset=CaptureDataset.PRICE,
        endpoint="inquire-price",
        payload={"rt_cd": "0", "output": {"vol_tnrt": "1"}},
        run_id="postclose-2026-11-04-abcd1234",
        symbol="004490",
    )
    recorded = _patch_main(monkeypatch, root)

    main(["--date", DATE])

    assert len(recorded) == 1
    job, outcome, kwargs = recorded[0]
    assert (job, outcome, kwargs["reason"]) == ("decision_extras", RUN_OUTCOME_SKIPPED, "no_decision_input")
    assert kwargs["metrics"]["no_decision_run"] == 1
    assert not research_partition_path(DECISION_EXTRAS_DATASET, DATE).exists()


def test_main_absent_raw_day_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = _patch_main(monkeypatch, _root(tmp_path))

    main(["--date", DATE])

    assert [(outcome, kwargs["reason"]) for _, outcome, kwargs in recorded] == [
        (RUN_OUTCOME_SKIPPED, "no_decision_input")
    ]
    assert recorded[0][2]["metrics"]["raw_absent"] == 1


def test_main_corrupt_raw_evidence_degrades_without_partitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _root(tmp_path)
    store = CaptureStore(root)
    _seed_decision_run(store)
    corrupt = root / "raw" / DATE / "kis" / "PRICE" / "inquire-price" / RUN / "000000-p0000-a09.json.gz"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"garbage")
    recorded = _patch_main(monkeypatch, root)

    main(["--date", DATE])

    assert len(recorded) == 1
    _, outcome, kwargs = recorded[0]
    assert (outcome, kwargs["reason"]) == (RUN_OUTCOME_DEGRADED, "invalid_raw_evidence")
    assert kwargs["metrics"]["invalid"] == 1
    assert not research_partition_path(DECISION_EXTRAS_DATASET, DATE).exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--date", DATE, "--from", DATE],
        ["--from", DATE],
        ["--to", DATE],
        ["--from", "2026-11-05", "--to", DATE],
        ["--date", "2026-13-01"],
    ],
)
def test_main_rejects_inconsistent_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> None:
    recorded = _patch_main(monkeypatch, _root(tmp_path))

    with pytest.raises(SystemExit):
        main(argv)
    assert recorded == []


def test_job_module_imports_no_broker_client() -> None:
    # src.tools.alerts (shared run-outcome dependency) loads only the KIS credential-pool reader, never a client.
    code = (
        "import sys\n"
        "import src.daily.extras_extract\n"
        "print(sorted(name for name in sys.modules if name.startswith('src.api')))\n"
    )
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and literal probe code, no untrusted input
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    loaded = set(ast.literal_eval(proc.stdout.strip()))
    assert loaded <= {"src.api", "src.api.kis", "src.api.kis.key_pool"}
