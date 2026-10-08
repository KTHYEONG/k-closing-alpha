"""Invariant guards for the incremental NXT calibration collection."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import weakref
from datetime import datetime

import pandas as pd
import pytest

from src.backfill.intraday import nxt_calibration_pairs as ncp
from src.backfill.intraday import nxt_calibration_run as ncr
from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger
from src.config.collection import CollectionSettings
from src.config.nxt_reconstruction import NxtReconstructionSettings
from src.data.capture_contracts import SEOUL, ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry
from src.data.capture_store import CaptureStore

_DAY1 = "2024-02-26"
_DAY2 = "2024-02-27"
_DAY3 = "2024-02-28"
_NOW = datetime(2024, 3, 5, 12, 0, tzinfo=SEOUL)


def _hhmmss(minute_of_day: int) -> int:
    return (minute_of_day // 60) * 10000 + (minute_of_day % 60) * 100


def _kis_frame(symbol: str, *, volume: int = 100, auction: int = 1000) -> pd.DataFrame:
    stamps = [_hhmmss(m) for m in range(9 * 60, 15 * 60 + 21)] + [153000]
    volumes = [volume] * (len(stamps) - 1) + [auction]
    return pd.DataFrame({
        "snapshot_date": ["x"] * len(stamps),
        "symbol": [symbol] * len(stamps),
        "ts_hms": stamps,
        "open": [70000] * len(stamps),
        "high": [70100] * len(stamps),
        "low": [69900] * len(stamps),
        "close": [70000] * len(stamps),
        "volume": volumes,
        "value_krw": [70000 * v for v in volumes],
        "has_trade": [True] * len(stamps),
        "vendor": ["kis"] * len(stamps),
    })


def _toss_frame(symbol: str, *, volume: int = 150) -> pd.DataFrame:
    stamps = [_hhmmss(m) for m in range(9 * 60 + 1, 15 * 60 + 21)]
    assert len(stamps) == 380
    return pd.DataFrame({
        "snapshot_date": ["x"] * len(stamps),
        "symbol": [symbol] * len(stamps),
        "ts_hms": stamps,
        "open": [70000] * len(stamps),
        "high": [70100] * len(stamps),
        "low": [69900] * len(stamps),
        "close": [70000] * len(stamps),
        "volume": [volume] * len(stamps),
        "value_krw": [70000 * volume] * len(stamps),
        "has_trade": [True] * len(stamps),
        "vendor": ["toss"] * len(stamps),
    })


_EOD = 380 * 100 + 1000


def _sample(day: str, symbol: str, *, raw: bool = True, eod: float = float(_EOD),
            partial: bool = False) -> ncp.CalibrationSample:
    frame = _kis_frame(symbol) if not partial else _kis_frame(symbol).iloc[:3].copy()
    return ncp.CalibrationSample(day, symbol, frame, float(eod), raw)


def _ref() -> ArtifactRef:
    return ArtifactRef(path="r/1.json", sha256="ab" * 32, bytes=10)


def _entry(symbol: str, status: CaptureStatus, reason: str, *, rows: int = 0) -> CoverageEntry:
    refs = () if status == CaptureStatus.FAILED and reason.startswith("transport:") else (_ref(),)
    if status in (CaptureStatus.NOT_APPLICABLE, CaptureStatus.NO_TRADES) and not refs:
        refs = (_ref(),)
    return CoverageEntry(
        symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
        scheduled_at=None, status=status, rows=rows,
        first_event_time=None, last_event_time=None, reason=reason, raw_refs=refs,
    )


class _StubAcquire:
    """Scripted acquire_toss_regular_bars stand-in keyed by (date, symbol)."""

    def __init__(self, script: dict[tuple[str, str], str]) -> None:
        self.script = dict(script)
        self.calls: list[tuple[str, str]] = []
        self.refs: list[weakref.ref] = []

    async def __call__(self, client, session, symbol, snapshot_date, *, eod_volume,
                       profile, capture_store, run_id):
        key = (str(snapshot_date), str(symbol))
        self.calls.append(key)
        mode = self.script.get(key, "pair")
        if mode == "pair":
            frame = _toss_frame(str(symbol))
            self.refs.append(weakref.ref(frame))
            return frame, _entry(str(symbol), CaptureStatus.NOT_APPLICABLE, "toss_consolidated_tape", rows=len(frame))
        if mode == "krx":
            frame = _toss_frame(str(symbol), volume=100)
            return frame, _entry(str(symbol), CaptureStatus.COMPLETE, "ok", rows=len(frame))
        if mode == "missing":
            return pd.DataFrame(), _entry(str(symbol), CaptureStatus.NOT_APPLICABLE, "toss_stock_not_found")
        if mode == "shortfall":
            return pd.DataFrame(), _entry(str(symbol), CaptureStatus.NOT_APPLICABLE, "toss_volume_shortfall")
        if mode == "transport":
            return pd.DataFrame(), _entry(str(symbol), CaptureStatus.FAILED, "transport:boom")
        if mode == "malformed":
            return pd.DataFrame(), _entry(str(symbol), CaptureStatus.FAILED, "toss_malformed_page")
        raise AssertionError(f"unknown stub mode {mode!r}")


def _profile(**overrides) -> CollectionSettings:
    base: dict = {
        "COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS": (),
        "COLLECTION_TOSS_OUTAGE_MIN_SAMPLE": 5,
        "_env_file": None,
    }
    base.update(overrides)
    return CollectionSettings(**base)


def _recon(**overrides) -> NxtReconstructionSettings:
    base: dict = {"_env_file": None}
    base.update(overrides)
    return NxtReconstructionSettings(**base)


def _run(samples, stub, ledger, table, *, clock=None, stop_at=None, profile=None,
         recon=None, tmp_path, run_id="test-run"):
    store = CaptureStore(tmp_path / "ev")
    return asyncio.run(ncr.run_calibration_collection(
        samples_by_date=samples, client=object(), session=object(),
        profile=profile or _profile(), settings=recon or _recon(),
        ledger=ledger, table_path=table, evidence_store=store,
        run_id=run_id, clock=clock or (lambda: _NOW), stop_at=stop_at,
    ))


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    table = tmp_path / "table.parquet"
    stub = _StubAcquire({})
    monkeypatch.setattr("src.backfill.intraday.toss_regular.acquire_toss_regular_bars", stub)
    return tmp_path, ledger, table, stub


def _bind(ctx, script):
    tmp_path, ledger, table, stub = ctx
    stub.script.update(script)
    return tmp_path, ledger, table, stub


def test_incremental_run_touches_only_unresolved_candidates(ctx) -> None:
    """Terminal ledger rows are never refetched; FAILED rows retry."""
    tmp_path, ledger, table, stub = _bind(ctx, {})
    old = [_sample(_DAY1, "000001"), _sample(_DAY2, "000001")]
    asyncio.run(ncr.run_calibration_collection(
        samples_by_date={_DAY1: old[:1], _DAY2: old[1:]}, client=object(), session=object(),
        profile=_profile(), settings=_recon(), ledger=ledger, table_path=table,
        evidence_store=CaptureStore(tmp_path / "ev"), run_id="seed",
        clock=lambda: _NOW,
    ))
    assert stub.calls
    stub.calls.clear()
    ledger.record(
        _DAY3, ncr.CALIBRATION_SESSION,
        [_entry("000009", CaptureStatus.FAILED, "transport:boom")],
        run_id="seed", attempted_at=_NOW, vendor="toss",
    )
    samples = {
        _DAY1: [_sample(_DAY1, "000001")],
        _DAY2: [_sample(_DAY2, "000001")],
        _DAY3: [_sample(_DAY3, "000001"), _sample(_DAY3, "000009")],
    }
    summary = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert sorted(stub.calls) == [(_DAY3, "000001"), (_DAY3, "000009")]
    assert summary.dates_done == 3
    assert summary.paired == 2
    frame = pd.read_parquet(table)
    assert set(frame["date"].tolist()) == {_DAY1, _DAY2, _DAY3}


def test_rerun_is_idempotent(ctx) -> None:
    """A completed run reruns with zero fetches and byte-identical stores."""
    tmp_path, ledger, table, stub = ctx
    samples = {_DAY1: [_sample(_DAY1, "000001"), _sample(_DAY1, "000002")]}
    first = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert first.paired == 2
    table_bytes = table.read_bytes()
    ledger_bytes = (tmp_path / "ledger.parquet").read_bytes()
    stub.calls.clear()
    second = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert stub.calls == []
    assert second.paired == 0
    assert table.read_bytes() == table_bytes
    assert (tmp_path / "ledger.parquet").read_bytes() == ledger_bytes


def test_crash_between_table_and_ledger_repairs_without_duplicates(ctx) -> None:
    """Rows upserted without ledger rows are idempotent on rerun; the ledger completes."""
    tmp_path, ledger, table, stub = ctx
    pair = {
        "date": _DAY1, "symbol": "000001",
        "kis_bars": _kis_frame("000001"), "toss_bars": _toss_frame("000001"),
        "eod_volume": float(_EOD),
    }
    ncr._upsert_table_rows(table, [ncp.calibration_row_from_pair(pair)])
    summary = _run({_DAY1: [_sample(_DAY1, "000001")]}, stub, ledger, table, tmp_path=tmp_path)
    assert summary.paired == 1
    frame = pd.read_parquet(table)
    assert len(frame) == 1
    assert ledger.terminal_symbols(_DAY1, ncr.CALIBRATION_SESSION) == frozenset({"000001"})


def test_not_consolidated_and_identity_violations_are_remembered(ctx) -> None:
    """KRX-only and identity-violating days are NOT_APPLICABLE, absent, never refetched."""
    tmp_path, ledger, table, stub = _bind(ctx, {(_DAY1, "000001"): "krx", (_DAY1, "000002"): "pair"})
    samples = {
        _DAY1: [
            _sample(_DAY1, "000001"),
            _sample(_DAY1, "000002"),
            _sample(_DAY1, "000003", eod=float(_EOD) * 10.0),
        ],
    }
    summary = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert summary.paired == 1
    assert summary.not_applicable == 2
    frame = pd.read_parquet(table)
    assert frame["symbol"].tolist() == ["000002"]
    assert ledger.terminal_symbols(_DAY1, ncr.CALIBRATION_SESSION) == frozenset({"000001", "000002", "000003"})
    stub.calls.clear()
    again = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert stub.calls == []
    assert again.paired == 0


def test_bounded_memory_releases_completed_date_bars(ctx) -> None:
    """Bars of a committed date are not referenced by the run after its commit."""
    tmp_path, ledger, table, stub = ctx
    days = [f"2024-03-{d:02d}" for d in range(1, 7)]
    samples = {day: [_sample(day, "000001"), _sample(day, "000002")] for day in days}
    summary = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert summary.paired == 12
    assert len(stub.refs) == 12
    gc.collect()
    assert all(ref() is None for ref in stub.refs)
    for value in vars(summary).values():
        assert not isinstance(value, pd.DataFrame)


def test_lock_exclusion_reads_and_writes_nothing(tmp_path, monkeypatch) -> None:
    """A second instance under the Toss run lock raises before touching the table."""
    from src.utils.file_lock import exclusive_file_lock
    from src.backfill.intraday.toss_regular_backfill import toss_run_lock_path

    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    table = tmp_path / "table.parquet"
    with exclusive_file_lock(toss_run_lock_path(), timeout_seconds=0.0, purpose="test"), pytest.raises(TimeoutError):
        asyncio.run(ncr.run_calibration_collection(
            samples_by_date={_DAY1: [_sample(_DAY1, "000001")]},
            client=object(), session=object(), profile=_profile(), settings=_recon(),
            ledger=ledger, table_path=table, evidence_store=CaptureStore(tmp_path / "ev"),
            run_id="locked", clock=lambda: _NOW,
        ))
    assert not table.exists()
    assert not (tmp_path / "ledger.parquet").exists()


def test_main_lock_contention_exits_nonzero_without_touching_table(tmp_path, monkeypatch) -> None:
    """The CLI exits non-zero under lock contention and leaves stores absent."""
    from src.utils.file_lock import exclusive_file_lock
    from src.backfill.intraday.toss_regular_backfill import toss_run_lock_path

    _main_env(tmp_path, monkeypatch)
    def forbidden_client():
        raise AssertionError("Lock contention must be resolved before opening the client")

    monkeypatch.setattr(ncr, "_open_toss_client", forbidden_client)
    with exclusive_file_lock(toss_run_lock_path(), timeout_seconds=0.0, purpose="test"):
        code = ncr.main(["--start", _DAY1, "--end", _DAY3])
    assert code != 0
    assert not (tmp_path / ncr.CALIBRATION_TABLE_FILENAME).exists()
    assert not (tmp_path / "intraday" / "backfill_ledger" / ncr.CALIBRATION_LEDGER_FILENAME).exists()


def test_cli_holds_shared_lock_during_auth_collection_and_fit(tmp_path, monkeypatch) -> None:
    from src.utils.file_lock import exclusive_file_lock

    _main_env(tmp_path, monkeypatch)
    stages = []

    def assert_locked(stage):
        with pytest.raises(TimeoutError), exclusive_file_lock(
            ncr.toss_run_lock_path(), timeout_seconds=0, purpose="other-worker",
        ):
            raise AssertionError("Shared lock was released")
        stages.append(stage)

    class Client(_FakeClient):
        async def ensure_token(self, session):
            assert_locked("auth")

    stub = _StubAcquire({})

    async def acquire(*args, **kwargs):
        assert_locked("collection")
        return await stub(*args, **kwargs)

    def fit(table, recon):
        assert_locked("fit")
        assert len(pd.read_parquet(table)) == 1
        return 0

    monkeypatch.setattr(ncr, "_open_toss_client", Client)
    monkeypatch.setattr("src.backfill.intraday.toss_regular.acquire_toss_regular_bars", acquire)
    monkeypatch.setattr(ncr, "_run_fit", fit)
    assert ncr.main(["--as-of", "2025-04-10", "--start", "2025-03-05", "--end", "2025-03-05", "--fit"]) == 0
    assert stages == ["auth", "collection", "fit"]


def test_cli_auth_timeout_is_not_misreported_as_lock_contention(tmp_path, monkeypatch) -> None:
    _main_env(tmp_path, monkeypatch)

    class Client(_FakeClient):
        async def ensure_token(self, session):
            raise TimeoutError("Token request timed out")

    monkeypatch.setattr(ncr, "_open_toss_client", Client)
    with pytest.raises(TimeoutError, match="Token request"):
        ncr.main(["--as-of", "2025-04-10"])
    assert not ncr.toss_run_lock_path().exists()
    assert not ncr.default_calibration_table_path().exists()


def test_lazy_samples_release_both_vendors_before_next_partition(tmp_path, monkeypatch) -> None:
    from datetime import date

    _main_env(tmp_path, monkeypatch)
    wide, prepared, _ = ncr._load_history(date(2025, 4, 10))
    days = ["2025-03-05", "2025-03-06"]
    from src.data.intraday_store import intraday_partition_path

    second_path = intraday_partition_path(1, days[1], "regular")
    _kis_frame("000001").to_parquet(second_path)
    original_read = pd.read_parquet
    loaded = []
    kis_refs = []
    stub = _StubAcquire({})

    def read(path, **kwargs):
        if path == second_path:
            gc.collect()
            assert kis_refs and all(ref() is None for ref in kis_refs)
            assert stub.refs and all(ref() is None for ref in stub.refs)
        loaded.append(path)
        return original_read(path, **kwargs)

    original_derive = ncr.calibration_row_from_pair

    def derive(pair):
        kis_refs.append(weakref.ref(pair["kis_bars"]))
        return original_derive(pair)

    monkeypatch.setattr(pd, "read_parquet", read)
    samples = ncr._build_samples(as_of=date(2025, 4, 10), start=None, end=None,
                                 wide_history=wide, prepared=prepared, calendar=days)
    assert loaded == []
    assert list(samples) == days
    monkeypatch.setattr(ncr, "calibration_row_from_pair", derive)
    monkeypatch.setattr("src.backfill.intraday.toss_regular.acquire_toss_regular_bars", stub)
    summary = _run(samples, stub, ExtendedBackfillLedger(tmp_path / "lazy-ledger.parquet"),
                   tmp_path / "lazy-table.parquet", tmp_path=tmp_path)
    assert summary.paired == 2
    assert summary.table_rows == 2


def test_fit_digest_and_rows_share_one_snapshot(tmp_path, monkeypatch) -> None:
    import io

    from src.data.nxt_decomposition import load_fit_diagnostics
    from src.data.pit1520_panel import default_decomposition_config_path

    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    days = pd.bdate_range("2024-01-01", periods=12).strftime("%Y-%m-%d").tolist()
    table = tmp_path / "calib.parquet"
    _constant_table(days).to_parquet(table)
    expected_digest = hashlib.sha256(table.read_bytes()).hexdigest()
    original_read = pd.read_parquet

    def replace_during_parse(source, **kwargs):
        assert isinstance(source, io.BytesIO)
        _constant_table([days[0]]).to_parquet(table)
        return original_read(source, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", replace_during_parse)
    assert ncr._run_fit(table, _small_recon()) == 0
    diagnostics = load_fit_diagnostics(default_decomposition_config_path().parent / "nxt_decomposition_fit_report.json")
    assert diagnostics.table_sha256 == expected_digest
    assert diagnostics.n_fit_rows + diagnostics.n_holdout_rows == 24


def test_blackout_starts_no_date_until_window_ends_or_deadline(ctx) -> None:
    """Inside a blackout window no date starts; a deadline inside the window stops the run."""
    tmp_path, ledger, table, stub = ctx
    clock_time = datetime(2024, 3, 5, 9, 0, tzinfo=SEOUL)
    profile = _profile(COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=("0850-0940",))
    stop_at = datetime(2024, 3, 5, 9, 30, tzinfo=SEOUL)
    summary = _run(
        {_DAY1: [_sample(_DAY1, "000001")]}, stub, ledger, table,
        clock=lambda: clock_time, stop_at=stop_at, profile=profile, tmp_path=tmp_path,
    )
    assert stub.calls == []
    assert summary.stopped_by_deadline is True
    assert summary.dates_done == 0


def test_outage_abort_records_nothing(ctx) -> None:
    """A date failing by transport on most attempts aborts with nothing recorded."""
    tmp_path, ledger, table, stub = _bind(ctx, {
        (_DAY1, f"{i:06d}"): "transport" for i in range(1, 7)
    })
    samples = {_DAY1: [_sample(_DAY1, f"{i:06d}") for i in range(1, 7)]}
    summary = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert summary.outage_aborted is True
    assert summary.paired == 0
    assert ledger._read_all().empty
    assert not table.exists()


def test_non_raw_or_partial_kis_days_are_skipped_without_call_or_ledger(ctx) -> None:
    """Adjusted-basis or partial-session KIS bars cause no Toss call and no ledger row."""
    tmp_path, ledger, table, stub = ctx
    samples = {_DAY1: [
        _sample(_DAY1, "000001"),
        _sample(_DAY1, "000002", raw=False),
        _sample(_DAY1, "000003", partial=True),
    ]}
    summary = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert stub.calls == [(_DAY1, "000001")]
    assert summary.paired == 1
    frame = ledger._read_all()
    assert set(frame["symbol"].tolist()) == {"000001"}


def test_row_function_equals_the_table_path() -> None:
    """The row-wise derivation and build_calibration_table yield identical rows and exclusions."""
    pairs = pd.DataFrame([
        {"date": _DAY1, "symbol": "000001", "kis_bars": _kis_frame("000001"),
         "toss_bars": _toss_frame("000001"), "eod_volume": float(_EOD)},
        {"date": _DAY1, "symbol": "000002", "kis_bars": _kis_frame("000002"),
         "toss_bars": _toss_frame("000002"), "eod_volume": float(_EOD) * 10.0},
    ])
    rows = [ncp.calibration_row_from_pair(row) for _, row in pairs.iterrows()]
    table = ncp.build_calibration_table(pairs)
    assert table.attrs["n_identity_excluded"] == 1
    assert len(rows) == len(table) + int(table.attrs["n_identity_excluded"])
    kept = table.set_index("symbol").to_dict(orient="index")
    assert set(kept) == {"000001"}
    for row in rows:
        if row["symbol"] in kept:
            for key, value in kept[row["symbol"]].items():
                assert row[key] == value


def test_plan_only_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    """--plan-only prints candidates and estimates without touching table, ledger or lock."""
    from src.backfill.intraday.toss_regular_backfill import toss_run_lock_path

    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    samples = {_DAY1: [_sample(_DAY1, "000001"), _sample(_DAY1, "000002", raw=False)]}
    monkeypatch.setattr(ncr, "_load_history", lambda as_of: (pd.DataFrame(), pd.DataFrame(), []))
    monkeypatch.setattr(ncr, "_build_samples", lambda **kw: samples)
    code = ncr.main(["--plan-only"])
    assert code == 0
    out = capsys.readouterr().out
    assert "planned_symbol_days=1" in out
    assert "estimated_calls=2" in out
    assert not (tmp_path / ncr.CALIBRATION_TABLE_FILENAME).exists()
    assert not (tmp_path / "intraday").exists()
    assert not toss_run_lock_path().exists()


def test_failed_rows_retry_then_exhaust(ctx) -> None:
    """FAILED rows retry on later runs; the attempt cap retires them as EXHAUSTED."""
    tmp_path, ledger, table, stub = _bind(ctx, {(_DAY1, "000001"): "transport"})
    samples = {_DAY1: [_sample(_DAY1, "000001")]}
    profile = _profile(COLLECTION_TOSS_OUTAGE_MIN_SAMPLE=100)
    for _ in range(3):
        summary = _run(samples, stub, ledger, table, profile=profile, tmp_path=tmp_path)
        assert summary.outage_aborted is False
    assert len(stub.calls) == 3
    frame = ledger._read_all()
    assert frame["status"].tolist() == ["EXHAUSTED"]
    assert ledger.terminal_symbols(_DAY1, ncr.CALIBRATION_SESSION) == frozenset({"000001"})
    stub.calls.clear()
    stub.script[(_DAY1, "000001")] = "pair"
    summary = _run(samples, stub, ledger, table, profile=profile, tmp_path=tmp_path)
    assert stub.calls == []
    assert summary.paired == 0


def _constant_table(days: list[str], *, share: float = 0.85) -> pd.DataFrame:
    rows = []
    for day in days:
        for symbol in ("000001", "000002"):
            v_cons = 10000.0
            v_krx = share * v_cons
            auction = 365.0
            eod = v_krx + auction
            rows.append({
                "date": day, "symbol": symbol, "v_krx_1520": v_krx,
                "auction_volume": auction, "v_cons_1520": v_cons,
                "v_cons_full": v_cons + 500.0, "close_krx_1519": 50000.0,
                "close_cons_1519": 50000.0, "high_krx": 50100.0, "low_krx": 49900.0,
                "high_cons": 50100.0, "low_cons": 49900.0, "eod_volume": eod,
                "identity_residual": 0.0,
            })
    return pd.DataFrame(rows, columns=list(ncp.CALIBRATION_TABLE_COLUMNS))


def _small_recon() -> NxtReconstructionSettings:
    return _recon(
        NXT_RECON_ALPHAS=(0.3, 0.5, 0.7),
        NXT_RECON_STRUCTURES=((2, 30),),
        NXT_RECON_HOLDOUT_FRACTION=0.25,
        NXT_RECON_MIN_HOLDOUT_DAYS=2,
        NXT_RECON_MAX_SELECTION_REL_ERR_P90=0.5,
    )


def _seed_config_and_diagnostics(directory) -> tuple:
    from src.data.nxt_decomposition import (
        DECOMPOSITION_FIT_REPORT_FILENAME,
        DecompositionConfig,
        FitDiagnostics,
        FrontierPoint,
        save_decomposition_config,
        save_fit_diagnostics,
    )
    from src.data.pit1520_panel import default_decomposition_config_path

    config = DecompositionConfig(
        ewma_alpha=0.5, min_prior_days=2, max_gap_days=30,
        auction_fraction_mean=0.04, bias_correction=1.0,
        volume_rel_err_p90=0.1, close_bp_err_p90=5.0,
        calibrated_through="2024-02-01", fit_start="2024-01-10",
        holdout_start="2024-01-02", holdout_end="2024-01-05",
    )
    diagnostics = FitDiagnostics(
        fit_start="2024-01-10", fit_end="2024-02-01",
        holdout_start="2024-01-02", holdout_end="2024-01-05",
        n_fit_rows=10, n_holdout_rows=4, n_symbols=2,
        holdout_rel_err_p50=0.05, holdout_rel_err_p90=0.1, holdout_rel_err_p99=0.2,
        holdout_coverage=0.8, abar_fit=0.04, abar_holdout=0.041,
        bias_fit=1.0, bias_holdout=1.01, alpha_at_boundary=False,
        frontier=(FrontierPoint(0.5, 2, 30, 0.8, 0.05, 0.1, 0.2),),
        table_sha256="seed",
    )
    config_path = default_decomposition_config_path()
    save_decomposition_config(config, config_path)
    save_fit_diagnostics(diagnostics, config_path.parent / DECOMPOSITION_FIT_REPORT_FILENAME)
    return config_path, config_path.parent / DECOMPOSITION_FIT_REPORT_FILENAME


def test_fit_failure_leaves_previous_config_and_diagnostics_untouched(tmp_path, monkeypatch) -> None:
    """A fit that raises never rewrites the previous config or diagnostics."""
    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    config_path, diagnostics_path = _seed_config_and_diagnostics(tmp_path)
    before_config = config_path.read_bytes()
    before_diagnostics = diagnostics_path.read_bytes()
    table = tmp_path / "tiny.parquet"
    _constant_table(["2024-01-02"]).to_parquet(table)
    with pytest.raises(ValueError, match="min_holdout_days"):
        ncr._run_fit(table, _small_recon())
    assert config_path.read_bytes() == before_config
    assert diagnostics_path.read_bytes() == before_diagnostics


def test_fit_success_orders_diagnostics_before_config_and_matches_digest(tmp_path, monkeypatch) -> None:
    """On success the config holdout window equals the diagnostics' and the table digest matches."""
    import json

    from src.data.nxt_decomposition import load_decomposition_config, load_fit_diagnostics

    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    days = pd.bdate_range("2024-01-01", periods=12).strftime("%Y-%m-%d").tolist()
    table = tmp_path / "calib.parquet"
    _constant_table(days).to_parquet(table)
    assert ncr._run_fit(table, _small_recon()) == 0
    from src.data.pit1520_panel import default_decomposition_config_path

    config_path = default_decomposition_config_path()
    diagnostics_path = config_path.parent / "nxt_decomposition_fit_report.json"
    config = load_decomposition_config(config_path)
    diagnostics = load_fit_diagnostics(diagnostics_path)
    assert (config.fit_start, config.holdout_start, config.holdout_end) == (
        diagnostics.fit_start, diagnostics.holdout_start, diagnostics.holdout_end,
    )
    assert diagnostics.table_sha256 == hashlib.sha256(table.read_bytes()).hexdigest()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["table_sha256"] == diagnostics.table_sha256


def test_collection_rejects_bad_contracts(ctx) -> None:
    """Empty run_id, missing client/session/stores and naive deadlines fail closed."""
    import datetime as dt

    tmp_path, ledger, table, _stub = ctx
    store = CaptureStore(tmp_path / "ev")
    kw = {
        "samples_by_date": {}, "client": object(), "session": object(),
        "profile": _profile(), "settings": _recon(), "ledger": ledger,
        "table_path": table, "evidence_store": store, "run_id": "ok",
        "clock": lambda: _NOW,
    }
    with pytest.raises(ValueError, match="run_id"):
        asyncio.run(ncr.run_calibration_collection(**{**kw, "run_id": "  "}))
    with pytest.raises(ValueError, match="client and session"):
        asyncio.run(ncr.run_calibration_collection(**{**kw, "client": None}))
    with pytest.raises(ValueError, match="ledger, evidence_store"):
        asyncio.run(ncr.run_calibration_collection(**{**kw, "ledger": None}))
    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(ncr.run_calibration_collection(
            **{**kw, "stop_at": dt.datetime(2024, 3, 1, 12, 0)},
        ))


def test_deadline_stop_starts_no_date(ctx) -> None:
    """A stop_at at or before now stops the run before the first date."""
    tmp_path, ledger, table, stub = ctx
    stop_at = datetime(2024, 3, 1, 12, 0, tzinfo=SEOUL)
    summary = _run(
        {_DAY1: [_sample(_DAY1, "000001")]}, stub, ledger, table,
        clock=lambda: _NOW, stop_at=stop_at, tmp_path=tmp_path,
    )
    assert stub.calls == []
    assert summary.stopped_by_deadline is True
    assert summary.dates_done == 0
    assert summary.dates_remaining == 1


def test_blackout_wait_then_proceeds_or_stops(ctx, monkeypatch) -> None:
    """After a blackout wait the run proceeds, or stops when the deadline passed."""
    tmp_path, ledger, table, stub = ctx
    profile = _profile(COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=("0850-0940",))
    clock_time = datetime(2024, 3, 5, 9, 0, tzinfo=SEOUL)
    now = {"at": clock_time}

    async def _fake_wait(windows, *, now_fn=None, sleep_fn=None, weekdays_only=True):
        now["at"] = datetime(2024, 3, 5, 10, 0, tzinfo=SEOUL)

    monkeypatch.setattr(ncr, "wait_for_blackout", _fake_wait)
    samples = {_DAY1: [_sample(_DAY1, "000001")]}
    summary = _run(
        samples, stub, ledger, table, clock=lambda: now["at"],
        profile=profile, tmp_path=tmp_path,
    )
    assert stub.calls == [(_DAY1, "000001")]
    assert summary.stopped_by_deadline is False
    now["at"] = clock_time
    stub.calls.clear()
    ledger2 = ExtendedBackfillLedger(tmp_path / "ledger2.parquet")
    summary = _run(
        samples, stub, ledger2, tmp_path / "table2.parquet",
        clock=lambda: now["at"], stop_at=datetime(2024, 3, 5, 9, 50, tzinfo=SEOUL),
        profile=profile, tmp_path=tmp_path,
    )
    assert stub.calls == []
    assert summary.stopped_by_deadline is True


def test_gate_rejections_without_consolidation_are_terminal(ctx) -> None:
    """Missing or shortfall verdicts are NOT_APPLICABLE and never refetched."""
    tmp_path, ledger, table, stub = _bind(ctx, {
        (_DAY1, "000001"): "missing", (_DAY1, "000002"): "shortfall",
    })
    samples = {_DAY1: [_sample(_DAY1, "000001"), _sample(_DAY1, "000002")]}
    summary = _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert summary.not_applicable == 2
    assert summary.paired == 0
    assert not table.exists()
    assert ledger.terminal_symbols(_DAY1, ncr.CALIBRATION_SESSION) == frozenset({"000001", "000002"})
    stub.calls.clear()
    _run(samples, stub, ledger, table, tmp_path=tmp_path)
    assert stub.calls == []


def test_upsert_rejects_a_foreign_table(ctx) -> None:
    """A table lacking calibration columns fails closed instead of being destroyed."""
    tmp_path, _ledger, table, _stub = ctx
    pd.DataFrame({"date": [_DAY1]}).to_parquet(table)
    before = table.read_bytes()
    with pytest.raises(OSError, match="lacks columns"):
        ncr._upsert_table_rows(table, [{"date": _DAY1, "symbol": "000001"}])
    assert table.read_bytes() == before


def test_http_session_shapes() -> None:
    """The session helper serves bare fakes, factories and context-manager factories."""

    class _Bare:
        pass

    class _Factory:
        def create_session(self):
            return _Bare()

    class _Context:
        def __init__(self):
            self.entered = False

        async def __aenter__(self):
            self.entered = True
            return _Bare()

        async def __aexit__(self, *args):
            return False

    class _ContextFactory:
        def __init__(self):
            self.produced = _Context()

        def create_session(self):
            return self.produced

    async def _main() -> None:
        import aiohttp

        async with ncr._http_session(_Bare()) as first:
            assert isinstance(first, aiohttp.ClientSession)
            assert not first.closed
        assert first.closed
        factory = _Factory()
        async with ncr._http_session(factory) as second:
            assert isinstance(second, _Bare)
        ctx_factory = _ContextFactory()
        async with ncr._http_session(ctx_factory) as third:
            assert isinstance(third, _Bare)
        assert ctx_factory.produced.entered is True

    asyncio.run(_main())


def test_open_toss_client_requires_credentials(monkeypatch) -> None:
    """Missing Toss credentials fail closed; configured ones open a client."""
    from src.config import settings as config_singleton

    monkeypatch.setattr(config_singleton, "TOSS_APP_KEY", "", raising=False)
    monkeypatch.setattr(config_singleton, "TOSS_APP_SECRET", "", raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        ncr._open_toss_client()
    monkeypatch.setattr(config_singleton, "TOSS_APP_KEY", "key", raising=False)
    monkeypatch.setattr(config_singleton, "TOSS_APP_SECRET", "secret", raising=False)
    client = ncr._open_toss_client()
    assert client.app_key == "key"


def _history_harness(tmp_path):
    import pandas as pd

    from src.data.intraday_store import intraday_partition_path

    days = ["2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-04-15"]
    rows = []
    for day in days:
        for symbol, volume in (("000001", 39000.0), ("000002", 0.0)):
            rows.append({
                "date": day, "symbol": symbol, "open": 70000.0, "high": 70100.0,
                "low": 69900.0, "close": 70000.0, "prev_close": 69900.0,
                "volume": volume, "market_cap_100m": 8000.0, "trade_value_100m": 250.0,
                "close_raw": 70000.0, "market": "KOSPI",
            })
    wide = pd.DataFrame(rows)
    (tmp_path / "price_history.parquet").parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(tmp_path / "price_history.parquet")
    ls_only = _kis_frame("000001")
    ls_only["vendor"] = "ls"
    target_ls = intraday_partition_path(1, "2025-03-04", "regular")
    target_ls.parent.mkdir(parents=True, exist_ok=True)
    ls_only.reset_index(drop=True).to_parquet(target_ls)
    kis = pd.concat([_kis_frame("000001"), _kis_frame("000002")], ignore_index=True)
    target = intraday_partition_path(1, "2025-03-05", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    kis.to_parquet(target)
    return days


def _main_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(
        ncr.app_settings, "PRICE_HISTORY_PARQUET_PATH",
        tmp_path / "price_history.parquet", raising=False,
    )
    _history_harness(tmp_path)


class _FakeClient:
    def __init__(self) -> None:
        self.token_calls = 0

    async def ensure_token(self, session) -> None:
        self.token_calls += 1


def test_main_collects_through_history_and_partitions(tmp_path, monkeypatch) -> None:
    """The CLI builds samples from stored history and collects one symbol-day."""
    _main_env(tmp_path, monkeypatch)
    monkeypatch.setattr("src.backfill.intraday.toss_regular.acquire_toss_regular_bars", _StubAcquire({}))
    monkeypatch.setattr(ncr, "_open_toss_client", lambda: _FakeClient())
    code = ncr.main(["--as-of", "2025-04-10", "--start", "2025-03-03", "--end", "2025-03-06"])
    assert code == 0
    table = pd.read_parquet(tmp_path / ncr.CALIBRATION_TABLE_FILENAME)
    assert set(zip(table["date"].tolist(), table["symbol"].tolist(), strict=True)) == {("2025-03-05", "000001")}


def test_main_fit_failure_exits_nonzero_with_table_kept(tmp_path, monkeypatch) -> None:
    """--fit on an unfittable table exits non-zero; the collected rows stay."""
    _main_env(tmp_path, monkeypatch)
    monkeypatch.setattr("src.backfill.intraday.toss_regular.acquire_toss_regular_bars", _StubAcquire({}))
    monkeypatch.setattr(ncr, "_open_toss_client", lambda: _FakeClient())
    code = ncr.main(["--as-of", "2025-04-10", "--start", "2025-03-05", "--end", "2025-03-05", "--fit"])
    assert code == 1
    table = pd.read_parquet(tmp_path / ncr.CALIBRATION_TABLE_FILENAME)
    assert len(table) == 1


def test_main_outage_abort_exits_nonzero(tmp_path, monkeypatch) -> None:
    """A vendor outage during the CLI run exits non-zero with nothing recorded."""
    _main_env(tmp_path, monkeypatch)
    script = {("2025-03-05", f"{i:06d}"): "transport" for i in range(1, 8)}
    kis = pd.concat([_kis_frame(f"{i:06d}") for i in range(1, 8)], ignore_index=True)
    from src.data.intraday_store import intraday_partition_path

    target = intraday_partition_path(1, "2025-03-05", "regular")
    kis.to_parquet(target)
    extra = pd.DataFrame([{
        "date": "2025-03-05", "symbol": f"{i:06d}", "open": 70000.0, "high": 70100.0,
        "low": 69900.0, "close": 70000.0, "prev_close": 69900.0,
        "volume": 39000.0, "market_cap_100m": 8000.0, "trade_value_100m": 250.0,
        "close_raw": 70000.0, "market": "KOSPI",
    } for i in range(3, 8)])
    wide = pd.read_parquet(tmp_path / "price_history.parquet")
    pd.concat([wide, extra], ignore_index=True).to_parquet(tmp_path / "price_history.parquet")
    monkeypatch.setattr(
        "src.backfill.intraday.toss_regular.acquire_toss_regular_bars", _StubAcquire(script),
    )
    monkeypatch.setattr(ncr, "_open_toss_client", lambda: _FakeClient())
    code = ncr.main(["--as-of", "2025-04-10", "--start", "2025-03-05", "--end", "2025-03-05"])
    assert code == 1
    assert not (tmp_path / ncr.CALIBRATION_TABLE_FILENAME).exists()


def test_main_rejects_bad_range_and_stop_at(tmp_path, monkeypatch) -> None:
    """Inverted ranges and malformed --stop-at fail closed with distinct messages."""
    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    with pytest.raises(ValueError, match="Invalid plan range"):
        ncr.main(["--start", "2025-03-10", "--end", "2025-03-01"])
    with pytest.raises(ValueError, match="Invalid --stop-at"):
        ncr.main(["--stop-at", "nope"])
    _main_env(tmp_path, monkeypatch)
    monkeypatch.setattr(ncr, "_load_history", lambda as_of: (pd.DataFrame(), pd.DataFrame(), []))
    monkeypatch.setattr(ncr, "_build_samples", lambda **kw: {})
    assert ncr.main(["--plan-only", "--stop-at", "000000"]) == 0


def test_plan_only_skips_terminal_candidates(tmp_path, monkeypatch, capsys) -> None:
    """--plan-only excludes ledger-terminal symbol-days from the estimate."""
    monkeypatch.setattr(ncr.app_settings, "HISTORY_DIR", tmp_path, raising=False)
    samples = {_DAY1: [_sample(_DAY1, "000001"), _sample(_DAY1, "000002")]}
    monkeypatch.setattr(ncr, "_load_history", lambda as_of: (pd.DataFrame(), pd.DataFrame(), []))
    monkeypatch.setattr(ncr, "_build_samples", lambda **kw: samples)
    ledger = ExtendedBackfillLedger(ncr.default_calibration_ledger_path())
    ledger.record(
        _DAY1, ncr.CALIBRATION_SESSION,
        [_entry("000001", CaptureStatus.NOT_APPLICABLE, "toss_stock_not_found")],
        run_id="seed", attempted_at=_NOW, vendor="toss",
    )
    assert ncr.main(["--plan-only"]) == 0
    assert "planned_symbol_days=1" in capsys.readouterr().out
