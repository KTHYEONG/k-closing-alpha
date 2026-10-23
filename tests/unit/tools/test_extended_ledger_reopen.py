"""Invariant guards for the EXHAUSTED reopen tool and the exhaustion audit detail."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from src.backfill.intraday.extended_session_backfill import (
    EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS,
    _LEDGER_COLUMNS,
    ExtendedBackfillLedger,
    ReopenSelector,
)
from src.data.capture_contracts import SEOUL, CaptureDataset, CaptureStatus, CoverageEntry
from src.data.io_utils import atomic_write_parquet

AS_OF = date(2026, 9, 14)
RETENTION = 30
SESSION = "nxt_aftermarket"


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(_LEDGER_COLUMNS))


def _row(day: str, session: str, symbol: str, status: str, reason: str, attempted: str, attempts: int = 3) -> dict:
    return {
        "snapshot_date": day,
        "session": session,
        "symbol": symbol,
        "status": status,
        "rows": 0,
        "reason": reason,
        "run_id": "run-seed",
        "attempted_at": attempted,
        "vendor": "kis",
        "price_basis": "kis_raw",
        "attempts": attempts,
    }


def _seed(path, rows: list[dict]) -> ExtendedBackfillLedger:
    atomic_write_parquet(_frame(rows), path)
    return ExtendedBackfillLedger(path)


def _exhausted(day: str, symbol: str, reason: str, attempted: str, session: str = SESSION) -> dict:
    return _row(day, session, symbol, "EXHAUSTED", f"exhausted:{reason}", attempted)


def _selector(**kw) -> ReopenSelector:
    base: dict = {"sessions": frozenset(), "symbols": frozenset()}
    base.update(kw)
    return ReopenSelector(**base)


def _run_id() -> str:
    return "reopen-testrun"


def _at() -> datetime:
    return datetime(2026, 9, 14, 23, 30, tzinfo=SEOUL)


def test_dry_run_leaves_file_byte_identical(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    before = path.read_bytes()
    plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=False, run_id=_run_id(), attempted_at=_at())
    assert plan.matched == 1 and plan.reopened == 0 and plan.backup_path is None
    assert path.read_bytes() == before
    assert [entry.name for entry in tmp_path.iterdir() if not entry.name.endswith(".lock")] == ["ledger.parquet"]


def test_apply_rewrites_only_in_retention(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
            _exhausted("2026-07-01", "000660", "price_basis_raw_basis_unavailable", "2026-07-02T00:10:00+09:00"),
        ],
    )
    plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id=_run_id(), attempted_at=_at())
    assert (plan.matched, plan.skipped_out_of_retention, plan.reopened) == (1, 1, 1)
    assert plan.keys == (("2026-09-10", SESSION, "005930"),)
    frame = pd.read_parquet(path)
    latest = frame.drop_duplicates(subset=["snapshot_date", "session", "symbol"], keep="last")
    a = latest[latest["symbol"] == "005930"].iloc[0]
    assert a["status"] == "FAILED" and int(a["attempts"]) == EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS - 1
    assert str(a["reason"]) == "reopened:exhausted:price_basis_raw_basis_unavailable"
    b = latest[latest["symbol"] == "000660"].iloc[0]
    assert b["status"] == "EXHAUSTED"


def test_retention_boundary(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(
        path,
        [
            _exhausted("2026-08-15", "005930", "price_basis_raw_basis_unavailable", "2026-08-16T00:10:00+09:00"),
            _exhausted("2026-08-14", "000660", "price_basis_raw_basis_unavailable", "2026-08-15T00:10:00+09:00"),
            _exhausted("2026-09-14", "000270", "price_basis_raw_basis_unavailable", "2026-09-14T00:10:00+09:00"),
        ],
    )
    plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=False, run_id=_run_id(), attempted_at=_at())
    assert plan.matched == 1 and plan.skipped_out_of_retention == 2
    assert plan.keys == (("2026-08-15", SESSION, "005930"),)


def test_backup_precedes_write_and_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
            _exhausted("2026-09-10", "000660", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
        ],
    )
    before = path.read_bytes()
    only_first = _selector(symbols=frozenset({"005930"}))
    plan = ledger.reopen_exhausted(only_first, as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    assert plan.matched == 1 and plan.backup_path is not None and plan.backup_path.suffix == ".bak"
    assert plan.backup_path.read_bytes() == before
    assert path.read_bytes() != before
    ledger2 = ExtendedBackfillLedger(path)
    with pytest.raises(FileExistsError):
        ledger2.reopen_exhausted(_selector(symbols=frozenset({"000660"})), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    empty_path = tmp_path / "empty.parquet"
    _seed(empty_path, [_row("2026-09-10", SESSION, "005930", "COMPLETE", "", "2026-09-11T00:10:00+09:00", 0)])
    plan2 = ExtendedBackfillLedger(empty_path).reopen_exhausted(
        _selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r2", attempted_at=_at()
    )
    assert plan2.matched == 0 and plan2.backup_path is None
    assert list(tmp_path.glob("empty*.bak")) == []


def test_idempotent_second_apply(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    after = path.read_bytes()
    plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    assert plan.matched == 0 and path.read_bytes() == after


def _entry(symbol: str, status: CaptureStatus, reason: str) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.MINUTE_BARS,
        venue="NXT",
        session=SESSION,
        scheduled_at=None,
        status=status,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=(),
    )


def test_reopened_key_retried_then_reexhausts(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    now = datetime(2026, 9, 15, 1, 0, tzinfo=SEOUL)
    turned = ledger.record(
        "2026-09-10", SESSION, [_entry("005930", CaptureStatus.FAILED, "price_basis_raw_basis_unavailable")],
        run_id="r2", attempted_at=now, is_transient=None,
    )
    assert turned.newly_exhausted == 1
    assert turned.newly_exhausted_actionable == 0
    latest = pd.read_parquet(path).drop_duplicates(subset=["snapshot_date", "session", "symbol"], keep="last").iloc[0]
    assert latest["status"] == "EXHAUSTED" and int(latest["attempts"]) == EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS


def test_reopened_key_transient_stays_failed_and_complete_resets(tmp_path) -> None:
    from src.backfill.intraday.extended_session_backfill import is_transient_failure

    for reason, check in (
        ("transport:RuntimeError", lambda s, a: (s == "FAILED", a == EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS - 1)),
        ("", None),
    ):
        path = tmp_path / f"ledger_{len(reason)}.parquet"
        ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
        ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
        now = datetime(2026, 9, 15, 1, 0, tzinfo=SEOUL)
        if reason:
            ledger.record(
                "2026-09-10", SESSION, [_entry("005930", CaptureStatus.FAILED, reason)],
                run_id="r2", attempted_at=now, is_transient=is_transient_failure,
            )
            latest = pd.read_parquet(path).drop_duplicates(subset=["snapshot_date", "session", "symbol"], keep="last").iloc[0]
            ok_status, ok_attempts = check(latest["status"], int(latest["attempts"]))
            assert ok_status and ok_attempts
        else:
            ledger.record(
                "2026-09-10", SESSION, [_entry("005930", CaptureStatus.COMPLETE, "")],
                run_id="r2", attempted_at=now,
            )
            latest = pd.read_parquet(path).drop_duplicates(subset=["snapshot_date", "session", "symbol"], keep="last").iloc[0]
            assert latest["status"] == "COMPLETE" and int(latest["attempts"]) == 0


def test_reopened_key_not_terminal(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    assert "005930" in ledger.terminal_symbols("2026-09-10", SESSION)
    ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    assert "005930" not in ledger.terminal_symbols("2026-09-10", SESSION)


def test_filters_combined(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
            _exhausted("2026-09-10", "000660", "price_basis_no_raw_source", "2026-09-11T00:10:00+09:00", session="krx_aftermarket"),
            _exhausted("2026-09-09", "005930", "price_basis_no_raw_source", "2026-09-11T22:00:00+09:00"),
        ],
    )
    kw = {"as_of": AS_OF, "retention_days": RETENTION, "apply": False, "run_id": _run_id(), "attempted_at": _at()}
    assert ledger.reopen_exhausted(_selector(sessions=frozenset({"krx_aftermarket"})), **kw).keys == (
        ("2026-09-10", "krx_aftermarket", "000660"),
    )
    assert ledger.reopen_exhausted(_selector(date_from="2026-09-10", date_to="2026-09-10"), **kw).matched == 2
    prefixed = ledger.reopen_exhausted(_selector(reason_prefix="price_basis_raw_basis"), **kw)
    assert prefixed.keys == (("2026-09-10", SESSION, "005930"),)
    by_symbol = ledger.reopen_exhausted(_selector(symbols=frozenset({"005930"})), **kw)
    assert by_symbol.matched == 2
    since = ledger.reopen_exhausted(
        _selector(attempted_since=datetime(2026, 9, 11, 12, 0, tzinfo=SEOUL)), **kw
    )
    assert since.keys == (("2026-09-09", SESSION, "005930"),)
    both = ledger.reopen_exhausted(
        _selector(sessions=frozenset({SESSION}), symbols=frozenset({"005930"}), reason_prefix="price_basis_no_raw"), **kw
    )
    assert both.keys == (("2026-09-09", SESSION, "005930"),)


def test_dry_run_missing_ledger_is_empty(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "absent.parquet")
    plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=False, run_id="r", attempted_at=_at())
    assert plan == ExtendedBackfillLedger(tmp_path / "absent.parquet").reopen_exhausted(
        _selector(), as_of=AS_OF, retention_days=RETENTION, apply=False, run_id="r", attempted_at=_at()
    )
    assert (plan.matched, plan.skipped_out_of_retention, plan.reopened) == (0, 0, 0)
    assert plan.keys == () and plan.backup_path is None
    assert not (tmp_path / "absent.parquet").exists()


def test_unparseable_and_naive_attempted_at_never_match_filter(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(
        path,
        [
            _row("2026-09-10", SESSION, "005930", "EXHAUSTED", "exhausted:price_basis_raw_basis_unavailable", "garbage"),
            _row("2026-09-10", SESSION, "000660", "EXHAUSTED", "exhausted:price_basis_raw_basis_unavailable", "2026-09-12T00:10:00"),
            _row("2026-09-10", SESSION, "000270", "EXHAUSTED", "exhausted:price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
        ],
    )
    plan = ledger.reopen_exhausted(
        _selector(attempted_since=datetime(2026, 9, 11, 0, 0, tzinfo=SEOUL)),
        as_of=AS_OF, retention_days=RETENTION, apply=False, run_id="r", attempted_at=_at(),
    )
    assert plan.keys == (("2026-09-10", SESSION, "000270"),)


def test_date_to_filter_excludes_later_keys(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
            _exhausted("2026-09-09", "000660", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
        ],
    )
    plan = ledger.reopen_exhausted(
        _selector(date_to="2026-09-09"), as_of=AS_OF, retention_days=RETENTION, apply=False, run_id="r", attempted_at=_at()
    )
    assert plan.keys == (("2026-09-09", SESSION, "000660"),)


def test_backup_size_mismatch_writes_nothing(tmp_path, monkeypatch) -> None:
    from src.backfill.intraday import extended_session_backfill as mod

    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    before = path.read_bytes()
    def _short(source, target):
        target.write(source.read(1))

    monkeypatch.setattr(mod.shutil, "copyfileobj", _short)
    with pytest.raises(OSError, match="size mismatch"):
        ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    assert path.read_bytes() == before


def test_cli_argument_errors_and_aware_instant(tmp_path, capsys) -> None:
    from src.tools import extended_ledger_reopen

    path = tmp_path / "ledger.parquet"
    _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    with pytest.raises(SystemExit) as exc:
        extended_ledger_reopen.main(["--ledger", str(path), "--date-from", "bogus"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        extended_ledger_reopen.main(["--ledger", str(path), "--attempted-since", "bogus"])
    assert exc.value.code == 2
    assert extended_ledger_reopen.main(["--ledger", str(path), "--date-from", "2026-09-10", "--date-to", "2026-09-09"]) == 2
    assert extended_ledger_reopen.main(
        ["--ledger", str(path), "--attempted-since", "2026-09-10T00:00:00+09:00", "--as-of", "2026-09-14"]
    ) == 0
    assert "matched=1" in capsys.readouterr().out


def test_cli_empty_prefix_cannot_authorize_apply(tmp_path) -> None:
    from src.tools.extended_ledger_reopen import main

    path = tmp_path / "ledger.parquet"
    _seed(path, [_exhausted("2026-09-10", "005930", "vendor_failure", "2026-09-12T00:10:00+09:00")])
    before = path.read_bytes()
    assert main(["--ledger", str(path), "--reason-prefix", "", "--apply"]) == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", ["20260914", "2026-W38-1"])
def test_cli_rejects_non_calendar_date_spelling(value) -> None:
    from src.tools.extended_ledger_reopen import main

    with pytest.raises(SystemExit) as exc:
        main(["--as-of", value])
    assert exc.value.code == 2


def test_existing_backup_symlink_is_not_followed(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "vendor_failure", "2026-09-12T00:10:00+09:00")])
    backup = tmp_path / "ledger.parquet.pre-reopen-r1.bak"
    target = tmp_path / "absent-target"
    backup.symlink_to(target)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    assert not target.exists() and path.read_bytes() == before


def test_apply_reselects_after_intervening_record(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "vendor_failure", "2026-09-12T00:10:00+09:00")])
    options = {"as_of": AS_OF, "retention_days": RETENTION, "run_id": "r1", "attempted_at": _at()}
    assert ledger.reopen_exhausted(_selector(), apply=False, **options).matched == 1
    ledger.record("2026-09-10", SESSION, [_entry("005930", CaptureStatus.COMPLETE, "")], run_id="nightly", attempted_at=_at())
    before = path.read_bytes()
    assert ledger.reopen_exhausted(_selector(), apply=True, **options).matched == 0
    assert path.read_bytes() == before and not list(tmp_path.glob("*.bak"))


def test_latest_only_rewrite_preserves_history_and_unselected_rows(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    older = _exhausted("2026-09-10", "005930", "vendor_failure", "2026-09-11T00:10:00+09:00")
    newest = dict(older, run_id="newest")
    resolved = _row("2026-09-10", SESSION, "000660", "COMPLETE", "", "2026-09-12T00:10:00+09:00", 0)
    ledger = _seed(path, [older, dict(older, symbol="000660"), newest, resolved])
    before = pd.read_parquet(path)
    plan = ledger.reopen_exhausted(_selector(date_from="", date_to=""), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    after = pd.read_parquet(path)
    assert plan.matched == 1 and len(after) == len(before)
    pd.testing.assert_frame_equal(after.iloc[[0, 1, 3]], before.iloc[[0, 1, 3]])
    assert after.iloc[2]["status"] == "FAILED"


def test_naive_attempted_since_read_as_kst_via_cli(tmp_path) -> None:
    from src.tools import extended_ledger_reopen

    path = tmp_path / "ledger.parquet"
    _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    assert extended_ledger_reopen.main(["--ledger", str(path), "--attempted-since", "2026-09-10", "--as-of", "2026-09-14"]) == 0


def test_only_exhausted_touched_and_columns_preserved(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    rows = [
        _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
        _row("2026-09-10", SESSION, "000660", "FAILED", "transport:boom", "2026-09-11T00:10:00+09:00", 1),
        _row("2026-09-10", SESSION, "000270", "COMPLETE", "", "2026-09-11T00:10:00+09:00", 0),
        _row("2026-09-10", SESSION, "000020", "NO_TRADES", "no_trades_in_window", "2026-09-11T00:10:00+09:00", 0),
        _row("2026-09-10", SESSION, "000010", "NOT_APPLICABLE", "toss_stock_not_found", "2026-09-11T00:10:00+09:00", 0),
    ]
    ledger = _seed(path, rows)
    plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    assert plan.matched == 1
    frame = pd.read_parquet(path)
    assert list(frame.columns) == list(_LEDGER_COLUMNS)
    by_symbol = {str(r["symbol"]): r for r in frame.to_dict(orient="records")}
    assert by_symbol["000660"]["status"] == "FAILED" and str(by_symbol["000660"]["reason"]) == "transport:boom"
    assert by_symbol["000270"]["status"] == "COMPLETE" and by_symbol["000020"]["status"] == "NO_TRADES"
    assert by_symbol["000010"]["status"] == "NOT_APPLICABLE"
    assert by_symbol["005930"]["vendor"] == "kis" and by_symbol["005930"]["price_basis"] == "kis_raw"


def test_legacy_ledger_reopened(tmp_path) -> None:
    path = tmp_path / "legacy.parquet"
    legacy = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-09-10",
                "session": SESSION,
                "symbol": "005930",
                "status": "EXHAUSTED",
                "rows": 0,
                "reason": "exhausted:price_basis_raw_basis_unavailable",
                "run_id": "run-seed",
                "attempted_at": "2026-09-11T00:10:00+09:00",
            }
        ]
    )
    legacy.to_parquet(path, index=False)
    ledger = ExtendedBackfillLedger(path)
    plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
    assert plan.matched == 1
    frame = pd.read_parquet(path)
    assert list(frame.columns) == list(_LEDGER_COLUMNS)
    assert int(frame.iloc[0]["attempts"]) == EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS - 1


def test_cli_unfiltered_apply_refused_and_dry_run_ok(tmp_path, capsys) -> None:
    from src.tools import extended_ledger_reopen

    path = tmp_path / "ledger.parquet"
    before_seed = [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")]
    _seed(path, before_seed)
    before = path.read_bytes()
    assert extended_ledger_reopen.main(["--ledger", str(path), "--apply", "--as-of", "2026-09-14"]) == 2
    assert path.read_bytes() == before
    assert extended_ledger_reopen.main(["--ledger", str(path), "--as-of", "2026-09-14"]) == 0
    assert path.read_bytes() == before
    out = capsys.readouterr().out
    assert "matched=1" in out


def test_cli_no_vendor_access_and_missing_and_corrupt(tmp_path, monkeypatch) -> None:
    import builtins

    from src.tools import extended_ledger_reopen

    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name.startswith(("src.api.kis", "src.api.kiwoom")) or name == "aiohttp":
            raise AssertionError(f"vendor import blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)
    path = tmp_path / "ledger.parquet"
    _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    assert extended_ledger_reopen.main(["--ledger", str(path), "--session", SESSION, "--as-of", "2026-09-14"]) == 0
    assert extended_ledger_reopen.main(["--ledger", str(path), "--session", SESSION, "--apply", "--as-of", "2026-09-14"]) == 0
    assert extended_ledger_reopen.main(["--ledger", str(tmp_path / "absent.parquet"), "--apply", "--session", SESSION]) == 0
    assert not (tmp_path / "absent.parquet").exists()
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not parquet")
    with pytest.raises(OSError, match="could not be read"):
        extended_ledger_reopen.main(["--ledger", str(corrupt), "--session", SESSION, "--apply"])
    assert corrupt.read_bytes() == b"not parquet"


def test_lock_honored_and_dry_run_without_wait(tmp_path, monkeypatch) -> None:
    from src.backfill.intraday import extended_session_backfill as mod
    from src.utils.file_lock import exclusive_file_lock, sidecar_lock_path

    path = tmp_path / "ledger.parquet"
    _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    monkeypatch.setattr(mod, "DEFAULT_LOCK_TIMEOUT_SECONDS", 0.05)
    ledger = ExtendedBackfillLedger(path)
    with exclusive_file_lock(sidecar_lock_path(path), timeout_seconds=5.0, purpose="test"):
        with pytest.raises(TimeoutError):
            ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=True, run_id="r1", attempted_at=_at())
        plan = ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=RETENTION, apply=False, run_id="r1", attempted_at=_at())
        assert plan.matched == 1


def test_naive_attempted_at_rejected(tmp_path) -> None:
    from datetime import datetime as _dt

    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.reopen_exhausted(
            _selector(), as_of=AS_OF, retention_days=RETENTION, apply=False, run_id="r", attempted_at=_dt(2026, 9, 14, 12, 0)
        )


def test_retention_days_rejected(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    ledger = _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    with pytest.raises(ValueError, match="retention_days"):
        ledger.reopen_exhausted(_selector(), as_of=AS_OF, retention_days=0, apply=False, run_id="r", attempted_at=_at())


def test_cli_deterministic_output(tmp_path, capsys) -> None:
    from src.tools import extended_ledger_reopen

    path = tmp_path / "ledger.parquet"
    _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00"),
            _exhausted("2026-09-09", "000660", "price_basis_no_raw_source", "2026-09-11T00:10:00+09:00"),
        ],
    )
    args = ["--ledger", str(path), "--as-of", "2026-09-14"]
    assert extended_ledger_reopen.main(args) == 0
    first = capsys.readouterr().out
    assert extended_ledger_reopen.main(args) == 0
    assert capsys.readouterr().out == first


def test_cli_bounds_large_key_sample(tmp_path, capsys) -> None:
    from src.tools import extended_ledger_reopen

    path = tmp_path / "ledger.parquet"
    _seed(
        path,
        [_exhausted("2026-09-10", f"{900000 + i:06d}", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00") for i in range(12)],
    )
    assert extended_ledger_reopen.main(["--ledger", str(path), "--as-of", "2026-09-14"]) == 0
    assert "외 2건" in capsys.readouterr().out
    assert extended_ledger_reopen.main(["--ledger", str(path), "--reason-prefix", "price_basis", "--apply", "--as-of", "2026-09-14"]) == 0
    out = capsys.readouterr().out
    assert "외 2건" in out and "backup=" in out


def test_cli_symbol_zero_padding(tmp_path) -> None:
    from src.tools import extended_ledger_reopen

    path = tmp_path / "ledger.parquet"
    _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T00:10:00+09:00")])
    assert extended_ledger_reopen.main(["--ledger", str(path), "--symbol", "5930", "--apply", "--as-of", "2026-09-14"]) == 0
    frame = pd.read_parquet(path)
    assert str(frame.iloc[0]["status"]) == "FAILED"


def _audit_profile(tmp_path, daily_max: int = 100):
    from src.config.collection import CollectionSettings

    return CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture",
        COLLECTION_EXTENDED_EXHAUSTED_DAILY_MAX=daily_max,
        _env_file=None,
    )


def test_audit_breakdown_and_deterministic_new_info_only(tmp_path) -> None:
    from src.tools import daily_audit

    path = tmp_path / "ledger.parquet"
    _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
            _exhausted("2026-09-10", "000660", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
            _exhausted("2026-09-10", "000270", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
            _exhausted("2026-09-09", "000020", "price_basis_no_raw_source", "2026-09-12T00:10:00+09:00"),
        ],
    )
    result = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=_audit_profile(tmp_path))
    assert result.issues == ()
    assert "intraday:extended_exhausted_reasons:price_basis_raw_basis_unavailable=3,price_basis_no_raw_source=1" in result.info_lines
    assert "intraday:extended_exhausted_new:4" in result.info_lines


def test_audit_unknown_and_transient_warn(tmp_path) -> None:
    from src.tools import daily_audit

    for reason in ("exhausted:vendor_failure", "exhausted:regular_empty_without_proof", "exhausted:transport:RuntimeError"):
        path = tmp_path / f"ledger_{abs(hash(reason))}.parquet"
        _seed(path, [_exhausted("2026-09-10", "005930", reason.removeprefix("exhausted:"), "2026-09-12T00:10:00+09:00")])
        result = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=_audit_profile(tmp_path))
        assert result.issues == ("collection:extended_exhausted:1:nondeterministic_reason",)


def test_audit_surge_and_both_conditions(tmp_path) -> None:
    from src.tools import daily_audit

    path = tmp_path / "ledger.parquet"
    _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
            _exhausted("2026-09-10", "000660", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
            _exhausted("2026-09-09", "000270", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
        ],
    )
    result = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=_audit_profile(tmp_path, daily_max=2))
    assert result.issues == ("collection:extended_exhausted:3:daily_surge",)
    exact = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=_audit_profile(tmp_path, daily_max=3))
    assert exact.issues == ()
    mixed = tmp_path / "mixed.parquet"
    _seed(
        mixed,
        [
            _exhausted("2026-09-10", "005930", "vendor_failure", "2026-09-12T00:10:00+09:00"),
            _exhausted("2026-09-10", "000660", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
            _exhausted("2026-09-09", "000270", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00"),
        ],
    )
    both = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=mixed, profile=_audit_profile(tmp_path, daily_max=2))
    assert both.issues == (
        "collection:extended_exhausted:1:nondeterministic_reason",
        "collection:extended_exhausted:3:daily_surge",
    )


def test_audit_empty_without_exhausted_and_bad_profile(tmp_path) -> None:
    from src.config.collection import CollectionSettings
    from src.tools import daily_audit

    path = tmp_path / "ledger.parquet"
    _seed(path, [_row("2026-09-10", SESSION, "005930", "COMPLETE", "", "2026-09-12T00:10:00+09:00", 0)])
    assert daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=_audit_profile(tmp_path)) == daily_audit.ExtendedExhaustedAudit((), ())
    broken = CollectionSettings.model_construct(
        COLLECTION_EXTENDED_EXHAUSTED_DAILY_MAX=100,
        COLLECTION_EXTENDED_EXHAUSTED_AUDIT_CUTOFF_HHMMSS="25:00",
    )
    seeded = tmp_path / "seeded.parquet"
    _seed(seeded, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00")])
    assert daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=seeded, profile=broken) == daily_audit.ExtendedExhaustedAudit((), ())


def test_audit_breakdown_bounds_many_classes(tmp_path) -> None:
    from src.tools import daily_audit

    path = tmp_path / "ledger.parquet"
    _seed(
        path,
        [
            _exhausted("2026-09-10", f"{900000 + i:06d}", f"custom{i}:detail", "2026-09-12T00:10:00+09:00")
            for i in range(11)
        ],
    )
    result = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=_audit_profile(tmp_path))
    assert len(result.info_lines) == 2
    assert "외 1건" in result.info_lines[0]


def test_audit_detail_wired_into_daily_digest(monkeypatch, tmp_path, caplog) -> None:
    from src.tools import daily_audit

    profile = _audit_profile(tmp_path)
    monkeypatch.setattr(daily_audit, "CollectionSettings", lambda *a, **k: profile)
    monkeypatch.setattr(daily_audit, "audit_daily_completeness", lambda d: dict.fromkeys(daily_audit.AUDIT_STEPS, True))
    monkeypatch.setattr(daily_audit, "audit_collection_manifests", lambda *a, **k: ())
    monkeypatch.setattr(daily_audit, "audit_intraday_partitions", lambda *a, **k: ())
    monkeypatch.setattr(daily_audit, "audit_tape_sweep", lambda *a, **k: ())
    monkeypatch.setattr(daily_audit, "audit_extended_exhausted", lambda **k: ())
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay, SessionKind

    monkeypatch.setattr(
        daily_audit, "resolve_session_day",
        lambda day: SessionDay(day, SessionKind.STANDARD, SessionClock.standard(day), "standard"),
    )
    monkeypatch.setattr(
        daily_audit,
        "audit_extended_exhausted_detail",
        lambda *a, **k: daily_audit.ExtendedExhaustedAudit(
            ("intraday:extended_exhausted_reasons:vendor_failure=1", "intraday:extended_exhausted_new:1"),
            ("collection:extended_exhausted:1:nondeterministic_reason",),
        ),
    )
    from src.data.capture_store import CaptureStore, resolve_capture_root

    store = CaptureStore(resolve_capture_root(profile))
    from tests.unit.tools.test_daily_audit import _publish_cohort_decision

    _publish_cohort_decision(store, "2026-09-14", ["005930"])
    sent: list[tuple[str, str]] = []

    def _dispatch(subject: str, body: str) -> dict[str, bool]:
        sent.append((subject, body))
        return {"webhook": True}

    with caplog.at_level("WARNING"):
        subject = daily_audit.run_daily_audit(
            "2026-09-14",
            trading_day_fn=lambda _d: True,
            failed_units_fn=list,
            stale_tokens_fn=lambda _d: [],
            dispatch_fn=_dispatch,
            backup_issues_fn=lambda _at: [],
            ops_issues_fn=lambda _at: [],
        )
    assert subject is not None and subject.startswith("[kca] 🚨")
    assert "collection:extended_exhausted:1:nondeterministic_reason" in sent[0][1]
    assert "intraday_complete=OK" in sent[0][1]
    assert "exhausted_detail=" in caplog.text


def test_audit_window_edges_and_old_keys(tmp_path) -> None:
    from src.tools import daily_audit

    path = tmp_path / "ledger.parquet"
    _seed(
        path,
        [
            _exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-11T22:59:59+09:00"),
            _exhausted("2026-09-10", "000660", "price_basis_raw_basis_unavailable", "2026-09-11T23:00:00+09:00"),
            _exhausted("2026-09-09", "000270", "price_basis_raw_basis_unavailable", "2026-09-14T22:59:59+09:00"),
            _exhausted("2026-09-09", "000020", "price_basis_raw_basis_unavailable", "2026-09-14T23:00:00+09:00"),
            _exhausted("2026-08-01", "000010", "price_basis_raw_basis_unavailable", "2026-08-02T00:10:00+09:00"),
        ],
    )
    result = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=_audit_profile(tmp_path))
    assert "intraday:extended_exhausted_new:2" in result.info_lines
    assert result.issues == ()


def test_audit_idempotent_and_legacy_malformed_inputs(tmp_path) -> None:
    from src.tools import daily_audit

    path = tmp_path / "ledger.parquet"
    _seed(path, [_exhausted("2026-09-10", "005930", "price_basis_raw_basis_unavailable", "2026-09-12T00:10:00+09:00")])
    profile = _audit_profile(tmp_path)
    first = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=profile)
    assert daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=path, profile=profile) == first
    later = daily_audit.audit_extended_exhausted_detail(date(2026, 9, 16), ledger_path=path, profile=profile)
    assert later.issues == () and all("new" not in line for line in later.info_lines if "reasons" not in line)
    weird = tmp_path / "weird.parquet"
    legacy = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-09-10",
                "session": SESSION,
                "symbol": "005930",
                "status": "EXHAUSTED",
                "rows": 0,
                "reason": "exhausted:price_basis_raw_basis_unavailable",
                "run_id": "r",
                "attempted_at": "garbage",
            },
            {
                "snapshot_date": "2026-09-10",
                "session": SESSION,
                "symbol": "000660",
                "status": "EXHAUSTED",
                "rows": 0,
                "reason": "exhausted:price_basis_raw_basis_unavailable",
                "run_id": "r",
                "attempted_at": "2026-09-12T00:10:00",
            },
        ]
    )
    legacy.to_parquet(weird, index=False)
    result = daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=weird, profile=profile)
    assert result.issues == () and result.info_lines and all("new" not in line for line in result.info_lines if "reasons" not in line)
    assert daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=tmp_path / "absent.parquet", profile=profile) == daily_audit.ExtendedExhaustedAudit((), ())
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not parquet")
    assert daily_audit.audit_extended_exhausted_detail(AS_OF, ledger_path=corrupt, profile=profile) == daily_audit.ExtendedExhaustedAudit((), ())
