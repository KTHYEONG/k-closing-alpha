from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data import intraday_store
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame


def _canon_bar(symbol: str, snapshot_date: str = "2026-09-03") -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "time": ["090300"],
            "open": [70000],
            "high": [70100],
            "low": [69900],
            "close": [70000],
            "jdiff_vol": [1000],
            "value": [70],
        }
    )
    return normalize_bar_frame(raw, "ls", snapshot_date, symbol)


def test_intraday_store_write_partition_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    df_in = _canon_bar("005930")
    rows_written = intraday_store.write_intraday_partition(df_in, 1, "2026-09-03", "regular")
    assert rows_written == 1

    df_out_of_range = _canon_bar("000660", "2026-08-01")
    intraday_store.write_intraday_partition(df_out_of_range, 1, "2026-08-01", "regular")

    target = intraday_store.intraday_partition_path(1, "2026-09-03", "regular")
    result = pd.read_parquet(target)

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "005930"
    other = intraday_store.intraday_partition_path(1, "2026-08-01", "regular")
    assert other.exists()
    assert other != target


def test_intraday_store_write_empty_df_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    rows_written = intraday_store.write_intraday_partition(pd.DataFrame(), 1, "2026-09-03", "regular")

    assert rows_written == 0


def test_intraday_store_write_and_read_tick_partition_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    raw = pd.DataFrame({"time": ["090300"], "close": [70000], "jdiff_vol": [1000]})
    df_in = normalize_tick_frame(raw, "ls", "2026-09-03", "005930")
    rows_written = intraday_store.write_tick_partition(df_in, "2026-09-03", "regular")
    assert rows_written == 1

    target = intraday_store.tick_partition_path("2026-09-03", "regular")
    assert target.exists()
    stored = pd.read_parquet(target)
    assert stored.iloc[0]["symbol"] == "005930"


def test_write_intraday_partition_merges_instead_of_overwriting(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.data import intraday_store
    from src.data.intraday_schema import normalize_bar_frame

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    raw = pd.DataFrame({"time": ["090300"], "open": [9100], "high": [9100], "low": [9100], "close": [9100], "jdiff_vol": [55226], "value": [498]})
    first = normalize_bar_frame(raw, "ls", "2026-09-04", "009900")
    second = normalize_bar_frame(raw, "ls", "2026-09-04", "005930")

    intraday_store.write_intraday_partition(first, 1, "2026-09-04", "regular")
    merged_rows = intraday_store.write_intraday_partition(second, 1, "2026-09-04", "regular")

    assert merged_rows == 2
    stored = pd.read_parquet(intraday_store.intraday_partition_path(1, "2026-09-04", "regular"))
    assert set(stored["symbol"]) == {"009900", "005930"}


def test_write_intraday_partition_rejects_non_canonical_frame(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    legacy_raw = pd.DataFrame({"종목코드": ["005930"], "stck_prpr": [70000], "acml_tr_pbmn": ["123"]})

    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts gate rejection
        intraday_store.write_intraday_partition(legacy_raw, 1, "2026-09-04", "regular")

    assert not intraday_store.intraday_partition_path(1, "2026-09-04", "regular").exists()


def test_log_session_coverage_outliers_flags_symbol_below_peer_ratio(caplog) -> None:
    import logging

    import pandas as pd

    from src.data import intraday_store

    # Given: symbol A has 10 bars (peer max), symbol B has only 3 (30% of peer max, below default 0.8)
    merged = pd.DataFrame({"symbol": ["A"] * 10 + ["B"] * 3})

    # When
    with caplog.at_level(logging.WARNING, logger=intraday_store.logger.name):
        report = intraday_store.log_session_coverage_outliers(merged, 1, "2026-09-11", "regular")

    # Then
    assert report == {"n_symbols": 2, "n_low_coverage": 1, "n_truncated": 0}
    assert any("session_coverage" in rec.message and "B" in rec.message for rec in caplog.records)


def test_log_session_coverage_outliers_returns_zero_when_all_symbols_at_peer_level(caplog) -> None:
    import logging

    import pandas as pd

    from src.data import intraday_store

    # Given: three symbols, all with identical bar counts (a healthy, fully-collected batch)
    merged = pd.DataFrame({"symbol": ["A"] * 5 + ["B"] * 5 + ["C"] * 5})

    # When
    with caplog.at_level(logging.WARNING, logger=intraday_store.logger.name):
        report = intraday_store.log_session_coverage_outliers(merged, 1, "2026-09-11", "regular")

    # Then
    assert report == {"n_symbols": 3, "n_low_coverage": 0, "n_truncated": 0}
    assert not any("session_coverage" in rec.message for rec in caplog.records)


def test_log_session_coverage_outliers_handles_empty_frame() -> None:
    import pandas as pd

    from src.data import intraday_store

    # Given: an empty frame (write_intraday_partition never reaches this point with one,
    # since it returns 0 early, but the utility must still be safe standalone)
    report = intraday_store.log_session_coverage_outliers(pd.DataFrame(), 1, "2026-09-11", "regular")

    # Then
    assert report == {"n_symbols": 0, "n_low_coverage": 0, "n_truncated": 0}


def test_log_session_coverage_outliers_respects_custom_min_peer_ratio_boundary() -> None:
    import pandas as pd

    from src.data import intraday_store

    # Given: peer max = 10; B sits exactly at 80% (8, must NOT be flagged -- strict less-than),
    # C sits just below (7, MUST be flagged), under an explicit min_peer_ratio=0.8
    merged = pd.DataFrame({"symbol": ["A"] * 10 + ["B"] * 8 + ["C"] * 7})

    # When
    report = intraday_store.log_session_coverage_outliers(merged, 1, "2026-09-11", "regular", min_peer_ratio=0.8)

    # Then
    assert report == {"n_symbols": 3, "n_low_coverage": 1, "n_truncated": 0}

    # And: a lone symbol (no peer) is never flagged, regardless of ratio
    solo = pd.DataFrame({"symbol": ["Z"] * 2})
    solo_report = intraday_store.log_session_coverage_outliers(solo, 1, "2026-09-11", "regular", min_peer_ratio=0.99)
    assert solo_report == {"n_symbols": 1, "n_low_coverage": 0, "n_truncated": 0}


def test_write_intraday_partition_logs_low_coverage_symbol_without_changing_return_value(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pandas as pd
    import pytest

    from src.data import intraday_store
    from src.data.intraday_schema import normalize_bar_frame

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    # Given: 005930 has 3 bars (full coverage), 000660 has only 1 (sparse, mirrors the
    # empirically-confirmed 181710/Toss-backfill pattern)
    raw_full = pd.DataFrame({
        "time": ["090100", "090200", "090300"],
        "open": [70000, 70000, 70000], "high": [70100, 70100, 70100],
        "low": [69900, 69900, 69900], "close": [70000, 70000, 70000],
        "jdiff_vol": [1000, 1000, 1000], "value": [70, 70, 70],
    })
    df_full = normalize_bar_frame(raw_full, "ls", "2026-09-11", "005930")
    raw_sparse = pd.DataFrame({
        "time": ["090100"], "open": [50000], "high": [50100], "low": [49900],
        "close": [50000], "jdiff_vol": [500], "value": [25],
    })
    df_sparse = normalize_bar_frame(raw_sparse, "ls", "2026-09-11", "000660")
    df_in = pd.concat([df_full, df_sparse], ignore_index=True)

    # When
    with caplog.at_level(logging.WARNING, logger=intraday_store.logger.name):
        rows_written = intraday_store.write_intraday_partition(df_in, 1, "2026-09-11", "regular")

    # Then: return contract unchanged (still total row count), diagnostic visible via log
    assert rows_written == 4
    assert any("session_coverage" in rec.message and "000660" in rec.message for rec in caplog.records)


def test_log_session_coverage_outliers_does_not_flag_full_range_scattered_gaps() -> None:
    import pandas as pd

    from src.data import intraday_store

    # Given: A has 12 bars spanning the full session (peer reference); B has only 5 bars
    # (below the 0.8 peer-ratio threshold) but its first and last bar still match A's
    # floor/ceiling -- scattered internal gaps only, not a truncated session
    merged = pd.DataFrame({
        "symbol": ["A"] * 12 + ["B"] * 5,
        "ts_hms": (
            [90000, 90100, 90200, 90300, 100000, 110000, 120000, 130000, 140000, 150000, 152900, 153000]  # noqa: RUF005 - contract skeleton verbatim
            + [90000, 100000, 120000, 140000, 153000]
        ),
    })

    # When
    report = intraday_store.log_session_coverage_outliers(merged, 1, "2026-09-11", "regular")

    # Then: B is flagged low-coverage but NOT truncated
    assert report == {"n_symbols": 2, "n_low_coverage": 1, "n_truncated": 0}


def test_log_session_coverage_outliers_flags_truncated_start_symbol() -> None:
    import pandas as pd

    from src.data import intraday_store

    # Given: A spans the full session (floor=90000); C starts late at 121100
    # (mirrors the confirmed 2023-07-05 Toss backfill defect)
    merged = pd.DataFrame({
        "symbol": ["A"] * 12 + ["C"] * 6,
        "ts_hms": (
            [90000, 90100, 90200, 90300, 100000, 110000, 120000, 130000, 140000, 150000, 152900, 153000]  # noqa: RUF005 - contract skeleton verbatim
            + [121100, 130000, 140000, 143000, 150000, 153000]
        ),
    })

    # When
    report = intraday_store.log_session_coverage_outliers(merged, 1, "2026-09-11", "regular")

    # Then: C is flagged both low-coverage (6 < 12*0.8) and truncated (starts 31100 past floor)
    assert report == {"n_symbols": 2, "n_low_coverage": 1, "n_truncated": 1}


def test_log_session_coverage_outliers_flags_truncated_end_symbol() -> None:
    import pandas as pd

    from src.data import intraday_store

    # Given: A spans the full session ending at 153000; D has almost as many bars (11 vs
    # peer 12, NOT below the 0.8 peer-ratio threshold) but ends early at 151900
    merged = pd.DataFrame({
        "symbol": ["A"] * 12 + ["D"] * 11,
        "ts_hms": (
            [90000, 90100, 90200, 90300, 100000, 110000, 120000, 130000, 140000, 150000, 152900, 153000]  # noqa: RUF005 - contract skeleton verbatim
            + [90000, 90100, 90200, 90300, 100000, 110000, 120000, 130000, 140000, 150000, 151900]
        ),
    })

    # When
    report = intraday_store.log_session_coverage_outliers(merged, 1, "2026-09-11", "regular")

    # Then: D is truncated (ends 1100 before ceiling) but NOT flagged low-coverage
    assert report == {"n_symbols": 2, "n_low_coverage": 0, "n_truncated": 1}


def test_write_intraday_partition_logs_truncation_warning_for_partial_session_symbol(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.data import intraday_store
    from src.data.intraday_schema import normalize_bar_frame

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    # Given: 005930 spans 09:00-09:10 (11 bars, full range); 000660 only has 2 bars
    # starting at 09:09 (900 HHMMSS units past the batch floor 090000, past the 500 default)
    times_full = ["090000", "090100", "090200", "090300", "090400", "090500", "090600", "090700", "090800", "090900", "091000"]
    raw_full = pd.DataFrame({
        "time": times_full,
        "open": [70000] * 11, "high": [70100] * 11, "low": [69900] * 11, "close": [70000] * 11,
        "jdiff_vol": [1000] * 11, "value": [70] * 11,
    })
    df_full = normalize_bar_frame(raw_full, "ls", "2026-09-11", "005930")
    raw_late = pd.DataFrame({
        "time": ["090900", "091000"],
        "open": [50000, 50000], "high": [50100, 50100], "low": [49900, 49900], "close": [50000, 50000],
        "jdiff_vol": [500, 500], "value": [25, 25],
    })
    df_late = normalize_bar_frame(raw_late, "ls", "2026-09-11", "000660")
    df_in = pd.concat([df_full, df_late], ignore_index=True)

    # When
    with caplog.at_level(logging.WARNING, logger=intraday_store.logger.name):
        rows_written = intraday_store.write_intraday_partition(df_in, 1, "2026-09-11", "regular")

    # Then: return contract unchanged (total row count), truncation diagnostic visible via log
    assert rows_written == 13
    assert any("session_truncation" in rec.message and "000660" in rec.message for rec in caplog.records)


def _tick_rows(symbol: str, snapshot_date: str, rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    raw = pd.DataFrame({
        "time": [item[0] for item in rows],
        "close": [item[1] for item in rows],
        "jdiff_vol": [item[2] for item in rows],
    })
    return normalize_tick_frame(raw, "ls", snapshot_date, symbol)


def _tick_entry(symbol: str, status: str = "COMPLETE", session: str = "regular", venue: str = "KRX") -> object:
    from src.data.capture_contracts import ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry

    refs: tuple[object, ...] = ()
    reason = "sweep done"
    if status in ("NO_TRADES", "NOT_APPLICABLE"):
        refs = (ArtifactRef(path=f"raw/{symbol}.json.gz", sha256="a" * 64, bytes=8, rows=0),)
        reason = "verified empty with proof"
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.TRADE_TICKS,
        venue=venue,
        session=session,
        scheduled_at=None,
        status=CaptureStatus(status),
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=refs,  # type: ignore[arg-type]
    )


def _bar_entry(symbol: str, status: str = "COMPLETE", session: str = "regular", venue: str = "KRX") -> object:
    from src.data.capture_contracts import ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry

    refs: tuple[object, ...] = ()
    reason = "sweep done"
    if status in ("NO_TRADES", "NOT_APPLICABLE"):
        refs = (ArtifactRef(path=f"raw/{symbol}.json.gz", sha256="a" * 64, bytes=8, rows=0),)
        reason = "verified empty with proof"
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.MINUTE_BARS,
        venue=venue,
        session=session,
        scheduled_at=None,
        status=CaptureStatus(status),
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=refs,  # type: ignore[arg-type]
    )


def _canon_bar_at(symbol: str, snapshot_date: str, time: str, close: int, vol: int = 100) -> pd.DataFrame:
    raw = pd.DataFrame({"time": [time], "open": [close], "high": [close], "low": [close], "close": [close], "jdiff_vol": [vol], "value": [10]})
    return normalize_bar_frame(raw, "ls", snapshot_date, symbol)


def test_write_tick_partition_preserves_same_second_trades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    df = _tick_rows("005930", "2026-09-03", [("090300", 70000, 10), ("090300", 70100, 10)])
    cov = {"005930": _tick_entry("005930")}
    first = intraday_store.write_tick_partition(df, "2026-09-03", "regular", coverage=cov)  # type: ignore[arg-type]
    assert first == 2
    second = intraday_store.write_tick_partition(df, "2026-09-03", "regular", coverage=cov)  # type: ignore[arg-type]
    assert second == 2
    stored = pd.read_parquet(intraday_store.tick_partition_path("2026-09-03", "regular"))
    assert len(stored) == 2
    assert int(stored["volume"].sum()) == 20
    assert sorted(stored["price"].tolist()) == [70000, 70100]


def test_write_tick_partition_preserves_identical_repeated_trades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    df = _tick_rows("005930", "2026-09-03", [("090300", 70000, 5), ("090300", 70000, 5)])
    cov = {"005930": _tick_entry("005930")}
    assert intraday_store.write_tick_partition(df, "2026-09-03", "regular", coverage=cov) == 2  # type: ignore[arg-type]
    assert intraday_store.write_tick_partition(df, "2026-09-03", "regular", coverage=cov) == 2  # type: ignore[arg-type]
    stored = pd.read_parquet(intraday_store.tick_partition_path("2026-09-03", "regular"))
    assert len(stored) == 2


def test_write_tick_partition_conserves_unaffected_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    old_a = _tick_rows("005930", date, [("090300", 70000, 10)])
    old_b = _tick_rows("000660", date, [("090300", 50000, 7)])
    both = pd.concat([old_a, old_b], ignore_index=True)
    cov_both = {"005930": _tick_entry("005930"), "000660": _tick_entry("000660")}
    intraday_store.write_tick_partition(both, date, "regular", coverage=cov_both)  # type: ignore[arg-type]
    before_b = pd.read_parquet(intraday_store.tick_partition_path(date, "regular")).query("symbol == '000660'")
    new_a = _tick_rows("005930", date, [("090400", 70100, 3), ("090500", 70200, 4)])
    total = intraday_store.write_tick_partition(new_a, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    assert total == 3
    stored = pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))
    after_b = stored.query("symbol == '000660'")
    assert len(after_b) == 1
    assert after_b.reset_index(drop=True).astype(str).equals(before_b.reset_index(drop=True).astype(str))
    assert len(stored.query("symbol == '005930'")) == 2


def test_write_tick_partition_rejects_partial_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    full = _tick_rows("005930", date, [("090300", 70000, 10)])
    intraday_store.write_tick_partition(full, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    before = pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))
    partial = _tick_rows("005930", date, [("090400", 70100, 2)])
    with pytest.raises(ValueError, match="Non-certified"):
        intraday_store.write_tick_partition(partial, date, "regular", coverage={"005930": _tick_entry("005930", status="PARTIAL")})  # type: ignore[arg-type]
    after = pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))
    assert after.reset_index(drop=True).astype(str).equals(before.reset_index(drop=True).astype(str))
    assert (tmp_path / "capture" / "staging" / "intraday").exists()


def test_write_tick_partition_repair_removes_contamination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    legacy = _tick_rows("005930", date, [("090000", 70000, 1), ("200000", 70000, 9)])
    intraday_store.write_tick_partition(legacy, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    clean = _tick_rows("005930", date, [("090100", 70100, 2), ("090200", 70200, 3)])
    total = intraday_store.write_tick_partition(clean, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    assert total == 2
    stored = pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))
    assert 200000 not in stored["ts_hms"].tolist()
    assert sorted(stored["ts_hms"].tolist()) == [90100, 90200]
    backups = list((tmp_path / "capture" / "backups").rglob("*.parquet"))
    assert len(backups) >= 1


def test_write_tick_partition_fails_on_corrupt_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    target = intraday_store.tick_partition_path("2026-09-03", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("corrupt bytes")
    df = _tick_rows("005930", "2026-09-03", [("090300", 70000, 1)])
    with pytest.raises(OSError, match="Cannot read existing"):
        intraday_store.write_tick_partition(df, "2026-09-03", "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    assert target.read_text() == "corrupt bytes"


def test_write_tick_partition_failed_staging_keeps_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    first = _tick_rows("005930", date, [("090300", 70000, 1)])
    intraday_store.write_tick_partition(first, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    before_bytes = intraday_store.tick_partition_path(date, "regular").read_bytes()
    real_writer = intraday_store.pq.ParquetWriter

    class _BoomWriter(real_writer):  # type: ignore[misc]
        def write_table(self, *args: object, **kwargs: object) -> None:
            raise OSError("arrow write boom")

    monkeypatch.setattr(intraday_store.pq, "ParquetWriter", _BoomWriter)
    second = _tick_rows("000660", date, [("090300", 50000, 2)])
    with pytest.raises(OSError, match="arrow write boom"):
        intraday_store.write_tick_partition(second, date, "regular", coverage={"000660": _tick_entry("000660")})  # type: ignore[arg-type]
    assert intraday_store.tick_partition_path(date, "regular").read_bytes() == before_bytes
    monkeypatch.setattr(intraday_store.pq, "ParquetWriter", real_writer)
    monkeypatch.setattr(intraday_store.os, "replace", lambda _s, _d: (_ for _ in ()).throw(OSError("rename boom")))
    with pytest.raises(OSError, match="publication failed"):
        intraday_store.write_tick_partition(second, date, "regular", coverage={"000660": _tick_entry("000660")})  # type: ignore[arg-type]
    assert intraday_store.tick_partition_path(date, "regular").read_bytes() == before_bytes


def test_write_tick_partition_serializes_concurrent_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    df_a = _tick_rows("005930", date, [("090300", 70000, 1)])
    df_b = _tick_rows("000660", date, [("090300", 50000, 2)])
    cov_a = {"005930": _tick_entry("005930")}
    cov_b = {"000660": _tick_entry("000660")}

    def _write_a() -> int:
        return intraday_store.write_tick_partition(df_a, date, "regular", coverage=cov_a)  # type: ignore[arg-type]

    def _write_b() -> int:
        return intraday_store.write_tick_partition(df_b, date, "regular", coverage=cov_b)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda fn: fn(), [_write_a, _write_b]))
    assert sorted(results) in ([1, 2], [2, 2], [1, 1])
    stored = pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))
    assert set(stored["symbol"].tolist()) == {"005930", "000660"}


def test_write_tick_partition_empty_frame_preserves_trades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    df = _tick_rows("005930", date, [("090300", 70000, 4)])
    intraday_store.write_tick_partition(df, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    total = intraday_store.write_tick_partition(pd.DataFrame(), date, "regular")
    assert total == 1
    total_none = intraday_store.write_tick_partition(None, date, "regular")  # type: ignore[arg-type]
    assert total_none == 1
    stored = pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))
    assert len(stored) == 1
    cleared = intraday_store.write_tick_partition(
        pd.DataFrame(), date, "regular", coverage={"005930": _tick_entry("005930", status="NO_TRADES")}  # type: ignore[arg-type]
    )
    assert cleared == 0
    target = intraday_store.tick_partition_path(date, "regular")
    if target.exists():
        assert len(pd.read_parquet(target)) == 0



def test_write_intraday_partition_rejects_contradictory_bar_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-04"
    first = _canon_bar_at("005930", date, "090300", 70000)
    intraday_store.write_intraday_partition(first, 1, date, "regular", coverage={"005930": _bar_entry("005930")})  # type: ignore[arg-type]
    before = pd.read_parquet(intraday_store.intraday_partition_path(1, date, "regular"))
    bad = pd.concat([_canon_bar_at("005930", date, "090400", 70000), _canon_bar_at("005930", date, "090400", 71000)], ignore_index=True)
    with pytest.raises(ValueError, match="Contradictory"):
        intraday_store.write_intraday_partition(bad, 1, date, "regular", coverage={"005930": _bar_entry("005930")})  # type: ignore[arg-type]
    after = pd.read_parquet(intraday_store.intraday_partition_path(1, date, "regular"))
    assert after.reset_index(drop=True).astype(str).equals(before.reset_index(drop=True).astype(str))
    dup = pd.concat([_canon_bar_at("005930", date, "090500", 70000), _canon_bar_at("005930", date, "090500", 70000)], ignore_index=True)
    total = intraday_store.write_intraday_partition(dup, 1, date, "regular", coverage={"005930": _bar_entry("005930")})  # type: ignore[arg-type]
    assert total == 1
    assert pd.read_parquet(intraday_store.intraday_partition_path(1, date, "regular"))["ts_hms"].tolist() == [90500]


def test_write_tick_partition_processes_large_partition_in_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    seed = _tick_rows("005930", date, [("090300", 70000, 1)])
    intraday_store.write_tick_partition(seed, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    big_parts = [_tick_rows(f"{900000 + idx:06d}", date, [("090300", 60000, 1)]) for idx in range(30)]
    big = pd.concat(big_parts, ignore_index=True)
    big_cov = {str(symbol): _tick_entry(str(symbol)) for symbol in big["symbol"].unique().tolist()}
    intraday_store.write_tick_partition(big, date, "regular", coverage=big_cov, batch_rows=8)  # type: ignore[arg-type]
    seen_sizes: list[int] = []
    import pyarrow.parquet as pq_mod

    real_file = pq_mod.ParquetFile
    real_read = pd.read_parquet
    target = intraday_store.tick_partition_path(date, "regular")

    def _guarded_read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if str(path) == str(target):
            raise AssertionError("bounded rewrite must not pandas-read the whole partition")
        return real_read(path, *args, **kwargs)  # type: ignore[arg-type]

    class _SpyFile(real_file):  # type: ignore[misc]
        def iter_batches(self, batch_size: int | None = None, **kwargs: object):  # type: ignore[override]
            seen_sizes.append(int(batch_size or 0))
            yield from super().iter_batches(batch_size=batch_size, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _guarded_read)
    monkeypatch.setattr(pq_mod, "ParquetFile", _SpyFile)
    monkeypatch.setattr(intraday_store.pq, "ParquetFile", _SpyFile)
    small = _tick_rows("000001", date, [("090400", 50000, 2)])
    total = intraday_store.write_tick_partition(small, date, "regular", coverage={"000001": _tick_entry("000001")}, batch_rows=8)  # type: ignore[arg-type]
    assert total == 32
    assert seen_sizes and max(seen_sizes) <= 8


def test_write_tick_partition_legacy_changed_attempt_requires_certification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    first = _tick_rows("005930", date, [("090300", 70000, 1)])
    assert intraday_store.write_tick_partition(first, date, "regular") == 1
    assert intraday_store.write_tick_partition(first, date, "regular") == 1
    changed = _tick_rows("005930", date, [("090400", 70100, 9)])
    with pytest.raises(ValueError, match="requires certification"):
        intraday_store.write_tick_partition(changed, date, "regular")
    stored = pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))
    assert len(stored) == 1
    assert int(stored.iloc[0]["ts_hms"]) == 90300
    with pytest.raises(ValueError, match="batch_rows must be positive"):
        intraday_store.write_tick_partition(first, date, "regular", batch_rows=0)
    with pytest.raises(ValueError, match="UNKNOWN venue"):
        intraday_store.write_tick_partition(first, date, "regular", coverage={"005930": _tick_entry("005930", status="PARTIAL", venue="UNKNOWN")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="session mismatch"):
        intraday_store.write_tick_partition(first, date, "regular", coverage={"005930": _tick_entry("005930", session="nxt_aftermarket")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lack coverage"):
        intraday_store.write_tick_partition(first, date, "regular", coverage={"000660": _tick_entry("000660")})  # type: ignore[arg-type]
    bad_date = _tick_rows("005930", "2026-09-04", [("090300", 70000, 1)])
    with pytest.raises(ValueError, match="snapshot_date"):
        intraday_store.write_tick_partition(bad_date, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    longer = _tick_rows("005930", date, [("090300", 70000, 1), ("090400", 70100, 2)])
    with pytest.raises(ValueError, match="requires certification"):
        intraday_store.write_tick_partition(longer, date, "regular")


def test_write_partitions_reject_invalid_frames_and_certification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(intraday_store.settings, "COLLECTION_ROOT", tmp_path / "custom-capture")
    date = "2026-09-03"
    base_tick = _tick_rows("005930", date, [("090300", 70000, 1)])
    bad_ts = base_tick.copy()
    bad_ts["ts_hms"] = -5
    with pytest.raises(ValueError, match="HHMMSS"):
        intraday_store.write_tick_partition(bad_ts, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    non_numeric = base_tick.copy()
    non_numeric["ts_hms"] = "not-a-time"
    with pytest.raises(ValueError, match="non-numeric"):
        intraday_store.write_tick_partition(non_numeric, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    neg_vol = base_tick.copy()
    neg_vol["volume"] = -1
    with pytest.raises(ValueError, match="negative volume"):
        intraday_store.write_tick_partition(neg_vol, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    base_bar = _canon_bar_at("005930", date, "090300", 70000)
    bad_bar_ts = base_bar.copy()
    bad_bar_ts["ts_hms"] = 999999
    with pytest.raises(ValueError, match="HHMMSS"):
        intraday_store.write_intraday_partition(bad_bar_ts, 1, date, "regular", coverage={"005930": _bar_entry("005930")})  # type: ignore[arg-type]
    bad_bar_num = base_bar.copy()
    bad_bar_num["ts_hms"] = "xx"
    with pytest.raises(ValueError, match="non-numeric"):
        intraday_store.write_intraday_partition(bad_bar_num, 1, date, "regular", coverage={"005930": _bar_entry("005930")})  # type: ignore[arg-type]
    bad_bar_vol = base_bar.copy()
    bad_bar_vol["volume"] = -2
    with pytest.raises(ValueError, match="negative volume"):
        intraday_store.write_intraday_partition(bad_bar_vol, 1, date, "regular", coverage={"005930": _bar_entry("005930")})  # type: ignore[arg-type]
    bad_bar_date = _canon_bar_at("005930", "2026-09-04", "090300", 70000)
    with pytest.raises(ValueError, match="snapshot_date"):
        intraday_store.write_intraday_partition(bad_bar_date, 1, date, "regular", coverage={"005930": _bar_entry("005930")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lack coverage"):
        intraday_store.write_intraday_partition(base_bar, 1, date, "regular", coverage={"000660": _bar_entry("000660")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="session mismatch"):
        intraday_store.write_intraday_partition(base_bar, 1, date, "regular", coverage={"005930": _bar_entry("005930", session="nxt_aftermarket")})  # type: ignore[arg-type]
    for status in ("FAILED", "UNKNOWN", "PENDING", "NOT_APPLICABLE"):
        with pytest.raises(ValueError, match="Non-certified"):
            intraday_store.write_tick_partition(base_tick, date, "regular", coverage={"005930": _tick_entry("005930", status=status)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="UNKNOWN venue"):
        intraday_store.write_intraday_partition(base_bar, 1, date, "regular", coverage={"005930": _bar_entry("005930", venue="UNKNOWN")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="session mismatch"):
        intraday_store.write_intraday_partition(base_bar, 1, date, "regular", coverage={"005930": _bar_entry("005930", session="nxt_aftermarket")})  # type: ignore[arg-type]


def test_write_partitions_handle_storage_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    assert intraday_store._capture_root() == tmp_path / "capture"
    assert intraday_store._is_valid_hhmmss(-1) is False
    assert intraday_store._is_valid_hhmmss(240000) is False
    assert intraday_store._is_valid_hhmmss(126060) is False
    assert intraday_store._is_valid_hhmmss(90300) is True
    target = intraday_store.tick_partition_path("2026-09-10", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("broken")
    legacy = _tick_rows("005930", "2026-09-10", [("090300", 70000, 1)])
    with pytest.raises(OSError, match="Cannot read existing"):
        intraday_store.write_tick_partition(legacy, "2026-09-10", "regular")
    legacy_bad = tmp_path / "legacy.parquet"
    pd.DataFrame({"ts_hms": [90300]}).to_parquet(legacy_bad, index=False)
    with pytest.raises(ValueError, match="missing key columns"):
        intraday_store._collect_legacy_overlap(legacy_bad, {"005930"}, 16)
    with pytest.raises(OSError, match="Cannot read existing"):
        intraday_store._collect_legacy_overlap(target, {"005930"}, 16)
    good = _tick_rows("005930", date, [("090300", 70000, 1)])
    intraday_store.write_tick_partition(good, date, "regular", coverage={"005930": _tick_entry("005930")})  # type: ignore[arg-type]
    held_target = intraday_store.tick_partition_path(date, "regular")
    before_bytes = held_target.read_bytes()
    sidecar = held_target.parent / (held_target.name + ".lock")
    holder_fd = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(intraday_store, "_LOCK_TIMEOUT_SECONDS", 0.0)
        blocked = _tick_rows("000660", date, [("090310", 50000, 1)])
        with pytest.raises(OSError, match="timed out acquiring partition lock"):
            intraday_store.write_tick_partition(blocked, date, "regular", coverage={"000660": _tick_entry("000660")})  # type: ignore[arg-type]
    finally:
        os.close(holder_fd)
    assert held_target.read_bytes() == before_bytes
    monkeypatch.setattr(intraday_store, "_LOCK_TIMEOUT_SECONDS", 30.0)
    real_count = intraday_store._partition_row_count
    monkeypatch.setattr(intraday_store, "_partition_row_count", lambda p: 999 if p.name.startswith(".stage-") else real_count(p))
    extra = _tick_rows("000660", date, [("090300", 50000, 1)])
    with pytest.raises(OSError, match="verification failed"):
        intraday_store.write_tick_partition(extra, date, "regular", coverage={"000660": _tick_entry("000660")})  # type: ignore[arg-type]
    assert intraday_store._partition_row_count(intraday_store.tick_partition_path(date, "regular")) == 1
    monkeypatch.setattr(intraday_store, "_partition_row_count", real_count)
    assert intraday_store.write_tick_partition(pd.DataFrame(), "2026-09-20", "regular", coverage={"005930": _tick_entry("005930", status="NO_TRADES")}) == 0  # type: ignore[arg-type]
    assert intraday_store.write_tick_partition(pd.DataFrame(), date, "regular", coverage={"005930": _tick_entry("005930")}) == 1  # type: ignore[arg-type]
    old_with_b = pd.concat([good, _tick_rows("000660", date, [("090310", 50000, 3)])], ignore_index=True)
    intraday_store.write_tick_partition(old_with_b, date, "regular", coverage={"005930": _tick_entry("005930"), "000660": _tick_entry("000660")})  # type: ignore[arg-type]
    new_a_only = _tick_rows("005930", date, [("090320", 70200, 5)])
    mixed_cov = {"005930": _tick_entry("005930"), "000660": _tick_entry("000660", status="NO_TRADES")}
    total = intraday_store.write_tick_partition(new_a_only, date, "regular", coverage=mixed_cov)  # type: ignore[arg-type]
    assert total == 1
    assert pd.read_parquet(intraday_store.tick_partition_path(date, "regular"))["symbol"].tolist() == ["005930"]
    assert intraday_store.write_intraday_partition(pd.DataFrame(), 1, "2026-09-21", "regular") == 0
    assert intraday_store.write_intraday_partition(pd.DataFrame(), 1, date, "regular", coverage={"005930": _bar_entry("005930")}) == 0  # type: ignore[arg-type]
    bar_gone = intraday_store.write_intraday_partition(pd.DataFrame(), 1, "2026-09-22", "regular", coverage={"005930": _bar_entry("005930", status="NO_TRADES")})  # type: ignore[arg-type]
    assert bar_gone == 0


def test_tick_write_with_stale_sidecar_succeeds_and_leaves_no_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    date = "2026-09-03"
    good = _tick_rows("005930", date, [("090300", 70000, 1)])
    assert intraday_store.write_tick_partition(good, date, "regular", coverage={"005930": _tick_entry("005930")}) == 1  # type: ignore[arg-type]
    target = intraday_store.tick_partition_path(date, "regular")
    (target.parent / (target.name + ".lock")).write_text("stale", encoding="utf-8")
    more = _tick_rows("000660", date, [("090310", 50000, 1)])
    assert intraday_store.write_tick_partition(more, date, "regular", coverage={"000660": _tick_entry("000660")}) == 2  # type: ignore[arg-type]
    assert list(target.parent.glob("*.lock")) == []
    vacated = intraday_store.write_tick_partition(pd.DataFrame(), date, "regular", coverage={"005930": _tick_entry("005930", status="NO_TRADES")})  # type: ignore[arg-type]
    assert vacated == 1
    assert list(target.parent.glob("*.lock")) == []



