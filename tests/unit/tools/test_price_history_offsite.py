from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _panel(symbols: tuple[str, ...], days: pd.DatetimeIndex, *, scale: float = 1.0) -> pd.DataFrame:
    from src.strategy.contract import derive_chg_ratio

    rows = []
    for i, symbol in enumerate(symbols):
        price = 1000.0 * (i + 1)
        prev = price
        for j, day in enumerate(days):
            close = round((price + 10 * j) * scale)
            rows.append({"date": day, "symbol": symbol, "open": close, "high": close + 5, "low": close - 5, "close": close,
                         "prev_close": round(prev), "close_raw": float(price + 10 * j), "volume": 100 + j, "market": "KOSPI"})
            prev = close
    frame = pd.DataFrame(rows)
    frame["daily_change_pct"] = derive_chg_ratio(frame["close"].to_numpy(float), frame["prev_close"].to_numpy(float))
    return frame


def _write(frame: pd.DataFrame, path: Path) -> Path:
    from src.data.parquet_codec import write_price_history_parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    write_price_history_parquet(frame, path)
    return path


DAYS = pd.bdate_range("2026-07-28", "2026-09-04")
SYMBOLS = ("000020", "005930", "035720")


def test_round_trip_is_exact_and_restore_writes_canonical_monolith(tmp_path: Path) -> None:
    from src.tools.price_history_offsite import export_offsite_form, rebuild_frame, restore_monolith

    # Given a canonical monolith spanning three calendar months
    source = _write(_panel(SYMBOLS, DAYS), tmp_path / "price_history.parquet")

    # When exported and rebuilt
    report = export_offsite_form(source, tmp_path / "form")

    # Then the first export is a base of one file per month and rebuilds the monolith exactly
    assert report.rebased and report.patch_rows == 0 and report.delete_rows == 0
    assert len([p for p in report.written if p.startswith("base/")]) == 3
    pd.testing.assert_frame_equal(rebuild_frame(tmp_path / "form"), pd.read_parquet(source), check_exact=True)
    rows = restore_monolith(tmp_path / "form", tmp_path / "restored.parquet")
    assert rows == len(pd.read_parquet(source))
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "restored.parquet"), pd.read_parquet(source), check_exact=True)


def test_back_adjustment_and_new_day_touch_only_the_patch(tmp_path: Path) -> None:
    from src.tools.price_history_offsite import export_offsite_form, rebuild_frame

    # Given an exported base
    form = tmp_path / "form"
    export_offsite_form(_write(_panel(SYMBOLS, DAYS), tmp_path / "v1.parquet"), form)
    base_before = {p.relative_to(form).as_posix(): p.stat().st_mtime_ns for p in (form / "base").rglob("*.parquet")}

    # When one symbol's whole history is back-adjusted (touches every month) and a new day is appended
    adjusted = _panel(SYMBOLS, DAYS.append(pd.DatetimeIndex(["2026-09-07"])))
    target = adjusted["symbol"] == "005930"
    for column in ("open", "high", "low", "close", "prev_close"):
        adjusted.loc[target, column] = (adjusted.loc[target, column] // 2).astype(int)
    from src.strategy.contract import derive_chg_ratio

    adjusted["daily_change_pct"] = derive_chg_ratio(adjusted["close"].to_numpy(float), adjusted["prev_close"].to_numpy(float))
    source = _write(adjusted, tmp_path / "v2.parquet")
    report = export_offsite_form(source, form, rebase_ratio=0.9)

    # Then base files are untouched, the patch carries exactly the changed rows, and the rebuild is exact
    base_after = {p.relative_to(form).as_posix(): p.stat().st_mtime_ns for p in (form / "base").rglob("*.parquet")}
    assert not report.rebased
    assert base_after == base_before
    assert all(not p.startswith("base/") for p in report.written)
    assert report.patch_rows == len(DAYS) + len(SYMBOLS)
    pd.testing.assert_frame_equal(rebuild_frame(form), pd.read_parquet(source), check_exact=True)


def test_deleted_rows_are_recorded_and_unlisted_files_removed(tmp_path: Path) -> None:
    from src.tools.price_history_offsite import export_offsite_form, rebuild_frame

    # Given an exported base and a first patch
    form = tmp_path / "form"
    export_offsite_form(_write(_panel(SYMBOLS, DAYS), tmp_path / "v1.parquet"), form)
    first = export_offsite_form(_write(_panel(SYMBOLS, DAYS[:-1]), tmp_path / "v2.parquet"), form, rebase_ratio=0.9)

    # When another symbol disappears
    source = _write(_panel(SYMBOLS[:2], DAYS[:-1]), tmp_path / "v3.parquet")
    report = export_offsite_form(source, form, rebase_ratio=0.9)

    # Then deleted keys cover both removals, the superseded deletes file is gone, and the rebuild is exact
    assert first.delete_rows == len(SYMBOLS)
    assert report.delete_rows == len(SYMBOLS) + len(DAYS) - 1
    assert any(p.startswith("deletes-") for p in report.removed)
    assert len(list(form.glob("deletes-*.parquet"))) == 1
    pd.testing.assert_frame_equal(rebuild_frame(form), pd.read_parquet(source), check_exact=True)


def test_patch_above_ratio_rebases_and_is_deterministic(tmp_path: Path) -> None:
    from src.tools.price_history_offsite import export_offsite_form

    # Given two independent mirrors of the same history
    v1 = _write(_panel(SYMBOLS, DAYS), tmp_path / "v1.parquet")
    v2 = _write(_panel(SYMBOLS, DAYS, scale=1.5), tmp_path / "v2.parquet")
    export_offsite_form(v1, tmp_path / "a")

    # When every row changes (patch share 1.0 > ratio)
    report = export_offsite_form(v2, tmp_path / "a")
    export_offsite_form(v2, tmp_path / "b")
    again = export_offsite_form(v2, tmp_path / "a")

    # Then the base is rebuilt, mirrors are byte-identical, and an unchanged source writes nothing
    assert report.rebased and report.patch_rows == 0
    names_a = {p.relative_to(tmp_path / "a").as_posix(): p.read_bytes() for p in (tmp_path / "a").rglob("*.parquet")}
    names_b = {p.relative_to(tmp_path / "b").as_posix(): p.read_bytes() for p in (tmp_path / "b").rglob("*.parquet")}
    assert names_a == names_b
    assert again.written == () and again.removed == ()


def test_corrupt_file_is_refused_on_restore_and_rebased_on_export(tmp_path: Path) -> None:
    from src.tools.price_history_offsite import export_offsite_form, rebuild_frame

    # Given an exported form with one corrupted base file
    source = _write(_panel(SYMBOLS, DAYS), tmp_path / "v1.parquet")
    form = tmp_path / "form"
    export_offsite_form(source, form)
    victim = sorted((form / "base").rglob("*.parquet"))[0]
    victim.write_bytes(victim.read_bytes()[:-8] + b"\x00" * 8)

    # When restoring / Then the md5 check refuses
    with pytest.raises(ValueError, match="md5 mismatch"):
        rebuild_frame(form)

    # When exporting again / Then the unreadable base is rebuilt and the form is valid again
    report = export_offsite_form(source, form, rebase_ratio=0.9)
    assert report.rebased
    pd.testing.assert_frame_equal(rebuild_frame(form), pd.read_parquet(source), check_exact=True)


def test_categorical_and_plain_frames_hash_alike() -> None:
    from src.tools.price_history_offsite import _hashes

    # Given the same rows as categorical and as plain strings
    frame = pd.DataFrame({"symbol": pd.Categorical(["a", "b"], categories=["b", "a", "z"]), "date": pd.to_datetime(["2026-01-02"] * 2)})
    plain = frame.assign(symbol=frame["symbol"].astype(str))

    # Then key and row hashes agree (the diff relies on it)
    for left, right in zip(_hashes(frame), _hashes(plain), strict=True):
        assert np.array_equal(left, right)


def test_manifest_without_patch_entry_is_rejected(tmp_path: Path) -> None:
    import json

    from src.tools.price_history_offsite import MANIFEST_NAME, export_offsite_form, read_manifest

    # Given a manifest whose patch entry was dropped
    form = tmp_path / "form"
    export_offsite_form(_write(_panel(SYMBOLS, DAYS), tmp_path / "v1.parquet"), form)
    raw = json.loads((form / MANIFEST_NAME).read_text())
    raw["files"] = [entry for entry in raw["files"] if entry["kind"] != "patch"]
    (form / MANIFEST_NAME).write_text(json.dumps(raw))

    # Then it is refused
    with pytest.raises(ValueError, match="one patch"):
        read_manifest(form)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"symbol": ["a"]}), "no date column"),
        (pd.DataFrame({"symbol": ["a"], "date": [pd.NaT]}), "null dates"),
    ],
)
def test_base_files_refuses_frames_without_valid_dates(frame: pd.DataFrame, message: str) -> None:
    from src.tools.price_history_offsite import base_files

    with pytest.raises(ValueError, match=message):
        base_files(frame)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": 99}, "unknown schema"),
        ({"schema_version": 1, "files": {}, "columns": []}, "no file or column list"),
        ({"schema_version": 1, "files": [{"relpath": "x"}], "columns": [], "rows": 0, "source_sha256": ""}, "malformed manifest"),
        ({"schema_version": 1, "files": [], "columns": [], "source_sha256": ""}, "malformed manifest"),
    ],
)
def test_parse_manifest_rejects_malformed_payloads(payload: dict, message: str) -> None:
    import json

    from src.tools.price_history_offsite import parse_manifest

    with pytest.raises(ValueError, match=message):
        parse_manifest(json.dumps(payload))


def test_rebuild_refuses_row_count_disagreements(tmp_path: Path) -> None:
    import json

    from src.tools.price_history_offsite import MANIFEST_NAME, export_offsite_form, rebuild_frame

    # Given an exported form
    form = tmp_path / "form"
    export_offsite_form(_write(_panel(SYMBOLS, DAYS), tmp_path / "v1.parquet"), form)
    original = (form / MANIFEST_NAME).read_text()

    # When a file's row count is misstated / Then the file is refused
    raw = json.loads(original)
    raw["files"][0]["rows"] += 1
    (form / MANIFEST_NAME).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="row count mismatch"):
        rebuild_frame(form)

    # When the panel total is misstated / Then the rebuild is refused
    raw = json.loads(original)
    raw["rows"] += 1
    (form / MANIFEST_NAME).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not match the manifest"):
        rebuild_frame(form)


def test_rebase_removes_emptied_partition_directories(tmp_path: Path) -> None:
    from src.tools.price_history_offsite import export_offsite_form

    # Given a base spanning two years
    form = tmp_path / "form"
    export_offsite_form(_write(_panel(SYMBOLS, pd.bdate_range("2025-12-22", "2026-01-09")), tmp_path / "v1.parquet"), form)
    assert (form / "base/year=2025").is_dir()

    # When the first year disappears and the base is rebuilt
    report = export_offsite_form(_write(_panel(SYMBOLS, pd.bdate_range("2026-01-05", "2026-01-09")), tmp_path / "v2.parquet"), form)

    # Then its partition files and directory are gone
    assert report.rebased
    assert not (form / "base/year=2025").exists()


def test_failed_rebuild_check_forces_rebase_then_raises(tmp_path: Path, monkeypatch) -> None:
    import src.tools.price_history_offsite as module

    # Given an exported base and a changed source
    form = tmp_path / "form"
    module.export_offsite_form(_write(_panel(SYMBOLS, DAYS), tmp_path / "v1.parquet"), form)
    source = _write(_panel(SYMBOLS, DAYS[:-1]), tmp_path / "v2.parquet")
    verdicts = iter([False, True])
    monkeypatch.setattr(module, "_rebuilds_rows", lambda out_dir, rows: next(verdicts))

    # When the patch form fails the check once / Then the export rebases
    assert module.export_offsite_form(source, form, rebase_ratio=0.9).rebased

    # When even a fresh base fails / Then the export raises
    monkeypatch.setattr(module, "_rebuilds_rows", lambda out_dir, rows: False)
    with pytest.raises(ValueError, match="does not rebuild"):
        module.export_offsite_form(source, form, rebase_ratio=0.9)
