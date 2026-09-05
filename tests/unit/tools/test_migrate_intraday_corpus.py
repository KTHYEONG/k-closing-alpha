from __future__ import annotations


def test_detect_intraday_vendor_identifies_ls_kis_and_rejects_unknown() -> None:
    import pandas as pd
    import pytest

    from src.tools.migrate_intraday_corpus import detect_intraday_vendor

    ls_df = pd.DataFrame({"date": ["20260904"], "time": ["090300"], "open": [1], "high": [1], "low": [1], "close": [1], "jdiff_vol": [1], "value": [1]})
    assert detect_intraday_vendor(ls_df) == "ls"

    kis_df = pd.DataFrame({"stck_cntg_hour": ["090300"], "stck_oprc": ["1"], "stck_hgpr": ["1"], "stck_lwpr": ["1"], "stck_prpr": ["1"], "cntg_vol": ["1"], "acml_tr_pbmn": ["1"]})
    assert detect_intraday_vendor(kis_df) == "kis"

    unknown_df = pd.DataFrame({"weird_col": [1]})
    with pytest.raises(ValueError, match="weird_col"):
        detect_intraday_vendor(unknown_df)


def test_migrate_intraday_partition_file_writes_truncation_audit_without_fabricating_flag(tmp_path, monkeypatch) -> None:
    import json

    import pandas as pd

    from src.data import intraday_store
    from src.tools import migrate_intraday_corpus as mod

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path / "history")

    source = tmp_path / "legacy_ticks.parquet"
    pd.DataFrame(
        {
            "date": ["20260904", "20260904"],
            "time": ["133544", "133600"],
            "close": [10000, 10010],
            "jdiff_vol": [10, 5],
            "종목코드": ["032940", "032940"],
            "스냅샷_날짜": ["2026-09-04", "2026-09-04"],
        }
    ).to_parquet(source, index=False)

    result = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=False)

    assert result["status"] == "migrated"
    audit_path = source.with_suffix(source.suffix + ".truncation_audit.json")
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text())
    assert audit["032940"]["earliest_ts_hms"] == 133544
    assert "truncated" not in audit["032940"]


def test_migrate_intraday_partition_file_rejects_multi_date_file(tmp_path) -> None:
    import pandas as pd

    from src.tools import migrate_intraday_corpus as mod

    source = tmp_path / "bad_partition.parquet"
    pd.DataFrame(
        {
            "stck_cntg_hour": ["090000", "090100"],
            "stck_oprc": ["1", "1"], "stck_hgpr": ["1", "1"], "stck_lwpr": ["1", "1"], "stck_prpr": ["1", "1"],
            "cntg_vol": ["1", "1"], "acml_tr_pbmn": ["1", "2"],
            "종목코드": ["005930", "005930"],
            "스냅샷_날짜": ["2026-09-03", "2026-09-04"],
        }
    ).to_parquet(source, index=False)
    original_bytes = source.read_bytes()

    result = mod.migrate_intraday_partition_file(source, kind="bar", dry_run=False)

    assert result["status"] == "verification_failed"
    assert source.read_bytes() == original_bytes


def _legacy_tick_frame() -> dict:
    return {
        "date": ["20260904", "20260904"],
        "time": ["133544", "133600"],
        "close": [10000, 10010],
        "jdiff_vol": [10, 5],
        "종목코드": ["032940", "032940"],
        "스냅샷_날짜": ["2026-09-04", "2026-09-04"],
    }


def _legacy_bar_frame() -> dict:
    return {
        "stck_cntg_hour": ["090000", "090100"],
        "stck_oprc": ["10000", "10010"], "stck_hgpr": ["10000", "10010"],
        "stck_lwpr": ["10000", "10010"], "stck_prpr": ["10000", "10010"],
        "cntg_vol": ["10", "5"], "acml_tr_pbmn": ["100000", "200000"],
        "종목코드": ["005930", "005930"],
        "스냅샷_날짜": ["2026-09-04", "2026-09-04"],
    }


def test_migrate_intraday_partition_file_reports_missing(tmp_path) -> None:
    from src.tools import migrate_intraday_corpus as mod

    assert mod.migrate_intraday_partition_file(tmp_path / "absent.parquet", kind="tick")["status"] == "missing"


def test_migrate_intraday_partition_file_rejects_unknown_kind(tmp_path) -> None:
    import pytest

    from src.tools import migrate_intraday_corpus as mod

    with pytest.raises(ValueError, match="kind"):
        mod.migrate_intraday_partition_file(tmp_path / "f.parquet", kind="1h")


def test_migrate_intraday_partition_file_detects_already_migrated(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.data import intraday_store
    from src.tools import migrate_intraday_corpus as mod

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path / "history")
    store_dir = tmp_path / "history" / "intraday" / "ticks" / "regular" / "2026-09"
    store_dir.mkdir(parents=True)
    source = store_dir / "2026-09-04.parquet"
    pd.DataFrame(_legacy_tick_frame()).to_parquet(source, index=False)

    first = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=False)

    assert first["status"] == "migrated"
    assert pq.ParquetFile(source).metadata.row_group(0).column(0).compression.upper() == "ZSTD"
    stored = pd.read_parquet(source)
    assert "jdiff_vol" not in stored.columns
    assert set(stored.columns) == {
        "snapshot_date", "symbol", "ts_hms", "price", "volume",
        "trade_strength", "ask1", "bid1", "truncated", "vendor",
    }

    result = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=False)

    assert result["status"] == "already_migrated"


def test_migrate_intraday_partition_file_rejects_empty_frame(tmp_path) -> None:
    import pandas as pd

    from src.tools import migrate_intraday_corpus as mod

    source = tmp_path / "empty.parquet"
    pd.DataFrame({"종목코드": pd.Series([], dtype="str"), "스냅샷_날짜": pd.Series([], dtype="str")}).to_parquet(
        source, index=False
    )

    result = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=False)

    assert result["status"] == "verification_failed"


def test_migrate_intraday_partition_file_rejects_missing_keys(tmp_path) -> None:
    import pandas as pd

    from src.tools import migrate_intraday_corpus as mod

    source = tmp_path / "nokeys.parquet"
    pd.DataFrame({"time": ["090300"], "close": [100]}).to_parquet(source, index=False)
    original_bytes = source.read_bytes()

    result = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=False)

    assert result["status"] == "verification_failed"
    assert source.read_bytes() == original_bytes


def test_migrate_intraday_partition_file_rejects_unknown_vendor_rows(tmp_path) -> None:
    import pandas as pd

    from src.tools import migrate_intraday_corpus as mod

    source = tmp_path / "weird.parquet"
    pd.DataFrame({"weird_col": [1], "종목코드": ["005930"], "스냅샷_날짜": ["2026-09-04"]}).to_parquet(
        source, index=False
    )
    original_bytes = source.read_bytes()

    result = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=False)

    assert result["status"] == "verification_failed"
    assert source.read_bytes() == original_bytes


def test_migrate_intraday_partition_file_dry_run_never_writes(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.data import intraday_store
    from src.tools import migrate_intraday_corpus as mod

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path / "history")
    source = tmp_path / "legacy_ticks.parquet"
    pd.DataFrame(_legacy_tick_frame()).to_parquet(source, index=False)
    original_bytes = source.read_bytes()

    result = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=True)

    assert result["status"] == "would_migrate"
    assert result["rows"] == 2
    assert source.read_bytes() == original_bytes
    assert not source.with_suffix(".parquet.bak").exists()
    assert not source.with_suffix(source.suffix + ".truncation_audit.json").exists()


def test_migrate_intraday_partition_file_never_overwrites_existing_backup(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.data import intraday_store
    from src.tools import migrate_intraday_corpus as mod

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path / "history")
    source = tmp_path / "legacy_ticks.parquet"
    pd.DataFrame(_legacy_tick_frame()).to_parquet(source, index=False)
    bak = source.with_suffix(".parquet.bak")
    bak.write_bytes(b"prior-backup")

    result = mod.migrate_intraday_partition_file(source, kind="tick", dry_run=False)

    assert result["status"] == "migrated"
    assert bak.read_bytes() == b"prior-backup"


def test_migrate_intraday_partition_file_migrates_bar_successfully(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.data import intraday_store
    from src.tools import migrate_intraday_corpus as mod

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path / "history")
    bar_dir = tmp_path / "1m"
    bar_dir.mkdir()
    source = bar_dir / "legacy_bars.parquet"
    pd.DataFrame(_legacy_bar_frame()).to_parquet(source, index=False)

    result = mod.migrate_intraday_partition_file(source, kind="bar", dry_run=False)

    assert result["status"] == "migrated"
    assert result["rows"] == 2
    target = intraday_store.intraday_partition_path(1, "2026-09-04", "regular")
    assert target.exists()
    stored = pd.read_parquet(target)
    assert set(stored["symbol"]) == {"005930"}
    assert source.with_suffix(".parquet.bak").exists()


def test_migrate_intraday_partition_file_migrates_nxt_aftermarket_without_corrupting_regular(tmp_path, monkeypatch) -> None:
    """회귀 테스트: nxt_aftermarket 원본이 무조건 'regular'로 하드코딩되어 실제로는 마이그레이션되지 않고
    같은 날짜의 regular 파티션이 오염되던 버그. 세션은 반드시 원본 경로에서 그대로 추출해야 한다."""
    import pandas as pd

    from src.data import intraday_store
    from src.tools import migrate_intraday_corpus as mod

    history = tmp_path / "history"
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", history)

    # Given: 같은 날짜에 regular 파티션이 이미 존재
    regular_target = intraday_store.intraday_partition_path(1, "2026-09-04", "regular")
    regular_target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_legacy_bar_frame()).to_parquet(regular_target, index=False)
    regular_bytes_before = regular_target.read_bytes()

    nxt_source = history / "intraday" / "1m" / "nxt_aftermarket" / "2026-09" / "2026-09-04.parquet"
    nxt_source.parent.mkdir(parents=True, exist_ok=True)
    nxt_frame = dict(_legacy_bar_frame())
    nxt_frame["종목코드"] = ["000660", "000660"]
    pd.DataFrame(nxt_frame).to_parquet(nxt_source, index=False)

    # When
    result = mod.migrate_intraday_partition_file(nxt_source, kind="bar", dry_run=False)

    # Then: regular 파티션은 손대지 않고 원본(nxt) 파일 그 자리가 정준 스키마로 교체된다.
    assert result["status"] == "migrated"
    assert regular_target.read_bytes() == regular_bytes_before
    assert mod._stored_codec(nxt_source) == "ZSTD"
    migrated_nxt = pd.read_parquet(nxt_source)
    assert set(migrated_nxt["symbol"]) == {"000660"}
    assert intraday_store.intraday_partition_path(1, "2026-09-04", "nxt_aftermarket").resolve() == nxt_source.resolve()


def test_migrate_intraday_bar_defaults_to_1m_interval_without_path_hint(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.data import intraday_store
    from src.tools import migrate_intraday_corpus as mod

    assert mod._infer_bar_interval_minutes(tmp_path / "plain.parquet") == 1
    assert mod._infer_bar_interval_minutes(tmp_path / "5m" / "plain.parquet") == 5

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path / "history")
    source = tmp_path / "legacy_bars.parquet"
    pd.DataFrame(_legacy_bar_frame()).to_parquet(source, index=False)

    result = mod.migrate_intraday_partition_file(source, kind="bar", dry_run=False)

    assert result["status"] == "migrated"
    assert intraday_store.intraday_partition_path(1, "2026-09-04", "regular").exists()


def test_migrate_intraday_stored_codec_handles_unreadable_file(tmp_path) -> None:
    from src.tools import migrate_intraday_corpus as mod

    path = tmp_path / "note.parquet"
    path.write_text("not a parquet file", encoding="utf-8")

    assert mod._stored_codec(path) == ""


def test_migrate_intraday_stored_codec_handles_zero_row_groups(tmp_path, monkeypatch) -> None:
    from src.tools import migrate_intraday_corpus as mod

    class _Meta:
        num_row_groups = 0

    class _File:
        metadata = _Meta()

    class _FakePq:
        @staticmethod
        def ParquetFile(path):  # noqa: N802
            return _File()

    monkeypatch.setattr(mod, "pq", _FakePq)

    assert mod._stored_codec(tmp_path / "empty.parquet") == ""


def test_migrate_intraday_corpus_main_dry_run(tmp_path, monkeypatch, capsys) -> None:
    import sys

    import pandas as pd

    from src.tools import migrate_intraday_corpus as mod

    history = tmp_path / "history"
    tick_dir = history / "intraday" / "ticks"
    tick_dir.mkdir(parents=True)
    pd.DataFrame(_legacy_tick_frame()).to_parquet(tick_dir / "legacy_ticks.parquet", index=False)
    monkeypatch.setattr(mod.settings, "HISTORY_DIR", history)
    monkeypatch.setattr(sys, "argv", ["migrate_intraday_corpus", "--dry-run"])

    mod.main()

    out = capsys.readouterr().out
    assert "[DATA]" in out
    assert "would_migrate" in out
    assert not (history / "intraday" / "_migration_manifest.json").exists()


def test_migrate_intraday_corpus_main_migrates_and_writes_manifest(tmp_path, monkeypatch, capsys) -> None:
    import json
    import sys

    import pandas as pd

    from src.tools import migrate_intraday_corpus as mod

    history = tmp_path / "history"
    bar_dir = history / "intraday" / "1m"
    bar_dir.mkdir(parents=True)
    pd.DataFrame(_legacy_bar_frame()).to_parquet(bar_dir / "legacy_bars.parquet", index=False)
    monkeypatch.setattr(mod.settings, "HISTORY_DIR", history)
    monkeypatch.setattr(sys, "argv", ["migrate_intraday_corpus", "--no-backup"])

    mod.main()

    out = capsys.readouterr().out
    assert "[DATA]" in out
    assert "migrated" in out
    manifest = json.loads((history / "intraday" / "_migration_manifest.json").read_text())
    assert len(manifest["files"]) == 1


def test_migrate_intraday_corpus_main_empty_dir_writes_empty_manifest(tmp_path, monkeypatch, capsys) -> None:
    import json
    import sys

    from src.tools import migrate_intraday_corpus as mod

    history = tmp_path / "history"
    monkeypatch.setattr(mod.settings, "HISTORY_DIR", history)
    monkeypatch.setattr(sys, "argv", ["migrate_intraday_corpus"])

    mod.main()

    manifest = json.loads((history / "intraday" / "_migration_manifest.json").read_text())
    assert manifest["files"] == []
    assert "[DATA]" not in capsys.readouterr().out
