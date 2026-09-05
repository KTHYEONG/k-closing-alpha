from __future__ import annotations


def _panel_cols() -> dict:
    import pandas as pd

    return {
        "date": pd.to_datetime(["2026-01-02", "2026-01-01"]),
        "symbol": ["000660", "005930"],
        "program_net_value": [-193758630, 42251260],
    }


def test_migrate_altdata_panel_file_migrates_successfully_with_backup(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.tools.migrate_altdata_storage import migrate_altdata_panel_file

    path = tmp_path / "program_trade_daily.parquet"
    pd.DataFrame(_panel_cols()).to_parquet(path, index=False)

    result = migrate_altdata_panel_file(path, dry_run=False)

    assert result["status"] == "migrated"
    assert result["rows"] == 2
    assert path.with_suffix(".parquet.bak").exists()
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression.upper() == "ZSTD"
    stored = pd.read_parquet(path)
    assert stored["symbol"].tolist() == ["000660", "005930"]
    assert stored["program_net_value"].tolist() == [-193758630, 42251260]


def test_migrate_altdata_panel_file_dry_run_never_writes(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_altdata_storage import migrate_altdata_panel_file

    path = tmp_path / "shorting.parquet"
    pd.DataFrame(_panel_cols()).to_parquet(path, index=False)
    original_bytes = path.read_bytes()

    result = migrate_altdata_panel_file(path, dry_run=True)

    assert result["status"] == "would_migrate"
    assert path.read_bytes() == original_bytes
    assert not path.with_suffix(".parquet.bak").exists()


def test_migrate_altdata_panel_file_detects_already_migrated(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_altdata_storage import migrate_altdata_panel_file

    path = tmp_path / "shorting.parquet"
    pd.DataFrame(_panel_cols()).to_parquet(path, index=False)
    assert migrate_altdata_panel_file(path, dry_run=False)["status"] == "migrated"

    result = migrate_altdata_panel_file(path, dry_run=False)

    assert result["status"] == "already_migrated"


def test_migrate_altdata_panel_file_reports_missing(tmp_path) -> None:
    from src.tools.migrate_altdata_storage import migrate_altdata_panel_file

    assert migrate_altdata_panel_file(tmp_path / "absent.parquet")["status"] == "missing"


def test_migrate_altdata_panel_file_respects_no_backup(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_altdata_storage import migrate_altdata_panel_file

    path = tmp_path / "shorting.parquet"
    pd.DataFrame(_panel_cols()).to_parquet(path, index=False)

    result = migrate_altdata_panel_file(path, dry_run=False, backup=False)

    assert result["status"] == "migrated"
    assert not path.with_suffix(".parquet.bak").exists()


def test_migrate_altdata_panel_file_never_overwrites_existing_backup(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_altdata_storage import migrate_altdata_panel_file

    path = tmp_path / "shorting.parquet"
    pd.DataFrame(_panel_cols()).to_parquet(path, index=False)
    bak = path.with_suffix(".parquet.bak")
    bak.write_bytes(b"prior-backup")

    result = migrate_altdata_panel_file(path, dry_run=False)

    assert result["status"] == "migrated"
    assert bak.read_bytes() == b"prior-backup"


def test_migrate_altdata_panel_file_skips_write_on_verification_failure(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.tools.migrate_altdata_storage as mod

    path = tmp_path / "shorting.parquet"
    pd.DataFrame(_panel_cols()).to_parquet(path, index=False)
    original_bytes = path.read_bytes()

    def corrupt_downcast(df):
        out = df.copy()
        out["program_net_value"] = out["program_net_value"] + 1
        return out

    monkeypatch.setattr(mod, "downcast_altdata_panel_frame", corrupt_downcast)

    result = mod.migrate_altdata_panel_file(path, dry_run=False)

    assert result["status"] == "verification_failed"
    assert path.read_bytes() == original_bytes
    assert not path.with_suffix(".parquet.bak").exists()


def test_verify_altdata_panel_detects_problems() -> None:
    import pandas as pd

    import src.tools.migrate_altdata_storage as mod

    frame = pd.DataFrame(_panel_cols())
    assert mod._verify_altdata_panel(frame, frame.sort_values("date").reset_index(drop=True)) is None
    assert mod._verify_altdata_panel(frame, frame.iloc[:1]) is not None

    renamed = frame.rename(columns={"program_net_value": "other"})
    assert mod._verify_altdata_panel(frame, renamed) is not None

    drifted = frame.copy()
    drifted.loc[0, "program_net_value"] = 0
    assert mod._verify_altdata_panel(frame, drifted) is not None

    keyless = pd.DataFrame({"value": [1, 2]})
    assert mod._verify_altdata_panel(keyless, keyless) is None


def test_stored_codec_handles_unreadable_file(tmp_path) -> None:
    import src.tools.migrate_altdata_storage as mod

    path = tmp_path / "note.parquet"
    path.write_text("not a parquet file", encoding="utf-8")

    assert mod._stored_codec(path) == ""


def test_stored_codec_handles_zero_row_groups(tmp_path, monkeypatch) -> None:
    import src.tools.migrate_altdata_storage as mod

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


def test_migrate_altdata_main_reports_missing_panels(tmp_path, monkeypatch, capsys) -> None:
    import sys

    import src.tools.migrate_altdata_storage as mod

    monkeypatch.setattr(mod.settings, "ALTDATA_DIR", tmp_path / "altdata")
    monkeypatch.setattr(sys, "argv", ["migrate_altdata_storage", "--dry-run"])

    mod.main()

    out = capsys.readouterr().out
    assert out.count("[DATA]") == len(mod._ALTDATA_PANELS)
    assert "status=missing" in out


def test_migrate_altdata_main_migrates_existing_panel(tmp_path, monkeypatch, capsys) -> None:
    import sys

    import pandas as pd

    import src.tools.migrate_altdata_storage as mod

    out_dir = tmp_path / "altdata"
    out_dir.mkdir()
    pd.DataFrame(_panel_cols()).to_parquet(out_dir / "shorting.parquet", index=False)
    monkeypatch.setattr(mod.settings, "ALTDATA_DIR", out_dir)
    monkeypatch.setattr(sys, "argv", ["migrate_altdata_storage", "--no-backup"])

    mod.main()

    out = capsys.readouterr().out
    assert "status=migrated" in out
    assert "status=missing" in out
