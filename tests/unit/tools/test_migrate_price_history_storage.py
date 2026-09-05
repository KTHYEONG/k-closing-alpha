from __future__ import annotations


def test_migrate_price_history_file_dry_run_never_writes(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_price_history_storage import migrate_price_history_file

    path = tmp_path / "price_history.parquet"
    cols = {
        "date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"],
        "open": [70000.0], "high": [70500.0], "low": [69800.0], "close": [70200.0], "prev_close": [69200.0],
        "market_cap_100m": [1.0], "trade_value_100m": [1.0], "daily_change_pct": [0.01],
        "market": ["KOSPI"], "volume": [1000],
        "foreign_netbuy": [1.0], "inst_netbuy": [1.0], "program_netbuy": [1.0],
        "kospi_pct": [0.01], "kosdaq_pct": [0.01], "v_kospi": [18.5], "v_kosdaq": [18.5],
    }
    pd.DataFrame(cols).to_parquet(path, index=False)
    original_bytes = path.read_bytes()

    result = migrate_price_history_file(path, dry_run=True)

    assert result["status"] in ("would_migrate", "migrated")
    assert path.read_bytes() == original_bytes
    assert not path.with_suffix(".parquet.bak").exists()


def test_migrate_price_history_file_skips_write_on_verification_failure(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.tools.migrate_price_history_storage as mod

    path = tmp_path / "price_history.parquet"
    cols = {
        "date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"],
        "open": [70000.0], "high": [70500.0], "low": [69800.0], "close": [70200.0], "prev_close": [69200.0],
        "market_cap_100m": [1.0], "trade_value_100m": [1.0], "daily_change_pct": [0.01],
        "market": ["KOSPI"], "volume": [1000],
        "foreign_netbuy": [1.0], "inst_netbuy": [1.0], "program_netbuy": [1.0],
        "kospi_pct": [0.01], "kosdaq_pct": [0.01], "v_kospi": [18.5], "v_kosdaq": [18.5],
    }
    pd.DataFrame(cols).to_parquet(path, index=False)
    original_bytes = path.read_bytes()

    def corrupt_downcast(df):
        out = df.copy()
        out["daily_change_pct"] = 999.0  # 검증 실패를 강제로 유발
        return out

    monkeypatch.setattr(mod, "downcast_price_history_frame", corrupt_downcast)

    result = mod.migrate_price_history_file(path, dry_run=False)

    assert result["status"] == "verification_failed"
    assert path.read_bytes() == original_bytes
    assert not path.with_suffix(".parquet.bak").exists()


def _price_cols() -> dict:
    import pandas as pd

    return {
        "date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"],
        "open": [70000.0], "high": [70500.0], "low": [69800.0], "close": [70200.0], "prev_close": [69200.0],
        "market_cap_100m": [1.0], "trade_value_100m": [1.0], "daily_change_pct": [0.01],
        "market": ["KOSPI"], "volume": [1000],
        "foreign_netbuy": [1.0], "inst_netbuy": [1.0], "program_netbuy": [1.0],
        "kospi_pct": [0.01], "kosdaq_pct": [0.01], "v_kospi": [18.5], "v_kosdaq": [18.5],
    }


def test_migrate_price_history_file_migrates_successfully_with_backup(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.tools.migrate_price_history_storage import migrate_price_history_file

    path = tmp_path / "price_history.parquet"
    pd.DataFrame(_price_cols()).to_parquet(path, index=False)

    result = migrate_price_history_file(path, dry_run=False)

    assert result["status"] == "migrated"
    assert result["rows"] == 1
    assert result["new_size_bytes"] > 0
    bak = path.with_suffix(".parquet.bak")
    assert bak.exists()
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression.upper() == "ZSTD"
    stored = pd.read_parquet(path)
    assert str(stored["open"].dtype) == "Int32"
    assert str(stored["daily_change_pct"].dtype) == "float64"


def test_migrate_price_history_file_detects_already_migrated(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_price_history_storage import migrate_price_history_file

    path = tmp_path / "price_history.parquet"
    pd.DataFrame(_price_cols()).to_parquet(path, index=False)
    assert migrate_price_history_file(path, dry_run=False)["status"] == "migrated"

    result = migrate_price_history_file(path, dry_run=False)

    assert result["status"] == "already_migrated"
    assert result["new_size_bytes"] == result["original_size_bytes"]


def test_migrate_price_history_file_reports_missing(tmp_path) -> None:
    from src.tools.migrate_price_history_storage import migrate_price_history_file

    result = migrate_price_history_file(tmp_path / "absent.parquet", dry_run=False)

    assert result["status"] == "missing"


def test_migrate_price_history_file_respects_no_backup(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_price_history_storage import migrate_price_history_file

    path = tmp_path / "price_history.parquet"
    pd.DataFrame(_price_cols()).to_parquet(path, index=False)

    result = migrate_price_history_file(path, dry_run=False, backup=False)

    assert result["status"] == "migrated"
    assert not path.with_suffix(".parquet.bak").exists()


def test_migrate_price_history_file_never_overwrites_existing_backup(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_price_history_storage import migrate_price_history_file

    path = tmp_path / "price_history.parquet"
    pd.DataFrame(_price_cols()).to_parquet(path, index=False)
    bak = path.with_suffix(".parquet.bak")
    bak.write_bytes(b"prior-backup")

    result = migrate_price_history_file(path, dry_run=False)

    assert result["status"] == "migrated"
    assert bak.read_bytes() == b"prior-backup"


def test_stored_codec_handles_unreadable_file(tmp_path) -> None:
    import src.tools.migrate_price_history_storage as mod

    path = tmp_path / "note.parquet"
    path.write_text("not a parquet file", encoding="utf-8")

    assert mod._stored_codec(path) == ""


def test_stored_codec_handles_zero_row_groups(tmp_path, monkeypatch) -> None:
    import src.tools.migrate_price_history_storage as mod

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


def test_verify_price_history_detects_problems() -> None:
    import pandas as pd

    import src.tools.migrate_price_history_storage as mod

    keys = {"date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"]}
    thin = pd.DataFrame(keys)
    assert mod._verify_price_history(thin, thin) is None
    assert mod._verify_price_history(thin, pd.concat([thin, thin], ignore_index=True)) is not None

    keyless = pd.DataFrame({"value": [1.0]})
    assert mod._verify_price_history(keyless, keyless) is None

    drifted = pd.DataFrame({**keys, "open": pd.array([70001], dtype="Int32")})
    assert "open" in (mod._verify_price_history(pd.DataFrame({**keys, "open": [70000.0]}), drifted) or "")

    nan_mismatch = pd.DataFrame({**keys, "market_cap_100m": [float("nan")]})
    assert "market_cap_100m" in (mod._verify_price_history(pd.DataFrame({**keys, "market_cap_100m": [1.0]}), nan_mismatch) or "")

    rel_err = pd.DataFrame({**keys, "market_cap_100m": [2.0]})
    assert "market_cap_100m" in (mod._verify_price_history(pd.DataFrame({**keys, "market_cap_100m": [1.0]}), rel_err) or "")

    retained = pd.DataFrame({**keys, "daily_change_pct": [0.02]})
    assert "daily_change_pct" in (mod._verify_price_history(pd.DataFrame({**keys, "daily_change_pct": [0.01]}), retained) or "")


def test_migrate_price_history_main_dry_run(tmp_path, monkeypatch, capsys) -> None:
    import sys

    import pandas as pd

    import src.tools.migrate_price_history_storage as mod

    path = tmp_path / "price_history.parquet"
    pd.DataFrame(_price_cols()).to_parquet(path, index=False)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", path)
    monkeypatch.setattr(sys, "argv", ["migrate_price_history_storage", "--dry-run"])

    mod.main()

    out = capsys.readouterr().out
    assert "[DATA]" in out
    assert "would_migrate" in out
    assert not path.with_suffix(".parquet.bak").exists()


def test_migrate_price_history_main_migrates_without_backup(tmp_path, monkeypatch, capsys) -> None:
    import sys

    import pandas as pd

    import src.tools.migrate_price_history_storage as mod

    path = tmp_path / "price_history.parquet"
    pd.DataFrame(_price_cols()).to_parquet(path, index=False)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", path)
    monkeypatch.setattr(sys, "argv", ["migrate_price_history_storage", "--no-backup"])

    mod.main()

    out = capsys.readouterr().out
    assert "[DATA]" in out
    assert "migrated" in out
    assert not path.with_suffix(".parquet.bak").exists()
