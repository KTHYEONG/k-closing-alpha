


def test_classify_partition_distinguishes_canonical_legacy_and_unknown() -> None:
    from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, CANONICAL_TICK_COLUMNS
    from src.tools.migrate_intraday_schema import classify_partition

    assert classify_partition(list(CANONICAL_BAR_COLUMNS), is_tick=False) == "canonical"
    assert classify_partition(list(CANONICAL_TICK_COLUMNS), is_tick=True) == "canonical"
    assert classify_partition(["stck_cntg_hour", "stck_prpr", "종목코드"], is_tick=False) == "legacy_kis"
    # KIS/LS 혼합(22컬럼 파티션)은 기존 cost_model 판별 관례대로 LS 우선
    assert classify_partition(["jdiff_vol", "stck_cntg_hour", "time", "종목코드"], is_tick=False) == "legacy_ls"
    assert classify_partition(["foo", "bar"], is_tick=False) == "unknown"


def test_migrate_frame_normalizes_per_symbol_without_mixing_cumulative_series() -> None:
    import pandas as pd

    from src.data.intraday_schema import CANONICAL_BAR_COLUMNS
    from src.tools.migrate_intraday_schema import migrate_frame

    raw = pd.DataFrame({
        "stck_bsop_date": ["20260501"] * 4,
        "stck_cntg_hour": ["090100", "090200", "090100", "090200"],
        "stck_oprc": ["1000", "1010", "500", "505"],
        "stck_hgpr": ["1010", "1020", "510", "515"],
        "stck_lwpr": ["995", "1005", "495", "500"],
        "stck_prpr": ["1005", "1015", "505", "510"],
        "cntg_vol": ["100", "200", "10", "20"],
        "acml_tr_pbmn": ["100000", "300000", "5000", "15000"],
        "종목코드": ["005930", "005930", "000660", "000660"],
    })

    out = migrate_frame(raw, "legacy_kis", "2026-05-01", is_tick=False)

    assert list(out.columns) == list(CANONICAL_BAR_COLUMNS)
    assert len(out) == 4
    a = out[out["symbol"] == "005930"].sort_values("ts_hms")
    b = out[out["symbol"] == "000660"].sort_values("ts_hms")
    assert a["value_krw"].tolist() == [100000, 200000]
    assert b["value_krw"].tolist() == [5000, 10000]


def test_migrate_frame_rejects_unknown_kind_and_missing_symbol_column() -> None:
    import pandas as pd
    import pytest

    from src.tools.migrate_intraday_schema import migrate_frame

    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts fail-closed kind
        migrate_frame(pd.DataFrame({"foo": [1]}), "unknown", "2026-05-01", is_tick=False)

    no_symbol = pd.DataFrame({
        "stck_bsop_date": ["20260501"],
        "stck_cntg_hour": ["090100"],
        "stck_oprc": ["1000"], "stck_hgpr": ["1010"], "stck_lwpr": ["995"], "stck_prpr": ["1005"],
        "cntg_vol": ["100"], "acml_tr_pbmn": ["100000"],
    })
    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts fail-closed symbol
        migrate_frame(no_symbol, "legacy_kis", "2026-05-01", is_tick=False)


def test_migrate_partition_file_dry_run_preserves_file_and_reports_counts(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_intraday_schema import migrate_partition_file

    path = tmp_path / "2026-05-01.parquet"
    pd.DataFrame({
        "stck_bsop_date": ["20260501", "20250829"],
        "stck_cntg_hour": ["090100", "090200"],
        "stck_oprc": ["1000", "1"], "stck_hgpr": ["1010", "1"],
        "stck_lwpr": ["995", "1"], "stck_prpr": ["1005", "1"],
        "cntg_vol": ["100", "1"], "acml_tr_pbmn": ["100000", "1"],
        "종목코드": ["005930", "005930"],
    }).to_parquet(path, index=False)

    res = migrate_partition_file(path, dry_run=True, backup=True)

    assert res["kind"] == "legacy_kis"
    assert res["n_before"] == 2
    assert res["n_after"] == 1
    assert res["n_date_dropped"] == 1
    assert res["migrated"] is False
    assert "stck_cntg_hour" in pd.read_parquet(path).columns
    assert not (tmp_path / "2026-05-01.parquet.legacy.bak").exists()


def test_migrate_partition_file_apply_writes_canonical_and_backs_up_original(tmp_path) -> None:
    import pandas as pd

    from src.data.intraday_schema import CANONICAL_BAR_COLUMNS
    from src.tools.migrate_intraday_schema import migrate_partition_file

    path = tmp_path / "2026-05-01.parquet"
    pd.DataFrame({
        "stck_bsop_date": ["20260501", "20260501"],
        "stck_cntg_hour": ["090100", "090200"],
        "stck_oprc": ["1000", "1010"], "stck_hgpr": ["1010", "1020"],
        "stck_lwpr": ["995", "1005"], "stck_prpr": ["1005", "1015"],
        "cntg_vol": ["100", "200"], "acml_tr_pbmn": ["100000", "300000"],
        "종목코드": ["005930", "005930"],
    }).to_parquet(path, index=False)

    res = migrate_partition_file(path, dry_run=False, backup=True)

    assert res["migrated"] is True
    assert res["n_before"] == 2 and res["n_after"] == 2 and res["n_date_dropped"] == 0
    out = pd.read_parquet(path)
    assert list(out.columns) == list(CANONICAL_BAR_COLUMNS)
    assert out["symbol"].tolist() == ["005930", "005930"]
    backup = tmp_path / "2026-05-01.parquet.legacy.bak"
    assert backup.exists()
    assert "stck_cntg_hour" in pd.read_parquet(backup).columns


def test_migrate_partition_file_skips_already_canonical(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_intraday_schema import migrate_partition_file

    path = tmp_path / "2026-05-01.parquet"
    pd.DataFrame({
        "snapshot_date": ["2026-05-01"],
        "symbol": ["005930"],
        "ts_hms": [90100],
        "open": [1000], "high": [1010], "low": [995], "close": [1005],
        "volume": [100], "value_krw": [100000],
        "has_trade": [True], "vendor": ["kis"],
    }).to_parquet(path, index=False)

    res = migrate_partition_file(path, dry_run=False, backup=True)

    assert res["migrated"] is False
    assert res["reason"] == "already_canonical"
    assert not (tmp_path / "2026-05-01.parquet.legacy.bak").exists()


def test_run_migration_aggregates_and_isolates_file_failures(tmp_path) -> None:
    import pandas as pd

    from src.tools.migrate_intraday_schema import run_migration

    good_dir = tmp_path / "1m" / "regular" / "2026-05"
    good_dir.mkdir(parents=True)
    pd.DataFrame({
        "stck_bsop_date": ["20260501"],
        "stck_cntg_hour": ["090100"],
        "stck_oprc": ["1000"], "stck_hgpr": ["1010"], "stck_lwpr": ["995"], "stck_prpr": ["1005"],
        "cntg_vol": ["100"], "acml_tr_pbmn": ["100000"],
        "종목코드": ["005930"],
    }).to_parquet(good_dir / "2026-05-01.parquet", index=False)

    bad_dir = tmp_path / "1m" / "regular" / "2026-06"
    bad_dir.mkdir(parents=True)
    pd.DataFrame({"foo": [1], "bar": [2]}).to_parquet(bad_dir / "2026-06-01.parquet", index=False)

    res = run_migration(root=tmp_path, dry_run=False, backup=False)

    assert res["n_files"] == 2
    assert res["n_migrated"] == 1
    assert res["n_failed"] == 1
    assert len(res["failures"]) == 1
    assert "2026-06-01" in str(res["failures"][0]["path"])


def test_count_stale_business_date_rows_is_measured_from_source_not_derived() -> None:
    """행수 불변식이 항등식이 되지 않도록 게이트 제거건수를 원천에서 독립 산출한다."""
    import pandas as pd

    from src.tools.migrate_intraday_schema import count_stale_business_date_rows

    raw = pd.DataFrame({"stck_bsop_date": ["20260501", "20250829", "20260501", "20250830"]})
    assert count_stale_business_date_rows(raw, "legacy_kis", "2026-05-01") == 2
    # 영업일 필드가 없으면 기대 제거건수 0 (검증 불가 구간)
    assert count_stale_business_date_rows(pd.DataFrame({"stck_prpr": ["1"]}), "legacy_kis", "2026-05-01") == 0


def test_migrate_partition_file_flags_row_count_mismatch_without_writing(tmp_path, monkeypatch) -> None:
    """게이트 제거분으로 설명되지 않는 정규화 손실은 쓰기 없이 row_count_mismatch로 거부한다."""
    import pandas as pd

    from src.tools import migrate_intraday_schema as mod

    path = tmp_path / "2026-05-01.parquet"
    original = pd.DataFrame({
        "stck_bsop_date": ["20260501", "20260501"],
        "stck_cntg_hour": ["090100", "090200"],
        "stck_oprc": ["1000", "1010"], "stck_hgpr": ["1010", "1020"],
        "stck_lwpr": ["995", "1005"], "stck_prpr": ["1005", "1015"],
        "cntg_vol": ["100", "200"], "acml_tr_pbmn": ["100000", "300000"],
        "종목코드": ["005930", "005930"],
    })
    original.to_parquet(path, index=False)

    real_migrate = mod.migrate_frame

    def _lossy(raw, kind, snapshot_date, *, is_tick):
        return real_migrate(raw, kind, snapshot_date, is_tick=is_tick).head(1)

    monkeypatch.setattr(mod, "migrate_frame", _lossy)

    res = mod.migrate_partition_file(path, dry_run=False, backup=True)

    assert res["reason"] == "row_count_mismatch"
    assert res["migrated"] is False
    assert res["n_date_dropped"] == 0
    assert "stck_cntg_hour" in pd.read_parquet(path).columns
    assert not (tmp_path / "2026-05-01.parquet.legacy.bak").exists()


def test_migrate_partition_file_allows_repeated_tick_timestamps(tmp_path) -> None:
    """같은 초에 복수 체결이 정상인 틱 파티션이 duplicate_key로 거부되지 않아야 한다."""
    import pandas as pd

    from src.data.intraday_schema import CANONICAL_TICK_COLUMNS
    from src.tools.migrate_intraday_schema import migrate_partition_file

    tick_dir = tmp_path / "ticks" / "regular" / "2026-05"
    tick_dir.mkdir(parents=True)
    path = tick_dir / "2026-05-01.parquet"
    pd.DataFrame({
        "stck_bsop_date": ["20260501"] * 3,
        "stck_cntg_hour": ["090100", "090100", "090100"],
        "stck_prpr": ["1005", "1006", "1007"],
        "cnqn": ["10", "20", "30"],
        "종목코드": ["005930"] * 3,
    }).to_parquet(path, index=False)

    res = migrate_partition_file(path, dry_run=False, backup=False)

    assert res["reason"] is None
    assert res["migrated"] is True
    assert res["n_after"] == 3
    out = pd.read_parquet(path)
    assert list(out.columns) == list(CANONICAL_TICK_COLUMNS)
    assert out["volume"].tolist() == [10, 20, 30]
