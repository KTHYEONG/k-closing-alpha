from __future__ import annotations

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


def test_intraday_store_write_and_range_read_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    df_in = _canon_bar("005930")
    rows_written = intraday_store.write_intraday_partition(df_in, 1, "2026-09-03", "regular")
    assert rows_written == 1

    df_out_of_range = _canon_bar("000660", "2026-08-01")
    intraday_store.write_intraday_partition(df_out_of_range, 1, "2026-08-01", "regular")

    result = intraday_store.read_intraday_range(1, "2026-09-01", "2026-09-30", session="regular")

    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "005930"


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


def test_merge_partition_frame_recovers_from_unreadable_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """기존 파티션 파일이 손상되어 읽기 실패해도 신규 데이터만으로 안전하게 계속 진행한다."""
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    target = intraday_store.intraday_partition_path(1, "2026-09-05", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not a valid parquet file")

    new_df = _canon_bar("005930", "2026-09-05")
    merged = intraday_store.merge_partition_frame(new_df, target, ("symbol", "ts_hms"))

    assert len(merged) == 1
    assert merged.iloc[0]["symbol"] == "005930"


def test_write_intraday_partition_rejects_symbol_coverage_reduction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """기존 파티션에 있던 종목이 신규 병합 결과에서 사라지면 커버리지 축소로 거부한다."""
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    first = _canon_bar("005930", "2026-09-06")
    intraday_store.write_intraday_partition(first, 1, "2026-09-06", "regular")

    target = intraday_store.intraday_partition_path(1, "2026-09-06", "regular")
    # 서로 다른 종목이지만 동일 ts_hms를 가지는 신규 프레임을 symbol을 뺀 key_cols로
    # 병합하면 drop_duplicates가 기존 종목 행을 통째로 지워버리는 오용 시나리오를 재현한다.
    colliding = _canon_bar("000660", "2026-09-06")
    colliding["ts_hms"] = first.iloc[0]["ts_hms"]

    with pytest.raises(ValueError, match="reduce symbol coverage"):
        intraday_store.merge_partition_frame(colliding, target, ("ts_hms",))



def test_merge_partition_frame_rejects_legacy_existing_partition(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.data.intraday_store import merge_partition_frame

    legacy = pd.DataFrame({
        "stck_bsop_date": ["20260501"] * 3,
        "stck_cntg_hour": ["090100", "090200", "090300"],
        "stck_prpr": ["1000", "1010", "1020"],
        "종목코드": ["005930"] * 3,
    })
    target = tmp_path / "2026-05-01.parquet"
    legacy.to_parquet(target, index=False)

    canonical = pd.DataFrame({
        "snapshot_date": ["2026-05-01"] * 2,
        "symbol": ["000660"] * 2,
        "ts_hms": [90100, 90200],
        "open": [500, 505], "high": [510, 512], "low": [498, 503],
        "close": [505, 510], "volume": [10, 20], "value_krw": [5050, 10200],
        "has_trade": [True, True], "vendor": ["kis", "kis"],
    })

    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts fail-closed merge
        merge_partition_frame(canonical, target, ("symbol", "ts_hms"))

    # 원본 파일은 그대로 보존되어야 한다
    assert len(pd.read_parquet(target)) == 3


def test_merge_partition_frame_still_merges_canonical_partitions(tmp_path) -> None:
    import pandas as pd

    from src.data.intraday_store import merge_partition_frame

    def _frame(symbol: str, ts: list[int], close: list[int]) -> pd.DataFrame:
        n = len(ts)
        return pd.DataFrame({
            "snapshot_date": ["2026-05-01"] * n,
            "symbol": [symbol] * n,
            "ts_hms": ts,
            "open": close, "high": close, "low": close, "close": close,
            "volume": [1] * n, "value_krw": [1] * n,
            "has_trade": [True] * n, "vendor": ["kis"] * n,
        })

    target = tmp_path / "2026-05-01.parquet"
    _frame("005930", [90100, 90200], [1000, 1010]).to_parquet(target, index=False)

    merged = merge_partition_frame(_frame("005930", [90200, 90300], [9999, 1020]), target, ("symbol", "ts_hms"))

    assert len(merged) == 3
    assert merged.loc[merged["ts_hms"] == 90200, "close"].iloc[0] == 9999


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
    assert report == {"n_symbols": 2, "n_low_coverage": 1}
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
    assert report == {"n_symbols": 3, "n_low_coverage": 0}
    assert not any("session_coverage" in rec.message for rec in caplog.records)


def test_log_session_coverage_outliers_handles_empty_frame() -> None:
    import pandas as pd

    from src.data import intraday_store

    # Given: an empty frame (write_intraday_partition never reaches this point with one,
    # since it returns 0 early, but the utility must still be safe standalone)
    report = intraday_store.log_session_coverage_outliers(pd.DataFrame(), 1, "2026-09-11", "regular")

    # Then
    assert report == {"n_symbols": 0, "n_low_coverage": 0}


def test_log_session_coverage_outliers_respects_custom_min_peer_ratio_boundary() -> None:
    import pandas as pd

    from src.data import intraday_store

    # Given: peer max = 10; B sits exactly at 80% (8, must NOT be flagged -- strict less-than),
    # C sits just below (7, MUST be flagged), under an explicit min_peer_ratio=0.8
    merged = pd.DataFrame({"symbol": ["A"] * 10 + ["B"] * 8 + ["C"] * 7})

    # When
    report = intraday_store.log_session_coverage_outliers(merged, 1, "2026-09-11", "regular", min_peer_ratio=0.8)

    # Then
    assert report == {"n_symbols": 3, "n_low_coverage": 1}

    # And: a lone symbol (no peer) is never flagged, regardless of ratio
    solo = pd.DataFrame({"symbol": ["Z"] * 2})
    solo_report = intraday_store.log_session_coverage_outliers(solo, 1, "2026-09-11", "regular", min_peer_ratio=0.99)
    assert solo_report == {"n_symbols": 1, "n_low_coverage": 0}


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
