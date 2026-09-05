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

