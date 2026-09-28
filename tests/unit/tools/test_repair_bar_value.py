"""Invariant scenarios for the stored-partition KIS bar-value repair."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.intraday_schema import normalize_bar_frame
from src.data.intraday_store import intraday_partition_path
from src.tools.repair_bar_value import repair_bar_value_partition

DAY = "2026-09-22"
SESSION = "krx_aftermarket"


def _kis(symbol: str, rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {"stck_bsop_date": DAY.replace("-", ""), "stck_cntg_hour": h, "stck_oprc": c, "stck_hgpr": c,
             "stck_lwpr": c, "stck_prpr": c, "cntg_vol": v, "acml_tr_pbmn": cum}
            for h, c, v, cum in rows
        ]
    )
    return normalize_bar_frame(raw, "kis", DAY, symbol)


def _kiwoom(symbol: str) -> pd.DataFrame:
    raw = pd.DataFrame([{"cntr_tm": f"{DAY.replace('-', '')}160100", "cur_prc": "+500", "open_pric": "+500",
                         "high_pric": "+500", "low_pric": "+500", "trde_qty": "4"}])
    return normalize_bar_frame(raw, "kiwoom", DAY, symbol)


@pytest.fixture
def partition(tmp_path, monkeypatch):
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    carried = _kis("000150", [("160000", "1471000", "3", "0"), ("160100", "1466000", "2", "0")])
    # 저장 당시(수정 전) 결함 재현: 첫 봉에 정규장 누적대금이 실려 있다
    carried.loc[0, "value_krw"] = 128_866_376_000
    carried.loc[1, "value_krw"] = 2_932_000
    clean = _kis("000660", [("160000", "1000", "10", "10000")])
    ls_noise = _kiwoom("005930")
    ls_noise["vendor"] = "ls"
    ls_noise.loc[0, "value_krw"] = 2_100  # LS 벤더 잡음(범위 밖)도 손대지 않는다
    frame = pd.concat([carried, clean, ls_noise], ignore_index=True)
    target = intraday_partition_path(1, DAY, SESSION)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target


def _read(target) -> pd.DataFrame:
    return pd.read_parquet(target).sort_values(["symbol", "ts_hms"]).reset_index(drop=True)


def test_repair_rewrites_only_violating_kis_rows(partition) -> None:
    before = _read(partition)
    report = repair_bar_value_partition(DAY, SESSION, apply=True)
    after = _read(partition)
    assert report.symbols_repaired == ("000150",)
    assert report.rows_repaired == 1
    assert report.written is True
    row = after[(after["symbol"] == "000150") & (after["ts_hms"] == 160000)].iloc[0]
    assert int(row["value_krw"]) == 1_471_000 * 3
    untouched = after[~((after["symbol"] == "000150") & (after["ts_hms"] == 160000))].reset_index(drop=True)
    expected = before[~((before["symbol"] == "000150") & (before["ts_hms"] == 160000))].reset_index(drop=True)
    pd.testing.assert_frame_equal(untouched[expected.columns], expected, check_dtype=False)


def test_dry_run_never_writes(partition) -> None:
    before_bytes = partition.read_bytes()
    report = repair_bar_value_partition(DAY, SESSION, apply=False)
    assert report.rows_repaired == 1
    assert report.written is False
    assert partition.read_bytes() == before_bytes


def test_repair_is_idempotent(partition) -> None:
    repair_bar_value_partition(DAY, SESSION, apply=True)
    again = repair_bar_value_partition(DAY, SESSION, apply=True)
    assert again.rows_repaired == 0
    assert again.written is False


def test_non_kis_rows_are_ignored(partition) -> None:
    repair_bar_value_partition(DAY, SESSION, apply=True)
    ls_row = _read(partition).query("symbol == '005930'").iloc[0]
    assert int(ls_row["value_krw"]) == 2_100


def test_missing_partition_raises(tmp_path, monkeypatch) -> None:
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    with pytest.raises(FileNotFoundError):
        repair_bar_value_partition(DAY, SESSION)


def test_cli_walks_range_and_skips_missing_dates(partition, monkeypatch, caplog) -> None:
    import logging
    import sys

    from src.tools import repair_bar_value

    monkeypatch.setattr(sys, "argv", ["repair", "--session", SESSION, "--from", "2026-09-21", "--to", DAY, "--apply"])
    with caplog.at_level(logging.INFO, logger=repair_bar_value.__name__):
        repair_bar_value.main()
    text = caplog.text
    assert "date=2026-09-21" in text and "missing_partition" in text
    assert f"date={DAY} session={SESSION} symbols=1 rows=1 written=True" in text


def test_cli_rejects_inverted_range(monkeypatch) -> None:
    import sys

    from src.tools import repair_bar_value

    monkeypatch.setattr(sys, "argv", ["repair", "--session", SESSION, "--from", "2026-09-23", "--to", "2026-09-22"])
    with pytest.raises(ValueError, match="Invalid date range"):
        repair_bar_value.main()


def test_empty_partition_reports_nothing(tmp_path, monkeypatch) -> None:
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    empty = _kis("000660", [("160000", "1000", "10", "10000")]).iloc[0:0]
    target = intraday_partition_path(1, DAY, SESSION)
    target.parent.mkdir(parents=True, exist_ok=True)
    empty.to_parquet(target, index=False)
    report = repair_bar_value_partition(DAY, SESSION, apply=True)
    assert report.rows_repaired == 0 and report.written is False


def test_cli_rejects_malformed_date(monkeypatch) -> None:
    import sys

    from src.tools import repair_bar_value

    monkeypatch.setattr(sys, "argv", ["repair", "--session", SESSION, "--from", "2026-13-01", "--to", DAY])
    with pytest.raises(ValueError, match="Invalid --from/--to"):
        repair_bar_value.main()
