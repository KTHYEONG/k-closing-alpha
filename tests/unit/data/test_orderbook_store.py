from __future__ import annotations



def test_build_orderbook_rows_preserves_full_payload_verbatim() -> None:
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    output1 = {f"askp{i}": str(70000 + i * 100) for i in range(1, 11)}
    output1.update({f"bidp{i}": str(69900 - i * 100) for i in range(1, 11)})
    output1.update({f"askp_rsqn{i}": str(100 * i) for i in range(1, 11)})
    output1.update({f"bidp_rsqn{i}": str(200 * i) for i in range(1, 11)})
    output1.update({"total_askp_rsqn": "1200", "total_bidp_rsqn": "1500", "antc_cnpr": "70050", "antc_cnqn": "3300"})

    ts = datetime(2026, 9, 4, 15, 22, 0)
    rows = build_orderbook_rows({"rt_cd": "0", "output1": output1}, "005930", "J", "auction", ts)

    assert len(rows) == 1
    row = rows[0]
    for key in output1:
        assert key in row
    assert row["askp10"] == 71000
    assert row["antc_cnqn"] == 3300
    assert row["symbol"] == "005930"
    assert row["venue"] == "J"
    assert row["capture_reason"] == "auction"


def test_build_orderbook_rows_passes_through_non_string_and_non_numeric_string_values() -> None:
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    output1 = {
        "askp1": 70000,  # 이미 숫자형 -> 그대로 통과
        "is_paused": True,  # bool -> 그대로 통과
        "missing_field": None,  # None -> 그대로 통과
        "status_text": "정상",  # 숫자가 아닌 문자열 -> 그대로 통과
    }
    ts = datetime(2026, 9, 4, 15, 22, 0)

    rows = build_orderbook_rows({"rt_cd": "0", "output1": output1}, "005930", "J", "auction", ts)

    row = rows[0]
    assert row["askp1"] == 70000
    assert row["is_paused"] is True
    assert row["missing_field"] is None
    assert row["status_text"] == "정상"


def test_build_orderbook_rows_passes_through_non_scalar_value() -> None:
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    output1 = {"nested": {"a": 1}}
    ts = datetime(2026, 9, 4, 15, 22, 0)

    rows = build_orderbook_rows({"rt_cd": "0", "output1": output1}, "005930", "J", "auction", ts)

    assert rows[0]["nested"] == {"a": 1}


def test_build_orderbook_rows_failed_response_is_empty() -> None:
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    assert build_orderbook_rows({"rt_cd": "1"}, "005930", "J", "decision", datetime(2026, 9, 4)) == []
    assert build_orderbook_rows({"rt_cd": "0", "output1": None}, "005930", "J", "decision", datetime(2026, 9, 4)) == []


def test_append_orderbook_snapshots_merges_column_union_across_sweeps(tmp_path, monkeypatch) -> None:
    from datetime import datetime

    import pandas as pd

    from src.data import orderbook_store

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path)

    first = [{"capture_ts": datetime(2026, 9, 4, 15, 20), "symbol": "005930", "venue": "J", "capture_reason": "auction", "askp1": 70000}]
    second = [{"capture_ts": datetime(2026, 9, 4, 15, 20, 10), "symbol": "005930", "venue": "J", "capture_reason": "auction", "askp1": 70100, "antc_cnpr": 70050}]

    assert orderbook_store.append_orderbook_snapshots(first, "2026-09-04") == 1
    assert orderbook_store.append_orderbook_snapshots(second, "2026-09-04") == 2
    assert orderbook_store.append_orderbook_snapshots([], "2026-09-04") == 0

    stored = pd.read_parquet(orderbook_store.orderbook_partition_path("2026-09-04"))
    assert len(stored) == 2
    assert "antc_cnpr" in stored.columns
    assert stored["antc_cnpr"].isna().sum() == 1


def test_append_orderbook_snapshots_recovers_from_unreadable_existing_partition(tmp_path, monkeypatch) -> None:
    """기존 파티션 파일이 손상되어 읽기 실패해도 신규 행만으로 안전하게 계속 진행한다."""
    from datetime import datetime

    from src.data import orderbook_store

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path)

    target = orderbook_store.orderbook_partition_path("2026-09-05")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not a valid parquet file")

    rows = [{"capture_ts": datetime(2026, 9, 5, 15, 20), "symbol": "005930", "venue": "J", "capture_reason": "auction", "askp1": 70000}]

    assert orderbook_store.append_orderbook_snapshots(rows, "2026-09-05") == 1

