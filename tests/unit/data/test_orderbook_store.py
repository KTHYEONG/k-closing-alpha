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


def test_build_orderbook_rows_preserves_iscd_fields_as_string() -> None:
    """iscd 종목코드 필드는 숫자만이어도(선행 0 보존), 문자가 섞여도 문자열로 남는다."""
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    output2 = {"stck_shrn_iscd": "005930", "mksc_shrn_iscd": "0220W0"}
    ts = datetime(2026, 9, 4, 15, 22, 0)

    rows = build_orderbook_rows({"rt_cd": "0", "output2": output2}, "005930", "J", "decision", ts)

    row = rows[0]
    assert row["stck_shrn_iscd"] == "005930"
    assert row["mksc_shrn_iscd"] == "0220W0"


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


def test_append_orderbook_refuses_to_overwrite_unreadable_partition(tmp_path, monkeypatch) -> None:
    """기존 파티션이 손상되어 읽을 수 없으면 쓰지 않고 typed error로 실패한다."""
    from datetime import datetime

    import pytest

    from src.data import orderbook_store
    from src.data.io_utils import ExistingStoreUnreadableError

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path)

    target = orderbook_store.orderbook_partition_path("2026-09-05")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a valid parquet file")

    rows = [{"capture_ts": datetime(2026, 9, 5, 15, 20), "symbol": "005930", "venue": "J", "capture_reason": "auction", "askp1": 70000}]

    with pytest.raises(ExistingStoreUnreadableError, match="2026-09-05"):
        orderbook_store.append_orderbook_snapshots(rows, "2026-09-05")

    assert target.read_bytes() == b"not a valid parquet file"


def test_append_orderbook_empty_rows_noop_even_if_partition_corrupt(tmp_path, monkeypatch) -> None:
    """빈 입력은 읽기 전에 0을 반환하므로 손상된 파티션도 건드리지 않는다."""
    from src.data import orderbook_store

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path)

    target = orderbook_store.orderbook_partition_path("2026-09-05")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a valid parquet file")

    assert orderbook_store.append_orderbook_snapshots([], "2026-09-05") == 0
    assert target.read_bytes() == b"not a valid parquet file"


def test_build_orderbook_rows_merges_output1_and_output2() -> None:
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    output1 = {"askp1": "70000", "bidp1": "69900", "total_askp_rsqn": "1200"}
    output2 = {
        "antc_mkop_cls_code": "112",
        "stck_prpr": "70000",
        "antc_cnpr": "70050",
        "antc_cntg_vrss": "50",
        "antc_vol": "12345",
    }
    ts = datetime(2026, 9, 10, 15, 22, 0)

    rows = build_orderbook_rows(
        {"rt_cd": "0", "output1": output1, "output2": output2}, "005930", "J", "decision", ts
    )

    assert len(rows) == 1
    row = rows[0]
    for key in output1:
        assert key in row
    for key in output2:
        assert key in row
    assert row["askp1"] == 70000
    assert row["antc_cnpr"] == 70050
    assert row["antc_vol"] == 12345
    assert row["symbol"] == "005930"


def test_build_orderbook_rows_output2_wins_on_key_collision() -> None:
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    output1 = {"stck_prpr": "11111"}
    output2 = {"stck_prpr": "22222"}
    ts = datetime(2026, 9, 10, 15, 22, 0)

    rows = build_orderbook_rows(
        {"rt_cd": "0", "output1": output1, "output2": output2}, "005930", "J", "decision", ts
    )

    assert rows[0]["stck_prpr"] == 22222


def test_build_orderbook_rows_works_with_only_output2() -> None:
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    ts = datetime(2026, 9, 10, 15, 22, 0)
    rows = build_orderbook_rows(
        {"rt_cd": "0", "output2": {"antc_cnpr": "70050"}}, "005930", "J", "decision", ts
    )
    assert len(rows) == 1
    assert rows[0]["antc_cnpr"] == 70050

    empty = build_orderbook_rows({"rt_cd": "0"}, "005930", "J", "decision", ts)
    assert empty == []



def test_orderbook_partition_path_supports_named_sessions(tmp_path, monkeypatch) -> None:
    """Session-specific partition path."""
    from src.data import orderbook_store

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path)

    assert orderbook_store.orderbook_partition_path("2026-09-29") == tmp_path / "orderbook" / "2026-09" / "2026-09-29.parquet"
    assert orderbook_store.orderbook_partition_path("2026-09-29", session="aftermarket") == (
        tmp_path / "orderbook" / "aftermarket" / "2026-09" / "2026-09-29.parquet"
    )


def test_orderbook_partition_path_rejects_unsafe_sessions(tmp_path, monkeypatch) -> None:
    """Unsafe session names never escape the orderbook tree."""
    import pytest

    from src.data import orderbook_store

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path)

    for bad in ("", "../escape", "a/b", ".."):
        with pytest.raises(ValueError, match="orderbook session"):
            orderbook_store.orderbook_partition_path("2026-09-29", session=bad)


def test_append_orderbook_snapshots_supports_named_sessions(tmp_path, monkeypatch) -> None:
    """Named sessions merge into their own partition without touching the regular one."""
    from datetime import datetime

    import pandas as pd

    from src.data import orderbook_store

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path)

    rows = [{"capture_ts": datetime(2026, 9, 29, 17, 0), "symbol": "005930", "venue": "NXT", "capture_reason": "aftermarket-sparse"}]
    assert orderbook_store.append_orderbook_snapshots(rows, "2026-09-29", session="aftermarket") == 1
    assert not orderbook_store.orderbook_partition_path("2026-09-29").exists()
    stored = pd.read_parquet(orderbook_store.orderbook_partition_path("2026-09-29", session="aftermarket"))
    assert len(stored) == 1


def test_build_orderbook_rows_carries_optional_timing_columns() -> None:
    """Optional timing columns."""
    from datetime import datetime

    from src.data.orderbook_store import build_orderbook_rows

    res = {"rt_cd": "0", "output1": {"aspr_acpt_hour": "170001"}}
    capture_ts = datetime(2026, 9, 29, 17, 0, 1)
    scheduled_at = datetime(2026, 9, 29, 17, 0, 0)
    started = datetime(2026, 9, 29, 17, 0, 0, 500000)

    timed = build_orderbook_rows(
        res, "005930", "NXT", "aftermarket-dense", capture_ts, scheduled_at=scheduled_at, request_started_at=started
    )
    assert timed[0]["scheduled_at"] == scheduled_at
    assert timed[0]["request_started_at"] == started

    plain = build_orderbook_rows(res, "005930", "NXT", "aftermarket-dense", capture_ts)
    assert "scheduled_at" not in plain[0]
    assert "request_started_at" not in plain[0]
