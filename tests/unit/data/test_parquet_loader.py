from pathlib import Path
import pandas as pd
import pytest

from src import settings
from src.data.parquet_loader import (
    _atomic_write_parquet,
    upsert_condition_parquet,
)


@pytest.fixture
def tmp_parquet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p_dir = tmp_path / "parquet"
    p_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "PARQUET_DIR", p_dir)
    monkeypatch.setattr(settings, "HISTORY_PARQUET_PATH", p_dir / "condition_history.parquet")
    return p_dir


def test_upsert_and_load_condition_parquet(tmp_parquet_dir: Path) -> None:
    data_day1 = {
        "스냅샷_날짜": ["2026-08-01 15:30:00", "2026-08-01 15:30:00"],
        "종목코드": ["005930", "000660"],
        "순위": [1, 2],
    }
    df1 = pd.DataFrame(data_day1)
    upsert_condition_parquet(df1)

    df_loaded = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
    assert len(df_loaded) == 2

    # 중복 업서트 테스트 (동일 날짜/종목코드)
    upsert_condition_parquet(df1)
    df_loaded_dedup = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
    assert len(df_loaded_dedup) == 2


def test_parquet_loader_atomic_write_delegates_and_still_works(tmp_path: Path) -> None:
    df = pd.DataFrame({"종목코드": ["005930"], "종가": [70000]})
    target = tmp_path / "legacy.parquet"

    _atomic_write_parquet(df, target)

    assert target.exists()
    loaded = pd.read_parquet(target)
    assert loaded.loc[0, "종목코드"] == "005930"


def test_parquet_loader_module_no_longer_exposes_trade_log_functions() -> None:
    import src.data.parquet_loader as mod

    assert not hasattr(mod, "save_trade_log_to_parquet")
    assert not hasattr(mod, "load_trade_log_from_parquet")
    assert not hasattr(mod, "save_theme_to_parquet")
    assert not hasattr(mod, "load_theme_from_parquet")
    assert not hasattr(mod, "ThemeMapUnreadableError")
    assert hasattr(mod, "upsert_condition_parquet")
    assert not hasattr(mod, "load_condition_data_from_parquet")


def test_load_condition_data_from_parquet_removed_as_orphaned() -> None:
    import src.data.parquet_loader as mod

    assert not hasattr(mod, "load_condition_data_from_parquet")
    assert hasattr(mod, "upsert_condition_parquet")


def _condition_frame(date_str: str, code: str) -> pd.DataFrame:
    return pd.DataFrame({"스냅샷_날짜": [date_str], "종목코드": [code], "순위": [1]})


def _hold_archive_sidecar(monkeypatch: pytest.MonkeyPatch):
    import contextlib
    import fcntl

    from src.data import io_utils
    from src.utils.file_lock import sidecar_lock_path

    monkeypatch.setattr(io_utils, "STORE_LOCK_TIMEOUT_SECONDS", 0.0)

    @contextlib.contextmanager
    def _guard():
        lock_path = sidecar_lock_path(settings.HISTORY_PARQUET_PATH)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        holder = open(lock_path, "w")  # noqa: PTH123, SIM115 - lock held across the block
        try:
            fcntl.flock(holder, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

    return _guard()


def test_interleaved_upserts_both_survive(tmp_parquet_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    import src.data.parquet_loader as loader

    upsert_condition_parquet(_condition_frame("2026-08-01 15:30:00", "005930"))
    df_a = _condition_frame("2026-08-02 15:30:00", "005930")
    df_b = _condition_frame("2026-08-03 15:30:00", "000660")
    real_write = loader._atomic_write_parquet
    state: dict = {"calls": 0}

    def _wrapping(df: pd.DataFrame, target: Path):
        if state["calls"] == 0:
            state["calls"] += 1

            def _run_b() -> None:
                try:
                    upsert_condition_parquet(df_b)
                except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
                    state["error"] = exc

            worker = threading.Thread(target=_run_b)
            state["worker"] = worker
            worker.start()
            time.sleep(0.5)
            state["blocked"] = worker.is_alive()
        return real_write(df, target)

    monkeypatch.setattr(loader, "_atomic_write_parquet", _wrapping)

    upsert_condition_parquet(df_a)
    worker = state["worker"]
    worker.join(timeout=30)
    state["joined"] = not worker.is_alive()

    assert state.get("error") is None
    assert state["blocked"] is True
    assert state["joined"] is True
    final = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
    assert len(final) == 3
    assert set(final["스냅샷_날짜"].astype(str)) == {
        "2026-08-01 15:30:00",
        "2026-08-02 15:30:00",
        "2026-08-03 15:30:00",
    }


def test_upsert_lock_timeout_leaves_archive_untouched(
    tmp_parquet_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data import io_utils

    upsert_condition_parquet(_condition_frame("2026-08-01 15:30:00", "005930"))
    before = settings.HISTORY_PARQUET_PATH.read_bytes()

    with (
        _hold_archive_sidecar(monkeypatch),
        pytest.raises(io_utils.StoreLockTimeoutError),
    ):
        upsert_condition_parquet(_condition_frame("2026-08-02 15:30:00", "005930"))

    assert settings.HISTORY_PARQUET_PATH.read_bytes() == before


def test_empty_upsert_takes_no_lock(tmp_parquet_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _hold_archive_sidecar(monkeypatch):
        assert upsert_condition_parquet(pd.DataFrame()) is None


def test_upsert_condition_parquet_replaces_by_snapshot_identity(tmp_parquet_dir: Path) -> None:
    seed = pd.DataFrame({
        "스냅샷_날짜": ["2026-08-01 15:30:00", "2026-08-01 15:30:00"],
        "snapshot_timestamp": ["2026-08-01T15:20:00+09:00", "2026-08-01T15:30:00+09:00"],
        "종목코드": ["005930", "005930"],
        "signal": ["old-1520", "old-1530"],
    })
    upsert_condition_parquet(seed)

    upsert_condition_parquet(
        pd.DataFrame({
            "스냅샷_날짜": ["2026-08-01 15:30:00", "2026-08-01 15:30:00"],
            "snapshot_timestamp": ["2026-08-01T15:30:00+09:00", "2026-08-01T15:30:00+09:00"],
            "종목코드": ["005930", "000660"],
            "signal": ["new-1530", "new-0660"],
        })
    )

    final = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
    assert len(final) == 3
    by_key = {
        (str(ts), str(code)): str(signal)
        for ts, code, signal in zip(
            final["snapshot_timestamp"], final["종목코드"], final["signal"], strict=True
        )
    }
    assert by_key == {
        ("2026-08-01T15:20:00+09:00", "005930"): "old-1520",
        ("2026-08-01T15:30:00+09:00", "005930"): "new-1530",
        ("2026-08-01T15:30:00+09:00", "000660"): "new-0660",
    }

