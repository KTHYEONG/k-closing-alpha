"""Invariant guards for the date-keyed research partition store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import settings
from src.data import research_store
from src.data.research_store import (
    research_partition_path,
    read_research_partition,
    write_research_partition,
)

DATE = "2026-11-04"


@pytest.fixture(autouse=True)
def _history(monkeypatch: Any, tmp_path: Path) -> Path:
    history = tmp_path / "history"
    monkeypatch.setattr(settings, "HISTORY_DIR", history)
    return history


def test_partition_path_layout() -> None:
    assert research_partition_path("postclose_strength", DATE) == (
        Path(settings.HISTORY_DIR) / "research" / "postclose_strength" / "2026-11" / f"{DATE}.parquet"
    )


@pytest.mark.parametrize("dataset", ["../x", "Upper", "1abc", "", "a-b"])
def test_dataset_name_validated(dataset: str) -> None:
    with pytest.raises(ValueError, match="invalid research dataset name"):
        research_partition_path(dataset, DATE)


@pytest.mark.parametrize("snapshot_date", ["2026-11-4", "20261104", "../2026-11-04"])
def test_snapshot_date_validated(snapshot_date: str) -> None:
    with pytest.raises(ValueError, match="snapshot_date must be ISO"):
        research_partition_path("postclose_strength", snapshot_date)


def test_no_overwrite_raises_and_keeps_existing_file() -> None:
    write_research_partition(pd.DataFrame({"symbol": ["A"]}), "postclose_strength", DATE, overwrite=False)
    path = research_partition_path("postclose_strength", DATE)
    before = path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        write_research_partition(pd.DataFrame({"symbol": ["B"]}), "postclose_strength", DATE, overwrite=False)

    assert path.read_bytes() == before


def test_overwrite_replaces_atomically_without_temp_leftovers() -> None:
    write_research_partition(pd.DataFrame({"symbol": ["A"]}), "postclose_strength", DATE, overwrite=False)
    written = write_research_partition(pd.DataFrame({"symbol": ["B", "C"]}), "postclose_strength", DATE, overwrite=True)

    assert written == 2
    assert read_research_partition("postclose_strength", DATE)["symbol"].tolist() == ["B", "C"]
    assert [p.name for p in research_partition_path("postclose_strength", DATE).parent.iterdir()] == [f"{DATE}.parquet"]


def test_empty_frame_writes_nothing() -> None:
    assert write_research_partition(pd.DataFrame(), "postclose_strength", DATE, overwrite=False) == 0
    assert not research_partition_path("postclose_strength", DATE).exists()


def test_read_missing_partition_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_research_partition("postclose_strength", DATE)


def test_module_exposes_research_directory_name() -> None:
    assert research_store.RESEARCH_DIR_NAME == "research"


def test_research_writer_refuses_evicted_partition() -> None:
    from src.data.partition_eviction import EvictedPartitionError, tombstone_path

    target = research_partition_path("postclose_strength", DATE)
    target.parent.mkdir(parents=True)
    tombstone_path(target).write_text("{}")

    with pytest.raises(EvictedPartitionError):
        write_research_partition(pd.DataFrame({"symbol": ["A"]}), "postclose_strength", DATE, overwrite=True)
    assert not target.exists()
