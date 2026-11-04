"""Date-keyed research partitions kept beside, not inside, the decision archive."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet

RESEARCH_DIR_NAME: str = "research"
_DATASET_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def research_partition_path(dataset: str, snapshot_date: str) -> Path:
    """Return the Parquet path of one research dataset on one trading date.

    Raises:
        ValueError: The dataset name is not a lowercase identifier, or snapshot_date is not ISO YYYY-MM-DD.
    """
    if not _DATASET_RE.fullmatch(dataset):
        raise ValueError(f"invalid research dataset name: {dataset!r}")
    try:
        canonical = date.fromisoformat(snapshot_date).isoformat()
    except ValueError:
        canonical = None
    if canonical != snapshot_date:
        raise ValueError(f"snapshot_date must be ISO YYYY-MM-DD: {snapshot_date!r}")
    return Path(settings.HISTORY_DIR) / RESEARCH_DIR_NAME / dataset / snapshot_date[:7] / f"{snapshot_date}.parquet"


def write_research_partition(frame: pd.DataFrame, dataset: str, snapshot_date: str, *, overwrite: bool) -> int:
    """Atomically publish one research partition and return the row count written.

    Raises:
        FileExistsError: The partition exists and overwrite is False.
        ValueError: The dataset name or snapshot_date is invalid.
        OSError: Publication failed; the destination is untouched.
    """
    path = research_partition_path(dataset, snapshot_date)
    if path.exists() and not overwrite:
        raise FileExistsError(f"research partition already exists: {path}")
    if len(frame) == 0:
        return 0
    atomic_write_parquet(frame, path)
    return len(frame)


def read_research_partition(dataset: str, snapshot_date: str) -> pd.DataFrame:
    """Read one research partition.

    Raises:
        FileNotFoundError: The partition does not exist.
    """
    return pd.read_parquet(research_partition_path(dataset, snapshot_date))
