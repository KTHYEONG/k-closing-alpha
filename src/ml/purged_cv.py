"""Purged walk-forward splitter ported (archival)."""
from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pandas as pd


class PurgedGroupTimeSeriesSplit:
    """Group(날짜) 단위 Purged Time-Series Walk-Forward Splitter."""

    def __init__(self, n_splits: int = 5, purge_gap: int = 1) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Walk-Forward 로 train/validation index 쌍을 순차 생성합니다."""
        if groups is None:
            raise ValueError("groups must be provided for PurgedGroupTimeSeriesSplit")
        group_values = pd.Series(groups).to_numpy()
        unique_groups = np.unique(group_values)
        n_groups = len(unique_groups)
        if n_groups < self.n_splits + 1:
            raise ValueError(
                f"n_splits={self.n_splits} requires at least {self.n_splits + 1} "
                f"unique groups, got {n_groups}"
            )
        group_positions = pd.Series(group_values).map(
            {group: i for i, group in enumerate(unique_groups)}
        ).to_numpy(dtype=np.int64)
        test_size = n_groups // (self.n_splits + 1)
        test_starts = range(n_groups - self.n_splits * test_size, n_groups, test_size)
        for test_start in test_starts:
            test_end = test_start + test_size
            train_end = test_start - self.purge_gap
            train_mask = group_positions < train_end
            test_mask = (group_positions >= test_start) & (group_positions < test_end)
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def chrono_fit_calibration_split(
    group_values: np.ndarray,
    train_idx: np.ndarray,
    calib_frac: float = 0.3,
    embargo: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 인덱스를 이른 fit 구간과 이후 보정 구간으로 순차 분할합니다."""
    if train_idx.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    train_groups = group_values[train_idx]
    unique = np.unique(train_groups)
    n = len(unique)
    if n < 3:  # _MIN_FIT_GROUPS(2) + 1
        return train_idx, np.array([], dtype=np.intp)
    calib_size = max(1, min(n - 1, int(np.ceil(n * calib_frac))))
    fit_size = n - calib_size
    calib_groups = set(unique[fit_size:].tolist())
    fit_groups = set(unique[: max(0, fit_size - embargo)].tolist())
    if not fit_groups:
        return np.array([], dtype=np.intp), train_idx
    fit_mask = np.isin(train_groups, list(fit_groups))
    calib_mask = np.isin(train_groups, list(calib_groups))
    return train_idx[fit_mask], train_idx[calib_mask]
