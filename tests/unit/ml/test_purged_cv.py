"""Purged Group Walk-Forward CV Splitter 단위 테스트.

SCENARIO_PURGED_CV_SPLIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.purged_cv import PurgedGroupTimeSeriesSplit


def _make_fixture(
    n_groups: int = 16, rows_per_group: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = [f"2024-01-{i + 1:02d}" for i in range(n_groups)]
    group_values: list[str] = []
    for date in dates:
        group_values.extend([date] * rows_per_group)
    size = len(group_values)
    X = pd.DataFrame(
        {"f1": rng.normal(size=size), "f2": rng.normal(size=size)}
    )
    y = pd.Series(rng.normal(size=size))
    return X, y, pd.Series(group_values)


def _fold_group_positions(
    splitter: PurgedGroupTimeSeriesSplit,
    groups: pd.Series,
) -> list[tuple[set[int], set[int]]]:
    """각 fold 의 train/val 에 속한 group 고유 위치 집합을 반환합니다."""
    unique = np.unique(groups.to_numpy())
    position_of = {group: i for i, group in enumerate(unique)}
    positions: list[tuple[set[int], set[int]]] = []
    X = pd.DataFrame(index=groups.index)
    for train_idx, val_idx in splitter.split(X, groups=groups):
        train_pos = {position_of[g] for g in groups.iloc[train_idx]}
        val_pos = {position_of[g] for g in groups.iloc[val_idx]}
        positions.append((train_pos, val_pos))
    return positions


def test_split_yields_expected_number_of_folds() -> None:
    X, y, groups = _make_fixture(n_groups=16, rows_per_group=4)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=1)
    splits = list(splitter.split(X, y, groups))
    assert len(splits) == 5


def test_split_never_overlaps_train_and_val_indices() -> None:
    X, y, groups = _make_fixture()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=1)
    for train_idx, val_idx in splitter.split(X, y, groups):
        assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_split_keeps_same_group_within_single_fold() -> None:
    X, y, groups = _make_fixture()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=1)
    for train_idx, val_idx in splitter.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        assert not (train_groups & val_groups)


def test_split_validation_indices_are_disjoint_across_folds() -> None:
    """Walk-Forward 특성상 최초 구간은 warm-up(Train 전용)이므로, 검증은
    샘플 간 중복 없이 뒤쪽 구간만 순차 검증합니다."""
    X, y, groups = _make_fixture(n_groups=16, rows_per_group=4)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=1)
    seen: list[int] = []
    for _, val_idx in splitter.split(X, y, groups):
        seen.extend(val_idx.tolist())
    assert len(seen) == len(set(seen))
    assert len(seen) > 0
    assert set(seen) <= set(range(len(groups)))


def test_split_maintains_purge_gap_barrier() -> None:
    """Train end group 와 Val start group 사이에 최소 purge_gap 만큼 공백 유지."""
    X, y, groups = _make_fixture(n_groups=18, rows_per_group=3)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=2)
    for train_pos, val_pos in _fold_group_positions(splitter, groups):
        max_train = max(train_pos)
        min_val = min(val_pos)
        assert max_train < min_val
        assert min_val - max_train - 1 == splitter.purge_gap


def test_split_purge_gap_zero_is_contiguous() -> None:
    X, y, groups = _make_fixture(n_groups=18, rows_per_group=3)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=0)
    for train_pos, val_pos in _fold_group_positions(splitter, groups):
        assert max(train_pos) < min(val_pos)
        assert min(val_pos) - max(train_pos) == 1


def test_split_train_is_strictly_walk_forward() -> None:
    """모든 fold 에서 Train 은 Validation 보다 과거 group 으로만 구성."""
    X, y, groups = _make_fixture()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=1)
    for train_pos, val_pos in _fold_group_positions(splitter, groups):
        assert max(train_pos) < min(val_pos)


def test_split_requires_groups_argument() -> None:
    X, y, _ = _make_fixture()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=3, purge_gap=1)
    with pytest.raises(ValueError, match="groups must be provided"):
        next(splitter.split(X, y))


def test_split_rejects_insufficient_groups() -> None:
    X, y, groups = _make_fixture(n_groups=4, rows_per_group=2)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=1)
    with pytest.raises(ValueError, match="requires at least 6 unique groups"):
        list(splitter.split(X, y, groups))


def test_split_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedGroupTimeSeriesSplit(n_splits=0)
    with pytest.raises(ValueError, match="purge_gap"):
        PurgedGroupTimeSeriesSplit(n_splits=3, purge_gap=-1)
