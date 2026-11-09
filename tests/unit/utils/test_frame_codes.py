"""Shared-object and group-code views of label columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.frame_codes import group_codes, shared_object_values


def test_shared_object_values_equal_plain_object_array_and_share_objects() -> None:
    values = pd.Series(["005930", "000660", None, "005930"], dtype="str")

    out = shared_object_values(values)

    plain = values.to_numpy(dtype=object)
    assert out.dtype == object
    assert [a == b or (pd.isna(a) and pd.isna(b)) for a, b in zip(out, plain, strict=True)] == [True] * 4
    assert out[0] is out[3]
    assert pd.isna(out[2])
    assert shared_object_values(pd.Series([], dtype="str")).shape == (0,)


def test_group_codes_partition_like_labels_with_nan_for_missing() -> None:
    values = pd.Series(["b", "a", None, "b"], dtype="str")
    data = pd.Series([1.0, 2.0, 3.0, 4.0])

    codes = group_codes(values)

    np.testing.assert_array_equal(codes[[0, 1, 3]], [0.0, 1.0, 0.0])
    assert np.isnan(codes[2])
    by_codes = data.groupby(codes, sort=False).sum().to_numpy()
    by_labels = data.groupby(values.to_numpy(), sort=False).sum().to_numpy()
    np.testing.assert_array_equal(by_codes, by_labels)
