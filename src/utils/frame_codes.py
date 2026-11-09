"""Memory-lean views of high-cardinality label columns (no src imports)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shared_object_values(values: pd.Series) -> np.ndarray:
    """Element-wise equal to ``values.to_numpy(dtype=object)``, with one shared object per distinct value.

    Millions of rows over a few thousand symbols otherwise allocate one Python string per row.

    Returns:
        Object array aligned with values; missing entries are ``np.nan``.
    """
    codes, uniques = pd.factorize(values)
    out = np.full(len(codes), np.nan, dtype=object)
    present = codes >= 0
    out[present] = np.asarray(uniques, dtype=object)[codes[present]]
    return out


def group_codes(values: pd.Series) -> np.ndarray:
    """Float group codes in first-appearance order: the same partition as ``values`` with NaN for missing.

    groupby drops NaN keys, so these codes group exactly like the original labels without re-hashing strings.
    """
    codes: np.ndarray = np.asarray(pd.factorize(values)[0], dtype=np.float64)
    codes[codes < 0] = np.nan
    return codes
