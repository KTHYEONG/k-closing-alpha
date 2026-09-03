import numpy as np
import pandas as pd

from src.serving.realtime.features import _apply_robust_z


def test_apply_robust_z_vectorized_matches_lambda_reference() -> None:
    # Given: a 3-date panel; date D3 has a constant column value (MAD == 0)
    rows = []
    for d, vals in {
        "2024-01-02": [1.0, 2.0, 3.0, 10.0],
        "2024-01-03": [-1.0, 0.0, 0.5, 4.0],
        "2024-01-04": [7.0, 7.0, 7.0, 7.0],
    }.items():
        for v in vals:
            rows.append({"trade_date": pd.Timestamp(d), "change_rate": v})  # noqa: PERF401
    df = pd.DataFrame(rows)

    # Reference: explicit per-group MAD via lambda (pre-change semantics)
    ref = df.copy()
    grouped = ref.groupby("trade_date")["change_rate"]
    median = grouped.transform("median")
    mad = grouped.transform(lambda x: (x - x.median()).abs().median()).replace(0, np.nan)
    ref_z = ((ref["change_rate"] - median) / mad).clip(-5, 5)

    # When
    out = _apply_robust_z(df.copy(), ("change_rate",))

    # Then
    got = out["change_rate_z"]
    assert got.isna().tolist()[-4:] == [True, True, True, True]  # zero-MAD group -> NaN
    np.testing.assert_allclose(
        got.to_numpy(dtype=float), ref_z.to_numpy(dtype=float), rtol=1e-12, atol=1e-15, equal_nan=True
    )
