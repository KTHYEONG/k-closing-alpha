"""Vectorized mean-group rank-IC contract."""
from __future__ import annotations


def test_mean_group_rank_ic_matches_scipy_per_group_mean() -> None:
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    from src.ml.metrics import mean_group_rank_ic

    rng = np.random.default_rng(7)
    frames = []
    for d in range(40):
        n = int(rng.integers(4, 15))
        sig = rng.normal(size=n)
        cont = pd.DataFrame({"g": d, "score": sig + rng.normal(size=n), "target": sig + rng.normal(size=n)})
        # tie-heavy block: integer target, duplicated scores
        tie = pd.DataFrame({"g": 100 + d, "score": rng.integers(0, 3, size=n).astype(float), "target": rng.integers(-1, 2, size=n).astype(float)})
        frames.extend([cont, tie])
    df = pd.concat(frames, ignore_index=True)

    ref = []
    for _, grp in df.groupby("g", sort=False):
        if len(grp) < 2:
            continue
        if float(np.std(grp["score"].to_numpy())) == 0.0 or float(np.std(grp["target"].to_numpy())) == 0.0:
            continue
        rho = spearmanr(grp["score"], grp["target"]).statistic
        if np.isfinite(rho):
            ref.append(float(rho))
    expected = float(np.mean(ref))

    got = mean_group_rank_ic(df, ["g"], "score", "target", min_group_size=2)
    assert abs(got - expected) < 1e-9


def test_mean_group_rank_ic_skips_degenerate_groups_and_returns_nan() -> None:
    import pytest
    import numpy as np
    import pandas as pd

    from src.ml.metrics import mean_group_rank_ic

    # one clean group, one too-small, one constant-target, one constant-score
    df = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "c", "c", "c", "d", "d", "d"],
            "score": [1.0, 2.0, 3.0, 5.0, 4.0, 1.0, 2.0, 7.0, 7.0, 7.0],
            "target": [3.0, 2.0, 1.0, 9.0, 6.0, 6.0, 6.0, 1.0, 2.0, 3.0],
        }
    )
    # only group 'a' qualifies: perfectly anti-monotonic -> rho == -1
    got = mean_group_rank_ic(df, ["g"], "score", "target", min_group_size=3)
    assert got == pytest.approx(-1.0)

    degenerate = pd.DataFrame({"g": ["x", "x"], "score": [1.0, 1.0], "target": [2.0, 3.0]})
    assert np.isnan(mean_group_rank_ic(degenerate, ["g"], "score", "target", min_group_size=2))


def test_rank_ic_delegates_without_behaviour_change() -> None:
    import pytest
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    from src.ml.metrics import rank_ic

    rng = np.random.default_rng(3)
    frames = []
    for d in range(25):
        n = int(rng.integers(3, 12))
        sig = rng.normal(size=n)
        frames.append(pd.DataFrame({"trade_date": f"2026-01-{d + 1:02d}", "pred": sig + rng.normal(size=n), "target_return": sig + rng.normal(size=n)}))
    oof = pd.concat(frames, ignore_index=True)

    ref = [float(spearmanr(g["pred"], g["target_return"]).statistic) for _, g in oof.groupby("trade_date", sort=False) if len(g) >= 2]
    expected = float(np.mean([r for r in ref if np.isfinite(r)]))

    assert rank_ic(oof, "trade_date", "target_return", "pred") == pytest.approx(expected, abs=1e-9)


def test_mean_group_rank_ic_empty_frame_returns_nan() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.metrics import mean_group_rank_ic

    empty = pd.DataFrame({"g": [], "score": [], "target": []})
    assert np.isnan(mean_group_rank_ic(empty, ["g"], "score", "target", min_group_size=2))
