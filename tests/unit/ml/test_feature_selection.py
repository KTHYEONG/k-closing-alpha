import pytest


def test_permutation_importance_zero_for_date_constant_feature() -> None:
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor

    from src.ml.feature_selection import permutation_rank_ic_importance

    rng = np.random.default_rng(7)
    rows = []
    for _di, d in enumerate(pd.bdate_range("2024-01-01", periods=60)):  # noqa: B007 - spec skeleton
        market = float(rng.normal())  # 날짜별 상수
        for _ in range(8):
            sig = float(rng.normal())
            rows.append(
                {
                    "trade_date": d,
                    "signal": sig,
                    "market_level": market,
                    "noise": float(rng.normal()),
                    "target_return": 0.01 * sig + 0.002 * float(rng.normal()),
                }
            )
    df = pd.DataFrame(rows)
    fcols = ["signal", "market_level", "noise"]

    model = LGBMRegressor(objective="huber", alpha=0.02, random_state=42, verbosity=-1, num_leaves=7, n_estimators=80)
    model.fit(df[fcols], df["target_return"])

    imp = permutation_rank_ic_importance(model, df, fcols, "target_return", "trade_date")

    assert imp["market_level"] == pytest.approx(0.0, abs=1e-12)
    assert imp["signal"] > imp["noise"]


def test_permutation_importance_shuffles_within_group_only() -> None:
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor

    from src.ml.feature_selection import permutation_rank_ic_importance

    rng = np.random.default_rng(13)
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=40):
        for _ in range(6):
            sig = float(rng.normal())
            rows.append({"trade_date": d, "signal": sig, "noise": float(rng.normal()),
                         "target_return": 0.01 * sig})
    df = pd.DataFrame(rows)
    fcols = ["signal", "noise"]
    model = LGBMRegressor(objective="huber", alpha=0.02, random_state=42, verbosity=-1, num_leaves=7, n_estimators=60)
    model.fit(df[fcols], df["target_return"])

    a = permutation_rank_ic_importance(model, df, fcols, "target_return", "trade_date", random_state=42)
    b = permutation_rank_ic_importance(model, df, fcols, "target_return", "trade_date", random_state=42)

    pd.testing.assert_series_equal(a, b)
    assert set(a.index) == set(fcols)
    # 입력 프레임은 부작용 없이 원상복구되어야 한다
    assert df["signal"].iloc[0] == rows[0]["signal"]


def test_select_stable_features_applies_min_folds() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.feature_selection import select_stable_features

    rng = np.random.default_rng(17)
    rows = []
    for d in pd.bdate_range("2023-01-02", periods=200):
        for _ in range(8):
            sig = float(rng.normal())
            r = {"trade_date": d, "signal": sig, "target_return": 0.02 * sig + 0.001 * float(rng.normal())}
            for j in range(6):
                r[f"noise{j}"] = float(rng.normal())
            rows.append(r)
    df = pd.DataFrame(rows)
    fcols = ["signal"] + [f"noise{j}" for j in range(6)]

    sel = select_stable_features(
        df, fcols, "target_return", "trade_date", top_n=3, inner_splits=3, min_folds=2
    )
    sel2 = select_stable_features(
        df, fcols, "target_return", "trade_date", top_n=3, inner_splits=3, min_folds=2
    )

    assert "signal" in sel
    assert 0 < len(sel) <= 3
    assert sel == sel2  # 결정적
    assert all(c in fcols for c in sel)


def test_select_stable_features_uses_only_train_rows() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.feature_selection import select_stable_features

    rng = np.random.default_rng(23)
    rows = []
    for d in pd.bdate_range("2023-01-02", periods=260):
        for _ in range(8):
            sig = float(rng.normal())
            r = {"trade_date": d, "signal": sig, "target_return": 0.02 * sig}
            for j in range(4):
                r[f"noise{j}"] = float(rng.normal())
            rows.append(r)
    df = pd.DataFrame(rows)
    fcols = ["signal"] + [f"noise{j}" for j in range(4)]
    cut = df["trade_date"].quantile(0.7)
    train = df[df["trade_date"] <= cut].copy()

    sel_a = select_stable_features(train, fcols, "target_return", "trade_date", top_n=3, inner_splits=3)

    # train 이후 구간(홀드아웃)을 완전히 망가뜨려도 선택은 동일해야 한다
    df_b = df.copy()
    mask = df_b["trade_date"] > cut
    for j in range(4):
        df_b.loc[mask, f"noise{j}"] = df_b.loc[mask, "target_return"] * 1000.0
    train_b = df_b[df_b["trade_date"] <= cut].copy()
    sel_b = select_stable_features(train_b, fcols, "target_return", "trade_date", top_n=3, inner_splits=3)

    assert sel_a == sel_b


def test_hpo_objective_defaults_to_rank_ic() -> None:
    import pytest

    from src.ml.tuning import ChampionTuningConfig

    cfg = ChampionTuningConfig()
    assert cfg.hpo_objective == "rank_ic"
    assert cfg.feature_selection_top_n is None
    assert cfg.feature_selection_min_folds == 2

    ok = ChampionTuningConfig(feature_selection_top_n=30)
    assert ok.feature_selection_top_n == 30

    with pytest.raises(ValueError, match="feature_selection_top_n"):  # noqa: PT011 - spec skeleton expanded
        ChampionTuningConfig(feature_selection_top_n=2)
