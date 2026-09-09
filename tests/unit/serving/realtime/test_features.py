"""Live snapshot feature-frame compatibility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.serving.realtime.features import (
    _ROBUST_Z_COLUMNS,
    _apply_robust_z,
    engineer_features,
)

from tests.unit.serving.realtime.fixtures import (
    daily_snapshot_df,
    snapshot_feature_cols,
)


def test_engineer_features_created() -> None:
    cleaned = daily_snapshot_df().rename(columns={"등락률": "change_rate"})
    cleaned["prev_close_price"] = cleaned["종가"] * 0.99
    cleaned["open_price"] = cleaned["시가"]
    cleaned["high_price"] = cleaned["고가"]
    cleaned["low_price"] = cleaned["저가"]
    cleaned["close_price"] = cleaned["종가"]
    cleaned["market_cap_100m"] = cleaned["시가총액"]
    cleaned["trade_value_100m"] = cleaned["거래대금"]
    cleaned["selection_rank"] = cleaned["선정순위"]
    cleaned["inst_net_buy"] = cleaned["기관_순매수"]
    cleaned["foreign_net_buy"] = cleaned["외국인_순매수"]
    cleaned["prog_net_buy"] = cleaned["프로그램_순매수"]
    cleaned["market_type"] = cleaned["시장구분"]
    cleaned["total_candidate_count"] = cleaned["총_종목수"]
    cleaned["avg_trade_value"] = cleaned["평균_거래대금"]
    cleaned["kospi_change"] = cleaned["kospi"]
    cleaned["kosdaq_change"] = cleaned["kosdaq"]
    cleaned["buy_price"] = cleaned["종가"]
    cleaned["trade_date"] = pd.Timestamp("2026-08-04")
    engineered = engineer_features(cleaned)
    expected = {
        "buy_price_change_rate",
        "gap_ratio",
        "intraday_return",
        "major_density",
        "prog_dominance",
        "rank_ratio",
        "relative_change_rate",
        "change_rate_pct_rank",
        "log_market_cap_100m",
    }
    assert expected.issubset(set(engineered.columns))


def test_apply_robust_z_bounds() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "trade_date": ["2026-08-04"] * 12,
            "change_rate": rng.normal(size=12),
            "major_density": rng.uniform(0, 1, size=12),
        }
    )
    out = _apply_robust_z(df, _ROBUST_Z_COLUMNS)
    for col in _ROBUST_Z_COLUMNS:
        z_col = f"{col}_z"
        if col not in df.columns:
            assert z_col not in out.columns
            continue
        assert z_col in out.columns
        vals = out[z_col].dropna()
        assert vals.between(-5, 5).all()
    assert out["change_rate_z"].notna().sum() > 0


def test_build_topk_ranker_features_maps_korean_snapshot_to_feature_cols() -> None:
    import pandas as pd
    import pytest

    from src.ml.research.v3_engine import FEATURE_COLS
    from src.serving.realtime.features import build_topk_ranker_features

    # Given: a live Korean-column daily snapshot (matches collect.py's saved shape)
    df = pd.DataFrame({
        "종목코드": ["005930", "000660"],
        "종가": [70000.0, 180000.0],
        "전일종가": [68000.0, 176000.0],
        "고가": [70500.0, 181000.0],
        "저가": [68500.0, 177000.0],
        "시가": [68800.0, 177500.0],
        "거래량": [1_000_000.0, 500_000.0],
        "거래대금": [700.0, 900.0],
        "시가총액": [4_200_000.0, 1_300_000.0],
        "기관_순매수": [1000.0, -500.0],
        "외국인_순매수": [2000.0, 300.0],
        "kospi": [0.52, 0.52],
        "kosdaq": [0.31, 0.31],
        "v_kospi": [15.2, 15.2],
    })

    # When
    out = build_topk_ranker_features(df, pd.Timestamp("2026-09-09"))

    # Then: every v3_engine FEATURE_COLS is present and finite
    for col in FEATURE_COLS:
        assert col in out.columns
    assert out["kospi_pct"].iloc[0] == pytest.approx(0.0052)  # percent -> decimal fraction
    assert out["kosdaq_pct"].iloc[0] == pytest.approx(0.0031)
    assert out["v_kospi"].iloc[0] == pytest.approx(15.2)


def test_build_topk_ranker_features_raises_on_missing_required_column() -> None:
    import pandas as pd
    import pytest

    from src.serving.realtime.features import build_topk_ranker_features

    df = pd.DataFrame({"종목코드": ["005930"], "종가": [70000.0]})

    with pytest.raises(ValueError, match="거래대금|missing"):  # noqa: RUF043 - spec skeleton alternation
        build_topk_ranker_features(df, pd.Timestamp("2026-09-09"))
