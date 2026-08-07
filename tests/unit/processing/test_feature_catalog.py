"""causal_expanded_v1 카탈로그 단위 테스트.

SCENARIO_FEATURE_CATALOG_01: 카탈로그는 결정적이고, 벡터화되며, 중복 없이
600--1000 후보로 제한되고, 타깃/미래 입력을 배제합니다.
SCENARIO_FEATURE_CATALOG_02: 동일 날짜 운영 시트 필드는 허용되고, 동일 날짜
외부 EOD/백필 이력은 제외되며, 이후 날짜 변경은 이전 피처 행을 바꾸지 않습니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_catalog import (
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    build_catalog,
    build_causal_feature_matrix,
    validate_price_history,
)


def _synthetic_history(n_dates: int = 90, n_symbols: int = 15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="D")
    rows: list[dict[str, object]] = []
    for s in range(n_symbols):
        symbol = f"{100000 + s:06d}"
        close = 50000.0 + s * 1000
        for date in dates:
            close = close * (1 + rng.normal(0.0, 0.02))
            open_ = close * (1 + rng.normal(0.0, 0.005))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            volume = abs(rng.normal(1_000_000, 200_000))
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "prev_close": close / (1 + rng.normal(0, 0.02)),
                    "volume": volume,
                    "trade_value_100m": volume * close / 100_000_000,
                    "market_cap_100m": close * 1_000_000 / 100_000_000,
                    "daily_change_pct": rng.normal(0.0, 2.0),
                    "market": "KOSPI",
                    "inst_net_buy": rng.normal(0, 1e8),
                    "foreign_net_buy": rng.normal(0, 1e8),
                    "prog_net_buy": rng.normal(0, 1e8),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_snapshot(n_panel_dates: int = 40, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-04-01", periods=n_panel_dates, freq="D")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        n_stocks = 8 + (i % 5)
        for j in range(n_stocks):
            base = 50000.0 + j * 500
            rows.append(
                {
                    "trade_date": date,
                    "stock_code": f"{100000 + (i + j) % 15:06d}",
                    "change_rate": float(rng.normal(0.5, 2.0)),
                    "open_price": base,
                    "high_price": base * 1.02,
                    "low_price": base * 0.98,
                    "close_price": base * 1.01,
                    "prev_close_price": base,
                    "market_cap_100m": 1000.0 + j * 100,
                    "trade_value_100m": 100.0 + j * 20,
                    "volume": 1_000_000 + j * 1000,
                    "selection_rank": float(j + 1),
                    "total_candidate_count": float(n_stocks),
                    "inst_net_buy": float((j - 2) * 5e7),
                    "foreign_net_buy": float(j * 3e7),
                    "prog_net_buy": float((j - 1) * 1e7),
                    "volume_power": 110.0 + j,
                    "avg_trade_value": 90.0 + j,
                    "kospi_change": 0.3,
                    "kosdaq_change": 0.1,
                    "v_kospi": 15.0,
                    "v_kosdaq": 18.0,
                    "buy_price": base * 1.01,
                    "market_type": "KOSPI" if j % 2 == 0 else "KOSDAQ",
                    "theme_sector": f"theme{j % 3}",
                    "net_return": float(rng.normal(0.2, 1.2)),
                }
            )
    return pd.DataFrame(rows)


def test_catalog_is_deterministic_and_distinct() -> None:
    """SCENARIO_FEATURE_CATALOG_01: 결정적, 중복 없는 후보 이름."""
    first = build_catalog()
    second = build_catalog()
    names1 = [d.name for d in first]
    names2 = [d.name for d in second]
    assert names1 == names2
    assert len(names1) == len(set(names1))


def test_catalog_candidate_count_within_research_target() -> None:
    """SCENARIO_FEATURE_CATALOG_01: 600--1000 후보 수 불변량."""
    definitions = build_catalog()
    assert MIN_CANDIDATES <= len(definitions) <= MAX_CANDIDATES


def test_catalog_rejects_duplicate_definitions() -> None:
    """카탈로그는 중복 소스/변환 선언을 거부합니다."""
    snapshot, history = _synthetic_snapshot(), _synthetic_history()
    build_causal_feature_matrix(snapshot, history)
    definitions = build_catalog()
    transform_keys = {
        (d.family, d.source_columns, d.lookback_groups, d.unit) for d in definitions
    }
    assert len(transform_keys) == len(definitions)


def test_catalog_matrix_is_unique_numeric_finite_or_nan() -> None:
    """SCENARIO_FEATURE_CATALOG_01: 고유 컬럼, 수치 유한 또는 NaN, inf 금지."""
    snapshot, history = _synthetic_snapshot(), _synthetic_history()
    matrix, manifest = build_causal_feature_matrix(snapshot, history)
    assert matrix.columns.is_unique
    assert MIN_CANDIDATES <= matrix.shape[1] <= MAX_CANDIDATES
    arr = matrix.to_numpy(dtype=np.float64)
    assert np.isfinite(arr[np.isfinite(arr)]).all()
    assert not np.isinf(arr).any()
    assert set(manifest["feature_name"]) == set(matrix.columns)


def test_catalog_excludes_target_future_columns() -> None:
    """SCENARIO_FEATURE_CATALOG_01: 타깃/미래/매도가 파생 컬럼을 생성하지 않습니다."""
    snapshot, history = _synthetic_snapshot(), _synthetic_history()
    matrix, _manifest = build_causal_feature_matrix(snapshot, history)
    banned = {"net_return", "sell_price", "target_return", "target_rank", "target_good", "target_bad"}
    for col in matrix.columns:
        assert col not in banned
        assert "sell_price" not in col
        assert "target" not in col
        assert "net_return" not in col


def test_catalog_manifest_records_family_and_lookback() -> None:
    """카탈로그 매니페스트는 family/source_columns/lookback_groups/availability_rule 을 기록합니다."""
    snapshot, history = _synthetic_snapshot(), _synthetic_history()
    _matrix, manifest = build_causal_feature_matrix(snapshot, history)
    assert {
        "feature_name",
        "family",
        "source_columns",
        "lookback_groups",
        "availability_rule",
        "unit",
        "panel_scope",
    }.issubset(set(manifest.columns))
    sheet = manifest[manifest["family"] == "sheet_level"]
    assert (sheet["availability_rule"] == "at_decision_time").all()
    lagged = manifest[manifest["family"] == "lagged_state"]
    assert (lagged["availability_rule"] == "prior_date_history_only").all()


def test_same_date_sheet_fields_accepted_and_history_prior_only() -> None:
    """SCENARIO_FEATURE_CATALOG_02: 동일 날짜 시트 필드 허용, 동일 날짜 EOD 이력 제외."""
    snapshot, history = _synthetic_snapshot(), _synthetic_history()
    matrix, _manifest = build_causal_feature_matrix(snapshot, history)
    # 동일 날짜 시트 파생 피처가 유한하게 생성됩니다.
    assert np.isfinite(matrix["snap_change_rate"].to_numpy(dtype=np.float64)).any()
    assert np.isfinite(matrix["snap_buy_gap"].to_numpy(dtype=np.float64)).any()

    # 스냅샷 특정 (stock, date) 의 직전 이력 행 값을 인위적으로 고정해,
    # as-of 조인이 동일 날짜 EOD 행이 아니라 직전 날짜 행을 사용하는지 확인합니다.
    target_date = snapshot["trade_date"].iloc[5]
    target_stock = snapshot["stock_code"].iloc[5]
    hist = history.copy()
    hist.loc[(hist["symbol"] == target_stock) & (hist["date"] == target_date), "close"] = 12345.0

    single = snapshot[snapshot["trade_date"] == target_date]
    single_stock = single[single["stock_code"] == target_stock]
    if single_stock.empty:
        pytest.skip("target snapshot row not available")
    matrix2, _ = build_causal_feature_matrix(snapshot, hist)
    idx = single_stock.index[0]
    row_prev = matrix2.loc[idx, "hist_prev_ret"]
    # 동일 날짜 close=12345 는 지난 수익률에 반영되면 안 됩니다.
    prior_rows = hist[(hist["symbol"] == target_stock) & (hist["date"] < target_date)]
    if len(prior_rows) >= 2:
        last_prior = prior_rows.iloc[-1]
        second_prior = prior_rows.iloc[-2]
        expected_prev_ret = last_prior["close"] / second_prior["close"] - 1
        assert row_prev == pytest.approx(expected_prev_ret)


def test_later_history_dates_do_not_change_prior_rows() -> None:
    """SCENARIO_FEATURE_CATALOG_02: 이후 날짜 변경은 이전 피처 행을 바꾸지 않습니다."""
    snapshot, history = _synthetic_snapshot(), _synthetic_history()
    matrix1, _ = build_causal_feature_matrix(snapshot, history)

    extra_dates = pd.date_range("2025-07-01", periods=10, freq="D")
    rng = np.random.default_rng(9)
    extra = _synthetic_history(n_dates=10, n_symbols=15, seed=11)
    extra["date"] = extra_dates[0] + pd.to_timedelta(np.arange(len(extra)) % 10, unit="D")
    history2 = pd.concat([history, extra], ignore_index=True)
    matrix2, _ = build_causal_feature_matrix(snapshot, history2)

    before = matrix1.to_numpy(dtype=np.float64)
    after = matrix2.to_numpy(dtype=np.float64)
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_validate_price_history_rejects_duplicates_and_unparseable() -> None:
    """캐노니컬 로더는 (symbol, date) 중복/파싱 불가 날짜/수치 오염을 거부합니다."""
    history = _synthetic_history(n_dates=5, n_symbols=2)
    dup = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_price_history(dup)

    bad_date = history.copy()
    bad_date["date"] = bad_date["date"].astype(object)
    bad_date.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="unparseable"):
        validate_price_history(bad_date)

    bad_num = history.copy()
    bad_num["close"] = "abc"
    with pytest.raises(ValueError, match="numeric"):
        validate_price_history(bad_num)

    missing_col = history.drop(columns=["volume"])
    with pytest.raises(ValueError, match="required columns"):
        validate_price_history(missing_col)


def test_causal_matrix_requires_snapshot_sources_and_valid_history() -> None:
    """필수 스냅샷 소스 누락/이력 누락은 fail-closed 로 실패합니다."""
    snapshot, history = _synthetic_snapshot(), _synthetic_history()
    missing = snapshot.drop(columns=["change_rate"])
    with pytest.raises(ValueError, match="required source"):
        build_causal_feature_matrix(missing, history)
    with pytest.raises(ValueError, match="price history is required"):
        build_causal_feature_matrix(snapshot, pd.DataFrame())
