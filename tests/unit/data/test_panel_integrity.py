"""Panel integrity scenarios (contract: ml_dataset_integrity)."""
from __future__ import annotations




def test_prepare_price_panel_overwrites_percent_encoded_vendor_change() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import PANEL_INTEGRITY_COLUMNS, prepare_price_panel

    # Given: row 0 is percent-encoded (8.0 == +8%), the rest are ratio-encoded
    ph = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-01", "2023-02-02"]),
        "symbol": ["1", "1", "2", "2"],
        "open": [10000.0, 10800.0, 20000.0, 20200.0],
        "high": [10900.0, 11100.0, 20300.0, 20500.0],
        "low": [9900.0, 10700.0, 19900.0, 20100.0],
        "close": [10800.0, 11000.0, 20200.0, 20400.0],
        "prev_close": [10000.0, 10800.0, 20000.0, 20200.0],
        "volume": [1000.0, 1000.0, 2000.0, 2000.0],
        "market_cap_100m": [900.0, 900.0, 1200.0, 1200.0],
        "trade_value_100m": [300.0, 300.0, 500.0, 500.0],
        "market": ["KOSPI", "KOSPI", "KOSDAQ", "KOSDAQ"],
        "daily_change_pct": [8.0, 11000.0 / 10800.0 - 1.0, 0.01, 20400.0 / 20200.0 - 1.0],
    })

    # When
    out, prov = prepare_price_panel(ph)

    # Then: no row is lost, symbols are zero-padded, and the vendor column is now the derived ratio
    assert len(out) == 4
    assert out["symbol"].tolist() == ["000001", "000001", "000002", "000002"]
    assert PANEL_INTEGRITY_COLUMNS.issubset(set(out.columns))
    assert np.isclose(out["chg_ratio"].iloc[0], 0.08)
    assert np.isclose(out["daily_change_pct"].iloc[0], 0.08)
    np.testing.assert_allclose(
        out["daily_change_pct"].to_numpy(dtype=float),
        out["chg_ratio"].to_numpy(dtype=float),
    )
    assert prov.n_unit_mismatch_vendor == 1
    assert prov.n_valid_chg == 4
    assert prov.n_input == 4
    assert out.attrs["panel_provenance"]["n_unit_mismatch_vendor"] == 1

    # And: running the normalizer again changes nothing
    again, _ = prepare_price_panel(out)
    np.testing.assert_allclose(
        again["chg_ratio"].to_numpy(dtype=float),
        out["chg_ratio"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        again["tick_cost_bp"].to_numpy(dtype=float),
        out["tick_cost_bp"].to_numpy(dtype=float),
    )


def test_prepare_price_panel_nans_invalid_rows_without_dropping_them() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel

    # Given: row 1 is a +100% split artefact, row 2 has an unusable prev_close
    ph = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-03"]),
        "symbol": ["000001", "000001", "000001"],
        "open": [1000.0, 1100.0, 2100.0],
        "high": [1050.0, 2100.0, 2200.0],
        "low": [990.0, 1050.0, 2000.0],
        "close": [1000.0, 2000.0, 2100.0],
        "prev_close": [1000.0, 1000.0, 0.0],
        "volume": [100.0, 100.0, 100.0],
        "market_cap_100m": [900.0, np.nan, 900.0],
        "trade_value_100m": [300.0, np.nan, 300.0],
        "market": ["KOSPI", "KOSPI", "KOSPI"],
        "daily_change_pct": [0.0, 1.0, 0.05],
    })

    # When
    out, prov = prepare_price_panel(ph)

    # Then: every row survives, but the two bad ones carry NaN rather than a wrong number
    assert len(out) == 3
    chg = out["chg_ratio"].to_numpy(dtype=float)
    assert np.isclose(chg[0], 0.0)
    assert np.isnan(chg[1])
    assert np.isnan(chg[2])
    assert prov.n_invalid_limit_violation == 1
    assert prov.n_invalid_prev_close == 1
    assert prov.n_invalid_price == 0
    assert prov.n_valid_chg == 1
    # mc_clean forward fills within the symbol; tv_clean falls back to close * volume / 1e8
    assert np.isclose(out["mc_clean"].iloc[1], 900.0)
    assert np.isclose(out["tv_clean"].iloc[1], 2000.0 * 100.0 / 1e8)
    # A NaN change can never satisfy a screen floor
    assert bool(chg[1] >= 0.02) is False


def test_prepare_price_panel_raises_on_missing_required_columns() -> None:
    import pandas as pd
    import pytest

    from src.data.panel_integrity import prepare_price_panel

    # Given: prev_close and volume are absent
    ph = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"]),
        "symbol": ["000001"],
        "open": [1000.0],
        "high": [1050.0],
        "low": [990.0],
        "close": [1020.0],
        "market_cap_100m": [900.0],
        "trade_value_100m": [300.0],
    })

    # When / Then
    with pytest.raises(ValueError, match="prepare_price_panel is missing required columns") as excinfo:
        prepare_price_panel(ph)
    message = str(excinfo.value)
    assert "prepare_price_panel is missing required columns" in message
    assert "prev_close" in message
    assert "volume" in message


def test_prepare_price_panel_tick_cost_is_point_in_time() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel

    # Given: 15,000원 either side of the 2023-01-25 reform, and 150,000원 on both boards pre-reform
    ph = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-24", "2023-01-25", "2023-01-24", "2023-01-24"]),
        "symbol": ["000001", "000001", "000002", "000003"],
        "open": [15000.0, 15000.0, 150000.0, 150000.0],
        "high": [15100.0, 15100.0, 151000.0, 151000.0],
        "low": [14900.0, 14900.0, 149000.0, 149000.0],
        "close": [15000.0, 15000.0, 150000.0, 150000.0],
        "prev_close": [15000.0, 15000.0, 150000.0, 150000.0],
        "volume": [100.0, 100.0, 100.0, 100.0],
        "market_cap_100m": [900.0, 900.0, 5000.0, 5000.0],
        "trade_value_100m": [300.0, 300.0, 800.0, 800.0],
        "market": ["KOSPI", "KOSPI", "KOSDAQ", "KOSPI"],
        "daily_change_pct": [0.0, 0.0, 0.0, 0.0],
    })

    # When
    out, prov = prepare_price_panel(ph)
    by_key = {(r.symbol, r.date.strftime("%Y-%m-%d")): r.tick_cost_bp for r in out.itertuples()}

    # Then: pre-reform KOSPI 15,000원 -> 50원 tick; post-reform -> 10원 tick
    assert np.isclose(by_key[("000001", "2023-01-24")], 50.0 / 15000.0 * 1e4)
    assert np.isclose(by_key[("000001", "2023-01-25")], 10.0 / 15000.0 * 1e4)
    # Pre-reform 150,000원: KOSDAQ tops out at 100원 while KOSPI charges 500원
    assert np.isclose(by_key[("000002", "2023-01-24")], 100.0 / 150000.0 * 1e4)
    assert np.isclose(by_key[("000003", "2023-01-24")], 500.0 / 150000.0 * 1e4)
    assert prov.n_rows_post_tick_reform == 1
    assert prov.n_days_post_tick_reform == 1


def test_prepare_price_panel_provenance_reports_ceiling_and_survivorship() -> None:
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel

    # Given: 000001 hits a +29.5% close-at-high on day 1; 000002 disappears after day 1
    ph = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-01"]),
        "symbol": ["000001", "000001", "000002"],
        "open": [10100.0, 12950.0, 5000.0],
        "high": [12950.0, 13100.0, 5100.0],
        "low": [10000.0, 12800.0, 4950.0],
        "close": [12950.0, 13000.0, 5050.0],
        "prev_close": [10000.0, 12950.0, 5000.0],
        "volume": [100.0, 100.0, 100.0],
        "market_cap_100m": [900.0, 900.0, 700.0],
        "trade_value_100m": [300.0, 300.0, 200.0],
        "market": ["KOSPI", "KOSPI", "KOSDAQ"],
        "daily_change_pct": [0.295, 0.00386, 0.01],
    })

    # When
    out, prov = prepare_price_panel(ph)

    # Then
    assert prov.n_ceiling == 1
    assert bool(out.loc[out["date"] == pd.Timestamp("2023-02-01"), "is_ceiling"].iloc[0]) is True
    assert prov.n_symbols == 2
    assert prov.n_days == 2
    assert prov.n_symbols_absent_at_end == 1
    assert prov.date_min == "2023-02-01"
    assert prov.date_max == "2023-02-02"
    assert "n_symbols_absent_at_end=1" in prov.to_log_kv()
    assert prov.to_dict()["n_ceiling"] == 1


def test_load_price_panel_reads_parquet_and_raises_when_absent(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.data.panel_integrity import load_price_panel

    # Given: a parquet whose vendor change column is percent-encoded
    ph = pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-02"]),
        "symbol": ["000001", "000001"],
        "open": [10000.0, 10800.0],
        "high": [10900.0, 11100.0],
        "low": [9900.0, 10700.0],
        "close": [10800.0, 11000.0],
        "prev_close": [10000.0, 10800.0],
        "volume": [1000.0, 1000.0],
        "market_cap_100m": [900.0, 900.0],
        "trade_value_100m": [300.0, 300.0],
        "market": ["KOSPI", "KOSPI"],
        "daily_change_pct": [8.0, 1.8518518518518516],
    })
    path = tmp_path / "price_history.parquet"
    ph.to_parquet(path)

    # When
    out, prov = load_price_panel(path)

    # Then
    assert len(out) == 2
    assert np.isclose(out["chg_ratio"].iloc[0], 0.08)
    assert float(np.nanmax(np.abs(out["daily_change_pct"].to_numpy(dtype=float)))) < 1.0
    assert prov.n_unit_mismatch_vendor == 2

    # And: a missing path fails closed
    with pytest.raises(FileNotFoundError, match="price_history parquet not found"):
        load_price_panel(tmp_path / "nope.parquet")




def test_prepare_price_panel_empty_frame_provenance_is_zeroed_not_crashed() -> None:
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel

    # Given: a zero-row frame with every required column present and correctly typed
    ph = pd.DataFrame({
        "date": pd.Series([], dtype="datetime64[ns]"),
        "symbol": pd.Series([], dtype=str),
        "open": pd.Series([], dtype=float),
        "high": pd.Series([], dtype=float),
        "low": pd.Series([], dtype=float),
        "close": pd.Series([], dtype=float),
        "prev_close": pd.Series([], dtype=float),
        "volume": pd.Series([], dtype=float),
        "market_cap_100m": pd.Series([], dtype=float),
        "trade_value_100m": pd.Series([], dtype=float),
    })

    # When
    out, prov = prepare_price_panel(ph)

    # Then: no crash on empty datetime min/max, and every count is honestly zero
    assert len(out) == 0
    assert prov.n_input == 0
    assert prov.n_symbols == 0
    assert prov.n_days == 0
    assert prov.date_min == ""
    assert prov.date_max == ""
    assert prov.n_symbols_absent_at_end == 0
    assert prov.n_valid_chg == 0
    assert prov.n_ceiling == 0
    assert prov.n_rows_post_tick_reform == 0
    assert prov.n_days_post_tick_reform == 0


def test_heal_price_history_panel_heals_slice_boundary_prev_close() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import heal_price_history_panel

    # Given: rows 0-1 came from an earlier backfill run, row 2 is the first row of a
    # later incremental slice and therefore carries a NaN prev_close.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"]),
        "symbol": ["5930", "5930", "5930"],
        "close": [10000.0, 10500.0, 10920.0],
        "prev_close": [np.nan, 10000.0, np.nan],
        "daily_change_pct": [np.nan, 0.05, np.nan],
    })

    # When
    out = heal_price_history_panel(df)

    # Then: no row is lost, the slice boundary is healed from the prior bar,
    # and the first bar of the symbol legitimately stays NaN.
    assert len(out) == 3
    assert out["symbol"].tolist() == ["005930", "005930", "005930"]
    assert np.isnan(out["prev_close"].to_numpy(dtype=float)[0])
    assert np.isnan(out["daily_change_pct"].to_numpy(dtype=float)[0])
    np.testing.assert_allclose(out["prev_close"].to_numpy(dtype=float)[1:], [10000.0, 10500.0])
    np.testing.assert_allclose(out["daily_change_pct"].to_numpy(dtype=float)[1:], [0.05, 0.04])
    np.testing.assert_allclose(
        out["chg_ratio"].to_numpy(dtype=float)[1:],
        out["daily_change_pct"].to_numpy(dtype=float)[1:],
    )
def test_heal_price_history_panel_overwrites_percent_encoded_vendor_column() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import heal_price_history_panel

    # Given: symbol 000001 stores percent (5.0 meaning +5%), symbol 000002 stores ratio.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-03"]),
        "symbol": ["1", "1", "2", "2"],
        "close": [10000.0, 10500.0, 20000.0, 20400.0],
        "prev_close": [9900.0, 10000.0, 19800.0, 20000.0],
        "daily_change_pct": [1.0101, 5.0, 0.0101, 0.02],
    })

    # When
    out = heal_price_history_panel(df)

    # Then: every row is the deterministic derivation regardless of the stored unit.
    expected = np.array([10000.0 / 9900.0 - 1.0, 0.05, 20000.0 / 19800.0 - 1.0, 0.02])
    np.testing.assert_allclose(out["daily_change_pct"].to_numpy(dtype=float), expected, rtol=1e-12)
    np.testing.assert_allclose(out["chg_ratio"].to_numpy(dtype=float), expected, rtol=1e-12)
def test_heal_price_history_panel_is_idempotent_and_slice_independent() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import heal_price_history_panel

    # Given: one full history, and the same history split into two contiguous slices
    # whose second slice lost prev_close at its boundary (what incremental backfill does).
    full = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"]),
        "symbol": ["000660"] * 4,
        "close": [10000.0, 10500.0, 10920.0, 11000.0],
        "prev_close": [9800.0, 10000.0, 10500.0, 10920.0],
        "daily_change_pct": [0.0204, 0.05, 0.04, 0.0073],
    })
    slice_a = full.iloc[:2].copy()
    slice_b = full.iloc[2:].copy()
    slice_b.loc[slice_b.index[0], "prev_close"] = np.nan
    slice_b.loc[slice_b.index[0], "daily_change_pct"] = np.nan
    merged = pd.concat([slice_a, slice_b], ignore_index=True)

    # When
    healed_full = heal_price_history_panel(full)
    healed_merged = heal_price_history_panel(merged)
    healed_twice = heal_price_history_panel(healed_merged)

    # Then: slice independence and idempotence both hold.
    np.testing.assert_allclose(
        healed_merged["daily_change_pct"].to_numpy(dtype=float),
        healed_full["daily_change_pct"].to_numpy(dtype=float),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        healed_twice["daily_change_pct"].to_numpy(dtype=float),
        healed_merged["daily_change_pct"].to_numpy(dtype=float),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        healed_twice["prev_close"].to_numpy(dtype=float),
        healed_merged["prev_close"].to_numpy(dtype=float),
        rtol=1e-12,
    )
def test_heal_price_history_panel_limit_violation_and_bad_prev_close_yield_nan() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import heal_price_history_panel

    # Given: symbol 000003 halves overnight (unadjusted split -> limit breach);
    # symbol 000004 has a zero prev_close carried in from a bad vendor row.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-03"]),
        "symbol": ["000003", "000003", "000004", "000004"],
        "close": [10000.0, 5000.0, 0.0, 500.0],
        "prev_close": [9900.0, 10000.0, np.nan, 0.0],
        "daily_change_pct": [0.0101, -0.5, np.nan, np.nan],
    })

    # When
    out = heal_price_history_panel(df)

    # Then: no row is dropped and both pathologies are NaN, never 0.0.
    assert len(out) == 4
    chg = out["daily_change_pct"].to_numpy(dtype=float)
    assert np.isnan(chg[1]), "limit violation must be NaN, not a clipped value"
    assert np.isnan(chg[2])
    assert np.isnan(chg[3]), "prev_close <= 0 must be NaN, not 0.0"
    assert not np.any(chg[np.isfinite(chg)] == 0.0)
def test_heal_price_history_panel_accepts_downcast_nullable_int_columns() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import heal_price_history_panel

    # Given: prices stored as pandas nullable Int32, as downcast_price_history_frame emits.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-03-03"]),
        "symbol": ["005380", "005380"],
        "close": pd.array([10000, 10500], dtype="Int32"),
        "prev_close": pd.array([9900, pd.NA], dtype="Int32"),
        "daily_change_pct": [0.0101, np.nan],
    })

    # When
    out = heal_price_history_panel(df)

    # Then: the Int32 NA at the boundary is healed from the prior close.
    assert len(out) == 2
    np.testing.assert_allclose(out["daily_change_pct"].to_numpy(dtype=float)[1], 0.05, rtol=1e-12)
def test_assert_price_history_units_clean_raises_on_percent_encoded_row() -> None:
    import pandas as pd
    import pytest

    from src.data.panel_integrity import PanelIntegrityError, assert_price_history_units_clean

    # Given: row 1 stores +5% as 5.0 instead of 0.05.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-03-03"]),
        "symbol": ["005930", "005930"],
        "close": [10000.0, 10500.0],
        "prev_close": [9900.0, 10000.0],
        "daily_change_pct": [10000.0 / 9900.0 - 1.0, 5.0],
    })

    # When / Then
    with pytest.raises(PanelIntegrityError) as excinfo:
        assert_price_history_units_clean(df)
    assert "005930" in str(excinfo.value)
def test_assert_price_history_units_clean_passes_on_healed_panel_and_rejects_missing_columns() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.data.panel_integrity import (
        PanelIntegrityError,
        assert_price_history_units_clean,
        heal_price_history_panel,
    )

    # Given: a panel whose first bar per symbol is legitimately NaN.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-02"]),
        "symbol": ["005930", "005930", "000660"],
        "close": [10000.0, 10500.0, 20000.0],
        "prev_close": [np.nan, 10000.0, np.nan],
        "daily_change_pct": [np.nan, 0.05, np.nan],
    })
    healed = heal_price_history_panel(df)

    # When / Then: the healed panel passes the gate (returns None).
    assert assert_price_history_units_clean(healed) is None

    # And: a frame without prev_close fails closed instead of skipping the check.
    with pytest.raises(PanelIntegrityError) as excinfo:
        assert_price_history_units_clean(healed.drop(columns=["prev_close"]))
    assert "prev_close" in str(excinfo.value)
def test_unit_sniffing_heuristic_is_absent_from_the_repository() -> None:
    import importlib
    from pathlib import Path

    import pytest

    # Given: the two modules that previously sniffed units from a median.
    for rel in ("src/backfill/price/normalize.py", "src/ml/exit_policy.py"):
        source = Path(rel).read_text(encoding="utf-8")
        # Then: no median-based unit inference and no divide-by-100 rescue remains.
        assert "median" not in source, f"{rel} still infers a unit from a distribution statistic"
        assert "/ 100.0" not in source, f"{rel} still rescales a change column by 100"

    # And: the dead module carrying the third copy is deleted.
    assert not Path("src/ml/forward_path.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.ml.forward_path")
