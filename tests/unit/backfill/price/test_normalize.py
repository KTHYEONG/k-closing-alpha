

def test_normalize_symbol_history_derives_ratio_without_unit_sniffing() -> None:
    import numpy as np
    import pandas as pd

    from src.backfill.price.normalize import _normalize_symbol_history

    # Given: a pykrx-shaped slice whose vendor 등락률 is percent-encoded with a median
    # BELOW 1.0, which is exactly the case the old median>1.0 sniff failed to divide.
    idx = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"])
    ohlcv = pd.DataFrame(
        {
            "시가": [9900.0, 10000.0, 10500.0],
            "고가": [10100.0, 10600.0, 11000.0],
            "저가": [9850.0, 9950.0, 10450.0],
            "종가": [10000.0, 10500.0, 10920.0],
            "거래량": [1000.0, 1100.0, 1200.0],
            "거래대금": [1.0e10, 1.1e10, 1.2e10],
            "등락률": [0.5, 5.0, 4.0],
        },
        index=idx,
    )

    # When
    out = _normalize_symbol_history(ohlcv, pd.DataFrame(), "005930", "KOSPI")

    # Then: the vendor 등락률 is ignored entirely and the ratio comes from close/prev_close.
    chg = out["daily_change_pct"].to_numpy(dtype=float)
    assert np.isnan(chg[0]), "first bar of the slice has no predecessor"
    np.testing.assert_allclose(chg[1:], [0.05, 0.04], rtol=1e-12)
    assert "daily_change_pct_raw" not in out.columns
