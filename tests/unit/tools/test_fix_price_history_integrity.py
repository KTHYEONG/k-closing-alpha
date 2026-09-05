"""test_fix_price_history_integrity.py 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.tools.fix_price_history_integrity import fix_price_history_frame, run_fix_price_history


def test_fix_price_history_frame_restores_prev_close() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["005930", "005930", "005930", "000660", "000660"],
            "date": pd.to_datetime(["2025-07-30", "2025-07-31", "2025-08-01", "2025-07-31", "2025-08-01"]),
            "close": [70000.0, 71000.0, 68000.0, 120000.0, 115000.0],
            # 오염된 prev_close: 첫날 diff(-1000), 2025-08-01 diff(-3000, -5000)
            "prev_close": [-1000.0, 70000.0, -3000.0, 500.0, -5000.0],
            "open": [70000.0, 70500.0, 68500.0, 119000.0, 116000.0],
            "high": [70500.0, 71500.0, 68500.0, 121000.0, 117000.0],
            "low": [69500.0, 70000.0, 67500.0, 118000.0, 114000.0],
            "volume": [100, 200, 300, 400, 500],
            "market": ["KOSPI", "KOSPI", "KOSPI", "KOSPI", "KOSPI"],
        }
    )

    fixed = fix_price_history_frame(df)
    # 정렬 후: 000660 (행 0, 1), 005930 (행 2, 3, 4)

    # 000660
    assert fixed.loc[0, "symbol"] == "000660"
    assert pd.isna(fixed.loc[0, "prev_close"])  # 첫 거래일은 NA
    assert fixed.loc[1, "symbol"] == "000660"
    assert fixed.loc[1, "prev_close"] == 120000.0  # 2025-08-01 prev_close 복원

    # 005930
    assert fixed.loc[2, "symbol"] == "005930"
    assert pd.isna(fixed.loc[2, "prev_close"])  # 첫 거래일은 NA
    assert fixed.loc[3, "symbol"] == "005930"
    assert fixed.loc[3, "prev_close"] == 70000.0
    assert fixed.loc[4, "symbol"] == "005930"
    assert fixed.loc[4, "prev_close"] == 71000.0  # 2025-08-01 prev_close 복원


def test_run_fix_price_history_file_dry_run_and_backup(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "symbol": ["005930", "005930"],
            "date": pd.to_datetime(["2025-07-31", "2025-08-01"]),
            "open": [70000.0, 70500.0],
            "high": [70500.0, 71500.0],
            "low": [69500.0, 70000.0],
            "close": [71000.0, 68000.0],
            "prev_close": [-1000.0, -3000.0],
            "market_cap_100m": [1.0, 1.0],
            "trade_value_100m": [1.0, 1.0],
            "daily_change_pct": [0.01, -0.04],
            "market": ["KOSPI", "KOSPI"],
            "volume": [100, 200],
            "foreign_netbuy": [1.0, 1.0],
            "inst_netbuy": [1.0, 1.0],
            "program_netbuy": [1.0, 1.0],
            "kospi_pct": [0.01, -0.01],
            "kosdaq_pct": [0.01, -0.01],
            "v_kospi": [18.0, 19.0],
            "v_kosdaq": [18.0, 19.0],
        }
    )
    p = tmp_path / "price_history.parquet"
    df.to_parquet(p, index=False)

    # Dry run
    res_dry = run_fix_price_history(p, dry_run=True)
    assert res_dry["status"] == "dry_run_success"
    assert not p.with_suffix(".parquet.bak").exists()

    # Actual run
    res = run_fix_price_history(p, backup=True, dry_run=False)
    assert res["status"] == "success"
    assert p.with_suffix(".parquet.bak").exists()

    reloaded = pd.read_parquet(p)
    assert pd.isna(reloaded.loc[0, "prev_close"])
    assert reloaded.loc[1, "prev_close"] == 71000
