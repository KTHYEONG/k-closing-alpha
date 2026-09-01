"""Synthetic trade log fixture for champion integration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_trade_log(n_dates: int = 200, per_day: int = 8) -> pd.DataFrame:
    """Generate a synthetic Korean-header trade log with learnable edge."""
    rng = np.random.default_rng(7)
    # Generate dates ending near 2024 so OOS reserve_start 2023-11-01 splits
    end = pd.Timestamp("2024-01-01")
    dates = pd.bdate_range(end=end, periods=n_dates)
    rows: list[dict[str, str]] = []
    for d in dates:
        for j in range(per_day):
            e = rng.normal()
            change = 2 + 0.5 * e
            rows.append(
                {
                    "매수날짜": d.strftime("%Y-%m-%d"),
                    "종목코드": f"{j:06d}",
                    "(시가)": "10000",
                    "(고가)": "10400",
                    "(저가)": "9800",
                    "(종가)": "10200",
                    "(전일종가)": "10000",
                    "(시가총액, 억)": "5000",
                    "(거래대금, 억)": "300",
                    "(등락률)": f"{change:.2f}",
                    "(선정 순위)": str(j + 1),
                    "(기관_순매수)": f"{e*100:.0f}",
                    "(외국인_순매수)": f"{e*80:.0f}",
                    "(프로그램_순매수)": f"{e*50:.0f}",
                    "(체결강도)": "120",
                    "(시장구분)": "KOSPI",
                    "(총 종목 수)": str(per_day),
                    "(평균 거래대금)": "250",
                    "(kospi, %)": "0.3",
                    "(kosdaq, %)": "0.1",
                    "v_kospi": "18",
                    "v_kosdaq": "20",
                    "(거래량)": "100000",
                    "(테마/섹터)": "반도체",
                    "(차트분석)": "거래량 폭증",
                    "(매수 가격)": "10200",
                    "(매도 가격)": f"{10200*(1+0.01*e):.0f}",
                    "(수익률, %)": f"{e:.2f}",
                }
            )
    return pd.DataFrame(rows)
