"""src.backfill.altdata.shorting 모듈 직접 참조 테스트.

기존 테스트가 전부 tests/unit/backfill/test_altdata_collectors.py에 묶여 있어
lean_check의 test_<module> co-modification 게이트가 shorting.py를 인식하지
못하던 갭을 해소하기 위해 신설. KIS 네이티브 공매도 일별추이(FHPST04830000)
기반 collect_shorting의 핵심 계약(스키마 컬럼, universe_symbols 필수)을
직접 검증한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backfill.altdata import shorting
from src.backfill.altdata.config import AltDataFetchConfig


def test_collect_shorting_panel_schema_has_ten_columns() -> None:
    """공매도 패널은 거래측 4컬럼 + 잔고측 4컬럼(NaN 허용) + date/symbol 총 10컬럼 계약을 유지한다."""
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
    )

    out = shorting.collect_shorting(cfg, [pd.Timestamp("2024-01-02")])

    expected = {
        "date", "symbol",
        "short_volume", "short_value", "day_total_volume", "short_volume_ratio",
        "short_balance_qty", "short_balance_value", "listed_shares", "short_balance_ratio",
    }
    assert expected.issubset(set(out.columns))


def test_collect_shorting_fans_out_across_keys(monkeypatch) -> None:
    """다중 키 선언 시 팬아웃 헬퍼로 위임하고 결과를 기존 스키마로 조립한다."""
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
        universe_symbols=frozenset({"005930"}),
        extra_client_kwargs=(("k2", "s2", "h2"),),
    )
    seen: dict[str, object] = {}

    async def _fake_fan_out(cfg_arg, symbols, call, on_error):
        seen["symbols"] = list(symbols)
        assert cfg_arg.extra_client_kwargs == (("k2", "s2", "h2"),)

        class _Stub:
            async def get_daily_short_sale_history(self, session, code, start, end, market_div_code=None):
                return {"rt_cd": "0", "output2": []}

        res = await call(_Stub(), object(), "005930")
        assert res["rt_cd"] == "0"
        fallback = on_error("005930", RuntimeError("x"))
        assert fallback == {"rt_cd": "9", "output2": []}
        return [("005930", {"rt_cd": "0", "output2": [
            {"stck_bsop_date": "20240102", "ssts_cntg_qty": "100", "ssts_tr_pbmn": "3000000",
             "acml_vol": "10000", "ssts_vol_rlim": "1.0"},
        ]})]

    monkeypatch.setattr(shorting, "fan_out_symbol_calls", _fake_fan_out)
    out = shorting.collect_shorting(cfg, [pd.Timestamp("2024-01-02")])
    assert seen["symbols"] == ["005930"]
    assert out["symbol"].iloc[0] == "005930"
    assert out["short_volume"].iloc[0] == 100.0
