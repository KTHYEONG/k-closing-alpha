from __future__ import annotations

import pandas as pd


def test_trade_log_symbols_normalizes_codes_and_ignores_blanks() -> None:
    from src.backfill.backfill_altdata import trade_log_symbols

    frame = pd.DataFrame({"종목코드": ["5930", " 000660 ", None, "nan", "", "0"]})
    assert trade_log_symbols(frame) == frozenset({"005930", "000660"})
    assert trade_log_symbols(pd.DataFrame({"x": [1]})) == frozenset()
