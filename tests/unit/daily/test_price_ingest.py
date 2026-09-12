"""Nightly price_history ingest tests: KRX bulk rows, KIS/Kiwoom flows, index, corporate actions."""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

CAL = pd.bdate_range("2015-11-02", "2026-09-11")


def _krx_raw(rows: list[dict], market: str = "KOSPI") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ISU_CD": r["symbol"], "MKT_NM": market, "TDD_OPNPRC": str(r.get("open", r["close"])),
            "TDD_HGPRC": str(r.get("high", r["close"])), "TDD_LWPRC": str(r.get("low", r["close"])),
            "TDD_CLSPRC": str(r["close"]), "CMPPREVDD_PRC": str(r["close"] - r["prev_close"]),
            "ACC_TRDVOL": str(r.get("volume", 1000)), "ACC_TRDVAL": str(r.get("value", 5_000_000_000)),
            "MKTCAP": str(r.get("mcap", 100_000_000_000)),
        }
        for r in rows
    ])


class FakeKis:
    base_url = "https://kis.test"

    def __init__(self, calendar=CAL, fail_investor=(), fail_program=(), index_rt="0"):
        self.calendar = list(calendar)
        self.fail_investor = set(fail_investor)
        self.fail_program = set(fail_program)
        self.index_rt = index_rt
        self.index_calls = 0

    async def ensure_token(self, session):
        return "token"

    def _get_headers(self, tr):
        return {"tr_id": tr}

    def _close(self, code, d):
        base = 2000.0 if code == "0001" else 800.0
        return base * (1.0 + 0.0005 * self.calendar.index(d))

    async def get_market_index_history(self, session, code, start, end):
        self.index_calls += 1
        if self.index_rt != "0":
            return {"rt_cd": self.index_rt, "msg1": "boom"}
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        days = [d for d in self.calendar if s <= d <= e][-50:]
        return {"rt_cd": "0", "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "bstp_nmix_prpr": f"{self._close(code, d):.2f}"} for d in reversed(days)]}

    async def _handle_request(self, method, url, headers=None, params=None):
        tr, sym = headers["tr_id"], params["FID_INPUT_ISCD"]
        anchor = pd.Timestamp(params["FID_INPUT_DATE_1"])
        days = [d for d in self.calendar if d <= anchor][-30:]
        if tr == "FHPTJ04160001":
            if sym in self.fail_investor:
                return {"rt_cd": "2", "msg1": "TIME LIMIT 00:00 ~ 15:40"}
            return {"rt_cd": "0", "output1": {}, "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "orgn_ntby_tr_pbmn": "100", "frgn_ntby_tr_pbmn": "-50"} for d in reversed(days)]}
        if sym in self.fail_program:
            return {"rt_cd": "1", "msg1": "program fail"}
        return {"rt_cd": "0", "output": [{"stck_bsop_date": d.strftime("%Y%m%d"), "whol_smtn_ntby_tr_pbmn": "7"} for d in reversed(days)]}


class FakeKiwoom:
    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []

    async def _post_tr(self, session, api_id, path, body, cont_yn="N", next_key=""):
        self.calls.append((api_id, body["stk_cd"]))
        if body["stk_cd"] in self.fail:
            return {"return_code": 1, "return_msg": "kiwoom fail"}, {}
        anchor = pd.Timestamp(body["dt"])
        days = [d for d in CAL if d <= anchor][-100:]
        return {"return_code": 0, "stk_invsr_orgn": [{"dt": d.strftime("%Y%m%d"), "orgn": "+30", "frgnr_invsr": "-20", "natfor": "--1"} for d in reversed(days)]}, {}


class FakeToss:
    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls: list[str] = []

    async def get_program_trades(self, session, symbol, count=100, until=None):
        self.calls.append(symbol)
        if symbol in self.fail:
            return {"error": {"code": "invalid-request", "message": "toss fail"}}
        return {"result": {"records": [
            {"date": "2026-09-10", "arbitrage": {"netBuyVolume": "5"}, "nonArbitrage": {"netBuyVolume": "6"}},
        ]}}


class _Session:
    get = None


def _panel_rows(symbol: str, days: list[pd.Timestamp], closes: list[float], volume: float = 1000.0) -> list[dict]:
    rows, prev = [], closes[0]
    for d, c in zip(days, closes, strict=True):
        rows.append({"date": d, "symbol": symbol, "open": c, "high": c, "low": c, "close": c, "prev_close": prev,
                     "market_cap_100m": 1000.0, "trade_value_100m": 50.0, "daily_change_pct": 0.0, "market": "KOSPI",
                     "volume": volume, "foreign_netbuy": 1.0, "inst_netbuy": 1.0, "program_netbuy": 1.0,
                     "kospi_pct": 0.0, "kosdaq_pct": 0.0, "v_kospi": 0.0, "v_kosdaq": 0.0, "chg_ratio": 0.0})
        prev = c
    return rows


def _write_panel(path, rows: list[dict]) -> None:
    from src.data.panel_integrity import heal_price_history_panel
    from src.data.parquet_codec import write_price_history_parquet

    write_price_history_parquet(heal_price_history_panel(pd.DataFrame(rows)), path)


def test_normalize_krx_daily_maps_columns_and_base_price() -> None:
    from src.daily.price_ingest import KRX_ROW_COLUMNS, normalize_krx_daily

    # Given: KRX strings with thousands separators; base price = close - change
    raw = _krx_raw([{"symbol": "005930", "close": 70000, "prev_close": 68000, "open": 68800, "high": 70500, "low": 68500, "volume": 1_000_000, "value": 70_000_000_000, "mcap": 4_200_000_000_000}], "KOSPI")
    raw["TDD_CLSPRC"] = "70,000"

    # When
    out = normalize_krx_daily(raw, pd.Timestamp("2026-09-10"))

    # Then
    assert list(out.columns) == list(KRX_ROW_COLUMNS)
    row = out.iloc[0]
    assert row["symbol"] == "005930" and row["market"] == "KOSPI"
    assert row["close"] == 70000 and row["prev_close"] == 68000 and row["open"] == 68800
    assert row["trade_value_100m"] == pytest.approx(700.0)
    assert row["market_cap_100m"] == pytest.approx(42000.0)
    assert row["date"] == pd.Timestamp("2026-09-10")


def test_normalize_krx_daily_empty_missing_and_duplicate() -> None:
    from src.daily.price_ingest import KRX_ROW_COLUMNS, normalize_krx_daily

    empty = normalize_krx_daily(pd.DataFrame(), pd.Timestamp("2026-09-10"))
    assert empty.empty and list(empty.columns) == list(KRX_ROW_COLUMNS)
    raw = _krx_raw([{"symbol": "000001", "close": 100, "prev_close": 100}])
    with pytest.raises(ValueError, match="MKTCAP"):
        normalize_krx_daily(raw.drop(columns=["MKTCAP"]), pd.Timestamp("2026-09-10"))
    with pytest.raises(ValueError, match="repeats"):
        normalize_krx_daily(pd.concat([raw, raw]), pd.Timestamp("2026-09-10"))


def test_fetch_krx_daily_unpublished_partial_and_full(monkeypatch) -> None:
    import src.daily.price_ingest as mod

    kospi = _krx_raw([{"symbol": "000001", "close": 100, "prev_close": 99}], "KOSPI")
    kosdaq = _krx_raw([{"symbol": "900001", "close": 50, "prev_close": 50}], "KOSDAQ")

    def _fetch(kosdaq_frame):
        return lambda ep, ymd, cfg: kospi if ep == mod.KRX_ENDPOINT_STK_DAILY else kosdaq_frame

    # Given/When/Then: both markets empty -> unpublished (empty), one empty -> partial (raise), both -> concat
    monkeypatch.setattr(mod, "fetch_krx_openapi_day_strict", lambda ep, ymd, cfg: pd.DataFrame())
    assert mod.fetch_krx_daily(pd.Timestamp("2026-09-10"), cfg=None).empty
    monkeypatch.setattr(mod, "fetch_krx_openapi_day_strict", _fetch(pd.DataFrame()))
    with pytest.raises(RuntimeError, match="partial"):
        mod.fetch_krx_daily(pd.Timestamp("2026-09-10"), cfg=None)
    monkeypatch.setattr(mod, "fetch_krx_openapi_day_strict", _fetch(kosdaq))
    out = mod.fetch_krx_daily(pd.Timestamp("2026-09-10"), cfg=None)
    assert sorted(out["symbol"]) == ["000001", "900001"]
    assert set(out["market"]) == {"KOSPI", "KOSDAQ"}


def test_fetch_index_closes_pages_backwards_until_start() -> None:
    from src.daily.price_ingest import fetch_index_closes

    # Given: a 120-day calendar served 50 latest rows per call
    cal = list(pd.bdate_range("2026-03-02", periods=120))
    kis = FakeKis(calendar=cal)

    # When
    out = asyncio.run(fetch_index_closes(kis, _Session(), "0001", cal[0], cal[-1]))

    # Then: every date collected once, ascending, in 3 pages
    assert list(out["date"]) == cal
    assert kis.index_calls == 3
    assert out["close"].iloc[1] == pytest.approx(kis._close("0001", cal[1]))


def test_fetch_index_closes_raises_on_vendor_error() -> None:
    from src.daily.price_ingest import fetch_index_closes

    kis = FakeKis(calendar=list(pd.bdate_range("2026-03-02", periods=10)), index_rt="1")
    with pytest.raises(RuntimeError, match="rt_cd=1"):
        asyncio.run(fetch_index_closes(kis, _Session(), "0001", pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-13")))


def test_fetch_index_closes_raises_when_paging_stalls() -> None:
    from src.daily.price_ingest import fetch_index_closes

    class StuckKis(FakeKis):
        async def get_market_index_history(self, session, code, start, end):
            days = self.calendar[-50:]
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "bstp_nmix_prpr": "1"} for d in days]}

    kis = StuckKis(calendar=list(pd.bdate_range("2026-01-01", periods=80)))
    with pytest.raises(RuntimeError, match="stalled"):
        asyncio.run(fetch_index_closes(kis, _Session(), "0001", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")))


def test_fetch_index_closes_returns_empty_when_range_has_no_rows() -> None:
    from src.daily.price_ingest import fetch_index_closes

    kis = FakeKis(calendar=list(pd.bdate_range("2026-03-02", periods=5)))
    out = asyncio.run(fetch_index_closes(kis, _Session(), "0001", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")))
    assert out.empty and list(out.columns) == ["date", "close"]


def test_compute_index_columns_returns_and_volatility() -> None:
    from src.backfill.price.factors import compute_vkospi_proxy
    from src.daily.price_ingest import compute_index_columns

    days = pd.bdate_range("2026-01-02", periods=30)
    kospi = pd.DataFrame({"date": days, "close": 100.0 * np.cumprod(1 + 0.01 * np.sin(np.arange(30)))})
    kosdaq = pd.DataFrame({"date": days, "close": 50.0 + np.arange(30)})

    out = compute_index_columns(kospi, kosdaq).set_index("date")

    # Then: close-to-close returns per index, 20-day HV proxy, NaN warm-up
    assert out.loc[days[1], "kosdaq_pct"] == pytest.approx(51.0 / 50.0 - 1.0)
    assert out.loc[days[5], "kospi_pct"] == pytest.approx(kospi.close[5] / kospi.close[4] - 1.0)
    expect = compute_vkospi_proxy(kospi, output_col="v").set_index("date")["v"]
    assert np.isnan(out.loc[days[10], "v_kospi"])
    assert out.loc[days[25], "v_kospi"] == pytest.approx(expect.loc[days[25]])


def test_attach_index_columns_overwrites_and_fails_on_missing_date() -> None:
    from src.daily.price_ingest import attach_index_columns

    table = pd.DataFrame({"date": pd.to_datetime(["2026-09-09", "2026-09-10"]), "kospi_pct": [0.01, -0.02], "kosdaq_pct": [0.03, 0.0], "v_kospi": [20.0, 21.0], "v_kosdaq": [30.0, 31.0]})
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-09"]), "symbol": ["A", "B"], "kospi_pct": [9.0, 9.0], "kosdaq_pct": [9.0, 9.0], "v_kospi": [9.0, 9.0], "v_kosdaq": [9.0, 9.0]})

    out = attach_index_columns(panel, table)

    assert out["kospi_pct"].tolist() == [-0.02, 0.01]
    assert out["v_kosdaq"].tolist() == [31.0, 30.0]
    assert panel["kospi_pct"].tolist() == [9.0, 9.0]
    with pytest.raises(ValueError, match="missing 1 panel dates"):
        attach_index_columns(pd.concat([panel, panel.assign(date=pd.Timestamp("2026-09-11"))]), table)


def test_plan_new_dates_bounds_and_gap_limit() -> None:
    from src.daily.price_ingest import FLOW_WINDOW_TRADING_DAYS, plan_new_dates

    cal = list(pd.bdate_range("2026-08-03", "2026-09-11"))
    out = plan_new_dates(pd.Timestamp("2026-09-08"), cal, pd.Timestamp("2026-09-10 21:30"))
    assert out == [pd.Timestamp("2026-09-09"), pd.Timestamp("2026-09-10")]
    assert plan_new_dates(pd.Timestamp("2026-09-11"), cal, pd.Timestamp("2026-09-11")) == []
    long_cal = list(pd.bdate_range("2026-01-02", periods=FLOW_WINDOW_TRADING_DAYS + 5))
    with pytest.raises(ValueError, match="flow window"):
        plan_new_dates(pd.Timestamp("2025-12-31"), long_cal, long_cal[-1])


def test_select_tail_rows_never_overwrites_existing() -> None:
    from src.daily.price_ingest import select_tail_rows

    rows = pd.DataFrame({"date": pd.to_datetime(["2026-09-08", "2026-09-09", "2026-09-09", "2026-09-10"]), "symbol": ["A", "A", "B", "C"]})
    last = {"A": pd.Timestamp("2026-09-08"), "B": pd.Timestamp("2026-09-09")}

    out = select_tail_rows(rows, last)

    assert list(zip(out["symbol"], out["date"].dt.strftime("%m-%d"), strict=True)) == [("A", "09-09"), ("C", "09-10")]


def test_parsers_normalize_vendor_rows_and_raise_on_errors() -> None:
    from src.daily.price_ingest import VendorResponseError, parse_kis_investor_rows, parse_kis_program_rows, parse_kiwoom_investor_rows

    inv = parse_kis_investor_rows({"rt_cd": "0", "output2": [{"stck_bsop_date": "20260910", "orgn_ntby_tr_pbmn": "-568864", "frgn_ntby_tr_pbmn": "+911744"}]})
    assert inv.iloc[0]["inst_netbuy"] == -568864 and inv.iloc[0]["foreign_netbuy"] == 911744
    prg = parse_kis_program_rows({"rt_cd": "0", "output": [{"stck_bsop_date": "20260910", "whol_smtn_ntby_tr_pbmn": "1,234"}]})
    assert prg.iloc[0]["program_netbuy"] == 1234
    kw = parse_kiwoom_investor_rows({"return_code": 0, "rows": [{"dt": "20260910", "orgn": "+1152626", "frgnr_invsr": "-1546879", "natfor": "--4705"}]})
    # Then: KIS foreign = Kiwoom frgnr_invsr + natfor
    assert kw.iloc[0]["inst_netbuy"] == 1152626 and kw.iloc[0]["foreign_netbuy"] == -1551584
    with pytest.raises(VendorResponseError, match="TIME LIMIT"):
        parse_kis_investor_rows({"rt_cd": "2", "msg1": "TIME LIMIT 00:00 ~ 15:40"})
    with pytest.raises(VendorResponseError, match="program"):
        parse_kis_program_rows({"rt_cd": "1", "msg1": "program fail"})
    with pytest.raises(VendorResponseError, match="return_code=1"):
        parse_kiwoom_investor_rows({"return_code": 1, "return_msg": "x"})


def test_fetch_symbol_flows_uses_kis_then_kiwoom_then_none() -> None:
    from src.daily.price_ingest import FLOW_COLUMNS, fetch_symbol_flows

    kis = FakeKis(fail_investor={"000002", "000003"}, fail_program={"000003"})
    kiwoom = FakeKiwoom(fail={"000003"})

    ok, src_ok, prg_ok = asyncio.run(fetch_symbol_flows(kis, kiwoom, _Session(), "000001", "20260910"))
    fb, src_fb, prg_fb = asyncio.run(fetch_symbol_flows(kis, kiwoom, _Session(), "000002", "20260910"))
    none, src_none, prg_none = asyncio.run(fetch_symbol_flows(kis, kiwoom, _Session(), "000003", "20260910"))

    assert (src_ok, src_fb, src_none) == ("kis", "kiwoom", "none")
    # Then: 000003 has no toss client injected, so its failed program flow has no fallback -> "none"
    assert (prg_ok, prg_fb, prg_none) == ("kis", "kis", "none")
    last_ok = ok[ok["date"] == pd.Timestamp("2026-09-10")].iloc[0]
    assert (last_ok["inst_netbuy"], last_ok["foreign_netbuy"], last_ok["program_netbuy"]) == (100, -50, 7)
    last_fb = fb[fb["date"] == pd.Timestamp("2026-09-10")].iloc[0]
    assert (last_fb["inst_netbuy"], last_fb["foreign_netbuy"], last_fb["program_netbuy"]) == (30, -21, 7)
    assert none.empty or none[list(FLOW_COLUMNS)].isna().all().all()
    assert kiwoom.calls == [("ka10059", "000002"), ("ka10059", "000003")]


def test_fetch_symbol_flows_without_kiwoom_client_marks_none() -> None:
    from src.daily.price_ingest import fetch_symbol_flows

    flows, source, program_source = asyncio.run(
        fetch_symbol_flows(FakeKis(fail_investor={"000009"}), None, _Session(), "000009", "20260910")
    )

    assert source == "none"
    assert program_source == "kis"
    assert flows["inst_netbuy"].isna().all()


def test_fetch_all_flows_counts_sources() -> None:
    from src.daily.price_ingest import fetch_all_flows

    flows, sources = asyncio.run(
        fetch_all_flows(FakeKis(fail_investor={"000002"}), FakeKiwoom(), _Session(), ["000001", "000002"], "20260910")
    )

    assert sources == {"investor": {"kis": 1, "kiwoom": 1}, "program": {"kis": 2}}
    assert set(flows["symbol"]) == {"000001", "000002"}


def test_fetch_all_flows_empty_symbol_list() -> None:
    from src.daily.price_ingest import FLOW_COLUMNS, fetch_all_flows

    flows, sources = asyncio.run(fetch_all_flows(FakeKis(), None, _Session(), [], "20260910"))

    assert flows.empty and list(flows.columns) == ["date", "symbol", *FLOW_COLUMNS]
    assert sources == {"investor": {}, "program": {}}


def test_assemble_new_rows_and_check_flow_coverage() -> None:
    from src.daily.price_ingest import assemble_new_rows, check_flow_coverage, normalize_krx_daily

    krx = normalize_krx_daily(_krx_raw([
        {"symbol": "A", "close": 110, "prev_close": 100, "volume": 10},
        {"symbol": "B", "close": 50, "prev_close": 50, "volume": 0},
        {"symbol": "C", "close": 20, "prev_close": 20, "volume": 5},
    ]), pd.Timestamp("2026-09-10"))
    flows = pd.DataFrame({
        "date": ["2026-09-10", "2026-09-10"], "symbol": ["A", "C"],
        "inst_netbuy": [1.0, np.nan], "foreign_netbuy": [2.0, 3.0], "program_netbuy": [0.0, np.nan],
    })

    rows = assemble_new_rows(krx, flows)

    assert rows.set_index("symbol").loc["A", "chg_ratio"] == pytest.approx(0.1)
    assert rows["daily_change_pct"].equals(rows["chg_ratio"])
    assert np.isnan(rows.set_index("symbol").loc["B", "inst_netbuy"])
    # Then: halted B (volume 0) is excluded; C lacks inst -> 1/2 traded rows covered
    assert check_flow_coverage(rows, min_coverage=0.5) == {"2026-09-10": 0.5}
    with pytest.raises(ValueError, match="coverage below"):
        check_flow_coverage(rows)
    # Then: C also lacks program -> the program gate is independent of the investor gate
    assert check_flow_coverage(rows, columns=("program_netbuy",), min_coverage=0.5, label="program") == {"2026-09-10": 0.5}
    with pytest.raises(ValueError, match="program flow coverage below"):
        check_flow_coverage(rows, columns=("program_netbuy",), label="program")


def test_merge_and_adjust_scales_history_for_events_and_ignores_gaps() -> None:
    from src.daily.price_ingest import merge_and_adjust

    d = [pd.Timestamp(x) for x in ("2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10")]
    panel = pd.DataFrame(
        _panel_rows("A", d[:3], [10000.0, 10000.0, 10000.0], volume=1000.0)
        + _panel_rows("B", d[:1], [3000.0])
        + _panel_rows("D", d[:2], [800.0, 800.0], volume=500.0)
    )
    new = pd.DataFrame([
        {"date": d[3], "symbol": "A", "open": 2100.0, "high": 2100.0, "low": 2100.0, "close": 2100.0, "prev_close": 2000.0, "volume": 5000.0},
        {"date": d[3], "symbol": "B", "open": 900.0, "high": 900.0, "low": 900.0, "close": 900.0, "prev_close": 1000.0, "volume": 10.0},
        {"date": d[2], "symbol": "D", "open": 400.0, "high": 400.0, "low": 400.0, "close": 400.0, "prev_close": 400.0, "volume": 50.0},
        {"date": d[3], "symbol": "D", "open": 800.0, "high": 800.0, "low": 800.0, "close": 800.0, "prev_close": 800.0, "volume": 50.0},
        {"date": d[3], "symbol": "E", "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "prev_close": 5.0, "volume": 1.0},
    ])

    merged, events = merge_and_adjust(panel, new, d)

    m = merged.set_index(["symbol", "date"])
    # Then: A split 5:1 on 09-10 -> earlier rows x0.2, volume x5; the event row itself untouched
    assert m.loc[("A", d[0]), "close"] == pytest.approx(2000.0)
    assert m.loc[("A", d[2]), "volume"] == pytest.approx(5000.0)
    assert m.loc[("A", d[3]), "close"] == pytest.approx(2100.0)
    # Then: B's prior row is 09-07 (not the previous trading day) -> a gap, never an event
    assert m.loc[("B", d[0]), "close"] == pytest.approx(3000.0)
    # Then: D has two events (x0.5 on 09-09, x2 on 09-10): oldest rows scaled by both, 09-09 row by the later one
    assert m.loc[("D", d[0]), "close"] == pytest.approx(800.0 * 0.5 * 2.0)
    assert m.loc[("D", d[2]), "close"] == pytest.approx(800.0)
    assert m.loc[("D", d[0]), "volume"] == pytest.approx(500.0)
    assert ("E", d[3]) in m.index
    got = sorted(zip(events["symbol"], events["date"].dt.strftime("%m-%d"), events["factor"].round(6), strict=True))
    assert got == [("A", "09-10", 0.2), ("D", "09-09", 0.5), ("D", "09-10", 2.0)]


def _orchestrate_fakes(monkeypatch, published: set[str]):
    import src.daily.price_ingest as mod

    days = {x: pd.Timestamp(x) for x in ("2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11")}
    krx = {
        "2026-09-08": [{"symbol": "000001", "close": 10000, "prev_close": 10000}, {"symbol": "000002", "close": 20000, "prev_close": 20000}, {"symbol": "000003", "close": 500, "prev_close": 500}],
        "2026-09-09": [{"symbol": "000001", "close": 10000, "prev_close": 10000}, {"symbol": "000002", "close": 21000, "prev_close": 20000}, {"symbol": "000003", "close": 510, "prev_close": 500}],
        "2026-09-10": [{"symbol": "000001", "close": 10500, "prev_close": 10000}, {"symbol": "000002", "close": 4300, "prev_close": 4200}, {"symbol": "000003", "close": 520, "prev_close": 510}],
    }
    calls: list[str] = []

    def _fetch(d, cfg):
        key = pd.Timestamp(d).strftime("%Y-%m-%d")
        calls.append(key)
        if key not in published:
            return pd.DataFrame(columns=list(mod.KRX_ROW_COLUMNS))
        return mod.normalize_krx_daily(_krx_raw(krx[key]), pd.Timestamp(key))

    monkeypatch.setattr(mod, "fetch_krx_daily", _fetch)
    return mod, days, calls


def test_run_price_ingest_fills_new_date_stale_tail_and_new_listing(monkeypatch, tmp_path) -> None:
    # Given: panel ends 09-09; 000002 is stale since 09-07; 000003 is a new listing; KRX published 09-10 only as new
    path = tmp_path / "ph.parquet"
    mod, days, calls = _orchestrate_fakes(monkeypatch, {"2026-09-08", "2026-09-09", "2026-09-10"})
    _write_panel(path, _panel_rows("000001", [days["2026-09-07"], days["2026-09-08"], days["2026-09-09"]], [10000.0] * 3)
                 + _panel_rows("000002", [days["2026-09-03"], days["2026-09-04"], days["2026-09-07"]], [20000.0] * 3, volume=100.0))

    # When
    report = asyncio.run(mod.run_price_ingest(today=pd.Timestamp("2026-09-11"), path=path, krx_cfg=object(), kis=FakeKis(), kiwoom=FakeKiwoom()))

    # Then: 09-11 unpublished stops the new-date loop; stale tail 09-08..09-09 is back-filled
    assert calls[:2] == ["2026-09-10", "2026-09-11"]
    assert report.ingested_dates == ["2026-09-08", "2026-09-09", "2026-09-10"]
    assert report.wrote is True and report.n_corporate_events == 1
    assert report.investor_sources == {"kis": 3}
    # Then: no symbol fails program on the default FakeKis fixture -> every program source is "kis"
    assert report.program_sources == {"kis": 3}
    out = pd.read_parquet(path)
    out["symbol"] = out["symbol"].astype(str)
    m = out.set_index(["symbol", "date"])
    # Then: 000001 keeps its stored rows, gains 09-10 only
    assert int((out["symbol"] == "000001").sum()) == 4
    # Then: 000002 split on 09-10 (base 4200 vs prior close 21000) -> history x0.2, volume x5
    assert m.loc[("000002", days["2026-09-03"]), "close"] == 4000
    assert m.loc[("000002", days["2026-09-03"]), "volume"] == 500
    assert m.loc[("000002", days["2026-09-09"]), "close"] == 4200
    assert int((out["symbol"] == "000003").sum()) == 3
    # Then: prev_close equals the prior close on every consecutive row; index columns from composites
    o = out.sort_values(["symbol", "date"])
    prior = o.groupby("symbol")["close"].shift(1)
    assert ((o["prev_close"].astype(float) - prior.astype(float)).abs().dropna() <= 0.5).all()
    kis = FakeKis()
    expect = kis._close("0001", days["2026-09-10"]) / kis._close("0001", days["2026-09-09"]) - 1.0
    assert m.loc[("000001", days["2026-09-10"]), "kospi_pct"] == pytest.approx(expect)
    assert m.loc[("000001", days["2026-09-10"]), "inst_netbuy"] == 100


def test_run_price_ingest_noop_when_unpublished_and_index_unchanged(monkeypatch, tmp_path) -> None:
    path = tmp_path / "ph.parquet"
    mod, days, calls = _orchestrate_fakes(monkeypatch, {"2026-09-08", "2026-09-09", "2026-09-10"})
    _write_panel(path, _panel_rows("000001", [days["2026-09-08"], days["2026-09-09"], days["2026-09-10"]], [10000.0] * 3))
    first = asyncio.run(mod.run_price_ingest(today=pd.Timestamp("2026-09-11"), path=path, krx_cfg=object(), kis=FakeKis(), kiwoom=FakeKiwoom()))
    assert first.wrote is True and first.ingested_dates == []
    before = path.stat().st_mtime_ns

    # When: nothing new published and index columns already corrected
    second = asyncio.run(mod.run_price_ingest(today=pd.Timestamp("2026-09-11"), path=path, krx_cfg=object(), kis=FakeKis(), kiwoom=FakeKiwoom()))

    # Then
    assert second.wrote is False and second.n_new_rows == 0
    assert path.stat().st_mtime_ns == before


def test_run_price_ingest_fails_closed_on_low_flow_coverage(monkeypatch, tmp_path) -> None:
    path = tmp_path / "ph.parquet"
    mod, days, _ = _orchestrate_fakes(monkeypatch, {"2026-09-10"})
    _write_panel(path, _panel_rows("000001", [days["2026-09-08"], days["2026-09-09"]], [10000.0] * 2))
    before = path.stat().st_mtime_ns

    # When: KIS and Kiwoom both fail investor flow for every symbol
    with pytest.raises(ValueError, match="coverage below"):
        asyncio.run(mod.run_price_ingest(today=pd.Timestamp("2026-09-11"), path=path, krx_cfg=object(), kis=FakeKis(fail_investor={"000001", "000002", "000003"}), kiwoom=FakeKiwoom(fail={"000001", "000002", "000003"})))

    # Then: nothing written
    assert path.stat().st_mtime_ns == before


def test_run_price_ingest_raises_when_past_trading_day_missing(monkeypatch, tmp_path) -> None:
    # Given: 000002 is stale since 09-07 but KRX has no rows for the past day 09-08
    path = tmp_path / "ph.parquet"
    mod, days, _ = _orchestrate_fakes(monkeypatch, {"2026-09-10", "2026-09-09"})
    _write_panel(path, _panel_rows("000001", [days["2026-09-07"], days["2026-09-08"], days["2026-09-09"]], [10000.0] * 3)
                 + _panel_rows("000002", [days["2026-09-04"], days["2026-09-07"]], [20000.0] * 2))

    with pytest.raises(RuntimeError, match="no rows for past trading day 2026-09-08"):
        asyncio.run(mod.run_price_ingest(today=pd.Timestamp("2026-09-11"), path=path, krx_cfg=object(), kis=FakeKis(), kiwoom=FakeKiwoom()))


def test_run_price_ingest_builds_default_clients(monkeypatch, tmp_path) -> None:
    import src.api.kis.client as kis_mod
    import src.api.kiwoom.client as kiwoom_mod
    import src.api.toss.client as toss_mod

    path = tmp_path / "ph.parquet"
    mod, days, _ = _orchestrate_fakes(monkeypatch, set())
    _write_panel(path, _panel_rows("000001", [days["2026-09-09"], days["2026-09-10"]], [10000.0] * 2))
    built: list[str] = []
    monkeypatch.setattr(kis_mod, "KisApiClient", lambda *a, **k: built.append("kis") or FakeKis())
    monkeypatch.setattr(kiwoom_mod, "KiwoomApiClient", lambda *a, **k: built.append("kiwoom") or FakeKiwoom())
    monkeypatch.setattr(toss_mod, "TossApiClient", lambda *a, **k: built.append("toss") or FakeToss())

    # When: no clients and no KRX config injected (production call shape)
    report = asyncio.run(mod.run_price_ingest(today=pd.Timestamp("2026-09-11"), path=path))

    # Then
    assert built == ["kis", "kiwoom", "toss"]
    assert report.ingested_dates == []


def test_run_price_ingest_missing_panel_raises(tmp_path) -> None:
    from src.daily.price_ingest import run_price_ingest

    with pytest.raises(FileNotFoundError, match="price_history not found"):
        asyncio.run(run_price_ingest(today=pd.Timestamp("2026-09-11"), path=tmp_path / "none.parquet", krx_cfg=object(), kis=FakeKis(), kiwoom=FakeKiwoom()))


def test_main_runs_ingest_with_defaults(monkeypatch) -> None:
    import src.daily.price_ingest as mod

    seen = []

    async def _fake(**kwargs):
        seen.append(kwargs)
        return mod.IngestReport(ingested_dates=[], n_new_rows=0, n_corporate_events=0)

    monkeypatch.setattr(mod, "run_price_ingest", _fake)
    mod.main()
    assert seen == [{}]


def test_kca_price_ingest_service_runs_module() -> None:
    with open("deploy/systemd/kca-price-ingest.service", encoding="utf-8") as f:
        content = f.read()
    assert "Type=oneshot" in content
    assert "ExecStart=%h/.local/bin/uv run python -m src.daily.price_ingest" in content


def test_kca_price_ingest_timer_slots_avoid_decision_and_flow_windows() -> None:
    import re

    with open("deploy/systemd/kca-price-ingest.timer", encoding="utf-8") as f:
        content = f.read()
    slots = [(int(h), int(m)) for h, m in re.findall(r"OnCalendar=Mon\.\.Fri (\d{2}):(\d{2}):\d{2} Asia/Seoul", content)]
    # Then: an evening slot (same-day publication) plus morning catch-up slots, none in 15:00-15:45 or 08:55-09:10 (paper-exit)
    assert len(slots) >= 3
    assert any(h >= 18 for h, _ in slots) and any(h < 12 for h, _ in slots)
    for h, m in slots:
        t = h * 60 + m
        assert not (15 * 60 <= t <= 15 * 60 + 45)
        assert not (8 * 60 + 55 <= t <= 9 * 60 + 10)
        assert t < 15 * 60 or t >= 20 * 60 + 30
    assert "Persistent=true" in content and "Unit=kca-price-ingest.service" in content


def test_parse_toss_program_rows_normalizes_and_raises() -> None:
    import pandas as pd
    import pytest

    from src.daily.price_ingest import VendorResponseError, parse_toss_program_rows

    ok = parse_toss_program_rows({"result": {"records": [
        {"date": "2026-09-10", "arbitrage": {"netBuyVolume": "3"}, "nonArbitrage": {"netBuyVolume": "4"}},
    ]}})
    assert ok.iloc[0]["program_netbuy"] == 7
    assert ok.iloc[0]["date"] == pd.Timestamp("2026-09-10")

    empty = parse_toss_program_rows({"result": {"records": []}})
    assert list(empty.columns) == ["date", "program_netbuy"] and empty.empty

    with pytest.raises(VendorResponseError, match="Toss program-trades"):
        parse_toss_program_rows({"error": {"code": "invalid-request", "message": "bad symbol"}})


def test_fetch_symbol_flows_program_falls_back_to_toss_when_kis_fails() -> None:
    from src.daily.price_ingest import fetch_symbol_flows

    kis = FakeKis(fail_program={"000004"})
    toss = FakeToss()

    flows, inv_src, prg_src = asyncio.run(
        fetch_symbol_flows(kis, None, _Session(), "000004", "20260910", toss)
    )

    assert inv_src == "kis"
    assert prg_src == "toss"
    last = flows[flows["date"] == pd.Timestamp("2026-09-10")].iloc[0]
    assert last["program_netbuy"] == 11
    assert toss.calls == ["000004"]


def test_fetch_all_flows_uses_toss_program_fallback() -> None:
    from src.daily.price_ingest import fetch_all_flows

    kis = FakeKis(fail_program={"000004"})
    toss = FakeToss()

    flows, sources = asyncio.run(fetch_all_flows(kis, None, _Session(), ["000001", "000004"], "20260910", toss))

    assert sources["program"] == {"kis": 1, "toss": 1}
    assert toss.calls == ["000004"]


def test_run_price_ingest_fails_closed_on_low_program_flow_coverage(monkeypatch, tmp_path) -> None:
    # Given: investor flow is healthy but every symbol's KIS program call fails,
    # and the Toss fallback also fails for every symbol (no viable recovery).
    # toss=None would let production build a *real* TossApiClient (its
    # documented default-construction convenience for callers), which reaches
    # the live network under real credentials -- passing a failing FakeToss
    # keeps this hermetic while preserving the "no working fallback" intent.
    path = tmp_path / "ph.parquet"
    mod, days, _ = _orchestrate_fakes(monkeypatch, {"2026-09-10"})
    _write_panel(path, _panel_rows("000001", [days["2026-09-08"], days["2026-09-09"]], [10000.0] * 2))
    before = path.stat().st_mtime_ns

    # When
    with pytest.raises(ValueError, match="program flow coverage below"):
        asyncio.run(mod.run_price_ingest(
            today=pd.Timestamp("2026-09-11"), path=path, krx_cfg=object(),
            kis=FakeKis(fail_program={"000001", "000002", "000003"}), kiwoom=FakeKiwoom(),
            toss=FakeToss(fail={"000001", "000002", "000003"}),
        ))

    # Then: nothing written
    assert path.stat().st_mtime_ns == before
