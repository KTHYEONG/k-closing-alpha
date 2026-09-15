"""KRX PIT panel rebuild scenarios."""
from __future__ import annotations


def test_fetch_krx_history_checkpoints_and_skips_holidays(tmp_path) -> None:
    import pandas as pd

    from src.backfill import krx_panel_rebuild as mod
    from src.daily.price_ingest import KRX_ROW_COLUMNS

    fetched: list[str] = []

    def _fetch(date, cfg):
        key = pd.Timestamp(date).strftime("%Y-%m-%d")
        fetched.append(key)
        if key == "2026-09-09":
            return pd.DataFrame(columns=list(KRX_ROW_COLUMNS))
        return pd.DataFrame([{"date": pd.Timestamp(date), "symbol": "005930", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                              "prev_close": 1.0, "volume": 10.0, "trade_value_100m": 1.0, "market_cap_100m": 1.0, "market": "KOSPI"}])

    # When: 9/08(화)~9/10(목), 9/09 휴장
    out = mod.fetch_krx_history(pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-10"), object(), tmp_path, fetch_fn=_fetch)
    again = mod.fetch_krx_history(pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-10"), object(), tmp_path, fetch_fn=_fetch)

    # Then
    assert fetched == ["2026-09-08", "2026-09-09", "2026-09-10"]
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["2026-09-08.parquet", "2026-09-09.parquet", "2026-09-10.parquet"]
    assert list(out.columns) == list(KRX_ROW_COLUMNS)
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-09-08", "2026-09-10"]
    assert len(again) == 2
    empty = mod.fetch_krx_history(pd.Timestamp("2026-09-09"), pd.Timestamp("2026-09-09"), object(), tmp_path, fetch_fn=_fetch)
    assert empty.empty and list(empty.columns) == list(KRX_ROW_COLUMNS)



def test_derive_adjusted_prices_scales_history_keeps_raw_and_volume() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.backfill.krx_panel_rebuild import derive_adjusted_prices

    def _raw(rows):
        cols = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "trade_value_100m", "market_cap_100m", "market"]
        return pd.DataFrame([dict(zip(cols, r, strict=True)) for r in rows], columns=cols).assign(date=lambda x: pd.to_datetime(x["date"]))

    raw = _raw([
        ("2018-04-27", "005930", 2600000, 2660000, 2590000, 2650000, 2607000, 606216, 16112.4, 3402242.0, "KOSPI"),
        ("2018-04-30", "005930", 2650000, 2650000, 2650000, 2650000, 2650000, 0, 0.0, 3402242.0, "KOSPI"),
        ("2018-05-04", "005930", 53000, 53900, 51800, 51900, 53000, 39565391, 20780.2, 3331629.5, "KOSPI"),
        ("2018-05-08", "005930", 52600, 53200, 51900, 52600, 51900, 23104720, 12100.0, 3376000.0, "KOSPI"),
        ("2018-04-27", "000001", 1000, 1000, 1000, 1000, 1000, 100, 0.001, 10.0, "KOSDAQ"),
        ("2018-05-08", "000001", 1200, 1200, 1200, 1200, 1100, 100, 0.001, 10.0, "KOSDAQ"),
    ])

    # When
    out = derive_adjusted_prices(raw)

    # Then: 삼성전자 05-04 기준가 53,000 = 직전종가 2,650,000 x 0.02 -> 이전 행 전부 x0.02
    s = out[out["symbol"] == "005930"].set_index(out.loc[out["symbol"] == "005930", "date"].dt.strftime("%m-%d"))
    assert s.loc["04-27", "close"] == pytest.approx(53000.0)
    assert s.loc["04-27", "open"] == pytest.approx(52000.0)
    assert s.loc["04-27", "prev_close"] == pytest.approx(52140.0)
    assert s.loc["04-30", "close"] == pytest.approx(53000.0)
    assert s.loc["05-04", "close"] == pytest.approx(51900.0)
    assert s.loc["04-27", "close_raw"] == 2650000
    assert s.loc["04-27", "volume"] == 606216
    assert s.loc["04-30", "volume"] == 0
    # 05-08 prev_close 51,900 == 직전 종가 -> 이벤트 아님
    assert s.loc["05-08", "close"] == pytest.approx(52600.0)
    # 000001: 04-27 -> 05-08 은 달력상 연속 거래일이 아님(사이에 05-04 존재) -> 갭, 이벤트 아님
    a = out[out["symbol"] == "000001"]
    assert np.allclose(a["close"].to_numpy(), [1000.0, 1200.0])
    assert list(out.columns[:12]) == ["date", "symbol", "open", "high", "low", "close", "prev_close", "close_raw", "volume", "trade_value_100m", "market_cap_100m", "market"]



def test_carry_existing_flows_fills_only_matching_rows() -> None:
    import numpy as np
    import pandas as pd

    from src.backfill.krx_panel_rebuild import carry_existing_flows

    panel = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-10", "2026-09-11"]), "symbol": ["005930", "000660", "005930"], "close": [1.0, 2.0, 3.0]})
    old = pd.DataFrame({"date": pd.to_datetime(["2026-09-10", "2026-09-09"]), "symbol": ["005930", "005930"],
                        "inst_netbuy": [10.0, 99.0], "foreign_netbuy": [20.0, 99.0], "program_netbuy": [30.0, 99.0]})

    # When
    out = carry_existing_flows(panel, old)

    # Then
    assert len(out) == 3
    row = out[(out["symbol"] == "005930") & (out["date"] == "2026-09-10")].iloc[0]
    assert (row["inst_netbuy"], row["foreign_netbuy"], row["program_netbuy"]) == (10.0, 20.0, 30.0)
    assert np.isnan(out.loc[out["symbol"] == "000660", "inst_netbuy"]).all()
    assert np.isnan(out.loc[out["date"] == "2026-09-11", "program_netbuy"]).all()



def test_validate_rebuild_rejects_duplicates_shrink_and_reports_match_rate() -> None:
    import pandas as pd
    import pytest

    from src.backfill import krx_panel_rebuild as mod

    def _panel(rows):
        return pd.DataFrame(rows, columns=["date", "symbol", "close", "close_raw"]).assign(date=lambda x: pd.to_datetime(x["date"]))

    old = _panel([("2026-09-10", "005930", 100.0, 100.0), ("2026-09-10", "000660", 50.0, 50.0), ("2026-09-11", "005930", 101.0, 101.0)])
    new = _panel([("2026-09-10", "005930", 100.0, 100.0), ("2026-09-10", "000660", 50.0, 50.0), ("2026-09-10", "000001", 5.0, 5.0),
                  ("2026-09-11", "005930", 101.0, 101.0), ("2026-09-11", "000001", 5.0, 5.0)])

    # When
    metrics = mod.validate_rebuild(new, old)

    # Then
    assert not hasattr(mod, "REBUILD_MIN_MATCH_RATE")
    assert metrics == {"n_rows": 5, "n_symbols": 3, "n_common_rows": 3, "match_rate": 1.0, "n_dates": 2}

    with pytest.raises(ValueError, match="duplicate"):
        mod.validate_rebuild(pd.concat([new, new.iloc[[0]]], ignore_index=True), old)
    with pytest.raises(ValueError, match="fewer rows"):
        mod.validate_rebuild(new.iloc[[0]], old)
    # 구 패널과의 수정종가 괴리는 진단 지표로만 보고되고 실패 사유가 아니다
    drift = new.copy()
    drift.loc[drift["symbol"] == "005930", "close"] = 999.0
    assert mod.validate_rebuild(drift, old)["match_rate"] == pytest.approx(1 / 3)
    bad_raw = new.copy()
    bad_raw.loc[0, "close_raw"] = 0.0
    with pytest.raises(ValueError, match="close_raw"):
        mod.validate_rebuild(bad_raw, old)



def test_run_panel_rebuild_writes_validated_panel(tmp_path, monkeypatch) -> None:
    import asyncio

    import numpy as np
    import pandas as pd

    from src.backfill import krx_panel_rebuild as mod
    from src.data.panel_integrity import heal_price_history_panel
    from src.data.parquet_codec import write_price_history_parquet

    days = [pd.Timestamp(d) for d in ("2026-09-08", "2026-09-09", "2026-09-10")]
    cal = pd.bdate_range("2015-11-02", "2026-09-10")

    class _Kis:
        async def ensure_token(self, session):
            return "t"

        async def get_market_index_history(self, session, code, start, end):
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            base = 2000.0 if code == "0001" else 800.0
            sel = [d for d in cal if s <= d <= e][-50:]
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "bstp_nmix_prpr": f"{base * (1 + 0.0005 * cal.get_loc(d)):.2f}"} for d in reversed(sel)]}

    def _fetch(date, cfg):
        d = pd.Timestamp(date)
        rows = [{"date": d, "symbol": "005930", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "prev_close": 100.0, "volume": 10.0, "trade_value_100m": 1.0, "market_cap_100m": 1000.0, "market": "KOSPI"},
                {"date": d, "symbol": "999999", "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "prev_close": 5.0, "volume": 1.0, "trade_value_100m": 0.1, "market_cap_100m": 10.0, "market": "KOSDAQ"}]
        return pd.DataFrame(rows)

    old_path = tmp_path / "old.parquet"
    old_rows = [{"date": d, "symbol": "005930", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "prev_close": 100.0, "volume": 10.0,
                 "trade_value_100m": 1.0, "market_cap_100m": 1000.0, "market": "KOSPI", "daily_change_pct": 0.0, "chg_ratio": 0.0,
                 "inst_netbuy": 7.0, "foreign_netbuy": 8.0, "program_netbuy": 9.0, "kospi_pct": 0.0, "kosdaq_pct": 0.0, "v_kospi": 0.0, "v_kosdaq": 0.0} for d in days]
    write_price_history_parquet(heal_price_history_panel(pd.DataFrame(old_rows)), old_path)
    out_path = tmp_path / "rebuild.parquet"
    monkeypatch.setattr(mod, "fetch_krx_daily", _fetch)

    # When
    report = asyncio.run(mod.run_panel_rebuild(start=days[0], end=days[-1], checkpoint_dir=tmp_path / "cp", out_path=out_path, old_path=old_path, krx_cfg=object(), kis=_Kis()))

    # Then
    assert out_path.exists() and not (tmp_path / "price_history.parquet").exists()
    written = pd.read_parquet(out_path)
    written["symbol"] = written["symbol"].astype(str)
    assert report.n_rows == 6 and report.n_symbols == 2 and report.match_rate == 1.0
    assert sorted(written["symbol"].unique()) == ["005930", "999999"]
    assert (written.loc[written["symbol"] == "005930", "inst_netbuy"].astype(float) == 7.0).all()
    assert written.loc[written["symbol"] == "999999", "inst_netbuy"].isna().all()
    assert np.isfinite(written["kospi_pct"].astype(float)).all()
    assert (written["close_raw"].astype(float) == written["close"].astype(float)).all()



def test_run_panel_rebuild_fails_closed_on_missing_old_or_empty_krx(tmp_path, monkeypatch) -> None:
    import asyncio

    import pandas as pd
    import pytest

    import src.api.kis.client as kis_mod
    from src.backfill import krx_panel_rebuild as mod
    from src.daily.price_ingest import KRX_ROW_COLUMNS

    built: list[dict] = []
    monkeypatch.setattr(kis_mod, "KisApiClient", lambda *a, **k: built.append(k) or object())
    monkeypatch.setattr(kis_mod, "kis_data_client_kwargs", lambda: {"app_key": "DATA"})
    monkeypatch.setattr(mod, "fetch_krx_daily", lambda date, cfg: pd.DataFrame(columns=list(KRX_ROW_COLUMNS)))
    old_path = tmp_path / "old.parquet"

    # When/Then: 이전 패널 없음
    with pytest.raises(FileNotFoundError, match="old panel"):
        asyncio.run(mod.run_panel_rebuild(start=pd.Timestamp("2026-09-08"), end=pd.Timestamp("2026-09-09"), checkpoint_dir=tmp_path / "cp", out_path=tmp_path / "out.parquet", old_path=old_path, krx_cfg=object()))
    assert built == []

    # When/Then: KRX 응답 0행 (kis=None -> 데이터 계좌 클라이언트 생성)
    pd.DataFrame({"date": pd.to_datetime(["2026-09-08"]), "symbol": ["005930"], "close": [1.0]}).to_parquet(old_path, index=False)
    with pytest.raises(ValueError, match="no rows"):
        asyncio.run(mod.run_panel_rebuild(start=pd.Timestamp("2026-09-08"), end=pd.Timestamp("2026-09-09"), checkpoint_dir=tmp_path / "cp", out_path=tmp_path / "out.parquet", old_path=old_path, krx_cfg=object()))
    assert built == [{"app_key": "DATA"}]
    assert not (tmp_path / "out.parquet").exists()



def test_swap_panel_backs_up_live_file_then_replaces(tmp_path) -> None:
    import pytest

    from src.backfill.krx_panel_rebuild import swap_panel

    live = tmp_path / "price_history.parquet"
    live.write_bytes(b"old")
    rebuilt = tmp_path / "price_history_rebuild.parquet"
    rebuilt.write_bytes(b"new")

    # When
    backup = swap_panel(rebuilt, live)

    # Then
    assert live.read_bytes() == b"new" and not rebuilt.exists()
    assert backup.parent == tmp_path and backup.name.startswith("price_history.") and backup.name.endswith(".bak.parquet")
    assert backup.read_bytes() == b"old"
    with pytest.raises(FileNotFoundError):
        swap_panel(tmp_path / "absent.parquet", live)
    assert live.read_bytes() == b"new"



def test_main_runs_rebuild_and_swaps_only_with_flag(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.backfill import krx_panel_rebuild as mod

    seen: list[dict] = []
    swaps: list[tuple] = []

    async def _fake_rebuild(**kwargs):
        seen.append(kwargs)
        return mod.RebuildReport(n_rows=1, n_symbols=1, n_dates=1, match_rate=1.0, n_common_rows=1)

    monkeypatch.setattr(mod, "run_panel_rebuild", _fake_rebuild)
    monkeypatch.setattr(mod, "swap_panel", lambda rebuilt, live: swaps.append((rebuilt, live)) or live)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "price_history.parquet", raising=False)

    # When
    mod.main(["--checkpoint-dir", str(tmp_path / "cp")])
    mod.main(["--checkpoint-dir", str(tmp_path / "cp"), "--start", "2020-01-02", "--end", "2020-01-31", "--swap"])

    # Then
    assert seen[0]["start"] == mod.REBUILD_START_DATE
    assert seen[0]["end"] == pd.Timestamp.today().normalize()
    assert seen[0]["out_path"] == tmp_path / "price_history_rebuild.parquet"
    assert seen[0]["old_path"] == tmp_path / "price_history.parquet"
    assert seen[0]["checkpoint_dir"] == tmp_path / "cp"
    assert seen[1]["start"] == pd.Timestamp("2020-01-02") and seen[1]["end"] == pd.Timestamp("2020-01-31")
    assert swaps == [(tmp_path / "price_history_rebuild.parquet", tmp_path / "price_history.parquet")]


def test_validate_rebuild_ignores_non_kospi_kosdaq_rows_for_shrink_check() -> None:
    """비-코스피/코스닥 표식 행(예: 조건검색 경유로 우연히 백필된 ETF)은 KRX 전종목 소스의
    스코프 밖이므로, 새 패널에서 사라져도 행수 감소로 취급하지 않는다 (2026-09-14 실측:
    243880 TIGER 200IT레버리지 market='ETF' 4개 날짜에서 fail-closed 오탐)."""
    import pandas as pd

    from src.backfill import krx_panel_rebuild as mod

    def _panel(rows):
        return pd.DataFrame(rows, columns=["date", "symbol", "close", "close_raw", "market"]).assign(
            date=lambda x: pd.to_datetime(x["date"])
        )

    old = _panel([
        ("2026-09-07", "005930", 100.0, 100.0, "KOSPI"),
        ("2026-09-07", "243880", 350000.0, 350000.0, "ETF"),
    ])
    new = _panel([
        ("2026-09-07", "005930", 100.0, 100.0, "KOSPI"),
    ])

    # When: ETF 행 하나가 새 패널에는 없지만(코스피/코스닥 스코프 밖)
    metrics = mod.validate_rebuild(new, old)

    # Then: 행수 감소로 실패하지 않고, 코스피/코스닥 행만으로 정상 검증
    assert metrics["n_rows"] == 1
    assert metrics["match_rate"] == 1.0


def test_compute_base_price_factors_flags_events_gaps_and_share_ratio() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.backfill.krx_panel_rebuild import compute_base_price_factors

    cols = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "trade_value_100m", "market_cap_100m", "market"]
    rows = [
        ("2026-09-07", "000002", 200, 200, 200, 200, 200, 10, 1.0, 2000.0, "KOSDAQ"),
        ("2026-09-08", "000002", 100, 100, 100, 100, 100, 10, 1.0, 2000.0, "KOSDAQ"),
        ("2026-09-09", "000002", 101, 101, 101, 101, 100, 10, 1.0, 2020.0, "KOSDAQ"),
        ("2026-09-07", "000003", 50, 50, 50, 50, 50, 10, 1.0, 500.0, "KOSPI"),
        ("2026-09-09", "000003", 80, 80, 80, 80, 70, 10, 1.0, 800.0, "KOSPI"),
    ]
    raw = pd.DataFrame([dict(zip(cols, r, strict=True)) for r in rows], columns=cols).assign(date=lambda x: pd.to_datetime(x["date"]))

    # When
    out = compute_base_price_factors(raw)

    # Then: 000002 09-08 기준가 100 / 직전종가 200 = 0.5 이벤트, 시총 불변 -> 주식수 2배
    a = out[out["symbol"] == "000002"].set_index(out.loc[out["symbol"] == "000002", "date"].dt.strftime("%m-%d"))
    assert bool(a.loc["09-08", "is_event"]) and a.loc["09-08", "factor"] == pytest.approx(0.5)
    assert a.loc["09-08", "share_ratio"] == pytest.approx(2.0)
    assert not bool(a.loc["09-09", "is_event"]) and a.loc["09-09", "factor"] == 1.0
    assert np.isnan(a.loc["09-07", "share_ratio"]) and pd.isna(a.loc["09-07", "prior_date"])
    assert a.loc["09-08", "prior_close"] == 200.0 and a.loc["09-08", "close_raw"] == 100.0
    # 000003: 09-08 결측(거래정지) 후 09-09 기준가 70 != 직전종가 50 -> 체인 미반영 갭 점프
    b = out[out["symbol"] == "000003"].set_index(out.loc[out["symbol"] == "000003", "date"].dt.strftime("%m-%d"))
    assert bool(b.loc["09-09", "is_gap_jump"]) and not bool(b.loc["09-09", "is_event"]) and b.loc["09-09", "factor"] == 1.0
    assert int(out["is_gap_jump"].sum()) == 1 and int(out["is_event"].sum()) == 1


def test_select_adjudication_events_finds_old_seams_and_controls() -> None:
    def _events(rows):
        import pandas as pd

        cols = ["date", "symbol", "kind", "new_factor", "old_factor", "share_ratio", "kis_factor"]
        return pd.DataFrame(rows, columns=cols).assign(date=lambda x: pd.to_datetime(x["date"]))

    import pandas as pd
    import pytest

    from src.backfill.krx_panel_rebuild import compute_base_price_factors, select_adjudication_events

    cols = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "trade_value_100m", "market_cap_100m", "market"]
    rows = [
        ("2026-09-07", "005930", 100, 100, 100, 100, 100, 10, 1.0, 1000.0, "KOSPI"),
        ("2026-09-08", "005930", 100, 100, 100, 100, 100, 10, 1.0, 1000.0, "KOSPI"),
        ("2026-09-09", "005930", 100, 100, 100, 100, 100, 10, 1.0, 1000.0, "KOSPI"),
        ("2026-09-07", "000002", 200, 200, 200, 200, 200, 10, 1.0, 2000.0, "KOSDAQ"),
        ("2026-09-08", "000002", 100, 100, 100, 100, 100, 10, 1.0, 2000.0, "KOSDAQ"),
        ("2026-09-09", "000002", 100, 100, 100, 100, 100, 10, 1.0, 2000.0, "KOSDAQ"),
        ("2026-09-07", "000004", 300, 300, 300, 300, 300, 10, 1.0, 3000.0, "KOSDAQ"),
        ("2026-09-08", "000004", 100, 100, 100, 100, 100, 10, 1.0, 3000.0, "KOSDAQ"),
        ("2026-09-09", "000004", 100, 100, 100, 100, 100, 10, 1.0, 3000.0, "KOSDAQ"),
    ]
    raw = pd.DataFrame([dict(zip(cols, r, strict=True)) for r in rows], columns=cols).assign(date=lambda x: pd.to_datetime(x["date"]))
    factors = compute_base_price_factors(raw)
    # 구 패널: 005930 은 09-07 에 가짜 x0.2 이음매, 000002 는 새 체인과 동일한 0.5 소급, 000004 는 0.3333 이벤트를 누락
    old = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-07", "2026-09-08", "2026-09-09"] * 3),
        "symbol": ["005930"] * 3 + ["000002"] * 3 + ["000004"] * 3,
        "close": [20.0, 100.0, 100.0, 100.0, 100.0, 100.0, 300.0, 100.0, 100.0],
    })

    # When
    events = select_adjudication_events(factors, old, n_controls=5)

    # Then
    assert list(events.columns) == ["date", "symbol", "kind", "new_factor", "old_factor", "share_ratio"]
    assert events["kind"].tolist() == ["disagree", "disagree", "control"]
    assert events["symbol"].tolist() == ["000004", "005930", "000002"]
    dis = events.set_index("symbol")
    assert dis.loc["005930", "new_factor"] == pytest.approx(1.0) and dis.loc["005930", "old_factor"] == pytest.approx(0.2)
    assert dis.loc["000004", "new_factor"] == pytest.approx(1 / 3) and dis.loc["000004", "old_factor"] == pytest.approx(1.0)
    assert dis.loc["000002", "new_factor"] == pytest.approx(0.5) and dis.loc["000002", "old_factor"] == pytest.approx(0.5)
    assert dis.loc["000002", "share_ratio"] == pytest.approx(2.0)
    # 대조군 상한 0 -> 대조군 없음
    assert select_adjudication_events(factors, old, n_controls=0)["kind"].tolist() == ["disagree", "disagree"]


def test_fetch_kis_event_factors_uses_adjusted_flag_and_checkpoints(tmp_path) -> None:
    import asyncio

    import numpy as np
    import pandas as pd
    import pytest

    from src.backfill.krx_panel_rebuild import fetch_kis_event_factors

    calls: list[tuple[str, str, str, str]] = []

    class _Kis:
        async def get_stock_ohlcv_history(self, session, stock_code, start_date, end_date, period_code="D", adj_price="0", market_div_code=None):
            calls.append((stock_code, start_date, end_date, adj_price))
            if stock_code == "999999":
                return {"rt_cd": "7", "output2": []}
            # fid_org_adj_prc="0" = 수정주가, "1" = 원주가; 09-09 에 1:2 분할
            closes = {"20260908": "100", "20260909": "100"} if adj_price == "0" else {"20260908": "200", "20260909": "100"}
            return {"rt_cd": "0", "output2": [{"stck_bsop_date": d, "stck_clpr": c} for d, c in sorted(closes.items(), reverse=True)]}

    class _Boom:
        async def get_stock_ohlcv_history(self, *a, **k):
            raise AssertionError("checkpointed events must not be re-queried")

    events = pd.DataFrame({"date": pd.to_datetime(["2026-09-09", "2026-09-09"]), "symbol": ["000002", "999999"], "kind": ["control", "disagree"],
                           "new_factor": [0.5, 2.0], "old_factor": [0.5, 1.0], "share_ratio": [2.0, 1.0]})
    cp = tmp_path / "referee.parquet"

    # When
    out = asyncio.run(fetch_kis_event_factors(_Kis(), None, events, cp))
    again = asyncio.run(fetch_kis_event_factors(_Boom(), None, events, cp))
    empty = asyncio.run(fetch_kis_event_factors(_Boom(), None, events.iloc[0:0], tmp_path / "never.parquet"))

    # Then
    assert out["kis_factor"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(out["kis_factor"].iloc[1])
    assert ("000002", "20260828", "20260912", "0") in calls and ("000002", "20260828", "20260912", "1") in calls
    assert len(calls) == 4 and cp.exists()
    assert again["kis_factor"].iloc[0] == pytest.approx(0.5) and np.isnan(again["kis_factor"].iloc[1])
    assert list(again.columns) == ["date", "symbol", "kind", "new_factor", "old_factor", "share_ratio", "kis_factor"]
    assert empty.empty and "kis_factor" in empty.columns and not (tmp_path / "never.parquet").exists()


def test_adjudicate_events_accepts_confirmed_corroborated_and_unverified() -> None:
    def _events(rows):
        import pandas as pd

        cols = ["date", "symbol", "kind", "new_factor", "old_factor", "share_ratio", "kis_factor"]
        return pd.DataFrame(rows, columns=cols).assign(date=lambda x: pd.to_datetime(x["date"]))

    import math

    from src.backfill import krx_panel_rebuild as mod

    nan = float("nan")
    events = _events([
        ("2026-01-02", "000002", "control", 0.5, 0.5, 2.0, 0.5),
        ("2025-09-01", "032860", "disagree", 1.0, 0.2, 1.0, 1.0),
        ("2026-05-11", "009310", "disagree", 5.0, 1.0, 0.2, 1.0),
        ("2023-03-30", "033790", "disagree", 2.0, 1.0, 1.0, nan),
    ])

    # When
    metrics = mod.adjudicate_events(events, n_gap_jumps=1)
    empty = mod.adjudicate_events(events.iloc[0:0], n_gap_jumps=0)

    # Then
    assert metrics == {"n_disagreements": 3, "n_kis_confirmed": 1, "n_share_corroborated": 1, "n_unverified": 1,
                       "n_controls": 1, "control_agree_rate": 1.0, "n_gap_jumps": 1}
    assert empty["n_disagreements"] == 0 and empty["n_controls"] == 0 and math.isnan(empty["control_agree_rate"])


def test_adjudicate_events_fails_closed() -> None:
    def _events(rows):
        import pandas as pd

        cols = ["date", "symbol", "kind", "new_factor", "old_factor", "share_ratio", "kis_factor"]
        return pd.DataFrame(rows, columns=cols).assign(date=lambda x: pd.to_datetime(x["date"]))

    import pytest

    from src.backfill import krx_panel_rebuild as mod

    nan = float("nan")
    control = ("2026-01-02", "000002", "control", 0.5, 0.5, 2.0, 0.5)

    # When/Then: KIS 가 구 패널 편(0.2)을 들고 주식수도 불변 -> 새 배율 기각
    with pytest.raises(ValueError, match="contradicted"):
        mod.adjudicate_events(_events([control, ("2025-09-01", "032860", "disagree", 1.0, 0.2, 1.0, 0.2)]), n_gap_jumps=0)
    # 조회 불가 + 주식수 미뒷받침 불일치가 상한 초과
    unverified = [("2023-03-30", f"{i:06d}", "disagree", 2.0, 1.0, 1.0, nan) for i in range(mod.REBUILD_MAX_UNVERIFIED + 1)]
    with pytest.raises(ValueError, match="unverified"):
        mod.adjudicate_events(_events([control, *unverified]), n_gap_jumps=0)
    # 플래그 반전(대조군 KIS 배율이 역수) -> 심판 자체를 신뢰할 수 없음
    inverted = ("2026-01-02", "000002", "control", 0.5, 0.5, 2.0, 2.0)
    with pytest.raises(ValueError, match="control"):
        mod.adjudicate_events(_events([inverted, ("2025-09-01", "032860", "disagree", 1.0, 0.2, 1.0, 1.0)]), n_gap_jumps=0)
    # 불일치가 있는데 유효 대조군이 없음
    with pytest.raises(ValueError, match="control"):
        mod.adjudicate_events(_events([("2025-09-01", "032860", "disagree", 1.0, 0.2, 1.0, 1.0)]), n_gap_jumps=0)
    # 거래정지 갭 기준가 점프 상한 초과
    with pytest.raises(ValueError, match="gap"):
        mod.adjudicate_events(_events([control]), n_gap_jumps=mod.REBUILD_MAX_GAP_JUMPS + 1)


def test_run_panel_rebuild_adjudicates_old_seam_with_kis_referee(tmp_path, monkeypatch) -> None:
    def _seam_fixture(tmp_path, monkeypatch, kis_split_adjusted: bool):
        import pandas as pd

        from src.backfill import krx_panel_rebuild as mod

        days = [pd.Timestamp(d) for d in ("2026-09-08", "2026-09-09", "2026-09-10")]
        cal = pd.bdate_range("2015-11-02", "2026-09-10")

        class _Kis:
            def __init__(self):
                self.ohlcv_calls = 0

            async def ensure_token(self, session):
                return "t"

            async def get_market_index_history(self, session, code, start, end):
                s, e = pd.Timestamp(start), pd.Timestamp(end)
                base = 2000.0 if code == "0001" else 800.0
                sel = [d for d in cal if s <= d <= e][-50:]
                return {"rt_cd": "0", "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "bstp_nmix_prpr": f"{base * (1 + 0.0005 * cal.get_loc(d)):.2f}"} for d in reversed(sel)]}

            async def get_stock_ohlcv_history(self, session, stock_code, start_date, end_date, period_code="D", adj_price="0", market_div_code=None):
                self.ohlcv_calls += 1
                if stock_code == "000002":
                    raw_c = ["200", "100", "100"]
                    closes = ["100", "100", "100"] if (adj_price == "0" and kis_split_adjusted) else raw_c
                else:
                    closes = ["100", "100", "100"]
                return {"rt_cd": "0", "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "stck_clpr": c} for d, c in reversed(list(zip(days, closes, strict=True)))]}

        def _fetch(date, cfg):
            d = pd.Timestamp(date)
            split = d >= days[1]
            return pd.DataFrame([
                {"date": d, "symbol": "005930", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "prev_close": 100.0, "volume": 10.0, "trade_value_100m": 1.0, "market_cap_100m": 1000.0, "market": "KOSPI"},
                {"date": d, "symbol": "000002", "open": 100.0 if split else 200.0, "high": 100.0 if split else 200.0, "low": 100.0 if split else 200.0, "close": 100.0 if split else 200.0,
                 "prev_close": 100.0 if split else 200.0, "volume": 10.0, "trade_value_100m": 1.0, "market_cap_100m": 2000.0, "market": "KOSDAQ"},
            ])

        flows = {"inst_netbuy": 7.0, "foreign_netbuy": 8.0, "program_netbuy": 9.0}
        old_rows = [{"date": d, "symbol": "005930", "close": 20.0 if i == 0 else 100.0, "market": "KOSPI", **flows} for i, d in enumerate(days)]
        old_rows += [{"date": d, "symbol": "000002", "close": 100.0, "market": "KOSDAQ", **flows} for d in days]
        old_path = tmp_path / "old.parquet"
        pd.DataFrame(old_rows).to_parquet(old_path, index=False)
        monkeypatch.setattr(mod, "fetch_krx_daily", _fetch)
        return mod, days, old_path, _Kis()

    import asyncio

    import pandas as pd
    import pytest

    mod, days, old_path, kis = _seam_fixture(tmp_path, monkeypatch, kis_split_adjusted=True)
    out_path = tmp_path / "rebuild.parquet"

    # When: 구 패널 005930 의 가짜 x0.2 이음매 -> KIS 가 새 체인(무이벤트) 확인, 000002 분할은 대조군
    report = asyncio.run(mod.run_panel_rebuild(start=days[0], end=days[-1], checkpoint_dir=tmp_path / "cp", out_path=out_path, old_path=old_path, krx_cfg=object(), kis=kis))

    # Then
    assert out_path.exists()
    assert (report.n_disagreements, report.n_kis_confirmed, report.n_share_corroborated, report.n_unverified) == (1, 1, 0, 0)
    assert report.match_rate == pytest.approx(5 / 6)
    assert (tmp_path / "cp" / mod.REBUILD_REFEREE_CHECKPOINT_NAME).exists() and kis.ohlcv_calls == 4
    written = pd.read_parquet(out_path)
    written["symbol"] = written["symbol"].astype(str)
    s = written[written["symbol"] == "000002"].sort_values("date")
    assert s["close"].astype(float).tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert s["close_raw"].astype(float).tolist() == pytest.approx([200.0, 100.0, 100.0])


def test_run_panel_rebuild_does_not_write_when_referee_control_fails(tmp_path, monkeypatch) -> None:
    def _seam_fixture(tmp_path, monkeypatch, kis_split_adjusted: bool):
        import pandas as pd

        from src.backfill import krx_panel_rebuild as mod

        days = [pd.Timestamp(d) for d in ("2026-09-08", "2026-09-09", "2026-09-10")]
        cal = pd.bdate_range("2015-11-02", "2026-09-10")

        class _Kis:
            def __init__(self):
                self.ohlcv_calls = 0

            async def ensure_token(self, session):
                return "t"

            async def get_market_index_history(self, session, code, start, end):
                s, e = pd.Timestamp(start), pd.Timestamp(end)
                base = 2000.0 if code == "0001" else 800.0
                sel = [d for d in cal if s <= d <= e][-50:]
                return {"rt_cd": "0", "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "bstp_nmix_prpr": f"{base * (1 + 0.0005 * cal.get_loc(d)):.2f}"} for d in reversed(sel)]}

            async def get_stock_ohlcv_history(self, session, stock_code, start_date, end_date, period_code="D", adj_price="0", market_div_code=None):
                self.ohlcv_calls += 1
                if stock_code == "000002":
                    raw_c = ["200", "100", "100"]
                    closes = ["100", "100", "100"] if (adj_price == "0" and kis_split_adjusted) else raw_c
                else:
                    closes = ["100", "100", "100"]
                return {"rt_cd": "0", "output2": [{"stck_bsop_date": d.strftime("%Y%m%d"), "stck_clpr": c} for d, c in reversed(list(zip(days, closes, strict=True)))]}

        def _fetch(date, cfg):
            d = pd.Timestamp(date)
            split = d >= days[1]
            return pd.DataFrame([
                {"date": d, "symbol": "005930", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "prev_close": 100.0, "volume": 10.0, "trade_value_100m": 1.0, "market_cap_100m": 1000.0, "market": "KOSPI"},
                {"date": d, "symbol": "000002", "open": 100.0 if split else 200.0, "high": 100.0 if split else 200.0, "low": 100.0 if split else 200.0, "close": 100.0 if split else 200.0,
                 "prev_close": 100.0 if split else 200.0, "volume": 10.0, "trade_value_100m": 1.0, "market_cap_100m": 2000.0, "market": "KOSDAQ"},
            ])

        flows = {"inst_netbuy": 7.0, "foreign_netbuy": 8.0, "program_netbuy": 9.0}
        old_rows = [{"date": d, "symbol": "005930", "close": 20.0 if i == 0 else 100.0, "market": "KOSPI", **flows} for i, d in enumerate(days)]
        old_rows += [{"date": d, "symbol": "000002", "close": 100.0, "market": "KOSDAQ", **flows} for d in days]
        old_path = tmp_path / "old.parquet"
        pd.DataFrame(old_rows).to_parquet(old_path, index=False)
        monkeypatch.setattr(mod, "fetch_krx_daily", _fetch)
        return mod, days, old_path, _Kis()

    import asyncio

    import pytest

    mod, days, old_path, kis = _seam_fixture(tmp_path, monkeypatch, kis_split_adjusted=False)
    out_path = tmp_path / "rebuild.parquet"

    # When/Then: KIS 가 분할을 반영하지 않은 값을 주면(플래그 반전 등) 대조군 검증 실패 -> 기록 없음
    with pytest.raises(ValueError, match="control"):
        asyncio.run(mod.run_panel_rebuild(start=days[0], end=days[-1], checkpoint_dir=tmp_path / "cp", out_path=out_path, old_path=old_path, krx_cfg=object(), kis=kis))
    assert not out_path.exists()
