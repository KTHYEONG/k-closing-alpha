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



def test_validate_rebuild_rejects_duplicates_shrink_and_mismatch() -> None:
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
    assert mod.REBUILD_MIN_MATCH_RATE == 0.98
    assert metrics == {"n_rows": 5, "n_symbols": 3, "n_common_rows": 3, "match_rate": 1.0, "n_dates": 2}

    with pytest.raises(ValueError, match="duplicate"):
        mod.validate_rebuild(pd.concat([new, new.iloc[[0]]], ignore_index=True), old)
    with pytest.raises(ValueError, match="fewer rows"):
        mod.validate_rebuild(new.iloc[[0]], old)
    drift = new.copy()
    drift.loc[drift["symbol"] == "005930", "close"] = 999.0
    with pytest.raises(ValueError, match="match rate"):
        mod.validate_rebuild(drift, old)
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

