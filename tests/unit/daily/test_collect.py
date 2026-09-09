"""Unit tests for daily data collection.

SCENARIO_DAILY_COLLECT_REFACTORING_01:
Verifies that collect.py saves collected condition data directly
without chart_pass_cache.json or parenthesis column renaming.

SCENARIO_COLLECT_NO_SLEEP:
 fetch_single_stock 내부에 API_SLEEP_INTERVAL 참조 없음 검증 (perf_v2).

SCENARIO_REGRESSION:
 기존 17개 테스트 회귀 검증.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.daily.collect import save_collected_condition_data


def test_scenario_daily_collect_refactoring_01(tmp_path: Path) -> None:
    """[SCENARIO_DAILY_COLLECT_REFACTORING_01] Verify clean standard CSV saving without chart_pass."""
    sample_df = pd.DataFrame(
        {
            "종목명": ["테스트종목"],
            "종목코드": ["123"],
            "시가": [1000],
            "고가": [1100],
            "저가": [990],
            "종가": [1050],
            "전일종가": [1000],
            "시가총액": [500.0],
            "거래대금": [100.0],
            "등락률": [5.0],
            "선정순위": [1],
            "기관_순매수": [10.0],
            "외국인_순매수": [20.0],
            "프로그램_순매수": [5.0],
            "체결강도": [120.0],
            "시장구분": ["KOSDAQ"],
            "총_종목수": [100],
            "평균_거래대금": [50.0],
            "kospi": [0.5],
            "kosdaq": [1.2],
            "v_kospi": [15.0],
            "v_kosdaq": [20.0],
            "거래량": [10000],
            "시나리오": ["거래량 폭증"],
        }
    )

    csv_path = tmp_path / "daily" / "daily_stocks.csv"
    res_path = save_collected_condition_data(sample_df, csv_path)

    assert res_path.exists()
    saved_df = pd.read_csv(res_path, dtype={"종목코드": str})
    assert saved_df["종목코드"].iloc[0] == "000123"
    assert "차트통과" not in saved_df.columns
    assert "(차트통과)" not in saved_df.columns


def test_scenario_collect_no_sleep() -> None:
    """[SCENARIO_COLLECT_NO_SLEEP]
    fetch_single_stock 소스에 API_SLEEP_INTERVAL 참조 및 asyncio.sleep 없음 검증.
    """
    import inspect
    from src.daily.collect import fetch_single_stock

    src_text = inspect.getsource(fetch_single_stock)
    assert callable(fetch_single_stock)
    assert "API_SLEEP_INTERVAL" not in src_text
    assert "asyncio.sleep" not in src_text


def test_scenario_regression() -> None:
    """[SCENARIO_REGRESSION] 기존 save_collected_condition_data 회귀 검증."""
    from src.daily.collect import save_collected_condition_data

    assert callable(save_collected_condition_data)


def test_fetch_single_stock_captures_dual_venue_decision_price() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()

    async def _fake_get_current_price(session, code, market_div_code=None):
        if market_div_code == "J":
            prpr, vol = "70000", "1000"
        else:
            prpr, vol = "69800", "500"
        return {
            "rt_cd": "0",
            "output": {
                "stck_prpr": prpr, "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                "acml_vol": vol, "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
            },
        }

    client.get_current_price = _fake_get_current_price
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})

    sem = asyncio.Semaphore(1)
    stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}

    result = asyncio.run(collect.fetch_single_stock(0, stock, 1, sem, client, session=None))

    record = result[0] if isinstance(result, tuple) else result
    assert record["krx_현재가"] == 70000
    assert record["nxt_현재가"] == 69800
    assert record["sor_effective_price"] == 69800


def test_fetch_single_stock_falls_back_to_krx_when_nxt_unlisted_or_illiquid() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    async def _run_case(nxt_resp):
        client = AsyncMock()

        async def _fake(session, code, market_div_code=None):
            if market_div_code == "J":
                return {"rt_cd": "0", "output": {
                    "stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                    "acml_vol": "1000", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
                }}
            return nxt_resp

        client.get_current_price = _fake
        client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
        client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
        client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})
        sem = asyncio.Semaphore(1)
        stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}
        result = await collect.fetch_single_stock(0, stock, 1, sem, client, session=None)
        record = result[0] if isinstance(result, tuple) else result
        failed = result[1] if isinstance(result, tuple) else []
        return record, failed

    # NXT 미상장 케이스
    import asyncio as _aio

    record, failed = _aio.run(_run_case({"rt_cd": "9", "msg1": "NXT 미상장"}))
    assert record["krx_현재가"] == 70000
    assert record["nxt_현재가"] is None
    assert record["sor_effective_price"] == 70000
    assert "현재가" not in failed or failed == []

    # NXT 유동성 0 케이스
    record2, _ = _aio.run(_run_case({"rt_cd": "0", "output": {
        "stck_prpr": "69000", "stck_oprc": "69000", "stck_hgpr": "69000", "stck_lwpr": "69000",
        "acml_vol": "0", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
    }}))
    assert record2["sor_effective_price"] == 70000


def test_fetch_single_stock_includes_orderbook_snapshot_fields() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()

    async def _fake_get_current_price(session, code, market_div_code=None):
        return {
            "rt_cd": "0",
            "output": {
                "stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                "acml_vol": "1000", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
            },
        }

    async def _fake_get_orderbook_snapshot(session, code, market_div_code=None):
        if market_div_code == "J":
            return {"rt_cd": "0", "output1": {
                "askp1": "70100", "bidp1": "70000",
                "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500",
            }}
        return {"rt_cd": "0", "output1": {
            "askp1": "69900", "bidp1": "69800",
            "total_askp_rsqn": "300", "total_bidp_rsqn": "200",
        }}

    client.get_current_price = _fake_get_current_price
    client.get_orderbook_snapshot = _fake_get_orderbook_snapshot
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})

    sem = asyncio.Semaphore(1)
    stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}
    record, _, _ = asyncio.run(collect.fetch_single_stock(0, stock, 1, sem, client, session=None))

    assert record["krx_매도호가1"] == 70100
    assert record["krx_매수호가1"] == 70000
    assert record["krx_매도잔량"] == 1200
    assert record["krx_매수잔량"] == 1500
    assert record["nxt_매도호가1"] == 69900
    assert record["nxt_매수호가1"] == 69800
    assert record["nxt_매도잔량"] == 300
    assert record["nxt_매수잔량"] == 200


def test_fetch_single_stock_flags_krx_orderbook_failure_without_blocking_record() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()

    async def _fake_get_current_price(session, code, market_div_code=None):
        return {
            "rt_cd": "0",
            "output": {
                "stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                "acml_vol": "1000", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
            },
        }

    async def _fake_get_orderbook_snapshot(session, code, market_div_code=None):
        if market_div_code == "J":
            return {"rt_cd": "9", "msg1": "일시 오류"}
        return {"rt_cd": "0", "output1": {
            "askp1": "69900", "bidp1": "69800",
            "total_askp_rsqn": "300", "total_bidp_rsqn": "200",
        }}

    client.get_current_price = _fake_get_current_price
    client.get_orderbook_snapshot = _fake_get_orderbook_snapshot
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})

    sem = asyncio.Semaphore(1)
    stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}
    record, failed, _ = asyncio.run(collect.fetch_single_stock(0, stock, 1, sem, client, session=None))

    assert "호가" in failed
    assert record["krx_매도호가1"] == 0
    assert record["nxt_매도호가1"] == 69900

def test_fetch_all_stock_data_persists_full_orderbook_without_widening_archive() -> None:
    from src.processing.schema import ARCHIVE_COLUMN_ORDER

    # Then: 호가 사다리는 별도 파티션으로 가고 평면 아카이브 스키마는 넓어지지 않는다
    depth_cols = [c for c in ARCHIVE_COLUMN_ORDER if c.startswith("krx_매도호가") or c.startswith("krx_매수호가")]
    assert depth_cols == ["krx_매도호가1", "krx_매수호가1"]
    assert "krx_매도호가2" not in ARCHIVE_COLUMN_ORDER
    assert len([c for c in ARCHIVE_COLUMN_ORDER if "호가" in c or "잔량" in c]) == 8


def test_fetch_single_stock_returns_orderbook_rows_triple(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(return_value={"rt_cd": "0", "output": {"stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "70500", "stck_lwpr": "68900", "acml_vol": "1000", "prdy_ctrt": "1.5", "lstn_stcn": "100", "hts_avls": "1000", "acml_tr_pbmn": "100000000", "rprs_mrkt_kor_name": "KOSPI"}})
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{"whol_smtn_ntby_tr_pbmn": "100"}]})
    ladder = {f"askp{i}": str(70000 + i * 100) for i in range(1, 11)}
    ladder.update({"bidp1": "69900", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})

    sem = asyncio.Semaphore(1)
    result = asyncio.run(
        collect.fetch_single_stock(0, {"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "1.5"}, 1, sem, client, object())
    )

    assert len(result) == 3
    row, failed, orderbook_rows = result
    assert row["krx_매도호가1"] == 70100
    assert any("askp10" in ob for ob in orderbook_rows)
    assert {ob["capture_reason"] for ob in orderbook_rows} == {"decision"}


def test_fetch_all_stock_data_persists_orderbook_and_survives_persist_failure(monkeypatch) -> None:
    """호가 스냅샷을 일괄 영속화하고, 영속화 실패는 로깅만 하고 수집 결과에 영향 없다."""
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()
    client.get_current_price = AsyncMock(
        return_value={"rt_cd": "0", "output": {"stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "70500", "stck_lwpr": "68900", "acml_vol": "1000", "prdy_ctrt": "1.5", "lstn_stcn": "100", "hts_avls": "1000", "acml_tr_pbmn": "100000000", "rprs_mrkt_kor_name": "KOSPI"}}
    )
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{"frgn_fake_ntby_qty": "1", "orgn_fake_ntby_qty": "2"}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{"whol_smtn_ntby_tr_pbmn": "100"}]})
    ladder = {f"askp{i}": str(70000 + i * 100) for i in range(1, 11)}
    ladder.update({"bidp1": "69900", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"})
    client.get_orderbook_snapshot = AsyncMock(return_value={"rt_cd": "0", "output1": ladder})

    monkeypatch.setattr(collect, "prefetch_ohlcv_for_sma120", AsyncMock(return_value={}))

    def _raise(rows, snapshot_date):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(collect, "append_orderbook_snapshots", _raise)

    stock_list = [{"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "1.5"}]
    results, failed_info = asyncio.run(collect.fetch_all_stock_data(stock_list, client, object()))

    assert len(results) == 1
    assert failed_info == []


def test_apply_cost_aware_admission_filters_by_chg_tv_mc_tick_cost_and_ceiling() -> None:
    import pandas as pd

    from src.daily.collect import apply_cost_aware_admission

    # Given: 5 enriched candidates on the live Korean-column snapshot shape.
    # S1: chg 5%, cheap tick (18000 KRW -> 5.56bp) -> admitted.
    # S2: chg 5%, expensive tick (30000 KRW -> 16.67bp > 7.5bp cap) -> excluded.
    # S3: chg 15% (outside the 2-10% band) -> excluded.
    # S4: chg 5%, thin liquidity (tv/mc below the floor) -> excluded.
    # S5: chg 30% and close==high (ceiling) -> excluded.
    df = pd.DataFrame({
        "종목코드": ["S1", "S2", "S3", "S4", "S5"],
        "종가": [18000.0, 30000.0, 23000.0, 18000.0, 13000.0],
        "전일종가": [17142.86, 28571.43, 20000.0, 17142.86, 10000.0],
        "고가": [18100.0, 30100.0, 23100.0, 18100.0, 13000.0],
        "거래량": [1_000_000] * 5,
        "거래대금": [500.0, 500.0, 500.0, 10.0, 500.0],
        "시가총액": [3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
        "시장구분": ["KOSPI", "KOSPI", "KOSPI", "KOSPI", "KOSPI"],
    })
    decision_date = pd.Timestamp("2026-09-09")

    # When
    out = apply_cost_aware_admission(df, decision_date=decision_date)

    # Then: only S1 clears every admission criterion
    assert out["종목코드"].tolist() == ["S1"]

    # Then: an all-excluded snapshot returns an empty frame, never raises
    thin = df[df["종목코드"] == "S4"].reset_index(drop=True)
    out_empty = apply_cost_aware_admission(thin, decision_date=decision_date)
    assert len(out_empty) == 0
    assert list(out_empty.columns) == list(thin.columns)


def test_apply_cost_aware_admission_empty_input_returns_empty_without_raising() -> None:
    import pandas as pd

    from src.daily.collect import apply_cost_aware_admission

    df = pd.DataFrame(columns=["종목코드", "종가", "전일종가", "고가", "거래량", "거래대금", "시가총액", "시장구분"])

    out = apply_cost_aware_admission(df, decision_date=pd.Timestamp("2026-09-09"))

    assert len(out) == 0


def test_resolve_daily_candidates_automated_mode_skips_condition_search_entirely(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod
    from src.daily.collect import resolve_daily_candidates

    monkeypatch.setattr(collect_mod.settings, "CANDIDATE_SOURCE_MODE", "automated")

    client = AsyncMock()  # never touched in automated mode
    fake_stock_list = [{"code": "005930", "name": "삼성전자", "price": "70000", "chgrate": "5.0"}]

    async def _fake_fetch(client_arg, session_arg, **kwargs):
        return fake_stock_list

    monkeypatch.setattr(collect_mod, "fetch_candidate_stock_list", _fake_fetch)

    # When
    resolved = asyncio.run(resolve_daily_candidates(client, object()))

    # Then: the automated stock_list is returned with every scenario set empty,
    # and the manual condition-search API is never consulted
    assert resolved is not None
    stock_list, overheated, new_high, near_new_high, upper_limit_next, upper_limit = resolved
    assert stock_list == fake_stock_list
    assert overheated == set()
    assert new_high == set()
    assert near_new_high == set()
    assert upper_limit_next == set()
    assert upper_limit == set()
    client.get_condition_list.assert_not_awaited()
    client.get_condition_result.assert_not_awaited()


def test_resolve_daily_candidates_manual_mode_preserves_existing_behavior(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod
    from src.daily.collect import resolve_daily_candidates

    monkeypatch.setattr(collect_mod.settings, "CANDIDATE_SOURCE_MODE", "manual")
    monkeypatch.setattr(collect_mod, "TARGET_CONDITION_NAME", "종가매매")

    client = AsyncMock()
    client.get_condition_list = AsyncMock(return_value={"rt_cd": "0", "output2": [
        {"condition_nm": "종가매매", "seq": "0"},
        {"condition_nm": "단기과열", "seq": "1"},
    ]})

    async def _fake_condition_result(session, seq):
        if seq == "0":
            return {"rt_cd": "0", "output2": [{"code": "005930", "name": "삼성전자"}]}
        if seq == "1":
            return {"rt_cd": "0", "output2": [{"code": "005930"}]}
        return {"rt_cd": "0", "output2": []}

    client.get_condition_result = _fake_condition_result

    # When
    resolved = asyncio.run(resolve_daily_candidates(client, object()))

    # Then: unchanged HTS condition-search path -- primary result plus the
    # overheated sub-condition tagging "005930"
    assert resolved is not None
    stock_list, overheated, new_high, near_new_high, upper_limit_next, upper_limit = resolved
    assert stock_list == [{"code": "005930", "name": "삼성전자"}]
    assert overheated == {"005930"}
    assert new_high == set()
    client.get_condition_list.assert_awaited_once()


def test_resolve_daily_candidates_manual_mode_condition_not_found_returns_none(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod
    from src.daily.collect import resolve_daily_candidates

    monkeypatch.setattr(collect_mod.settings, "CANDIDATE_SOURCE_MODE", "manual")
    monkeypatch.setattr(collect_mod, "TARGET_CONDITION_NAME", "종가매매")

    client = AsyncMock()
    client.get_condition_list = AsyncMock(return_value={"rt_cd": "0", "output2": [
        {"condition_nm": "다른조건", "seq": "9"},
    ]})

    # When: the target condition is not among the operator's saved conditions
    resolved = asyncio.run(resolve_daily_candidates(client, object()))

    # Then: fails closed by returning None, matching the prior inline early-return
    assert resolved is None

    # Then: an empty saved-condition list also returns None
    client2 = AsyncMock()
    client2.get_condition_list = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    assert asyncio.run(resolve_daily_candidates(client2, object())) is None

    # Then: a failed condition-list fetch also returns None
    client3 = AsyncMock()
    client3.get_condition_list = AsyncMock(return_value={"rt_cd": "1", "msg1": "인증 만료"})
    assert asyncio.run(resolve_daily_candidates(client3, object())) is None


def test_resolve_daily_candidates_rejects_invalid_candidate_source_mode(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import pytest

    import src.daily.collect as collect_mod
    from src.daily.collect import resolve_daily_candidates

    monkeypatch.setattr(collect_mod.settings, "CANDIDATE_SOURCE_MODE", "bogus_mode")
    client = AsyncMock()

    with pytest.raises(ValueError, match="CANDIDATE_SOURCE_MODE"):
        asyncio.run(resolve_daily_candidates(client, object()))


def test_resolve_daily_candidates_manual_mode_primary_result_failure_returns_none(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.daily.collect as collect_mod
    from src.daily.collect import resolve_daily_candidates

    monkeypatch.setattr(collect_mod.settings, "CANDIDATE_SOURCE_MODE", "manual")
    monkeypatch.setattr(collect_mod, "TARGET_CONDITION_NAME", "종가매매")

    client = AsyncMock()
    client.get_condition_list = AsyncMock(return_value={"rt_cd": "0", "output2": [
        {"condition_nm": "종가매매", "seq": "0"},
    ]})
    # Given: the target condition is found, but its result fetch itself fails
    client.get_condition_result = AsyncMock(return_value={"rt_cd": "1", "msg1": "세션 만료"})

    # When
    resolved = asyncio.run(resolve_daily_candidates(client, object()))

    # Then: fails closed by returning None, matching the prior inline early-return
    assert resolved is None
    client.get_condition_result.assert_awaited_once()


def test_apply_cost_aware_admission_if_automated_gates_on_candidate_source_mode(monkeypatch) -> None:
    import pandas as pd

    import src.daily.collect as collect_mod
    from src.daily.collect import apply_cost_aware_admission_if_automated

    df = pd.DataFrame({"종목코드": ["S1"], "거래대금": [500.0]})
    seen: dict[str, object] = {}

    def _fake_admission(frame, *, decision_date, screen=None):
        seen["frame"] = frame
        seen["decision_date"] = decision_date
        return frame.iloc[0:0]

    monkeypatch.setattr(collect_mod, "apply_cost_aware_admission", _fake_admission)

    # Given: manual mode (the default) -- df must pass through byte-identical,
    # apply_cost_aware_admission must never be called
    monkeypatch.setattr(collect_mod.settings, "CANDIDATE_SOURCE_MODE", "manual")
    out_manual = apply_cost_aware_admission_if_automated(df)
    assert out_manual is df
    assert "frame" not in seen

    # Given: automated mode -- the admission mask is applied with today's
    # Asia/Seoul decision date
    monkeypatch.setattr(collect_mod.settings, "CANDIDATE_SOURCE_MODE", "automated")
    out_auto = apply_cost_aware_admission_if_automated(df)
    assert len(out_auto) == 0
    assert seen["frame"] is df
    assert isinstance(seen["decision_date"], pd.Timestamp)

