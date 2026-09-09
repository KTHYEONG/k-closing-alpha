import asyncio
# ruff: noqa: I001 - contract mandates contiguous wiring import block after Colors
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd

from src import settings

# 커스텀 모듈 임포트
from src.api.kis_client import KisApiClient
from src.data.orderbook_store import append_orderbook_snapshots, build_orderbook_rows
from src.utils.display import Colors
from src.daily import archive
from src.daily.universe_scan import fetch_candidate_stock_list
from src.execution.cost_model import tick_cost_bp
from src.strategy.contract import COST_AWARE_UNIVERSE, UniverseSpec, derive_chg_ratio, mark_ceiling, select_universe

logger = logging.getLogger(__name__)

# =========================================================
# [설정] API 접속 정보
# =========================================================
APP_KEY = settings.KIS_API_CONFIG["app_key"]
APP_SECRET = settings.KIS_API_CONFIG["app_secret"]
ACCOUNT_ID = settings.KIS_API_CONFIG.get("account_id", "")
HTS_ID = settings.KIS_API_CONFIG.get("hts_id")

TARGET_CONDITION_NAME = settings.TARGET_CONDITION_NAME
TOKEN_FILE = str(settings.TOKEN_FILE)

logger.debug("일일 수집 시작...")


def _validate_hts_id() -> None:
    if not HTS_ID or "여기에" in HTS_ID:
        raise RuntimeError(
            ".env 파일의 'KIS_HTS_ID'에 본인의 HTS ID를 입력해주세요!"
        )


def safe_float(value, default=0.0):
    """문자열이나 None 값을 안전하게 float로 변환"""
    if value is None:
        return default
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------
# 헬퍼 함수: 시장 지수 등락률 파싱
# ---------------------------------------------------------
def parse_market_index_rate(data):
    if not data or data.get("rt_cd") != "0":
        return 0.0
    out1 = data.get("output1")
    if not out1:
        return 0.0
    rate_str = out1.get("bstp_nmix_prdy_ctrt") or out1.get("prdy_ctrt")
    try:
        if rate_str and float(rate_str) != 0.0:
            return float(rate_str)
        current_price = float(out1.get("bstp_nmix_prpr", "0"))
        change_amount = float(out1.get("bstp_nmix_prdy_vrss", "0"))
        prev_close = current_price - change_amount
        if prev_close != 0:
            return round((change_amount / prev_close) * 100, 2)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------
# 표준 CSV 저장 (utf-8-sig) & Parquet
# ---------------------------------------------------------


def flag_cost_aware_admission(
    df: pd.DataFrame, *, decision_date: pd.Timestamp, screen: UniverseSpec = COST_AWARE_UNIVERSE
) -> pd.DataFrame:
    """Flag every row of the daily snapshot with the COST_AWARE_UNIVERSE verdict.

    Args:
        df: Enriched Korean-column snapshot frame.
        decision_date: Decision date for point-in-time tick costing.
        screen: Universe admission spec.

    Returns:
        A copy of the input with a bool ``admitted`` column; no rows dropped.
        apply_cost_aware_admission wraps this helper and drops non-admitted rows.
    """
    import numpy as np

    if len(df) == 0:
        out = df.copy()
        out["admitted"] = np.zeros(0, dtype=bool)
        logger.info("[DATA] stage=cost_aware_admission n_raw=0 n_admitted=0 n_ceiling_excluded=0")
        return out
    close = pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = pd.to_numeric(df["전일종가"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["고가"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(df["거래량"], errors="coerce").to_numpy(dtype=np.float64)
    tv_clean = pd.to_numeric(df["거래대금"], errors="coerce").to_numpy(dtype=np.float64)
    mc_clean = pd.to_numeric(df["시가총액"], errors="coerce").to_numpy(dtype=np.float64)
    chg_ratio = derive_chg_ratio(close, prev_close)
    is_ceiling = mark_ceiling(pd.DataFrame({"chg_ratio": chg_ratio, "close": close, "high": high}))
    dates = np.full(len(df), np.datetime64(decision_date.strftime("%Y-%m-%d")))
    market = df["시장구분"].astype(str).to_numpy(dtype=object)
    tick_bp = tick_cost_bp(close, dates, market)
    mapped = pd.DataFrame(
        {
            "chg_ratio": chg_ratio,
            "is_ceiling": is_ceiling,
            "tick_cost_bp": tick_bp,
            "tv_clean": tv_clean,
            "mc_clean": mc_clean,
            "close": close,
            "volume": volume,
        }
    )
    mask = select_universe(mapped, screen)
    n_ceiling_excluded = int(np.asarray(is_ceiling, dtype=bool).sum())
    logger.info(
        "[DATA] stage=cost_aware_admission n_raw=%d n_admitted=%d n_ceiling_excluded=%d",
        len(df),
        int(np.asarray(mask, dtype=bool).sum()),
        n_ceiling_excluded,
    )
    flagged = df.copy()
    flagged["admitted"] = np.asarray(mask, dtype=bool)
    return flagged


async def resolve_daily_candidates(client, session) -> list[dict]:
    """자동 비용축 스캔 결과를 그대로 반환합니다.

    Args:
        client: KIS API client.
        session: HTTP session.

    Returns:
        자동 스캔 후보 리스트. 스캔이 비면 빈 리스트를 반환한다.
    """
    return await fetch_candidate_stock_list(client, session) or []


# ---------------------------------------------------------
# 상세 정보 조회 및 데이터 매핑 (비동기)
# ---------------------------------------------------------


async def fetch_single_stock(
    i,
    stock,
    total,
    sem,
    client,
    session,
):
    """단일 종목의 상세 데이터를 수집합니다."""
    async with sem:
        code = stock["code"]
        name = stock["name"]

        price = int(float(stock.get("price", 0)))
        rate = float(stock.get("chgrate", 0))

        open_price = 0
        high_price = 0
        low_price = 0
        close_price = price
        prev_close_price = price
        vol_acml = 0
        market_name = ""
        mkt_cap_eok = 0.0
        trade_amt_eok = 0.0

        # 종목당 4회로 한정: 현재가(KRX) + 투자자추정 + 호가(KRX/NXT)
        from src.config.market_session import DECISION_PRICE_MARKET_DIV_CODES

        _krx_div, _nxt_div = DECISION_PRICE_MARKET_DIV_CODES
        (
            res_detail,
            res_investor,
            res_ob_krx,
            res_ob_nxt,
        ) = await asyncio.gather(
            client.get_current_price(session, code, market_div_code=_krx_div),
            client.get_investor_trend_estimate(session, code),
            client.get_orderbook_snapshot(session, code, market_div_code=_krx_div),
            client.get_orderbook_snapshot(session, code, market_div_code=_nxt_div),
        )

        # 실패한 API 체크 (유지 3종만 판정)
        failed_apis = []
        if res_detail.get("rt_cd") != "0":
            failed_apis.append("현재가")
        if res_investor.get("rt_cd") != "0":
            failed_apis.append("투자자추정")
        if res_ob_krx.get("rt_cd") != "0" or res_ob_nxt.get("rt_cd") != "0":
            failed_apis.append("호가")

        # 데이터 파싱
        detail = res_detail.get("output") if res_detail.get("rt_cd") == "0" else None

        frgn_qty, orgn_qty = 0, 0
        if res_investor.get("rt_cd") == "0" and res_investor.get("output2"):
            latest = res_investor["output2"][0]
            frgn_qty = int(safe_float(latest.get("frgn_fake_ntby_qty", 0)))
            orgn_qty = int(safe_float(latest.get("orgn_fake_ntby_qty", 0)))

        if detail:
            close_price = int(safe_float(detail.get("stck_prpr"), price))
            open_price = int(safe_float(detail.get("stck_oprc"), 0))
            high_price = int(safe_float(detail.get("stck_hgpr"), 0))
            low_price = int(safe_float(detail.get("stck_lwpr"), 0))
            vol_acml = int(safe_float(detail.get("acml_vol"), 0))
            rate = safe_float(detail.get("prdy_ctrt"), rate)
            prev_close_price = int(close_price / (1 + rate / 100)) if rate != 0 else close_price
            price = close_price
            shares = safe_float(detail.get("lstn_stcn"), 0)
            raw_market = str(detail.get("rprs_mrkt_kor_name", "")).upper()
            market_name = "KOSPI" if "KOSPI" in raw_market or "유가" in raw_market else "KOSDAQ" if "KOSDAQ" in raw_market else raw_market
            raw_mkt_cap = safe_float(detail.get("hts_avls")) * 100_000_000 or shares * price
            mkt_cap_eok = round(raw_mkt_cap / 100_000_000, 2)
            trade_amt_eok = round(safe_float(detail.get("acml_tr_pbmn")) / 100_000_000, 2)

        capture_ts = datetime.now(ZoneInfo("Asia/Seoul"))
        orderbook_rows: list[dict] = []
        orderbook_rows.extend(build_orderbook_rows(res_ob_krx, code, _krx_div, "decision", capture_ts))
        orderbook_rows.extend(build_orderbook_rows(res_ob_nxt, code, _nxt_div, "decision", capture_ts))

        frgn_net_eok = round((frgn_qty * price) / 100_000_000, 2)
        orgn_net_eok = round((orgn_qty * price) / 100_000_000, 2)

        return {
            "종목명": name,
            "종목코드": code,
            "시장구분": market_name,
            "시가": open_price,
            "고가": high_price,
            "저가": low_price,
            "종가": close_price,
            "전일종가": prev_close_price,
            "거래량": vol_acml,
            "거래대금": trade_amt_eok,
            "시가총액": mkt_cap_eok,
            "기관_순매수": orgn_net_eok,
            "외국인_순매수": frgn_net_eok,
            "등락률": rate,
        }, failed_apis, orderbook_rows


async def fetch_all_stock_data(
    stock_list,
    client,
    session,
):
    """모든 종목의 상세 데이터를 수집합니다."""
    import sys

    sem = asyncio.Semaphore(settings.API_SEMAPHORE_LIMIT)
    total = len(stock_list)
    completed_count = 0

    async def _track_task(i, stock):
        nonlocal completed_count
        res = await fetch_single_stock(
            i,
            stock,
            total,
            sem,
            client,
            session,
        )
        completed_count += 1
        pct = (completed_count / total) * 100 if total > 0 else 100.0
        bar_len = 25
        filled = int(bar_len * completed_count // total) if total > 0 else bar_len
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(
            f"\r⏳ [수집 진행] [{bar}] {pct:5.1f}% ({completed_count}/{total})"
        )
        sys.stdout.flush()
        return res

    tasks = [_track_task(i, stock) for i, stock in enumerate(stock_list)]
    all_res = await asyncio.gather(*tasks)
    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    results = [r for r, _f, _o in all_res]
    failed_info = [
        (stock_list[i]["name"], stock_list[i]["code"], f)
        for i, (r, f, _o) in enumerate(all_res)
        if f
    ]
    orderbook_rows: list[dict] = []
    for _r, _f, o in all_res:
        if o:
            orderbook_rows.extend(o)
    snapshot_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    try:
        append_orderbook_snapshots(orderbook_rows, snapshot_date)
    except Exception as e:
        logger.warning("[DATA] Orderbook decision snapshot persist failed: %s", e)

    logger.info(f"{Colors.GREEN}✅ 데이터 수집 완료{Colors.RESET}")
    return results, failed_info


def persist_daily_snapshot(df: pd.DataFrame, snapshot_date: str) -> int:
    """일일 wide 스냅샷을 아카이브 저장소에 직접 기록한다.

    Args:
        df: admitted 플래그를 포함한 wide 단면 프레임.
        snapshot_date: 스냅샷 날짜 (YYYY-MM-DD).

    Returns:
        저장된 행수. 빈 프레임은 저장 없이 0을 반환한다.
    """
    if df.empty:
        return 0
    return archive.upsert_archive_snapshot(df, snapshot_date=snapshot_date)


async def main():
    from aiohttp.resolver import ThreadedResolver

    _validate_hts_id()

    # aiohttp 세션 설정 강화 (네트워크 안정성 향상 + DNS 해결)
    timeout = aiohttp.ClientTimeout(
        total=60,  # 전체 요청 타임아웃
        connect=10,  # 연결 타임아웃
        sock_read=30,  # 소켓 읽기 타임아웃
    )
    connector = aiohttp.TCPConnector(
        limit=20,  # 최대 동시 연결 수
        ttl_dns_cache=300,  # DNS 캐시 TTL (5분)
        force_close=False,  # Keep-Alive 유지
        resolver=ThreadedResolver(),  # [Fix] Windows aiodns 이슈 방지용 표준 리졸버 사용
    )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 1. 클라이언트 초기화 및 토큰 확보
        client = KisApiClient(
            APP_KEY, APP_SECRET, ACCOUNT_ID, HTS_ID, token_file=TOKEN_FILE
        )
        await client.ensure_token(session)

        # 2. 시장 지수 조회 (병렬 gather)
        res_kospi, res_kosdaq = await asyncio.gather(
            client.get_market_index_rate(session, "0001"),
            client.get_market_index_rate(session, "1001"),
        )

        kospi_rate = parse_market_index_rate(res_kospi)
        kosdaq_rate = parse_market_index_rate(res_kosdaq)

        # 3. 후보 종목 리스트 확보 (자동 비용축 스캔 단일 경로)
        stock_list = await resolve_daily_candidates(client, session)
        if not stock_list:
            logger.info(f"{Colors.YELLOW}⚠ 자동 스캔 후보가 없습니다.{Colors.RESET}")
            return

        logger.info(
            f"{Colors.BOLD}🚀 K-CLOSING ALPHA :: 실시간 종가매매 데이터 수집 ({len(stock_list)}종목 포착){Colors.RESET}"
        )

        # 4. 상세 데이터 수집
        results, failed_info = await fetch_all_stock_data(stock_list, client, session)

        # 5. wide 단면 구성 후 PIT admitted 플래그 부여 및 저장소 직접 기록
        snapshot_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        df = pd.DataFrame(results)
        df = flag_cost_aware_admission(df, decision_date=pd.Timestamp(snapshot_date))
        df["kospi"] = kospi_rate
        df["kosdaq"] = kosdaq_rate

        # V-KOSPI만 부착 (V-KOSDAQ 조회 제거)
        try:
            from src.api.kis_client import fetch_index_and_calculate_volatility

            (vkospi_val, _vkospi_chg) = await fetch_index_and_calculate_volatility(
                "1028", session=session
            )
        except Exception:
            vkospi_val = 0.0
        df["v_kospi"] = round(vkospi_val, 2)

        stored_rows = persist_daily_snapshot(df, snapshot_date)

        success_count = len(results) - len(failed_info)
        logger.info(f"\n{Colors.BOLD}📊 [데이터 수집 요약]{Colors.RESET}")
        logger.info(f"   ✅ 성공: {Colors.GREEN}{success_count}{Colors.RESET} 종목")
        logger.info(f"{Colors.GREEN}📂 저장소 직접 기록 완료: {snapshot_date} ({stored_rows}행){Colors.RESET}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
