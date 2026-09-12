import asyncio
# ruff: noqa: I001 - contract mandates contiguous wiring import block after Colors
import logging
import sys
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd

from src import settings

# 커스텀 모듈 임포트
from src.api.kis.client import KisApiClient
from src.data.orderbook_store import append_orderbook_snapshots, build_orderbook_rows
from src.utils.display import Colors
from src.daily import archive
from src.daily.universe_scan import fetch_candidate_stock_list
from src.data.trading_calendar import is_kis_trading_day
from src.execution.cost_model import tick_cost_bp
from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL, QUOTE_FAILED_COL
from src.strategy.contract import COST_AWARE_UNIVERSE, UniverseSpec, derive_chg_ratio, mark_ceiling, select_universe

logger = logging.getLogger(__name__)


class NonTradingDayError(RuntimeError):
    """휴장일에 결정 파이프라인이 발화할 때의 fail-closed 오류."""


def build_kiwoom_scan_client() -> Any | None:
    """Kiwoom 스캔 클라이언트를 구성한다. 자격증명이 없으면 None을 반환한다."""
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient()
    if not client.app_key:
        return None
    return client


def build_toss_scan_client() -> Any | None:
    """Toss 랭킹 폴백 클라이언트를 구성한다. 자격증명이 없으면 None을 반환한다."""
    from src.api.toss.client import TossApiClient

    client = TossApiClient()
    if not client.app_key:
        return None
    return client


async def _validate_trading_day(client, session, snapshot_date: str, *, force: bool = False) -> None:
    """휴장일 실행을 차단한다. --force가 유일한 우회 경로다."""
    if force:
        return
    if not await is_kis_trading_day(client, session, snapshot_date):
        raise NonTradingDayError(f"non-trading day: {snapshot_date}")

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


def _validate_decision_window(now: datetime, *, force: bool = False) -> None:
    if force:
        return
    from src.config.market_session import (
        DECISION_WINDOW_END_HHMMSS,
        DECISION_WINDOW_START_HHMMSS,
    )

    hhmmss = now.strftime("%H%M%S")
    if not (DECISION_WINDOW_START_HHMMSS <= hhmmss <= DECISION_WINDOW_END_HHMMSS):
        raise RuntimeError(
            f"결정 창({DECISION_WINDOW_START_HHMMSS}~{DECISION_WINDOW_END_HHMMSS} KST) 밖 실행은 금지됩니다. --force로 우회 가능합니다."
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
        return None
    out1 = data.get("output1")
    if not out1:
        return None
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
    return None


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


async def resolve_daily_candidates(client, session, *, kiwoom_client: Any | None = None, toss_client: Any | None = None) -> list[dict]:
    """자동 비용축 스캔 결과를 그대로 반환합니다.

    Args:
        client: KIS API client.
        session: HTTP session.
        kiwoom_client: Kiwoom scan client (1순위 후보 소스).
        toss_client: Toss scan client (Kiwoom 실패 시 폴백, 선택).

    Returns:
        자동 스캔 후보 리스트. 스캔이 비면 빈 리스트를 반환한다.
    """
    return await fetch_candidate_stock_list(client, session, kiwoom_client=kiwoom_client, toss_client=toss_client) or []


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

        # 종목당 3회로 한정: 현재가(KRX) + 투자자추정 + 호가(KRX)
        from src.config.market_session import KRX_CLOSE_MARKET_DIV_CODE

        _krx_div = KRX_CLOSE_MARKET_DIV_CODE
        (
            res_detail,
            res_investor,
            res_ob_krx,
        ) = await asyncio.gather(
            client.get_current_price(session, code, market_div_code=_krx_div),
            client.get_investor_trend_estimate(session, code),
            client.get_orderbook_snapshot(session, code, market_div_code=_krx_div),
        )

        # 실패한 API 체크 (유지 3종만 판정)
        failed_apis = []
        quote_failed = res_detail.get("rt_cd") != "0"
        if quote_failed:
            failed_apis.append("현재가")
        if res_investor.get("rt_cd") != "0":
            failed_apis.append("투자자추정")
        if res_ob_krx.get("rt_cd") != "0":
            failed_apis.append("호가")

        # 데이터 파싱
        detail = res_detail.get("output") if res_detail.get("rt_cd") == "0" else None

        supply_failed = False
        frgn_qty, orgn_qty = 0, 0
        if res_investor.get("rt_cd") != "0":
            supply_failed = True
        elif res_investor.get("output2"):
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
            sdpr = safe_float(detail.get("stck_sdpr"), 0)
            if sdpr > 0:
                prev_close_price = int(sdpr)
            elif rate != 0:
                prev_close_price = int(close_price / (1 + rate / 100))
            else:
                prev_close_price = close_price
            price = close_price
            shares = safe_float(detail.get("lstn_stcn"), 0)
            raw_market = str(detail.get("rprs_mrkt_kor_name", "")).upper()
            market_name = "KOSPI" if "KOSPI" in raw_market or "유가" in raw_market else "KOSDAQ" if "KOSDAQ" in raw_market else raw_market
            raw_mkt_cap = safe_float(detail.get("hts_avls")) * 100_000_000 or shares * price
            mkt_cap_eok = round(raw_mkt_cap / 100_000_000, 2)
            trade_amt_eok = round(safe_float(detail.get("acml_tr_pbmn")) / 100_000_000, 2)

        if quote_failed:
            # 현재가 실패: 스캔 행 값(종가/전일종가)만 유지하고 OHLCV는 NaN (0 위조 금지)
            close_price = price
            prev_close_price = int(price / (1 + rate / 100)) if rate != 0 else price
            open_price = float("nan")
            high_price = float("nan")
            low_price = float("nan")
            vol_acml = float("nan")
            mkt_cap_eok = float("nan")
            trade_amt_eok = float("nan")

        capture_ts = datetime.now(ZoneInfo("Asia/Seoul"))
        orderbook_rows: list[dict] = []
        orderbook_rows.extend(build_orderbook_rows(res_ob_krx, code, _krx_div, "decision", capture_ts))

        if supply_failed:
            frgn_net_eok = float("nan")
            orgn_net_eok = float("nan")
        else:
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
            "수급_실패": supply_failed,
            QUOTE_FAILED_COL: quote_failed,
            DECISION_CLOSE_COL: close_price,
            CLOSE_CONFIRMED_COL: False,
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


async def main(force: bool = False):
    from aiohttp.resolver import ThreadedResolver

    _validate_hts_id()
    _validate_decision_window(datetime.now(ZoneInfo("Asia/Seoul")), force=force)

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

        snapshot_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        kiwoom_client = build_kiwoom_scan_client()
        toss_client = build_toss_scan_client()
        await _validate_trading_day(client, session, snapshot_date, force=force)

        # 2. 시장 지수 조회 (병렬 gather)
        res_kospi, res_kosdaq = await asyncio.gather(
            client.get_market_index_rate(session, "0001"),
            client.get_market_index_rate(session, "1001"),
        )

        kospi_rate = parse_market_index_rate(res_kospi)
        kosdaq_rate = parse_market_index_rate(res_kosdaq)

        # 3. 후보 종목 리스트 확보 (자동 비용축 스캔 단일 경로, Toss 폴백 포함)
        stock_list = await resolve_daily_candidates(client, session, kiwoom_client=kiwoom_client, toss_client=toss_client)
        if not stock_list:
            logger.info(f"{Colors.YELLOW}⚠ 자동 스캔 후보가 없습니다.{Colors.RESET}")
            return

        logger.info(
            f"{Colors.BOLD}🚀 [1/3] 후보 종목 스캔 (Kiwoom / KIS){Colors.RESET}\n"
            f"   - 대상: {Colors.CYAN}{len(stock_list)}{Colors.RESET}개 종목 포착 (등락률/유니버스 필터)"
        )

        # 4. 상세 데이터 수집
        logger.info(f"\n{Colors.BOLD}⏳ [2/3] 실시간 단면 데이터 수집{Colors.RESET}")
        results, failed_info = await fetch_all_stock_data(stock_list, client, session)

        # 5. wide 단면 구성 후 PIT admitted 플래그 부여 및 저장소 직접 기록
        logger.info(f"\n{Colors.BOLD}📊 [3/3] 유니버스 적격성(Admission) 평가 및 저장{Colors.RESET}")
        capture_ts = pd.Timestamp.now(tz="Asia/Seoul")
        df = pd.DataFrame(results)
        df["snapshot_timestamp"] = capture_ts
        df = flag_cost_aware_admission(df, decision_date=pd.Timestamp(snapshot_date))
        index_failed = False
        if kospi_rate is None:
            df["kospi"] = float("nan")
            index_failed = True
        else:
            df["kospi"] = kospi_rate
        if kosdaq_rate is None:
            df["kosdaq"] = float("nan")
            index_failed = True
        else:
            df["kosdaq"] = kosdaq_rate

        # V-KOSPI만 부착 (V-KOSDAQ 조회 제거)
        try:
            from src.api.kis.indicators import fetch_index_and_calculate_volatility

            (vkospi_val, _vkospi_chg) = await fetch_index_and_calculate_volatility(
                "1028", session=session
            )
        except Exception:
            vkospi_val = float("nan")
            index_failed = True
        df["v_kospi"] = round(float(vkospi_val), 2)
        df["지수_실패"] = index_failed

        stored_rows = persist_daily_snapshot(df, snapshot_date)

        n_raw = len(df)
        n_admitted = int(df["admitted"].sum()) if "admitted" in df.columns else 0
        n_failed = len(failed_info)
        success_count = n_raw - n_failed

        divider = "─" * 60
        box_top = "━" * 60
        logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")
        logger.info(f" {Colors.GREEN}{Colors.BOLD}📋 [데이터 수집 & 적재 요약]{Colors.RESET} ({snapshot_date})")
        logger.info(f"{Colors.BOLD}{divider}{Colors.RESET}")
        logger.info(f"   • 스캔 및 수집 시도 : {n_raw:>3} 종목 (성공: {Colors.GREEN}{success_count}{Colors.RESET}, 실패: {Colors.RED if n_failed > 0 else Colors.GRAY}{n_failed}{Colors.RESET})")
        logger.info(f"   • 유니버스 적격 통과: {Colors.CYAN}{Colors.BOLD}{n_admitted:>3}{Colors.RESET} 종목 (비용/거래대금 필터 통과)")
        logger.info(f"   • 스냅샷 저장소 적재: {Colors.GREEN}{stored_rows:>3}{Colors.RESET} 행 적재 완료")
        logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main(force="--force" in sys.argv))  # pragma: no cover
