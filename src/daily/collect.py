import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd

from src import settings

# 커스텀 모듈 임포트
from src.api.kis_client import KisApiClient, prefetch_ohlcv_for_sma120
from src.data.db_loader import load_theme_from_db
from src.data.orderbook_store import append_orderbook_snapshots, build_orderbook_rows
from src.data.theme_resolver import batch_resolve_missing_themes
from src.processing.schema import STANDARD_COLUMN_ORDER
from src.utils.display import Colors

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

logger.debug("조건검색 (표준 CSV 저장) 시작...")


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
def save_collected_condition_data(
    df: pd.DataFrame,
    csv_path: Path | str,
    excel_path: Path | str | None = None,
) -> Path:
    """조건검색 DataFrame 을 표준 열 순서로 저장한다.

    - utf-8-sig 인코딩 CSV 로 저장하여 Excel / Google Sheets 에서 한글 깨짐 없이 열림.
    - ``STANDARD_COLUMN_ORDER`` 순서를 엄격히 유지하고, 종목코드는 6자리 zero-fill 문자열로 저장.
    - ``excel_path`` 가 주어지면 레거시 호환용 xlsx 도 함께 기록한다 (셀 포맷팅 없음).
    """
    out = df.copy()
    out = out.reindex(columns=STANDARD_COLUMN_ORDER)
    if "종목코드" in out.columns:
        out["종목코드"] = out["종목코드"].astype(str).str.zfill(6)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    parquet_path = csv_path.with_suffix(".parquet")
    try:
        out.to_parquet(parquet_path, index=False)
        logger.debug("Parquet 저장 완료: %s", parquet_path)
    except Exception:
        logger.exception("Parquet 저장 실패 (CSV 는 정상 저장됨): %s", parquet_path)

    if excel_path is not None:
        out.to_excel(Path(excel_path), index=False)

    logger.debug("표준 CSV 저장 완료: %s (%d행)", csv_path, len(out))
    return csv_path


# ---------------------------------------------------------
# 상세 정보 조회 및 데이터 매핑 (비동기)
# ---------------------------------------------------------
async def _empty_orderbook_result() -> dict:
    return {"rt_cd": "0", "output1": {}, "output2": []}


async def fetch_single_stock(
    i,
    stock,
    total,
    sem,
    client,
    session,
    overheated_stock_codes=None,
    new_high_stock_codes=None,
    near_new_high_stock_codes=None,
    upper_limit_next_day_stock_codes=None,
    upper_limit_stock_codes=None,
    ohlcv_cache=None,
):
    """단일 종목의 상세 데이터를 수집합니다."""
    if overheated_stock_codes is None:
        overheated_stock_codes = set()
    if new_high_stock_codes is None:
        new_high_stock_codes = set()
    if near_new_high_stock_codes is None:
        near_new_high_stock_codes = set()
    if upper_limit_next_day_stock_codes is None:
        upper_limit_next_day_stock_codes = set()
    if upper_limit_stock_codes is None:
        upper_limit_stock_codes = set()

    async with sem:
        code = stock["code"]
        name = stock["name"]

        try:
            price = int(float(stock.get("price", 0)))
            rate = float(stock.get("chgrate", 0))
        except:
            price, rate = 0, 0.0

        open_price = 0
        high_price = 0
        low_price = 0
        close_price = price
        prev_close_price = price
        vol_acml = 0
        market_name = ""
        mkt_cap_eok = 0.0
        trade_amt_eok = 0.0

        # KisApiClient를 사용하여 비동기로 여러 API 동시 호출
        # 실계좌 SOR 대응: 의사결정시점 현재가를 J/NX 각각 명시 조회해 병기한다.
        from src.config.market_session import DECISION_PRICE_MARKET_DIV_CODES

        _krx_div, _nxt_div = DECISION_PRICE_MARKET_DIV_CODES
        ob_krx_coro = (
            client.get_orderbook_snapshot(session, code, market_div_code=_krx_div)
            if hasattr(client, "get_orderbook_snapshot")
            else _empty_orderbook_result()
        )
        ob_nxt_coro = (
            client.get_orderbook_snapshot(session, code, market_div_code=_nxt_div)
            if hasattr(client, "get_orderbook_snapshot")
            else _empty_orderbook_result()
        )
        (
            res_krx,
            res_nxt,
            res_strength,
            res_investor,
            res_program,
            res_ob_krx,
            res_ob_nxt,
        ) = await asyncio.gather(
            client.get_current_price(session, code, market_div_code=_krx_div),
            client.get_current_price(session, code, market_div_code=_nxt_div),
            client.get_trade_strength(session, code),
            client.get_investor_trend_estimate(session, code),
            client.get_program_net_buy(session, code),
            ob_krx_coro,
            ob_nxt_coro,
        )
        res_detail = res_krx

        # 실패한 API 체크 (NXT 미상장 NX 조회 실패는 정상 케이스로 흡수)
        failed_apis = []
        if res_detail.get("rt_cd") != "0":
            failed_apis.append("현재가")
        if res_strength.get("rt_cd") != "0":
            failed_apis.append("체결강도")
        if res_investor.get("rt_cd") != "0":
            failed_apis.append("투자자추정")
        if res_program.get("rt_cd") != "0":
            failed_apis.append("프로그램")
        if not isinstance(res_ob_krx, dict) or res_ob_krx.get("rt_cd") != "0":
            failed_apis.append("호가")

        # 데이터 파싱
        detail = res_detail.get("output") if res_detail.get("rt_cd") == "0" else None

        vol_power = 0.0
        if res_strength.get("rt_cd") == "0" and res_strength.get("output"):
            latest = res_strength["output"][0]
            vol_power = safe_float(latest.get("tday_rltv") or latest.get("ctrb_strgt"))

        frgn_qty, orgn_qty = 0, 0
        if res_investor.get("rt_cd") == "0" and res_investor.get("output2"):
            latest = res_investor["output2"][0]
            frgn_qty = int(safe_float(latest.get("frgn_fake_ntby_qty", 0)))
            orgn_qty = int(safe_float(latest.get("orgn_fake_ntby_qty", 0)))

        program_amt_won = 0.0
        if res_program.get("rt_cd") == "0" and res_program.get("output"):
            program_amt_won = safe_float(
                res_program["output"][0].get("whol_smtn_ntby_tr_pbmn", 0)
            )
        if detail:
            try:
                # 가격 데이터 추출
                close_price = int(safe_float(detail.get("stck_prpr"), price))
                open_price = int(safe_float(detail.get("stck_oprc"), 0))
                high_price = int(safe_float(detail.get("stck_hgpr"), 0))
                low_price = int(safe_float(detail.get("stck_lwpr"), 0))
                vol_acml = int(safe_float(detail.get("acml_vol"), 0))

                rate = safe_float(detail.get("prdy_ctrt"), rate)

                # 전일 종가 계산
                if rate != 0:
                    prev_close_price = int(close_price / (1 + rate / 100))
                else:
                    prev_close_price = close_price

                price = close_price
                shares = safe_float(detail.get("lstn_stcn"), 0)
                raw_market = str(detail.get("rprs_mrkt_kor_name", "")).upper()
                market_name = (
                    "KOSPI"
                    if "KOSPI" in raw_market or "유가" in raw_market
                    else "KOSDAQ" if "KOSDAQ" in raw_market else raw_market
                )
                raw_mkt_cap = safe_float(detail.get("hts_avls")) * 100_000_000
                if raw_mkt_cap == 0:
                    raw_mkt_cap = shares * price
                mkt_cap_eok = round(raw_mkt_cap / 100_000_000, 2)
                trade_amt_eok = round(
                    safe_float(detail.get("acml_tr_pbmn")) / 100_000_000, 2
                )
            except Exception as e:
                logger.info(f"\n {Colors.RED}⚠️ [{name}] 파싱 에러: {e}{Colors.RESET}")

        # 듀얼벤뉴 의사결정 가격: NXT는 rt_cd 0 + acml_vol>0일 때만 신뢰한다.
        nxt_price: int | None = None
        try:
            if res_nxt.get("rt_cd") == "0" and res_nxt.get("output"):
                nxt_out = res_nxt["output"]
                nxt_vol = int(safe_float(nxt_out.get("acml_vol"), 0))
                if nxt_vol > 0:
                    nxt_price = int(safe_float(nxt_out.get("stck_prpr"), 0)) or None
        except Exception:
            nxt_price = None
        krx_price = close_price
        if nxt_price is not None:
            sor_effective_price = min(krx_price, nxt_price)
        else:
            sor_effective_price = krx_price

        def _parse_orderbook(res):
            if not isinstance(res, dict) or res.get("rt_cd") != "0":
                return 0, 0, 0, 0
            out1 = res.get("output1")
            if not isinstance(out1, dict):
                return 0, 0, 0, 0
            ask1 = int(safe_float(out1.get("askp1"), 0))
            bid1 = int(safe_float(out1.get("bidp1"), 0))
            ask_rsqn = int(safe_float(out1.get("total_askp_rsqn"), 0))
            bid_rsqn = int(safe_float(out1.get("total_bidp_rsqn"), 0))
            return ask1, bid1, ask_rsqn, bid_rsqn

        krx_ask1, krx_bid1, krx_ask_rsqn, krx_bid_rsqn = _parse_orderbook(res_ob_krx)
        nxt_ask1, nxt_bid1, nxt_ask_rsqn, nxt_bid_rsqn = _parse_orderbook(res_ob_nxt)

        capture_ts = datetime.now(ZoneInfo("Asia/Seoul"))
        orderbook_rows: list[dict] = []
        orderbook_rows.extend(build_orderbook_rows(res_ob_krx, code, _krx_div, "decision", capture_ts))
        orderbook_rows.extend(build_orderbook_rows(res_ob_nxt, code, _nxt_div, "decision", capture_ts))

        frgn_net_eok = round((frgn_qty * price) / 100_000_000, 2)
        orgn_net_eok = round((orgn_qty * price) / 100_000_000, 2)
        program_net_eok = round(program_amt_won / 100_000_000, 2)

        scenario = ""
        data_count = 0
        sma_value, sma_success = 0, False

        # 1차 시나리오 우선판단 (상따, 상한가 다음날, 상승형 음봉, 신고가, 신고가 근접)
        if code in upper_limit_stock_codes:
            scenario = "상따"
        elif code in upper_limit_next_day_stock_codes:
            scenario = "상한가 다음날"
        elif rate > 0 and close_price < open_price:
            scenario = "상승형 음봉"
        elif code in new_high_stock_codes:
            scenario = "신고가"
        elif code in near_new_high_stock_codes:
            scenario = "신고가 근접"

        # 1차 시나리오 미확정 시 120일 이동평균 계산 진행
        if not scenario:
            try:
                from src.api.kis_client import calculate_all_moving_averages

                prefetched = (ohlcv_cache or {}).get(code)
                (_, (_, _, data_count), _, (sma120_val, sma120_ok)) = (
                    await calculate_all_moving_averages(
                        code, session=session, client=client,
                        prefetched_records=prefetched,
                    )
                )
                sma_value, sma_success = float(sma120_val), sma120_ok
                if sma_success and sma_value > 0 and prev_close_price < sma_value <= close_price:
                    scenario = "120 돌파"
            except Exception:
                pass

        if not scenario:
            scenario = "거래량 폭증"

        return {
            "종목명": name,
            "종목코드": code,
            "시가": open_price,
            "고가": high_price,
            "저가": low_price,
            "종가": close_price,
            "전일종가": prev_close_price,
            "시장구분": market_name,
            "시가총액": mkt_cap_eok,
            "거래대금": trade_amt_eok,
            "체결강도": vol_power,
            "등락률": rate,
            "선정순위": 0,
            "기관_순매수": orgn_net_eok,
            "외국인_순매수": frgn_net_eok,
            "프로그램_순매수": program_net_eok,
            "시나리오": scenario,
            "거래량": vol_acml,
            "krx_현재가": krx_price,
            "nxt_현재가": nxt_price,
            "sor_effective_price": sor_effective_price,
            "krx_매도호가1": krx_ask1,
            "krx_매수호가1": krx_bid1,
            "krx_매도잔량": krx_ask_rsqn,
            "krx_매수잔량": krx_bid_rsqn,
            "nxt_매도호가1": nxt_ask1,
            "nxt_매수호가1": nxt_bid1,
            "nxt_매도잔량": nxt_ask_rsqn,
            "nxt_매수잔량": nxt_bid_rsqn,
        }, failed_apis, orderbook_rows


async def fetch_all_stock_data(
    stock_list,
    client,
    session,
    overheated_stock_codes=None,
    new_high_stock_codes=None,
    near_new_high_stock_codes=None,
    upper_limit_next_day_stock_codes=None,
    upper_limit_stock_codes=None,
):
    """모든 종목의 상세 데이터를 수집합니다."""
    import sys

    if overheated_stock_codes is None:
        overheated_stock_codes = set()
    if new_high_stock_codes is None:
        new_high_stock_codes = set()
    if near_new_high_stock_codes is None:
        near_new_high_stock_codes = set()
    if upper_limit_next_day_stock_codes is None:
        upper_limit_next_day_stock_codes = set()
    if upper_limit_stock_codes is None:
        upper_limit_stock_codes = set()

    # Phase A: 1차 시나리오 확정 가능 종목 분류 (SMA120 조회 불필요)
    primary_matched_codes = (
        upper_limit_stock_codes
        | upper_limit_next_day_stock_codes
        | new_high_stock_codes
        | near_new_high_stock_codes
    )
    sma_needed_codes = [
        stock["code"] for stock in stock_list
        if stock["code"] not in primary_matched_codes
    ]

    # Phase B: SMA120 필요 종목 OHLCV 사전 병렬 선조회
    if sma_needed_codes and client is not None and session is not None:
        ohlcv_cache = await prefetch_ohlcv_for_sma120(sma_needed_codes, session=session, client=client)
    else:
        ohlcv_cache = {}

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
            overheated_stock_codes,
            new_high_stock_codes,
            near_new_high_stock_codes,
            upper_limit_next_day_stock_codes,
            upper_limit_stock_codes,
            ohlcv_cache,
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

        # 3. 조건 목록 및 대상 조건 찾기
        res_cond_list = await client.get_condition_list(session)
        if res_cond_list.get("rt_cd") != "0":
            logger.info(
                f"{Colors.RED}조건식 목록 조회 실패: "
                f"rt_cd={res_cond_list.get('rt_cd')}, "
                f"msg={res_cond_list.get('msg1', 'N/A')}{Colors.RESET}"
            )
            return
        my_conditions = res_cond_list.get("output2", [])

        if not my_conditions:
            logger.info(
                f"{Colors.YELLOW}⚠ 저장된 조건이 없습니다. "
                f"(HTS ID: {HTS_ID}, output2 비어 있음){Colors.RESET}"
            )
            return
        target_cond = next(
            (c for c in my_conditions if TARGET_CONDITION_NAME in c["condition_nm"]),
            None,
        )
        if not target_cond:
            logger.info(
                f"{Colors.RED}❌ '{TARGET_CONDITION_NAME}' 조건을 찾을 수 없습니다.{Colors.RESET}"
            )
            return

        # 4. 조건검색 결과 조회 (종가매매)
        res_cond_res = await client.get_condition_result(session, target_cond["seq"])
        if res_cond_res.get("rt_cd") != "0":
            logger.error(f"{Colors.RED}❌ 검색 실패: {res_cond_res.get('msg1')}{Colors.RESET}")
            return

        stock_list = res_cond_res.get("output2", [])

        # 4-1~4-5. 조건검색 결과 병렬 조회 (Phase 0 최적화)
        condition_candidates = [
            (settings.OVERHEATED_CONDITION_NAME, "overheated"),
            (settings.NEW_HIGH_CONDITION_NAME, "new_high"),
            (settings.NEAR_NEW_HIGH_CONDITION_NAME, "near_new_high"),
            (settings.UPPER_LIMIT_NEXT_DAY_CONDITION_NAME, "upper_limit_next"),
            (settings.UPPER_LIMIT_CONDITION_NAME, "upper_limit"),
        ]
        cond_seq_map = {}
        for cond_name, key in condition_candidates:
            matched = next(
                (c for c in my_conditions if c["condition_nm"] == cond_name),
                None,
            )
            if matched:
                cond_seq_map[key] = matched["seq"]

        async def _fetch_condition_codes(seq):
            if seq is None:
                return set()
            res = await client.get_condition_result(session, seq)
            if res.get("rt_cd") != "0":
                return set()
            return {stock.get("code") for stock in res.get("output2", [])}

        (
            overheated_stock_codes,
            new_high_stock_codes,
            near_new_high_stock_codes,
            upper_limit_next_day_stock_codes,
            upper_limit_stock_codes,
        ) = await asyncio.gather(
            _fetch_condition_codes(cond_seq_map.get("overheated")),
            _fetch_condition_codes(cond_seq_map.get("new_high")),
            _fetch_condition_codes(cond_seq_map.get("near_new_high")),
            _fetch_condition_codes(cond_seq_map.get("upper_limit_next")),
            _fetch_condition_codes(cond_seq_map.get("upper_limit")),
        )

        # 조건검색 결과 통합 카드 리포트 출력
        logger.info(
            f"\n{Colors.BOLD}================================================================================{Colors.RESET}"
        )
        logger.info(
            f"{Colors.BOLD}🚀 K-CLOSING ALPHA :: 실시간 종가매매 데이터 수집 ({len(stock_list)}종목 포착){Colors.RESET}"
        )
        logger.info(
            f"{Colors.BOLD}================================================================================{Colors.RESET}"
        )
        logger.info(f"   • 🔺 상한가       : {len(upper_limit_stock_codes)}종목")
        logger.info(f"   • ⬆️ 상한가 다음날 : {len(upper_limit_next_day_stock_codes)}종목")
        logger.info(f"   • 🚀 신고가       : {len(new_high_stock_codes)}종목")
        logger.info(f"   • 📈 신고가 근접  : {len(near_new_high_stock_codes)}종목")
        logger.info(f"   • 🔥 단기과열     : {len(overheated_stock_codes)}종목")
        logger.info(
            f"{Colors.BOLD}================================================================================{Colors.RESET}\n"
        )

        # 5. 상세 데이터 수집 (HTS 조건검색 결과 전달)
        results, failed_info = await fetch_all_stock_data(
            stock_list,
            client,
            session,
            overheated_stock_codes,
            new_high_stock_codes,
            near_new_high_stock_codes,
            upper_limit_next_day_stock_codes,
            upper_limit_stock_codes,
        )
        if results:
            # 데이터 저장 및 후처리 로직 (표준 CSV 저장)
            save_path = str(settings.CONDITION_CSV_PATH)

            df = pd.DataFrame(results)
            df = df.sort_values(by="거래대금", ascending=False)
            df["선정순위"] = range(1, len(df) + 1)

            # 추가 지표 계산
            total_count = len(df)
            df["총_종목수"] = total_count
            df["평균_거래대금"] = round(df["거래대금"].mean(), 2)
            df["kospi"] = kospi_rate
            df["kosdaq"] = kosdaq_rate

            # [New] V-KOSPI & V-KOSDAQ 계산 및 추가
            try:
                from src.api.kis_client import fetch_index_and_calculate_volatility

                # V-KOSPI (KOSPI 200: 1028) & V-KOSDAQ (KOSDAQ 150: 2203) 병렬 조회
                (vkospi_val, vkospi_chg), (vkosdaq_val, vkosdaq_chg) = await asyncio.gather(
                    fetch_index_and_calculate_volatility("1028", session=session),
                    fetch_index_and_calculate_volatility("2203", session=session),
                )
                df["v_kospi"] = round(vkospi_val, 2)
                df["v_kosdaq"] = round(vkosdaq_val, 2)

            except Exception:
                df["v_kospi"] = 0.0
                df["v_kosdaq"] = 0.0

            # 표준 열 순서로 utf-8-sig CSV (+ Parquet) 저장
            save_collected_condition_data(df, save_path)

            # 수집 결과 요약 출력
            success_count = len(results) - len(failed_info)

            # 테마 미매칭 종목 자동 분류 및 로컬 Parquet/DB 캐시 갱신 (구글 시트 수동 의존 제거)
            theme_map = load_theme_from_db()
            df["테마"] = df["종목코드"].map(theme_map)
            no_theme_df = df[df["테마"].isna() | (df["테마"] == "")]

            if not no_theme_df.empty:
                no_theme_list = no_theme_df[["종목코드", "종목명", "시장구분"]].to_dict("records")
                logger.info(
                    f"\n{Colors.BOLD}⚠️ [신규/미분류 종목] {Colors.YELLOW}{len(no_theme_list)}{Colors.RESET} 종목 발견 -> 자동 분류 및 로컬 캐시 갱신..."
                )
                resolved_list = batch_resolve_missing_themes(no_theme_list)
                for s in resolved_list:
                    theme_val = s.get("테마", "기타")
                    mkt_val = s.get("시장구분", "")
                    logger.info(
                        f"   👉 {s.get('종목명', '')}({s.get('종목코드', '')}) -> 테마: {theme_val} | 시장: {mkt_val}"
                    )
                # 갱신된 로컬 테마 DB 재매핑
                theme_map = load_theme_from_db()
                df["테마"] = df["종목코드"].map(theme_map).fillna("기타")
                save_collected_condition_data(df, save_path)
            logger.info(f"\n{Colors.BOLD}� [데이터 수집 요약]{Colors.RESET}")
            logger.info(f"   ✅ 성공: {Colors.GREEN}{success_count}{Colors.RESET} 종목")
            if failed_info:
                logger.info(f"   ❌ 실패: {Colors.RED}{len(failed_info)}{Colors.RESET} 종목")
                for name, code, apis in failed_info:
                    logger.info(
                        f"      - {name} ({code}) [실패항목: {Colors.YELLOW}{', '.join(apis)}{Colors.RESET}]"
                    )

            logger.info(f"\n{Colors.GREEN}📂 표준 CSV 저장 완료: {save_path}{Colors.RESET}")
        else:
            logger.info(f"{Colors.YELLOW}⚠ 검색된 종목이 없습니다.{Colors.RESET}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
