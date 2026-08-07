"""한국투자증권 REST API 클라이언트 (HTTP/토큰/요청 동작)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta

import aiohttp

from src import settings
from src.api.kis.rate_limit import AsyncRateLimiter

logger = logging.getLogger(__name__)


class KisApiClient:
    def __init__(
        self,
        app_key=None,
        app_secret=None,
        account_id=None,
        hts_id=None,
        base_url=None,
        token_file=None,
    ):
        self.app_key = app_key or settings.KIS_API_CONFIG.get("app_key")
        self.app_secret = app_secret or settings.KIS_API_CONFIG.get("app_secret")
        self.account_id = account_id or settings.KIS_API_CONFIG.get("account_id")
        self.hts_id = hts_id or settings.KIS_API_CONFIG.get("hts_id")
        self.base_url = base_url or settings.KIS_BASE_URL
        self.token = None
        self.token_file = str(token_file or settings.TOKEN_FILE)
        self._market_div_cache = {}
        # 동시성 및 레이트 리밋 제어를 위한 세마포어와 AsyncRateLimiter
        self.semaphore = asyncio.Semaphore(10)
        self.rate_limiter = AsyncRateLimiter(max_rate=18.0, time_period=1.0)

    def create_session(self, *, timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientSession:
        """최적화된 커넥터와 bounded request timeout을 가진 세션을 생성합니다."""
        from aiohttp.resolver import ThreadedResolver
        resolver = ThreadedResolver()
        request_timeout = timeout or aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        connector = aiohttp.TCPConnector(
            limit=50,            # 동시 연결 수 제한
            ttl_dns_cache=300,  # DNS 캐시 유지 시간
            use_dns_cache=True,  # DNS 캐시 사용
            resolver=resolver   # DNS 해석기 추가
        )
        return aiohttp.ClientSession(timeout=request_timeout, connector=connector)

    @staticmethod
    def _normalize_market_div_code(market_div_code):
        if market_div_code is None:
            return ""
        code = str(market_div_code).strip().upper()
        return code if code in {"J", "NX", "UN"} else ""

    def _market_div_candidates(self, preferred_market_div_code=None):
        preferred = self._normalize_market_div_code(preferred_market_div_code)
        candidates = []
        if preferred:
            candidates.append(preferred)
        for code in ("UN", "J", "NX"):
            if code not in candidates:
                candidates.append(code)
        return candidates

    async def _request_with_market_div_fallback(
        self,
        session,
        url,
        tr_id,
        params,
        market_div_param_key,
        preferred_market_div_code=None,
        require_non_empty_output2=False,
    ):
        last_res = {"rt_cd": "9", "msg1": "market_div fallback failed"}
        for market_div_code in self._market_div_candidates(preferred_market_div_code):
            req_params = dict(params)
            req_params[market_div_param_key] = market_div_code
            res = await self._handle_request(
                session.get,
                url,
                headers=self._get_headers(tr_id),
                params=req_params,
            )
            if res.get("rt_cd") == "0":
                if require_non_empty_output2 and not res.get("output2"):
                    last_res = {
                        **res,
                        "msg1": f"{res.get('msg1', 'empty output2')} [market_div={market_div_code}]",
                    }
                    continue
                return res, market_div_code
            last_res = res
        return last_res, None

    async def resolve_stock_market_div_code(
        self, session: aiohttp.ClientSession, code: str, preferred_market_div_code=None
    ) -> str:
        cached = self._market_div_cache.get(code)
        if cached:
            return cached

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {"fid_input_iscd": code}
        res, used_market_div = await self._request_with_market_div_fallback(
            session=session,
            url=url,
            tr_id="FHKST01010100",
            params=params,
            market_div_param_key="fid_cond_mrkt_div_code",
            preferred_market_div_code=preferred_market_div_code,
        )
        if res.get("rt_cd") == "0" and used_market_div:
            self._market_div_cache[code] = used_market_div
            return used_market_div

        # Fallback default for resiliency
        self._market_div_cache[code] = "J"
        return "J"

    async def ensure_token(self, session: aiohttp.ClientSession, force_refresh: bool = False):
        """토큰 유효성을 확인하고 필요시 갱신합니다."""
        if not force_refresh and os.path.exists(self.token_file):
            try:
                with open(self.token_file, encoding="utf-8") as f:
                    saved_data = json.load(f)
                if saved_data.get("app_key") == self.app_key:
                    expired_at = datetime.strptime(
                        saved_data["expired_at"], "%Y-%m-%d %H:%M:%S"
                    )
                    if datetime.now() < expired_at - timedelta(minutes=10):
                        self.token = saved_data["access_token"]
                        return self.token
            except Exception:
                pass

        # 새 토큰 발급 요청
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        async with session.post(url, headers=headers, json=body) as resp:
            data = await resp.json()
            if "access_token" not in data:
                raise Exception(f"토큰 발급 실패: {data}")

            self.token = data["access_token"]
            expires_in = data.get("expires_in", 86400)
            expired_at_str = (datetime.now() + timedelta(seconds=expires_in)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(self.token_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "access_token": self.token,
                        "expired_at": expired_at_str,
                        "app_key": self.app_key,
                    },
                    f,
                )
            return self.token

    def _get_headers(self, tr_id):
        """공통 헤더 생성"""
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appKey": self.app_key,
            "appSecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    async def _handle_request(self, session_method, url, **kwargs):
        """재시도 로직을 포함한 공통 요청 처리 (네트워크 및 토큰 재발급 에러 처리 강화)"""
        import aiohttp
        
        session = getattr(session_method, "__self__", None)
        await self.rate_limiter.acquire()
        for attempt in range(5):
            try:
                # 최신 토큰으로 headers의 authorization 동기화
                if "headers" in kwargs and isinstance(kwargs["headers"], dict):
                    kwargs["headers"]["authorization"] = f"Bearer {self.token}"

                async with session_method(url, **kwargs) as resp:
                    if resp.status == 429:  # Too Many Requests
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue

                    data = await resp.json()
                    # 토큰 만료/유효하지 않음 에러 자동 재발급 처리
                    msg_cd = data.get("msg_cd", "")
                    msg1 = data.get("msg1", "")
                    if (
                        data.get("rt_cd") != "0"
                        and (msg_cd in ("EGW00121", "EGW00123") or "token" in msg1.lower() or "토큰" in msg1)
                        and session
                        and attempt < 2
                    ):
                        logger.warning("유효하지 않은 토큰 감지 (%s). 토큰 강제 재발급 진행...", msg_cd or msg1)
                        await self.ensure_token(session, force_refresh=True)
                        await asyncio.sleep(0.2)
                        continue

                    # KIS 특유의 TPS 초과 메시지 처리
                    if data.get("rt_cd") != "0" and "초당 거래건수" in msg1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    return data
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientError) as e:
                # 네트워크 연결 에러 시 지수 백오프로 재시도
                if attempt < 4:  # 마지막 시도가 아니면
                    wait_time = 0.5 * (2 ** attempt)  # 0.5초, 1초, 2초, 4초
                    logger.warning(
                        "네트워크 에러 (%s), %.1f초 후 재시도... (%d/5)",
                        type(e).__name__,
                        wait_time,
                        attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # 마지막 시도에서도 실패하면 에러 반환
                    return {"rt_cd": "9", "msg1": f"네트워크 연결 실패: {str(e)[:100]}"}
        return {"rt_cd": "9", "msg1": "최대 재시도 횟수 초과 (TPS 제한)"}

    async def get_current_price(self, session, code, market_div_code=None):
        """주식 현재가 시세 조회 (FHKST01010100)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {"fid_input_iscd": code}
        preferred_market_div = market_div_code or self._market_div_cache.get(code)
        res, used_market_div = await self._request_with_market_div_fallback(
            session=session,
            url=url,
            tr_id="FHKST01010100",
            params=params,
            market_div_param_key="fid_cond_mrkt_div_code",
            preferred_market_div_code=preferred_market_div,
        )
        if res.get("rt_cd") == "0" and used_market_div:
            self._market_div_cache[code] = used_market_div
        return res

    async def get_program_net_buy(self, session, code, market_div_code=None):
        """종목별 프로그램 매매 추이 (FHPPG04650101)"""
        url = (
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/program-trade-by-stock"
        )
        params = {"FID_INPUT_ISCD": code}
        preferred_market_div = market_div_code or self._market_div_cache.get(code)
        res, used_market_div = await self._request_with_market_div_fallback(
            session=session,
            url=url,
            tr_id="FHPPG04650101",
            params=params,
            market_div_param_key="FID_COND_MRKT_DIV_CODE",
            preferred_market_div_code=preferred_market_div,
        )
        if res.get("rt_cd") == "0" and used_market_div:
            self._market_div_cache[code] = used_market_div
        return res

    async def get_market_index_rate(self, session, market_code):
        """시장 지수 등락률 조회 (FHKUP03500100) - 최근 5일"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
        now = datetime.now()
        str_today = now.strftime("%Y%m%d")
        str_past = (now - timedelta(days=5)).strftime("%Y%m%d")
        params = {
            "fid_cond_mrkt_div_code": "U",
            "fid_input_iscd": market_code,
            "fid_input_date_1": str_past,
            "fid_input_date_2": str_today,
            "fid_period_div_code": "D",
            "fid_org_adj_prc": "0",
        }
        return await self._handle_request(
            session.get, url, headers=self._get_headers("FHKUP03500100"), params=params
        )

    async def get_market_index_history(
        self, session, market_code, start_date, end_date, period_code="D"
    ):
        """시장 지수/업종 기간별 시세 조회 (FHKUP03500100) - 기간 지정 가능
        market_code: 업종/지수 코드 (KOSPI: 0001, KOSDAQ: 1001, V-KOSPI: 200 등)
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        period_code: D(일), W(주), M(월), Y(년)
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
        params = {
            "fid_cond_mrkt_div_code": "U",
            "fid_input_iscd": market_code,
            "fid_input_date_1": start_date,
            "fid_input_date_2": end_date,
            "fid_period_div_code": period_code,
            "fid_org_adj_prc": "0",
        }
        return await self._handle_request(
            session.get, url, headers=self._get_headers("FHKUP03500100"), params=params
        )

    async def get_investor_trend_estimate(self, session, code):
        """외인/기관 추정가집계 (HHPTJ04160200)"""
        url = (
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/investor-trend-estimate"
        )
        params = {"MKSC_SHRN_ISCD": code}
        return await self._handle_request(
            session.get, url, headers=self._get_headers("HHPTJ04160200"), params=params
        )

    async def get_trade_strength(self, session, code, market_div_code=None):
        """종목별 체결강도 조회 (FHKST01010300)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-ccnl"
        params = {"FID_INPUT_ISCD": code}
        preferred_market_div = market_div_code or self._market_div_cache.get(code)
        res, used_market_div = await self._request_with_market_div_fallback(
            session=session,
            url=url,
            tr_id="FHKST01010300",
            params=params,
            market_div_param_key="FID_COND_MRKT_DIV_CODE",
            preferred_market_div_code=preferred_market_div,
        )
        if res.get("rt_cd") == "0" and used_market_div:
            self._market_div_cache[code] = used_market_div
        return res

    async def get_condition_list(self, session):
        """내 조건 목록 가져오기 (HHKST03900300)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/psearch-title"
        params = {"user_id": self.hts_id}
        return await self._handle_request(
            session.get, url, headers=self._get_headers("HHKST03900300"), params=params
        )

    async def get_condition_result(self, session, seq):
        """조건검색 결과 조회 (HHKST03900400)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/psearch-result"
        params = {"user_id": self.hts_id, "seq": seq}
        return await self._handle_request(
            session.get, url, headers=self._get_headers("HHKST03900400"), params=params
        )

    async def get_stock_ohlcv_history(
        self,
        session,
        stock_code,
        start_date,
        end_date,
        period_code="D",
        adj_price="0",
        market_div_code=None,
    ):
        """국내주식 기간별 시세 조회 (FHKST03010100)
        
        Args:
            session: aiohttp ClientSession
            stock_code: 종목코드 (6자리)
            start_date: 시작일자 (YYYYMMDD)
            end_date: 종료일자 (YYYYMMDD)
            period_code: D(일), W(주), M(월), Y(년)
            adj_price: 수정주가 반영여부 (0: 미반영, 1: 반영)
        
        Returns:
            dict: API 응답 데이터
                - rt_cd: 성공여부 ("0": 성공)
                - output2: 시세 데이터 리스트
                    - stck_bsop_date: 주식영업일자
                    - stck_oprc: 시가
                    - stck_hgpr: 고가
                    - stck_lwpr: 저가
                    - stck_clpr: 종가
                    - acml_vol: 누적거래량

        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        params = {
            "fid_input_iscd": stock_code,
            "fid_input_date_1": start_date,
            "fid_input_date_2": end_date,
            "fid_period_div_code": period_code,
            "fid_org_adj_prc": adj_price,
        }
        preferred_market_div = market_div_code or self._market_div_cache.get(stock_code)
        res, used_market_div = await self._request_with_market_div_fallback(
            session=session,
            url=url,
            tr_id="FHKST03010100",
            params=params,
            market_div_param_key="fid_cond_mrkt_div_code",
            preferred_market_div_code=preferred_market_div,
            require_non_empty_output2=True,
        )
        if res.get("rt_cd") == "0" and used_market_div:
            self._market_div_cache[stock_code] = used_market_div
        return res
