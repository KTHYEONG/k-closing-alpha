import asyncio
import json
import logging
import os
from datetime import datetime, timedelta

import aiohttp

from src import settings

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """한국투자증권 REST API 요청용 비동기 레이트 리미터 (슬라이딩 윈도우)."""

    def __init__(self, max_rate: float = 18.0, time_period: float = 1.0):
        self.max_rate = max_rate
        self.time_period = time_period
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Rate limit 초과 시 대기 후 권한 획득."""
        import time
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self.time_period]
            if len(self._timestamps) >= self.max_rate:
                sleep_time = self.time_period - (now - self._timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < self.time_period]
            self._timestamps.append(now)


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

    def create_session(self) -> aiohttp.ClientSession:
        """최적화된 커넥터를 가진 세션을 생성합니다."""
        from aiohttp.resolver import ThreadedResolver
        resolver = ThreadedResolver()
        connector = aiohttp.TCPConnector(
            limit=50,            # 동시 연결 수 제한
            ttl_dns_cache=300,  # DNS 캐시 유지 시간
            use_dns_cache=True,  # DNS 캐시 사용
            resolver=resolver   # DNS 해석기 추가
        )
        return aiohttp.ClientSession(connector=connector)

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

    async def ensure_token(self, session: aiohttp.ClientSession):
        """토큰 유효성을 확인하고 필요시 갱신합니다."""
        if os.path.exists(self.token_file):
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
        """재시도 로직을 포함한 공통 요청 처리 (네트워크 에러 처리 강화)"""
        import aiohttp
        
        await self.rate_limiter.acquire()
        for attempt in range(5):
            try:
                async with session_method(url, **kwargs) as resp:
                    if resp.status == 429:  # Too Many Requests
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue

                    data = await resp.json()
                    # KIS 특유의 TPS 초과 메시지 처리
                    if data.get("rt_cd") != "0" and "초당 거래건수" in data.get("msg1", ""):
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


async def fetch_index_and_calculate_volatility(index_code="1028", session=None):
    """지수 코드를 받아 최근 데이터를 가져와 역사적 변동성(HV)을 계산합니다.
    기본값 1028은 KOSPI 200입니다. KOSDAQ 150은 2203(예상)입니다.
    
    Returns:
        tuple: (hv_today, hv_change)

    """
    from datetime import datetime, timedelta

    import aiohttp
    import numpy as np
    import pandas as pd
    
    client = KisApiClient()
    
    # 최근 30일 데이터 (영업일 기준 약 21일)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=34)).strftime("%Y%m%d") # 여유있게 34일로 늘림
    
    # 세션 관리: 전달받은 세션이 있으면 사용, 없으면 생성
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
        
    try:
        await client.ensure_token(session)
        
        resp = await client.get_market_index_history(
            session, index_code, start_date, end_date
        )
        
        if resp.get('rt_cd') == '0':
            items = resp.get('output2', [])
            
            if len(items) >= 2:
                # 데이터 정리 및 정렬
                records = []
                for item in items:
                    date = item.get('stck_bsop_date')
                    close = float(item.get('bstp_nmix_prpr') or 0)
                    if date and close > 0:
                        records.append({'date': date, 'close': close})
                
                df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
                
                if len(df) >= 2:
                    # 로그 수익률 계산
                    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
                    
                    # 최근 20일 표준편차 (마지막 행 기준)
                    if len(df) >= 21:
                        recent_returns = df['log_ret'].iloc[-20:]
                    else:
                        recent_returns = df['log_ret'].dropna()
                    
                    std = recent_returns.std()
                    
                    # 연율화 HV
                    hv_today = std * np.sqrt(252) * 100
                    
                    # 어제 HV 계산 (전일 대비 변화율용)
                    if len(df) >= 22:
                        prev_returns = df['log_ret'].iloc[-21:-1]
                        prev_std = prev_returns.std()
                        hv_yesterday = prev_std * np.sqrt(252) * 100
                        hv_change = (hv_today - hv_yesterday) / hv_yesterday if hv_yesterday != 0 else 0
                    else:
                        hv_change = 0
                    
                    return hv_today, hv_change
        
        return 0.0, 0.0

    finally:
        if local_session:
            await session.close()

async def fetch_kospi200_and_calculate_vkospi():
    """KOSPI 200(1028) 기반 V-KOSPI 계산 (Legacy Wrapper)"""
    return await fetch_index_and_calculate_volatility("1028")


async def calculate_stock_sma(stock_code, sma_period=120, lookback_days=200, session=None):
    """특정 종목의 SMA (단순이동평균)를 계산합니다.
    
    한국투자증권 API는 한 번에 약 100일의 데이터만 반환하므로,
    충분한 데이터를 확보하기 위해 여러 번 호출합니다.
    """
    from datetime import datetime, timedelta

    import aiohttp
    import pandas as pd
    
    client = KisApiClient()
    all_records = []
    
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
    
    try:
        await client.ensure_token(session)
        
        # 여러 번 API 호출하여 충분한 데이터 확보
        # 한 번에 약 100일씩, 최대 3번 호출 (총 300일치)
        for chunk in range(3):
            # 각 청크의 종료일과 시작일 계산
            end_date = (datetime.now() - timedelta(days=100 * chunk)).strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=100 * (chunk + 1) + 50)).strftime("%Y%m%d")
            
            # API 호출
            resp = await client.get_stock_ohlcv_history(
                session, stock_code, start_date, end_date
            )
            
            if resp.get('rt_cd') != '0':
                if chunk == 0:
                    # 첫 번째 호출 실패면 전체 실패
                    logger.warning(
                        "[SMA Debug] %s 일봉 조회 실패: rt_cd=%s, msg=%s, range=%s~%s, chunk=%s",
                        stock_code,
                        resp.get('rt_cd'),
                        resp.get('msg1', 'N/A'),
                        start_date,
                        end_date,
                        chunk,
                    )
                    return 0.0, False
                else:
                    # 이후 호출 실패는 무시 (이미 충분한 데이터가 있을 수 있음)
                    break
            
            items = resp.get('output2', [])
            
            # 데이터 파싱
            for item in items:
                date = item.get('stck_bsop_date')
                close = float(item.get('stck_clpr') or 0)
                if date and close > 0:
                    all_records.append({'date': date, 'close': close})
            
            # 충분한 데이터를 확보했으면 중단
            if len(all_records) >= sma_period + 10:  # 여유분 10일 추가
                break
            
            # API 부하 방지를 위한 짧은 대기
            await asyncio.sleep(0.1)
        
        if len(all_records) < sma_period:
            # 데이터가 부족하면 실패 (상장 초기 종목 등)
            return 0.0, False
        
        # 데이터프레임 생성 및 정렬
        df = pd.DataFrame(all_records)
        # 중복 제거 (날짜 기준)
        df = df.drop_duplicates(subset=['date'], keep='first')
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < sma_period:
            return 0.0, False
        
        # SMA 계산 (최근 N일의 평균)
        sma_value = df['close'].tail(sma_period).mean()
        
        return float(sma_value), True
            
    except Exception as e:
        logger.warning(
            "[SMA Debug] %s SMA 계산 예외: %s: %s", stock_code, type(e).__name__, e
        )
        return 0.0, False
    finally:
        if local_session:
            await session.close()


async def calculate_stock_ema(stock_code, ema_period=20, lookback_days=60, session=None):
    """특정 종목의 EMA (지수이동평균)를 계산합니다."""
    from datetime import datetime, timedelta

    import aiohttp
    import pandas as pd
    
    client = KisApiClient()
    
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
    
    try:
        await client.ensure_token(session)
        
        # 과거 데이터 조회
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        
        # API 호출
        resp = await client.get_stock_ohlcv_history(
            session, stock_code, start_date, end_date
        )
        
        if resp.get('rt_cd') != '0':
            logger.warning(
                "[EMA Debug] %s 일봉 조회 실패: rt_cd=%s, msg=%s, range=%s~%s",
                stock_code,
                resp.get('rt_cd'),
                resp.get('msg1', 'N/A'),
                start_date,
                end_date,
            )
            return 0.0, False, 0
        
        items = resp.get('output2', [])
        if not items:
            logger.warning(
                "[EMA Debug] %s 일봉 응답이 비어 있음: rt_cd=%s, msg=%s, range=%s~%s",
                stock_code,
                resp.get('rt_cd'),
                resp.get('msg1', 'N/A'),
                start_date,
                end_date,
            )
        
        # 데이터 파싱
        records = []
        for item in items:
            date = item.get('stck_bsop_date')
            close = float(item.get('stck_clpr') or 0)
            if date and close > 0:
                records.append({'date': date, 'close': close})
        
        if len(records) < ema_period:
            # 데이터가 부족하면 실패 (상장 초기 종목)
            return 0.0, False, len(records)
        
        # 데이터프레임 생성 및 정렬
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='first')
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < ema_period:
            return 0.0, False, len(df)
        
        # EMA 계산 (지수이동평균)
        ema_value = df['close'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
        
        return float(ema_value), True, len(df)
            
    except Exception as e:
        logger.warning(
            "[EMA Debug] %s EMA 계산 예외: %s: %s", stock_code, type(e).__name__, e
        )
        return 0.0, False, 0
    finally:
        if local_session:
            await session.close()


async def calculate_multiple_emas(stock_code, periods=[5, 10, 20], lookback_days=120, session=None):
    """한 번의 데이터 조회로 여러 EMA를 계산합니다.
    """
    from datetime import datetime, timedelta

    import aiohttp
    import pandas as pd
    
    client = KisApiClient()
    results = {}
    
    local_session = False
    if session is None:
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True
    
    try:
        await client.ensure_token(session)
        
        # 충분한 과거 데이터 조회
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        
        resp = await client.get_stock_ohlcv_history(
            session, stock_code, start_date, end_date
        )
        
        if resp.get('rt_cd') != '0':
            return {}
        
        items = resp.get('output2', [])
        records = []
        for item in items:
            date = item.get('stck_bsop_date')
            close = float(item.get('stck_clpr') or 0)
            if date and close > 0:
                records.append({'date': date, 'close': close})
        
        if not records:
            return {}
            
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['date'], keep='first')
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < max(periods):
            return {}
        
        for period in periods:
            if len(df) >= period:
                val = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
                results[period] = round(float(val), 2)
            else:
                results[period] = 0.0
                
        return results
            
    except Exception:
        return {}
    finally:
        if local_session:
            await session.close()


async def calculate_all_moving_averages(stock_code, session=None):
    """한 종목의 여러 이동평균(EMA 5/10/20, SMA 60/120)을 최소한의 API 호출로 통합 계산함.
    TPS 부하를 줄이기 위해 중복되는 OHLCV 데이터를 한 번에 가져와서 메모리에서 계산함.
    """
    import asyncio
    from datetime import datetime, timedelta

    import pandas as pd

    client = KisApiClient()
    all_records = []
    local_session = False

    if session is None:
        import aiohttp
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        session = aiohttp.ClientSession(connector=connector)
        local_session = True

    try:
        await client.ensure_token(session)

        # SMA 120까지 계산하기 위해 충분한 데이터 확보
        for chunk in range(2):
            end_dt = datetime.now() - timedelta(days=100 * chunk)
            start_dt = end_dt - timedelta(days=120)
            end_date = end_dt.strftime("%Y%m%d")
            start_date = start_dt.strftime("%Y%m%d")

            resp = await client.get_stock_ohlcv_history(session, stock_code, start_date, end_date)

            if resp.get("rt_cd") != "0":
                if chunk == 0:
                    return {}, (0.0, False, 0), (0.0, False), (0.0, False)
                break

            items = resp.get("output2", [])
            for item in items:
                date = item.get("stck_bsop_date")
                close = float(item.get("stck_clpr") or 0)
                if date and close > 0:
                    all_records.append({"date": date, "close": close})

            if len(all_records) >= 150:
                break

            if chunk < 1:
                await asyncio.sleep(0.05)

        if not all_records:
            return {}, (0.0, False, 0), (0.0, False), (0.0, False)

        df = pd.DataFrame(all_records).drop_duplicates(subset=["date"], keep="first").sort_values("date").reset_index(drop=True)
        data_count = len(df)

        ema_res = {}
        for period in [5, 10, 20]:
            if data_count >= period:
                val = df["close"].ewm(span=period, adjust=False).mean().iloc[-1]
                ema_res[period] = round(float(val), 2)
            else:
                ema_res[period] = 0.0

        ema20_val = ema_res.get(20, 0.0)
        ema_success = data_count >= 20

        sma60_val = round(float(df["close"].tail(60).mean()), 2) if data_count >= 60 else 0.0
        sma60_ok = data_count >= 60

        sma120_val = round(float(df["close"].tail(120).mean()), 2) if data_count >= 120 else 0.0
        sma120_ok = data_count >= 120

        return ema_res, (ema20_val, ema_success, data_count), (sma60_val, sma60_ok), (sma120_val, sma120_ok)

    except Exception as e:
        logger.warning(
            "[MA Calc Error] %s: %s: %s", stock_code, type(e).__name__, e
        )
        return {}, (0.0, False, 0), (0.0, False), (0.0, False)
    finally:
        if local_session:
            await session.close()
