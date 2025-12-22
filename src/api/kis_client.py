import aiohttp
import asyncio
import json
import os
from datetime import datetime, timedelta


from src import settings

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

    async def ensure_token(self, session: aiohttp.ClientSession):
        """토큰 유효성을 확인하고 필요시 갱신합니다."""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r", encoding="utf-8") as f:
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
        """재시도 로직을 포함한 공통 요청 처리"""
        for attempt in range(5):
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
        return {"rt_cd": "9", "msg1": "최대 재시도 횟수 초과 (TPS 제한)"}

    async def get_current_price(self, session, code):
        """주식 현재가 시세 조회 (FHKST01010100)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": code}
        return await self._handle_request(
            session.get, url, headers=self._get_headers("FHKST01010100"), params=params
        )

    async def get_program_net_buy(self, session, code):
        """종목별 프로그램 매매 추이 (FHPPG04650101)"""
        url = (
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/program-trade-by-stock"
        )
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
        return await self._handle_request(
            session.get, url, headers=self._get_headers("FHPPG04650101"), params=params
        )

    async def get_market_index_rate(self, session, market_code):
        """시장 지수 등락률 조회 (FHKUP03500100)"""
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

    async def get_investor_trend_estimate(self, session, code):
        """외인/기관 추정가집계 (HHPTJ04160200)"""
        url = (
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/investor-trend-estimate"
        )
        params = {"MKSC_SHRN_ISCD": code}
        return await self._handle_request(
            session.get, url, headers=self._get_headers("HHPTJ04160200"), params=params
        )

    async def get_trade_strength(self, session, code):
        """종목별 체결강도 조회 (FHKST01010300)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-ccnl"
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
        return await self._handle_request(
            session.get, url, headers=self._get_headers("FHKST01010300"), params=params
        )

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
