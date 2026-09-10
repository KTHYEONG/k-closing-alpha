"""키움증권 REST API 클라이언트 (HTTP/토큰/요청 동작, TR별 5req/s 서버 강제 리밋)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os

from src import settings
from src.api.kis.rate_limit import AsyncRateLimiter, get_shared_rate_limiter

logger = logging.getLogger(__name__)

# 서버 실측 확인 (scratch/probe_kiwoom_rate_limit_concurrency.json): 초과 시 HTTP 429 +
# 응답 바디에 "유량=5, API ID=<api_id>" 명시. api-id(TR)별로 독립된 버킷이 부과된다.
_KIWOOM_TR_RATE_PER_SEC = 5.0


class KiwoomApiClient:
    def __init__(self, app_key: str | None = None, secret_key: str | None = None, base_url: str | None = None) -> None:
        self.app_key = app_key or getattr(settings, "KIWOM_APP_KEY", "") or os.getenv("KIWOM_APP_KEY", "")
        self.secret_key = secret_key or getattr(settings, "KIWOM_SECRET_KEY", "") or os.getenv("KIWOM_SECRET_KEY", "")
        self.base_url = base_url or getattr(settings, "KIWOM_BASE_URL", "") or "https://api.kiwoom.com"
        self.token: str | None = None
        self._rate_limiters: dict[str, AsyncRateLimiter] = {}
        self._token_lock: asyncio.Lock | None = None

    def _limiter_for(self, api_id: str) -> AsyncRateLimiter:
        return get_shared_rate_limiter("kiwoom", f"{self.app_key}:{api_id}", _KIWOOM_TR_RATE_PER_SEC)

    async def ensure_token(self, session) -> str:
        if self.token:
            return self.token
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        async with self._token_lock:
            if self.token:
                return self.token
            payload = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.secret_key}
            raw = session.post(
                f"{self.base_url}/oauth2/token",
                headers={"Content-Type": "application/json;charset=UTF-8"},
                json=payload,
            )
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                body = await resp.json()
            token = str(body.get("token", ""))
            if not token:
                raise RuntimeError(f"Kiwoom token issuance failed: {body}")
            self.token = token
            return token

    async def _post_tr(
        self,
        session,
        api_id: str,
        path: str,
        body: dict,
        cont_yn: str = "N",
        next_key: str = "",
        max_retries: int = 3,
    ) -> tuple[dict, dict]:
        if not self.token:
            await self.ensure_token(session)
        limiter = self._limiter_for(api_id)

        for attempt in range(max_retries):
            await limiter.acquire()
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "authorization": f"Bearer {self.token}",
                "api-id": api_id,
                "cont-yn": cont_yn,
                "next-key": next_key,
            }
            raw = session.post(f"{self.base_url}{path}", headers=headers, json=body)
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                data = await resp.json()
                status = resp.status
                headers_raw = getattr(resp, "headers", None)
                if isinstance(headers_raw, dict):
                    resp_headers = headers_raw
                elif hasattr(headers_raw, "items") and not type(headers_raw).__name__.endswith("Mock"):
                    try:
                        resp_headers = dict(headers_raw)
                    except Exception:
                        resp_headers = {}
                else:
                    resp_headers = {}

            if status == 429 and attempt < max_retries - 1:
                logger.warning("Kiwoom rate limit hit (429) api_id=%s. Retrying in 1.2s... (attempt %d/%d)", api_id, attempt + 1, max_retries)
                await asyncio.sleep(1.2)
                continue
            return data, resp_headers
        return data, resp_headers

    async def get_fluctuation_ranking(self, session, *, rate_min_pct: float, rate_max_pct: float, market_type: str = "000", max_pages: int = 5, stex_tp: str = "3") -> dict:
        cont_yn, next_key = "N", ""
        collected: list[dict] = []
        try:
            for _ in range(max(1, int(max_pages))):
                data, resp_headers = await self._post_tr(
                    session, "ka10027", "/api/dostk/rkinfo",
                    {
                        "mrkt_tp": str(market_type),
                        "stex_tp": str(stex_tp),
                        "sort_tp": "1",
                        "trde_qty_cnd": "0000",
                        "stk_cnd": "0",
                        "crd_cnd": "0",
                        "updown_incls": "0",
                        "pric_cnd": "0",
                        "trde_prica_cnd": "0",
                    },
                    cont_yn=cont_yn, next_key=next_key,
                )
                if data.get("return_code") != 0:
                    return {"rt_cd": "1", "msg1": str(data.get("return_msg", "")), "output": [], "vendor": "kiwoom"}
                rows = data.get("pred_pre_flu_rt_upper") or []
                page_rates: list[float] = []
                for r in rows:
                    try:
                        rate = float(str(r.get("flu_rt", "")).strip())
                    except (ValueError, TypeError):
                        continue
                    page_rates.append(rate)
                    if rate < float(rate_min_pct) or rate > float(rate_max_pct):
                        continue
                    collected.append(dict(r))
                cont_yn = str(resp_headers.get("cont-yn", "N") or "N")
                next_key = str(resp_headers.get("next-key", "") or "")
                if cont_yn != "Y":
                    break
                if page_rates and min(page_rates) < float(rate_min_pct):
                    break
        except Exception as e:
            logger.warning("Kiwoom fluctuation ranking failed: %s", e)
            return {"rt_cd": "1", "msg1": str(e), "output": [], "vendor": "kiwoom"}
        return {"rt_cd": "0", "output": collected, "vendor": "kiwoom"}

    async def get_tick_chart(self, session, code: str, target_date: str, max_pages: int | None = None) -> dict:
        ymd = str(target_date).replace("-", "")
        page_budget = int(max_pages) if max_pages is not None else int(getattr(settings, "KIWOM_TICK_MAX_PAGES", 30) or 30)
        cont_yn, next_key = "N", ""
        all_rows: list[dict] = []
        reached_open = False
        try:
            for _ in range(max(1, page_budget)):
                data, resp_headers = await self._post_tr(
                    session, "ka10079", "/api/dostk/chart",
                    {"stk_cd": str(code), "tic_scope": "1", "upd_stkpc_tp": "1", "base_dt": ymd},
                    cont_yn=cont_yn, next_key=next_key,
                )
                if data.get("return_code") != 0:
                    return {"rt_cd": "1", "msg1": str(data.get("return_msg", "")), "output2": [], "vendor": "kiwoom", "truncated": False}
                rows = data.get("stk_tic_chart_qry") or []
                if rows:
                    all_rows.extend(rows)
                cont_yn = str(resp_headers.get("cont-yn", "N") or "N")
                next_key = str(resp_headers.get("next-key", "") or "")
                if cont_yn != "Y":
                    reached_open = True
                    break
                # 페이지 내 행은 시간 역순(최신->과거)이므로 마지막 행이 그 페이지의 최고령 행이다.
                oldest = str(rows[-1].get("cntr_tm", "")) if rows else ""
                if oldest and oldest[:8] < ymd:
                    reached_open = True
                    break
        except Exception as e:
            logger.warning("Kiwoom tick chart failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "kiwoom", "truncated": False}
        # base_dt로 앵커링된 페이지가 여러 날에 걸칠 수 있음을 실측 확인 -- target_date로 최종 필터.
        filtered = [dict(r) for r in all_rows if str(r.get("cntr_tm", "")).startswith(ymd)]
        truncated = not reached_open
        if truncated:
            logger.warning("[DATA] Kiwoom tick page budget exhausted code=%s pages=%d", code, page_budget)
        return {"rt_cd": "0", "output2": filtered, "vendor": "kiwoom", "truncated": truncated}

    async def get_nxt_premarket_chart(self, session, code: str, target_date: str) -> dict:
        ymd = str(target_date).replace("-", "")
        nx_code = f"{str(code).split('_')[0].zfill(6)}_NX"
        try:
            data, _headers = await self._post_tr(
                session, "ka10080", "/api/dostk/chart",
                {"stk_cd": nx_code, "base_dt": ymd, "tic_scope": "1", "upd_stkpc_tp": "0"},
            )
        except Exception as e:
            logger.warning("Kiwoom NXT premarket chart failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "kiwoom"}
        if data.get("return_code") != 0:
            return {"rt_cd": "1", "msg1": str(data.get("return_msg", "")), "output2": [], "vendor": "kiwoom"}
        rows = data.get("stk_min_pole_chart_qry") or []
        kept: list[dict] = []
        for r in rows:
            cntr_tm = str(r.get("cntr_tm", ""))
            if not cntr_tm.startswith(ymd):
                continue
            try:
                hms = int(cntr_tm[8:14])
            except (ValueError, TypeError):
                continue
            if hms < 80000 or hms > 85000:
                continue
            kept.append(dict(r))
        return {"rt_cd": "0", "output2": kept, "vendor": "kiwoom"}

    async def get_nxt_minute_chart(self, session, code: str, target_date: str) -> dict:
        ymd = str(target_date).replace("-", "")
        nx_code = f"{str(code).split('_')[0].zfill(6)}_NX"
        try:
            data, _headers = await self._post_tr(
                session, "ka10080", "/api/dostk/chart",
                {"stk_cd": nx_code, "base_dt": ymd, "tic_scope": "1", "upd_stkpc_tp": "0"},
            )
        except Exception as e:
            logger.warning("Kiwoom NXT minute chart failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "kiwoom"}
        if data.get("return_code") != 0:
            return {"rt_cd": "1", "msg1": str(data.get("return_msg", "")), "output2": [], "vendor": "kiwoom"}
        rows = data.get("stk_min_pole_chart_qry") or []
        kept: list[dict] = []
        for r in rows:
            cntr_tm = str(r.get("cntr_tm", ""))
            if not cntr_tm.startswith(ymd):
                continue
            try:
                hms = int(cntr_tm[8:14])
            except (ValueError, TypeError):
                continue
            if hms < 154000 or hms > 200000:
                continue
            kept.append(dict(r))
        return {"rt_cd": "0", "output2": kept, "vendor": "kiwoom"}
