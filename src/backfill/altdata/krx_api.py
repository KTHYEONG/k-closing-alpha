"""KRX Open API (data-dbg.krx.co.kr) 일별 조회 클라이언트.

``AUTH_KEY`` 헤더 인증. 파생상품/지수 일별매매정보의 주 경로이며, pykrx 가
KRX 안티봇으로 차단된 환경에서도 동작한다. 구독되지 않은 엔드포인트는
401(``Unauthorized API Call``), 오타 경로는 404 를 반환하며 두 경우 모두
빈 DataFrame 으로 fail-soft 처리한다 (호출부는 pykrx fallback 으로 진행).
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.ratelimit import retry_call, wait_for_krx_slot

logger = logging.getLogger(__name__)

_BASE_URL = "https://data-dbg.krx.co.kr"

# 구독 확인된 엔드포인트 (2026-09 기준). 미구독 경로는 401 로 fail-soft.
KRX_ENDPOINT_FUT_DAILY = "/svc/apis/drv/fut_bydd_trd"
KRX_ENDPOINT_KOSPI_INDEX_DAILY = "/svc/apis/idx/kospi_dd_trd"
KRX_ENDPOINT_STK_BASE_INFO = "/svc/apis/sto/stk_isu_base_info"
KRX_ENDPOINT_KSQ_BASE_INFO = "/svc/apis/sto/ksq_isu_base_info"
KRX_ENDPOINT_STK_DAILY = "/svc/apis/sto/stk_bydd_trd"


def fetch_krx_openapi_day(
    endpoint: str, date_ymd: str, cfg: AltDataFetchConfig
) -> pd.DataFrame:
    """단일 기준일(``basDd=YYYYMMDD``) KRX Open API 응답을 DataFrame 으로 반환합니다.

    Args:
        endpoint: ``/svc/apis/...`` 경로.
        date_ymd: 기준일 (``YYYYMMDD``).
        cfg: Alt-data 설정 (``krx_api_key`` 필수).

    Returns:
        ``OutBlock_1`` 행들의 DataFrame. 키 미설정·401·404·비200·JSON 오류·빈
        응답은 모두 빈 DataFrame.
    """
    key = str(cfg.krx_api_key).strip()
    if not key:
        logger.warning("[DATA] stage=altdata_krx status=SKIP reason=no_key endpoint=%s", endpoint)
        return pd.DataFrame()

    url = f"{_BASE_URL}{endpoint}"

    def _call() -> pd.DataFrame:
        wait_for_krx_slot(cfg)
        resp = requests.get(
            url, params={"basDd": date_ymd}, headers={"AUTH_KEY": key}, timeout=20
        )
        if resp.status_code in (401, 404):
            # 미구독/오타 경로 — 재시도 무의미, 빈 프레임으로 확정.
            logger.warning(
                "[DATA] stage=altdata_krx status=UNAVAILABLE code=%s endpoint=%s",
                resp.status_code,
                endpoint,
            )
            return pd.DataFrame()
        if resp.status_code != 200:
            raise RuntimeError(f"krx http_status={resp.status_code}")
        payload = resp.json()
        if not isinstance(payload, dict):
            return pd.DataFrame()
        block = next((k for k in payload if k.startswith("OutBlock")), None)
        rows = payload.get(block, []) if block else []
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    out = retry_call(_call, cfg, label=f"krx {endpoint} {date_ymd}")
    return out if out is not None else pd.DataFrame()


def fetch_krx_openapi_day_strict(
    endpoint: str, date_ymd: str, cfg: AltDataFetchConfig
) -> pd.DataFrame:
    """단일 기준일 KRX Open API 응답을 fail-closed 계약으로 반환합니다.

    기존 :func:`fetch_krx_openapi_day` 의 fail-soft 와 정반대 계약으로,
    폴백 없는 설계에서 조용한 결손을 막기 위해 401/404/비200 을 절대 빈
    DataFrame 으로 삼키지 않고 :class:`RuntimeError` 로 드러냅니다.
    휴장일(200 + 0행)은 정상 응답이므로 빈 DataFrame 을 반환합니다.

    Args:
        endpoint: ``/svc/apis/...`` 경로.
        date_ymd: 기준일 (``YYYYMMDD``).
        cfg: Alt-data 설정 (``krx_api_key`` 필수).

    Returns:
        ``OutBlock_1`` 행들의 DataFrame. 0행이면 휴장일의 정상 빈 프레임.

    Raises:
        ValueError: ``krx_api_key`` 가 비어 있을 때.
        RuntimeError: HTTP 401/404/기타 비200 및 재시도 전체 실패 시.
    """
    key = str(cfg.krx_api_key).strip()
    if not key:
        raise ValueError("krx api key is required for strict fetch")
    url = f"{_BASE_URL}{endpoint}"

    def _call() -> pd.DataFrame:
        wait_for_krx_slot(cfg)
        resp = requests.get(
            url, params={"basDd": date_ymd}, headers={"AUTH_KEY": key}, timeout=20
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"krx http_status={resp.status_code} endpoint={endpoint} date={date_ymd}"
            )
        payload = resp.json()
        block = next((k for k in payload if k.startswith("OutBlock")), None)
        rows = payload.get(block, []) if block else []
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    out = retry_call(_call, cfg, label=f"krx {endpoint} {date_ymd}")
    if out is None:
        raise RuntimeError(
            f"krx http_status=unknown endpoint={endpoint} date={date_ymd} (all retries failed)"
        )
    return out
