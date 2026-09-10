"""KRX 거래일 판정 (KRX 공식 지수 일별매매정보 기반)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import settings
from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.krx_api import (
    KRX_ENDPOINT_KOSPI_INDEX_DAILY,
    fetch_krx_openapi_day_strict,
)

_TRADING_DAY_CACHE: dict[str, bool] = {}


def is_krx_trading_day(date: pd.Timestamp | str, cfg: AltDataFetchConfig | None = None) -> bool:
    """KRX 공식 응답 행수>0을 거래일 권위 판정으로 사용합니다.

    지수 일별매매정보(51행/0.15s)의 행 존재 여부를 판정 근거로 씁니다.
    네트워크/인증 장애는 ``False`` 로 삼키지 않고 그대로 전파합니다.

    Args:
        date: 판정 대상일.
        cfg: Alt-data 설정. ``None`` 이면 기본 설정을 생성합니다.

    Returns:
        거래일이면 ``True``, 휴장일(0행)이면 ``False``.
    """
    key = pd.Timestamp(date).strftime("%Y%m%d")
    if key in _TRADING_DAY_CACHE:
        return _TRADING_DAY_CACHE[key]
    if cfg is None:
        # 이 경로는 cfg의 API 접근 필드(krx_api_key/레이트리밋)만 사용한다.
        # start/end/out_dir는 수집 창 설정이라 여기선 의미가 없지만 필수 인자라 채운다.
        # krx_api_key를 빠뜨리면 strict 페처가 ValueError로 죽으므로 settings에서 주입한다.
        target = pd.Timestamp(date).normalize()
        cfg = AltDataFetchConfig(
            start=target,
            end=target + pd.Timedelta(days=1),
            out_dir=Path("."),
            krx_api_key=settings.KRX_OPENAPI_KEY,
        )
    rows = fetch_krx_openapi_day_strict(KRX_ENDPOINT_KOSPI_INDEX_DAILY, key, cfg)
    result = len(rows) > 0
    _TRADING_DAY_CACHE[key] = result
    return result
