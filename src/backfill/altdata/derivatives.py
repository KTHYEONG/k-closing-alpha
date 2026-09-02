"""KOSPI200 지수-선물 베이시스 수집기 (KRX Open API 주 경로 + pykrx fallback)."""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.krx_api import KRX_ENDPOINT_FUT_DAILY, fetch_krx_openapi_day
from src.backfill.altdata.ratelimit import retry_call, wait_for_pykrx_slot

logger = logging.getLogger(__name__)

try:
    from pykrx import stock
except ImportError:  # pragma: no cover
    stock = None  # type: ignore[assignment]

_COLS = [
    "date",
    "kospi200_close",
    "k200_future_close",
    "basis",
    "basis_pct",
    "future_volume",
    "future_open_interest",
]

# KRX drv/fut_bydd_trd 의 코스피200 정규 선물 (미니/위클리/스프레드 제외).
_K200_PROD = "코스피200 선물"
_EXPIRY_RE = re.compile(r"(\d{6})")


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _krx_front_month_row(raw: pd.DataFrame, ymd: str) -> dict[str, float] | None:
    """KRX 선물 일별매매 응답에서 코스피200 최근월물 1행을 고릅니다."""
    if raw is None or raw.empty or "PROD_NM" not in raw.columns:
        return None
    work = raw[raw["PROD_NM"].astype(str).str.strip() == _K200_PROD].copy()
    if "MKT_NM" in work.columns:
        work = work[work["MKT_NM"].astype(str).str.contains("정규", na=False)]
    # 스프레드(SP) 종목 제외
    if "ISU_NM" in work.columns:
        work = work[~work["ISU_NM"].astype(str).str.contains(" SP ", na=False)]
    if work.empty:
        return None
    # 만기(YYYYMM) 파싱 → 기준일 이후 최근월물
    ym = work["ISU_NM"].astype(str).str.extract(_EXPIRY_RE, expand=False)
    work = work.assign(_ym=pd.to_numeric(ym, errors="coerce"))
    cur_ym = int(ymd[:6])
    fwd = work[work["_ym"] >= cur_ym]
    picked = (fwd if not fwd.empty else work).sort_values("_ym").iloc[0]

    spot = pd.to_numeric(picked.get("SPOT_PRC"), errors="coerce")
    fut = pd.to_numeric(picked.get("TDD_CLSPRC"), errors="coerce")
    vol = pd.to_numeric(picked.get("ACC_TRDVOL"), errors="coerce")
    oi = pd.to_numeric(picked.get("ACC_OPNINT_QTY"), errors="coerce")
    basis = float(fut - spot) if pd.notna(fut) and pd.notna(spot) else float("nan")
    basis_pct = (
        float(basis / spot) if pd.notna(spot) and float(spot) != 0.0 and pd.notna(basis) else float("nan")
    )
    return {
        "kospi200_close": float(spot) if pd.notna(spot) else float("nan"),
        "k200_future_close": float(fut) if pd.notna(fut) else float("nan"),
        "basis": basis,
        "basis_pct": basis_pct,
        "future_volume": float(vol) if pd.notna(vol) else float("nan"),
        "future_open_interest": float(oi) if pd.notna(oi) else float("nan"),
    }


def _collect_via_krx(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in business_days:
        ymd = _to_ymd(day)
        raw = fetch_krx_openapi_day(KRX_ENDPOINT_FUT_DAILY, ymd, cfg)
        picked = _krx_front_month_row(raw, ymd)
        if picked is None:
            continue
        rows.append({"date": pd.Timestamp(day).normalize(), **picked})
    return pd.DataFrame(rows, columns=_COLS)


def _collect_via_pykrx(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """pykrx fallback (환경에 따라 KRX 차단으로 빈 결과일 수 있음)."""
    if stock is None or not business_days:
        return pd.DataFrame(columns=_COLS)

    def _spot_call() -> pd.DataFrame:
        wait_for_pykrx_slot(cfg)
        return stock.get_index_ohlcv(_to_ymd(business_days[0]), _to_ymd(business_days[-1]), "1028")

    spot_df = retry_call(_spot_call, cfg, label="derivatives spot 1028")
    spot_map: dict[pd.Timestamp, float] = {}
    if spot_df is not None and not spot_df.empty:
        idx = pd.to_datetime(spot_df.index, errors="coerce")
        close_col = next((c for c in ("종가", "Close", "close") if c in spot_df.columns), None)
        if close_col is not None:
            for d, v in zip(idx, spot_df[close_col], strict=False):
                spot_map[pd.Timestamp(d).normalize()] = float(pd.to_numeric(v, errors="coerce"))

    rows: list[dict[str, object]] = []
    for day in business_days:
        d_norm = pd.Timestamp(day).normalize()
        spot = spot_map.get(d_norm, float("nan"))

        def _fut_call(_ymd: str = _to_ymd(day)) -> pd.DataFrame:
            wait_for_pykrx_slot(cfg)
            return stock.get_future_ohlcv_by_ticker(_ymd)

        fut_df = retry_call(_fut_call, cfg, label=f"derivatives future {_to_ymd(day)}")
        fut = vol = oi = float("nan")
        if fut_df is not None and not fut_df.empty:
            close_c = next((c for c in ("종가", "Close", "close") if c in fut_df.columns), None)
            vol_c = next((c for c in ("거래량", "Volume", "volume") if c in fut_df.columns), None)
            oi_c = next((c for c in ("미결제약정", "open_interest") if c in fut_df.columns), None)
            first = fut_df.iloc[0]
            if close_c is not None:
                fut = float(pd.to_numeric(first[close_c], errors="coerce"))
            if vol_c is not None:
                vol = float(pd.to_numeric(first[vol_c], errors="coerce"))
            if oi_c is not None:
                oi = float(pd.to_numeric(first[oi_c], errors="coerce"))
        if pd.isna(spot) and pd.isna(fut):
            # 휴장일/무자료일 — 빈 행을 만들지 않는다.
            continue
        basis = float(fut - spot) if pd.notna(fut) and pd.notna(spot) else float("nan")
        basis_pct = (
            float(basis / spot) if pd.notna(spot) and float(spot) != 0.0 and pd.notna(basis) else float("nan")
        )
        rows.append(
            {
                "date": d_norm,
                "kospi200_close": spot,
                "k200_future_close": fut,
                "basis": basis,
                "basis_pct": basis_pct,
                "future_volume": vol,
                "future_open_interest": oi,
            }
        )
    return pd.DataFrame(rows, columns=_COLS)


def collect_derivatives_basis(
    cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]
) -> pd.DataFrame:
    """KOSPI200 지수-선물 베이시스 패널을 수집합니다.

    KRX Open API(``drv/fut_bydd_trd``) 를 주 경로로 사용하고, 유효 행이 하나도
    없으면 pykrx 로 fallback 합니다.

    Args:
        cfg: Alt-data 설정.
        business_days: 영업일 목록.

    Returns:
        수집된 원시 DataFrame (``date`` 키, 시장 레벨).
    """
    if not business_days:
        return pd.DataFrame(columns=_COLS)

    krx = _collect_via_krx(cfg, business_days)
    if not krx.empty:
        # KRX 주 경로가 하나라도 유효 행을 냈으면 그대로 사용 (부분 커버리지 허용).
        return krx

    logger.warning("[DATA] stage=altdata_deriv status=KRX_EMPTY fallback=pykrx")
    return _collect_via_pykrx(cfg, business_days)
