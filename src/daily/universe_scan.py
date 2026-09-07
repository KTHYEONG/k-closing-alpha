from __future__ import annotations

import logging

import pandas as pd

from src.daily import archive
from src.strategy.contract import DEFAULT_UNIVERSE, UniverseSpec

logger = logging.getLogger(__name__)

UNIVERSE_SCAN_SCENARIO_TAG: str = "등락률스캔"

__all__ = [
    "UNIVERSE_SCAN_SCENARIO_TAG",
    "archive_universe_snapshot",
    "collect_universe_scan",
    "map_ranking_rows_to_archive_frame",
]

_ARCHIVE_COLUMNS: list[str] = [
    "스냅샷_날짜",
    "종목코드",
    "종목명",
    "종가",
    "전일종가",
    "등락률",
    "거래량",
    "거래대금",
    "시가총액",
    "시나리오",
    "snapshot_timestamp",
]


def _to_float(value: object) -> float:
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return float("nan")


def map_ranking_rows_to_archive_frame(
    rows: list[dict], snapshot_date: str, snapshot_timestamp: pd.Timestamp
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_ARCHIVE_COLUMNS)
    records: list[dict] = []
    for row in rows:
        code_raw = str(row.get("stck_shrn_iscd") or row.get("mksc_shrn_iscd") or "").strip()
        code = code_raw.zfill(6) if code_raw else float("nan")
        name_raw = row.get("hts_kor_isnm", None)
        name = name_raw if name_raw is not None and str(name_raw).strip() != "" else float("nan")
        close = _to_float(row.get("stck_prpr", None))
        prev_close = _to_float(row.get("stck_sdpr", None))
        chg = _to_float(row.get("prdy_ctrt", None))
        volume = _to_float(row.get("acml_vol", None))
        # UNVERIFIED: acml_tr_pbmn 단위가 이 TR에서 원화 raw값인지 미확인 — 억원 환산(1e8 나눔) 가정.
        tr_amount_raw = _to_float(row.get("acml_tr_pbmn", None))
        tr_amount = tr_amount_raw / 1e8 if tr_amount_raw == tr_amount_raw else float("nan")
        # 이 TR은 시가총액 반환이 확인되지 않음 — 없으면 NaN 유지하며 오류로 취급하지 않음.
        market_cap = _to_float(row.get("stck_avls"))
        records.append(
            {
                "스냅샷_날짜": snapshot_date,
                "종목코드": code,
                "종목명": name,
                "종가": close,
                "전일종가": prev_close,
                "등락률": chg,
                "거래량": volume,
                "거래대금": tr_amount,
                "시가총액": market_cap,
                "시나리오": UNIVERSE_SCAN_SCENARIO_TAG,
                "snapshot_timestamp": snapshot_timestamp,
            }
        )
    df = pd.DataFrame(records, columns=_ARCHIVE_COLUMNS)
    for col in ("종가", "전일종가", "등락률", "거래량", "거래대금", "시가총액"):
        df[col] = df[col].astype("float64")
    return df


async def collect_universe_scan(
    client,
    session,
    snapshot_date: str,
    snapshot_timestamp: pd.Timestamp,
    *,
    universe: UniverseSpec = DEFAULT_UNIVERSE,
) -> pd.DataFrame:
    res = await client.get_fluctuation_ranking(
        session,
        rate_min_pct=universe.chg_min * 100.0,
        rate_max_pct=universe.chg_max * 100.0,
        market_div_code="J",
    )
    if res.get("rt_cd") != "0":
        logger.warning("Universe scan ranking failed rt_cd=%s msg=%s", res.get("rt_cd"), res.get("msg1", ""))
        return map_ranking_rows_to_archive_frame([], snapshot_date, snapshot_timestamp)
    return map_ranking_rows_to_archive_frame(res.get("output") or [], snapshot_date, snapshot_timestamp)


def archive_universe_snapshot(df: pd.DataFrame, snapshot_date: str) -> int:
    return archive.upsert_archive_snapshot(df, snapshot_date=snapshot_date)
