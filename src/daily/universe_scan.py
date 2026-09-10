from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.daily import archive
from src.strategy.contract import DEFAULT_UNIVERSE, UniverseSpec

logger = logging.getLogger(__name__)

UNIVERSE_SCAN_SCENARIO_TAG: str = "등락률스캔"

SCAN_PRIMARY_VENDOR: str = "kiwoom"


class UniverseScanCoverageError(RuntimeError):
    """Kiwoom 커버리지 없이 거래 후보 리스트를 만들 수 없을 때의 fail-closed 오류."""


RANKING_SCAN_INPUT_CNT: str = "200"

__all__ = [
    "RANKING_SCAN_INPUT_CNT",
    "SCAN_PRIMARY_VENDOR",
    "UNIVERSE_SCAN_SCENARIO_TAG",
    "UniverseScanCoverageError",
    "archive_universe_snapshot",
    "collect_universe_scan",
    "fetch_candidate_stock_list",
    "map_kiwoom_ranking_rows_to_archive_frame",
    "map_kiwoom_ranking_rows_to_stock_list",
    "map_ranking_rows_to_archive_frame",
    "map_ranking_rows_to_stock_list",
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


def map_kiwoom_ranking_rows_to_archive_frame(
    rows: list[dict], snapshot_date: str, snapshot_timestamp: pd.Timestamp
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_ARCHIVE_COLUMNS)
    records: list[dict] = []
    for row in rows:
        code_raw = str(row.get("stk_cd", "") or "").split("_")[0].strip()
        code = code_raw.zfill(6) if code_raw else float("nan")
        name_raw = row.get("stk_nm", None)
        name = name_raw if name_raw is not None and str(name_raw).strip() != "" else float("nan")
        close = _to_float(row.get("cur_prc", None))
        pred_pre = _to_float(row.get("pred_pre", None))
        prev_close = close - pred_pre
        chg = _to_float(row.get("flu_rt", None))
        volume = _to_float(row.get("now_trde_qty", None))
        tr_amount = round(close * volume / 1e8, 4)
        market_cap = float("nan")
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
    kiwoom_client: Any | None = None,
) -> pd.DataFrame:
    if kiwoom_client is not None:
        try:
            kw_res = await kiwoom_client.get_fluctuation_ranking(
                session,
                rate_min_pct=universe.chg_min * 100.0,
                rate_max_pct=universe.chg_max * 100.0,
            )
        except Exception as e:
            logger.warning("Universe scan kiwoom ranking failed, falling back to KIS: %s", e)
            kw_res = {"rt_cd": "1", "output": []}
        if kw_res.get("rt_cd") == "0":
            return map_kiwoom_ranking_rows_to_archive_frame(kw_res.get("output") or [], snapshot_date, snapshot_timestamp)
        logger.warning(
            "Universe scan kiwoom ranking failed rt_cd=%s, falling back to KIS",
            kw_res.get("rt_cd"),
        )
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


def map_ranking_rows_to_stock_list(rows: list[dict]) -> list[dict]:
    """Map KIS fluctuation-ranking rows to collect.py stock_list shape.

    Args:
        rows: Raw KIS ranking rows.

    Returns:
        List of {code, name, price, chgrate} dicts; codeless rows skipped.
    """
    out: list[dict] = []
    for row in rows:
        code_raw = str(row.get("stck_shrn_iscd") or row.get("mksc_shrn_iscd") or "").strip()
        if not code_raw:
            continue
        out.append(
            {
                "code": code_raw.zfill(6),
                "name": row.get("hts_kor_isnm"),
                "price": row.get("stck_prpr"),
                "chgrate": row.get("prdy_ctrt"),
            }
        )
    return out


def map_kiwoom_ranking_rows_to_stock_list(rows: list[dict]) -> list[dict]:
    """Map Kiwoom ka10027 ranking rows to collect.py stock_list shape.

    Args:
        rows: Raw Kiwoom ranking rows.

    Returns:
        List of {code, name, price, chgrate} dicts; codeless rows skipped.
    """
    out: list[dict] = []
    for row in rows:
        code_raw = str(row.get("stk_cd", "") or "").split("_")[0].strip()
        if not code_raw:
            continue
        out.append(
            {
                "code": code_raw.zfill(6),
                "name": row.get("stk_nm"),
                "price": row.get("cur_prc"),
                "chgrate": row.get("flu_rt"),
            }
        )
    return out


async def fetch_candidate_stock_list(
    client, session, *, universe: UniverseSpec = DEFAULT_UNIVERSE, kiwoom_client: Any | None = None
) -> list[dict]:
    """Kiwoom ka10027을 유일한 유니버스 소스로 후보 stock_list를 조회한다 (fail-closed).

    Args:
        client: KIS API client (사용하지 않음; 시그니처 호환용).
        session: HTTP session.
        universe: Universe bounds for the ranking call.
        kiwoom_client: Kiwoom vendor client (필수).

    Returns:
        Candidate stock_list in collect.py shape.

    Raises:
        UniverseScanCoverageError: kiwoom_client 미주입/호출 실패/논리 실패 시.
    """
    if kiwoom_client is None:
        raise UniverseScanCoverageError("kiwoom_client is required for the tradeable candidate list")
    try:
        kw_res = await kiwoom_client.get_fluctuation_ranking(
            session,
            rate_min_pct=universe.chg_min * 100.0,
            rate_max_pct=universe.chg_max * 100.0,
        )
    except Exception as e:
        raise UniverseScanCoverageError(f"kiwoom ranking call failed: {e}") from e
    if kw_res.get("rt_cd") != "0":
        raise UniverseScanCoverageError(
            f"kiwoom ranking failed rt_cd={kw_res.get('rt_cd')} msg={kw_res.get('msg1', '')}"
        )
    rows = kw_res.get("output") or []
    out = map_kiwoom_ranking_rows_to_stock_list(rows)
    logger.info("[DATA] stage=universe_scan vendor=%s n_rows=%d", SCAN_PRIMARY_VENDOR, len(out))
    return out
