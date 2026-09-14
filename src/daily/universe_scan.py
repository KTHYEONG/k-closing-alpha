from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.daily import archive
from src.strategy.contract import DEFAULT_UNIVERSE, UniverseSpec

logger = logging.getLogger(__name__)

UNIVERSE_SCAN_SCENARIO_TAG: str = "등락률스캔"

SCAN_PRIMARY_VENDOR: str = "kiwoom"

TOSS_RANKING_TYPE_TOP_GAINERS: str = "TOP_GAINERS"

TOSS_RANKING_TYPE_TRADE_VALUE: str = "MARKET_TRADING_AMOUNT"


class UniverseScanCoverageError(RuntimeError):
    """Kiwoom 커버리지 없이 거래 후보 리스트를 만들 수 없을 때의 fail-closed 오류."""


RANKING_SCAN_INPUT_CNT: str = "200"
# 실측(2026-09-14): KIS 등락률순위(FHPST01700000)는 INPUT_CNT와 무관하게 요청당 최대 30행이며 연속조회(tr_cont)가 없다.
KIS_RANKING_ROW_CAP: int = 30
# 실측 2~10% 밴드 390종목을 41콜(약 3~5초)로 전수 복원. 급등장 후보 3~4배와 이분할 오버헤드를 덮는 상한(데이터 계좌 리미터 기준 약 11초).
KIS_RANKING_MAX_CALLS: int = 200

__all__ = [
    "KIS_RANKING_MAX_CALLS",
    "KIS_RANKING_ROW_CAP",
    "RANKING_SCAN_INPUT_CNT",
    "SCAN_PRIMARY_VENDOR",
    "TOSS_RANKING_TYPE_TOP_GAINERS",
    "TOSS_RANKING_TYPE_TRADE_VALUE",
    "UNIVERSE_SCAN_SCENARIO_TAG",
    "UniverseScanCoverageError",
    "archive_universe_snapshot",
    "collect_universe_scan",
    "fetch_candidate_stock_list",
    "fetch_kis_band_ranking",
    "fetch_trade_value_union",
    "map_kiwoom_ranking_rows_to_archive_frame",
    "map_kiwoom_ranking_rows_to_stock_list",
    "map_ranking_rows_to_archive_frame",
    "map_ranking_rows_to_stock_list",
    "map_toss_ranking_rows_to_stock_list",
    "map_toss_trade_value_rows_to_stock_list",
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
        ETN Q prefix is normalized to the bare 6-digit code.
    """
    out: list[dict] = []
    for row in rows:
        code_raw = str(row.get("stck_shrn_iscd") or row.get("mksc_shrn_iscd") or "").strip()
        if not code_raw:
            continue
        if len(code_raw) == 7 and code_raw.startswith("Q"):
            # KIS 순위 TR은 ETN을 Q접두어로 반환하지만 1순위(Kiwoom) 후보는 접두어 없는 6자리이므로 동일 표기로 맞춘다(시세 경로의 미해석 처리와 일치).
            code_raw = code_raw[1:]
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


def map_toss_ranking_rows_to_stock_list(rows: list[dict], universe: UniverseSpec = DEFAULT_UNIVERSE) -> list[dict]:
    """Map Toss `/api/v1/rankings` TOP_GAINERS rows to collect.py stock_list shape.

    Toss rankings carry no company-name field (unlike KIS/Kiwoom); name is
    always None for these rows -- an explicit gap, not fabricated data. Rows
    are client-side filtered to [universe.chg_min, universe.chg_max) since
    Toss has no min/max band query, only a top-N-by-rank list, so coverage
    may be narrower than Kiwoom's dedicated band query during a real outage.

    Args:
        rows: Raw Toss rankings result.rankings rows.
        universe: Change-rate band to keep (mirrors select_universe's >=min, <max convention).

    Returns:
        List of {code, name, price, chgrate} dicts; codeless or out-of-band rows skipped.
    """
    out: list[dict] = []
    for row in rows:
        code_raw = str(row.get("symbol", "") or "").strip()
        if not code_raw:
            continue
        price_block = row.get("price") or {}
        chg = _to_float(price_block.get("changeRate"))
        if not (float(universe.chg_min) <= chg < float(universe.chg_max)):
            continue
        out.append(
            {
                "code": code_raw.zfill(6),
                "name": None,
                "price": price_block.get("lastPrice"),
                "chgrate": price_block.get("changeRate"),
            }
        )
    return out


def map_toss_trade_value_rows_to_stock_list(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        code_raw = str(row.get("symbol", "") or "").strip()
        if not code_raw:
            continue
        code = code_raw.zfill(6)
        price_block = row.get("price") or {}
        out.append(
            {
                "code": code,
                "name": None,
                "price": price_block.get("lastPrice"),
                "chgrate": price_block.get("changeRate"),
            }
        )
    return out


async def fetch_trade_value_union(session, *, toss_client: Any | None = None, count: int = 100) -> list[dict]:
    if toss_client is None:
        return []
    try:
        res = await toss_client.get_rankings(session, ranking_type=TOSS_RANKING_TYPE_TRADE_VALUE, market_country="KR", duration="1d", count=count)
    except Exception as e:
        logger.warning("[DATA] stage=universe_scan_trade_value_union vendor=toss status=FAILED reason=%s", e)
        return []
    if "error" in res:
        err = res["error"]
        logger.warning("[DATA] stage=universe_scan_trade_value_union vendor=toss status=FAILED reason=code=%s msg=%s", err.get("code"), err.get("message", ""))
        return []
    rows = (res.get("result") or {}).get("rankings") or []
    out = map_toss_trade_value_rows_to_stock_list(rows)
    logger.info("[DATA] stage=universe_scan_trade_value_union vendor=toss status=OK n_rows=%d", len(out))
    return out


async def fetch_kis_band_ranking(
    client: Any, session: Any, *, rate_min_pct: float, rate_max_pct: float, max_calls: int = KIS_RANKING_MAX_CALLS
) -> list[dict]:
    """Fetch full KIS band coverage despite the 30-row cap via adaptive bisection.

    The KIS fluctuation ranking returns at most 30 rows per call with no
    continuation, so saturated bands are bisected on a 0.01%p grid.

    Args:
        client: KIS API client.
        session: HTTP session.
        rate_min_pct: Band lower bound (percent).
        rate_max_pct: Band upper bound (percent).
        max_calls: Call budget guard.

    Returns:
        Deduplicated ranking rows across all leaves.

    Raises:
        UniverseScanCoverageError: On vendor failure, budget exhaustion, or a
            saturated 1bp leaf band.
    """
    lo_bp = int(round(rate_min_pct * 100))
    hi_bp = int(round(rate_max_pct * 100))
    rows_by_code: dict[str, dict] = {}
    n_calls = 0
    stack: list[tuple[int, int]] = [(lo_bp, hi_bp)]
    while stack:
        a, b = stack.pop()
        if n_calls >= max_calls:
            raise UniverseScanCoverageError(f"kis ranking call budget {max_calls} exhausted")
        n_calls += 1
        res = await client.get_fluctuation_ranking(
            session, rate_min_pct=a / 100.0, rate_max_pct=b / 100.0, market_div_code="J"
        )
        if res.get("rt_cd") != "0":
            raise UniverseScanCoverageError(f"kis ranking failed rt_cd={res.get('rt_cd')} msg={res.get('msg1', '')}")
        rows = res.get("output") or []
        if len(rows) >= KIS_RANKING_ROW_CAP:
            if b - a <= 1:
                raise UniverseScanCoverageError(
                    f"kis ranking band [{a / 100:.2f}, {b / 100:.2f}] saturated at {KIS_RANKING_ROW_CAP} rows"
                )
            mid = (a + b) // 2
            stack.append((mid, b))
            stack.append((a, mid))
            continue
        for row in rows:
            code = str(row.get("stck_shrn_iscd") or "").strip()
            if code:
                # 밴드 경계값은 양쪽 하위 밴드에 모두 포함되므로 코드 기준 중복 제거로 흡수한다.
                rows_by_code[code] = row
    logger.info("[DATA] stage=universe_scan_kis_band status=OK n_rows=%d n_calls=%d", len(rows_by_code), n_calls)
    return list(rows_by_code.values())


async def fetch_candidate_stock_list(
    client, session, *, universe: UniverseSpec = DEFAULT_UNIVERSE, kiwoom_client: Any | None = None, toss_client: Any | None = None, kis_band_fallback: bool = False
) -> list[dict]:
    """Kiwoom ka10027을 1순위, KIS 밴드 스캔을 2순위, Toss TOP_GAINERS 랭킹을 3순위 폴백으로 후보 stock_list를 조회한다 (fail-closed).

    Args:
        client: KIS data client used for the band fallback.
        session: HTTP session.
        universe: Universe bounds for the ranking call.
        kiwoom_client: Kiwoom vendor client (필수).
        toss_client: Toss vendor client; Kiwoom·KIS 실패 시 폴백으로 사용된다 (선택,
            None이면 폴백 없이 Kiwoom 실패 즉시 fail-closed).
        kis_band_fallback: Kiwoom -> KIS band scan -> Toss top-100; default False keeps the Kiwoom-or-Toss behaviour.
            Truncation only aborts when the flag is off.

    Returns:
        Candidate stock_list in collect.py shape. Toss-sourced rows carry
        name=None (Toss rankings have no company-name field) and are
        client-side filtered to [universe.chg_min, universe.chg_max) from at
        most 100 top-gainer rows, so coverage may be narrower than Kiwoom's
        dedicated band query during a genuine Kiwoom outage.

    Raises:
        UniverseScanCoverageError: kiwoom_client 미주입, 또는 Kiwoom과(toss_client가
            주입된 경우) Toss 모두 실패 시, 또는 truncated Kiwoom scan
            (no Toss fallback, which is narrower).
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
        kw_reason = f"kiwoom ranking call failed: {e}"
    else:
        if kw_res.get("truncated"):
            if not kis_band_fallback:
                raise UniverseScanCoverageError(f"kiwoom ranking truncated: {kw_res.get('msg1', '')}")
            kw_reason = f"kiwoom ranking truncated: {kw_res.get('msg1', '')}"
        elif kw_res.get("rt_cd") == "0":
            rows = kw_res.get("output") or []
            out = map_kiwoom_ranking_rows_to_stock_list(rows)
            logger.info("[DATA] stage=universe_scan vendor=%s n_rows=%d", SCAN_PRIMARY_VENDOR, len(out))
            return out
        else:
            kw_reason = f"kiwoom ranking failed rt_cd={kw_res.get('rt_cd')} msg={kw_res.get('msg1', '')}"
    logger.warning("[DATA] stage=universe_scan vendor=kiwoom status=FAILED reason=%s", kw_reason)
    if kis_band_fallback:
        try:
            kis_rows = await fetch_kis_band_ranking(
                client, session, rate_min_pct=universe.chg_min * 100.0, rate_max_pct=universe.chg_max * 100.0
            )
        except UniverseScanCoverageError as exc:
            logger.warning("[DATA] stage=universe_scan vendor=kis status=FAILED reason=%s", exc)
            kw_reason = f"{kw_reason}; {exc}"
        else:
            out = map_ranking_rows_to_stock_list(kis_rows)
            logger.warning("[DATA] stage=universe_scan vendor=kis status=FALLBACK n_rows=%d", len(out))
            return out
    if toss_client is None:
        raise UniverseScanCoverageError(kw_reason)
    try:
        toss_res = await toss_client.get_rankings(
            session, ranking_type=TOSS_RANKING_TYPE_TOP_GAINERS, market_country="KR", duration="1d", count=100,
        )
    except Exception as e:
        raise UniverseScanCoverageError(f"{kw_reason}; toss ranking call failed: {e}") from e
    if "error" in toss_res:
        err = toss_res["error"]
        raise UniverseScanCoverageError(f"{kw_reason}; toss ranking failed code={err.get('code')} msg={err.get('message', '')}")
    rows = (toss_res.get("result") or {}).get("rankings") or []
    out = map_toss_ranking_rows_to_stock_list(rows, universe)
    logger.warning(
        "[DATA] stage=universe_scan vendor=toss status=FALLBACK n_rows=%d note=degraded_top100_band_filter",
        len(out),
    )
    return out
