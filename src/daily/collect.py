import asyncio
# ruff: noqa: I001 - contract mandates contiguous wiring import block after Colors
import logging
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd

from src import settings

# 커스텀 모듈 임포트
from src.api.kis.client import KisApiClient, kis_data_client_kwargs, kis_decision_shard_client_kwargs
from src.config.market_session import REALTIME_REQUOTE_DEADLINE_HHMMSS
from src.data.capture_contracts import (
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureStatus,
    Cohort,
    CoverageEntry,
    build_cohort,
)
from src.data.capture_store import CaptureStore
from src.data.orderbook_store import append_orderbook_snapshots, build_orderbook_rows
from src.utils.display import Colors
from src.daily import archive
from src.daily.universe_scan import fetch_candidate_stock_list, fetch_trade_value_union
from src.data.trading_calendar import is_kis_trading_day
from src.ml.topk_history_features import MAX_PREV_TRADING_DAY_LOOKBACK
from src.daily.universe_screen import build_screen_frame
from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL, PRICE_ANOMALY_COL, QUOTE_FAILED_COL
from src.strategy.contract import COST_AWARE_UNIVERSE, UniverseSpec, select_universe

logger = logging.getLogger(__name__)


class NonTradingDayError(RuntimeError):
    """휴장일에 결정 파이프라인이 발화할 때의 fail-closed 오류."""


def build_kiwoom_scan_client() -> Any | None:
    """Kiwoom 스캔 클라이언트를 구성한다. 자격증명이 없으면 None을 반환한다."""
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient()
    if not client.app_key:
        return None
    return client


def build_toss_scan_client() -> Any | None:
    """Toss 랭킹 폴백 클라이언트를 구성한다. 자격증명이 없으면 None을 반환한다."""
    from src.api.toss.client import TossApiClient

    client = TossApiClient()
    if not client.app_key:
        return None
    return client


async def _validate_trading_day(client, session, snapshot_date: str, *, force: bool = False) -> None:
    """휴장일 실행을 차단한다. --force가 유일한 우회 경로다."""
    if force:
        return
    if not await is_kis_trading_day(client, session, snapshot_date):
        raise NonTradingDayError(f"non-trading day: {snapshot_date}")

TARGET_CONDITION_NAME = settings.TARGET_CONDITION_NAME

logger.debug("일일 수집 시작...")


def _validate_hts_id(hts_id: str) -> None:
    if not hts_id or "여기에" in hts_id:
        raise RuntimeError(
            ".env 파일의 'KIS_HTS_ID'에 본인의 HTS ID를 입력해주세요!"
        )


def _validate_decision_window(now: datetime, *, force: bool = False) -> None:
    if force:
        return
    from src.config.market_session import (
        DECISION_WINDOW_END_HHMMSS,
        DECISION_WINDOW_START_HHMMSS,
    )

    hhmmss = now.strftime("%H%M%S")
    if not (DECISION_WINDOW_START_HHMMSS <= hhmmss <= DECISION_WINDOW_END_HHMMSS):
        raise RuntimeError(
            f"결정 창({DECISION_WINDOW_START_HHMMSS}~{DECISION_WINDOW_END_HHMMSS} KST) 밖 실행은 금지됩니다. --force로 우회 가능합니다."
        )


def safe_float(value, default=0.0):
    """문자열이나 None 값을 안전하게 float로 변환"""
    if value is None:
        return default
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------
# 헬퍼 함수: 시장 지수 등락률 파싱
# ---------------------------------------------------------
def parse_market_index_rate(data):
    if not data or data.get("rt_cd") != "0":
        return None
    out1 = data.get("output1")
    if not out1:
        return None
    rate_str = out1.get("bstp_nmix_prdy_ctrt") or out1.get("prdy_ctrt")
    try:
        if rate_str and float(rate_str) != 0.0:
            return float(rate_str)
        current_price = float(out1.get("bstp_nmix_prpr", "0"))
        change_amount = float(out1.get("bstp_nmix_prdy_vrss", "0"))
        prev_close = current_price - change_amount
        if prev_close != 0:
            return round((change_amount / prev_close) * 100, 2)
    except Exception:
        pass
    return None


# ---------------------------------------------------------
# 표준 CSV 저장 (utf-8-sig) & Parquet
# ---------------------------------------------------------


def flag_cost_aware_admission(
    df: pd.DataFrame, *, decision_date: pd.Timestamp, screen: UniverseSpec = COST_AWARE_UNIVERSE
) -> pd.DataFrame:
    """Flag every row of the daily snapshot with the COST_AWARE_UNIVERSE verdict.

    Args:
        df: Enriched Korean-column snapshot frame.
        decision_date: Decision date for point-in-time tick costing.
        screen: Universe admission spec.

    Returns:
        A copy of the input with a bool ``admitted`` column; no rows dropped.
        apply_cost_aware_admission wraps this helper and drops non-admitted rows.
    """
    import numpy as np

    if len(df) == 0:
        out = df.copy()
        out["admitted"] = np.zeros(0, dtype=bool)
        logger.info("[DATA] stage=cost_aware_admission n_raw=0 n_admitted=0 n_ceiling_excluded=0")
        return out
    mapped = build_screen_frame(df, decision_date=decision_date)
    mask = select_universe(mapped, screen)
    n_ceiling_excluded = int(mapped["is_ceiling"].to_numpy(dtype=bool).sum())
    logger.info(
        "[DATA] stage=cost_aware_admission n_raw=%d n_admitted=%d n_ceiling_excluded=%d",
        len(df),
        int(np.asarray(mask, dtype=bool).sum()),
        n_ceiling_excluded,
    )
    flagged = df.copy()
    flagged["admitted"] = np.asarray(mask, dtype=bool)
    return flagged


# 실시간 결정 스냅샷의 degraded-row(벤더 실패 또는 값-비정상) 허용 임계치.
# price_ingest.MIN_FLOW_COVERAGE(0.99, 야간 벌크 흐름 게이트)와 동일 철학을 실시간
# 경로에 이식한 값이다 -- 두 파이프라인은 별개 흐름이라 상수도 분리해 둔다.
REALTIME_MIN_QUOTE_COVERAGE: float = 0.99

# fetch_single_stock의 failed_apis 태그: 벤더가 rt_cd=0으로 응답했지만 종목코드를 해석하지 못한 경우
QUOTE_UNRESOLVED_API: str = "현재가_미해석"

_DECISION_ELIGIBILITY_RULE_VERSION: str = "price_history_panel@v1"


def _capture_root() -> Path:
    root = settings.COLLECTION_ROOT
    if root is not None:
        return Path(root)
    return Path(settings.HISTORY_DIR) / "capture"


def _validate_capture_context(
    capture_store: CaptureStore | None, cohort: Cohort | None, run_id: str | None
) -> bool:
    if capture_store is None and cohort is None and run_id is None:
        return False
    if capture_store is None or cohort is None or not run_id:
        raise ValueError("capture context is incomplete or inconsistent")
    return True


def _capture_context_for(
    trading_day: date, run_id: str, cohort_id: str | None, dataset: CaptureDataset, symbol: str | None, endpoint: str
) -> CaptureContext:
    return CaptureContext(
        trading_date=trading_day,
        run_id=run_id,
        dataset=dataset,
        vendor="kis",
        endpoint=endpoint,
        symbol=symbol,
        venue="KRX",
        session="regular",
        capture_reason="decision-input",
        cohort_id=cohort_id,
        scheduled_at=None,
    )


def _append_with_unique_attempt(store: CaptureStore, response: CapturedResponse) -> None:
    """Persist one observed response, shifting attempt_index on identity conflict.

    The raw artifact path carries vendor/endpoint/page/attempt, so a repeated
    call (e.g. requote) with the same identifiers but different bytes must keep
    both evidences under distinct identities instead of dropping either one.
    """
    base_attempt = int(response.attempt_index)
    for offset in range(6):
        candidate = response if offset == 0 else response.model_copy(update={"attempt_index": base_attempt + offset})
        try:
            store.append_response(candidate)
            return
        except ValueError as exc:
            if "conflicting immutable artifact identity" not in str(exc):
                raise
    raise OSError("required decision evidence cannot be published")


def _persist_market_response(
    store: CaptureStore,
    context: CaptureContext,
    payload: dict[str, Any] | None,
    started: datetime,
    received: datetime,
) -> None:
    status = CaptureStatus.COMPLETE if payload is not None and payload.get("rt_cd") == "0" else CaptureStatus.FAILED
    body = dict(payload) if isinstance(payload, dict) else None
    _append_with_unique_attempt(
        store,
        CapturedResponse(
            context=context,
            request_started_at=started,
            received_at=received,
            payload=body,
            status=status,
            source_timestamp=None,
            source_published_at=None,
            page_index=0,
            attempt_index=0,
            continuation={},
            error_type=None if status == CaptureStatus.COMPLETE else "vendor_failure",
        ),
    )


def _persist_observed_market_page(
    store: CaptureStore,
    context: CaptureContext,
    payload: Any,
    metadata: Mapping[str, Any],
    started: datetime,
    received: datetime,
    page_index: int,
    attempt_index: int,
) -> None:
    """Immediately persist one retry-level vendor response observed via callback."""
    body = dict(payload) if isinstance(payload, dict) else None
    status = CaptureStatus.COMPLETE if isinstance(payload, dict) and payload.get("rt_cd") == "0" else CaptureStatus.FAILED
    meta = dict(metadata) if isinstance(metadata, Mapping) else {}
    if status == CaptureStatus.COMPLETE:
        error_type = None
    else:
        raw_err = meta.get("error_type")
        error_type = str(raw_err) if isinstance(raw_err, str) and raw_err.strip() else "vendor_failure"
    _append_with_unique_attempt(
        store,
        CapturedResponse(
            context=context,
            request_started_at=started,
            received_at=received,
            payload=body,
            status=status,
            source_timestamp=None,
            source_published_at=None,
            page_index=int(page_index),
            attempt_index=int(attempt_index),
            continuation={str(k): str(v) for k, v in meta.items()},
            error_type=error_type,
        ),
    )


_MARKET_LABEL_ROUTE: dict[str, tuple[CaptureDataset, str]] = {
    "price": (CaptureDataset.PRICE, "inquire-price"),
    "investor": (CaptureDataset.INVESTOR_ESTIMATE, "investor-trend-estimate"),
    "orderbook": (CaptureDataset.ORDERBOOK, "inquire-asking-price"),
}


async def resolve_prev_trading_day_kis(client: Any, session: Any, decision_date: pd.Timestamp, *, krx_is_trading_day: Callable[[pd.Timestamp], bool] | None = None, max_lookback_days: int = MAX_PREV_TRADING_DAY_LOOKBACK) -> pd.Timestamp:
    """Resolve the previous trading day via KIS first, KRX fallback per date.

    Args:
        client: KIS API client.
        session: HTTP session.
        decision_date: Decision date (time component ignored).
        krx_is_trading_day: KRX oracle override (tests); None selects is_krx_trading_day.
        max_lookback_days: Calendar-day search bound.

    Returns:
        The previous trading day, normalized to midnight.

    Raises:
        RuntimeError: When both oracles fail.
        ValueError: When no trading day exists within the bound.
    """
    d = pd.Timestamp(decision_date).normalize()
    for k in range(1, int(max_lookback_days) + 1):
        cand = d - pd.Timedelta(days=k)
        if cand.weekday() >= 5:
            continue
        try:
            is_open = await is_kis_trading_day(client, session, cand)
        except RuntimeError as exc:
            logger.warning("[DATA] stage=prev_trading_day vendor=kis status=FAILED date=%s reason=%s fallback=krx", cand.date(), exc)
            oracle = krx_is_trading_day
            if oracle is None:
                from src.data.trading_calendar import is_krx_trading_day as oracle
            is_open = await asyncio.to_thread(oracle, cand)
        if is_open:
            return cand
    raise ValueError(f"no trading day within {int(max_lookback_days)} days before {d.date()}")


def load_eligible_codes(decision_date: pd.Timestamp, *, prev_trading_day: pd.Timestamp, path: str | os.PathLike[str] | None = None) -> frozenset[str]:
    """Return the symbols listed in price_history on the previous trading day.

    Args:
        decision_date: Decision date; only rows strictly before it are read.
        prev_trading_day: Previous trading day resolved via resolve_prev_trading_day_kis.
        path: Parquet path; None selects settings.PRICE_HISTORY_PARQUET_PATH.

    Returns:
        Symbols present on the latest trading day strictly before decision_date.

    Raises:
        FileNotFoundError: When the parquet does not exist.
        ValueError: When prev_trading_day is not before decision_date, or when
            no rows exist on the previous trading day.
    """
    prev = pd.Timestamp(prev_trading_day).normalize()
    if prev >= pd.Timestamp(decision_date).normalize():
        raise ValueError(f"prev_trading_day {prev.date()} must be before decision_date {pd.Timestamp(decision_date).date()}")
    src_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if path is None else path)
    if not src_path.exists():
        raise FileNotFoundError(f"price_history not found: {src_path}")
    rows = pd.read_parquet(src_path, columns=["date", "symbol"], filters=[("date", "==", prev)])
    if rows.empty:
        raise ValueError(f"stale price_history: no rows on prev_trading_day={prev.date()}")
    return frozenset(rows["symbol"].astype(str))


async def resolve_eligible_codes(client: Any, session: Any, decision_date: pd.Timestamp) -> frozenset[str]:
    """Resolve eligibility through the async KIS calendar before quoting."""
    prev = await resolve_prev_trading_day_kis(client, session, decision_date)
    listed = load_eligible_codes(decision_date, prev_trading_day=prev)
    logger.info("[DATA] stage=eligibility n_listed=%d", len(listed))
    return listed


def filter_eligible_candidates(stock_list: list[dict], eligible_codes: frozenset[str]) -> list[dict]:
    """Drop scanned candidates that are not listed in the research panel.

    Args:
        stock_list: Scan rows carrying ``code``.
        eligible_codes: Symbols from load_eligible_codes.

    Returns:
        Eligible rows in scan order.

    Raises:
        ValueError: When the scan is non-empty but nothing is eligible.
    """
    kept = [row for row in stock_list if str(row["code"]) in eligible_codes]
    dropped = [str(row["code"]) for row in stock_list if str(row["code"]) not in eligible_codes]
    logger.info(
        "[DATA] stage=instrument_eligibility n_raw=%d n_eligible=%d n_dropped=%d dropped_head=%s",
        len(stock_list),
        len(kept),
        len(dropped),
        dropped[:5],
    )
    if stock_list and not kept:
        raise ValueError(f"no scanned candidate is an eligible listed stock: n_raw={len(stock_list)}")
    return kept


def flag_price_anomaly(df: pd.DataFrame) -> pd.Series:
    """벤더가 성공(rt_cd='0')을 반환했더라도 값 자체가 비정상인 행을 표식한다.

    현재가_실패(QUOTE_FAILED_COL)는 벤더 호출 자체의 실패만 포착하므로, 호출은
    성공했지만 종가<=0이거나 OHLC 범위가 내부적으로 모순인 경우(레버리지/인버스
    ETN 등에서 실측된 패턴)는 별도로 잡아야 한다. NaN(quote_failed 경로)은 이미
    현재가_실패로 표식되므로 여기서는 이중 표식하지 않는다.

    Args:
        df: 종가/고가/저가/거래량 컬럼을 포함한 wide 단면 프레임.

    Returns:
        df와 같은 길이/인덱스의 bool Series. True면 값-비정상.
    """
    close = pd.to_numeric(df["종가"], errors="coerce")
    high = pd.to_numeric(df["고가"], errors="coerce")
    low = pd.to_numeric(df["저가"], errors="coerce")
    volume = pd.to_numeric(df["거래량"], errors="coerce")
    has_quote = close.notna() & high.notna() & low.notna() & volume.notna()
    non_positive_close = close <= 0.0
    inconsistent_range = low > high
    close_out_of_range = (close < low) | (close > high)
    negative_volume = volume < 0.0
    anomaly = has_quote & (non_positive_close | inconsistent_range | close_out_of_range | negative_volume)
    return anomaly.fillna(False).astype(bool)


def check_realtime_collection_coverage(
    df: pd.DataFrame, *, min_coverage: float = REALTIME_MIN_QUOTE_COVERAGE
) -> dict[str, Any]:
    """실시간 스냅샷의 degraded 비율이 임계치를 넘으면 fail-closed 한다.

    degraded 행은 QUOTE_FAILED_COL(벤더 호출 실패) 또는 PRICE_ANOMALY_COL
    (호출 성공했지만 값 비정상)이 True인 행이다. price_ingest.compute_flow_coverage의 커버리지 정의와
    동일한 철학을 실시간 단일 스냅샷에 적용한다.

    Args:
        df: PRICE_ANOMALY_COL과 QUOTE_FAILED_COL을 포함한 wide 단면 프레임.
        min_coverage: degraded 되지 않은 행이 차지해야 할 최소 비율.

    Returns:
        {"n_raw": 전체 행수, "n_degraded": degraded 행수, "coverage": 정상 비율}.

    Raises:
        ValueError: df가 비었거나 coverage가 min_coverage 미만인 경우.
    """
    if len(df) == 0:
        raise ValueError("check_realtime_collection_coverage received an empty snapshot")
    degraded = df[QUOTE_FAILED_COL].fillna(False).astype(bool) | df[PRICE_ANOMALY_COL].fillna(False).astype(bool)
    n_raw = len(df)
    n_degraded = int(degraded.sum())
    coverage = 1.0 - (n_degraded / n_raw)
    if coverage < float(min_coverage):
        raise ValueError(
            f"real-time collection coverage {coverage:.4f} below {min_coverage}: n_degraded={n_degraded}/{n_raw}"
        )
    return {"n_raw": n_raw, "n_degraded": n_degraded, "coverage": round(coverage, 6)}


async def resolve_daily_candidates(client, session, *, kiwoom_client: Any | None = None, toss_client: Any | None = None, on_page: Any | None = None) -> list[dict]:
    """자동 비용축 스캔 결과를 그대로 반환합니다.

    Args:
        client: KIS API client.
        session: HTTP session.
        kiwoom_client: Kiwoom scan client (1순위 후보 소스).
        toss_client: Toss scan client (Kiwoom·KIS 밴드 스캔 모두 실패 시 최종 폴백, 선택).
        on_page: 원시 스캔 페이지 관찰자.

    Returns:
        자동 스캔 후보 리스트. 스캔이 비면 빈 리스트를 반환한다.
    """
    scan_kwargs: dict[str, Any] = {}
    if on_page is not None:
        scan_kwargs["on_page"] = on_page
    primary = await fetch_candidate_stock_list(client, session, kiwoom_client=kiwoom_client, toss_client=toss_client, kis_band_fallback=True, **scan_kwargs) or []
    union_rows = await fetch_trade_value_union(session, toss_client=toss_client, **scan_kwargs)
    seen_codes = {row["code"] for row in primary}
    merged = primary + [row for row in union_rows if row["code"] not in seen_codes]
    return merged


# ---------------------------------------------------------
# 상세 정보 조회 및 데이터 매핑 (비동기)
# ---------------------------------------------------------


async def fetch_single_stock(
    i: int,
    stock: dict[str, Any],
    total: int,
    sem: asyncio.Semaphore,
    client: Any,
    session: Any,
    *,
    capture_store: CaptureStore | None = None,
    cohort: Cohort | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Preserve independently timed quote, investor, and book evidence before parsing.

    Args:
        i: Existing progress index.
        stock: Existing scan candidate row.
        total: Declared candidate count.
        sem: Existing bounded concurrency gate.
        client: Explicit-route KIS data client.
        session: Existing HTTP session.
        capture_store: Owner-local raw evidence store.
        cohort: Dated complete eligible and rejected research population.
        run_id: Shared acquisition identity.

    Returns:
        Compatible wide row, failed API tags, and legacy orderbook rows.

    Raises:
        ValueError: Capture context is incomplete or inconsistent.
        OSError: Required decision evidence cannot be published.
    """
    capture_on = _validate_capture_context(capture_store, cohort, run_id)
    async with sem:
        code = stock["code"]
        name = stock["name"]

        price = int(float(stock.get("price", 0)))
        rate = float(stock.get("chgrate", 0))

        open_price: Any = 0
        high_price: Any = 0
        low_price: Any = 0
        close_price: Any = price
        prev_close_price: Any = price
        vol_acml: Any = 0
        market_name = ""
        mkt_cap_eok: Any = 0.0
        trade_amt_eok: Any = 0.0

        from src.config.market_session import KRX_CLOSE_MARKET_DIV_CODE

        _krx_div = KRX_CLOSE_MARKET_DIV_CODE
        marks: dict[str, tuple[datetime, datetime]] = {}
        observed_pages: dict[str, list[tuple[Any, Mapping[str, Any], datetime, datetime, int, int]]] = {}

        async def _timed(label: str, coro: Any) -> Any:
            started = datetime.now(ZoneInfo("Asia/Seoul"))
            scope = getattr(client, "observe_market_responses", None)
            if callable(scope) and not asyncio.iscoroutinefunction(scope):
                events: list[tuple[Any, Mapping[str, Any], datetime, datetime, int, int]] = []

                def _collect(
                    _payload: Any,
                    _meta: Mapping[str, Any],
                    _s: datetime,
                    _r: datetime,
                    _p: int,
                    _a: int,
                ) -> None:
                    events.append((_payload, _meta, _s, _r, int(_p), int(_a)))

                with scope(_collect):
                    res = await coro
                if events:
                    observed_pages[label] = events
                    marks[label] = (
                        min(_s for _, _, _s, _, _, _ in events),
                        max(_r for _, _, _, _r, _, _ in events),
                    )
                else:
                    marks[label] = (started, datetime.now(ZoneInfo("Asia/Seoul")))
                return res
            res = await coro
            marks[label] = (started, datetime.now(ZoneInfo("Asia/Seoul")))
            return res

        (
            res_detail,
            res_investor,
            res_ob_krx,
        ) = await asyncio.gather(
            _timed("price", client.get_current_price(session, code, market_div_code=_krx_div, allow_market_div_fallback=False)),
            _timed("investor", client.get_investor_trend_estimate(session, code)),
            _timed("orderbook", client.get_orderbook_snapshot(session, code, market_div_code=_krx_div)),
        )
        row_receipt_max = max(r for _, r in marks.values()) if marks else datetime.now(ZoneInfo("Asia/Seoul"))

        if capture_on:
            assert capture_store is not None
            assert cohort is not None
            assert run_id is not None
            trading_day = cohort.trading_date
            cohort_id = cohort.cohort_id
            finals: dict[str, Any] = {
                "price": res_detail,
                "investor": res_investor,
                "orderbook": res_ob_krx,
            }
            for label, (dataset, endpoint) in _MARKET_LABEL_ROUTE.items():
                context = _capture_context_for(trading_day, run_id, cohort_id, dataset, str(code), endpoint)
                events = observed_pages.get(label)
                if events:
                    for _payload, _meta, _s, _r, _p, _a in events:
                        _persist_observed_market_page(
                            capture_store, context, _payload, _meta, _s, _r, _p, _a
                        )
                    continue
                mark = marks.get(label, (row_receipt_max, row_receipt_max))
                final = finals[label]
                _persist_market_response(
                    capture_store,
                    context,
                    final if isinstance(final, dict) else None,
                    mark[0],
                    mark[1],
                )

        detail = res_detail.get("output") if res_detail.get("rt_cd") == "0" else None
        quote_unresolved = bool(detail) and not str(detail.get("stck_shrn_iscd") or "").strip()
        if quote_unresolved:
            detail = None

        failed_apis = []
        quote_failed = res_detail.get("rt_cd") != "0" or quote_unresolved
        if quote_failed:
            logger.warning(
                "[DATA] stage=realtime_quote code=%s status=FAILED unresolved=%s rt_cd=%s msg_cd=%s msg1=%s",
                code,
                quote_unresolved,
                res_detail.get("rt_cd"),
                res_detail.get("msg_cd"),
                res_detail.get("msg1"),
            )
            failed_apis.append("현재가")
        if quote_unresolved:
            failed_apis.append(QUOTE_UNRESOLVED_API)
        if res_investor.get("rt_cd") != "0":
            failed_apis.append("투자자추정")
        if res_ob_krx.get("rt_cd") != "0":
            failed_apis.append("호가")

        supply_failed = False
        frgn_qty, orgn_qty = 0, 0
        investor_rows = res_investor.get("output2") if res_investor.get("rt_cd") == "0" else None
        if not investor_rows:
            supply_failed = True
        else:
            latest = investor_rows[0]
            frgn_qty = int(safe_float(latest.get("frgn_fake_ntby_qty", 0)))
            orgn_qty = int(safe_float(latest.get("orgn_fake_ntby_qty", 0)))

        if detail:
            close_price = int(safe_float(detail.get("stck_prpr"), price))
            open_price = int(safe_float(detail.get("stck_oprc"), 0))
            high_price = int(safe_float(detail.get("stck_hgpr"), 0))
            low_price = int(safe_float(detail.get("stck_lwpr"), 0))
            vol_acml = int(safe_float(detail.get("acml_vol"), 0))
            rate = safe_float(detail.get("prdy_ctrt"), rate)
            sdpr = safe_float(detail.get("stck_sdpr"), 0)
            if sdpr > 0:
                prev_close_price = int(sdpr)
            elif rate != 0:
                prev_close_price = int(close_price / (1 + rate / 100))
            else:
                prev_close_price = close_price
            price = close_price
            shares = safe_float(detail.get("lstn_stcn"), 0)
            raw_market = str(detail.get("rprs_mrkt_kor_name", "")).upper()
            market_name = "KOSPI" if "KOSPI" in raw_market or "유가" in raw_market else "KOSDAQ" if "KOSDAQ" in raw_market else raw_market
            raw_mkt_cap = safe_float(detail.get("hts_avls")) * 100_000_000 or shares * price
            mkt_cap_eok = round(raw_mkt_cap / 100_000_000, 2)
            trade_amt_eok = round(safe_float(detail.get("acml_tr_pbmn")) / 100_000_000, 2)

        if quote_failed:
            # 현재가 실패: 스캔 행 값(종가/전일종가)만 유지하고 OHLCV는 NaN (0 위조 금지)
            close_price = price
            prev_close_price = int(price / (1 + rate / 100)) if rate != 0 else price
            open_price = float("nan")
            high_price = float("nan")
            low_price = float("nan")
            vol_acml = float("nan")
            mkt_cap_eok = float("nan")
            trade_amt_eok = float("nan")

        capture_ts = datetime.now(ZoneInfo("Asia/Seoul"))
        orderbook_rows: list[dict] = []
        if not quote_unresolved:
            orderbook_rows.extend(build_orderbook_rows(res_ob_krx, code, _krx_div, "decision", capture_ts))

        if supply_failed:
            frgn_net_eok = float("nan")
            orgn_net_eok = float("nan")
        else:
            frgn_net_eok = round((frgn_qty * price) / 100_000_000, 2)
            orgn_net_eok = round((orgn_qty * price) / 100_000_000, 2)

        row: dict[str, Any] = {
            "종목명": name,
            "종목코드": code,
            "시장구분": market_name,
            "시가": open_price,
            "고가": high_price,
            "저가": low_price,
            "종가": close_price,
            "전일종가": prev_close_price,
            "거래량": vol_acml,
            "거래대금": trade_amt_eok,
            "시가총액": mkt_cap_eok,
            "기관_순매수": orgn_net_eok,
            "외국인_순매수": frgn_net_eok,
            "등락률": rate,
            "수급_실패": supply_failed,
            QUOTE_FAILED_COL: quote_failed,
            DECISION_CLOSE_COL: close_price,
            CLOSE_CONFIRMED_COL: False,
        }
        if capture_on:
            row["snapshot_timestamp"] = row_receipt_max
        return row, failed_apis, orderbook_rows


async def requote_failed_quotes(stock_list: list[dict], all_res: list[tuple[dict, list[str], list[dict]]], client: Any, session: Any, sem: asyncio.Semaphore, *, now_fn: Callable[[], datetime] | None = None, capture_store: CaptureStore | None = None, cohort: Cohort | None = None, run_id: str | None = None) -> list[tuple[dict, list[str], list[dict]]]:
    """Re-fetch transient quote failures once within the decision-window budget.

    Args:
        stock_list: Scan rows in fetch order.
        all_res: First-pass fetch results.
        client: KIS API client.
        session: HTTP session.
        sem: Concurrency limiter.
        now_fn: Clock override (tests); None uses KST now.
        capture_store: Owner-local raw evidence store.
        cohort: Dated research population.
        run_id: Shared acquisition identity.

    Returns:
        Results with recovered rows replaced in place order.
    """
    retry_idx = [i for i, (row, apis, _ob) in enumerate(all_res) if row.get(QUOTE_FAILED_COL) and QUOTE_UNRESOLVED_API not in apis]
    if not retry_idx:
        return all_res
    now = now_fn() if now_fn is not None else datetime.now(ZoneInfo("Asia/Seoul"))
    if now.strftime("%H%M%S") >= REALTIME_REQUOTE_DEADLINE_HHMMSS:
        logger.warning("[DATA] stage=realtime_requote status=SKIPPED reason=deadline n_failed=%d now=%s", len(retry_idx), now.strftime("%H%M%S"))
        return all_res
    retried = await asyncio.gather(*[fetch_single_stock(i, stock_list[i], len(stock_list), sem, client, session, capture_store=capture_store, cohort=cohort, run_id=run_id) for i in retry_idx])
    out = list(all_res)
    n_recovered = 0
    for i, res in zip(retry_idx, retried):
        if not res[0].get(QUOTE_FAILED_COL):
            n_recovered += 1
        out[i] = res
    logger.info("[DATA] stage=realtime_requote n_retry=%d n_recovered=%d", len(retry_idx), n_recovered)
    return out


async def fetch_all_stock_data(
    stock_list,
    client,
    session,
    *,
    now_fn: Callable[[], datetime] | None = None,
    capture_store: CaptureStore | None = None,
    cohort: Cohort | None = None,
    run_id: str | None = None,
):
    """모든 종목의 상세 데이터를 수집합니다."""
    import sys

    sem = asyncio.Semaphore(settings.API_SEMAPHORE_LIMIT)
    total = len(stock_list)
    completed_count = 0

    async def _track_task(i, stock):
        nonlocal completed_count
        res = await fetch_single_stock(
            i,
            stock,
            total,
            sem,
            client,
            session,
            capture_store=capture_store,
            cohort=cohort,
            run_id=run_id,
        )
        completed_count += 1
        pct = (completed_count / total) * 100 if total > 0 else 100.0
        bar_len = 25
        filled = int(bar_len * completed_count // total) if total > 0 else bar_len
        bar = "█" * filled + "░" * (bar_len - filled)
        if sys.stdout.isatty():
            sys.stdout.write(
                f"\r⏳ [수집 진행] [{bar}] {pct:5.1f}% ({completed_count}/{total})"
            )
            sys.stdout.flush()
        return res

    tasks = [_track_task(i, stock) for i, stock in enumerate(stock_list)]
    all_res = await asyncio.gather(*tasks)
    if total > 0 and sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()
    all_res = await requote_failed_quotes(stock_list, list(all_res), client, session, sem, now_fn=now_fn, capture_store=capture_store, cohort=cohort, run_id=run_id)
    logger.info("[DATA] stage=realtime_quote_batch n_rows=%d n_quote_failed=%d", total, sum(1 for r, _f, _o in all_res if r.get(QUOTE_FAILED_COL)))

    results = [r for r, _f, _o in all_res]
    failed_info = [
        (stock_list[i]["name"], stock_list[i]["code"], f)
        for i, (r, f, _o) in enumerate(all_res)
        if f
    ]
    orderbook_rows: list[dict] = []
    for _r, _f, o in all_res:
        if o:
            orderbook_rows.extend(o)
    snapshot_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    try:
        append_orderbook_snapshots(orderbook_rows, snapshot_date)
    except Exception as e:
        logger.warning("[DATA] Orderbook decision snapshot persist failed: %s", e)

    logger.info(f"{Colors.GREEN}✅ 데이터 수집 완료{Colors.RESET}")
    return results, failed_info


def _split_stock_list_evenly(stock_list: list[dict], n: int) -> list[list[dict]]:
    """원래 순서를 보존한 채 n개의 연속 구간으로 최대한 균등 분할한다."""
    base, rem = divmod(len(stock_list), n)
    chunks: list[list[dict]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        chunks.append(stock_list[start:start + size])
        start += size
    return chunks


async def fetch_all_stock_data_sharded(
    stock_list: list[dict],
    clients: list,
    session,
    *,
    now_fn: Callable[[], datetime] | None = None,
    capture_store: CaptureStore | None = None,
    cohort: Cohort | None = None,
    run_id: str | None = None,
) -> tuple[list[dict], list[tuple[str, str, list[str]]]]:
    """여러 KIS 키로 분할 수집하고 원래 순서로 병합한다."""
    if len(clients) <= 1:
        return await fetch_all_stock_data(stock_list, clients[0], session, now_fn=now_fn, capture_store=capture_store, cohort=cohort, run_id=run_id)
    chunks = _split_stock_list_evenly(stock_list, len(clients))
    pairs = [(chunk, client) for chunk, client in zip(chunks, clients) if chunk]
    if not pairs:
        return [], []
    gathered = await asyncio.gather(*[fetch_all_stock_data(chunk, client, session, now_fn=now_fn, capture_store=capture_store, cohort=cohort, run_id=run_id) for chunk, client in pairs])
    results: list[dict] = []
    failed_info: list = []
    for r, f in gathered:
        results.extend(r)
        failed_info.extend(f)
    return results, failed_info


def persist_daily_snapshot(df: pd.DataFrame, snapshot_date: str) -> int:
    """일일 wide 스냅샷을 아카이브 저장소에 직접 기록한다.

    Args:
        df: admitted 플래그를 포함한 wide 단면 프레임.
        snapshot_date: 스냅샷 날짜 (YYYY-MM-DD).

    Returns:
        저장된 행수. 빈 프레임은 저장 없이 0을 반환한다.
    """
    if df.empty:
        return 0
    return archive.upsert_archive_snapshot(df, snapshot_date=snapshot_date)


async def main(force: bool = False):
    # auction_capture is a separate command; it loads this job's published cohort
    # through CaptureStore rather than being awaited inside initial collect.
    from aiohttp.resolver import ThreadedResolver

    data_kwargs = kis_data_client_kwargs()
    _validate_hts_id(data_kwargs["hts_id"])
    _validate_decision_window(datetime.now(ZoneInfo("Asia/Seoul")), force=force)

    # aiohttp 세션 설정 강화 (네트워크 안정성 향상 + DNS 해결)
    timeout = aiohttp.ClientTimeout(
        total=60,  # 전체 요청 타임아웃
        connect=10,  # 연결 타임아웃
        sock_read=30,  # 소켓 읽기 타임아웃
    )
    connector = aiohttp.TCPConnector(
        limit=20,  # 최대 동시 연결 수
        ttl_dns_cache=300,  # DNS 캐시 TTL (5분)
        force_close=False,  # Keep-Alive 유지
        resolver=ThreadedResolver(),  # [Fix] Windows aiodns 이슈 방지용 표준 리졸버 사용
    )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 1. 클라이언트 초기화 및 토큰 확보
        client = KisApiClient(
            data_kwargs["app_key"],
            data_kwargs["app_secret"],
            data_kwargs["account_id"],
            data_kwargs["hts_id"],
            token_file=data_kwargs["token_file"],
        )
        await client.ensure_token(session)
        decision_shard_clients = [client]
        for shard_kwargs in kis_decision_shard_client_kwargs()[1:]:
            shard_client = KisApiClient(
                shard_kwargs["app_key"],
                shard_kwargs["app_secret"],
                shard_kwargs["account_id"],
                shard_kwargs["hts_id"],
                token_file=shard_kwargs["token_file"],
            )
            await shard_client.ensure_token(session)
            decision_shard_clients.append(shard_client)

        snapshot_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        kiwoom_client = build_kiwoom_scan_client()
        toss_client = build_toss_scan_client()
        try:
            await _validate_trading_day(client, session, snapshot_date, force=force)
        except NonTradingDayError:
            # 휴장일은 장애가 아니다: 정상 종료해 OnFailure 오탐 알림과 하위 단계 오류를 막는다
            logger.info("[DATA] stage=collect status=SKIP reason=non_trading_day date=%s", snapshot_date)
            return

        # 2. 시장 지수 조회 (병렬 gather)
        res_kospi, res_kosdaq = await asyncio.gather(
            client.get_market_index_rate(session, "0001"),
            client.get_market_index_rate(session, "1001"),
        )

        kospi_rate = parse_market_index_rate(res_kospi)
        kosdaq_rate = parse_market_index_rate(res_kosdaq)

        # 3. 후보 종목 리스트 확보 (자동 비용축 스캔 단일 경로, Toss 폴백 포함)
        raw_enabled = bool(settings.COLLECTION_RAW_ENABLED)
        store = CaptureStore(_capture_root()) if raw_enabled else None
        run_id = f"decision-{snapshot_date}-{uuid.uuid4().hex[:8]}" if raw_enabled else None
        trading_day = date.fromisoformat(snapshot_date)
        scan_observer = None
        if store is not None and run_id is not None:
            def scan_observer(
                payload: Any,
                metadata: Mapping[str, str],
                started: datetime,
                received: datetime,
                page_index: int,
                attempt: int,
            ) -> None:
                assert store is not None
                assert run_id is not None
                meta = dict(metadata) if isinstance(metadata, Mapping) else {}
                vendor = str(meta.get("vendor") or "scan").strip() or "scan"
                endpoint = str(meta.get("endpoint") or meta.get("scope") or "scan").strip() or "scan"
                _append_with_unique_attempt(
                    store,
                    CapturedResponse(
                        context=CaptureContext(
                            trading_date=trading_day,
                            run_id=run_id,
                            dataset=CaptureDataset.SCAN,
                            vendor=vendor,
                            endpoint=endpoint,
                            symbol=None,
                            venue="UNKNOWN",
                            session="regular",
                            capture_reason="decision-input",
                            cohort_id=None,
                            scheduled_at=None,
                        ),
                        request_started_at=started,
                        received_at=received,
                        payload=dict(payload) if isinstance(payload, dict) else None,
                        status=CaptureStatus.COMPLETE if isinstance(payload, dict) else CaptureStatus.FAILED,
                        source_timestamp=None,
                        source_published_at=None,
                        page_index=int(page_index),
                        attempt_index=int(attempt),
                        continuation={str(k): str(v) for k, v in meta.items()},
                        error_type=None if isinstance(payload, dict) else "vendor_failure",
                    ),
                )

        stock_list = await resolve_daily_candidates(client, session, kiwoom_client=kiwoom_client, toss_client=toss_client, on_page=scan_observer)
        if not stock_list:
            logger.info(f"{Colors.YELLOW}⚠ 자동 스캔 후보가 없습니다.{Colors.RESET}")
            return
        scanned_codes = [str(row["code"]) for row in stock_list]
        eligible_codes = await resolve_eligible_codes(client, session, pd.Timestamp(snapshot_date))
        cohort = None
        if store is not None and run_id is not None:
            eligible_in_scan = [c for c in scanned_codes if c in eligible_codes]
            rejections = {c: "not_listed_in_panel" for c in scanned_codes if c not in eligible_codes}
            cohort = build_cohort(
                trading_day,
                scanned_codes,
                eligible_in_scan,
                rejections,
                eligibility_rule_version=_DECISION_ELIGIBILITY_RULE_VERSION,
            )
        stock_list = filter_eligible_candidates(stock_list, eligible_codes)

        logger.info(
            f"{Colors.BOLD}🚀 [1/3] 후보 종목 스캔 (Kiwoom / KIS){Colors.RESET}\n"
            f"   - 대상: {Colors.CYAN}{len(stock_list)}{Colors.RESET}개 종목 포착 (등락률/유니버스 필터)"
        )

        # 4. 상세 데이터 수집
        logger.info(f"\n{Colors.BOLD}⏳ [2/3] 실시간 단면 데이터 수집{Colors.RESET}")
        results, failed_info = await fetch_all_stock_data_sharded(stock_list, decision_shard_clients, session, capture_store=store, cohort=cohort, run_id=run_id)

        # 5. wide 단면 구성 후 PIT admitted 플래그 부여 및 저장소 직접 기록
        logger.info(f"\n{Colors.BOLD}📊 [3/3] 유니버스 적격성(Admission) 평가 및 저장{Colors.RESET}")
        capture_ts = pd.Timestamp.now(tz="Asia/Seoul")
        df = pd.DataFrame(results)
        if "snapshot_timestamp" in df.columns:
            df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"]).fillna(capture_ts)
        else:
            df["snapshot_timestamp"] = capture_ts
        df = flag_cost_aware_admission(df, decision_date=pd.Timestamp(snapshot_date))
        index_failed = False
        if kospi_rate is None:
            df["kospi"] = float("nan")
            index_failed = True
        else:
            df["kospi"] = kospi_rate
        if kosdaq_rate is None:
            df["kosdaq"] = float("nan")
            index_failed = True
        else:
            df["kosdaq"] = kosdaq_rate

        # V-KOSPI만 부착 (V-KOSDAQ 조회 제거)
        try:
            from src.api.kis.indicators import fetch_index_and_calculate_volatility

            (vkospi_val, _vkospi_chg) = await fetch_index_and_calculate_volatility(
                "1028", session=session
            )
        except Exception:
            vkospi_val = float("nan")
            index_failed = True
        df["v_kospi"] = round(float(vkospi_val), 2)
        breadth_failed = False
        try:
            from src.data.panel_integrity import compute_latest_market_breadth, load_price_panel

            panel, _prov = load_price_panel(settings.PRICE_HISTORY_PARQUET_PATH)
            breadth_val = compute_latest_market_breadth(panel, snapshot_date)
            if breadth_val != breadth_val:
                breadth_failed = True
        except Exception:
            breadth_val = float("nan")
            breadth_failed = True
        df["market_breadth"] = breadth_val
        df["시장폭_실패"] = breadth_failed
        df["지수_실패"] = index_failed

        df[PRICE_ANOMALY_COL] = flag_price_anomaly(df).to_numpy()
        enrichment_completed_at = datetime.now(ZoneInfo("Asia/Seoul"))
        df["feature_available_timestamp"] = enrichment_completed_at
        if store is not None and cohort is not None and run_id is not None:
            entries = (
                CoverageEntry(
                    symbol=None,
                    dataset=CaptureDataset.PRICE,
                    venue="KRX",
                    session="regular",
                    scheduled_at=None,
                    status=CaptureStatus.COMPLETE,
                    rows=len(df),
                    first_event_time=None,
                    last_event_time=None,
                    reason="decision-input",
                    raw_refs=(),
                ),
            )
            store.publish_decision(df, cohort=cohort, run_id=run_id, completed_at=enrichment_completed_at, entries=entries)
            coverage_report = check_realtime_collection_coverage(df)
        else:
            coverage_report = check_realtime_collection_coverage(df)
        logger.info(
            "[DATA] stage=realtime_coverage n_raw=%d n_degraded=%d coverage=%.4f",
            coverage_report["n_raw"], coverage_report["n_degraded"], coverage_report["coverage"],
        )

        stored_rows = persist_daily_snapshot(df, snapshot_date)

        n_raw = len(df)
        n_admitted = int(df["admitted"].sum()) if "admitted" in df.columns else 0
        n_failed = len(failed_info)
        success_count = n_raw - n_failed

        divider = "─" * 60
        box_top = "━" * 60
        logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")
        logger.info(f" {Colors.GREEN}{Colors.BOLD}📋 [데이터 수집 & 적재 요약]{Colors.RESET} ({snapshot_date})")
        logger.info(f"{Colors.BOLD}{divider}{Colors.RESET}")
        logger.info(f"   • 스캔 및 수집 시도 : {n_raw:>3} 종목 (성공: {Colors.GREEN}{success_count}{Colors.RESET}, 실패: {Colors.RED if n_failed > 0 else Colors.GRAY}{n_failed}{Colors.RESET})")
        logger.info(f"   • 유니버스 적격 통과: {Colors.CYAN}{Colors.BOLD}{n_admitted:>3}{Colors.RESET} 종목 (비용/거래대금 필터 통과)")
        logger.info(f"   • 스냅샷 저장소 적재: {Colors.GREEN}{stored_rows:>3}{Colors.RESET} 행 적재 완료")
        logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main(force="--force" in sys.argv))  # pragma: no cover
