"""매 평일 EOD 이후 스케줄 실행되는 자동화 점검 + 일일 요약 발송.

평일마다 정확히 한 통의 요약(정상/경고/휴장일)을 보낸다. 요약이 오지 않는 것
자체가 스케줄러 중단 신호가 되도록 누락이 없어도 침묵하지 않는다. 누락 단계가
있어도 비정상 종료하지 않는다(가시화 전용).
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.api.kis.key_pool import load_kis_env, read_token_issued_date, resolve_host_issued_credentials, token_cache_path
from src.config.base import TOPK_DECISIONS_PARQUET_NAME
from src.config.collection import CollectionSettings
from src.config.market_session import (
    DECISION_WINDOW_END_HHMMSS,
    DECISION_WINDOW_START_HHMMSS,
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_REGULAR,
    KRX_AFTERMARKET_HOUR_CEIL,
    KRX_AFTERMARKET_HOUR_FLOOR,
    KRX_AFTERMARKET_START_DATE,
    NXT_AFTERMARKET_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_FLOOR,
    NXT_PREMARKET_HOUR_CEIL,
    NXT_PREMARKET_HOUR_FLOOR,
)
from src.daily.archive import fetch_archive_snapshot
from src.daily.archive_intraday import resolve_previous_archive_date
from src.daily.auction_capture import close_rounds, program_rounds
from src.data.altdata_health import AltdataVerdict, altdata_verdict
from src.data.capture_contracts import (
    SEOUL,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
    SessionClock,
)
from src.data.capture_store import (
    CaptureStore,
    resolve_capture_root,
)
from src.data.capture_store import (
    resolve_capture_root as _capture_root,
)
from src.data.intraday_store import intraday_partition_path, tick_partition_path
from src.data.io_utils import atomic_write_text
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.tick_bar_consistency import (
    CERTIFIED_SOURCE_DIFF_MAX_SHARE,
    CertifiedTickShortfall,
    TickBarRelation,
    classify_certified_tick_shortfall,
    classify_tick_bar_volume,
    comparable_bar_volumes,
    summed_tick_volumes,
)
from src.data.trading_calendar import (
    DAY_HOLIDAY,
    DAY_TRADING,
    DAY_UNKNOWN,
    DAY_WEEKEND,
    classify_day,
)
from src.execution.paper_broker import PaperLedger
from src.processing.schema import CLOSE_CONFIRMED_COL
from src.tools.alerts import dispatch_digest, drain_alert_outbox
from src.tools.expiry_notices import CALENDAR_EXPIRY_NAME, CALENDAR_RENEW_HINT, evaluate_expiries
from src.tools.offsite_backup import BACKUP_INFO_ISSUES, REPORT_RELPATH, backup_staleness_issues
from src.tools.run_outcome import RUN_OUTCOME_OK, load_run_outcomes, record_run_outcome
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)


def _format_count_kr(n: int) -> str:
    """건수를 한국어 만/건 단위로 축약 포맷팅한다."""
    if n >= 10_000:
        return f"{n / 10_000:.1f}만 건"
    return f"{n:,}건"


def _extract_paper_summary(snapshot_date: str) -> tuple[str, str]:
    """사람이 읽기 좋은 NAV 및 당일 매수 진입 종목 요약을 추출한다."""
    nav_str = "확인 불가"
    entry_str = "당일 진입 없음"
    try:
        nav_path = Path(settings.PAPER_DIR) / "nav.parquet"
        if nav_path.exists():
            nav_df = pd.read_parquet(nav_path)
            if not nav_df.empty:
                latest = nav_df.iloc[-1]
                val = latest.get("cash", latest.get("nav", None))
                if val is not None and pd.notna(val):
                    nav_str = f"{int(val):,}원"
        fills_path = Path(settings.PAPER_DIR) / "fills.parquet"
        if fills_path.exists():
            fdf = pd.read_parquet(fills_path)
            if not fdf.empty and "decision_date" in fdf.columns:
                day_fills = fdf[(fdf["decision_date"].astype(str) == str(snapshot_date)) & (fdf["side"] == "buy")]
                if not day_fills.empty and "symbol" in day_fills.columns:
                    syms = list(dict.fromkeys(day_fills["symbol"].astype(str).tolist()))
                    entry_str = f"{', '.join(syms[:3])} ({len(syms)}종목)"
    except Exception as exc:
        logger.debug("[SYS] stage=daily_audit extract_paper failed: %s", exc)
    return nav_str, entry_str


def _extract_intraday_summary(snapshot_date: str) -> tuple[str, str]:
    """사람이 읽기 좋은 당일 정규장 1분봉 및 체결 틱 건수 요약을 추출한다."""
    bars_str = "0건"
    ticks_str = "0건"
    try:
        import pyarrow.parquet as pq

        bp = intraday_partition_path(1, snapshot_date, "regular")
        if bp.exists():
            cnt = pq.ParquetFile(bp).metadata.num_rows
            bars_str = _format_count_kr(cnt)
        tp = tick_partition_path(snapshot_date, "regular")
        if tp.exists():
            cnt = pq.ParquetFile(tp).metadata.num_rows
            ticks_str = _format_count_kr(cnt)
    except Exception as exc:
        logger.debug("[SYS] stage=daily_audit extract_intraday failed: %s", exc)
    return bars_str, ticks_str


_CHART_DATASETS: tuple[CaptureDataset, CaptureDataset] = (CaptureDataset.MINUTE_BARS, CaptureDataset.TRADE_TICKS)
# Live pagination proofs plus the tape sweep's vendor-certified completion proofs: a tape-recovered entry
# (`tape_complete`, `tape_bracketed`) is as terminal as a live walk that exhausted its pages.
_TERMINAL_REASONS: frozenset[str] = frozenset(
    {"exhausted", "crossed_target_date", "tape_complete", "tape_bracketed"}
)
_SLOW_DATA_DUE_HHMMSS: str = "213500"
_SNAPSHOT_CATCHUP_CUTOFF_HOUR: int = 12
AUDIT_STEPS: tuple[str, ...] = (
    "archive",
    "close_confirmed",
    "decision",
    "paper_entry",
    "paper_exit",
    "minute_bars",
    "intraday_complete",
    "price_history_fresh",
)
SYSTEMCTL_TIMEOUT_SEC: int = 30


def _column_dates(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path)
    if frame.empty or column not in frame.columns:
        return set()
    return {str(value)[:10] for value in frame[column].dropna().tolist()}


def _entry_fill_dates(path: Path) -> set[str]:
    if not path.exists():
        return set()
    fills = pd.read_parquet(path)
    if fills.empty or not {"side", "decision_date"}.issubset(fills.columns):
        return set()
    buys = fills.loc[fills["side"].astype(str) == "buy", "decision_date"]
    return {str(value)[:10] for value in buys.dropna().tolist()}


def _price_history_fresh(snapshot_date: str) -> bool:
    # 감사가 당일 야간 적재(21:30) 전에 돌 수 있으므로 직전 아카이브 영업일까지 적재됐는지로 판정한다
    previous = resolve_previous_archive_date(snapshot_date)
    if previous is None:
        return True
    path = Path(settings.PRICE_HISTORY_PARQUET_PATH)
    if not path.exists():
        return False
    dates = pd.read_parquet(path, columns=["date"])["date"]
    if dates.empty:
        return False
    return str(pd.Timestamp(dates.max()).date()) >= previous


def audit_daily_completeness(snapshot_date: str) -> dict[str, bool]:
    """해당 일자의 단계별 산출물 존재 여부를 AUDIT_STEPS 키 bool 딕셔너리로 반환한다.

    존재 판정은 파일/행 존재만으로 하며 값 검증은 하지 않는다. 결정은 영속된 top-k 결정 또는 predict 실행결과 OK(정상 무결정 포함)일 때만 수행으로 인정하고, 페이퍼 무결정 기록은 페이퍼 진입 단계에만 인정한다. paper_exit는 당일 이전 결정의 미청산 로트가 없을 때 True이다.

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD, KST).

    Returns:
        AUDIT_STEPS 각 단계의 수행 여부.
    """
    frame = fetch_archive_snapshot(snapshot_date)
    archive_ok = frame is not None and len(frame) > 0
    close_confirmed_ok = (
        archive_ok
        and CLOSE_CONFIRMED_COL in frame.columns
        and bool(frame[CLOSE_CONFIRMED_COL].fillna(False).astype(bool).any())
    )
    topk_dates = _column_dates(Path(settings.PARQUET_DIR) / TOPK_DECISIONS_PARQUET_NAME, "decision_date")
    no_decision_dates = _column_dates(Path(settings.PAPER_DIR) / "decisions.parquet", "decision_date")
    entry_dates = _entry_fill_dates(Path(settings.PAPER_DIR) / "fills.parquet")
    outcomes = load_run_outcomes(snapshot_date)
    try:
        open_positions = PaperLedger(root=Path(settings.PAPER_DIR)).load_open_positions()
        # 감사(21:20)는 당일 09:00 청산 이후이므로 당일 이전 결정 로트가 남아 있으면 청산 누락이다
        stale_exit = bool((open_positions["decision_date"].astype(str) < snapshot_date).any())
    except (KeyError, ValueError) as exc:
        # 스키마가 깨졌거나 로트 링크를 위반한 원장은 청산 상태를 보증할 수 없으므로 누락으로 보고한다
        logger.warning(
            "[DATA] stage=daily_audit step=paper_exit status=LEDGER_INVALID reason=%s: %s", type(exc).__name__, exc
        )
        stale_exit = True
    return {
        "archive": bool(archive_ok),
        "close_confirmed": bool(close_confirmed_ok),
        "decision": snapshot_date in topk_dates or outcomes.get("predict") == RUN_OUTCOME_OK,
        "paper_entry": snapshot_date in entry_dates or snapshot_date in no_decision_dates,
        "paper_exit": not stale_exit,
        "minute_bars": bool(intraday_partition_path(1, snapshot_date, "regular").exists()),
        "price_history_fresh": _price_history_fresh(snapshot_date),
    }


def list_failed_kca_units(run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[str]:
    """systemd 유저 매니저에서 failed 상태인 kca-* 유닛 이름을 정렬해 반환한다.

    systemctl 자체를 실행할 수 없으면 빈 목록으로 숨기지 않고 표식 문자열을 반환해
    요약에 드러나게 한다.

    Args:
        run_fn: subprocess.run 호환 실행기(테스트 주입용).

    Returns:
        실패 유닛 이름 목록 또는 조회 불가 표식 1건.
    """
    try:
        result = run_fn(
            ["systemctl", "--user", "list-units", "--failed", "--plain", "--no-legend", "kca-*"],
            capture_output=True,
            text=True,
            timeout=SYSTEMCTL_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"<systemctl unavailable: {type(exc).__name__}>"]
    return sorted(line.split()[0] for line in result.stdout.splitlines() if line.strip())


def list_stale_kis_tokens(
    snapshot_date: str,
    *,
    env: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
    allow_newer: bool = False,
) -> list[str]:
    """이 호스트가 발급 책임을 지는 KIS 키 중 당일(snapshot_date, KST) 토큰이 없는 것을 반환한다.

    발급 자체는 kca-kis-token-warmup 책임이다. 여기서는 그 결과가 선언된 모든
    키(배정 슬롯 + 비풀 선언 키)에 실제로 도달했는지만 가시화한다(부분 성공이
    exit 0으로 숨는 것을 방지).

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD, KST).
        env: KIS 자격증명 env(테스트 주입용). None이면 호스트 .env를 읽는다.
        cache_dir: 토큰 캐시 디렉터리(테스트 주입용). None이면 settings.KIS_TOKEN_CACHE_DIR.
        allow_newer: Accept a later issuance when reconciling historical audit coverage.

    Returns:
        당일 발급 기록이 없는 슬롯 이름(예: "DATA_3", "PRIMARY") 목록, 정렬됨.
        키 선언 자체가 어긋나면(자격증명 누락 등) 숨기지 않고 표식 1건을 담아 반환한다.
    """
    source = env if env is not None else load_kis_env(Path(settings.BASE_DIR) / ".env")
    cache = cache_dir if cache_dir is not None else settings.KIS_TOKEN_CACHE_DIR
    try:
        creds = resolve_host_issued_credentials(source)
    except ValueError as exc:
        return [f"<kis host key config invalid: {exc}>"]
    stale = []
    for cred in creds:
        issued = read_token_issued_date(token_cache_path(cred.app_key, cache))
        if issued != snapshot_date and not (allow_newer and issued is not None and issued > snapshot_date):
            stale.append(cred.slot)
    return sorted(stale)


def _collection_issue(dataset: str, count: int, reason: str) -> str:
    return f"collection:{dataset}:{count}:{reason}"


def _terminal_proof(reason: str) -> bool:
    return reason.split(":")[0] in _TERMINAL_REASONS


def _outside_regular_session(entry: CoverageEntry, session_clock: SessionClock) -> bool:
    if entry.first_event_time is not None and entry.first_event_time < session_clock.open_at:
        return True
    return entry.last_event_time is not None and entry.last_event_time > session_clock.close_at


def _regular_entry_settled(dataset: CaptureDataset, entry: CoverageEntry) -> bool:
    # 거래정지 등 정규장 무거래 종목은 벤더 증명(no_trades_in_window)이 있는 틱 항목만 완료로 인정한다
    if entry.status == CaptureStatus.COMPLETE:
        return True
    return (
        dataset is CaptureDataset.TRADE_TICKS
        and entry.status == CaptureStatus.NO_TRADES
        and entry.reason == "no_trades_in_window"
    )


def _audit_regular_bars(
    manifests: tuple[CaptureManifest, ...],
    expected: tuple[str, ...],
    session_clock: SessionClock,
) -> list[str]:
    issues: list[str] = []
    for dataset in _CHART_DATASETS:
        label = "charts" if dataset is CaptureDataset.MINUTE_BARS else "ticks"
        by_symbol: dict[str, list[CoverageEntry]] = {}
        for manifest in manifests:
            if manifest.status == CaptureStatus.PENDING:
                continue
            for entry in manifest.entries:
                if entry.dataset is not dataset or entry.session != "regular" or entry.symbol is None:
                    continue
                by_symbol.setdefault(entry.symbol, []).append(entry)
        missing = sum(1 for symbol in expected if not by_symbol.get(symbol))
        incomplete = sum(
            1
            for symbol in expected
            if by_symbol.get(symbol) and not any(_regular_entry_settled(dataset, e) for e in by_symbol[symbol])
        )
        unproven = 0
        violations = 0
        for symbol in expected:
            entries = by_symbol.get(symbol, [])
            complete = [e for e in entries if e.status == CaptureStatus.COMPLETE]
            if entries and complete and any(not _terminal_proof(e.reason) for e in complete):
                unproven += 1
            if any(_outside_regular_session(e, session_clock) for e in entries):
                violations += 1
        if missing:
            issues.append(_collection_issue(label, missing, "missing_entries"))
        if incomplete:
            issues.append(_collection_issue(label, incomplete, "incomplete_entries"))
        if unproven:
            issues.append(_collection_issue(label, unproven, "terminal_proof_missing"))
        if violations:
            issues.append(_collection_issue(label, violations, "session_violation"))
    return issues


def _audit_auction_sweeps(
    manifests: tuple[CaptureManifest, ...],
    expected: tuple[str, ...],
    profile: CollectionSettings,
    session_clock: SessionClock,
    audit_at: datetime,
) -> list[str]:
    issues: list[str] = []
    if audit_at >= session_clock.close_at:
        rounds = close_rounds(session_clock, int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS))
        prog_rounds = program_rounds(session_clock)
        actual: set[tuple[str, str, datetime]] = set()
        incomplete = 0
        for manifest in manifests:
            if manifest.context.capture_reason != "auction-close" or manifest.status == CaptureStatus.PENDING:
                continue
            for entry in manifest.entries:
                if entry.status != CaptureStatus.COMPLETE:
                    incomplete += 1
                if entry.symbol is not None and entry.scheduled_at is not None:
                    actual.add((entry.symbol, entry.dataset.value, entry.scheduled_at))
        missing = 0
        for symbol in expected:
            for slot in rounds:
                if (symbol, CaptureDataset.ORDERBOOK.value, slot) not in actual:
                    missing += 1
            for slot in prog_rounds:
                if (symbol, CaptureDataset.PROGRAM.value, slot) not in actual:
                    missing += 1
        if missing:
            issues.append(_collection_issue("auction_close", missing, "missing_entries"))
        if incomplete:
            issues.append(_collection_issue("auction_close", incomplete, "incomplete_entries"))
    open_due_at = session_clock.open_at + timedelta(seconds=int(profile.COLLECTION_OPEN_CONFIRM_SECONDS))
    if audit_at >= open_due_at:
        terminal = [
            m for m in manifests if m.context.capture_reason == "auction-open" and m.status != CaptureStatus.PENDING
        ]
        if not terminal:
            issues.append(_collection_issue("auction_open", 1, "missing_manifest"))
        else:
            bad = sum(1 for m in terminal for e in m.entries if e.status != CaptureStatus.COMPLETE)
            if bad:
                issues.append(_collection_issue("auction_open", bad, "incomplete_entries"))
    return issues


def _audit_slow_data(
    manifests: tuple[CaptureManifest, ...],
    trading_date: date,
    audit_at: datetime,
) -> tuple[str, ...]:
    due_at = datetime(
        trading_date.year,
        trading_date.month,
        trading_date.day,
        int(_SLOW_DATA_DUE_HHMMSS[0:2]),
        int(_SLOW_DATA_DUE_HHMMSS[2:4]),
        int(_SLOW_DATA_DUE_HHMMSS[4:6]),
        tzinfo=SEOUL,
    )
    if audit_at < due_at:
        return ()
    terminal = [
        m for m in manifests if m.context.capture_reason == "altdata-backfill" and m.status != CaptureStatus.PENDING
    ]
    if not terminal:
        return (_collection_issue("slow_data", 1, "missing_run"),)
    if not any(altdata_verdict(m) in (AltdataVerdict.COMPLETE, AltdataVerdict.DEGRADED) for m in terminal):
        return (_collection_issue("slow_data", len(terminal), "incomplete_run"),)
    return ()


def _audit_aftermarket_book(manifests: Sequence[CaptureManifest]) -> list[str]:
    """Audit evening aftermarket order-book manifests for presence and entry completeness.

    Args:
        manifests: Verified manifests of the audited date.

    Returns:
        Issue strings ``collection:aftermarket_book:<count>:<reason>`` with reasons
        missing (no aftermarket-book manifest) or incomplete (PARTIAL/FAILED entries).
    """
    terminal = [m for m in manifests if m.context.capture_reason == "aftermarket-book"]
    if not terminal:
        return [_collection_issue("aftermarket_book", 0, "missing")]
    bad = sum(1 for m in terminal for e in m.entries if e.status in (CaptureStatus.PARTIAL, CaptureStatus.FAILED))
    if bad:
        return [_collection_issue("aftermarket_book", bad, "incomplete")]
    return []


def audit_collection_manifests(
    trading_date: date,
    *,
    store: CaptureStore,
    profile: CollectionSettings,
    session_clock: SessionClock,
    audit_at: datetime,
) -> tuple[str, ...]:
    """Explain collection gaps against the owner-local expected population and schedule.

    Args:
        trading_date: Actual audited market date.
        store: Immutable artifacts and coverage manifests.
        profile: Enabled dataset and acquisition profiles.
        session_clock: Verified session times for this date.
        audit_at: Aware cutoff; future scheduled tasks are pending.
    Returns:
        Stable credential-free issue strings with dataset, count and reason.
    Raises:
        ValueError: Inconsistent date or naive audit cutoff.
    """
    if session_clock.trading_date != trading_date:
        raise ValueError(
            f"session clock trading_date {session_clock.trading_date.isoformat()} != trading_date {trading_date.isoformat()}"
        )
    if audit_at.tzinfo is None or audit_at.utcoffset() is None:
        raise ValueError("audit_at must be timezone-aware")
    try:
        manifests = store.read_manifests(trading_date.isoformat())
    except (OSError, ValueError):
        return (_collection_issue("manifest", 1, "unreadable_evidence"),)
    try:
        cohort = store.read_cohort(trading_date.isoformat(), available_by=audit_at)
    except FileNotFoundError:
        cohort = None
    auction_enabled = bool(profile.COLLECTION_AUCTION_ENABLED)
    altdata_enabled = bool(profile.COLLECTION_ALTDATA_ENABLED)
    if cohort is None and not manifests and not auction_enabled and not altdata_enabled:
        return ()
    issues: list[str] = []
    expected: tuple[str, ...] = cohort.eligible_symbols if cohort is not None else ()
    if cohort is None:
        issues.append(_collection_issue("cohort", 0, "missing_cohort"))
    decision_manifests = [m for m in manifests if m.cohort is not None]
    if decision_manifests:
        qualified = False
        tampered = False
        for manifest in decision_manifests:
            try:
                store.read_decision(trading_date.isoformat(), available_by=audit_at, run_id=manifest.context.run_id)
                qualified = True
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                tampered = True
        if tampered:
            issues.append(_collection_issue("decision", len(decision_manifests), "integrity_failure"))
        elif not qualified:
            issues.append(_collection_issue("decision", len(decision_manifests), "missing_decision_input"))
    if cohort is not None:
        issues.extend(_audit_regular_bars(manifests, expected, session_clock))
    if auction_enabled:
        issues.extend(_audit_auction_sweeps(manifests, expected, profile, session_clock, audit_at))
    else:
        issues.append(_collection_issue("auction", 0, "disabled"))
    if profile.COLLECTION_AFTERMARKET_BOOK_ENABLED and session_clock == SessionClock.standard(trading_date):
        issues.extend(_audit_aftermarket_book(manifests))
    if altdata_enabled:
        issues.extend(_audit_slow_data(manifests, trading_date, audit_at))
    else:
        issues.append(_collection_issue("slow_data", 0, "disabled"))
    return tuple(issues)


def _intraday_issue(session: str, count: int, reason: str) -> str:
    return f"intraday:{session}:{count}:{reason}"


def _hhmm_to_minutes(hhmmss: str) -> int:
    return int(hhmmss[0:2]) * 60 + int(hhmmss[2:4])


def expected_regular_stamps(clock: SessionClock) -> tuple[int, ...]:
    """Expected end-labeled regular-session 1m stamps (HHMMSS ints) for a session clock.

    Continuous trading produces one bar per minute ending at open+1m through
    the start of the closing call; the closing auction prints once at close.
    The closing-call length is DECISION_WINDOW_END - DECISION_WINDOW_START.
    For the standard clock this is 09:01..15:20 plus 15:30 (381 stamps).
    """
    call_minutes = _hhmm_to_minutes(DECISION_WINDOW_END_HHMMSS) - _hhmm_to_minutes(DECISION_WINDOW_START_HHMMSS)
    tick = clock.open_at + timedelta(minutes=1)
    last_continuous = clock.close_at - timedelta(minutes=call_minutes)
    stamps: list[int] = []
    while tick <= last_continuous:
        stamps.append(tick.hour * 10000 + tick.minute * 100 + tick.second)
        tick += timedelta(minutes=1)
    stamps.append(clock.close_at.hour * 10000 + clock.close_at.minute * 100 + clock.close_at.second)
    return tuple(stamps)


def expected_krx_aftermarket_stamps() -> tuple[int, ...]:
    """Expected KRX aftermarket 1m stamps: KRX_AFTERMARKET_HOUR_FLOOR..CEIL inclusive (241)."""
    start = _hhmm_to_minutes(KRX_AFTERMARKET_HOUR_FLOOR)
    end = _hhmm_to_minutes(KRX_AFTERMARKET_HOUR_CEIL)
    return tuple((minute // 60) * 10000 + (minute % 60) * 100 for minute in range(start, end + 1))


def _read_stored_partition(snapshot_date: str, session: str) -> pd.DataFrame | None:
    path = intraday_partition_path(1, snapshot_date, session)
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=["symbol", "ts_hms"])


def _stamps_by_symbol(frame: pd.DataFrame) -> dict[str, set[int]]:
    symbols = frame["symbol"].astype(str).tolist()
    raw_stamps = frame["ts_hms"].tolist()
    by_symbol: dict[str, set[int]] = {}
    for index in range(len(frame)):
        by_symbol.setdefault(str(symbols[index]), set()).add(int(raw_stamps[index]))
    return by_symbol


def _audit_dense_partition(
    session: str,
    frame: pd.DataFrame | None,
    expected_stamps: tuple[int, ...],
    expected_symbols: tuple[str, ...],
) -> list[str]:
    if frame is None:
        return [_intraday_issue(session, 1, "missing_partition")]
    expected_set = set(expected_stamps)
    by_symbol = _stamps_by_symbol(frame)
    present = set(by_symbol)
    issues: list[str] = []
    missing_symbols = sorted(symbol for symbol in expected_symbols if symbol not in present)
    if missing_symbols:
        issues.append(_intraday_issue(session, len(missing_symbols), "missing_symbols"))
    missing_bar_symbols = sorted(symbol for symbol in present if not expected_set.issubset(by_symbol[symbol]))
    first_missing: list[int] = []
    if missing_bar_symbols:
        issues.append(_intraday_issue(session, len(missing_bar_symbols), "missing_bars"))
        union_missing: set[int] = set()
        for symbol in missing_bar_symbols:
            union_missing |= expected_set - by_symbol[symbol]
        first_missing = sorted(union_missing)[:5]
    unexpected_symbols = sorted(symbol for symbol in present if by_symbol[symbol] - expected_set)
    if unexpected_symbols:
        issues.append(_intraday_issue(session, len(unexpected_symbols), "unexpected_bars"))
    if issues:
        implicated = set(missing_symbols) | set(missing_bar_symbols) | set(unexpected_symbols)
        logger.debug(
            "[DATA] stage=daily_audit step=intraday_complete session=%s status=FAIL reasons=%s symbols=%d first_missing=%s",
            session,
            ",".join(issue.split(":")[-1] for issue in issues),
            len(implicated),
            first_missing,
        )
    return issues


def _audit_sparse_partition(
    session: str,
    frame: pd.DataFrame | None,
    floor_hhmmss: str,
    ceil_hhmmss: str,
) -> list[str]:
    if frame is None:
        return [_intraday_issue(session, 1, "missing_partition")]
    floor = int(floor_hhmmss)
    ceil = int(ceil_hhmmss)
    by_symbol = _stamps_by_symbol(frame)
    # 분봉 시각은 해당 분의 끝으로 표기되므로 ceil 시각 봉(예: 20:00:00 = 19:59~20:00)은 세션 안이다 -- 수집기 창([floor, ceil])과 동일
    bad_symbols = sorted(
        symbol for symbol, stamps in by_symbol.items() if any(stamp < floor or stamp > ceil for stamp in stamps)
    )
    if not bad_symbols:
        return []
    window_breaks = sorted(
        {stamp for symbol in bad_symbols for stamp in by_symbol[symbol] if stamp < floor or stamp > ceil}
    )[:5]
    logger.debug(
        "[DATA] stage=daily_audit step=intraday_complete session=%s status=FAIL reasons=out_of_window symbols=%d first_missing=%s",
        session,
        len(bad_symbols),
        window_breaks,
    )
    return [_intraday_issue(session, len(bad_symbols), "out_of_window")]


def audit_intraday_partitions(
    trading_date: date,
    *,
    clock: SessionClock,
    session_kind: SessionKind,
    cohort_symbols: tuple[str, ...],
    read_partition: Callable[[str], pd.DataFrame | None] | None = None,
) -> tuple[str, ...]:
    """Audit stored 1m partitions for presence, cohort coverage and grid completeness.

    Manifests prove what was attempted; this audit proves what was stored.
    Dense sessions (regular, krx_aftermarket) must hold every expected stamp
    for every expected symbol; sparse NXT sessions must exist and stay inside
    their session window.

    Args:
        trading_date: Audited KST date.
        clock: Verified session clock for the date.
        session_kind: Resolved session status; aftermarket grid checks are
            skipped (with an explicit issue) on non-STANDARD days.
        cohort_symbols: Declared eligible symbols of the day.
        read_partition: session -> frame with symbol and ts_hms, None when the
            partition file is absent (injectable for tests). None reads parquet
            with column pruning (symbol, ts_hms).

    Returns:
        Issue strings `intraday:<session>:<count>:<reason>` with reasons
        missing_partition, missing_symbols, missing_bars, unexpected_bars,
        out_of_window, aftermarket_unverified; empty when complete.
    """
    day_str = trading_date.isoformat()
    sessions = [INTRADAY_SESSION_REGULAR, INTRADAY_SESSION_NXT_PREMARKET, INTRADAY_SESSION_NXT_AFTERMARKET]
    if day_str >= KRX_AFTERMARKET_START_DATE:
        sessions.append(INTRADAY_SESSION_KRX_AFTERMARKET)
    reader = (
        read_partition if read_partition is not None else (lambda session: _read_stored_partition(day_str, session))
    )
    frames = {session: reader(session) for session in sessions}
    issues: list[str] = []
    issues.extend(
        _audit_dense_partition(
            INTRADAY_SESSION_REGULAR,
            frames[INTRADAY_SESSION_REGULAR],
            expected_regular_stamps(clock),
            tuple(cohort_symbols),
        )
    )
    if session_kind is not SessionKind.STANDARD:
        issues.append(_intraday_issue("aftermarket", 0, "aftermarket_unverified"))
        return tuple(issues)
    issues.extend(
        _audit_sparse_partition(
            INTRADAY_SESSION_NXT_PREMARKET,
            frames[INTRADAY_SESSION_NXT_PREMARKET],
            NXT_PREMARKET_HOUR_FLOOR,
            NXT_PREMARKET_HOUR_CEIL,
        )
    )
    issues.extend(
        _audit_sparse_partition(
            INTRADAY_SESSION_NXT_AFTERMARKET,
            frames[INTRADAY_SESSION_NXT_AFTERMARKET],
            NXT_AFTERMARKET_HOUR_FLOOR,
            NXT_AFTERMARKET_HOUR_CEIL,
        )
    )
    if INTRADAY_SESSION_KRX_AFTERMARKET in frames:
        regular_frame = frames[INTRADAY_SESSION_REGULAR]
        regular_symbols = (
            tuple(sorted({str(value) for value in regular_frame["symbol"].astype(str).tolist()}))
            if regular_frame is not None
            else ()
        )
        issues.extend(
            _audit_dense_partition(
                INTRADAY_SESSION_KRX_AFTERMARKET,
                frames[INTRADAY_SESSION_KRX_AFTERMARKET],
                expected_krx_aftermarket_stamps(),
                regular_symbols,
            )
        )
    return tuple(issues)


def audit_bar_value_consistency(
    trading_date: date,
    *,
    sessions: Sequence[str],
    read_partition: Callable[[str], pd.DataFrame | None] | None = None,
) -> tuple[str, ...]:
    """Audit stored 1m bars for traded-value consistency with their own volume and price range.

    Grid audits prove bars exist; this proves their values are physically possible. LS rows are exempt
    because LS reports its own per-bar value in million-KRW units with occasional vendor attribution
    noise; every other vendor's value is produced by our normalizer and must satisfy the bound exactly.

    Args:
        trading_date: Audited KST date.
        sessions: Session partitions to check.
        read_partition: session -> frame with vendor, volume, value_krw, low, high (None when absent);
            None reads parquet with column pruning.

    Returns:
        Issue strings `intraday:<session>:<count>:value_out_of_range` (count = violating rows); empty when clean.
    """
    day_str = trading_date.isoformat()

    def _default_reader(session: str) -> pd.DataFrame | None:
        path = intraday_partition_path(1, day_str, session)
        if not path.exists():
            return None
        return pd.read_parquet(path, columns=["vendor", "volume", "value_krw", "low", "high"])

    reader = read_partition if read_partition is not None else _default_reader
    issues: list[str] = []
    for session in sessions:
        frame = reader(session)
        if frame is None or len(frame) == 0:
            continue
        non_ls = frame["vendor"].astype(str) != "ls"
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        value = pd.to_numeric(frame["value_krw"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        bad = non_ls & (
            ((volume == 0) & (value != 0)) | ((volume > 0) & ((value < low * volume) | (value > high * volume)))
        )
        count = int(bad.sum())
        if count:
            issues.append(_intraday_issue(session, count, "value_out_of_range"))
    return tuple(issues)


def audit_aftermarket_ticks(
    trading_date: date,
    *,
    read_ticks: Callable[[str], pd.DataFrame | None] | None = None,
    read_bars: Callable[[str], pd.DataFrame | None] | None = None,
) -> tuple[str, ...]:
    """Audit same-day aftermarket tick partitions against the stored 1m bars of the same session.

    Ticks cannot be re-fetched after the day ends, so their absence or inconsistency must surface in the
    same evening's digest. Volume is compared per symbol over the whole session window, which is
    independent of the bar labelling convention. KRX aftermarket bars are start-labelled, so the bar stamped at the session
    ceiling opens after the tick window closes; it is excluded via `BAR_VOLUME_CUTOFF_HMS`.

    Args:
        trading_date: Audited KST date (checked on STANDARD days).
        read_ticks: session -> frame with symbol, ts_hms, volume (None when absent).
        read_bars: session -> frame with symbol, volume (None when absent).

    Returns:
        Issues `intraday:<session>_ticks:<count>:<reason>` with reasons missing_partition (bars exist but no
        tick partition) and volume_mismatch (tick shortfall or surplus beyond the session policy, not exact inequality).
    """
    day_str = trading_date.isoformat()

    def _default_ticks(session: str) -> pd.DataFrame | None:
        path = tick_partition_path(day_str, session)
        if not path.exists():
            return None
        return pd.read_parquet(path, columns=["symbol", "ts_hms", "volume"])

    def _default_bars(session: str) -> pd.DataFrame | None:
        path = intraday_partition_path(1, day_str, session)
        if not path.exists():
            return None
        return pd.read_parquet(path, columns=["symbol", "ts_hms", "volume"])

    ticks_reader = read_ticks if read_ticks is not None else _default_ticks
    bars_reader = read_bars if read_bars is not None else _default_bars
    issues: list[str] = []
    for session in (INTRADAY_SESSION_KRX_AFTERMARKET, INTRADAY_SESSION_NXT_AFTERMARKET):
        bars = bars_reader(session)
        ticks = ticks_reader(session)
        if bars is None or len(bars) == 0:
            continue
        if ticks is None:
            issues.append(f"intraday:{session}_ticks:1:missing_partition")
            continue
        bar_sum = comparable_bar_volumes(session, bars)
        tick_sum = summed_tick_volumes(ticks)
        mismatched = sum(
            1
            for symbol in set(bar_sum) | set(tick_sum)
            if classify_tick_bar_volume(session, float(bar_sum.get(symbol, 0)), float(tick_sum.get(symbol, 0)))
            is not TickBarRelation.CONSISTENT
        )
        if mismatched:
            issues.append(f"intraday:{session}_ticks:{mismatched}:volume_mismatch")
    return tuple(issues)


REGULAR_TICKS_AUDIT_START_DATE: str = "2026-09-28"
"""First date with the current archive path and certified LS/Kiwoom ticks; earlier dates predate the fix."""


def certified_tick_symbols(store: CaptureStore, trading_date: date, *, session: str = "regular") -> frozenset[str]:
    """Symbols whose stored ticks for the day are certified complete by the tape's vendor total.

    A symbol is certified when it has a `TRADE_TICKS` entry for `session` with status `COMPLETE`
    and `reason` prefix (text before the first `:`) exactly `tape_complete`. The latest manifest
    wins when entries conflict for a symbol; a later non-certified `COMPLETE` entry
    (live re-collection) un-certifies it. `tape_bracketed` and `tape_empty` never certify.

    Args:
        store: Immutable artifacts and coverage manifests.
        trading_date: Audited KST date.
        session: Session label to certify (regular ticks only in practice).

    Returns:
        Certified symbols; empty when manifests are missing or the store is unreadable
        (everything stays `lost`; fail closed toward warning). Never raises for absent days.
    """
    try:
        manifests = store.read_manifests(trading_date.isoformat())
    except (OSError, ValueError):
        return frozenset()
    certified: dict[str, bool] = {}
    for manifest in manifests:
        if manifest.status == CaptureStatus.PENDING:
            continue
        for entry in manifest.entries:
            if entry.dataset is not CaptureDataset.TRADE_TICKS or entry.session != session:
                continue
            if entry.symbol is None or entry.status != CaptureStatus.COMPLETE:
                continue
            certified[str(entry.symbol)] = entry.reason.split(":")[0] == "tape_complete"
    return frozenset(symbol for symbol, ok in certified.items() if ok)


@dataclass(frozen=True)
class TickSourceDiff:
    """One tape-certified regular-session shortfall attributed to vendor aggregation."""

    symbol: str
    bar_volume: float
    tick_volume: float
    relative_shortfall: float


@dataclass(frozen=True)
class TickAuditResult:
    """Classified regular-tick audit: warning issues plus informational source diffs."""

    issues: tuple[str, ...]
    source_diffs: tuple[TickSourceDiff, ...]
    compared: int


def classify_regular_ticks(
    trading_date: date,
    *,
    certified: frozenset[str] = frozenset(),
    read_ticks: Callable[[], pd.DataFrame | None] | None = None,
    read_bars: Callable[[], pd.DataFrame | None] | None = None,
) -> TickAuditResult:
    """Classify same-day regular-session tick volume against the 1m bars per symbol.

    Tape-certified shortfalls within tolerance are vendor disagreement (informational only);
    uncertified shortfalls stay actionable warnings the sweep targets.

    Args:
        trading_date: Audited KST date (checked only from REGULAR_TICKS_AUDIT_START_DATE).
        certified: Symbols certified complete by the tape vendor total.
        read_ticks: No-arg callable returning a frame with symbol and volume (None when the
            tick partition is absent). None reads the stored regular tick partition.
        read_bars: No-arg callable returning a frame with symbol, volume and ts_hms (None when
            absent). None reads the stored regular 1m partition.

    Returns:
        TickAuditResult with warning issues (`missing_partition`, `volume_gap` for lost only,
        `certified_gap`, `source_diff_systemic`), informational `source_diffs` sorted by
        relative shortfall descending, and `compared` (symbols with positive bar volume).
    """
    day_str = trading_date.isoformat()
    if day_str < REGULAR_TICKS_AUDIT_START_DATE:
        return TickAuditResult(issues=(), source_diffs=(), compared=0)

    def _default_ticks() -> pd.DataFrame | None:
        path = tick_partition_path(day_str, INTRADAY_SESSION_REGULAR)
        if not path.exists():
            return None
        return pd.read_parquet(path, columns=["symbol", "volume"])

    def _default_bars() -> pd.DataFrame | None:
        path = intraday_partition_path(1, day_str, INTRADAY_SESSION_REGULAR)
        if not path.exists():
            return None
        return pd.read_parquet(path, columns=["symbol", "ts_hms", "volume"])

    ticks_reader = read_ticks if read_ticks is not None else _default_ticks
    bars_reader = read_bars if read_bars is not None else _default_bars
    bars = bars_reader()
    ticks = ticks_reader()
    if bars is None or len(bars) == 0:
        return TickAuditResult(issues=(), source_diffs=(), compared=0)
    bar_sum = comparable_bar_volumes(INTRADAY_SESSION_REGULAR, bars)
    compared = sum(1 for total in bar_sum.values() if float(total) > 0)
    if ticks is None:
        return TickAuditResult(
            issues=(_intraday_issue("regular_ticks", 1, "missing_partition"),),
            source_diffs=(),
            compared=compared,
        )
    tick_sum = summed_tick_volumes(ticks)
    lost = 0
    excess = 0
    diffs: list[TickSourceDiff] = []
    for symbol, bar_total in bar_sum.items():
        bar_total = float(bar_total)
        if bar_total == 0:
            continue
        tick_total = float(tick_sum.get(symbol, 0))
        if (
            classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, bar_total, tick_total)
            is not TickBarRelation.TICK_SHORT
        ):
            continue
        verdict = classify_certified_tick_shortfall(
            INTRADAY_SESSION_REGULAR, bar_total, tick_total, tape_certified=symbol in certified
        )
        if verdict is CertifiedTickShortfall.SOURCE_DIFF:
            diffs.append(
                TickSourceDiff(
                    symbol=symbol,
                    bar_volume=bar_total,
                    tick_volume=tick_total,
                    relative_shortfall=(bar_total - tick_total) / bar_total if bar_total > 0 else 0.0,
                )
            )
        elif verdict is CertifiedTickShortfall.SOURCE_DIFF_EXCESS:
            excess += 1
        else:
            lost += 1
    diffs.sort(key=lambda d: (-d.relative_shortfall, d.symbol))
    issues: list[str] = []
    if lost:
        issues.append(_intraday_issue("regular_ticks", lost, "volume_gap"))
    if excess:
        issues.append(_intraday_issue("regular_ticks", excess, "certified_gap"))
    if diffs and compared > 0 and len(diffs) / compared > CERTIFIED_SOURCE_DIFF_MAX_SHARE:
        issues.append(_intraday_issue("regular_ticks", len(diffs), "source_diff_systemic"))
    return TickAuditResult(issues=tuple(issues), source_diffs=tuple(diffs), compared=compared)


def audit_regular_ticks(
    trading_date: date,
    *,
    certified: frozenset[str] = frozenset(),
    read_ticks: Callable[[], pd.DataFrame | None] | None = None,
    read_bars: Callable[[], pd.DataFrame | None] | None = None,
) -> tuple[str, ...]:
    """Compare same-day regular-session tick volume with the 1m bars per symbol.

    The bar stamped at the regular close carries closing-auction volume that is outside the tick
    window and is excluded via `BAR_VOLUME_CUTOFF_HMS`. Ticks cover the exchange tape while bars are vendor-aggregated, so small
    residuals are expected; only a tick shortfall beyond the shared tick-bar contract marks a
    symbol as short of ticks. Tape-certified shortfalls within tolerance are informational only.

    Args:
        trading_date: Audited KST date (checked only from REGULAR_TICKS_AUDIT_START_DATE; the
            caller gates STANDARD days).
        certified: Symbols certified complete by the tape vendor total.
        read_ticks: No-arg callable returning a frame with symbol and volume (None when the
            tick partition is absent). None reads the stored regular tick partition with
            column pruning.
        read_bars: No-arg callable returning a frame with symbol, volume and ts_hms (None when
            absent). None reads the stored regular 1m partition with column pruning.

    Returns:
        Issues `intraday:regular_ticks:<count>:<reason>` with reasons missing_partition (bars
        exist but no tick partition), volume_gap (uncertified symbols short of ticks),
        certified_gap and source_diff_systemic; empty when clean. Equals
        `classify_regular_ticks(...).issues`.
    """
    return classify_regular_ticks(
        trading_date, certified=certified, read_ticks=read_ticks, read_bars=read_bars
    ).issues


TAPE_DEPTH_DAYS: int = 30
"""Observed Kiwoom tape depth in days; needs older than this can never be recovered."""

TAPE_EXPIRY_WARNING_DAYS: int = 3
"""Needs within this many days of tape expiry are surfaced as digest warnings."""

TAPE_REPORT_MAX_AGE_DAYS: int = 4
"""A sweep report older than this (covers weekends and one holiday) is ignored by the digest."""


def _read_tape_sweep_report(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads((root / "staging" / "tape_sweep" / "last_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def audit_tape_sweep(
    trading_date: date,
    *,
    profile: CollectionSettings | None = None,
    report: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Surface tape-sweep expiry and disk-guard states from the last sweep report.

    The sweep already computed the needs against the stored partitions and the settled ledger, so the audit only reads
    its report: recomputing 30 days of needs here would rescan every partition inside a unit that must stay fast.

    Args:
        trading_date: Audited KST date (checked on STANDARD days).
        profile: Acquisition limits; None builds CollectionSettings().
        report: Injected sweep report; None reads `staging/tape_sweep/last_report.json`. An absent or stale
            (older than TAPE_REPORT_MAX_AGE_DAYS) report yields no issues, since no sweep has judged the window yet.

    Returns:
        `intraday:tape_expiring:<n>:expiring_need` when n symbol-days are within TAPE_EXPIRY_WARNING_DAYS of tape
        expiry, plus `intraday:tape_sweep:1:disk_guard` when the sweep stopped on low disk.
    """
    resolved = profile if profile is not None else CollectionSettings()
    data = dict(report) if report is not None else _read_tape_sweep_report(_capture_root(resolved))
    run_date = str(data.get("run_date", ""))
    try:
        age = (trading_date - date.fromisoformat(run_date)).days
    except ValueError:
        return ()
    if age < 0 or age > TAPE_REPORT_MAX_AGE_DAYS:
        return ()
    issues: list[str] = []
    if bool(data.get("disk_guard", False)):
        issues.append(_intraday_issue("tape_sweep", 1, "disk_guard"))
    expiring = int(data.get("expiring_needs", 0) or 0)
    if expiring > 0:
        issues.append(_intraday_issue("tape_expiring", expiring, "expiring_need"))
    return tuple(issues)


def audit_extended_exhausted(*, ledger_path: Path | None = None) -> tuple[str, ...]:
    """Count EXHAUSTED extended-backfill ledger keys (informational only, never a warning).

    Args:
        ledger_path: Ledger parquet location; None uses the history-tree default.

    Returns:
        A single `intraday:extended_exhausted:<n>` line for the report body;
        empty when the ledger is absent or unreadable.
    """
    from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger

    ledger = ExtendedBackfillLedger(ledger_path) if ledger_path is not None else ExtendedBackfillLedger()
    try:
        frame = ledger._read_all()
    except (OSError, ValueError):
        return ()
    if frame.empty or "status" not in frame.columns:
        return ()
    latest = frame.drop_duplicates(subset=["snapshot_date", "session", "symbol"], keep="last")
    count = int((latest["status"].astype(str) == "EXHAUSTED").sum())
    return (f"intraday:extended_exhausted:{count}",)


def _expiry_hint_lines(items: Sequence[str]) -> list[str]:
    """KRX 달력 수평선 자체가 만료 예정일 때 표시하는 고정 갱신 안내."""
    if any(item.startswith(f"{CALENDAR_EXPIRY_NAME}:") for item in items):
        return [f"• 달력 갱신: {CALENDAR_RENEW_HINT}"]
    return []


class DigestSeverity(enum.StrEnum):
    """Dispatch class of one daily digest.

    The digest subject is presentation text (emoji, Korean labels) and must never be parsed to route alerts;
    severity is decided together with the subject and is the only input to dispatch.
    """

    OK = "ok"
    WARNING = "warning"
    HOLIDAY_SKIP = "holiday_skip"


@dataclass(frozen=True)
class AuditIssue:
    """Stable warning-level audit finding with self-healing flag."""

    key: str
    transient: bool
    text: str


def is_transient_issue_key(key: str) -> bool:
    """Classify whether re-measurement alone can clear an issue key.

    Transient conditions heal without human repair; persistent ones are facts
    about the audited date that only a later full audit can retire.
    """
    if key == "undelivered_alerts":
        return True
    return key.startswith(("offsite_backup:", "failed_unit:", "stale_kis_token:"))


@dataclass(frozen=True)
class AuditDigest:
    """Rendered daily digest and its dispatch severity."""

    subject: str
    body: str
    severity: DigestSeverity
    provisional_reasons: tuple[str, ...] = ()
    issues: tuple[AuditIssue, ...] = ()


AUDIT_ALERT_STATE_RELPATH: str = "logs/heartbeat/audit_alert_state.json"
AUDIT_ALERT_STATE_SCHEMA_VERSION: int = 1
AUDIT_HEARTBEAT_SCHEMA_VERSION: int = 2


DIGEST_ISSUE_DISPLAY_LIMIT: int = 10
"""Maximum issue strings shown in a digest summary bullet or a summary log line; the remainder is reported as a count."""


def _format_bounded_issues(issues: Sequence[str]) -> str:
    """Join at most DIGEST_ISSUE_DISPLAY_LIMIT issues, appending an omitted count."""
    shown = list(issues[:DIGEST_ISSUE_DISPLAY_LIMIT])
    omitted = len(issues) - len(shown)
    text = ", ".join(shown)
    if omitted > 0:
        text += f" 외 {omitted}건"
    return text


_IGNORED_COLLECTION_SUFFIXES: tuple[str, ...] = (":incomplete_entries", ":disabled", ":raw_disabled")


def _critical_collection_issues(collection_issues: Sequence[str]) -> list[str]:
    return [iss for iss in collection_issues if not any(iss.endswith(suffix) for suffix in _IGNORED_COLLECTION_SUFFIXES)]


def collect_audit_issues(
    *,
    day_kind: str,
    missing_steps: Sequence[str],
    failed_units: Sequence[str],
    stale_kis_tokens: Sequence[str],
    critical_collection: Sequence[str],
    intraday_issues: Sequence[str],
    backup_warnings: Sequence[str],
    undelivered_alerts: int,
    expiry_warnings: Sequence[str],
    calendar_disagreement: bool,
) -> tuple[AuditIssue, ...]:
    """Build the warning-level issue set backing a digest subject."""
    issues: list[AuditIssue] = []
    issues.extend(AuditIssue(key=f"missing:{step}", transient=False, text=f"누락 단계: {step}") for step in missing_steps)
    issues.extend(
        AuditIssue(key=f"failed_unit:{unit}", transient=True, text=f"실패 유닛: {unit}") for unit in failed_units
    )
    issues.extend(
        AuditIssue(key=f"stale_kis_token:{token}", transient=True, text=f"KIS 토큰 누락: {token}")
        for token in stale_kis_tokens
    )
    if day_kind != DAY_HOLIDAY:
        issues.extend(AuditIssue(key=raw, transient=False, text=f"수집 이상: {raw}") for raw in critical_collection)
        issues.extend(AuditIssue(key=raw, transient=False, text=f"장중 이상: {raw}") for raw in intraday_issues)
    issues.extend(AuditIssue(key=raw, transient=True, text=f"백업 이상: {raw}") for raw in backup_warnings)
    if undelivered_alerts:
        issues.append(
            AuditIssue(key="undelivered_alerts", transient=True, text=f"미전송 알림: {undelivered_alerts}건 (outbox 적체)")
        )
    issues.extend(
        AuditIssue(key=f"expiry:{warn}", transient=False, text=f"만료 임박: {warn}") for warn in expiry_warnings
    )
    if calendar_disagreement:
        issues.append(
            AuditIssue(
                key="calendar_disagreement",
                transient=False,
                text="달력 불일치: 정적 달력은 개장(STANDARD)이나 KIS 오라클이 휴일로 응답",
            )
        )
    return tuple(issues)


def build_digest(
    snapshot_date: str,
    day_kind: str,
    result: dict[str, bool] | None,
    failed_units: list[str],
    stale_kis_tokens: list[str],
    *,
    collection_issues: Sequence[str] = (),
    intraday_issues: Sequence[str] = (),
    backup_issues: Sequence[str] = (),
    session_kind: str = "UNKNOWN",
    undelivered_alerts: int = 0,
    expiry_notices: Sequence[str] = (),
    expiry_warnings: Sequence[str] = (),
    info_lines: Sequence[str] = (),
) -> AuditDigest:
    """일일 요약의 제목·본문과 발송 심각도를 만든다.

    Severity is decided by the same conditions that choose the subject, so routing never depends on subject text.

    Args:
        snapshot_date: 점검 대상일.
        day_kind: classify_day 결과(주말 제외).
        result: audit_daily_completeness 결과. 휴장일에만 None을 허용한다.
        failed_units: list_failed_kca_units 결과.
        stale_kis_tokens: list_stale_kis_tokens 결과.
        collection_issues: Independently assessed raw-data and schedule gaps.
        intraday_issues: Stored-partition audit issues `intraday:<session>:<count>:<reason>`; listed in full in the detail
            section and, bounded by DIGEST_ISSUE_DISPLAY_LIMIT, in the warning summary. Must be empty on holidays.
        backup_issues: Offsite backup staleness issues.
        session_kind: Resolved SessionKind value for the date.
        undelivered_alerts: Outbox에 적체된 미전송 알림 수. 0보다 크면 경고.
        expiry_notices: D-30 이내 만료 예정 항목. 정상 요약을 경고로 바꾸지 않는다.
        expiry_warnings: D-7 이내(지난 항목 포함) 만료 항목. backup_issues처럼 경고로 격상한다.
        info_lines: 경고로 격상하지 않는 정보성 본문 라인(예: extended-backfill 소진 수).

    Returns:
        AuditDigest with severity WARNING for any warning digest (trading or holiday), HOLIDAY_SKIP for a clean
        holiday, OK otherwise.

    Raises:
        ValueError: unsupported day_kind; result None on a non-holiday; non-empty intraday_issues on a holiday.
    """
    if day_kind not in (DAY_WEEKEND, DAY_HOLIDAY, DAY_TRADING, DAY_UNKNOWN):
        raise ValueError(f"unsupported day_kind={day_kind!r}")
    if day_kind == DAY_HOLIDAY and intraday_issues:
        raise ValueError("intraday_issues must be empty on a holiday")
    lines = [f"date={snapshot_date}", f"day={day_kind}", f"session={session_kind}"]
    lines.extend(info_lines)
    if day_kind != DAY_HOLIDAY:
        if result is None:
            raise ValueError(f"audit result required for day_kind={day_kind!r}")
        lines += [f"{step}={'OK' if result.get(step, False) else 'MISSING'}" for step in AUDIT_STEPS]
    lines.append(f"failed_units={','.join(failed_units) if failed_units else 'none'}")
    lines.append(f"stale_kis_tokens={','.join(stale_kis_tokens) if stale_kis_tokens else 'none'}")
    lines.append(f"collection_issues={','.join(collection_issues) if collection_issues else 'none'}")
    if day_kind != DAY_HOLIDAY:
        lines.append(f"intraday_issues={','.join(intraday_issues) if intraday_issues else 'none'}")
    lines.append(f"backup_issues={','.join(backup_issues) if backup_issues else 'none'}")
    lines.append(f"undelivered_alerts={undelivered_alerts}")
    lines.append(f"expiry_notices={','.join(expiry_notices) if expiry_notices else 'none'}")
    lines.append(f"expiry_warnings={','.join(expiry_warnings) if expiry_warnings else 'none'}")
    backup_infos = tuple(issue for issue in backup_issues if issue in BACKUP_INFO_ISSUES)
    backup_warnings = tuple(issue for issue in backup_issues if issue not in BACKUP_INFO_ISSUES)
    running_line = "• 백업 진행 중: 완료 후 자동 재확인됩니다"
    critical_collection = _critical_collection_issues(collection_issues)
    _missing_steps: list[str] = []
    if day_kind != DAY_HOLIDAY:
        _missing_steps = [step for step in AUDIT_STEPS if not result.get(step, False)] if result is not None else []
    _calendar_disagreement = day_kind == DAY_HOLIDAY and session_kind == SessionKind.STANDARD.value
    issues = collect_audit_issues(
        day_kind=day_kind,
        missing_steps=_missing_steps,
        failed_units=list(failed_units),
        stale_kis_tokens=[] if day_kind == DAY_HOLIDAY else list(stale_kis_tokens),
        critical_collection=[] if day_kind == DAY_HOLIDAY else list(critical_collection),
        intraday_issues=() if day_kind == DAY_HOLIDAY else tuple(intraday_issues),
        backup_warnings=list(backup_warnings),
        undelivered_alerts=int(undelivered_alerts),
        expiry_warnings=list(expiry_warnings),
        calendar_disagreement=_calendar_disagreement,
    )
    _backup_warning_set = set(backup_warnings)
    issue_missing = [issue.key.split(":", 1)[1] for issue in issues if issue.key.startswith("missing:")]
    issue_failed = [issue.key.split(":", 1)[1] for issue in issues if issue.key.startswith("failed_unit:")]
    issue_stale = [issue.key.split(":", 1)[1] for issue in issues if issue.key.startswith("stale_kis_token:")]
    issue_critical = [issue.key for issue in issues if issue.key.startswith("collection:")]
    issue_intraday = [issue.key for issue in issues if issue.key.startswith("intraday:")]
    issue_backup = [issue.key for issue in issues if issue.key in _backup_warning_set]
    issue_expiry = [issue.key[len("expiry:") :] for issue in issues if issue.key.startswith("expiry:")]
    issue_undelivered = any(issue.key == "undelivered_alerts" for issue in issues)

    if day_kind == DAY_HOLIDAY:
        holiday_problems: list[str] = []
        if _calendar_disagreement:
            holiday_problems.append("calendar_disagreement")
        if issue_failed:
            holiday_problems.append(f"실패유닛 {','.join(issue_failed)}")
        if issue_backup:
            holiday_problems.append(f"백업이상 {','.join(issue_backup)}")
        if issue_undelivered:
            holiday_problems.append(f"미전송알림 {undelivered_alerts}건")
        if issue_expiry:
            holiday_problems.append(f"만료임박 {','.join(issue_expiry)}")
        if holiday_problems:
            summary_lines = [
                "==================================================",
                f"🚨 K-Closing Alpha 장애/누락 알림 ({snapshot_date})",
                "==================================================",
            ]
            if session_kind == SessionKind.STANDARD.value:
                summary_lines.append(
                    "• 달력 불일치: 정적 달력은 개장(STANDARD)이나 KIS 오라클이 휴일로 응답 (fail-closed, 무결정)"
                )
            if issue_failed:
                summary_lines.append(f"• 실패 유닛: {', '.join(issue_failed)}")
            if issue_backup:
                summary_lines.append(f"• 백업 이상: {', '.join(issue_backup)}")
            if backup_infos:
                summary_lines.append(running_line)
            if issue_undelivered:
                summary_lines.append(f"• 미전송 알림: {undelivered_alerts}건 (outbox 적체)")
            if issue_expiry:
                summary_lines.append(f"• 만료 임박: {', '.join(issue_expiry)}")
            summary_lines.extend(_expiry_hint_lines((*expiry_notices, *expiry_warnings)))
            summary_lines.append("• 조치 안내: or-vps 서버 상태 점검 요망")
            body = "\n".join(summary_lines) + "\n\n[상세 내역]\n" + "\n".join(lines)
            return AuditDigest(
                subject=f"[kca] 🚨 {snapshot_date} 일일점검 경고: {' / '.join(holiday_problems)}",
                body=body,
                severity=DigestSeverity.WARNING,
                provisional_reasons=backup_infos,
                issues=issues,
            )
        label = "휴장일 SKIP"
        header = (
            "==================================================\n"
            f"⏸️ K-Closing Alpha 휴장일 알림 ({snapshot_date})\n"
            "==================================================\n"
            "• 상태: ⏸️ 거래소 휴장일 (배치 스킵)\n\n"
        )
        if backup_infos:
            header = header.rstrip("\n") + "\n" + running_line + "\n\n"
        return AuditDigest(
            subject=f"[kca] ⏸️ {snapshot_date} {label}",
            body=header + "[상세 내역]\n" + "\n".join(lines),
            severity=DigestSeverity.HOLIDAY_SKIP,
            provisional_reasons=backup_infos,
            issues=issues,
        )

    is_warning = bool(issues)

    if not is_warning:
        nav_str, entry_str = _extract_paper_summary(snapshot_date)
        bars_str, ticks_str = _extract_intraday_summary(snapshot_date)
        subject = f"[kca] 🟢 {snapshot_date} 일일점검 완료 (정상)"
        summary_block = [
            "==================================================",
            f"📊 K-Closing Alpha 일일 운영 요약 ({snapshot_date})",
            "==================================================",
            "• 상태: 🟢 전 단계 정상 완료 (누락 0 / 실패 0)",
            f"• 자산: 💼 NAV {nav_str}",
            f"• 진입: 🎯 {entry_str}",
            f"• 데이터: 📦 1분봉 {bars_str} / 체결 틱 {ticks_str} 적재 완료",
        ]
        if backup_infos:
            summary_block.append(running_line)
        if expiry_notices:
            summary_block.append(f"🔑 갱신 필요: {', '.join(expiry_notices)}")
            subject += f" · 🔑갱신필요 {len(expiry_notices)}건"
        summary_block.extend(_expiry_hint_lines(expiry_notices))
        body = "\n".join(summary_block) + "\n\n[상세 내역]\n" + "\n".join(lines)
        return AuditDigest(
            subject=subject, body=body, severity=DigestSeverity.OK, provisional_reasons=backup_infos, issues=issues
        )

    problems = []
    summary_lines = [
        "==================================================",
        f"🚨 K-Closing Alpha 장애/누락 알림 ({snapshot_date})",
        "==================================================",
    ]
    if issue_missing:
        problems.append(f"누락 {','.join(issue_missing)}")
        summary_lines.append(f"• 누락 단계: {', '.join(issue_missing)}")
    if issue_failed:
        problems.append(f"실패유닛 {','.join(issue_failed)}")
        summary_lines.append(f"• 실패 유닛: {', '.join(issue_failed)}")
    if issue_stale:
        problems.append(f"KIS토큰누락 {','.join(issue_stale)}")
        summary_lines.append(f"• KIS 토큰 누락: {', '.join(issue_stale)}")
    if issue_critical:
        problems.append(f"수집이상 {','.join(issue_critical)}")
        summary_lines.append(f"• 수집 이상: {', '.join(issue_critical)}")
    if issue_intraday:
        summary_lines.append(f"• 장중 이상: {_format_bounded_issues(issue_intraday)}")
    if issue_backup:
        problems.append(f"백업이상 {','.join(issue_backup)}")
        summary_lines.append(f"• 백업 이상: {', '.join(issue_backup)}")
    if backup_infos:
        summary_lines.append(running_line)
    if issue_expiry:
        problems.append(f"만료임박 {','.join(issue_expiry)}")
        summary_lines.append(f"• 만료 임박: {', '.join(issue_expiry)}")
    if issue_undelivered:
        problems.append(f"미전송알림 {undelivered_alerts}건")
        summary_lines.append(f"• 미전송 알림: {undelivered_alerts}건 (outbox 적체)")
    if expiry_notices:
        summary_lines.append(f"🔑 갱신 필요: {', '.join(expiry_notices)}")
    summary_lines.extend(_expiry_hint_lines((*expiry_notices, *expiry_warnings)))
    summary_lines.append("• 조치 안내: or-vps 서버 상태 점검 요망")
    body = "\n".join(summary_lines) + "\n\n[상세 내역]\n" + "\n".join(lines)
    return AuditDigest(
        subject=f"[kca] 🚨 {snapshot_date} 일일점검 경고: {' / '.join(problems)}",
        body=body,
        severity=DigestSeverity.WARNING,
        provisional_reasons=backup_infos,
        issues=issues,
    )


def resolve_snapshot_date(now: pd.Timestamp, *, catchup_cutoff_hour: int = _SNAPSHOT_CATCHUP_CUTOFF_HOUR) -> str:
    """실행 시각 기준으로 감사 대상 영업일(KST)을 결정한다.

    daily_audit는 평일 21:20 KST 타이머로만 트리거되며(Mon..Fri 21:20:00 Asia/Seoul),
    선행 아카이브 잡과의 After= 순서 의존성 때문에 실제 프로세스 시작이 자정을 넘길
    수 있다. 이때 wall-clock '오늘'을 그대로 쓰면 아직 파이프라인이 전혀 돌지 않은
    새 영업일을 감사하게 되어, 정작 검증해야 할 전일 점검이 영구 누락된다.
    자정~catchup_cutoff_hour 사이의 실행은 전일 파이프라인의 지연 실행으로 간주해
    전일자를 반환한다. daily-audit의 유일한 정상 트리거가 평일 저녁이므로
    (다음 트리거는 최소 다음 평일 21:20), 이 창 안에서의 실행은 항상 '어제 저녁
    사이클의 지연분'이지 '오늘 저녁 사이클의 조기 실행'일 수 없다 — 따라서
    거래일력(공휴일) 보정 없이 달력일 -1만으로 충분하다.

    Args:
        now: 기준 시각(Asia/Seoul tz-aware Timestamp).
        catchup_cutoff_hour: 이 시각(KST, 0-23) 이전 실행은 전일 지연실행으로 간주.

    Returns:
        감사 대상 날짜(YYYY-MM-DD, KST).
    """
    if int(now.hour) < int(catchup_cutoff_hour):
        return str((now.normalize() - pd.Timedelta(days=1)).date())
    return str(now.date())


def _default_backup_issues(audit_at: datetime) -> list[str]:
    return backup_staleness_issues(_capture_root() / REPORT_RELPATH, audit_at)


AUDIT_HEARTBEAT_RELPATH: str = "logs/heartbeat/daily_audit.json"


def _infer_heartbeat_severity(subject: str | None, day_kind: str) -> str:
    if subject is not None and "경고" in subject:
        return "WARNING"
    if subject is not None and ("휴장일" in subject or "SKIP" in subject or "⏸" in subject):
        return "HOLIDAY_SKIP"
    if day_kind == DAY_HOLIDAY:
        return "HOLIDAY_SKIP"
    return "OK"


def _serialize_open_issues(open_issues: Sequence[AuditIssue | Mapping[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in open_issues:
        if isinstance(item, AuditIssue):
            serialized.append({"key": item.key, "transient": item.transient, "text": item.text})
        else:
            serialized.append(
                {"key": str(item["key"]), "transient": bool(item["transient"]), "text": str(item["text"])}
            )
    return serialized


def write_audit_heartbeat(
    snapshot_date: str,
    *,
    day_kind: str,
    subject: str | None,
    undelivered_alerts: int,
    finished_at: datetime,
    path: Path | None = None,
    severity: str | None = None,
    open_issues: Sequence[AuditIssue | Mapping[str, Any]] = (),
    provisional_reasons: Sequence[str] = (),
    audit_kind: str = "scheduled",
    reconciled_at: datetime | None = None,
) -> Path:
    """Persist proof that the weekday audit ran to completion.

    The external watchdog reads this file over SSH; its absence or staleness is
    the only signal that survives a dead alert channel or a stopped host, so it
    is written on every weekday run including holidays (where no digest is sent).

    Heartbeat schema v2 is additive: legacy keys keep their meaning so the
    current dashboard adapter keeps working.

    Args:
        snapshot_date: Audited KST date (YYYY-MM-DD).
        day_kind: classify_day result for the date.
        subject: Digest subject that was built (None only for weekends).
        undelivered_alerts: Outbox backlog observed by this run.
        finished_at: Timezone-aware completion time.
        path: Override target (tests); default DATA_DIR / AUDIT_HEARTBEAT_RELPATH.
        severity: OK, WARNING or HOLIDAY_SKIP; inferred from the subject when None.
        open_issues: Machine-readable issue list backing the subject.
        provisional_reasons: Informational running markers (never warnings).
        audit_kind: scheduled for full audits, reconcile for re-measurements.
        reconciled_at: Re-measurement time; None for scheduled audits.

    Returns:
        Path written.

    Raises:
        OSError: The heartbeat could not be persisted (the audit unit must fail
            so OnFailure fires; a silently missing heartbeat would page falsely
            the next morning without a cause).
    """
    target = Path(path) if path is not None else Path(settings.DATA_DIR) / AUDIT_HEARTBEAT_RELPATH
    payload = {
        "snapshot_date": snapshot_date,
        "day_kind": day_kind,
        "subject": subject,
        "undelivered_alerts": undelivered_alerts,
        "finished_at": finished_at.isoformat(),
        "schema_version": AUDIT_HEARTBEAT_SCHEMA_VERSION,
        "severity": severity if severity is not None else _infer_heartbeat_severity(subject, day_kind),
        "open_issues": _serialize_open_issues(open_issues),
        "provisional_reasons": list(provisional_reasons),
        "audit_kind": audit_kind,
        "reconciled_at": reconciled_at.isoformat() if reconciled_at is not None else None,
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False), mode=0o644)
    return target


def read_audit_heartbeat(path: Path | None = None) -> dict[str, Any] | None:
    """Read the last heartbeat; return None only when absent, fail on unreadable evidence."""
    target = Path(path) if path is not None else Path(settings.DATA_DIR) / AUDIT_HEARTBEAT_RELPATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise OSError("Unreadable audit heartbeat") from exc
    if not isinstance(raw, dict):
        raise OSError("Invalid audit heartbeat")
    return raw


def _alert_state_path(path: Path | None = None) -> Path:
    return Path(path) if path is not None else Path(settings.DATA_DIR) / AUDIT_ALERT_STATE_RELPATH


def load_audit_alert_state(path: Path | None = None) -> dict[str, Any] | None:
    """Load alert state; return None when absent and raise OSError for invalid evidence."""
    target = _alert_state_path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise OSError("Unreadable audit alert state") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("open"), dict):
        raise OSError("Invalid audit alert state")
    if any(not isinstance(entry, dict) for entry in raw["open"].values()):
        raise OSError("Invalid audit alert state entry")
    return raw


def write_audit_alert_state(
    *,
    snapshot_date: str,
    updated_at: datetime,
    open_entries: Mapping[str, Mapping[str, Any]],
    path: Path | None = None,
    pending_resolutions: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Atomically persist the alert state for reconcile runs to re-measure."""
    target = _alert_state_path(path)
    payload = {
        "schema_version": AUDIT_ALERT_STATE_SCHEMA_VERSION,
        "snapshot_date": snapshot_date,
        "updated_at": updated_at.isoformat(),
        "open": {
            key: {
                "transient": bool(entry["transient"]),
                "text": str(entry["text"]),
                "first_seen": str(entry["first_seen"]),
                "last_notified": str(entry["last_notified"]),
            }
            for key, entry in open_entries.items()
        },
    }
    if pending_resolutions:
        payload["pending_resolutions"] = dict(pending_resolutions)
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2), mode=0o644)
    return target


def sync_audit_alert_state_from_digest(
    digest: AuditDigest,
    snapshot_date: str,
    *,
    now: datetime,
    path: Path | None = None,
) -> Path:
    """Replace the alert state with a full audit evaluation, retaining history."""
    previous = load_audit_alert_state(path) or {}
    prev_open = previous.get("open", {})
    stamp = now.isoformat()
    entries: dict[str, dict[str, Any]] = {}
    for issue in digest.issues:
        prev = prev_open.get(issue.key)
        if isinstance(prev, dict) and "first_seen" in prev and "last_notified" in prev:
            entries[issue.key] = {
                "transient": issue.transient,
                "text": issue.text,
                "first_seen": str(prev["first_seen"]),
                "last_notified": str(prev["last_notified"]),
            }
        else:
            entries[issue.key] = {
                "transient": issue.transient,
                "text": issue.text,
                "first_seen": stamp,
                "last_notified": stamp,
            }
    return write_audit_alert_state(snapshot_date=snapshot_date, updated_at=now, open_entries=entries, path=path)


def run_daily_audit(
    snapshot_date: str,
    *,
    trading_day_fn: Callable[[str], bool] | None = None,
    failed_units_fn: Callable[[], list[str]] = list_failed_kca_units,
    stale_tokens_fn: Callable[[str], list[str]] = list_stale_kis_tokens,
    dispatch_fn: Callable[[str, str], dict[str, bool]] = dispatch_digest,
    backup_issues_fn: Callable[[datetime], list[str]] = _default_backup_issues,
) -> str | None:
    """평일 1회 점검 후 요약을 발송한다. 주말이면 아무것도 보내지 않는다.

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD, KST).
        trading_day_fn: 거래일 오라클 주입(테스트용).
        failed_units_fn: 실패 유닛 조회 주입(테스트용).
        stale_tokens_fn: 호스트 발급 KIS 토큰 커버리지 조회 주입(테스트용).
        dispatch_fn: 요약 발송 주입(테스트용).
        backup_issues_fn: 오프사이트 백업 신선도 조회 주입(테스트용).

    Returns:
        발송한 요약 제목. 주말이면 None.
    """
    day_kind = classify_day(snapshot_date, trading_day_fn)
    if day_kind == DAY_WEEKEND:
        logger.info("[DATA] stage=daily_audit status=SKIP reason=weekend date=%s", snapshot_date)
        return None
    result = None if day_kind == DAY_HOLIDAY else audit_daily_completeness(snapshot_date)
    trading_date = date.fromisoformat(snapshot_date)
    audit_at = datetime.now(SEOUL)
    session_day = resolve_session_day(trading_date)
    tick_source_diffs: tuple[TickSourceDiff, ...] = ()
    try:
        profile = CollectionSettings()
        session_clock = session_day.clock if session_day.clock is not None else SessionClock.standard(trading_date)
        store = CaptureStore(resolve_capture_root(profile))
        collection_issues = audit_collection_manifests(
            trading_date, store=store, profile=profile, session_clock=session_clock, audit_at=audit_at
        )
        try:
            cohort_symbols = store.read_cohort(trading_date.isoformat(), available_by=audit_at).eligible_symbols or ()
        except FileNotFoundError:
            intraday_issues: tuple[str, ...] = (_intraday_issue("regular", 0, "missing_cohort"),)
        else:
            intraday_issues = audit_intraday_partitions(
                trading_date,
                clock=session_clock,
                session_kind=session_day.kind,
                cohort_symbols=tuple(cohort_symbols),
            )
            value_sessions = [
                INTRADAY_SESSION_REGULAR,
                INTRADAY_SESSION_NXT_PREMARKET,
                INTRADAY_SESSION_NXT_AFTERMARKET,
            ]
            if trading_date.isoformat() >= KRX_AFTERMARKET_START_DATE:
                value_sessions.append(INTRADAY_SESSION_KRX_AFTERMARKET)
            if session_day.kind is not SessionKind.STANDARD:
                value_sessions = [INTRADAY_SESSION_REGULAR]
            intraday_issues = (*intraday_issues, *audit_bar_value_consistency(trading_date, sessions=value_sessions))
            if session_day.kind is SessionKind.STANDARD:
                certified = certified_tick_symbols(store, trading_date)
                tick_audit = classify_regular_ticks(trading_date, certified=certified)
                intraday_issues = (*intraday_issues, *tick_audit.issues)
                tick_source_diffs = tick_audit.source_diffs
                intraday_issues = (*intraday_issues, *audit_aftermarket_ticks(trading_date))
                intraday_issues = (*intraday_issues, *audit_tape_sweep(trading_date, profile=profile))
    except (OSError, ValueError) as exc:
        logger.warning("[DATA] stage=daily_audit collection_audit=UNAVAILABLE reason=%s", type(exc).__name__)
        collection_issues = (_collection_issue("audit", 1, "unavailable"),)
        intraday_issues = ()
    try:
        extended_info = audit_extended_exhausted()
    except (OSError, ValueError):
        extended_info = ()
    if extended_info:
        logger.info("[DATA] stage=daily_audit %s", extended_info[0])
    info_lines: tuple[str, ...] = tuple(extended_info)
    if tick_source_diffs:
        peak = max(diff.relative_shortfall for diff in tick_source_diffs)
        shown = ",".join(diff.symbol for diff in tick_source_diffs[:10])
        info_lines = (
            *info_lines,
            f"regular_ticks_source_diff={len(tick_source_diffs)} max={peak * 100:.1f}% symbols={shown}",
        )
        try:
            record_run_outcome(
                "tick_source_diff",
                RUN_OUTCOME_OK,
                run_date=trading_date.isoformat(),
                metrics={
                    "n": len(tick_source_diffs),
                    "max_relative_shortfall": peak,
                    "symbols": [diff.symbol for diff in tick_source_diffs[:20]],
                },
            )
        except Exception as exc:
            logger.warning("[SYS] stage=daily_audit tick_source_diff_record=FAILED reason=%s", type(exc).__name__)
    if result is not None:
        result["intraday_complete"] = not intraday_issues
    if result is not None and intraday_issues:
        omitted = max(0, len(intraday_issues) - DIGEST_ISSUE_DISPLAY_LIMIT)
        logger.warning(
            "[DATA] stage=daily_audit step=intraday_complete status=FAIL date=%s issues=%d shown=%s omitted=%d",
            snapshot_date,
            len(intraday_issues),
            _format_bounded_issues(intraday_issues),
            omitted,
        )
    if session_day.kind in (SessionKind.SHIFTED, SessionKind.UNKNOWN):
        collection_issues = (*collection_issues, _collection_issue("session", 0, session_day.kind.value.lower()))
    try:
        _, undelivered_alerts = drain_alert_outbox(max_items=settings.ALERT_OUTBOX_MAX_DRAIN)
    except Exception as exc:
        logger.warning("[DATA] stage=daily_audit alert_drain=FAILED reason=%s", type(exc).__name__)
        undelivered_alerts = 0
    report = evaluate_expiries(trading_date)
    digest = build_digest(
        snapshot_date,
        day_kind,
        result,
        failed_units_fn(),
        stale_tokens_fn(snapshot_date),
        collection_issues=collection_issues,
        intraday_issues=() if day_kind == DAY_HOLIDAY else intraday_issues,
        backup_issues=backup_issues_fn(audit_at),
        session_kind=session_day.kind.value,
        undelivered_alerts=undelivered_alerts,
        expiry_notices=report.notices,
        expiry_warnings=report.warnings,
        info_lines=info_lines,
    )
    if digest.severity is DigestSeverity.WARNING:
        logger.warning("[DATA] stage=daily_audit day=%s status=WARNING subject=%s", day_kind, digest.subject)
        dispatch_fn(digest.subject, digest.body)
    elif digest.severity is DigestSeverity.HOLIDAY_SKIP:
        logger.info(
            "[DATA] stage=daily_audit day=%s status=SKIP subject=%s (holiday dispatch skipped)",
            day_kind,
            digest.subject,
        )
    else:
        logger.info("[DATA] stage=daily_audit day=%s status=OK subject=%s", day_kind, digest.subject)
        dispatch_fn(digest.subject, digest.body)
    finished_at = datetime.now(SEOUL)
    sync_audit_alert_state_from_digest(digest, snapshot_date, now=finished_at)
    severity = (
        "WARNING"
        if digest.severity is DigestSeverity.WARNING
        else "HOLIDAY_SKIP"
        if digest.severity is DigestSeverity.HOLIDAY_SKIP
        else "OK"
    )
    write_audit_heartbeat(
        snapshot_date,
        day_kind=day_kind,
        subject=digest.subject,
        undelivered_alerts=undelivered_alerts,
        finished_at=finished_at,
        severity=severity,
        open_issues=digest.issues,
        provisional_reasons=digest.provisional_reasons,
        audit_kind="scheduled",
        reconciled_at=None,
    )
    return digest.subject


def main() -> None:  # pragma: no cover - CLI entry; logic covered via run_daily_audit scenarios
    parser = argparse.ArgumentParser(description="Daily automation audit + digest (every weekday after EOD)")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today in KST)")
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    snapshot_date: str = args.date or resolve_snapshot_date(now)
    run_daily_audit(snapshot_date)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
