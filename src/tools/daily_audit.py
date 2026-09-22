"""매 평일 EOD 이후 스케줄 실행되는 자동화 점검 + 일일 요약 발송.

평일마다 정확히 한 통의 요약(정상/경고/휴장일)을 보낸다. 요약이 오지 않는 것
자체가 스케줄러 중단 신호가 되도록 누락이 없어도 침묵하지 않는다. 누락 단계가
있어도 비정상 종료하지 않는다(가시화 전용).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.api.kis.key_pool import load_kis_env, read_token_issued_date, resolve_host_issued_credentials, token_cache_path
from src.config.collection import CollectionSettings
from src.daily.archive import fetch_archive_snapshot
from src.daily.archive_intraday import resolve_previous_archive_date
from src.daily.auction_capture import _close_rounds, _program_rounds
from src.data.capture_contracts import (
    SEOUL,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
    SessionClock,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_store import intraday_partition_path
from src.data.trading_calendar import is_kis_trading_day
from src.execution.paper_broker import PaperLedger
from src.processing.schema import CLOSE_CONFIRMED_COL
from src.tools.alerts import dispatch_digest
from src.tools.run_outcome import RUN_OUTCOME_OK, load_run_outcomes

logger = logging.getLogger(__name__)

DAY_WEEKEND: str = "weekend"
DAY_HOLIDAY: str = "holiday"
DAY_TRADING: str = "trading"
# 달력 조회 장애: 휴장일로 단정하지 않고 감사를 수행한다(장애 조기 발견 우선)
DAY_UNKNOWN: str = "unknown"
_CHART_DATASETS: tuple[CaptureDataset, CaptureDataset] = (CaptureDataset.MINUTE_BARS, CaptureDataset.TRADE_TICKS)
_TERMINAL_REASONS: frozenset[str] = frozenset({"exhausted", "crossed_target_date"})
_SLOW_DATA_DUE_HHMMSS: str = "213500"
_SNAPSHOT_CATCHUP_CUTOFF_HOUR: int = 12
AUDIT_STEPS: tuple[str, ...] = (
    "archive",
    "close_confirmed",
    "decision",
    "paper_entry",
    "paper_exit",
    "minute_bars",
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
    topk_dates = _column_dates(Path(settings.PARQUET_DIR) / "topk_decisions.parquet", "decision_date")
    no_decision_dates = _column_dates(Path(settings.PAPER_DIR) / "decisions.parquet", "decision_date")
    entry_dates = _entry_fill_dates(Path(settings.PAPER_DIR) / "fills.parquet")
    outcomes = load_run_outcomes(snapshot_date)
    try:
        open_positions = PaperLedger(root=Path(settings.PAPER_DIR)).load_open_positions()
        # 감사(20:15)는 당일 09:00 청산 이후이므로 당일 이전 결정 로트가 남아 있으면 청산 누락이다
        stale_exit = bool((open_positions["decision_date"].astype(str) < snapshot_date).any())
    except (KeyError, ValueError) as exc:
        # 스키마가 깨졌거나 로트 링크를 위반한 원장은 청산 상태를 보증할 수 없으므로 누락으로 보고한다
        logger.warning("[DATA] stage=daily_audit step=paper_exit status=LEDGER_INVALID reason=%s: %s", type(exc).__name__, exc)
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


def _kis_trading_day(snapshot_date: str) -> bool:  # pragma: no cover - live KIS boundary
    async def _run() -> bool:
        client = KisApiClient(**kis_data_client_kwargs())  # type: ignore[no-untyped-call]
        async with client.create_session() as session:
            await client.ensure_token(session)
            return await is_kis_trading_day(client, session, snapshot_date)

    return asyncio.run(_run())


def classify_day(snapshot_date: str, trading_day_fn: Callable[[str], bool] | None = None) -> str:
    """점검 대상일을 주말/휴장일/거래일/미상 중 하나로 분류한다.

    KIS 지수 일별시세는 당일 게시되므로 휴장일 판정에 쓴다(KRX 공식 지수는 1일 이상
    지연 게시되어 당일 휴장일 판정에 쓸 수 없다). 조회 장애는 미상으로 낮춰 감사를
    막지 않는다.

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD).
        trading_day_fn: 거래일 오라클. None이면 KIS 실조회를 사용한다.

    Returns:
        DAY_WEEKEND, DAY_HOLIDAY, DAY_TRADING, DAY_UNKNOWN 중 하나.
    """
    if pd.Timestamp(snapshot_date).weekday() >= 5:
        return DAY_WEEKEND
    oracle = trading_day_fn if trading_day_fn is not None else _kis_trading_day
    try:
        return DAY_TRADING if oracle(snapshot_date) else DAY_HOLIDAY
    except (RuntimeError, OSError, aiohttp.ClientError) as exc:
        logger.warning("[DATA] stage=daily_audit calendar_lookup=FAIL reason=%s", type(exc).__name__)
        return DAY_UNKNOWN


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
) -> list[str]:
    """이 호스트가 발급 책임을 지는 KIS 키 중 당일(snapshot_date, KST) 토큰이 없는 것을 반환한다.

    발급 자체는 kca-kis-token-warmup 책임이다. 여기서는 그 결과가 선언된 모든
    키(배정 슬롯 + 비풀 선언 키)에 실제로 도달했는지만 가시화한다(부분 성공이
    exit 0으로 숨는 것을 방지).

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD, KST).
        env: KIS 자격증명 env(테스트 주입용). None이면 호스트 .env를 읽는다.
        cache_dir: 토큰 캐시 디렉터리(테스트 주입용). None이면 settings.KIS_TOKEN_CACHE_DIR.

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
    return sorted(
        cred.slot
        for cred in creds
        if read_token_issued_date(token_cache_path(cred.app_key, cache)) != snapshot_date
    )


def _collection_issue(dataset: str, count: int, reason: str) -> str:
    return f"collection:{dataset}:{count}:{reason}"


def _terminal_proof(reason: str) -> bool:
    return reason.split(":")[0] in _TERMINAL_REASONS


def _outside_regular_session(entry: CoverageEntry, session_clock: SessionClock) -> bool:
    if entry.first_event_time is not None and entry.first_event_time < session_clock.open_at:
        return True
    return entry.last_event_time is not None and entry.last_event_time > session_clock.close_at


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
            if by_symbol.get(symbol) and not any(e.status == CaptureStatus.COMPLETE for e in by_symbol[symbol])
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
        rounds = _close_rounds(session_clock, int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS))
        program_rounds = _program_rounds(session_clock)
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
            for slot in program_rounds:
                if (symbol, CaptureDataset.PROGRAM.value, slot) not in actual:
                    missing += 1
        if missing:
            issues.append(_collection_issue("auction_close", missing, "missing_entries"))
        if incomplete:
            issues.append(_collection_issue("auction_close", incomplete, "incomplete_entries"))
    open_due_at = session_clock.open_at + timedelta(seconds=int(profile.COLLECTION_OPEN_CONFIRM_SECONDS))
    if audit_at >= open_due_at:
        terminal = [
            m
            for m in manifests
            if m.context.capture_reason == "auction-open" and m.status != CaptureStatus.PENDING
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
    if not any(m.status == CaptureStatus.COMPLETE for m in terminal):
        return (_collection_issue("slow_data", len(terminal), "incomplete_run"),)
    return ()


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
    if not profile.COLLECTION_RAW_ENABLED:
        return (_collection_issue("provenance", 0, "raw_disabled"),)
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
                store.read_decision(
                    trading_date.isoformat(), available_by=audit_at, run_id=manifest.context.run_id
                )
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
    if altdata_enabled:
        issues.extend(_audit_slow_data(manifests, trading_date, audit_at))
    else:
        issues.append(_collection_issue("slow_data", 0, "disabled"))
    return tuple(issues)


def _collection_capture_root(profile: CollectionSettings) -> Path:
    if profile.COLLECTION_ROOT is not None:
        return Path(profile.COLLECTION_ROOT)
    return Path(settings.HISTORY_DIR) / "capture"


def build_digest(
    snapshot_date: str,
    day_kind: str,
    result: dict[str, bool] | None,
    failed_units: list[str],
    stale_kis_tokens: list[str],
    *,
    collection_issues: Sequence[str] = (),
) -> tuple[str, str]:
    """일일 요약의 (제목, 본문)을 만든다.

    Args:
        snapshot_date: 점검 대상일.
        day_kind: classify_day 결과(주말 제외).
        result: audit_daily_completeness 결과. 휴장일에만 None을 허용한다.
        failed_units: list_failed_kca_units 결과.
        stale_kis_tokens: list_stale_kis_tokens 결과.
        collection_issues: Independently assessed raw-data and schedule gaps.

    Returns:
        (제목, 본문) 튜플.

    Raises:
        ValueError: 휴장일이 아닌데 result가 None인 경우.
        ValueError: Existing unsupported date/day-kind combinations.
    """
    if day_kind not in (DAY_WEEKEND, DAY_HOLIDAY, DAY_TRADING, DAY_UNKNOWN):
        raise ValueError(f"unsupported day_kind={day_kind!r}")
    lines = [f"date={snapshot_date}", f"day={day_kind}"]
    missing: list[str] = []
    if day_kind != DAY_HOLIDAY:
        if result is None:
            raise ValueError(f"audit result required for day_kind={day_kind!r}")
        missing = [step for step in AUDIT_STEPS if not result.get(step, False)]
        lines += [f"{step}={'OK' if result.get(step, False) else 'MISSING'}" for step in AUDIT_STEPS]
    lines.append(f"failed_units={','.join(failed_units) if failed_units else 'none'}")
    lines.append(f"stale_kis_tokens={','.join(stale_kis_tokens) if stale_kis_tokens else 'none'}")
    lines.append(f"collection_issues={','.join(collection_issues) if collection_issues else 'none'}")
    if not missing and not failed_units and not stale_kis_tokens and not collection_issues:
        label = "휴장일 SKIP" if day_kind == DAY_HOLIDAY else "일일점검 OK"
        return f"[KCA] {snapshot_date} {label}", "\n".join(lines)
    problems = []
    summary_lines = ["[🚨 일일점검 경고 요약]"]
    if missing:
        problems.append(f"누락 {','.join(missing)}")
        summary_lines.append(f"• 누락 단계: {', '.join(missing)}")
    if failed_units:
        problems.append(f"실패유닛 {','.join(failed_units)}")
        summary_lines.append(f"• 실패 유닛: {', '.join(failed_units)}")
    if stale_kis_tokens:
        problems.append(f"KIS토큰누락 {','.join(stale_kis_tokens)}")
        summary_lines.append(f"• KIS 토큰 누락: {', '.join(stale_kis_tokens)}")
    if collection_issues:
        problems.append(f"수집이상 {','.join(collection_issues)}")
        summary_lines.append(f"• 수집 이상: {', '.join(collection_issues)}")
    body = "\n".join(summary_lines) + "\n\n[상세 내역]\n" + "\n".join(lines)
    return f"[KCA] {snapshot_date} 일일점검 경고: {' / '.join(problems)}", body



def resolve_snapshot_date(now: pd.Timestamp, *, catchup_cutoff_hour: int = _SNAPSHOT_CATCHUP_CUTOFF_HOUR) -> str:
    """실행 시각 기준으로 감사 대상 영업일(KST)을 결정한다.

    daily_audit는 평일 20:15 KST 타이머로만 트리거되며(Mon..Fri 20:15:00 Asia/Seoul),
    선행 아카이브 잡과의 After= 순서 의존성 때문에 실제 프로세스 시작이 자정을 넘길
    수 있다. 이때 wall-clock '오늘'을 그대로 쓰면 아직 파이프라인이 전혀 돌지 않은
    새 영업일을 감사하게 되어, 정작 검증해야 할 전일 점검이 영구 누락된다.
    자정~catchup_cutoff_hour 사이의 실행은 전일 파이프라인의 지연 실행으로 간주해
    전일자를 반환한다. daily-audit의 유일한 정상 트리거가 평일 저녁이므로
    (다음 트리거는 최소 다음 평일 20:15), 이 창 안에서의 실행은 항상 '어제 저녁
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


def run_daily_audit(
    snapshot_date: str,
    *,
    trading_day_fn: Callable[[str], bool] | None = None,
    failed_units_fn: Callable[[], list[str]] = list_failed_kca_units,
    stale_tokens_fn: Callable[[str], list[str]] = list_stale_kis_tokens,
    dispatch_fn: Callable[[str, str], dict[str, bool]] = dispatch_digest,
) -> str | None:
    """평일 1회 점검 후 요약을 발송한다. 주말이면 아무것도 보내지 않는다.

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD, KST).
        trading_day_fn: 거래일 오라클 주입(테스트용).
        failed_units_fn: 실패 유닛 조회 주입(테스트용).
        stale_tokens_fn: 호스트 발급 KIS 토큰 커버리지 조회 주입(테스트용).
        dispatch_fn: 요약 발송 주입(테스트용).

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
    try:
        profile = CollectionSettings()
        session_clock = profile.COLLECTION_SESSION_OVERRIDES.get(
            snapshot_date, SessionClock.standard(trading_date)
        )
        store = CaptureStore(_collection_capture_root(profile))
        collection_issues = audit_collection_manifests(
            trading_date, store=store, profile=profile, session_clock=session_clock, audit_at=audit_at
        )
    except (OSError, ValueError) as exc:
        logger.warning("[DATA] stage=daily_audit collection_audit=UNAVAILABLE reason=%s", type(exc).__name__)
        collection_issues = (_collection_issue("audit", 1, "unavailable"),)
    subject, body = build_digest(
        snapshot_date, day_kind, result, failed_units_fn(), stale_tokens_fn(snapshot_date),
        collection_issues=collection_issues,
    )
    has_warning = "경고:" in subject
    if has_warning:
        logger.warning("[DATA] stage=daily_audit day=%s status=WARNING subject=%s", day_kind, subject)
        dispatch_fn(subject, body)
    else:
        logger.info("[DATA] stage=daily_audit day=%s status=OK subject=%s (dispatch skipped)", day_kind, subject)
    return subject



def main() -> None:  # pragma: no cover - CLI entry; logic covered via run_daily_audit scenarios
    parser = argparse.ArgumentParser(description="Daily automation audit + digest (every weekday after EOD)")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today in KST)")
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    snapshot_date: str = args.date or resolve_snapshot_date(now)
    run_daily_audit(snapshot_date)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
