"""Operator paging policy: which incidents page a human (CRITICAL) and which stay SILENT.

Mail is reserved for incidents that need a human now: money at risk (orders, positions) or
the day's decision lost (collect/predict/finalize inputs). Everything else is SILENT and is
recorded in logs, the audit heartbeat and digest files that the ops check reads. The policy is
pure (no I/O) so every entry path (systemd OnFailure, daily audit, reconcile, run outcomes)
classifies identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from zoneinfo import ZoneInfo

from src.config.market_session import DECISION_WINDOW_START_HHMMSS

ALERT_POLICY_TZ: ZoneInfo = ZoneInfo("Asia/Seoul")
PAGE_SUBJECT_PREFIX: str = "[긴급][kca] "
RESOLVED_SUBJECT_PREFIX: str = "[해결됨][kca] "
_UNIT_SUFFIX: str = ".service"
_KSE_ALERT_PREFIX: str = "kse-alert@"


class AlertClass(StrEnum):
    """Routing class of one alert candidate."""

    CRITICAL = "CRITICAL"
    SILENT = "SILENT"


class IncidentKind(StrEnum):
    """One pageable incident; the per-KST-day dedup identity shared by every entry path."""

    ORDER_ENTRY = "order_entry"
    ORDER_EXIT = "order_exit"
    ORDER_RECONCILE_ENTRY = "order_reconcile_entry"
    ORDER_RECONCILE_EXIT = "order_reconcile_exit"
    COLLECT = "collect"
    PREDICT = "predict"
    FINALIZE_CLOSE = "finalize_close"
    PRICE_PANEL_STALE = "price_panel_stale"
    KIS_TOKEN_WARMUP = "kis_token_warmup"  # noqa: S105 - incident label, not a credential
    KIS_TOKEN_STALE = "kis_token_stale"  # noqa: S105 - incident label, not a credential
    BACKUP_FAILING = "backup_failing"
    KRX_DEFERRED_RECREATE = "krx_deferred_recreate"
    KSE_PAGE_UNDELIVERED = "kse_page_undelivered"
    VPS_DISK_WATCH = "vps_disk_watch"


class SourceScope(StrEnum):
    """Lifetime of an incident source in the pager ledger.

    OPEN sources are measured conditions (failed kca units, backup state, token cache) that the
    reconcile tick re-measures and resolves. DAY sources are one-shot events (run outcomes,
    foreign units) that nothing re-measures; they expire silently at the KST day boundary.
    """

    OPEN = "open"
    DAY = "day"


@dataclass(frozen=True)
class IncidentText:
    """Korean operator text of one incident kind; `auto_action` is the default remediation line."""

    summary: str
    what: str
    auto_action: str
    action: str


@dataclass(frozen=True)
class OutcomeRule:
    """Run outcomes of one job that page; reasons in `silent_reasons` stay SILENT."""

    kind: IncidentKind
    outcomes: frozenset[str]
    silent_reasons: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AlertPolicyConfig:
    """Timing contract of the paging policy.

    Attributes:
        backup_page_after: An offsite backup failure pages only after it stays open this long, i.e.
            it also failed the next nightly slot (two consecutive days).
        kis_token_page_lead: A stale KIS data token pages only within this lead before the decision window.
        decision_window_start: KST wall-clock start of the closing decision window.
        reminder_earliest: A still-open incident is re-paged at most once per KST day, never before this time.
        max_remediation_hold: Upper bound on deferring a page while auto-remediation is still pending.
    """

    backup_page_after: timedelta = timedelta(hours=24)
    kis_token_page_lead: timedelta = timedelta(hours=2)
    decision_window_start: time = field(
        default_factory=lambda: time(
            int(DECISION_WINDOW_START_HHMMSS[0:2]), int(DECISION_WINDOW_START_HHMMSS[2:4]), int(DECISION_WINDOW_START_HHMMSS[4:6])
        )
    )
    reminder_earliest: time = time(8, 0)
    max_remediation_hold: timedelta = timedelta(hours=2)


DEFAULT_ALERT_POLICY: AlertPolicyConfig = AlertPolicyConfig()

INCIDENT_TEXTS: Mapping[IncidentKind, IncidentText] = MappingProxyType(
    {
        IncidentKind.ORDER_ENTRY: IncidentText(
            summary="종가 매수 주문 실패",
            what="오늘 종가 단일가 매수 주문이 정상 제출되지 않았습니다.",
            auto_action="없음(주문 단계는 자동 재실행하지 않음).",
            action="15:30 종가 마감 전에 증권사 앱에서 미체결·잔고를 확인하고 필요하면 수동 주문하세요.",
        ),
        IncidentKind.ORDER_EXIT: IncidentText(
            summary="시가 매도 주문 실패",
            what="오늘 시가 단일가 매도(청산) 주문이 정상 처리되지 않았습니다.",
            auto_action="없음(주문 단계는 자동 재실행하지 않음).",
            action="09:00 시가 전, 늦어도 오전 중에 증권사 앱에서 보유 잔고를 확인하고 수동 매도 여부를 결정하세요.",
        ),
        IncidentKind.ORDER_RECONCILE_ENTRY: IncidentText(
            summary="매수 체결 대사 불일치",
            what="종가 매수 체결과 장부가 일치하지 않습니다(미확인 주문·잔고 부족 등).",
            auto_action="다음 영업일 08:40 매도와 대사 실행이 매도 전에 미확정 매수를 증권사 체결 내역으로 다시 대사합니다.",
            action="오늘 18:00 전에 증권사 앱의 체결 내역과 잔고를 확인하세요.",
        ),
        IncidentKind.ORDER_RECONCILE_EXIT: IncidentText(
            summary="매도 체결 대사 불일치",
            what="시가 매도 체결과 장부가 일치하지 않습니다(미체결 잔량·미확인 주문 등).",
            auto_action="없음(대사 단계는 자동 재실행하지 않음).",
            action="오늘 정규장 중(15:20 전)에 증권사 앱에서 잔고를 확인하고 남은 포지션을 정리하세요.",
        ),
        IncidentKind.COLLECT: IncidentText(
            summary="결정 데이터 수집 실패",
            what="오늘 종가 결정용 시세 수집이 실패해 결정을 만들 수 없습니다.",
            auto_action="없음(결정 시점 수집은 재실행 불가).",
            action="15:28 전에 수집 실패 원인을 확인하세요. 시간이 지나면 오늘은 매매 없이 넘어가며 내일 전까지 원인을 수정하세요.",
        ),
        IncidentKind.PREDICT: IncidentText(
            summary="오늘 매수 결정 생성 실패",
            what="오늘 종가 매수 후보(결정)가 생성되지 않았습니다.",
            auto_action="없음(결정 단계는 자동 재실행하지 않음).",
            action="15:28 전에 원인을 확인하세요. 시간이 지나면 오늘은 매매 없이 넘어가며 내일 15:20 전까지 원인을 수정하세요.",
        ),
        IncidentKind.FINALIZE_CLOSE: IncidentText(
            summary="종가 확정 실패",
            what="오늘 결정 종목의 종가 확정이 완료되지 않았습니다.",
            auto_action="없음(종가 확정은 자동 재실행하지 않음).",
            action="오늘 18:00 전에 종가 확정 실패 원인을 확인하고 내일 08:40 청산 전에 장부를 바로잡으세요.",
        ),
        IncidentKind.PRICE_PANEL_STALE: IncidentText(
            summary="가격 이력 미갱신(오늘 결정 입력 누락)",
            what="11:30 안전 재실행 이후에도 전일 가격 이력이 반영되지 않았습니다.",
            auto_action="08:30 실행과 11:30 안전 재실행(KRX 재시도 포함) 모두 갱신에 실패했습니다.",
            action="15:20 전에 price-ingest를 수동 재실행하세요. 실패가 계속되면 오늘 결정은 생성되지 않습니다.",
        ),
        IncidentKind.KIS_TOKEN_WARMUP: IncidentText(
            summary="KIS 토큰 발급 실패",
            what="오늘 KIS 데이터 토큰 발급이 재시도 후에도 실패했습니다.",
            auto_action="systemd 재시도 후 자동 재실행을 시도했지만 실패했습니다.",
            action="15:20 결정창 전에 KIS 앱키 상태를 확인하고 kis-token-warmup을 수동 실행하세요.",
        ),
        IncidentKind.KIS_TOKEN_STALE: IncidentText(
            summary="결정창 직전 KIS 토큰 누락",
            what="결정창 2시간 이내인데 KIS 데이터 토큰이 오늘 날짜로 발급되어 있지 않습니다.",
            auto_action="토큰 재발급을 자동 재실행했지만 아직 누락 상태입니다.",
            action="15:20 전에 kis-token-warmup을 수동 실행하고 토큰 캐시를 확인하세요.",
        ),
        IncidentKind.BACKUP_FAILING: IncidentText(
            summary="오프사이트 백업 이틀 연속 실패",
            what="오프사이트 백업이 2일 연속 정상 완료되지 않았습니다.",
            auto_action="백업 자동 재실행을 시도했지만 해소되지 않았습니다.",
            action="오늘 22:15 다음 백업 전에 Drive 인증과 디스크 상태를 확인하세요.",
        ),
        IncidentKind.KRX_DEFERRED_RECREATE: IncidentText(
            summary="krx 수집기 재생성 실패",
            what="krx-alpha 수집기 지연 재생성(krx-deferred-recreate)이 실패했습니다.",
            auto_action="없음(외부 프로젝트 유닛).",
            action="다음 장 시작(08:00) 전에 krx-alpha 수집기 컨테이너 상태를 확인하고 재생성하세요.",
        ),
        IncidentKind.VPS_DISK_WATCH: IncidentText(
            summary="VPS 디스크 여유 경고",
            what="호스트 디스크 감시(vps-disk-watch)가 여유 공간 부족 또는 소진 예상 시점 임박을 알렸습니다(감시 자체 실패 포함).",
            auto_action="없음(정리는 각 프로젝트 보존 정책에 따름).",
            action="오늘 안에 vps-disk-watch 로그의 증가 상위 항목을 확인하고, 멈춘 정리나 업로드 작업이 있으면 복구하세요.",
        ),
        IncidentKind.KSE_PAGE_UNDELIVERED: IncidentText(
            summary="kse 긴급 알림 전달 실패",
            what="k-stock-engine의 긴급 알림을 kse가 직접 보내지 못해 kca가 대신 알립니다.",
            auto_action="kse 알림은 kse outbox에 적재되어 재전송 대기 중입니다.",
            action="지금 kse 로그와 kse outbox를 확인해 어떤 긴급 상황인지 파악하고 해당 작업의 마감 전에 조치하세요.",
        ),
    }
)

CRITICAL_UNITS: Mapping[str, IncidentKind] = MappingProxyType(
    {
        "kca-order-entry": IncidentKind.ORDER_ENTRY,
        "kca-order-exit": IncidentKind.ORDER_EXIT,
        "kca-order-reconcile-entry": IncidentKind.ORDER_RECONCILE_ENTRY,
        "kca-order-reconcile-exit": IncidentKind.ORDER_RECONCILE_EXIT,
        "kca-collect": IncidentKind.COLLECT,
        "kca-predict": IncidentKind.PREDICT,
        "kca-finalize-close": IncidentKind.FINALIZE_CLOSE,
        "kca-kis-token-warmup": IncidentKind.KIS_TOKEN_WARMUP,
        "krx-deferred-recreate": IncidentKind.KRX_DEFERRED_RECREATE,
        # krx deploy/host/vps_disk_watch.py signals WARN/CRIT only through its exit code (OnFailure relay).
        "vps-disk-watch": IncidentKind.VPS_DISK_WATCH,
    }
)


_ALERTED_OUTCOMES: frozenset[str] = frozenset({"DEGRADED", "NO_DECISION"})

CRITICAL_RUN_OUTCOMES: Mapping[str, OutcomeRule] = MappingProxyType(
    {
        "order_entry": OutcomeRule(IncidentKind.ORDER_ENTRY, _ALERTED_OUTCOMES),
        # A SHIFTED day exists only as a verified calendar entry or an operator override, so holding lots to the
        # next standard session is the planned outcome; the unverified-calendar hold still pages.
        "order_exit": OutcomeRule(IncidentKind.ORDER_EXIT, _ALERTED_OUTCOMES, frozenset({"shifted_session_hold"})),
        # An unfilled closing-auction limit buy leaves cash idle, not money at risk.
        "order_reconcile_entry": OutcomeRule(
            IncidentKind.ORDER_RECONCILE_ENTRY, _ALERTED_OUTCOMES, frozenset({"partial_or_unfilled"})
        ),
        "order_reconcile_exit": OutcomeRule(IncidentKind.ORDER_RECONCILE_EXIT, _ALERTED_OUTCOMES),
        "collect": OutcomeRule(IncidentKind.COLLECT, frozenset({"NO_DECISION"})),
        "predict": OutcomeRule(IncidentKind.PREDICT, frozenset({"NO_DECISION"}), frozenset({"session_shifted"})),
        "finalize_close": OutcomeRule(
            IncidentKind.FINALIZE_CLOSE,
            frozenset({"DEGRADED"}),
            frozenset({"calendar_unavailable"}),
        ),
    }
)

# price_ingest records `stale_panel` only from PRICE_INGEST_STALE_ALERT_FROM_HHMMSS on, i.e. after the
# 11:30 safety run; earlier stale runs log `stale_panel_pending_retry` and never reach this table.
PRICE_INGEST_JOB: str = "price_ingest"
PRICE_INGEST_STALE_REASON: str = "stale_panel"

# Offsite backup verdicts that mean "no good copy"; deferred/draining/running are progress states.
BACKUP_FAILING_KEYS: frozenset[str] = frozenset(
    {
        "offsite_backup:failed",
        "offsite_backup:missing",
        "offsite_backup:stale",
        "offsite_backup:unreadable",
        "offsite_backup:interrupted",
    }
)

FAILED_UNIT_PREFIX: str = "failed_unit:"
STALE_KIS_TOKEN_PREFIX: str = "stale_kis_token:"  # noqa: S105 - issue-key prefix, not a credential
RUN_SOURCE_PREFIX: str = "run:"

# Issue-key prefixes the reconcile tick re-measures every run; only these sources may be resolved by absence.
RECONCILE_MEASURED_PREFIXES: tuple[str, ...] = (f"{FAILED_UNIT_PREFIX}kca-", "offsite_backup:", STALE_KIS_TOKEN_PREFIX)


# Kinds backed by a condition the reconcile tick re-measures (and so can page after remediation and
# resolve by absence). Every other kind is a time-bound event owned by the path that observed it: a
# failed order/decision unit stays "failed" in systemd until its next daily run, so re-paging it from
# re-measurement would report yesterday's window as today's incident.
MEASURED_KINDS: frozenset[IncidentKind] = frozenset(
    {IncidentKind.KIS_TOKEN_WARMUP, IncidentKind.KIS_TOKEN_STALE, IncidentKind.BACKUP_FAILING}
)


def incident_scope(kind: IncidentKind) -> SourceScope:
    """OPEN for MEASURED_KINDS (resolved by re-measurement), DAY for one-shot events."""
    return SourceScope.OPEN if kind in MEASURED_KINDS else SourceScope.DAY


def unit_base_name(unit: str) -> str:
    """Strip every trailing `.service` (systemd `%n` relays can stack them) and surrounding blanks."""
    name = unit.strip()
    while name.endswith(_UNIT_SUFFIX):
        name = name[: -len(_UNIT_SUFFIX)]
    return name


def classify_unit(unit: str) -> IncidentKind | None:
    """Return the incident kind a failed systemd unit pages as, or None when it is SILENT.

    Every `kse-alert@<target>` relay is CRITICAL: the k-stock-engine alert unit exits non-zero only
    when a CRITICAL kse page could not be delivered and was queued to the kse outbox.
    """
    base = unit_base_name(unit)
    if base.startswith(_KSE_ALERT_PREFIX):
        return IncidentKind.KSE_PAGE_UNDELIVERED
    return CRITICAL_UNITS.get(base)


def classify_run_outcome(job: str, outcome: str, reason: str) -> IncidentKind | None:
    """Return the incident kind a recorded run outcome pages as, or None when it is SILENT."""
    if job == PRICE_INGEST_JOB:
        return IncidentKind.PRICE_PANEL_STALE if outcome == "DEGRADED" and reason == PRICE_INGEST_STALE_REASON else None
    rule = CRITICAL_RUN_OUTCOMES.get(job)
    if rule is None or outcome not in rule.outcomes or reason in rule.silent_reasons:
        return None
    return rule.kind


def classify_issue_key(key: str) -> IncidentKind | None:
    """Return the incident kind an audit/reconcile issue key belongs to, or None when it is SILENT.

    Timing gates (backup persistence, token lead) are applied separately by `issue_page_due`.
    """
    if key.startswith(FAILED_UNIT_PREFIX):
        return classify_unit(key[len(FAILED_UNIT_PREFIX) :])
    if key.startswith(STALE_KIS_TOKEN_PREFIX):
        return IncidentKind.KIS_TOKEN_STALE
    if key in BACKUP_FAILING_KEYS:
        return IncidentKind.BACKUP_FAILING
    return None


def issue_page_due(
    kind: IncidentKind, *, first_seen: datetime, now: datetime, cfg: AlertPolicyConfig = DEFAULT_ALERT_POLICY
) -> bool:
    """True when a classified issue's timing gate allows a page at `now` (both timezone-aware)."""
    if kind is IncidentKind.BACKUP_FAILING:
        return now - first_seen >= cfg.backup_page_after
    if kind is IncidentKind.KIS_TOKEN_STALE:
        local = now.astimezone(ALERT_POLICY_TZ)
        window_start = datetime.combine(local.date(), cfg.decision_window_start, tzinfo=ALERT_POLICY_TZ)
        return window_start - cfg.kis_token_page_lead <= local < window_start
    return True


def alert_class_of_unit(unit: str) -> AlertClass:
    """CRITICAL when `classify_unit` maps the unit to an incident kind, else SILENT."""
    return AlertClass.CRITICAL if classify_unit(unit) is not None else AlertClass.SILENT


def format_page(kind: IncidentKind, *, auto_action: str | None = None) -> tuple[str, str]:
    """Build the operator page: `[긴급][kca] <summary>` and exactly three Korean lines."""
    text = INCIDENT_TEXTS[kind]
    body = "\n".join(
        (
            f"무슨 일: {text.what}",
            f"자동 조치: {auto_action or text.auto_action}",
            f"지금 할 일: {text.action}",
        )
    )
    return f"{PAGE_SUBJECT_PREFIX}{text.summary}", body


def format_resolved(kind: IncidentKind, *, at: datetime) -> tuple[str, str]:
    """Build the single one-line resolution mail for a previously paged incident."""
    text = INCIDENT_TEXTS[kind]
    stamp = at.astimezone(ALERT_POLICY_TZ).strftime("%H:%M")
    return f"{RESOLVED_SUBJECT_PREFIX}{text.summary}", f"해결됨: {text.summary} ({stamp} KST 정상 확인, 조치 불필요)"
