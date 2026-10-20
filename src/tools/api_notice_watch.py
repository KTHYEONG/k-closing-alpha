"""Broker API notice watcher: surfaces new notices that touch the TRs/paths any host project calls.

Impact is judged against the `api_surface` manifest (kca, krx, kse). Actionable items persist as pending actions
in the state file until acknowledged; quant-dashboard renders them per project and an AI/ops check can list them
(`--pending`), fix the affected project, then `--ack`. Nothing is mailed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import requests

from src import settings
from src.tools.api_surface import BROKER_DBSEC, BROKER_KIS, BROKER_KIWOOM, BROKER_LS, BROKER_TOSS, Surface, load_surface
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
STATE_RELPATH: str = "logs/heartbeat/api_notice_watch_state.json"
STATE_SCHEMA_VERSION: int = 1
PENDING_HINT: str = (
    "처리: cd ~/k-closing-alpha && uv run python -m src.tools.api_notice_watch --pending "
    "→ 영향 프로젝트 점검·수정 후 --ack <id>"
)
FALLBACK_PROJECT = "kca"
_TAG_RE = re.compile(r"<[^>]+>")
_STYLE_BLOCK_RE = re.compile(r"<(script|style|svg)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_LS_ROW_RE = re.compile(
    r'<tr[^>]*hx-get="[^"]*/posts/(?P<id>[0-9a-f-]{36})[^"]*"[^>]*>(?P<row>.*?)</tr>', re.DOTALL
)
_LS_TITLE_RE = re.compile(r'class="t">(?P<v>[^<]*)<')
_LS_DATE_RE = re.compile(r'class="c-date">(?P<v>[^<]*)<')
_LS_DETAIL_CUT = "이전 글"


@dataclass(frozen=True)
class NoticeWatchConfig:
    """Tunables for one watcher run (all externalized; no literals in logic)."""

    first_run_lookback_days: int = 14
    max_pages: int = 3
    page_size: int = 10
    http_timeout_sec: float = 20.0
    request_gap_sec: float = 1.0
    seen_cap: int = 300
    surface_max_age_days: int = 30
    keywords: tuple[str, ...] = (
        "유량", "중단", "종료", "폐지", "점검", "도메인", "인증", "토큰", "세션", "순단",
        "nxt", "넥스트레이드", "애프터마켓", "rate limit", "deprecat", "maintenance",
    )  # fmt: skip


DEFAULT_CONFIG = NoticeWatchConfig()


@dataclass(frozen=True)
class NoticeRef:
    broker: str
    notice_id: str
    title: str
    posted: str
    body: str | None = None


@dataclass(frozen=True)
class ProjectImpact:
    """A used TR/path named by a notice, with the project files that reference it."""

    project: str
    token: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class NoticeVerdict:
    ref: NoticeRef
    impact: tuple[ProjectImpact, ...]
    keywords: tuple[str, ...]
    empty_body: bool
    broker_projects: tuple[str, ...] = ()

    @property
    def needs_attention(self) -> bool:
        return bool(self.impact or self.keywords or self.empty_body)


class NoticeSource(Protocol):
    broker: str
    board_url: str

    def list_page(self, page: int) -> list[NoticeRef]:
        """Newest-first notices of the 0-based page; empty list means no more pages."""

    def fetch_body(self, ref: NoticeRef) -> str:
        """Plain-text body of one notice."""


class SourceFaultError(Exception):
    """A listing/body fetch that cannot be trusted (HTTP error, schema drift, empty board)."""


def html_to_text(raw: str) -> str:
    """Strip tags/scripts and unescape entities; block-level closers become newlines."""
    text = _STYLE_BLOCK_RE.sub(" ", raw)
    text = re.sub(r"<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", text, flags=re.IGNORECASE)
    text = html.unescape(_TAG_RE.sub(" ", text)).replace("\xa0", " ")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def parse_ls_list(page_html: str) -> list[NoticeRef]:
    """Parse the LS notice list fragment; rows without id/title/date are schema drift."""
    refs: list[NoticeRef] = []
    for match in _LS_ROW_RE.finditer(page_html):
        title = _LS_TITLE_RE.search(match["row"])
        posted = _LS_DATE_RE.search(match["row"])
        if title is None or posted is None:
            raise SourceFaultError("ls row without title/date")
        refs.append(NoticeRef(BROKER_LS, match["id"], html.unescape(title["v"]).strip(), posted["v"].strip()))
    return refs


def parse_ls_body(detail_html: str) -> str:
    text = html_to_text(detail_html)
    return text.split(_LS_DETAIL_CUT)[0].strip()


def parse_uno_posts(payload: Any, broker: str) -> list[NoticeRef]:
    """Parse the KIS/DB 'uno' forum JSON list; a non-list payload is schema drift."""
    if not isinstance(payload, list):
        raise SourceFaultError(f"{broker} posts payload is {type(payload).__name__}")
    refs: list[NoticeRef] = []
    for item in payload:
        try:
            content = item.get("content")
            refs.append(
                NoticeRef(
                    broker,
                    str(item["id"]),
                    str(item["title"]).strip(),
                    str(item["createdDate"])[:10],
                    html_to_text(content) if isinstance(content, str) else None,
                )
            )
        except (KeyError, AttributeError) as exc:
            raise SourceFaultError(f"{broker} post schema drift: {exc!r}") from exc
    return refs


def parse_kiwoom_list(payload: Any) -> list[NoticeRef]:
    """Parse Kiwoom's notice list JSON, newest first (pinned rows are not trusted for order)."""
    try:
        items = payload["noticeList"]
        refs = [
            NoticeRef(
                BROKER_KIWOOM,
                str(item["seqid"]),
                str(item["ttl"]).strip(),
                str(item["makeTime"]).replace(".", "-"),
            )
            for item in items
        ]
    except (KeyError, TypeError) as exc:
        raise SourceFaultError(f"kiwoom list schema drift: {exc!r}") from exc
    return sorted(refs, key=lambda r: int(r.notice_id), reverse=True)


def parse_kiwoom_body(payload: Any) -> str:
    try:
        return html_to_text(str(payload["noticeDetail"]["dtlCntn"] or ""))
    except (KeyError, TypeError) as exc:
        raise SourceFaultError(f"kiwoom detail schema drift: {exc!r}") from exc


class _Http:
    """requests wrapper with a fixed timeout and inter-request gap (portals are not rate-limit friendly)."""

    def __init__(self, cfg: NoticeWatchConfig) -> None:
        self._cfg = cfg
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "Mozilla/5.0 (compatible; kca-api-notice-watch)"
        self._last = 0.0

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        wait = self._cfg.request_gap_sec - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self._session.request(method, url, timeout=self._cfg.http_timeout_sec, **kwargs)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SourceFaultError(f"{method} {url}: {exc!r}") from exc
        finally:
            self._last = time.monotonic()
        return resp

    def json(self, method: str, url: str, **kwargs: Any) -> Any:
        try:
            return self.request(method, url, **kwargs).json()
        except ValueError as exc:
            raise SourceFaultError(f"{method} {url}: non-JSON response") from exc


class LsSource:
    broker = BROKER_LS
    board_url = "https://openapi.ls-sec.co.kr/community-notice"
    _forum = "https://openapi.ls-sec.co.kr/community/10000000-0000-0011-0000-000000000001"

    def __init__(self, http: _Http, cfg: NoticeWatchConfig) -> None:
        self._http = http

    def list_page(self, page: int) -> list[NoticeRef]:
        resp = self._http.request("GET", self._forum, params={"tab": "", "page": page}, headers={"HX-Request": "true"})
        return parse_ls_list(resp.text)

    def fetch_body(self, ref: NoticeRef) -> str:
        resp = self._http.request(
            "GET", f"{self._forum}/posts/{ref.notice_id}", params={"page": 0, "tab": ""}, headers={"HX-Request": "true"}
        )
        return parse_ls_body(resp.text)


class UnoForumSource:
    """KIS and DB증권 share the same forum JSON API; `sort` is mandatory (server 500s without it)."""

    def __init__(self, broker: str, host: str, board_path: str, http: _Http, cfg: NoticeWatchConfig) -> None:
        self.broker = broker
        self.board_url = f"{host}{board_path}"
        self._api = f"{host}/api/forums/10000000-0000-0011-0000-000000000001/posts"
        self._http = http
        self._size = cfg.page_size

    def list_page(self, page: int) -> list[NoticeRef]:
        params = {"page": page, "size": self._size, "sort": "createdDate,desc"}
        return parse_uno_posts(self._http.json("GET", self._api, params=params), self.broker)

    def fetch_body(self, ref: NoticeRef) -> str:
        if ref.body is not None:
            return ref.body
        payload = self._http.json("GET", f"{self._api}/{ref.notice_id}")
        try:
            return html_to_text(str(payload["content"] or ""))
        except (KeyError, TypeError) as exc:
            raise SourceFaultError(f"{self.broker} detail schema drift: {exc!r}") from exc


class KiwoomSource:
    broker = BROKER_KIWOOM
    board_url = "https://openapi.kiwoom.com/board/Board0101View"
    _base = "https://bbn.kiwoom.com/bbs"

    def __init__(self, http: _Http, cfg: NoticeWatchConfig) -> None:
        self._http = http
        self._size = cfg.page_size
        self._headers = {"Referer": "https://bbn.kiwoom.com/"}

    def list_page(self, page: int) -> list[NoticeRef]:
        payload = self._http.json(
            "POST",
            f"{self._base}/SBbsNoticeNOPAZListAjax",
            data={"pageNo": page + 1, "pageSize": self._size},
            headers=self._headers,
        )
        return parse_kiwoom_list(payload)

    def fetch_body(self, ref: NoticeRef) -> str:
        payload = self._http.json(
            "POST", f"{self._base}/SBbsNoticeNOPAZDetailAjax", data={"seqid": ref.notice_id}, headers=self._headers
        )
        return parse_kiwoom_body(payload)


def build_sources(cfg: NoticeWatchConfig = DEFAULT_CONFIG) -> tuple[NoticeSource, ...]:
    http = _Http(cfg)
    return (
        LsSource(http, cfg),
        UnoForumSource(BROKER_KIS, "https://apiportal.koreainvestment.com", "/community-notice", http, cfg),
        KiwoomSource(http, cfg),
        UnoForumSource(BROKER_DBSEC, "https://openapi.dbsec.co.kr", "/community-notice", http, cfg),
    )


def _token_in_text(token: str, text_lower: str) -> bool:
    if token.startswith("/"):
        return token in text_lower
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?!\d)", text_lower) is not None


def judge_notice(ref: NoticeRef, body: str, surface: Surface, cfg: NoticeWatchConfig) -> NoticeVerdict:
    """Match title+body against every project's used TR/path tokens and risk keywords; empty body needs review."""
    haystack = f"{ref.title}\n{body}".lower()
    impact = tuple(
        ProjectImpact(project, token, surface.files(project, token))
        for project, tokens in sorted(surface.used(ref.broker).items())
        for token in sorted(t for t in tokens if _token_in_text(t, haystack))
    )
    keywords = tuple(k for k in cfg.keywords if k in haystack)
    return NoticeVerdict(
        ref, impact, keywords, empty_body=not body.strip(), broker_projects=surface.projects_using(ref.broker)
    )


def collect_new(
    source: NoticeSource, seen: frozenset[str] | None, *, today: date, cfg: NoticeWatchConfig
) -> list[NoticeRef]:
    """New notices since the last run; on first run (seen is None) only those inside the lookback window."""
    cutoff = None if seen is not None else (today - timedelta(days=cfg.first_run_lookback_days)).isoformat()
    fresh_all: list[NoticeRef] = []
    for page in range(cfg.max_pages):
        refs = source.list_page(page)
        if not refs:
            if page == 0:
                raise SourceFaultError(f"{source.broker} board returned no notices")
            break
        fresh = [r for r in refs if (r.notice_id not in seen if seen is not None else r.posted >= (cutoff or ""))]
        fresh_all.extend(fresh)
        if len(fresh) < len(refs):
            break
    return fresh_all


def _canonical_path(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path)


def _resolve_refs(node: Any, components: Mapping[str, Any], trail: tuple[str, ...] = ()) -> Any:
    """Inline `#/components/...` $refs so a schema change shows up in every path that uses it."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/"):
            if ref in trail:
                return {"$cycle": ref}
            target: Any = {"components": components}
            for part in ref[len("#/") :].split("/"):
                target = target.get(part, {}) if isinstance(target, dict) else {}
            return _resolve_refs(target, components, (*trail, ref))
        return {k: _resolve_refs(v, components, trail) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, components, trail) for v in node]
    return node


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class TossSpecSnapshot:
    openapi_sha: str
    asyncapi_sha: str
    used_path_sha: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        return {"openapi_sha": self.openapi_sha, "asyncapi_sha": self.asyncapi_sha, "used_path_sha": self.used_path_sha}

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> TossSpecSnapshot:
        return cls(str(raw["openapi_sha"]), str(raw["asyncapi_sha"]), dict(raw["used_path_sha"]))


def snapshot_toss_spec(openapi: Mapping[str, Any], asyncapi: Mapping[str, Any], used_paths: frozenset[str]) -> TossSpecSnapshot:
    """Hash the whole specs plus each used endpoint with its referenced schemas inlined."""
    components = openapi.get("components", {})
    wanted = {_canonical_path(p) for p in used_paths}
    used_sha = {
        path: _sha(_resolve_refs(item, components))
        for path, item in openapi.get("paths", {}).items()
        if _canonical_path(path) in wanted
    }
    return TossSpecSnapshot(_sha(openapi), _sha(asyncapi), used_sha)


def diff_toss_spec(prior: TossSpecSnapshot, now: TossSpecSnapshot, used_paths: frozenset[str]) -> tuple[list[str], list[str]]:
    """Return (actionable, informational) change lines for the Toss specs."""
    actionable: list[str] = []
    for path in sorted(set(prior.used_path_sha) | set(now.used_path_sha)):
        before, after = prior.used_path_sha.get(path), now.used_path_sha.get(path)
        if before is not None and after is None:
            actionable.append(f"사용 중인 엔드포인트 삭제: {path}")
        elif before is None and after is not None:
            actionable.append(f"엔드포인트 신규 매칭: {path}")
        elif before != after:
            actionable.append(f"사용 중인 엔드포인트 명세 변경: {path}")
    present = {_canonical_path(p) for p in now.used_path_sha}
    actionable.extend(
        f"사용 중인 경로가 명세에 없음: {p}" for p in sorted(used_paths) if _canonical_path(p) not in present
    )
    info: list[str] = []
    if prior.openapi_sha != now.openapi_sha and not actionable:
        info.append("openapi.json 변경(사용 엔드포인트 외)")
    if prior.asyncapi_sha != now.asyncapi_sha:
        info.append("asyncapi.json 변경")
    return actionable, info


def fetch_toss_specs(http: _Http) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    base = "https://openapi.tossinvest.com/openapi-docs/latest"
    openapi = http.json("GET", f"{base}/openapi.json")
    asyncapi = http.json("GET", f"{base}/asyncapi.json")
    if not isinstance(openapi, dict) or not openapi.get("paths") or not isinstance(asyncapi, dict):
        raise SourceFaultError("toss spec schema drift")
    return openapi, asyncapi


@dataclass
class WatchReport:
    attention: list[NoticeVerdict] = field(default_factory=list)
    quiet: list[NoticeVerdict] = field(default_factory=list)
    toss_changes: list[str] = field(default_factory=list)
    toss_info: list[str] = field(default_factory=list)
    new_faults: list[str] = field(default_factory=list)


_FILES_SHOWN: int = 2


def _impact_text(impact: Sequence[ProjectImpact]) -> str:
    return ", ".join(f"{i.project}:{i.token}({', '.join(i.files[:_FILES_SHOWN])})" for i in impact)


def format_report(report: WatchReport, boards: Mapping[str, str]) -> tuple[str, str]:
    count = len(report.attention) + len(report.toss_changes) + len(report.new_faults)
    lines: list[str] = []
    for verdict in report.attention:
        reasons = [f"영향 {_impact_text(verdict.impact)}"] if verdict.impact else []
        if verdict.keywords:
            reasons.append(f"키워드 {', '.join(verdict.keywords)} (브로커 공통, 사용 프로젝트 {', '.join(verdict.broker_projects)})")
        if verdict.empty_body:
            reasons.append("본문 없음(첨부/이미지 가능성, 직접 확인)")
        lines.append(f"• [{verdict.ref.broker}] {verdict.ref.posted} {verdict.ref.title} — {'; '.join(reasons)}")
    lines.extend(f"• [toss] {change}" for change in report.toss_changes)
    lines.extend(f"• [감시기 이상] {fault}" for fault in report.new_faults)
    lines.append(f"참고: 해당 없음으로 분류된 신규 공지 {len(report.quiet)}건")
    lines.append("게시판: " + " | ".join(f"{broker} {url}" for broker, url in boards.items()))
    lines.append(PENDING_HINT)
    return f"API 공지 감시: 확인 필요 {count}건", "\n".join(lines)


def _pending_from_verdict(verdict: NoticeVerdict, today: date) -> tuple[str, dict[str, Any]]:
    ref = verdict.ref
    projects = sorted({i.project for i in verdict.impact} or set(verdict.broker_projects)) or [FALLBACK_PROJECT]
    return f"{ref.broker}:{ref.notice_id}", {
        "kind": "notice",
        "broker": ref.broker,
        "posted": ref.posted,
        "title": ref.title,
        "projects": projects,
        "impact": [{"project": i.project, "token": i.token, "files": list(i.files)} for i in verdict.impact],
        "keywords": list(verdict.keywords),
        "broker_projects": list(verdict.broker_projects),
        "empty_body": verdict.empty_body,
        "first_seen": today.isoformat(),
    }


def _pending_from_toss(
    changes: Sequence[str], snapshot: TossSpecSnapshot, projects: Sequence[str], today: date
) -> tuple[str, dict[str, Any]]:
    return f"toss:{snapshot.openapi_sha[:8]}", {
        "kind": "toss_spec",
        "broker": BROKER_TOSS,
        "posted": today.isoformat(),
        "title": "토스 OpenAPI 명세 변경",
        "projects": list(projects) or [FALLBACK_PROJECT],
        "changes": list(changes),
        "first_seen": today.isoformat(),
    }


def format_pending(pending: Mapping[str, Mapping[str, Any]]) -> str:
    """Render open pending actions for an AI/ops check (id, what changed, which project files to inspect)."""
    if not pending:
        return "조치 대기 항목 없음\n"
    lines: list[str] = []
    for action_id, item in sorted(pending.items(), key=lambda kv: kv[1]["first_seen"]):
        lines.append(f"[{action_id}] {item['posted']} {item['title']}")
        lines.extend(
            f"    - {entry['project']}: {entry['token']} → {', '.join(entry['files'])}" for entry in item.get("impact", [])
        )
        lines.extend(f"    - {change}" for change in item.get("changes", []))
        if item.get("keywords"):
            lines.append(f"    - 키워드: {', '.join(item['keywords'])} (브로커 공통 영향 가능, 점검 대상 프로젝트: {', '.join(item.get('broker_projects', []))})")
        if item.get("empty_body"):
            lines.append("    - 본문 없음: 게시판에서 직접 확인")
    return "\n".join(lines) + "\n"


def load_state(path: Path) -> dict[str, Any]:
    """Return the persisted state, {} when absent; raises OSError on corrupt evidence (never silently resets)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise OSError(f"Unreadable api notice watch state: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != STATE_SCHEMA_VERSION:
        raise OSError(f"Invalid api notice watch state: {path}")
    return raw


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema_version": STATE_SCHEMA_VERSION, **state}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run_watch(
    sources: Sequence[NoticeSource],
    *,
    surface: Surface,
    state_path: Path,
    today: date,
    cfg: NoticeWatchConfig = DEFAULT_CONFIG,
    toss_fetch: Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None,
    persist: bool = True,
    now: datetime | None = None,
) -> WatchReport:
    """One pass: fetch new notices, judge impact per project, diff the Toss specs, then persist state.

    State advances only for sources that fetched cleanly; a faulted source is reported once per fault episode.
    Actionable items are stored as pending actions until `ack_pending`; `last_run_at` lets the dashboard detect
    a silent watcher. `persist=False` is a dry run (no state write).
    """
    state = load_state(state_path)
    brokers_state: dict[str, Any] = dict(state.get("brokers", {}))
    faults: dict[str, str] = dict(state.get("faults", {}))
    pending: dict[str, Any] = dict(state.get("pending", {}))
    report = WatchReport()

    def record_fault(key: str, message: str) -> None:
        if key not in faults:
            report.new_faults.append(f"{key}: {message}")
            faults[key] = today.isoformat()
        logger.error("[SYS] stage=api_notice_watch source=%s status=FAULT reason=%s", key, message)

    manifest_age = surface.age_days(datetime.combine(today, datetime.min.time(), tzinfo=KST))
    if manifest_age > cfg.surface_max_age_days:
        record_fault("surface", f"api_surface.json is {manifest_age}d old; run `python -m src.tools.api_surface --sync`")
    else:
        faults.pop("surface", None)

    for source in sources:
        prior = brokers_state.get(source.broker, {}).get("seen")
        seen = frozenset(prior) if prior is not None else None
        try:
            fresh = collect_new(source, seen, today=today, cfg=cfg)
            verdicts = [judge_notice(ref, source.fetch_body(ref), surface, cfg) for ref in fresh]
        except SourceFaultError as exc:
            record_fault(source.broker, str(exc))
            continue
        faults.pop(source.broker, None)
        report.attention.extend(v for v in verdicts if v.needs_attention)
        report.quiet.extend(v for v in verdicts if not v.needs_attention)
        ids = [r.notice_id for r in fresh]
        brokers_state[source.broker] = {"seen": (ids + list(prior or []))[: cfg.seen_cap]}
        logger.info(
            "[SYS] stage=api_notice_watch source=%s new=%d attention=%d", source.broker, len(fresh),
            sum(v.needs_attention for v in verdicts),
        )  # fmt: skip

    toss_state = state.get("toss")
    toss_snapshot: TossSpecSnapshot | None = None
    if toss_fetch is not None:
        toss_paths = surface.attributed(BROKER_TOSS)
        try:
            openapi, asyncapi = toss_fetch()
            toss_snapshot = snapshot_toss_spec(openapi, asyncapi, toss_paths)
        except SourceFaultError as exc:
            record_fault(BROKER_TOSS, str(exc))
        else:
            faults.pop(BROKER_TOSS, None)
            if toss_state is not None:
                report.toss_changes, report.toss_info = diff_toss_spec(
                    TossSpecSnapshot.from_json(toss_state), toss_snapshot, toss_paths
                )
            toss_state = toss_snapshot.to_json()

    if not persist:
        return report
    for verdict in report.attention:
        action_id, item = _pending_from_verdict(verdict, today)
        pending.setdefault(action_id, item)
    if report.toss_changes and toss_snapshot is not None:
        action_id, item = _pending_from_toss(report.toss_changes, toss_snapshot, surface.projects_using(BROKER_TOSS), today)
        pending.setdefault(action_id, item)
    write_state(
        state_path,
        {
            "brokers": brokers_state,
            "faults": faults,
            "toss": toss_state,
            "pending": pending,
            "watched_projects": sorted(surface.projects),
            "last_run_at": (now or datetime.now(KST)).isoformat(timespec="seconds"),
        },
    )
    return report


def ack_pending(state_path: Path, action_ids: Sequence[str]) -> list[str]:
    """Remove acknowledged pending actions; returns the ids that were not pending (typos are not silent)."""
    state = load_state(state_path)
    pending: dict[str, Any] = dict(state.get("pending", {}))
    unknown = [a for a in action_ids if a not in pending]
    for action_id in action_ids:
        pending.pop(action_id, None)
    write_state(state_path, {**{k: v for k, v in state.items() if k != "schema_version"}, "pending": pending})
    return unknown


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entry; logic covered via run_watch scenarios
    parser = argparse.ArgumentParser(description="Broker API notice watcher (records notices touching our API surface for quant-dashboard)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the report; keep state untouched")
    mode.add_argument("--pending", action="store_true", help="list open pending actions and exit")
    mode.add_argument("--ack", nargs="+", metavar="ID", help="acknowledge handled pending actions and exit")
    args = parser.parse_args(argv)
    configure_cli_logging("%(asctime)s %(levelname)s %(message)s")
    state_path = Path(settings.DATA_DIR) / STATE_RELPATH
    if args.pending:
        sys.stdout.write(format_pending(load_state(state_path).get("pending", {})))
        return 0
    if args.ack:
        unknown = ack_pending(state_path, args.ack)
        sys.stdout.write(f"acked {len(args.ack) - len(unknown)}; unknown: {unknown}\n")
        return 1 if unknown else 0
    cfg = DEFAULT_CONFIG
    http = _Http(cfg)
    report = run_watch(
        build_sources(cfg),
        surface=load_surface(),
        state_path=state_path,
        today=datetime.now(KST).date(),
        cfg=cfg,
        toss_fetch=lambda: fetch_toss_specs(http),
        persist=not args.dry_run,
    )
    if args.dry_run:
        subject, body = format_report(report, {})
        sys.stdout.write(f"{subject}\n{body}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
