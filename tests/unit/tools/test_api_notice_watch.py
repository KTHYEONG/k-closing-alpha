"""Invariants for the broker API notice watcher."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from src.tools import api_notice_watch as w
from src.tools.api_surface import Surface, TokenEntry

TODAY = date(2026, 10, 20)
NOW = datetime(2026, 10, 20, 17, 40, tzinfo=w.KST)
CFG = w.DEFAULT_CONFIG
LS_ID = "03f7e4d4-cdb3-4276-a7b6-0f27ca5ebfb5"


FRESH = "2026-10-19T00:00:00+09:00"


def _surface(
    ls: frozenset[str] = frozenset(), toss: frozenset[str] = frozenset(), extra: dict[str, dict[str, TokenEntry]] | None = None
) -> Surface:
    """kca uses `ls`/`toss` tokens; `extra` adds other projects (project -> token -> entry)."""
    kca = {t: TokenEntry(frozenset({"ls"}), ("src/api/ls/client.py",)) for t in ls}
    kca.update({t: TokenEntry(frozenset({"toss"}), ("src/api/toss/client.py",)) for t in toss})
    return Surface(FRESH, {"kca": kca, **(extra or {})})


def _ref(notice_id: str, posted: str, title: str = "공지", body: str | None = None, broker: str = "ls") -> w.NoticeRef:
    return w.NoticeRef(broker, notice_id, title, posted, body)


@dataclass
class FakeSource:
    broker: str = "ls"
    board_url: str = "https://example.test/notice"
    pages: list[list[w.NoticeRef]] = field(default_factory=list)
    bodies: dict[str, str] = field(default_factory=dict)
    fail: bool = False

    def list_page(self, page: int) -> list[w.NoticeRef]:
        if self.fail:
            raise w.SourceFaultError("boom")
        return self.pages[page] if page < len(self.pages) else []

    def fetch_body(self, ref: w.NoticeRef) -> str:
        return self.bodies.get(ref.notice_id, "")


def _run(
    source: FakeSource,
    state: Path,
    surface: Surface | None = None,
    **kwargs: Any,
) -> w.WatchReport:
    return w.run_watch(
        [source],
        surface=surface or _surface(ls=frozenset({"t8412"})),
        state_path=state,
        today=TODAY,
        now=NOW,
        **kwargs,
    )


def test_parse_ls_list_extracts_id_title_date_and_rejects_drift() -> None:
    row = (
        f'<tr hx-get="/community/x/posts/{LS_ID}?page=0&amp;tab="><td><span class="t">[공지] 제목</span></td>'
        '<td class="c-date">2026-10-16</td></tr>'
    )
    assert w.parse_ls_list(row) == [_ref(LS_ID, "2026-10-16", "[공지] 제목")]
    with pytest.raises(w.SourceFaultError):
        w.parse_ls_list(f'<tr hx-get="/community/x/posts/{LS_ID}"><td>no title</td></tr>')


def test_parse_ls_body_drops_neighbor_navigation_and_scripts() -> None:
    page = "<p>본문 t8412</p><script>var x=1</script><div>이전 글</div><div>[다른 공지] t1410</div>"
    assert w.parse_ls_body(page) == "본문 t8412"


def test_parse_uno_posts_keeps_inline_body_and_flags_drift() -> None:
    posts = [{"id": "a", "title": " T ", "createdDate": "2026-09-09T12:40:20.000+0900", "content": "<p>x&nbsp;y</p>"}]
    assert w.parse_uno_posts(posts, "kis") == [w.NoticeRef("kis", "a", "T", "2026-09-09", "x y")]
    assert w.parse_uno_posts([{"id": "b", "title": "t", "createdDate": "2026-08-12"}], "dbsec")[0].body is None
    with pytest.raises(w.SourceFaultError):
        w.parse_uno_posts({"error": 500}, "kis")
    with pytest.raises(w.SourceFaultError):
        w.parse_uno_posts([{"id": "a"}], "kis")


def test_parse_kiwoom_list_orders_by_seqid_not_server_order() -> None:
    payload = {
        "noticeList": [
            {"seqid": 28, "ttl": "pinned", "makeTime": "2025.11.06"},
            {"seqid": 63, "ttl": "new", "makeTime": "2026.09.21"},
        ]
    }
    refs = w.parse_kiwoom_list(payload)
    assert [r.notice_id for r in refs] == ["63", "28"]
    assert refs[0].posted == "2026-09-21"
    with pytest.raises(w.SourceFaultError):
        w.parse_kiwoom_list({"result": "N"})
    assert w.parse_kiwoom_body({"noticeDetail": {"dtlCntn": None}}) == ""


def test_judge_notice_matches_tr_on_word_boundary_and_names_every_affected_project() -> None:
    krx = {"t8412": TokenEntry(frozenset({"ls"}), ("src/realtime/adapters/ls.py",)), "s3_": TokenEntry(frozenset({"ls"}), ("x.py",))}
    surface = _surface(ls=frozenset({"t8412"}), extra={"krx": krx})
    hit = w.judge_notice(_ref("1", "2026-10-07"), "t8412InBlock 필드 추가", surface, CFG)
    assert [(i.project, i.token, i.files) for i in hit.impact] == [
        ("kca", "t8412", ("src/api/ls/client.py",)),
        ("krx", "t8412", ("src/realtime/adapters/ls.py",)),
    ]
    assert not w.judge_notice(_ref("2", "2026-10-07"), "t84120 및 xt8412", surface, CFG).impact


def test_judge_notice_matches_paths_case_insensitively_and_only_for_the_notice_broker() -> None:
    path = "/uapi/domestic-stock/v1/quotations/INQUIRE-price"
    kis = {path: TokenEntry(frozenset({"kis"}), ("k.py",))}
    surface = Surface(FRESH, {"kse": kis})
    kis_ref = _ref("3", "2026-10-07", broker="kis")
    assert w.judge_notice(kis_ref, f"경로 {path.upper()} 변경", surface, CFG).impact[0].project == "kse"
    assert not w.judge_notice(_ref("4", "2026-10-07", broker="ls"), f"경로 {path}", surface, CFG).impact


def test_judge_notice_flags_empty_body_and_keywords_but_ignores_irrelevant() -> None:
    surface = _surface(extra={"krx": {"s3_": TokenEntry(frozenset({"ls"}), ("x.py",))}})
    assert w.judge_notice(_ref("1", "2026-10-07"), "  ", surface, CFG).empty_body
    keyword = w.judge_notice(_ref("2", "2026-10-07"), "유량제어 시행", surface, CFG)
    assert keyword.keywords == ("유량",) and keyword.broker_projects == ("krx",)
    assert not w.judge_notice(_ref("3", "2026-10-07", title="이벤트"), "수수료 할인", surface, CFG).needs_attention


def test_collect_new_first_run_applies_lookback_and_later_runs_use_seen() -> None:
    source = FakeSource(pages=[[_ref("3", "2026-10-16"), _ref("2", "2026-10-10"), _ref("1", "2026-06-01")]])
    first = w.collect_new(source, None, today=TODAY, cfg=CFG)
    assert [r.notice_id for r in first] == ["3", "2"]
    later = w.collect_new(source, frozenset({"2", "1"}), today=TODAY, cfg=CFG)
    assert [r.notice_id for r in later] == ["3"]


def test_collect_new_pages_until_seen_id_and_caps_pages() -> None:
    pages = [[_ref("5", "2026-10-16"), _ref("4", "2026-10-15")], [_ref("3", "2026-10-14"), _ref("2", "2026-10-13")]]
    source = FakeSource(pages=pages)
    assert [r.notice_id for r in w.collect_new(source, frozenset({"2"}), today=TODAY, cfg=CFG)] == ["5", "4", "3"]
    capped = w.collect_new(source, frozenset({"0"}), today=TODAY, cfg=w.NoticeWatchConfig(max_pages=1))
    assert [r.notice_id for r in capped] == ["5", "4"]


def test_empty_first_page_is_a_fault_not_silence() -> None:
    with pytest.raises(w.SourceFaultError):
        w.collect_new(FakeSource(pages=[]), frozenset({"1"}), today=TODAY, cfg=CFG)


def test_run_watch_alerts_once_then_stays_quiet_and_persists_seen(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    source = FakeSource(
        pages=[[_ref("2", "2026-10-16", "TR 변경"), _ref("1", "2026-10-15", "이벤트")]],
        bodies={"2": "t8412 입력 필드 추가", "1": "수수료 할인"},
    )
    first = _run(source, state)
    assert [v.ref.notice_id for v in first.attention] == ["2"] and len(first.quiet) == 1
    saved = w.load_state(state)
    assert saved["last_run_at"] == "2026-10-20T17:40:00+09:00" and set(saved["pending"]) == {"ls:2"}
    assert saved["watched_projects"] == ["kca"]
    again = _run(source, state)
    assert not again.attention
    assert w.load_state(state)["brokers"]["ls"]["seen"] == ["2", "1"]


def test_run_watch_fault_alerts_once_per_episode_and_does_not_advance_state(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    healthy = FakeSource(pages=[[_ref("1", "2026-10-16")]], bodies={"1": "t8412"})
    _run(healthy, state)
    broken = FakeSource(fail=True)
    assert _run(broken, state).new_faults
    assert not _run(broken, state).new_faults
    saved = w.load_state(state)
    assert saved["brokers"]["ls"]["seen"] == ["1"] and "ls" in saved["faults"]
    _run(healthy, state)
    assert "ls" not in w.load_state(state)["faults"]


def test_dry_run_does_not_write_state(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    source = FakeSource(pages=[[_ref("1", "2026-10-16")]], bodies={"1": "t8412"})
    report = w.run_watch([source], surface=_surface(ls=frozenset({"t8412"})), state_path=state, today=TODAY, persist=False)
    assert report.attention and not state.exists()


def test_load_state_rejects_corrupt_and_wrong_schema(tmp_path: Path) -> None:
    assert w.load_state(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(OSError, match="Unreadable"):
        w.load_state(bad)
    bad.write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(OSError, match="Invalid"):
        w.load_state(bad)


_OPENAPI: dict[str, Any] = {
    "paths": {
        "/api/v1/prices": {"get": {"responses": {"200": {"schema": {"$ref": "#/components/schemas/Price"}}}}},
        "/api/v1/stocks/{symbol}/program-trades": {"get": {"summary": "pt"}},
        "/api/v1/unused": {"get": {"summary": "u"}},
    },
    "components": {"schemas": {"Price": {"type": "object", "properties": {"close": {"type": "number"}}}}},
}
_USED = frozenset({"/api/v1/prices", "/api/v1/stocks/{id}/program-trades"})


def test_toss_snapshot_resolves_refs_so_schema_change_flags_used_path() -> None:
    before = w.snapshot_toss_spec(_OPENAPI, {"a": 1}, _USED)
    changed = copy.deepcopy(_OPENAPI)
    changed["components"]["schemas"]["Price"]["properties"]["close"]["type"] = "string"
    actionable, _ = w.diff_toss_spec(before, w.snapshot_toss_spec(changed, {"a": 1}, _USED), _USED)
    assert actionable == ["사용 중인 엔드포인트 명세 변경: /api/v1/prices"]
    assert set(before.used_path_sha) == {"/api/v1/prices", "/api/v1/stocks/{symbol}/program-trades"}


def test_toss_unused_changes_are_informational_and_removed_used_path_is_actionable() -> None:
    before = w.snapshot_toss_spec(_OPENAPI, {"a": 1}, _USED)
    unused = copy.deepcopy(_OPENAPI)
    unused["paths"]["/api/v1/unused"]["get"]["summary"] = "changed"
    actionable, info = w.diff_toss_spec(before, w.snapshot_toss_spec(unused, {"a": 2}, _USED), _USED)
    assert not actionable and info == ["openapi.json 변경(사용 엔드포인트 외)", "asyncapi.json 변경"]
    removed = copy.deepcopy(_OPENAPI)
    del removed["paths"]["/api/v1/prices"]
    actionable, _ = w.diff_toss_spec(before, w.snapshot_toss_spec(removed, {"a": 1}, _USED), _USED)
    assert "사용 중인 엔드포인트 삭제: /api/v1/prices" in actionable
    assert "사용 중인 경로가 명세에 없음: /api/v1/prices" in actionable


def test_run_watch_toss_seeds_silently_then_alerts_on_used_path_change(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    source = FakeSource(pages=[[_ref("1", "2026-10-16")]], bodies={"1": "수수료 할인"})
    specs = {"openapi": _OPENAPI}
    kwargs = {"toss_fetch": lambda: (specs["openapi"], {"a": 1})}
    used = _surface(toss=_USED)
    seed = w.run_watch([source], surface=used, state_path=state, today=TODAY, **kwargs)
    assert not seed.toss_changes and not w.load_state(state)["pending"]
    changed = copy.deepcopy(_OPENAPI)
    changed["paths"]["/api/v1/prices"]["get"]["summary"] = "x"
    specs["openapi"] = changed
    report = w.run_watch([source], surface=used, state_path=state, today=TODAY, **kwargs)
    assert report.toss_changes and next(iter(w.load_state(state)["pending"])).startswith("toss:")


def test_format_report_lists_reasons_and_counts() -> None:
    verdict = w.NoticeVerdict(_ref("1", "2026-10-07", "TR 안내"), (w.ProjectImpact("krx", "t8412", ("a.py", "b.py")),), ("유량",), empty_body=False, broker_projects=("kca", "krx"))
    report = w.WatchReport(attention=[verdict], toss_changes=["x"], new_faults=["kis: boom"])
    subject, body = w.format_report(report, {"ls": "https://ls"})
    assert "3건" in subject
    assert "krx:t8412(a.py, b.py)" in body and "유량" in body and "kca, krx" in body and w.PENDING_HINT in body and "[감시기 이상] kis: boom" in body and "https://ls" in body


@dataclass
class _Resp:
    text: str = ""
    payload: Any = None
    status: int = 200

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise w.requests.HTTPError(f"{self.status}")

    def json(self) -> Any:
        if self.payload is None:
            raise ValueError("not json")
        return self.payload


@dataclass
class _FakeHttp:
    """Stands in for `_Http`; routes by URL suffix and records calls."""

    routes: dict[str, Any]
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def _route(self, method: str, url: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append((method, url, kwargs))
        return next(v for k, v in self.routes.items() if url.endswith(k))

    def request(self, method: str, url: str, **kwargs: Any) -> _Resp:
        return _Resp(text=self._route(method, url, kwargs))

    def json(self, method: str, url: str, **kwargs: Any) -> Any:
        return self._route(method, url, kwargs)


def test_http_wrapper_maps_failures_to_source_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    http = w._Http(w.NoticeWatchConfig(request_gap_sec=0.0))
    responses = iter([_Resp(payload={"ok": 1}), _Resp(status=500), _Resp(text="<html>")])
    monkeypatch.setattr(http._session, "request", lambda *a, **k: next(responses))
    assert http.json("GET", "https://x.test/a") == {"ok": 1}
    with pytest.raises(w.SourceFaultError, match="500"):
        http.request("GET", "https://x.test/b")
    with pytest.raises(w.SourceFaultError, match="non-JSON"):
        http.json("GET", "https://x.test/c")


def test_http_wrapper_sleeps_to_honour_request_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    http = w._Http(w.NoticeWatchConfig(request_gap_sec=5.0))
    slept: list[float] = []
    monkeypatch.setattr(w.time, "sleep", slept.append)
    monkeypatch.setattr(http._session, "request", lambda *a, **k: _Resp(payload={}))
    http.request("GET", "https://x.test/a")
    http.request("GET", "https://x.test/b")
    assert len(slept) == 1 and 0 < slept[0] <= 5.0


def test_ls_source_requests_htmx_fragments() -> None:
    row = f'<tr hx-get="/c/posts/{LS_ID}"><span class="t">T</span><td class="c-date">2026-10-16</td></tr>'
    http = _FakeHttp({"0000-000000000001": row, f"posts/{LS_ID}": "<p>t8412</p>이전 글"})
    source = w.LsSource(http, CFG)  # type: ignore[arg-type]
    refs = source.list_page(2)
    assert refs[0].notice_id == LS_ID and http.calls[0][2]["params"]["page"] == 2
    assert source.fetch_body(refs[0]) == "t8412"
    assert http.calls[1][2]["headers"] == {"HX-Request": "true"}


def test_uno_source_sorts_server_side_and_fetches_body_only_when_missing() -> None:
    inline = _ref("a", "2026-09-09", body="inline", broker="kis")
    http = _FakeHttp(
        {
            "/posts": [{"id": "b", "title": "t", "createdDate": "2026-08-12"}],
            "/posts/b": {"content": "<p>detail</p>"},
        }
    )
    source = w.UnoForumSource("dbsec", "https://db.test", "/community-notice", http, CFG)  # type: ignore[arg-type]
    refs = source.list_page(1)
    assert http.calls[0][2]["params"] == {"page": 1, "size": CFG.page_size, "sort": "createdDate,desc"}
    assert source.fetch_body(refs[0]) == "detail" and source.fetch_body(inline) == "inline"
    http.routes["/posts/b"] = {"nope": 1}
    with pytest.raises(w.SourceFaultError):
        source.fetch_body(refs[0])


def test_kiwoom_source_uses_one_based_pages_and_referer() -> None:
    http = _FakeHttp(
        {
            "ListAjax": {"noticeList": [{"seqid": 63, "ttl": "t", "makeTime": "2026.09.21"}]},
            "DetailAjax": {"noticeDetail": {"dtlCntn": "<p>본문</p>"}},
        }
    )
    source = w.KiwoomSource(http, CFG)  # type: ignore[arg-type]
    refs = source.list_page(0)
    assert http.calls[0][2]["data"]["pageNo"] == 1 and http.calls[0][2]["headers"]["Referer"].startswith("https://bbn")
    assert source.fetch_body(refs[0]) == "본문" and http.calls[1][2]["data"] == {"seqid": "63"}


def test_build_sources_covers_every_notice_board() -> None:
    assert [s.broker for s in w.build_sources()] == ["ls", "kis", "kiwoom", "dbsec"]


def test_fetch_toss_specs_rejects_schema_drift() -> None:
    good = _FakeHttp({"openapi.json": _OPENAPI, "asyncapi.json": {"a": 1}})
    assert w.fetch_toss_specs(good)[0] is _OPENAPI  # type: ignore[arg-type]
    bad = _FakeHttp({"openapi.json": {"paths": {}}, "asyncapi.json": {"a": 1}})
    with pytest.raises(w.SourceFaultError):
        w.fetch_toss_specs(bad)  # type: ignore[arg-type]


def test_kiwoom_body_schema_drift_is_a_fault() -> None:
    with pytest.raises(w.SourceFaultError):
        w.parse_kiwoom_body({"result": "N"})


def test_toss_ref_cycles_and_lists_are_hashed_without_recursion_error() -> None:
    spec = {
        "paths": {"/api/v1/prices": {"get": {"parameters": [{"$ref": "#/components/schemas/Node"}]}}},
        "components": {"schemas": {"Node": {"properties": {"child": {"$ref": "#/components/schemas/Node"}}}}},
    }
    snapshot = w.snapshot_toss_spec(spec, {}, frozenset({"/api/v1/prices"}))
    assert set(snapshot.used_path_sha) == {"/api/v1/prices"}


def test_toss_newly_matched_endpoint_is_actionable() -> None:
    before = w.snapshot_toss_spec(_OPENAPI, {}, frozenset({"/api/v1/prices"}))
    after = w.snapshot_toss_spec(_OPENAPI, {}, _USED)
    actionable, _ = w.diff_toss_spec(before, after, _USED)
    assert actionable == ["엔드포인트 신규 매칭: /api/v1/stocks/{symbol}/program-trades"]


def test_digest_flags_empty_body_for_manual_review() -> None:
    verdict = w.NoticeVerdict(_ref("1", "2026-10-07", "첨부 공지"), (), (), empty_body=True)
    _, body = w.format_report(w.WatchReport(attention=[verdict]), {})
    assert "본문 없음" in body


def test_toss_fetch_fault_alerts_once_and_keeps_prior_snapshot(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    source = FakeSource(pages=[[_ref("1", "2026-10-16")]], bodies={"1": "수수료"})
    used = _surface(toss=_USED)

    w.run_watch([source], surface=used, state_path=state, today=TODAY, toss_fetch=lambda: (_OPENAPI, {"a": 1}))
    seeded = w.load_state(state)["toss"]

    def broken() -> tuple[dict[str, Any], dict[str, Any]]:
        raise w.SourceFaultError("spec 403")

    for _ in range(2):
        w.run_watch([source], surface=used, state_path=state, today=TODAY, toss_fetch=broken)
    assert w.load_state(state)["faults"] == {"toss": "2026-10-20"}
    assert w.load_state(state)["toss"] == seeded


def test_pending_actions_persist_dedupe_render_and_ack(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    surface = _surface(ls=frozenset({"t8412"}), extra={"krx": {"s3_": TokenEntry(frozenset({"ls"}), ("x.py",))}})
    source = FakeSource(
        pages=[[_ref("2", "2026-10-16", "유량 공지"), _ref("1", "2026-10-15", "TR 변경")]],
        bodies={"2": "세션 3개로 제한", "1": "t8412 필드 추가"},
    )
    _run(source, state, surface=surface)
    pending = w.load_state(state)["pending"]
    assert set(pending) == {"ls:1", "ls:2"}
    assert pending["ls:1"]["impact"] == [{"project": "kca", "token": "t8412", "files": ["src/api/ls/client.py"]}]
    assert pending["ls:2"]["broker_projects"] == ["kca", "krx"]
    assert pending["ls:1"]["projects"] == ["kca"] and pending["ls:2"]["projects"] == ["kca", "krx"]
    rendered = w.format_pending(pending)
    assert "kca: t8412 → src/api/ls/client.py" in rendered and "점검 대상 프로젝트: kca, krx" in rendered
    _run(source, state, surface=surface)
    assert set(w.load_state(state)["pending"]) == {"ls:1", "ls:2"}
    assert w.ack_pending(state, ["ls:1", "nope"]) == ["nope"]
    after = w.load_state(state)
    assert set(after["pending"]) == {"ls:2"} and after["brokers"]["ls"]["seen"] == ["2", "1"]
    assert w.format_pending({}) == "조치 대기 항목 없음\n"


def test_empty_body_pending_is_rendered_for_manual_check() -> None:
    verdict = w.NoticeVerdict(_ref("9", "2026-10-07", "첨부"), (), (), empty_body=True)
    action_id, item = w._pending_from_verdict(verdict, TODAY)
    assert item["projects"] == ["kca"]  # no impact and no broker users: falls back to the watcher owner
    assert "게시판에서 직접 확인" in w.format_pending({action_id: item})


def test_toss_spec_change_becomes_a_pending_action(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    source = FakeSource(pages=[[_ref("1", "2026-10-16")]], bodies={"1": "수수료"})
    surface = _surface(toss=_USED)
    specs = {"openapi": _OPENAPI}

    def run() -> None:
        w.run_watch(
            [source], surface=surface, state_path=state, today=TODAY, toss_fetch=lambda: (specs["openapi"], {"a": 1})
        )

    run()
    changed = copy.deepcopy(_OPENAPI)
    changed["paths"]["/api/v1/prices"]["get"]["summary"] = "x"
    specs["openapi"] = changed
    run()
    (action_id, item), = w.load_state(state)["pending"].items()
    assert action_id.startswith("toss:") and "사용 중인 엔드포인트 명세 변경: /api/v1/prices" in w.format_pending({action_id: item})
    assert item["projects"] == ["kca"]


def test_stale_surface_manifest_raises_one_fault_per_episode(tmp_path: Path) -> None:
    state = tmp_path / "s.json"
    stale = Surface("2026-01-01T00:00:00+09:00", {"kca": {}})
    source = FakeSource(pages=[[_ref("1", "2026-10-16")]], bodies={"1": "수수료"})
    assert _run(source, state, surface=stale).new_faults[0].startswith("surface:")
    assert not _run(source, state, surface=stale).new_faults
    _run(source, state, surface=_surface())
    assert "surface" not in w.load_state(state)["faults"]


_TOSS_SPEC_FIXTURE: dict[str, Any] = {
    "paths": {
        "/api/v1/candles": {"get": {"summary": "candles"}},
        "/api/v1/rankings": {"get": {"summary": "rankings"}},
        "/api/v1/stocks/{symbol}/program-trades": {"get": {"summary": "program trades"}},
        "/api/v1/market-calendar/KR": {"get": {"summary": "kr calendar"}},
    },
    "components": {"schemas": {}},
}


def test_toss_exact_match_is_clean() -> None:
    used = frozenset({"/api/v1/market-calendar/KR"})
    before = w.snapshot_toss_spec(_TOSS_SPEC_FIXTURE, {"a": 1}, used)
    after = w.snapshot_toss_spec(_TOSS_SPEC_FIXTURE, {"a": 1}, used)
    actionable, info = w.diff_toss_spec(before, after, used)
    assert actionable == [] and info == []


def test_toss_case_mismatch_is_info_only() -> None:
    used = frozenset({"/api/v1/market-calendar/kr"})
    before = w.snapshot_toss_spec(_TOSS_SPEC_FIXTURE, {"a": 1}, used)
    after = w.snapshot_toss_spec(_TOSS_SPEC_FIXTURE, {"a": 1}, used)
    actionable, info = w.diff_toss_spec(before, after, used)
    assert actionable == []
    assert len(info) == 1 and "/api/v1/market-calendar/kr" in info[0] and "/api/v1/market-calendar/KR" in info[0]


def test_toss_truly_missing_stays_actionable() -> None:
    used = frozenset({"/api/v1/market-calendar/XX"})
    before = w.snapshot_toss_spec(_TOSS_SPEC_FIXTURE, {"a": 1}, used)
    after = w.snapshot_toss_spec(_TOSS_SPEC_FIXTURE, {"a": 1}, used)
    actionable, _ = w.diff_toss_spec(before, after, used)
    assert actionable == ["사용 중인 경로가 명세에 없음: /api/v1/market-calendar/XX"]


@pytest.mark.parametrize("prior_paths", [{}, {"/api/v1/market-calendar/kr": {"get": {}}}])
def test_toss_case_mismatch_does_not_create_pending_on_baseline_transition(
    tmp_path: Path, prior_paths: dict[str, Any]
) -> None:
    used = frozenset({"/api/v1/market-calendar/kr"})
    state = tmp_path / "s.json"
    prior = w.snapshot_toss_spec({"paths": prior_paths}, {}, used)
    w.write_state(state, {"toss": prior.to_json()})
    report = w.run_watch(
        [], surface=_surface(toss=used), state_path=state, today=TODAY,
        toss_fetch=lambda: (_TOSS_SPEC_FIXTURE, {}),
    )
    assert report.toss_changes == []
    assert "경로 대소문자 불일치: /api/v1/market-calendar/kr vs /api/v1/market-calendar/KR" in report.toss_info
    assert not w.load_state(state)["pending"]
    saved = w.load_state(state)["toss"]["used_path_sha"]
    assert set(saved) == {"/api/v1/market-calendar/KR"}


def test_manifest_krx_toss_paths_exist_exact_case_in_spec() -> None:
    from src.tools.api_surface import load_surface as _load_manifest

    surface = _load_manifest()
    krx_toss_api = {t for t, e in surface.projects["krx"].items() if "toss" in e.brokers and t.startswith("/api/")}
    assert krx_toss_api
    assert "/api/v1/market-calendar/kr" not in surface.projects["krx"]
    for path in krx_toss_api:
        assert path in _TOSS_SPEC_FIXTURE["paths"]
    assert set(_TOSS_SPEC_FIXTURE["paths"]) == {
        "/api/v1/candles",
        "/api/v1/rankings",
        "/api/v1/stocks/{symbol}/program-trades",
        "/api/v1/market-calendar/KR",
    }
