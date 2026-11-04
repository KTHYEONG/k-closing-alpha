"""Research roster job invariant guards."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.config.collection import CollectionSettings
from src.daily import roster_capture
from src.data.capture_contracts import SEOUL, SessionClock
from src.data.capture_store import CaptureStore
from src.data.research_roster import RESEARCH_ROSTER_REASON, read_research_roster
from src.data.session_calendar import SessionDay, SessionKind
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_OK, RUN_OUTCOME_SKIPPED

DATE = "2026-11-04"
PREV = pd.Timestamp("2026-11-03")
DECISION = pd.Timestamp(DATE)
OBSERVED = dt.datetime(2026, 11, 4, 15, 20, 30, tzinfo=SEOUL)
COMPLETED = dt.datetime(2026, 11, 4, 15, 20, 32, tzinfo=SEOUL)


def _clock() -> SessionClock:
    return SessionClock(
        trading_date=dt.date(2026, 11, 4),
        open_at=dt.datetime(2026, 11, 4, 9, 0, tzinfo=SEOUL),
        close_at=dt.datetime(2026, 11, 4, 15, 30, tzinfo=SEOUL),
        close_confirmation_deadline=dt.datetime(2026, 11, 4, 15, 33, tzinfo=SEOUL),
        provenance="test",
    )


def _profile(tmp_path: Path, **overrides: Any) -> CollectionSettings:
    base: dict[str, Any] = {"COLLECTION_ROOT": tmp_path / "capture", "COLLECTION_RESEARCH_ROSTER_ENABLED": True}
    base.update(overrides)
    return CollectionSettings(**base)


def _row(code: str, trde_prica: str, price: str = "+72000") -> dict[str, str]:
    return {"stk_cd": f"{code}_AL", "now_rank": "1", "trde_prica": trde_prica, "cur_prc": price, "flu_rt": "+1.20"}


class _FakeKiwoom:
    """Vendor boundary: replays ranking pages through the same observer contract as the real client."""

    def __init__(self, pages: list[list[dict[str, str]]], *, rt_cd: str = "0", truncated: bool = False) -> None:
        self._pages = pages
        self._rt_cd = rt_cd
        self._truncated = truncated

    async def get_trade_value_ranking(
        self, session: Any, *, max_pages: int, min_trade_value_krw: int, stex_tp: str = "1", on_page: Any = None
    ) -> dict[str, Any]:
        started = COMPLETED - dt.timedelta(seconds=2)
        received = COMPLETED - dt.timedelta(seconds=1)
        for index, rows in enumerate(self._pages):
            cont = "Y" if index < len(self._pages) - 1 else "N"
            metadata = {"vendor": "kiwoom", "endpoint": "trade-value-ranking", "cont-yn": cont, "next-key": f"k{index}"}
            on_page({"return_code": 0, "trde_prica_upper": rows}, metadata, started, received, index, 0)
        output = [row for rows in self._pages for row in rows]
        return {"rt_cd": self._rt_cd, "output": output, "vendor": "kiwoom", "truncated": self._truncated, "msg1": ""}


def _run(
    tmp_path: Path,
    kiwoom: _FakeKiwoom,
    *,
    profile: CollectionSettings | None = None,
    times: list[dt.datetime] | None = None,
) -> Any:
    store = CaptureStore(tmp_path / "capture")
    clock_values = iter(times if times is not None else [OBSERVED, COMPLETED])
    return store, asyncio.run(
        roster_capture.run_research_roster(
            DATE,
            profile=profile or _profile(tmp_path),
            store=store,
            kiwoom_client=kiwoom,
            session_clock=_clock(),
            listed=frozenset({"005930", "000660"}),
            screenable=None,
            shares_by_symbol={"005930": 6e8, "000660": 1e8},
            as_of_panel_date=PREV.date(),
            now_fn=lambda: next(clock_values),
        )
    )


def test_run_rejects_outside_pit_window(tmp_path: Path) -> None:
    kiwoom = _FakeKiwoom([[_row("005930", "50000")]])
    for observed in (
        dt.datetime(2026, 11, 4, 15, 19, 59, tzinfo=SEOUL),
        dt.datetime(2026, 11, 4, 15, 30, 0, tzinfo=SEOUL),
    ):
        store = CaptureStore(tmp_path / "capture")
        with pytest.raises(ValueError, match="outside_pit_window"):
            asyncio.run(
                roster_capture.run_research_roster(
                    DATE,
                    profile=_profile(tmp_path),
                    store=store,
                    kiwoom_client=kiwoom,
                    session_clock=_clock(),
                    listed=frozenset({"005930"}),
                    screenable=None,
                    shares_by_symbol={"005930": 6e8},
                    as_of_panel_date=PREV.date(),
                    now_fn=lambda observed=observed: observed,
                )
            )
        assert store.read_manifests(DATE) == ()


def test_run_publishes_and_persists_raw_pages(tmp_path: Path) -> None:
    kiwoom = _FakeKiwoom([[_row("005930", "50000")], [_row("000660", "20000", "+90000")]])

    store, roster = _run(tmp_path, kiwoom)

    manifests = store.read_manifests(DATE)
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.context.capture_reason == RESEARCH_ROSTER_REASON
    assert manifest.cohort is None
    assert len(manifest.entries[0].raw_refs) == 2
    assert all(
        ref.path.startswith(f"raw/{DATE}/kiwoom/SCAN/trade-value-ranking/{roster.roster_id}/")
        for ref in manifest.entries[0].raw_refs
    )
    assert roster.members == ("005930", "000660")
    assert read_research_roster(store, DATE, available_by=COMPLETED + dt.timedelta(minutes=1)).members == roster.members


def test_run_ranking_unavailable_publishes_nothing(tmp_path: Path) -> None:
    kiwoom = _FakeKiwoom([[]], rt_cd="1", truncated=False)
    store = CaptureStore(tmp_path / "capture")

    with pytest.raises(RuntimeError, match="ranking_unavailable"):
        asyncio.run(
            roster_capture.run_research_roster(
                DATE,
                profile=_profile(tmp_path),
                store=store,
                kiwoom_client=kiwoom,
                session_clock=_clock(),
                listed=frozenset(),
                screenable=None,
                shares_by_symbol={},
                as_of_panel_date=PREV.date(),
                now_fn=lambda: OBSERVED,
            )
        )
    assert store.read_manifests(DATE) == ()


def test_run_disabled_raises(tmp_path: Path) -> None:
    kiwoom = _FakeKiwoom([[_row("005930", "50000")]])

    with pytest.raises(ValueError, match="research roster disabled"):
        _run(tmp_path, kiwoom, profile=CollectionSettings(COLLECTION_ROOT=tmp_path / "capture"))


def test_shares_from_prev_day_only(tmp_path: Path) -> None:
    panel = tmp_path / "price_history.parquet"
    pd.DataFrame(
        {
            "date": [PREV, PREV, DECISION],
            "symbol": ["005930", "000660", "005930"],
            "market_cap_100m": [1000.0, 500.0, 2000.0],
            "close_raw": [10000.0, 0.0, 20000.0],
        }
    ).to_parquet(panel, index=False)

    shares = roster_capture.load_prev_day_shares(DECISION, prev_trading_day=PREV, path=panel)

    assert shares == pytest.approx({"005930": 1000.0 * 1e8 / 10000.0})


def test_latest_panel_date_before_returns_latest_prior_date(tmp_path: Path) -> None:
    panel = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2026-11-02"), DECISION], "symbol": ["005930", "005930"]}).to_parquet(
        panel, index=False
    )

    assert roster_capture.latest_panel_date_before(DECISION, path=panel) == pd.Timestamp("2026-11-02")


def test_latest_panel_date_before_stale(tmp_path: Path) -> None:
    panel = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2026-10-15")], "symbol": ["005930"]}).to_parquet(panel, index=False)

    with pytest.raises(ValueError, match="stale price_history"):
        roster_capture.latest_panel_date_before(DECISION, path=panel)


def test_main_disabled_records_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLLECTION_RESEARCH_ROSTER_ENABLED", raising=False)
    calls: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(
        roster_capture,
        "record_run_outcome",
        lambda job, outcome, **kw: calls.append((job, outcome, kw)) or {},
    )

    roster_capture.main(["--date", DATE])

    assert calls == [("research_roster", RUN_OUTCOME_SKIPPED, {"run_date": DATE, "reason": "disabled"})]


def test_now_is_kst_aware() -> None:
    assert roster_capture._now().utcoffset() == dt.timedelta(hours=9)


def test_run_rejects_session_clock_of_another_date(tmp_path: Path) -> None:
    other = SessionClock(
        trading_date=dt.date(2026, 11, 3),
        open_at=dt.datetime(2026, 11, 3, 9, 0, tzinfo=SEOUL),
        close_at=dt.datetime(2026, 11, 3, 15, 30, tzinfo=SEOUL),
        close_confirmation_deadline=dt.datetime(2026, 11, 3, 15, 33, tzinfo=SEOUL),
        provenance="test",
    )
    with pytest.raises(ValueError, match="session clock date"):
        asyncio.run(
            roster_capture.run_research_roster(
                DATE,
                profile=_profile(tmp_path),
                store=CaptureStore(tmp_path / "capture"),
                kiwoom_client=_FakeKiwoom([[_row("005930", "50000")]]),
                session_clock=other,
                listed=frozenset({"005930"}),
                screenable=None,
                shares_by_symbol={"005930": 6e8},
                as_of_panel_date=PREV.date(),
                now_fn=lambda: OBSERVED,
            )
        )


def test_shares_reject_non_prior_date_and_missing_panel(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be before decision_date"):
        roster_capture.load_prev_day_shares(PREV, prev_trading_day=DECISION, path=tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        roster_capture.load_prev_day_shares(DECISION, prev_trading_day=PREV, path=tmp_path / "missing.parquet")


def test_latest_panel_date_before_missing_panel(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        roster_capture.latest_panel_date_before(DECISION, path=tmp_path / "missing.parquet")


def _main_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    kind: SessionKind = SessionKind.STANDARD,
    app_key: str = "key",
    pages: list[list[dict[str, str]]] | None = None,
    rt_cd: str = "0",
    truncated: bool = False,
    stale: bool = False,
) -> list[tuple[str, str, dict[str, Any]]]:
    profile = _profile(tmp_path, COLLECTION_RESEARCH_ROSTER_ENABLED=True)
    calls: list[tuple[str, str, dict[str, Any]]] = []
    if pages is None:
        pages = [[_row("005930", "50000")], [_row("000660", "20000", "+90000")]]

    def _latest(decision_date: pd.Timestamp, **kwargs: Any) -> pd.Timestamp:
        if stale:
            raise ValueError("stale price_history")
        return PREV

    def _session(trading_date: dt.date, overrides: Any = None) -> SessionDay:
        clock = _clock() if kind in (SessionKind.STANDARD, SessionKind.SHIFTED) else None
        return SessionDay(trading_date=trading_date, kind=kind, clock=clock, provenance="test")

    monkeypatch.setattr(roster_capture, "CollectionSettings", lambda: profile)
    monkeypatch.setattr(roster_capture, "resolve_session_day", _session)
    monkeypatch.setattr(roster_capture, "latest_panel_date_before", _latest)
    monkeypatch.setattr(roster_capture, "load_eligible_codes", lambda *a, **k: frozenset({"005930", "000660"}))
    monkeypatch.setattr(roster_capture, "load_security_classification", lambda *a, **k: frozenset({"005930", "000660"}))
    monkeypatch.setattr(roster_capture, "load_prev_day_shares", lambda *a, **k: {"005930": 6e8, "000660": 1e8})
    monkeypatch.setattr(roster_capture.settings, "KIWOOM_APP_KEY", app_key, raising=False)
    monkeypatch.setattr(roster_capture, "KiwoomApiClient", lambda: _FakeKiwoom(pages, rt_cd=rt_cd, truncated=truncated))
    monkeypatch.setattr(roster_capture, "_now", lambda: OBSERVED)
    monkeypatch.setattr(
        roster_capture, "record_run_outcome", lambda job, outcome, **kw: calls.append((job, outcome, kw)) or {}
    )
    return calls


def test_main_success_records_ok_with_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _main_fakes(monkeypatch, tmp_path)

    roster_capture.main(["--date", DATE])

    assert calls == [
        (
            "research_roster",
            RUN_OUTCOME_OK,
            {"run_date": DATE, "metrics": {"members": 2, "dropped": 0, "pages": 2}},
        )
    ]


def test_main_truncated_records_degraded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _main_fakes(monkeypatch, tmp_path, rt_cd="1", truncated=True)

    roster_capture.main(["--date", DATE])

    assert calls[0][:2] == ("research_roster", RUN_OUTCOME_DEGRADED)
    assert calls[0][2]["reason"] == "truncated"


def test_main_ranking_unavailable_records_degraded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _main_fakes(monkeypatch, tmp_path, pages=[[]], rt_cd="1", truncated=False)

    roster_capture.main(["--date", DATE])

    assert calls == [("research_roster", RUN_OUTCOME_DEGRADED, {"run_date": DATE, "reason": "ranking_unavailable"})]


def test_main_stale_panel_records_degraded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _main_fakes(monkeypatch, tmp_path, stale=True)

    roster_capture.main(["--date", DATE])

    assert calls == [("research_roster", RUN_OUTCOME_DEGRADED, {"run_date": DATE, "reason": "stale price_history"})]


@pytest.mark.parametrize(
    ("kind", "app_key", "reason"),
    [
        (SessionKind.CLOSED, "key", "non_trading_day"),
        (SessionKind.UNKNOWN, "key", "session_unknown"),
        (SessionKind.STANDARD, "", "no_kiwoom_credentials"),
    ],
)
def test_main_skips_without_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: SessionKind, app_key: str, reason: str
) -> None:
    calls = _main_fakes(monkeypatch, tmp_path, kind=kind, app_key=app_key)

    roster_capture.main(["--date", DATE])

    assert calls == [("research_roster", RUN_OUTCOME_SKIPPED, {"run_date": DATE, "reason": reason})]
    assert CaptureStore(tmp_path / "capture").read_manifests(DATE) == ()
