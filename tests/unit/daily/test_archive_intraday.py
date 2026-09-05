from __future__ import annotations



def test_archive_target_codes_unions_previous_session_watchlist(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    # Given: 2026-09-04 이전 아카이브 영업일은 금요일이 아닌 2026-09-01 (휴장 가정)
    frames = {
        "2026-09-04": pd.DataFrame({"종목코드": ["005930", "000660"]}),
        "2026-09-01": pd.DataFrame({"종목코드": ["009900", "005930"]}),
    }
    all_rows = pd.DataFrame({"스냅샷_날짜": ["2026-09-01", "2026-09-04"], "종목코드": ["009900", "005930"]})

    def fake_fetch(snapshot_date=None, month=None, all_rows_flag=False, **kwargs):
        if kwargs.get("all_rows") or all_rows_flag:
            return all_rows
        return frames.get(snapshot_date, pd.DataFrame())

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", fake_fetch)

    # When
    codes = archive_intraday._archive_target_codes("2026-09-04")

    # Then
    assert set(codes) == {"005930", "000660", "009900"}
    assert len(codes) == 3
    assert archive_intraday.resolve_previous_archive_date("2026-09-04") == "2026-09-01"
    assert archive_intraday.resolve_previous_archive_date("2026-09-01") is None


def test_today_watchlist_codes_returns_empty_on_fetch_failure(monkeypatch) -> None:
    from src.daily import archive_intraday

    def _raise(*a, **kw):
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", _raise)

    assert archive_intraday._today_watchlist_codes("2026-09-04") == []


def test_today_watchlist_codes_returns_empty_when_column_missing(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", lambda **kw: pd.DataFrame())

    assert archive_intraday._today_watchlist_codes("2026-09-04") == []


def test_resolve_previous_archive_date_returns_none_on_fetch_failure(monkeypatch) -> None:
    from src.daily import archive_intraday

    def _raise(*a, **kw):
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", _raise)

    assert archive_intraday.resolve_previous_archive_date("2026-09-04") is None


def test_resolve_previous_archive_date_returns_none_when_column_missing(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", lambda **kw: pd.DataFrame())

    assert archive_intraday.resolve_previous_archive_date("2026-09-04") is None


def test_archive_intraday_main_invokes_run_intraday_archive(monkeypatch) -> None:
    from src.daily import archive_intraday

    captured: dict = {}

    def _fake_run(snapshot_date=None):
        captured["snapshot_date"] = snapshot_date
        return (1, 2, 3)

    monkeypatch.setattr(archive_intraday, "run_intraday_archive", _fake_run)
    monkeypatch.setattr("sys.argv", ["archive_intraday", "2026-09-04"])

    archive_intraday.main()

    assert captured == {"snapshot_date": "2026-09-04"}

