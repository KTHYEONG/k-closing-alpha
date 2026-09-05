from __future__ import annotations


def test_load_candidate_snapshot_panel_maps_condition_history_to_raw_headers(tmp_path) -> None:
    import pandas as pd

    from src.data.candidate_panel import (
        UNSCORED_SCENARIO_SENTINEL,
        load_candidate_snapshot_panel,
    )

    # Arrange: two clean rows plus one row with a blank 거래대금(억).
    raw = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02", "2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "000660", "035720"],
            "종목명": ["A", "B", "C"],
            "시가": [100.0, 200.0, 300.0],
            "고가": [110.0, 220.0, 330.0],
            "저가": [95.0, 190.0, 285.0],
            "종가": [108.0, 210.0, 320.0],
            "전일종가": [100.0, 200.0, 300.0],
            "등락률": [8.0, 5.0, 6.67],
            "체결강도": [120.0, 90.0, 110.0],
            "시장구분": ["KOSPI", "KOSPI", "KOSDAQ"],
            "시가총액(억)": [1000.0, 2000.0, 3000.0],
            "거래대금(억)": [500.0, 600.0, float("nan")],
            "순위": [1.0, 2.0, 3.0],
            "기관_순매수(억)": [1.0, 2.0, 3.0],
            "외국인_순매수(억)": [4.0, 5.0, 6.0],
            "프로그램_순매수(억)": [7.0, 8.0, 9.0],
            "전체종목수": [3.0, 3.0, 3.0],
            "평균거래대금(억)": [550.0, 550.0, 550.0],
            "KOSPI등락률": [1.0, 1.0, 1.0],
            "KOSDAQ등락률": [0.5, 0.5, 0.5],
            "(v-kospi)": [15.0, 15.0, 15.0],
            "(v-kosdaq)": [18.0, 18.0, 18.0],
            "(거래량)": [1000.0, 2000.0, 3000.0],
        }
    )
    path = tmp_path / "condition_history_cleaned.parquet"
    raw.to_parquet(path)

    # Act
    panel = load_candidate_snapshot_panel(condition_history_path=path, archive_df=pd.DataFrame())

    # Assert
    assert len(panel) == 2
    assert "매수날짜" in panel.columns
    assert "(거래대금, 억)" in panel.columns
    assert "(선정 순위)" in panel.columns
    assert "종목명" not in panel.columns
    assert set(panel["종목코드"]) == {"005930", "000660"}
    assert set(panel["(차트분석)"]) == {UNSCORED_SCENARIO_SENTINEL}


def test_attach_reconstructed_labels_preserves_sell_over_buy_invariant() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.data.candidate_panel import (
        LABEL_SOURCE_COLUMN,
        RECONSTRUCTED_LABEL_SOURCE,
        attach_reconstructed_labels,
    )

    # Arrange: 005930 has a next trading day, 000660 is the last row for its symbol.
    panel = pd.DataFrame(
        {
            "매수날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "000660"],
            "(종가)": [100.0, 200.0],
        }
    )
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-02"]),
            "symbol": ["005930", "005930", "000660"],
            "open": [99.0, 104.0, 199.0],
            "close": [100.0, 106.0, 200.0],
        }
    )

    # Act
    out = attach_reconstructed_labels(panel, price_history, execution_offset_pct=-0.30)

    # Assert: 000660 dropped, 005930 = (104/100 - 1)*100 - 0.30 = 3.70
    assert list(out["종목코드"]) == ["005930"]
    assert out["(수익률, %)"].to_numpy()[0] == pytest.approx(3.70, abs=1e-9)
    buy = out["(매수 가격)"].to_numpy(dtype=float)
    sell = out["(매도 가격)"].to_numpy(dtype=float)
    assert buy[0] == pytest.approx(100.0, abs=1e-9)
    assert (sell / buy - 1.0) * 100.0 == pytest.approx(3.70, rel=1e-9)
    assert set(out[LABEL_SOURCE_COLUMN]) == {RECONSTRUCTED_LABEL_SOURCE}
    assert np.isfinite(sell).all()


def test_measure_execution_offset_pct_rejects_thin_and_split_inconsistent_overlap() -> None:
    import pandas as pd
    import pytest

    from src.data.candidate_panel import measure_execution_offset_pct

    # Arrange: two consistent rows (offset -1.0pp) and one split-mismatched row that must be ignored.
    trade_log = pd.DataFrame(
        {
            "매수날짜": ["2026-03-02", "2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "000660", "035720"],
            "(종가)": [100.0, 200.0, 300.0],
            "(수익률, %)": [1.0, 1.0, 50.0],
            "(매수 가격)": [100.0, 200.0, 300.0],
            "(매도 가격)": [101.0, 202.0, 450.0],
        }
    )
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-03", "2026-03-02", "2026-03-03"]
            ),
            "symbol": ["005930", "005930", "000660", "000660", "035720", "035720"],
            "open": [100.0, 102.0, 200.0, 204.0, 30.0, 60.0],
            "close": [100.0, 102.0, 200.0, 204.0, 30.0, 60.0],
        }
    )

    # Act: theoretical = +2.0pp for both consistent rows, logged = +1.0pp -> median offset -1.0pp
    offset = measure_execution_offset_pct(trade_log, price_history, since="2026-01-01", min_rows=2)

    # Assert
    assert offset == pytest.approx(-1.0, abs=1e-9)
    with pytest.raises(ValueError, match="min_rows"):
        measure_execution_offset_pct(trade_log, price_history, since="2026-01-01", min_rows=99)


def test_build_restored_trade_log_keeps_executed_row_on_duplicate_key(tmp_path) -> None:
    import pandas as pd

    from src.data.candidate_panel import (
        EXECUTED_LABEL_SOURCE,
        LABEL_SOURCE_COLUMN,
        RECONSTRUCTED_LABEL_SOURCE,
        build_restored_trade_log,
    )

    # Arrange: 005930 exists in both sources; 000660 only in the snapshot panel.
    trade_log = pd.DataFrame(
        {
            "매수날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "(종가)": [100.0],
            "(수익률, %)": [7.77],
            "(매수 가격)": [100.0],
            "(매도 가격)": [107.77],
        }
    )
    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "000660"],
            "종가": [100.0, 200.0],
            "거래대금(억)": [500.0, 600.0],
            "등락률": [8.0, 5.0],
            "시장구분": ["KOSPI", "KOSPI"],
        }
    )
    path = tmp_path / "condition_history_cleaned.parquet"
    condition.to_parquet(path)
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-03"]),
            "symbol": ["005930", "005930", "000660", "000660"],
            "open": [100.0, 102.0, 200.0, 204.0],
            "close": [100.0, 102.0, 200.0, 204.0],
        }
    )

    # Act
    out = build_restored_trade_log(
        trade_log,
        price_history,
        condition_history_path=path,
        archive_df=pd.DataFrame(),
        offset_since="2026-01-01",
        offset_min_rows=1,
    )

    # Assert
    assert len(out) == 2
    keyed = out.set_index("종목코드")
    assert keyed.loc["005930", LABEL_SOURCE_COLUMN] == EXECUTED_LABEL_SOURCE
    assert float(keyed.loc["005930", "(수익률, %)"]) == 7.77
    assert keyed.loc["000660", LABEL_SOURCE_COLUMN] == RECONSTRUCTED_LABEL_SOURCE
    prov = out.attrs["panel_restoration"]
    assert prov["restored_rows"] == 1
    assert "execution_offset_pct" in prov

def test_load_candidate_snapshot_panel_uses_defaults_archive_and_theme(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src import settings
    from src.daily import archive
    from src.data import candidate_panel as mod

    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "000660"],
            "종가": [100.0, 200.0],
            "거래대금(억)": [500.0, 600.0],
            "등락률": [8.0, 5.0],
            "시장구분": ["KOSPI", "KOSPI"],
        }
    )
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    condition.to_parquet(history_dir / "condition_history_cleaned.parquet", index=False)
    monkeypatch.setattr(settings, "HISTORY_DIR", history_dir)

    arch = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "종가": [101.0],
            "거래대금": [510.0],
            "등락률": [8.1],
            "시장구분": ["KOSPI"],
            "시나리오": ["거래량 폭증"],
            "테마_섹터": ["반도체"],
        }
    )
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: arch)
    theme = pd.DataFrame({"종목코드": ["000660"], "테마": ["2차전지"]})

    panel = mod.load_candidate_snapshot_panel(theme_df=theme)

    assert len(panel) == 2
    keyed = panel.set_index("종목코드")
    # Archive row wins the duplicate key and keeps its scenario label.
    assert float(keyed.loc["005930", "(종가)"]) == 101.0
    assert keyed.loc["005930", "(차트분석)"] == "거래량 폭증"
    assert keyed.loc["005930", "(테마/섹터)"] == "반도체"
    assert keyed.loc["000660", "(테마/섹터)"] == "2차전지"


def test_load_candidate_snapshot_panel_empty_sources_returns_panel_columns(tmp_path) -> None:
    import pandas as pd

    from src.data import candidate_panel as mod

    panel = mod.load_candidate_snapshot_panel(
        condition_history_path=tmp_path / "missing.parquet",
        archive_df=pd.DataFrame(),
    )

    assert panel.empty
    assert list(panel.columns) == mod.PANEL_COLUMNS


def test_load_candidate_universe_symbols_prunes_and_dedups(tmp_path) -> None:
    import pandas as pd

    from src.data import candidate_panel as mod

    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "ABC"],
            "시장구분": ["KOSPI", "KOSPI"],
            "종가": [100.0, 200.0],
        }
    )
    path = tmp_path / "condition_history_cleaned.parquet"
    condition.to_parquet(path, index=False)
    arch = pd.DataFrame({"스냅샷_날짜": ["2026-03-03"], "종목코드": ["005930"]})

    out = mod.load_candidate_universe_symbols(condition_history_path=path, archive_df=arch)

    assert list(out.columns) == ["symbol", "market"]
    assert set(out["symbol"]) == {"005930"}
    # Archive row (no market) wins the duplicate key.
    assert out.set_index("symbol").loc["005930", "market"] == "UNKNOWN"


def test_load_candidate_universe_symbols_empty_sources_returns_empty_frame(tmp_path) -> None:
    import pandas as pd

    from src.data import candidate_panel as mod

    out = mod.load_candidate_universe_symbols(
        condition_history_path=tmp_path / "missing.parquet",
        archive_df=pd.DataFrame(),
    )

    assert out.empty
    assert list(out.columns) == ["symbol", "market"]


def test_measure_execution_offset_pct_raises_without_required_columns() -> None:
    import pandas as pd
    import pytest

    from src.data.candidate_panel import measure_execution_offset_pct

    with pytest.raises(ValueError, match="requires column"):
        measure_execution_offset_pct(pd.DataFrame({"매수날짜": ["2026-03-02"]}), pd.DataFrame(), min_rows=1)


def test_attach_reconstructed_labels_drops_nonpositive_close() -> None:
    import pandas as pd

    from src.data.candidate_panel import attach_reconstructed_labels

    panel = pd.DataFrame(
        {
            "매수날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "000660"],
            "(종가)": [100.0, 0.0],
        }
    )
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-03"]),
            "symbol": ["005930", "005930", "000660", "000660"],
            "open": [100.0, 102.0, 200.0, 204.0],
            "close": [100.0, 102.0, 200.0, 204.0],
        }
    )

    out = attach_reconstructed_labels(panel, price_history, execution_offset_pct=0.0)

    assert list(out["종목코드"]) == ["005930"]

def test_load_candidate_snapshot_panel_tolerates_empty_file_and_failing_archive(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive
    from src.data import candidate_panel as mod

    # Arrange: an empty condition file plus an archive fetch that raises.
    pd.DataFrame().to_parquet(tmp_path / "empty.parquet")

    def _boom(*args, **kwargs) -> pd.DataFrame:
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(archive, "fetch_archive_snapshot", _boom)

    # Act
    panel = mod.load_candidate_snapshot_panel(condition_history_path=tmp_path / "empty.parquet")

    # Assert
    assert panel.empty
    assert list(panel.columns) == mod.PANEL_COLUMNS


def test_load_candidate_snapshot_panel_empty_archive_snapshot_falls_back_to_condition(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive
    from src.data import candidate_panel as mod

    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "종가": [100.0],
            "거래대금(억)": [500.0],
            "등락률": [8.0],
            "시장구분": ["KOSPI"],
        }
    )
    path = tmp_path / "condition_history_cleaned.parquet"
    condition.to_parquet(path, index=False)
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: pd.DataFrame())

    panel = mod.load_candidate_snapshot_panel(condition_history_path=path)

    assert len(panel) == 1
    assert panel["(테마/섹터)"].to_numpy()[0] == "기타"


def test_load_candidate_snapshot_panel_keeps_archive_theme_without_theme_df(tmp_path) -> None:
    import pandas as pd

    from src.data import candidate_panel as mod

    arch = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "000660"],
            "종가": [101.0, 202.0],
            "거래대금": [510.0, 610.0],
            "등락률": [8.1, 5.2],
            "시장구분": ["KOSPI", "KOSPI"],
            "시나리오": ["거래량 폭증", None],
            "테마_섹터": ["반도체", None],
        }
    )

    panel = mod.load_candidate_snapshot_panel(
        condition_history_path=tmp_path / "missing.parquet", archive_df=arch
    )

    keyed = panel.set_index("종목코드")
    assert keyed.loc["005930", "(테마/섹터)"] == "반도체"
    assert keyed.loc["000660", "(테마/섹터)"] == "기타"


def test_load_candidate_snapshot_panel_all_nonfinite_returns_panel_columns(tmp_path) -> None:
    import pandas as pd

    from src.data import candidate_panel as mod

    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "종가": [100.0],
            "거래대금(억)": [float("nan")],
            "등락률": [8.0],
            "시장구분": ["KOSPI"],
        }
    )
    path = tmp_path / "condition_history_cleaned.parquet"
    condition.to_parquet(path, index=False)

    panel = mod.load_candidate_snapshot_panel(condition_history_path=path, archive_df=pd.DataFrame())

    assert panel.empty
    assert list(panel.columns) == mod.PANEL_COLUMNS


def test_measure_execution_offset_pct_raises_when_residuals_not_finite() -> None:
    import pandas as pd
    import pytest

    from src.data.candidate_panel import measure_execution_offset_pct

    # Arrange: close agrees and next_open exists, but the logged return is blank.
    trade_log = pd.DataFrame(
        {
            "매수날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "(종가)": [100.0],
            "(수익률, %)": [None],
            "(매수 가격)": [100.0],
            "(매도 가격)": [101.0],
        }
    )
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-03-03"]),
            "symbol": ["005930", "005930"],
            "open": [100.0, 102.0],
            "close": [100.0, 102.0],
        }
    )

    with pytest.raises(ValueError, match="finite residuals"):
        measure_execution_offset_pct(trade_log, price_history, since="2026-01-01", min_rows=1)


def test_attach_reconstructed_labels_raises_without_close_column() -> None:
    import pandas as pd
    import pytest

    from src.data.candidate_panel import attach_reconstructed_labels

    with pytest.raises(ValueError, match="requires column"):
        attach_reconstructed_labels(
            pd.DataFrame({"매수날짜": ["2026-03-02"], "종목코드": ["005930"]}),
            pd.DataFrame(),
            execution_offset_pct=0.0,
        )


def test_build_restored_trade_log_defaults_to_fail_closed_min_rows(tmp_path) -> None:
    """build_restored_trade_log must not silently accept a thin offset overlap by default."""
    import pandas as pd
    import pytest

    from src.data.candidate_panel import build_restored_trade_log

    # Arrange: only 1 overlap row -- fewer than the contract's min_rows=500 default.
    trade_log = pd.DataFrame(
        {
            "매수날짜": ["2026-03-02"],
            "종목코드": ["005930"],
            "(종가)": [100.0],
            "(수익률, %)": [7.77],
            "(매수 가격)": [100.0],
            "(매도 가격)": [107.77],
        }
    )
    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02"],
            "종목코드": ["000660"],
            "종가": [200.0],
            "거래대금(억)": [600.0],
            "등락률": [5.0],
            "시장구분": ["KOSPI"],
        }
    )
    path = tmp_path / "condition_history_cleaned.parquet"
    condition.to_parquet(path)
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-03"]),
            "symbol": ["005930", "005930", "000660", "000660"],
            "open": [100.0, 102.0, 200.0, 204.0],
            "close": [100.0, 102.0, 200.0, 204.0],
        }
    )

    # Act / Assert: default offset_min_rows (500) rejects the thin overlap.
    with pytest.raises(ValueError, match="min_rows"):
        build_restored_trade_log(
            trade_log,
            price_history,
            condition_history_path=path,
            archive_df=pd.DataFrame(),
            offset_since="2026-01-01",
        )

    # An explicit override still allows the thin-sample research path.
    out = build_restored_trade_log(
        trade_log,
        price_history,
        condition_history_path=path,
        archive_df=pd.DataFrame(),
        offset_since="2026-01-01",
        offset_min_rows=1,
    )
    assert len(out) == 2


def test_build_restored_trade_log_preserves_duplicate_executed_keys(tmp_path) -> None:
    """A (date, code) pair journaled twice in the raw trade log must not collapse to one row."""
    import pandas as pd

    from src.data.candidate_panel import EXECUTED_LABEL_SOURCE, LABEL_SOURCE_COLUMN, build_restored_trade_log

    # Arrange: 005930 journaled twice on the same day under two different scenarios.
    trade_log = pd.DataFrame(
        {
            "매수날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["005930", "005930"],
            "(종가)": [100.0, 100.0],
            "(수익률, %)": [7.77, -2.5],
            "(매수 가격)": [100.0, 100.0],
            "(매도 가격)": [107.77, 97.5],
        }
    )
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-02", "2026-03-03"]),
            "symbol": ["005930", "005930"],
            "open": [100.0, 102.0],
            "close": [100.0, 102.0],
        }
    )

    # Act
    out = build_restored_trade_log(
        trade_log,
        price_history,
        condition_history_path=tmp_path / "missing.parquet",
        archive_df=pd.DataFrame(),
        offset_since="2026-01-01",
        offset_min_rows=1,
    )

    # Assert: both journaled rows survive untouched.
    assert len(out) == 2
    assert set(out[LABEL_SOURCE_COLUMN]) == {EXECUTED_LABEL_SOURCE}
    assert sorted(out["(수익률, %)"].tolist()) == [-2.5, 7.77]


def test_load_candidate_snapshot_panel_prefers_archived_theme_over_theme_df(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src import settings
    from src.daily import archive
    from src.data import candidate_panel as mod

    # Arrange: no condition_history, one archive row already resolved to '반도체'.
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    monkeypatch.setattr(settings, "HISTORY_DIR", history_dir)
    arch = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-08-10"],
            "종목코드": ["005930"],
            "종가": [100.0],
            "거래대금": [500.0],
            "등락률": [1.0],
            "시장구분": ["KOSPI"],
            "시나리오": ["신고가"],
            "테마_섹터": ["반도체"],
        }
    )
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: arch)
    # A conflicting, stale theme_df entry for the same code.
    theme_df = pd.DataFrame({"종목코드": ["005930"], "테마": ["2차전지"]})

    # Act
    panel = mod.load_candidate_snapshot_panel(theme_df=theme_df)

    # Assert: the archive-resolved theme wins, not theme_df's conflicting entry.
    assert panel.loc[0, "(테마/섹터)"] == "반도체"
    assert panel.loc[0, "(차트분석)"] == "신고가"


def test_load_candidate_snapshot_panel_falls_back_to_theme_df_when_source_has_no_theme(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src import settings
    from src.daily import archive
    from src.data import candidate_panel as mod

    # Arrange: condition_history has no theme column at all.
    condition = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-03-02", "2026-03-02"],
            "종목코드": ["000660", "999999"],
            "종가": [200.0, 50.0],
            "거래대금(억)": [600.0, 10.0],
            "등락률": [5.0, 1.0],
            "시장구분": ["KOSPI", "KOSDAQ"],
        }
    )
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    condition.to_parquet(history_dir / "condition_history_cleaned.parquet", index=False)
    monkeypatch.setattr(settings, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: pd.DataFrame())
    theme_df = pd.DataFrame({"종목코드": ["000660"], "테마": ["2차전지"]})

    # Act
    panel = mod.load_candidate_snapshot_panel(theme_df=theme_df)

    # Assert
    keyed = panel.set_index("종목코드")
    assert keyed.loc["000660", "(테마/섹터)"] == "2차전지"
    assert keyed.loc["999999", "(테마/섹터)"] == "기타"


def test_load_candidate_snapshot_panel_unifies_no_theme_sentinels(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src import settings
    from src.daily import archive
    from src.data import candidate_panel as mod

    # Arrange: archive.py's own fallback sentinel for an unmapped code.
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    monkeypatch.setattr(settings, "HISTORY_DIR", history_dir)
    arch = pd.DataFrame(
        {
            "스냅샷_날짜": ["2026-08-10", "2026-08-10"],
            "종목코드": ["005930", "000660"],
            "종가": [100.0, 200.0],
            "거래대금": [500.0, 600.0],
            "등락률": [1.0, 2.0],
            "시장구분": ["KOSPI", "KOSPI"],
            "시나리오": ["신고가", "거래량 폭증"],
            "테마_섹터": ["테마 없음", "테마 없음"],
        }
    )
    monkeypatch.setattr(archive, "fetch_archive_snapshot", lambda all_rows=False, **kw: arch)
    theme_df = pd.DataFrame({"종목코드": ["005930"], "테마": ["반도체"]})

    # Act
    panel = mod.load_candidate_snapshot_panel(theme_df=theme_df)

    # Assert: theme_df resolves 005930; 000660 falls all the way through to '기타'.
    keyed = panel.set_index("종목코드")
    assert keyed.loc["005930", "(테마/섹터)"] == "반도체"
    assert keyed.loc["000660", "(테마/섹터)"] == "기타"
    assert "테마 없음" not in set(panel["(테마/섹터)"])


def test_check_price_history_freshness_flags_stale_data() -> None:
    import pandas as pd

    from src.data.candidate_panel import check_price_history_freshness

    # Arrange
    stale = pd.DataFrame({"date": pd.to_datetime(["2025-12-30"])})
    fresh = pd.DataFrame({"date": pd.to_datetime(["2026-09-03", "2026-09-04"])})
    as_of = pd.Timestamp("2026-09-05")

    # Act
    stale_result = check_price_history_freshness(stale, as_of=as_of, max_staleness_days=5)
    fresh_result = check_price_history_freshness(fresh, as_of=as_of, max_staleness_days=5)

    # Assert
    assert stale_result["is_stale"] is True
    assert stale_result["staleness_days"] == (as_of - pd.Timestamp("2025-12-30")).days
    assert fresh_result["is_stale"] is False
    assert fresh_result["staleness_days"] == 1


def test_attach_reconstructed_labels_drops_rows_with_close_mismatch() -> None:
    """A panel close disagreeing with price_history's close (corporate action / bad snapshot) must be dropped, not used to compute a wild return."""
    import pandas as pd

    from src.data.candidate_panel import attach_reconstructed_labels

    # Arrange: 005930's panel close (252.0) wildly disagrees with price_history's
    # close (12650.0) for the same day -- e.g. a stale/unadjusted snapshot value.
    # 000660 agrees exactly and must still produce a normal label.
    panel = pd.DataFrame(
        {
            "매수날짜": ["2026-04-29", "2026-04-29"],
            "종목코드": ["900300", "000660"],
            "(종가)": [252.0, 200.0],
        }
    )
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-29", "2026-04-30", "2026-04-29", "2026-04-30"]),
            "symbol": ["900300", "900300", "000660", "000660"],
            "open": [12500.0, 12600.0, 199.0, 204.0],
            "close": [12650.0, 10900.0, 200.0, 204.0],
        }
    )

    # Act
    out = attach_reconstructed_labels(panel, price_history, execution_offset_pct=0.0)

    # Assert: the mismatched row is dropped entirely; no ~4900% return is produced.
    assert list(out["종목코드"]) == ["000660"]
    assert out["(수익률, %)"].abs().max() < 50.0


def test_attach_reconstructed_labels_skips_zero_filled_placeholder_days() -> None:
    """A zero-OHLC placeholder row (halt/holiday gap) must not be treated as the next trading day's open."""
    import pandas as pd
    import pytest

    from src.data.candidate_panel import attach_reconstructed_labels

    # Arrange: the day right after entry is a zero-filled placeholder; the true
    # next trading day (with a real open) comes two rows later.
    panel = pd.DataFrame({"매수날짜": ["2026-05-22"], "종목코드": ["032580"], "(종가)": [8460.0]})
    price_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-22", "2026-05-26", "2026-05-27"]),
            "symbol": ["032580", "032580", "032580"],
            "open": [7030.0, 0.0, 9750.0],
            "close": [8460.0, 8460.0, 6280.0],
        }
    )

    # Act
    out = attach_reconstructed_labels(panel, price_history, execution_offset_pct=0.0)

    # Assert: the zero-open placeholder is skipped -- the label uses 9750.0, not 0.0 (-100%).
    assert len(out) == 1
    expected_ret = (9750.0 / 8460.0 - 1.0) * 100.0
    assert out["(수익률, %)"].to_numpy()[0] == pytest.approx(expected_ret, abs=1e-6)


def test_measure_execution_offset_pct_skips_zero_filled_placeholder_days() -> None:
    """The offset measurement's next_open lookup must also skip zero-OHLC placeholder rows."""
    import pandas as pd
    import pytest

    from src.data.candidate_panel import measure_execution_offset_pct

    # Arrange: 5 real rows so min_rows can be satisfied; one symbol's immediate
    # next day is a zero-filled placeholder, the real next trading day follows.
    trade_log = pd.DataFrame(
        {
            "매수날짜": ["2026-01-05"] * 5,
            "종목코드": ["005930", "000660", "035720", "051910", "005380"],
            "(종가)": [100.0, 100.0, 100.0, 100.0, 100.0],
            "(수익률, %)": [2.0, 2.0, 2.0, 2.0, 2.0],
            "(매수 가격)": [100.0] * 5,
            "(매도 가격)": [102.0] * 5,
        }
    )
    rows = []
    for code in ("000660", "035720", "051910", "005380"):
        rows += [
            {"date": "2026-01-05", "symbol": code, "open": 100.0, "close": 100.0},
            {"date": "2026-01-06", "symbol": code, "open": 102.0, "close": 102.0},
        ]
    # 005930: next calendar day is a zero-filled placeholder; real next trading
    # day (open=102.0, matching the other four) follows after it.
    rows += [
        {"date": "2026-01-05", "symbol": "005930", "open": 99.0, "close": 100.0},
        {"date": "2026-01-06", "symbol": "005930", "open": 0.0, "close": 100.0},
        {"date": "2026-01-07", "symbol": "005930", "open": 102.0, "close": 102.0},
    ]
    price_history = pd.DataFrame(rows)
    price_history["date"] = pd.to_datetime(price_history["date"])

    # Act
    offset = measure_execution_offset_pct(trade_log, price_history, since="2026-01-01", min_rows=5)

    # Assert: theoretical = +2.0pp for all five rows (102/100-1)*100, logged = +2.0pp -> offset 0.0.
    # If the zero-filled row had been used for 005930, its theoretical would be -100pp instead.
    assert offset == pytest.approx(0.0, abs=1e-9)
