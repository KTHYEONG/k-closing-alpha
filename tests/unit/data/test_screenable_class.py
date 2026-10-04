from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.screenable_class import (
    CLASSIFICATION_VERDICT_COLUMNS,
    ScreenableClassProvenance,
    attach_screenable_class,
    build_proxy_agreement_report,
    load_classification_panel,
    proxy_screenable,
)
from src.strategy.contract import SCREENABLE_CLASS_COL


def _class_frame(rows: list[tuple[str, str, bool]], **attrs) -> pd.DataFrame:
    data = {"date": pd.to_datetime([r[0] for r in rows]), "symbol": [r[1] for r in rows],
            "is_screenable": [r[2] for r in rows], **dict(attrs)}
    return pd.DataFrame(data)


def test_proxy_excludes_preferred_and_class_codes() -> None:
    out = proxy_screenable(pd.Series(["005930", "005935", "00088K", "0009K0"]))
    assert out.tolist() == [True, False, False, True]


def test_proxy_excludes_foreign_issuer_codes() -> None:
    out = proxy_screenable(pd.Series(["900140", "950130", "091990"]))
    assert out.tolist() == [False, False, True]


def test_proxy_rejects_malformed_codes() -> None:
    with pytest.raises(ValueError, match="short code"):
        proxy_screenable(pd.Series(["5930"]))
    with pytest.raises(ValueError, match="short code"):
        proxy_screenable(pd.Series(["A005930"]))
    assert proxy_screenable(pd.Series([], dtype=object)).size == 0


def test_real_verdict_uses_previous_trading_day() -> None:
    d1, d2, d3 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
    classification = _class_frame([(d1, "005930", True), (d2, "005930", False)])
    panel = pd.DataFrame({"date": [d2, d3], "symbol": ["005930", "005930"]})
    out, prov = attach_screenable_class(
        panel, classification, market_dates=np.array([d1, d2, d3]))
    assert out[SCREENABLE_CLASS_COL].tolist() == [True, False]
    assert (out["screenable_source"] == "real").all()
    assert prov.n_real == 2 and prov.n_proxy == 0


def test_future_classification_is_invisible() -> None:
    d1, d2, d3 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
    base = _class_frame([(d1, "005930", True), (d2, "005930", False), (d3, "005930", True)])
    flipped = _class_frame([(d1, "005930", True), (d2, "005930", False), (d3, "005930", False)])
    panel = pd.DataFrame({"date": [d1, d2, d3], "symbol": ["005930"] * 3})
    market = np.array([d1, d2, d3])
    out1, _ = attach_screenable_class(panel, base, market_dates=market)
    out2, _ = attach_screenable_class(panel, flipped, market_dates=market)
    assert out1[SCREENABLE_CLASS_COL].tolist() == out2[SCREENABLE_CLASS_COL].tolist()


def test_pre_coverage_rows_use_proxy() -> None:
    d1, d2, d3, d4 = (pd.Timestamp(x) for x in ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    classification = _class_frame([(d3, "005930", True), (d4, "005930", True)])
    panel = pd.DataFrame({"date": [d2, d2], "symbol": ["005935", "005930"]})
    out, prov = attach_screenable_class(
        panel, classification, market_dates=np.array([d1, d2, d3, d4]))
    assert out[SCREENABLE_CLASS_COL].tolist() == [False, True]
    assert (out["screenable_source"] == "proxy").all()
    assert prov.n_proxy == 2


def test_coverage_start_boundary() -> None:
    d1, d2, d3 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
    classification = _class_frame([(d2, "005930", True), (d3, "005930", True)])
    panel = pd.DataFrame({"date": [d3, d2], "symbol": ["005930", "005930"]})
    out, _ = attach_screenable_class(panel, classification, market_dates=np.array([d1, d2, d3]))
    assert out["screenable_source"].tolist() == ["real", "proxy"]


def test_gap_inside_coverage_fails_closed() -> None:
    d = [pd.Timestamp(x) for x in ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]]
    classification = _class_frame([(d[0], "005930", True), (d[2], "005930", True)])
    panel = pd.DataFrame({"date": [d[2]], "symbol": ["005930"]})
    with pytest.raises(ValueError, match="2026-01-05"):
        attach_screenable_class(panel, classification, market_dates=np.array(d))


def test_absent_symbol_on_covered_date_is_non_screenable() -> None:
    d1, d2 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")
    classification = _class_frame([(d1, "005930", True)])
    panel = pd.DataFrame({"date": [d2], "symbol": ["009990"]})
    out, prov = attach_screenable_class(panel, classification, market_dates=np.array([d1, d2]))
    assert out[SCREENABLE_CLASS_COL].tolist() == [False]
    assert prov.n_real_absent == 1


def test_first_market_date_after_coverage_start_fails_closed() -> None:
    d1, d5 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-08")
    classification = _class_frame([(d1, "005930", True)])
    panel = pd.DataFrame({"date": [d5], "symbol": ["005930"]})
    with pytest.raises(ValueError, match="coverage start"):
        attach_screenable_class(panel, classification, market_dates=np.array([d5]))


def test_input_not_mutated_and_shape_preserved() -> None:
    d1, d2, d3 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
    classification = _class_frame([(d1, "005930", True), (d1, "005935", False), (d2, "005930", False)])
    panel = pd.DataFrame({"date": [d2, d3], "symbol": ["005930", "005935"]}, index=[7, 9])
    before = panel.copy(deep=True)
    out, prov = attach_screenable_class(panel, classification, market_dates=np.array([d1, d2, d3]))
    pd.testing.assert_frame_equal(panel, before)
    assert out.index.tolist() == [7, 9]
    assert out[SCREENABLE_CLASS_COL].dtype == bool
    assert not out[SCREENABLE_CLASS_COL].isna().any()
    assert set(out["screenable_source"].unique()) <= {"real", "proxy"}
    assert prov.n_real + prov.n_proxy == prov.n_rows == 2
    assert "n_rows=" in prov.to_log_kv()


def test_unknown_panel_date_rejected() -> None:
    d1, d2 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")
    classification = _class_frame([(d1, "005930", True)])
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-06")], "symbol": ["005930"]})
    with pytest.raises(ValueError, match="market_dates"):
        attach_screenable_class(panel, classification, market_dates=np.array([d1, d2]))


def test_loader_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_classification_panel(tmp_path / "missing.parquet")
    empty = tmp_path / "empty.parquet"
    pd.DataFrame({"date": [], "symbol": [], "is_screenable": []}).to_parquet(empty)
    with pytest.raises(ValueError, match="empty"):
        load_classification_panel(empty)
    dup = tmp_path / "dup.parquet"
    _class_frame([("2026-01-02", "005930", True), ("2026-01-02", "005930", False)]).to_parquet(dup)
    with pytest.raises(ValueError, match="repeats"):
        load_classification_panel(dup)
    nullp = tmp_path / "null.parquet"
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"],
                       "is_screenable": pd.Series([None], dtype=object)})
    df.to_parquet(nullp)
    with pytest.raises(ValueError, match="null"):
        load_classification_panel(nullp)
    ok = tmp_path / "ok.parquet"
    _class_frame([("2026-01-02", "005930", True)]).to_parquet(ok)
    with pytest.raises(ValueError, match="must include"):
        load_classification_panel(ok, columns=["date", "symbol"])


def test_resolver_agrees_with_live_loader(tmp_path: Path) -> None:
    from src.daily.security_classification import load_security_classification

    d1, d2 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")
    frame = _class_frame([(d1, "005930", True), (d1, "005935", False), (d2, "005930", False)])
    path = tmp_path / "security_classification.parquet"
    frame.to_parquet(path)
    loaded = load_classification_panel(path)
    panel = pd.DataFrame({"date": [d2, d2], "symbol": ["005930", "005935"]})
    out, _ = attach_screenable_class(panel, loaded, market_dates=np.array([d1, d2]))
    live = load_security_classification(d2, prev_trading_day=d1, path=path)
    expected = [s in live for s in ["005930", "005935"]]
    assert out[SCREENABLE_CLASS_COL].tolist() == expected


def _agreement_panel() -> pd.DataFrame:
    dates = [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    syms = ["005930", "005935", "123450", "000015"]
    rows = [
        {"date": d, "symbol": s, "chg_ratio": 0.05, "tv_clean": 200.0,
         "mc_clean": 1000.0, "close": 1000.0, "volume": 10.0, "is_ceiling": False}
        for d in dates for s in syms
    ]
    return pd.DataFrame(rows)


def _agreement_classification() -> pd.DataFrame:
    dates = [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    rows = []
    for d in dates:
        rows.append((d, "005930", True, "주권", "보통주", ""))
        rows.append((d, "005935", False, "주권", "우선주", ""))
        rows.append((d, "123450", False, "주권", "보통주", "관리종목"))
        rows.append((d, "000015", True, "주권", "보통주", ""))
    return pd.DataFrame({
        "date": pd.to_datetime([r[0] for r in rows]), "symbol": [r[1] for r in rows],
        "is_screenable": [r[2] for r in rows], "security_group": [r[3] for r in rows],
        "security_kind": [r[4] for r in rows], "section_type": [r[5] for r in rows]})


def test_agreement_report_counts_each_outcome(caplog) -> None:
    import logging

    panel = _agreement_panel()
    classification = _agreement_classification()
    with caplog.at_level(logging.INFO):
        report = build_proxy_agreement_report(panel, classification)
    total = report[(report["scope"] == "panel") & (report["date"].isna())].iloc[0]
    assert total["n_rows"] == 8
    assert total["n_caught"] == 2
    assert total["n_missed"] == 2
    assert total["n_false_exclusion"] == 2
    assert total["recall"] == pytest.approx(0.5)
    assert any("관리종목" in r.message for r in caplog.records)


def test_report_cli_fails_on_any_false_exclusion(tmp_path: Path) -> None:
    from src.data.screenable_class import main as report_main

    price = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "symbol": ["005930", "000015"],
        "open": [1000.0, 1000.0], "high": [1010.0, 1010.0], "low": [990.0, 990.0],
        "close": [1005.0, 1005.0], "prev_close": [1000.0, 1000.0], "volume": [100, 100],
        "market_cap_100m": [1000.0, 1000.0], "trade_value_100m": [300.0, 300.0],
        "market": ["KOSPI", "KOSPI"]})
    ph_path = tmp_path / "price_history.parquet"
    price.to_parquet(ph_path)
    cls = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "symbol": ["005930", "000015"], "is_screenable": [True, True],
        "security_group": ["주권", "주권"], "security_kind": ["보통주", "보통주"],
        "section_type": ["", ""]})
    cls_path = tmp_path / "classification.parquet"
    cls.to_parquet(cls_path)
    out = tmp_path / "report.parquet"
    with pytest.raises(RuntimeError, match="over-excludes"):
        report_main(["--price-history", str(ph_path), "--classification", str(cls_path),
                     "--out", str(out)])
    assert out.exists()


def test_report_refuses_class_filtered_screen() -> None:
    from src.strategy.contract import UniverseSpec

    with pytest.raises(ValueError, match="class-flag-free"):
        build_proxy_agreement_report(
            _agreement_panel(), _agreement_classification(),
            screen=UniverseSpec(exclude_non_screenable_class=True))


def test_provenance_log_kv_field_order() -> None:
    prov = ScreenableClassProvenance(8, 6, 2, 1, 3, 1, "2026-01-02", "2026-01-05")
    assert prov.to_log_kv() == (
        "n_rows=8 n_real=6 n_proxy=2 n_real_absent=1 n_non_screenable_real=3 "
        "n_non_screenable_proxy=1 coverage_start=2026-01-02 coverage_end=2026-01-05")
    assert list(CLASSIFICATION_VERDICT_COLUMNS) == ["date", "symbol", "is_screenable"]


def test_loader_rejects_missing_column(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pandas as pd

    frame = _class_frame([("2026-01-02", "005930", True)])
    path = tmp_path / "cls.parquet"
    frame.to_parquet(path)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame({"date": ["2026-01-02"], "symbol": ["005930"]}))
    with pytest.raises(ValueError, match="missing columns"):
        load_classification_panel(path)


def test_loader_rejects_unparseable_dates(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": ["not-a-date"], "symbol": ["005930"], "is_screenable": [True]})
    path = tmp_path / "baddate.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="unparseable dates"):
        load_classification_panel(path)


def test_attach_rejects_empty_classification() -> None:
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "symbol": ["005930"]})
    with pytest.raises(ValueError, match="empty classification"):
        attach_screenable_class(panel, pd.DataFrame(), market_dates=np.array([pd.Timestamp("2026-01-02")]))


def test_attach_rejects_empty_or_unsorted_calendar() -> None:
    classification = _class_frame([("2026-01-02", "005930", True)])
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "symbol": ["005930"]})
    with pytest.raises(ValueError, match="empty market_dates"):
        attach_screenable_class(panel, classification, market_dates=np.array([]))
    d1, d2 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")
    with pytest.raises(ValueError, match="sorted ascending"):
        attach_screenable_class(panel, classification, market_dates=np.array([d2, d1]))
    bad_panel = pd.DataFrame({"date": ["not-a-date"], "symbol": ["005930"]})
    with pytest.raises(ValueError, match="unparseable panel dates"):
        attach_screenable_class(bad_panel, classification, market_dates=np.array([d1, d2]))


def test_report_rejects_disjoint_dates() -> None:
    panel = _agreement_panel()
    classification = _agreement_classification()
    classification["date"] = pd.to_datetime(["2020-01-02"] * len(classification))
    with pytest.raises(ValueError, match="shares no date"):
        build_proxy_agreement_report(panel, classification)


def test_report_without_attributes_logs_empty_breakdown(caplog) -> None:
    import logging

    panel = _agreement_panel()
    classification = _agreement_classification().drop(
        columns=["security_group", "security_kind", "section_type"])
    with caplog.at_level(logging.INFO):
        report = build_proxy_agreement_report(panel, classification)
    total = report[(report["scope"] == "panel") & (report["date"].isna())].iloc[0]
    assert total["n_missed"] == 2
    assert any("stage=screenable_proxy_missed_breakdown" in r.message for r in caplog.records)


def test_report_cli_warns_on_coverage_gap(tmp_path: Path, caplog) -> None:
    import logging

    from src.data.screenable_class import main as report_main

    price = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
        "symbol": ["005930", "005930"],
        "open": [1000.0, 1000.0], "high": [1010.0, 1010.0], "low": [990.0, 990.0],
        "close": [1005.0, 1005.0], "prev_close": [1000.0, 1000.0], "volume": [100, 100],
        "market_cap_100m": [1000.0, 1000.0], "trade_value_100m": [300.0, 300.0],
        "market": ["KOSPI", "KOSPI"]})
    ph_path = tmp_path / "price_history.parquet"
    price.to_parquet(ph_path)
    cls = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02"]),
        "symbol": ["005930"], "is_screenable": [True],
        "security_group": ["주권"], "security_kind": ["보통주"], "section_type": [""]})
    cls_path = tmp_path / "classification.parquet"
    cls.to_parquet(cls_path)
    out = tmp_path / "report.parquet"
    with caplog.at_level(logging.WARNING):
        report_main(["--price-history", str(ph_path), "--classification", str(cls_path),
                     "--out", str(out)])
    assert out.exists()
    assert any("n_coverage_gap_dates" in r.message for r in caplog.records)
