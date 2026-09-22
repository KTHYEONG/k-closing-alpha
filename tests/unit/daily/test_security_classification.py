"""Security classification capture invariant tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.backfill.altdata.config import AltDataFetchConfig
from src.daily.security_classification import (
    SECURITY_CLASSIFICATION_COLUMNS,
    fetch_security_classification,
    load_security_classification,
    merge_security_classification,
    normalize_security_classification,
    run_security_classification_ingest,
)


def _raw(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(code="005930", group="주권", kind="보통주", section=""):
    return {"ISU_SRT_CD": code, "SECUGRP_NM": group, "KIND_STKCERT_TP_NM": kind, "SECT_TP_NM": section}


def _cfg() -> AltDataFetchConfig:
    return AltDataFetchConfig(
        start=pd.Timestamp("2026-09-18"),
        end=pd.Timestamp("2026-09-19"),
        out_dir=Path("."),
        krx_api_key="test-key",
    )


def test_normalize_empty_raw_yields_empty_schema():
    out = normalize_security_classification(pd.DataFrame(), pd.Timestamp("2026-09-18"))
    assert list(out.columns) == list(SECURITY_CLASSIFICATION_COLUMNS)
    assert out.empty


def test_normalize_missing_required_column_fails_closed():
    raw = _raw([_row()])
    raw = raw.drop(columns=["KIND_STKCERT_TP_NM"])
    with pytest.raises(ValueError, match="KIND_STKCERT_TP_NM"):
        normalize_security_classification(raw, pd.Timestamp("2026-09-18"))


def test_normalize_plain_common_stock_is_screenable():
    out = normalize_security_classification(_raw([_row()]), pd.Timestamp("2026-09-18"))
    assert bool(out.loc[0, "is_screenable"]) is True


def test_normalize_preferred_excluded_via_kind_field():
    out = normalize_security_classification(
        _raw([_row(code="005931", kind="구형우선주")]), pd.Timestamp("2026-09-18")
    )
    assert bool(out.loc[0, "is_screenable"]) is False


def test_normalize_kind_stock_excluded():
    out = normalize_security_classification(
        _raw([_row(code="37550K", kind="종류주권")]), pd.Timestamp("2026-09-18")
    )
    assert bool(out.loc[0, "is_screenable"]) is False


def test_normalize_administrative_issue_excluded():
    out = normalize_security_classification(
        _raw([_row(section="관리종목(소속부없음)")]), pd.Timestamp("2026-09-18")
    )
    assert bool(out.loc[0, "is_screenable"]) is False


def test_normalize_investment_alert_excluded():
    out = normalize_security_classification(
        _raw([_row(section="투자주의환기종목(소속부없음)")]), pd.Timestamp("2026-09-18")
    )
    assert bool(out.loc[0, "is_screenable"]) is False


def test_normalize_spac_excluded():
    out = normalize_security_classification(
        _raw([_row(section="SPAC(소속부없음)")]), pd.Timestamp("2026-09-18")
    )
    assert bool(out.loc[0, "is_screenable"]) is False


def test_normalize_non_stock_group_excluded():
    out = normalize_security_classification(
        _raw([_row(group="부동산투자회사")]), pd.Timestamp("2026-09-18")
    )
    assert bool(out.loc[0, "is_screenable"]) is False


def test_normalize_duplicate_symbol_fails_closed():
    raw = _raw([_row(code="005930"), _row(code="005930")])
    with pytest.raises(ValueError, match="1"):
        normalize_security_classification(raw, pd.Timestamp("2026-09-18"))


def test_fetch_partial_publication_fails_closed():
    kospi = _raw([_row(code="005930")])
    with (
        patch(
            "src.daily.security_classification.fetch_krx_openapi_day_strict",
            side_effect=[kospi, pd.DataFrame()],
        ),
        pytest.raises(RuntimeError, match="partial"),
    ):
        fetch_security_classification(pd.Timestamp("2026-09-18"), _cfg())


def test_fetch_both_markets_empty_is_clean_holiday():
    with patch(
        "src.daily.security_classification.fetch_krx_openapi_day_strict",
        return_value=pd.DataFrame(),
    ):
        out = fetch_security_classification(pd.Timestamp("2026-09-18"), _cfg())
    assert list(out.columns) == list(SECURITY_CLASSIFICATION_COLUMNS)
    assert out.empty


def test_merge_recapture_upserts():
    day = pd.Timestamp("2026-09-18")
    existing = pd.DataFrame([{
        "date": day, "symbol": "X", "security_group": "주권",
        "security_kind": "보통주", "section_type": "", "is_screenable": True,
    }])
    new_rows = pd.DataFrame([{
        "date": day, "symbol": "X", "security_group": "주권",
        "security_kind": "보통주", "section_type": "관리종목(소속부없음)", "is_screenable": False,
    }])
    merged = merge_security_classification(existing, new_rows)
    assert len(merged) == 1
    assert merged.loc[0, "section_type"] == "관리종목(소속부없음)"


def test_run_ingest_confirmed_date_without_classification_fails_closed(tmp_path):
    target = tmp_path / "security_classification.parquet"
    with (
        patch(
            "src.daily.security_classification.fetch_security_classification",
            return_value=pd.DataFrame(columns=list(SECURITY_CLASSIFICATION_COLUMNS)),
        ),
        pytest.raises(RuntimeError, match="2026-09-18"),
    ):
        run_security_classification_ingest(
            [pd.Timestamp("2026-09-18")], path=target, krx_cfg=_cfg()
        )
    assert not target.exists()


def test_run_ingest_empty_trade_dates_noop(tmp_path):
    target = tmp_path / "security_classification.parquet"
    out = run_security_classification_ingest([], path=target, krx_cfg=_cfg())
    assert out == {"n_fetched": 0, "n_written": 0, "n_admin": 0, "n_alert": 0, "n_spac": 0, "n_non_common": 0}
    assert not target.exists()


def _panel_frame() -> pd.DataFrame:
    day = pd.Timestamp("2026-09-17")
    return pd.DataFrame([
        {"date": day, "symbol": "A", "security_group": "주권", "security_kind": "보통주",
         "section_type": "", "is_screenable": True},
        {"date": day, "symbol": "B", "security_group": "주권", "security_kind": "보통주",
         "section_type": "관리종목(소속부없음)", "is_screenable": False},
    ])


def test_load_rejects_non_past_prev_trading_day(tmp_path):
    target = tmp_path / "security_classification.parquet"
    _panel_frame().to_parquet(target, index=False)
    with pytest.raises(ValueError, match="must be before"):
        load_security_classification(
            pd.Timestamp("2026-09-18"), prev_trading_day=pd.Timestamp("2026-09-18"), path=target
        )


def test_load_fails_closed_on_coverage_gap(tmp_path):
    target = tmp_path / "security_classification.parquet"
    _panel_frame().to_parquet(target, index=False)
    with pytest.raises(ValueError, match="no rows"):
        load_security_classification(
            pd.Timestamp("2026-09-18"), prev_trading_day=pd.Timestamp("2026-09-16"), path=target
        )


def test_load_returns_only_screenable_set(tmp_path):
    target = tmp_path / "security_classification.parquet"
    _panel_frame().to_parquet(target, index=False)
    out = load_security_classification(
        pd.Timestamp("2026-09-18"), prev_trading_day=pd.Timestamp("2026-09-17"), path=target
    )
    assert out == frozenset({"A"})


def test_fetch_both_markets_concatenated():
    kospi = _raw([_row(code="005930")])
    kosdaq = _raw([_row(code="035720")])
    with patch(
        "src.daily.security_classification.fetch_krx_openapi_day_strict",
        side_effect=[kospi, kosdaq],
    ):
        out = fetch_security_classification(pd.Timestamp("2026-09-18"), _cfg())
    assert sorted(out["symbol"]) == ["005930", "035720"]
    assert bool(out["is_screenable"].all()) is True


def test_merge_empty_branches_preserve_without_mutation():
    full = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-18"), "symbol": "A", "security_group": "주권",
        "security_kind": "보통주", "section_type": "", "is_screenable": True,
    }])
    empty = pd.DataFrame(columns=["date", "symbol", "security_group", "security_kind", "section_type", "is_screenable"])
    out_new = merge_security_classification(empty, full)
    assert out_new.equals(full)
    assert out_new is not full
    out_existing = merge_security_classification(full, empty)
    assert out_existing.equals(full)
    assert out_existing is not full


def test_run_ingest_success_writes_panel(tmp_path):
    target = tmp_path / "nested" / "security_classification.parquet"
    day = pd.Timestamp("2026-09-18")
    batch = pd.DataFrame([
        {"date": day, "symbol": "A", "security_group": "주권", "security_kind": "보통주",
         "section_type": "", "is_screenable": True},
        {"date": day, "symbol": "B", "security_group": "주권", "security_kind": "구형우선주",
         "section_type": "관리종목(소속부없음)", "is_screenable": False},
        {"date": day, "symbol": "C", "security_group": "주권", "security_kind": "보통주",
         "section_type": "SPAC(소속부없음)", "is_screenable": False},
        {"date": day, "symbol": "D", "security_group": "주권", "security_kind": "보통주",
         "section_type": "투자주의환기종목(소속부없음)", "is_screenable": False},
    ])
    with patch(
        "src.daily.security_classification.fetch_security_classification",
        return_value=batch,
    ):
        out = run_security_classification_ingest([day], path=target, krx_cfg=_cfg())
    assert out == {"n_fetched": 4, "n_written": 4, "n_admin": 1, "n_alert": 1, "n_spac": 1, "n_non_common": 1}
    assert target.exists()
    stored = pd.read_parquet(target)
    assert len(stored) == 4
    with patch(
        "src.daily.security_classification.fetch_security_classification",
        return_value=batch,
    ):
        rerun = run_security_classification_ingest([day], path=target, krx_cfg=_cfg())
    assert rerun["n_fetched"] == 4
    assert len(pd.read_parquet(target)) == 4


def test_load_missing_panel_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="security_classification"):
        load_security_classification(
            pd.Timestamp("2026-09-18"),
            prev_trading_day=pd.Timestamp("2026-09-17"),
            path=tmp_path / "missing.parquet",
        )


def test_price_ingest_main_triggers_classification_capture(monkeypatch):
    import src.daily.price_ingest as price_ingest

    seen: dict = {}

    async def fake_ingest(**_kwargs):
        seen["ingest"] = True

        class _Report:
            ingested_dates = ["2026-09-18"]

        return _Report()

    def fake_classification(dates):
        seen["dates"] = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
        return {"n_fetched": 1, "n_written": 1, "n_admin": 0, "n_alert": 0, "n_spac": 0, "n_non_common": 0}

    import sys

    import src.daily.security_classification as classification

    monkeypatch.setattr(price_ingest, "run_price_ingest", fake_ingest)
    monkeypatch.setattr(classification, "run_security_classification_ingest", fake_classification)
    monkeypatch.setitem(sys.modules, "src.daily.security_classification", classification)
    monkeypatch.setattr("src.strategy.growth_shadow.run_growth_shadow", lambda: None)
    monkeypatch.setattr("src.strategy.t1_attribution.run_t1_attribution", lambda: None)
    price_ingest.main()
    assert seen["dates"] == ["2026-09-18"]
