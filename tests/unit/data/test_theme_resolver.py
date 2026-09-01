def test_resolve_stock_theme_from_cache() -> None:
    from src.data.theme_resolver import resolve_stock_theme_and_market

    custom_map = {"005930": "반도체"}
    theme, market = resolve_stock_theme_and_market(
        code="005930",
        name="삼성전자",
        kis_market="KOSPI",
        custom_theme_map=custom_map,
    )
    assert theme == "반도체"
    assert market == "KOSPI"


def test_resolve_stock_theme_from_name_rules() -> None:
    from src.data.theme_resolver import resolve_stock_theme_and_market

    theme_spac, market_spac = resolve_stock_theme_and_market(
        code="475150",
        name="하나29호스팩",
        kis_market="KOSDAQ",
        custom_theme_map={},
    )
    assert theme_spac == "기타"
    assert market_spac == "KOSDAQ"

    theme_hold, market_hold = resolve_stock_theme_and_market(
        code="005385",
        name="현대차우지주",
        kis_market="KOSPI",
        custom_theme_map={},
    )
    assert theme_hold == "지주사"
    assert market_hold == "KOSPI"


def test_resolve_stock_theme_fallback_and_market(monkeypatch) -> None:
    from src.data import theme_resolver

    monkeypatch.setattr(theme_resolver, "_fetch_naver_stock_metadata", lambda code, timeout: {})

    theme, market = theme_resolver.resolve_stock_theme_and_market(
        code="999999",
        name="알수없는미확인종목",
        kis_market="KSQ150",
        kis_upjong="기타",
        custom_theme_map={},
    )
    assert theme == "기타"
    assert market == "KOSDAQ"


import pandas as pd
def test_batch_resolve_missing_themes_updates_cache(monkeypatch, tmp_path) -> None:
    from src.data import theme_resolver
    from src import settings

    dummy_parquet = tmp_path / "theme.parquet"
    monkeypatch.setattr(settings, "THEME_PARQUET_PATH", dummy_parquet)
    monkeypatch.setattr(theme_resolver, "_fetch_naver_stock_metadata", lambda code, timeout: {"upjong": "반도체와반도체장비", "summary": "반도체 설계 전문"})

    missing_stocks = [
        {"종목코드": "452430", "종목명": "아이엠티", "시장구분": "KOSDAQ", "업종": ""},
        {"종목코드": "475150", "종목명": "하나스팩", "시장구분": "KOSDAQ", "업종": ""},
    ]

    resolved = theme_resolver.batch_resolve_missing_themes(missing_stocks, sync_gsheet=False)
    assert len(resolved) == 2
    assert resolved[0]["테마"] == "반도체"
    assert resolved[1]["테마"] == "기타"

    assert dummy_parquet.exists()
    saved_df = pd.read_parquet(dummy_parquet)
    assert "452430" in saved_df["종목코드"].values
