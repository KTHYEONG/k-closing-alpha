"""마켓 레짐 백필 진입점의 생존 여부를 검증하는 회귀 가드."""

from __future__ import annotations


def test_market_regime_entrypoint_survives_while_legacy_wrapper_is_gone() -> None:
    from src.backfill import backfill_regime

    # 살아남아야 하는 진입점 (동일 모듈 내부 호출로 실사용 중)
    assert callable(backfill_regime.run_backfill_market_regime_factors)
    assert callable(backfill_regime.main)
    # 이번 단계에서 삭제된 하위 호환 래퍼
    assert not hasattr(backfill_regime, "run_backfill_market_factors")


def test_krx_openapi_settings_replace_hand_rolled_env_parser() -> None:
    from src.backfill import backfill_regime
    from src.config.altdata import AltDataSettings
    from src.settings import Settings

    # Then: the duplicate .env reader is gone.
    assert not hasattr(backfill_regime, "_get_env_value")
    assert not hasattr(backfill_regime, "_get_env_csv")

    # And: every key it used to read is owned by the settings layer.
    for field in ("KRX_OPENAPI_KEY", "KRX_OPENAPI_BASE_URL", "KRX_OPENAPI_BASE_URLS", "KRX_OPENAPI_ENDPOINTS"):
        assert field in AltDataSettings.model_fields, f"{field} must be a settings field"
        assert hasattr(Settings(), field)

    # And: the replacement parser is pure - no I/O, no swallow.
    assert backfill_regime._split_csv("") == []
    assert backfill_regime._split_csv("   ") == []
    assert backfill_regime._split_csv("a,b") == ["a", "b"]
    assert backfill_regime._split_csv(' "a" , \'b\' ,, c ') == ["a", "b", "c"]

    # And: the module no longer reads .env by hand.
    import inspect

    source = inspect.getsource(backfill_regime)
    assert 'BASE_DIR / ".env"' not in source
    assert "except Exception: return default" not in source


def test_krx_openapi_env_precedence_is_preserved(monkeypatch) -> None:
    from src.backfill import backfill_regime

    # Given: a settings object whose CSV fields are empty.
    monkeypatch.setattr(backfill_regime.settings, "KRX_OPENAPI_BASE_URLS", "")
    monkeypatch.setattr(backfill_regime.settings, "KRX_OPENAPI_ENDPOINTS", "")

    # Then: an empty setting parses to an empty list, which is the falsy value
    # the `env or cfg_default` precedence relies on.
    assert backfill_regime._split_csv(backfill_regime.settings.KRX_OPENAPI_BASE_URLS) == []
    assert not backfill_regime._split_csv(backfill_regime.settings.KRX_OPENAPI_ENDPOINTS)

    # And: a populated setting wins and is fully parsed.
    monkeypatch.setattr(
        backfill_regime.settings, "KRX_OPENAPI_BASE_URLS", "https://a.example, https://b.example"
    )
    assert backfill_regime._split_csv(backfill_regime.settings.KRX_OPENAPI_BASE_URLS) == [
        "https://a.example",
        "https://b.example",
    ]


def test_krx_openapi_reads_come_from_settings_at_the_call_sites(monkeypatch) -> None:
    import pandas as pd

    from src.backfill import backfill_regime

    cfg = backfill_regime.MarketFactorFetchConfig()

    # Given: settings supply the base URLs and endpoints, and an EMPTY business-date
    # range so the fetch loop body never runs and no network call is made.
    monkeypatch.setattr(backfill_regime.settings, "KRX_OPENAPI_BASE_URLS", "https://a.example, https://b.example")
    monkeypatch.setattr(backfill_regime.settings, "KRX_OPENAPI_BASE_URL", "")
    monkeypatch.setattr(backfill_regime.settings, "KRX_OPENAPI_ENDPOINTS", "/svc/one, /svc/two")

    empty_start = pd.Timestamp("2026-01-10")
    empty_end = pd.Timestamp("2026-01-09")  # end before start -> zero business days
    out = backfill_regime._fetch_krx_breadth_openapi(
        start=empty_start, end=empty_end, cfg=cfg, auth_key="dummy"
    )
    assert out.empty

    # And: a single base URL setting overrides the list form.
    monkeypatch.setattr(backfill_regime.settings, "KRX_OPENAPI_BASE_URL", "https://only.example")
    out2 = backfill_regime._fetch_krx_breadth_openapi(
        start=empty_start, end=empty_end, cfg=cfg, auth_key="dummy"
    )
    assert out2.empty

    # And: the auth key read reaches settings; a non-empty key routes to the OpenAPI
    # branch, which we stub so the test stays offline.
    sentinel = pd.DataFrame({"date": [pd.Timestamp("2026-01-05")], "adv_count": [1]})
    monkeypatch.setattr(backfill_regime.settings, "KRX_OPENAPI_KEY", "live-key")
    monkeypatch.setattr(
        backfill_regime, "_fetch_krx_breadth_openapi", lambda **kwargs: sentinel
    )
    got = backfill_regime._fetch_krx_breadth(
        start=pd.Timestamp("2026-01-05"), end=pd.Timestamp("2026-01-06"), cfg=cfg
    )
    pd.testing.assert_frame_equal(got, sentinel)
