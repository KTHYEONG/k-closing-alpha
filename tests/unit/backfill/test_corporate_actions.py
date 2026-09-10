def test_detect_price_limit_breach_flags_extreme_drop_and_rise() -> None:
    import pandas as pd

    from src.backfill.price.corporate_actions import detect_price_limit_breach

    panel = pd.DataFrame({
        "symbol": ["000001", "000001", "000002", "000002", "000003", "000003"],
        "date": pd.to_datetime([
            "2026-01-01", "2026-01-02",  # 000001: 정상 등락(+5%)
            "2026-01-01", "2026-01-02",  # 000002: 분할 의심(-80%)
            "2026-01-01", "2026-01-02",  # 000003: 정상 등락(-10%)
        ]),
        "close": [10_000, 10_500, 50_000, 10_000, 20_000, 18_000],
    })

    # When
    breached = detect_price_limit_breach(panel)

    # Then: 가격제한을 벗어난 종목만 잡힌다
    assert breached == ["000002"]


def test_detect_price_limit_breach_ignores_first_row_and_empty_input() -> None:
    import pandas as pd

    from src.backfill.price.corporate_actions import detect_price_limit_breach

    # Given: 종목당 단 1행(shift 결과가 전부 NaN)
    single_row = pd.DataFrame({"symbol": ["000001"], "date": pd.to_datetime(["2026-01-01"]), "close": [10_000]})
    assert detect_price_limit_breach(single_row) == []

    # And: 완전히 빈 프레임
    assert detect_price_limit_breach(pd.DataFrame()) == []

    # And: 필수 컬럼(close) 없음
    no_close = pd.DataFrame({"symbol": ["000001", "000001"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    assert detect_price_limit_breach(no_close) == []


def test_heal_corporate_action_breach_replaces_only_flagged_symbol(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import corporate_actions
    from src.backfill.price.config import FetchConfig

    merged = pd.DataFrame({
        "symbol": ["000001", "000001", "000002", "000002"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
        "close": [10_000, 10_500, 50_000, 10_000],  # 000002가 -80% 이탈
    })
    fetch_cfg = FetchConfig(fixed_start_date=pd.Timestamp("2016-01-01"))

    refetched = pd.DataFrame({
        "symbol": ["000002", "000002"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "close": [1_000, 200],  # pykrx 재조회로 사후조정된 값
    })

    def _fake_fetch_one_symbol(symbol, start, end, market_hint, fetch_cfg):
        assert symbol == "000002"
        return refetched

    import src.backfill.price.runner as runner_mod
    monkeypatch.setattr(runner_mod, "fetch_one_symbol", _fake_fetch_one_symbol)

    # When
    out = corporate_actions.heal_corporate_action_breach(merged, fetch_cfg, {"000002": "KOSPI"})

    # Then: 000001은 그대로, 000002는 재조회 값으로 완전 치환
    sym1 = out[out["symbol"] == "000001"].sort_values("date")
    assert list(sym1["close"]) == [10_000, 10_500]
    sym2 = out[out["symbol"] == "000002"].sort_values("date")
    assert list(sym2["close"]) == [1_000, 200]


def test_heal_corporate_action_breach_skips_refetch_when_no_breach(monkeypatch) -> None:
    import pandas as pd

    from src.backfill.price import corporate_actions
    from src.backfill.price.config import FetchConfig

    merged = pd.DataFrame({
        "symbol": ["000001", "000001"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "close": [10_000, 10_500],
    })
    fetch_cfg = FetchConfig()

    calls = {"n": 0}

    def _counting(*a, **k):
        calls["n"] += 1
        return pd.DataFrame()

    import src.backfill.price.runner as runner_mod
    monkeypatch.setattr(runner_mod, "fetch_one_symbol", _counting)

    # When
    out = corporate_actions.heal_corporate_action_breach(merged, fetch_cfg, {})

    # Then: 재조회 호출 0회, 원본 그대로
    assert calls["n"] == 0
    assert len(out) == 2


def test_heal_corporate_action_breach_keeps_original_rows_on_refetch_failure(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.backfill.price import corporate_actions
    from src.backfill.price.config import FetchConfig

    merged = pd.DataFrame({
        "symbol": ["000002", "000002"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "close": [50_000, 10_000],  # -80% 이탈
    })
    fetch_cfg = FetchConfig()

    def _boom(symbol, start, end, market_hint, fetch_cfg):
        raise RuntimeError("pykrx unavailable")

    import src.backfill.price.runner as runner_mod
    monkeypatch.setattr(runner_mod, "fetch_one_symbol", _boom)

    # When: 재조회가 실패해도 예외가 전파되지 않는다
    with caplog.at_level(logging.WARNING):
        out = corporate_actions.heal_corporate_action_breach(merged, fetch_cfg, {"000002": "KOSPI"})

    # Then: 원본 행이 그대로 남고 실패가 로그로 드러난다
    assert list(out.sort_values("date")["close"]) == [50_000, 10_000]
    assert any("000002" in r.message and "REFETCH_FAILED" in r.message for r in caplog.records)


def test_heal_corporate_action_breach_keeps_original_rows_on_empty_refetch(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.backfill.price import corporate_actions
    from src.backfill.price.config import FetchConfig

    merged = pd.DataFrame({
        "symbol": ["000002", "000002"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "close": [50_000, 10_000],  # -80% 이탈
    })
    fetch_cfg = FetchConfig()

    # Given: 재조회는 예외 없이 성공했지만 빈 프레임(해당 구간 데이터 없음)
    def _empty(symbol, start, end, market_hint, fetch_cfg):
        return pd.DataFrame()

    import src.backfill.price.runner as runner_mod
    monkeypatch.setattr(runner_mod, "fetch_one_symbol", _empty)

    # When
    with caplog.at_level(logging.WARNING):
        out = corporate_actions.heal_corporate_action_breach(merged, fetch_cfg, {"000002": "KOSPI"})

    # Then: 예외 경로와 동일하게 원본 유지 + REFETCH_FAILED 로그
    assert list(out.sort_values("date")["close"]) == [50_000, 10_000]
    assert any("000002" in r.message and "REFETCH_FAILED" in r.message for r in caplog.records)
