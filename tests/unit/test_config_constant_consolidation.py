from __future__ import annotations


def test_live_entrypoints_import_after_constant_consolidation() -> None:
    import importlib

    entrypoints = [
        "src.daily.collect",
        "src.daily.paper_trade",
        "src.daily.predict",
        "src.daily.archive_intraday",
        "src.ml.retrain",
        "src.ml.costaware_topk",
        "src.ml.topk_ranker_research",
        "src.backfill.backfill_altdata",
        "src.backfill.kis_flow_backfill",
        "src.backfill.intraday.backfill_minute_history",
    ]

    # Then: no import cycle was introduced by the new edges into strategy.contract.
    for name in entrypoints:
        assert importlib.import_module(name) is not None, f"{name} failed to import"


def test_trading_paths_do_not_import_collect() -> None:
    import ast
    from pathlib import Path

    # Given: 거래 경로 4 모듈
    paths = [
        "src/daily/paper_trade.py",
        "src/daily/finalize_close.py",
        "src/daily/auction_capture.py",
        "src/daily/archive_intraday.py",
    ]

    # Then: 어떤 깊이에서도 src.daily.collect 참조 없음
    for rel in paths:
        tree = ast.parse(Path(rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("src.daily.collect" not in a.name for a in node.names), rel
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "src.daily.collect", rel
                assert node.module != "src.daily" or all(a.name != "collect" for a in node.names), rel


def test_market_div_codes_are_not_literals() -> None:
    import ast
    from pathlib import Path

    from src.config import market_session
    from src.daily import aftermarket_book

    # Given / Then: 호출 키워드는 상수, 리터럴 없음
    for rel in ("src/daily/auction_capture.py", "src/daily/universe_scan.py", "src/daily/aftermarket_book.py"):
        tree = ast.parse(Path(rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "market_div_code":
                        assert not isinstance(kw.value, ast.Constant), f"{rel}:{node.lineno}"

    # And: venue 매핑은 market_session 상수와 동일
    assert aftermarket_book._VENUE_DIV_CODE == {
        "KRX": market_session.KRX_CLOSE_MARKET_DIV_CODE,
        "NXT": market_session.NXT_MARKET_DIV_CODE,
    }


def test_prev_trading_day_lookback_single_definition() -> None:
    import ast
    from pathlib import Path

    from src.config import market_session
    from src.ml import topk_history_features

    # Given / Then: 값 15, 단일 정의
    assert market_session.MAX_PREV_TRADING_DAY_LOOKBACK == 15
    assert topk_history_features.MAX_PREV_TRADING_DAY_LOOKBACK == 15
    tree = ast.parse(Path("src/ml/topk_history_features.py").read_text())
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
    assert "MAX_PREV_TRADING_DAY_LOOKBACK" not in assigned
