"""Regression guards for the dead-code subtraction phase."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _raw_trade_log_for_digest(n_dates: int = 30, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=n_dates):
        for j in range(per_day):
            e = rng.normal()
            rows.append({
                "매수날짜": d.strftime("%Y-%m-%d"), "종목코드": f"{j:06d}",
                "(시가)": "10000", "(고가)": "10400", "(저가)": "9800", "(종가)": "10200", "(전일종가)": "10000",
                "(시가총액, 억)": "5000", "(거래대금, 억)": "300", "(등락률)": f"{2 + e:.2f}",
                "(선정 순위)": str(j + 1), "(기관_순매수)": f"{e*100:.0f}", "(외국인_순매수)": f"{e*80:.0f}",
                "(프로그램_순매수)": f"{e*50:.0f}", "(체결강도)": "120", "(시장구분)": "KOSPI",
                "(총 종목 수)": str(per_day), "(평균 거래대금)": "250", "(kospi, %)": "0.3", "(kosdaq, %)": "0.1",
                "v_kospi": "18", "v_kosdaq": "20", "(거래량)": "100000", "(테마/섹터)": "반도체",
                "(차트분석)": "거래량 폭증", "(매수 가격)": "10200",
                "(매도 가격)": f"{10200*(1+0.01*e):.0f}", "(수익률, %)": f"{e:.2f}",
            })
    return pd.DataFrame(rows)


def test_unreachable_modules_are_deleted() -> None:
    import importlib
    from pathlib import Path

    import pytest

    # Given: the four modules proven unreachable from every live entrypoint.
    deleted = {
        "src/data/theme_resolver.py": "src.data.theme_resolver",
        "src/execution/passive_fill.py": "src.execution.passive_fill",
        "src/ml/expected_value.py": "src.ml.expected_value",
        "src/backfill/price/krx_openapi.py": "src.backfill.price.krx_openapi",
    }

    # Then: each file is gone and its module no longer imports.
    for rel, mod in deleted.items():
        assert not Path(rel).exists(), f"{rel} should be deleted"
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)

    # And: the live look-alikes that were previously mis-flagged still exist.
    assert Path("src/backfill/altdata/krx_api.py").exists()
    assert Path("src/api/kiwoom/client.py").exists()
    assert importlib.import_module("src.api.kiwoom.client").KiwoomApiClient is not None


def test_dead_symbols_are_absent_but_live_siblings_remain() -> None:
    import importlib

    # Given: the 12 symbols proven dead by cross-module refs AND zero self-use.
    dead = {
        "src.ml.metrics": [
            "extract_year", "max_drawdown", "group_relevance",
            "ndcg_at_k", "top_k_return", "yearly_breakdown",
        ],
        "src.ml.research.v3_engine": [
            "execute_walk_forward_oof", "evaluate_all_pipelines", "evaluate_decision_gates",
        ],
        "src.ml.bundle": ["save_bundle"],
        "src.utils.display": ["apply_label_encodings"],
    }
    # And: symbols that look dead by cross-module grep but are alive via self-use.
    alive = {
        "src.ml.oof": ["sample_weight_for_fold"],
        "src.ml.costaware_topk": ["split_regime_masks"],
        "src.utils.display": ["get_decision_color"],
        "src.ml.robust_eval": ["BootstrapDelta"],
    }

    # Then
    for mod_name, symbols in dead.items():
        mod = importlib.import_module(mod_name)
        for sym in symbols:
            assert not hasattr(mod, sym), f"{mod_name}.{sym} should be deleted"

    for mod_name, symbols in alive.items():
        mod = importlib.import_module(mod_name)
        for sym in symbols:
            assert hasattr(mod, sym), f"{mod_name}.{sym} must NOT be deleted"


def test_metrics_survivors_unchanged_and_orphan_constants_removed() -> None:
    import numpy as np
    import pandas as pd

    from src.ml import metrics as m

    # Given: the exact deterministic input the pre-deletion baseline was taken on.
    rng = np.random.default_rng(11)
    n = 4000
    scores = rng.normal(size=n)
    labels = scores * 0.3 + rng.normal(size=n)
    groups = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=200), n // 200))
    ic_df = pd.DataFrame({"d": groups.to_numpy(), "score": scores, "y": labels})

    # When / Then: the survivor's value is bit-stable against the recorded baseline.
    got_ic = float(m.mean_group_rank_ic(ic_df, ["d"], "score", "y"))
    assert f"{got_ic:.17g}" == "0.2604736842105263"

    agg = m.aggregate_metrics(labels)
    assert set(agg) == {
        "top_1_return", "win_rate", "profit_factor", "mean_win", "mean_loss", "sharpe",
    }
    assert np.isfinite(agg["sharpe"])

    # And: constants the survivors need stay; constants orphaned by the deletions go.
    assert hasattr(m, "_DAILY_ANNUALIZATION")
    assert hasattr(m, "_BASE_METRIC_KEYS")
    assert not hasattr(m, "_YEAR_RE"), "orphaned by removing extract_year"
    assert not hasattr(m, "_MIN_YEAR_SAMPLES"), "orphaned by removing yearly_breakdown"
    assert not hasattr(m, "re"), "the re import is orphaned by removing extract_year"


def test_pyproject_drops_unused_dependencies() -> None:
    import tomllib
    from pathlib import Path

    # Given
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared = data["project"]["dependencies"]
    names = {
        d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        for d in declared
    }

    # Then: every distribution with zero imports across src/, tests/ and legacy/ is gone.
    # setuptools/pykrx retired together: pykrx (and its matplotlib/pillow transitive chain)
    # was the sole runtime importer of pkg_resources; the build backend still gets its own
    # setuptools from an isolated build environment via [build-system], not this list.
    removed = {
        "python-dotenv", "tenacity", "psutil", "tqdm", "openpyxl", "gspread",
        "oauth2client", "xgboost", "catboost", "huggingface-hub", "optuna",
        "pykrx", "setuptools",
    }
    assert names & removed == set(), f"still declared: {sorted(names & removed)}"

    # And: the build backend requirement is untouched.
    assert any("setuptools" in r for r in data["build-system"]["requires"])

    # And: dependencies actually in use are still declared.
    for kept in ("numpy", "pandas", "pyarrow", "scipy", "lightgbm", "scikit-learn", "joblib"):
        assert kept in names, f"{kept} must remain declared"


def test_live_entrypoints_still_import_after_dependency_removal() -> None:
    import importlib

    # Given: the operational entrypoints documented in README.md / docs/guide.md.
    entrypoints = [
        "src.daily.collect",
        "src.daily.predict",
        "src.daily.archive_intraday",
        "src.ml.retrain",
        "src.ml.costaware_topk",
        "src.ml.topk_ranker_research",
        "src.backfill.backfill_altdata",
        "src.backfill.kis_flow_backfill",
        "src.backfill.intraday.backfill_minute_history",
    ]

    # When / Then: each imports without error.
    for name in entrypoints:
        assert importlib.import_module(name) is not None, f"{name} failed to import"


def test_backfill_price_facade_symbol_is_fully_deleted() -> None:
    import importlib
    from pathlib import Path

    import pytest

    assert not Path("src/backfill/backfill_price.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.backfill.backfill_price")


def test_synthetic_output_digests_unchanged_after_subtraction() -> None:
    import hashlib
    import json

    import numpy as np
    import pandas as pd

    from src.ml.dataset import build_ml_dataset
    from src.ml.purged_cv import PurgedGroupTimeSeriesSplit

    # 부동소수점 배열은 CPU/BLAS 벤더별 리덕션 순서에 따라 마지막 ULP가 흔들릴 수
    # 있으므로(호스트 하드웨어에 따라 실측 확인됨: 구조/카테고리 해시 3종은 전
    # 환경 일치, float 해시만 불일치), 소수점 8자리로 반올림한 뒤 해싱해 하드웨어
    # 잡음을 흡수하면서도 그보다 큰(=1e-8 초과) 실제 로직 변경은 계속 잡아낸다.
    FLOAT_DIGEST_DECIMALS = 8

    def digest(*arrays: object) -> str:
        h = hashlib.sha256()
        for a in arrays:
            arr = np.asarray(a)
            if arr.dtype == object:
                h.update(json.dumps(arr.tolist(), sort_keys=True, default=str).encode())
            elif np.issubdtype(arr.dtype, np.floating):
                h.update(np.ascontiguousarray(np.round(arr, FLOAT_DIGEST_DECIMALS)).tobytes())
            else:
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]

    # Given: digests recorded on this exact input BEFORE any code was deleted.
    groups = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=40), 5))
    x = pd.DataFrame({"f": np.arange(len(groups))})
    splits = list(PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=1).split(x, groups=groups))

    gx, _gt, gcat, gproc = build_ml_dataset(
        _raw_trade_log_for_digest(), None, feature_set="close_morning61", panel_mode="scenario_action"
    )

    # Then: structural/categorical digests reproduce byte-for-byte; float digests
    # tolerate hardware-level ULP noise via FLOAT_DIGEST_DECIMALS rounding above.
    assert digest(*[a for pair in splits for a in pair]) == "82c66517bd8514cb"
    assert digest(np.array(sorted(gx.columns), dtype=object)) == "0cd73ada27529408"
    assert digest(np.array(sorted(gcat), dtype=object)) == "5ccc339c7392dbfc"
    assert digest(gproc.sort_index()["target_return"].to_numpy(np.float64)) == "192a90f712131e32"  # 2026-09-12: 반올림 해싱으로 갱신 (하드웨어 ULP 잡음 흡수)
    assert digest(gx.sort_index().select_dtypes("number").to_numpy(np.float64)) == "42e66c97d1138fca"  # 2026-09-12: 반올림 해싱으로 갱신 (하드웨어 ULP 잡음 흡수)


def test_removed_modules_are_gone_and_orderbook_store_survives() -> None:
    import importlib
    from pathlib import Path

    import pytest

    # Then: 소비자 0으로 실증된 두 모듈은 삭제된다
    for rel, mod in {
        "src/utils/export_archive.py": "src.utils.export_archive",
        "src/daily/collect_auction.py": "src.daily.collect_auction",
    }.items():
        assert not Path(rel).exists(), f"{rel} should be deleted"
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)

    # And: 테스트 전용이던 스프레드시트 렌더러도 함께 사라진다
    archive = importlib.import_module("src.daily.archive")
    assert not hasattr(archive, "export_archive_for_spreadsheet")

    # And: collect.py가 결정시점 캡처에 쓰는 호가 저장소는 반드시 존치한다
    assert Path("src/data/orderbook_store.py").exists()
    store = importlib.import_module("src.data.orderbook_store")
    assert callable(store.append_orderbook_snapshots)
    collect = importlib.import_module("src.daily.collect")
    assert callable(collect.persist_daily_snapshot)


def test_dead_kis_client_methods_absent_live_quote_methods_present() -> None:
    from src.api.kis.client import KisApiClient

    for name in (
        "resolve_stock_market_div_code",
        "get_trade_strength",
        "get_condition_list",
        "get_condition_result",
    ):
        assert not hasattr(KisApiClient, name), f"KisApiClient.{name} should be deleted"
    for name in (
        "get_current_price",
        "get_program_net_buy",
        "get_investor_trend_estimate",
        "get_orderbook_snapshot",
    ):
        assert hasattr(KisApiClient, name), f"KisApiClient.{name} must remain"


def test_altdata_config_has_no_pykrx_field() -> None:
    import dataclasses

    from src.backfill.altdata.config import AltDataFetchConfig

    names = {f.name for f in dataclasses.fields(AltDataFetchConfig)}
    assert "pykrx_requests_per_sec" not in names
    assert AltDataFetchConfig.dart_requests_per_sec == 8.0
    assert AltDataFetchConfig.krx_requests_per_sec == 4.0


def test_intraday_store_public_surface_shrinks() -> None:
    from src.data import intraday_store

    assert not hasattr(intraday_store, "merge_partition_frame")
    assert not hasattr(intraday_store, "read_intraday_range")
    for name in (
        "write_intraday_partition",
        "write_tick_partition",
        "intraday_partition_path",
        "tick_partition_path",
    ):
        assert hasattr(intraday_store, name), f"intraday_store.{name} must remain"


def test_write_partition_fails_closed_on_unreadable_existing(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    from src.data import intraday_store
    from src.data.intraday_schema import normalize_bar_frame

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    target = intraday_store.intraday_partition_path(1, "2026-09-05", "regular")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a valid parquet file")
    before = target.read_bytes()

    raw = pd.DataFrame(
        {
            "time": ["090300"],
            "open": [70000],
            "high": [70100],
            "low": [69900],
            "close": [70000],
            "jdiff_vol": [1000],
            "value": [70],
        }
    )
    new_df = normalize_bar_frame(raw, "ls", "2026-09-05", "005930")
    with pytest.raises(OSError, match="Cannot read existing partition evidence"):
        intraday_store.write_intraday_partition(new_df, 1, "2026-09-05", "regular")
    assert target.read_bytes() == before


def test_universe_scan_archive_surface_removed() -> None:
    from src.daily import universe_scan as mod

    for name in (
        "collect_universe_scan",
        "archive_universe_snapshot",
        "map_ranking_rows_to_archive_frame",
        "map_kiwoom_ranking_rows_to_archive_frame",
        "_ARCHIVE_COLUMNS",
        "RANKING_SCAN_INPUT_CNT",
        "UNIVERSE_SCAN_SCENARIO_TAG",
    ):
        assert not hasattr(mod, name), f"universe_scan.{name} should be deleted"
        assert name not in mod.__all__, f"universe_scan.__all__ still lists {name}"


def test_schema_spreadsheet_helpers_removed() -> None:
    from src.processing import schema as mod

    for name in (
        "StandardColumns",
        "LEGACY_RAW_TO_KOREAN_MAP",
        "STANDARD_TO_KOREAN_MAP",
        "normalize_column_names",
    ):
        assert not hasattr(mod, name), f"schema.{name} should be deleted"


def test_raw_to_standard_map_values_preserved() -> None:
    from src.processing.schema import RAW_TO_STANDARD_MAP

    assert RAW_TO_STANDARD_MAP["매수날짜"] == "trade_date"
    assert RAW_TO_STANDARD_MAP["(매수날짜)"] == "trade_date"
    assert RAW_TO_STANDARD_MAP["종목코드"] == "stock_code"
    assert RAW_TO_STANDARD_MAP["(종목코드)"] == "stock_code"
    assert RAW_TO_STANDARD_MAP["수익률"] == "net_return"
    assert RAW_TO_STANDARD_MAP["(수익률, %)"] == "net_return"
    assert RAW_TO_STANDARD_MAP["(Win)"] == "Win"


_DELETED_SETTINGS_FIELDS = (
    "TARGET_CONDITION_NAME",
    "OVERHEATED_CONDITION_NAME",
    "NEW_HIGH_CONDITION_NAME",
    "NEAR_NEW_HIGH_CONDITION_NAME",
    "UPPER_LIMIT_NEXT_DAY_CONDITION_NAME",
    "UPPER_LIMIT_CONDITION_NAME",
    "EMA_PERIOD",
    "SMA_PERIOD",
    "SMA60_PERIOD",
    "CANDLE_BODY_RATIO_THRESHOLD",
    "GAP_UP_THRESHOLD",
    "SMA_LOOKBACK_DAYS",
    "SMA60_LOOKBACK_DAYS",
    "EMA_LOOKBACK_DAYS",
    "DEFAULT_SCENARIOS",
    "DAY_NAME_MAP",
    "CONFIGS_DIR",
    "DAILY_DIR",
    "TOKEN_FILE",
    "DATA_TOKEN_FILE",
    "ORDERBOOK_DIR",
    "LABEL_ENCODER_PATH",
    "MODEL_PATH",
)


def test_deleted_settings_fields_gone() -> None:
    from src import settings as settings_module
    from src.config import Settings

    for name in _DELETED_SETTINGS_FIELDS:
        assert name not in Settings.model_fields, f"{name} should be deleted from Settings"
        assert not hasattr(settings_module.settings, name), f"{name} should be gone from Settings instance"
        assert not hasattr(settings_module, name), f"{name} should be gone from src.settings module"
        assert name not in settings_module.__all__, f"{name} should be gone from src.config.__all__"


def test_live_path_fields_follow_overrides(tmp_path) -> None:
    from src.config import Settings

    settings = Settings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", _env_file=None)
    for name in (
        "PARQUET_DIR",
        "HISTORY_DIR",
        "ALTDATA_DIR",
        "PRICE_HISTORY_PARQUET_PATH",
        "HISTORY_PARQUET_PATH",
        "MODELS_DIR",
    ):
        assert str(getattr(settings, name)).startswith(str(tmp_path)), f"{name} ignored the override"


def test_stale_env_names_ignored(monkeypatch) -> None:
    from src.config import Settings

    monkeypatch.setenv("EMA_PERIOD", "5")
    monkeypatch.setenv("CONFIGS_DIR", "/nonexistent")
    settings = Settings(_env_file=None)
    assert not hasattr(settings, "EMA_PERIOD")
    assert not hasattr(settings, "CONFIGS_DIR")


def test_indicators_module_uses_data_key_client_once() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/api/kis/indicators.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "KisApiClient"
    ]
    assert len(calls) == 1
    (call,) = calls
    assert call.args == []
    assert len(call.keywords) == 1
    (kw,) = call.keywords
    assert kw.arg is None
    assert isinstance(kw.value, ast.Call)
    assert isinstance(kw.value.func, ast.Name)
    assert kw.value.func.id == "kis_data_client_kwargs"
