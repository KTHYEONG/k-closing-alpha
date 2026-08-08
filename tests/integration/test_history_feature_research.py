"""History-feature real-data streaming benchmark (slow, opt-in).

`docs/specs/ml_history_feature_scaling.md` HFS-08: 실제 ``price_history.parquet``
입력으로 peak RSS/elapsed/batch count/OOF 날짜 일치/선정 피처 수를 기록하고,
활성 아티팩트를 승격·덮어쓰지 않습니다.

`docs/specs/ml_training_optimization.md` MTO-04: feature cache 지문이 일치하면
warm read, source/config/key/cutoff 가 바뀌면 재구성해 stale feature 를 재사용하지
않습니다.

기본 테스트 스위트에서는 slow 실행만 생략합니다. ``K_CLOSING_RUN_SLOW=1``
환경 변수로 HFS-08 실데이터 벤치마크를 실행합니다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_selection import FeatureSelectionConfig
from src.ml.history_feature_research import run_history_feature_research_experiment
from src.ml.history_features import HistoryFeatureExecutionConfig

HISTORY_PATH = Path("data/history/price_history.parquet")
TRADE_LOG_PATH = Path("data/parquet/trade_log.parquet")

_RUN_SLOW = os.environ.get("K_CLOSING_RUN_SLOW") == "1"


def _build_trade_log(n_dates: int = 18, n_stocks: int = 8, seed: int = 4) -> pd.DataFrame:
    """MTO-04 캐시 테스트용 합성 매매일지 (스프레드시트 헤더)."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime([f"2024-01-{2 + d:02d}" for d in range(n_dates)])
    rows: list[dict[str, object]] = []
    for date in dates:
        for i in range(n_stocks):
            code = f"{i + 1:06d}"
            close_p = int(10_000 * (1 + float(rng.normal(0.5, 2.0)) / 100))
            rows.append(
                {
                    "매수날짜": date,
                    "종목코드": code,
                    "(시가)": 10_000,
                    "(고가)": 10_500,
                    "(저가)": 9_700,
                    "(종가)": close_p,
                    "(전일종가)": 9_900,
                    "(시가총액, 억)": 1_000.0 + i * 100,
                    "(거래대금, 억)": 100.0 + i * 20,
                    "(등락률)": float(rng.normal(0.5, 2.0)),
                    "(선정 순위)": float(i + 1),
                    "(기관_순매수)": float((i - 2) * 50),
                    "(외국인_순매수)": float(i * 30),
                    "(프로그램_순매수)": float((i - 1) * 10),
                    "(체결강도)": 110.0 + i,
                    "(시장구분)": "KOSPI" if i % 2 == 0 else "KOSDAQ",
                    "(총 종목 수)": float(n_stocks),
                    "(평균 거래대금)": 90.0,
                    "(kospi, %)": 0.3,
                    "(kosdaq, %)": 0.1,
                    "v_kospi": 15.0,
                    "v_kosdaq": 18.0,
                    "(거래량)": 1_000_000 + i * 1_000,
                    "(테마/섹터)": f"theme{i % 3}",
                    "(차트분석)": "신고가 근접",
                    "(매수 가격)": float(close_p * 0.99),
                    "(매도 가격)": float(close_p * 1.02),
                    "(수익률, %)": f"{float(rng.normal(0.2, 1.2)):.4f}",
                }
            )
    return pd.DataFrame(rows)


def _build_price_history(trade_log: pd.DataFrame, seed: int = 4) -> pd.DataFrame:
    """MTO-04 캐시 테스트용 합성 EOD 판넬."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(sorted(trade_log["매수날짜"].unique()))
    symbols = sorted(trade_log["종목코드"].astype(str).str.zfill(6).unique())
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        price = 10_000.0
        for date in dates:
            c = price * (1 + rng.normal(0, 0.008))
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": max(price, c) * 1.003,
                    "low": min(price, c) * 0.997,
                    "close": c,
                    "prev_close": price,
                    "market_cap_100m": rng.uniform(100, 5_000),
                    "trade_value_100m": rng.uniform(10, 500),
                    "daily_change_pct": c / price - 1.0,
                    "market": "KOSPI",
                    "volume": rng.uniform(1e4, 1e7),
                    "foreign_netbuy": rng.normal(0, 1e5),
                    "inst_netbuy": rng.normal(0, 1e5),
                    "program_netbuy": rng.normal(0, 5e4),
                    "kospi_pct": rng.normal(0, 0.004),
                    "kosdaq_pct": rng.normal(0, 0.004),
                    "v_kospi": rng.uniform(15, 30),
                    "v_kosdaq": rng.uniform(12, 25),
                }
            )
            price = c
    return pd.DataFrame(rows)


def _small_config() -> FeatureSelectionConfig:
    return FeatureSelectionConfig(
        min_retained=1,
        max_retained=40,
        hard_max_retained=100,
        correlation_threshold=1.0,
    )


def _run_once(
    trade_log: pd.DataFrame,
    price_history: pd.DataFrame,
    cache_dir: Path,
    export_dir: Path,
    research_cutoff: str | None = None,
) -> dict:
    return run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history=price_history,
        n_splits=3,
        purge_gap=1,
        feature_selection_config=_small_config(),
        research_cutoff=research_cutoff,
        cache_dir=str(cache_dir),
        export_dir=str(export_dir),
    )


def test_mto04_feature_cache_warm_read_and_invalidation(tmp_path: Path) -> None:
    """MTO-04: 지문 일치 시 warm read, source/decision-key/cutoff 불일치 시 재구성."""
    trade_log = _build_trade_log(n_dates=18, n_stocks=8, seed=4)
    price_history = _build_price_history(trade_log, seed=4)
    cache_dir = tmp_path / "cache"
    export_dir = tmp_path / "research"

    first = _run_once(trade_log, price_history, cache_dir, export_dir)
    assert first["build_metrics"]["cache_state"] == "cold"
    assert list(cache_dir.glob("history_features_*.parquet"))

    second = _run_once(trade_log, price_history, cache_dir, export_dir)
    assert second["build_metrics"]["cache_state"] == "warm"
    assert second["comparison"]["control_oof_dates"] == first["comparison"]["control_oof_dates"]
    assert second["comparison"]["candidate_oof_dates"] == first["comparison"]["candidate_oof_dates"]

    # decision-key 불일치 (추가 매매일) → cold 재구성.
    wider_trade_log = _build_trade_log(n_dates=19, n_stocks=8, seed=4)
    wider_price = _build_price_history(wider_trade_log, seed=4)
    key_mismatch = _run_once(wider_trade_log, wider_price, cache_dir, export_dir)
    assert key_mismatch["build_metrics"]["cache_state"] == "cold"

    # 커트오프 불일치 → cold 재구성.
    cutoff_mismatch = _run_once(
        trade_log, price_history, cache_dir, export_dir, research_cutoff="2024-01-16"
    )
    assert cutoff_mismatch["build_metrics"]["cache_state"] == "cold"

    # 동일 입력 복귀 → warm read 로 stale feature 가 아닌 동일 판넬을 재사용.
    back = _run_once(trade_log, price_history, cache_dir, export_dir)
    assert back["build_metrics"]["cache_state"] == "warm"


@pytest.mark.skipif(
    not _RUN_SLOW,
    reason="real-data benchmark; set K_CLOSING_RUN_SLOW=1 to run",
)
@pytest.mark.slow
@pytest.mark.skipif(
    not (HISTORY_PATH.is_file() and TRADE_LOG_PATH.is_file()),
    reason="real data files are missing",
)
def test_hfs08_real_data_streaming_benchmark(tmp_path: Path) -> None:
    """실데이터 streaming 실행이 메모리 예산 내에서 불변량을 유지하고 지표를 기록합니다."""
    trade_log = pd.read_parquet(TRADE_LOG_PATH)
    export_dir = tmp_path / "research"
    config = FeatureSelectionConfig()
    execution = HistoryFeatureExecutionConfig(
        memory_budget_bytes=8 * 1024**3,
        enforce_memory_budget=True,
    )
    result = run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history_path=str(HISTORY_PATH),
        n_splits=5,
        purge_gap=1,
        feature_selection_config=config,
        execution_config=execution,
        export_dir=str(export_dir),
    )
    metrics = result["build_metrics"]
    assert metrics["input_history_rows"] > 0
    assert metrics["decision_key_rows"] >= 1
    assert metrics["output_rows"] == metrics["decision_key_rows"]
    assert metrics["batch_count"] >= 1
    assert metrics["peak_rss_bytes"] <= 8 * 1024**3
    assert metrics["elapsed_seconds"] > 0.0
    assert result["comparison"]["identical_oof_dates"] is True
    final_count = len(result["candidate"]["final_features"])
    assert 1 <= final_count <= 500
    # 활성 아티팩트를 승격하거나 덮어쓰지 않습니다.
    assert Path(result["candidate_bundle_path"]).is_file()
