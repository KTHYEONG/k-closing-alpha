"""Causal EOD price-history feature catalogue and streaming panel builder (research-only).

`docs/specs/ml_feature_selection_pipeline.md` 의 causal data/candidate 계약과
`docs/specs/ml_history_feature_scaling.md` 의 streaming 실행 계약을 구현합니다.

``data/history/price_history.parquet`` 의 EOD 판넬을 720 컬럼 후보 카탈로그로
변환하고, 각 decision key ``(stock_code, trade_date)`` 를 그 종목의 결정일
**엄격히 이전** 최신 EOD 행으로 ``merge_asof(allow_exact_matches=False)`` 조인해
미래 정보(당일 종가·수급·시가총액·지수)가 결정 시점에 노출되지 않도록 합니다.

Streaming 설계 (두 단계):
1. 시간축(source) 피처: 종목 사전순으로 정렬된 유한 배치 단위로 계산하고, 각
   배치를 즉시 결정 key 로 축소(merge_asof)합니다. 전체 ``(history_rows, 720)``
   행렬을 한 번에 구성하지 않습니다.
2. 횡단면(cross-sectional) 피처: 시간축 피처가 결정 key 판넬로 축소된 뒤, 해당
   결정일의 후보 판넬 내 순위/상호작용으로 계산합니다. 이는 기존 후보 판넬 모델
   범위와 일치하며 배치 분할과 무관하게 결정적입니다.

매니페스트는 ``history_temporal_panel``(시간축)과 ``decision_candidate_panel``
(횡단면) 범위를 구분합니다.

원칙:
- 모든 rolling/횡단면 값은 원천 EOD 날짜에서 계산되며, merge_asof 의 1일 지연을
  통해 이후 decision 날짜에만 노출됩니다.
- 종목 시리즈를 forward-fill 하거나 미래에서 채우지 않습니다. 이력이 없는
  decision key 는 ``NaN`` 을 유지합니다.
- ratio/log 연산은 안전한 벡터 나눗셈(``np.divide(..., where=denominator != 0)``)
  또는 유효 도메인 마스크를 사용하며, 출력 경계에서 모든 ``+/-inf`` 를 ``NaN`` 으로
  치환하고 남은 무한대가 있으면 fail-closed 합니다.
- Vectorized 전용이며 candidate 생성에 ``pd.apply`` 를 사용하지 않습니다.
"""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydantic

try:
    import psutil

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional telemetry dependency
    _HAS_PSUTIL = False

HISTORICAL_CATALOGUE_VERSION = "causal_history_v2"

# --- 카탈로그 lookback/그리드 상수 (버전화된 동결 값) -----------------------------------------

_RETURN_LAGS = (1, 2, 3, 5, 10, 20, 40, 60, 120, 180, 240)
_MA_WINDOWS = (3, 5, 10, 20, 40, 60, 120, 180, 240)
_ZSCORE_WINDOWS = (5, 10, 20, 40, 60, 120, 180, 240)
_DRAWDOWN_WINDOWS = (5, 10, 20, 40, 60, 120, 180, 240)
_VOL_WINDOWS = (5, 10, 20, 40, 60, 120, 180, 240)
_ACCEL_LAGS = (1, 2, 3, 5, 10, 20, 40, 60, 120)
_MA_CROSS_PAIRS = (
    (5, 20),
    (5, 60),
    (10, 20),
    (10, 60),
    (20, 60),
    (20, 120),
    (40, 120),
    (40, 240),
    (60, 240),
    (120, 240),
)
_CUM_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_EWMA_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_PCT_HIGH_WINDOWS = (60, 120, 240)
_RSI_WINDOWS = (5, 14, 20, 40, 60, 120, 240)
_BOLLINGER_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_MA_SLOPE_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_VOL_CHANGE_PAIRS = ((5, 20), (20, 60), (60, 240))
_TRIX_WINDOWS = (5, 10, 20, 60, 120)
_DOWNSIDE_VOL_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_PARKINSON_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_AUTOCORR_WINDOW = 60
_AUTOCORR_LAGS = (1, 2, 3, 5)
_OHLC_LAGS = (0, 1, 2, 3, 5, 10, 20)
_OHLC_SHORT_LAGS = (0, 1, 2, 3, 5, 10)
_RANGE_WINDOWS = (5, 10, 20, 40, 60, 120)
_RANGE_VOL_WINDOWS = (5, 20, 60)

_LIQUIDITY_LAGS = (0, 1, 2, 3, 5, 10, 20)
_LIQUIDITY_CHANGE_LAGS = (1, 2, 3, 5, 10, 20)
_LIQUIDITY_WINDOWS = (5, 10, 20, 40, 60, 120, 240)

_FLOW_SOURCES = ("foreign_netbuy", "inst_netbuy", "program_netbuy")
_FLOW_LAGS = (0, 1, 2, 3, 5, 10, 20)
_FLOW_CHANGE_LAGS = (1, 2, 3, 5, 10, 20)
_FLOW_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_FLOW_STREAK_WINDOWS = (5, 10, 20, 40, 60, 120)

_INDEX_LAGS = (0, 1, 2, 3, 5, 10, 20)
_INDEX_VOL_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
_INDEX_VOL_CHANGE_LAGS = (1, 2, 3, 5, 10, 20)
_VIX_WINDOWS = (5, 10, 20, 40, 60, 120)

_RANK_LAGS = (0, 1, 2, 3, 5, 10, 20)
_INTERACTION_LAGS = (0, 1, 2, 3, 5, 10)
_RATIO_LAGS = (0, 1, 2, 3)

# 패밀리별 예상 후보 수: Return/trend/mean-reversion 180, OHLC range/gap 96,
# Liquidity/size/turnover 120, Investor-flow 180, Market/regime 72, Cross-section 72.
_FAMILY_COUNTS: dict[str, int] = {
    "return_trend_mean_reversion": 180,
    "ohlc_range_gap": 96,
    "liquidity_size_turnover": 120,
    "investor_flow_dynamics": 180,
    "market_regime_context": 72,
    "cross_sectional_interactions": 72,
}
HISTORICAL_CATALOGUE_COUNT = 720

# 결정 후보 판넬(decision_candidate_panel)에서 계산되는 변환 집합. 이 변환들은
# 시간축 피처가 결정 key 판넬로 축소된 뒤에 계산되며, 나머지는 history_temporal_panel
# 시간축 변환입니다.
_DECISION_PANEL_TRANSFORMS: frozenset[str] = frozenset(
    {
        "cross_sectional_rank",
        "cross_sectional_rank_change",
        "rank_log_return",
        "rank_realised_vol",
        "rank_turnover",
        "rank_flow_density",
        "interaction",
        "interaction_ratio",
    }
)

_MARKET_CONTEXT_FAMILY = "market_regime_context"

REQUIRED_HISTORY_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "market_cap_100m",
    "trade_value_100m",
    "daily_change_pct",
    "volume",
    "foreign_netbuy",
    "inst_netbuy",
    "program_netbuy",
    "kospi_pct",
    "kosdaq_pct",
    "v_kospi",
    "v_kosdaq",
)


class HistoricalFeatureConfig(pydantic.BaseModel):
    """Causal history 카탈로그 빌드 설정 (불변)."""

    model_config = pydantic.ConfigDict(frozen=True)

    catalogue_version: str = HISTORICAL_CATALOGUE_VERSION
    history_date_col: str = "date"
    history_symbol_col: str = "symbol"
    decision_date_col: str = "trade_date"
    decision_symbol_col: str = "stock_code"


class HistoryFeatureExecutionConfig(pydantic.BaseModel):
    """Streaming 실행 설정 (불변).

    ``symbols_per_batch`` 를 명시하면 그대로 사용하고, ``memory_budget_bytes`` 만
    주어지면 대표 pilot 배치로 행당 RSS 를 측정해 예산 내 최대 배치 크기를
    선택합니다. 둘 다 없으면 단일 배치(기존 DataFrame 호출자 호환)로 동작합니다.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    memory_budget_bytes: int | None = None
    symbols_per_batch: int | None = None
    parquet_columns: tuple[str, ...] = REQUIRED_HISTORY_COLUMNS
    enforce_memory_budget: bool = True
    parquet_batch_rows: int = 250_000
    n_jobs: int = 4

    @pydantic.model_validator(mode="after")
    def _validate(self) -> HistoryFeatureExecutionConfig:
        if self.symbols_per_batch is not None and self.symbols_per_batch < 1:
            raise ValueError(f"symbols_per_batch must be >= 1, got {self.symbols_per_batch}")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError(
                f"memory_budget_bytes must be > 0, got {self.memory_budget_bytes}"
            )
        if self.parquet_batch_rows < 1:
            raise ValueError(f"parquet_batch_rows must be >= 1, got {self.parquet_batch_rows}")
        if self.n_jobs != -1 and self.n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 (all cores) or >= 1, got {self.n_jobs}")
        missing = [col for col in REQUIRED_HISTORY_COLUMNS if col not in set(self.parquet_columns)]
        if missing:
            raise ValueError(
                f"parquet_columns must include required history columns: {missing}"
            )
        return self


class HistoryFeatureBuildMetrics(pydantic.BaseModel):
    """Streaming 실행 텔레메트리 (불변)."""

    model_config = pydantic.ConfigDict(frozen=True)

    input_history_rows: int
    decision_key_rows: int
    output_rows: int
    batch_count: int
    estimated_bytes_per_source_row: float
    peak_rss_bytes: int
    elapsed_seconds: float
    nonfinite_to_nan_count: int


def _zfill_symbols(series: pd.Series) -> pd.Series:
    """종목 코드를 6 자리로 정규화합니다 (결측은 그대로 결측으로 유지, index 보존)."""
    return series.astype("string").str.strip().str.zfill(6).astype("object")


def _build_catalogue() -> tuple[dict[str, object], ...]:
    """720 컬럼 후보 카탈로그를 패밀리 순서로 생성합니다 (동결, 결정적)."""
    entries: list[dict[str, object]] = []

    def add(family: str, name: str, source_field: str, transform: str, lookback: object) -> None:
        entries.append(
            {
                "feature_name": name,
                "family": family,
                "source_field": source_field,
                "transform": transform,
                "lookback": lookback,
            }
        )

    # --- 패밀리 1: Return, trend, mean reversion (180) ----------------------------------------
    for lag in _RETURN_LAGS:
        add("return_trend_mean_reversion", f"ret_log_{lag}", "close", "log_return", lag)
    for lag in _RETURN_LAGS:
        add("return_trend_mean_reversion", f"ret_simple_{lag}", "close", "simple_return", lag)
    for window in _MA_WINDOWS:
        add("return_trend_mean_reversion", f"ma_dist_{window}", "close", "ma_distance", window)
    for window in _ZSCORE_WINDOWS:
        add("return_trend_mean_reversion", f"close_zscore_{window}", "close", "zscore", window)
    for window in _DRAWDOWN_WINDOWS:
        add("return_trend_mean_reversion", f"drawdown_{window}", "close", "drawdown", window)
    for window in _VOL_WINDOWS:
        add("return_trend_mean_reversion", f"realised_vol_{window}", "close", "realised_vol", window)
    for window in _VOL_WINDOWS:
        add("return_trend_mean_reversion", f"vol_scaled_ret_{window}", "close", "vol_scaled_return", window)
    for lag in _ACCEL_LAGS:
        add("return_trend_mean_reversion", f"ret_accel_{lag}", "close", "return_acceleration", lag)
    for fast, slow in _MA_CROSS_PAIRS:
        add("return_trend_mean_reversion", f"ma_cross_{fast}_{slow}", "close", "ma_cross", f"{fast}_{slow}")
    for window in _CUM_WINDOWS:
        add("return_trend_mean_reversion", f"cum_change_{window}", "daily_change_pct", "cum_change", window)
    for window in _EWMA_WINDOWS:
        add("return_trend_mean_reversion", f"ewma_dist_{window}", "close", "ewma_distance", window)
    for window in _EWMA_WINDOWS:
        add("return_trend_mean_reversion", f"ewma_zscore_{window}", "close", "ewma_zscore", window)
    for window in _PCT_HIGH_WINDOWS:
        add("return_trend_mean_reversion", f"pct_of_high_{window}", "close", "pct_of_high", window)
    for window in _PCT_HIGH_WINDOWS:
        add("return_trend_mean_reversion", f"pct_of_low_{window}", "close", "pct_of_low", window)
    for window in _RSI_WINDOWS:
        add("return_trend_mean_reversion", f"rsi_{window}", "close", "rsi", window)
    for window in _BOLLINGER_WINDOWS:
        add("return_trend_mean_reversion", f"bollinger_upper_dist_{window}", "close", "bollinger_upper", window)
    for window in _BOLLINGER_WINDOWS:
        add("return_trend_mean_reversion", f"bollinger_lower_dist_{window}", "close", "bollinger_lower", window)
    for window in _CUM_WINDOWS:
        add("return_trend_mean_reversion", f"cum_vol_norm_{window}", "daily_change_pct", "cum_vol_norm", window)
    for window in _MA_SLOPE_WINDOWS:
            # ``ma_slope`` compares two rolling windows separated by ``window``;
            # its effective source history is therefore 2 * window observations.
            add(
                "return_trend_mean_reversion",
                f"ma_slope_{window}",
                "close",
                "ma_slope",
                2 * window,
            )
    for fast, slow in _VOL_CHANGE_PAIRS:
        add("return_trend_mean_reversion", f"vol_change_{fast}_{slow}", "close", "volatility_change", f"{fast}_{slow}")
    for window in _TRIX_WINDOWS:
        add("return_trend_mean_reversion", f"trix_{window}", "close", "trix", window)
    for fast, slow in _MA_CROSS_PAIRS:
        add("return_trend_mean_reversion", f"momentum_osc_{fast}_{slow}", "close", "momentum_osc", f"{fast}_{slow}")
    for window in _DOWNSIDE_VOL_WINDOWS:
        add("return_trend_mean_reversion", f"downside_vol_{window}", "close", "downside_vol", window)
    for window in _PARKINSON_WINDOWS:
        add("return_trend_mean_reversion", f"parkinson_vol_{window}", "close", "parkinson_vol", window)
    for lag in _AUTOCORR_LAGS:
        add("return_trend_mean_reversion", f"ret_autocorr_{lag}", "close", "return_autocorrelation", lag)

    # --- 패밀리 2: OHLC range and gap (96) -----------------------------------------------------
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"candle_body_{lag}", "open", "candle_body", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"candle_range_{lag}", "close", "candle_range", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"gap_ratio_{lag}", "prev_close", "gap_ratio", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"close_location_{lag}", "close", "close_location", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"upper_shadow_{lag}", "close", "upper_shadow", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"lower_shadow_{lag}", "close", "lower_shadow", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"body_to_range_{lag}", "close", "body_to_range", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"high_close_dist_{lag}", "high", "high_close_dist", lag)
    for lag in _OHLC_LAGS:
        add("ohlc_range_gap", f"low_close_dist_{lag}", "low", "low_close_dist", lag)
    for lag in _OHLC_SHORT_LAGS:
        add("ohlc_range_gap", f"intraday_return_{lag}", "open", "intraday_return", lag)
    for lag in _OHLC_SHORT_LAGS:
        add("ohlc_range_gap", f"overnight_return_{lag}", "prev_close", "overnight_return", lag)
    for window in _RANGE_WINDOWS:
        add("ohlc_range_gap", f"high_roll_max_dist_{window}", "high", "high_roll_max_dist", window)
    for window in _RANGE_WINDOWS:
        add("ohlc_range_gap", f"low_roll_min_dist_{window}", "low", "low_roll_min_dist", window)
    for window in _RANGE_WINDOWS:
        add("ohlc_range_gap", f"range_expansion_{window}", "close", "range_expansion", window)
    for window in _RANGE_VOL_WINDOWS:
        add("ohlc_range_gap", f"range_volatility_{window}", "close", "range_volatility", window)

    # --- 패밀리 3: Liquidity, size, turnover (120) ----------------------------------------------
    for lag in _LIQUIDITY_LAGS:
        add("liquidity_size_turnover", f"log_volume_{lag}", "volume", "log1p", lag)
    for lag in _LIQUIDITY_LAGS:
        add("liquidity_size_turnover", f"log_value_{lag}", "trade_value_100m", "log1p", lag)
    for lag in _LIQUIDITY_LAGS:
        add("liquidity_size_turnover", f"log_cap_{lag}", "market_cap_100m", "log1p", lag)
    for lag in _LIQUIDITY_LAGS:
        add("liquidity_size_turnover", f"turnover_{lag}", "trade_value_100m", "turnover", lag)
    for lag in _LIQUIDITY_LAGS:
        add("liquidity_size_turnover", f"volume_pct_rank_{lag}", "volume", "cross_sectional_rank", lag)
    for lag in _LIQUIDITY_LAGS:
        add("liquidity_size_turnover", f"value_pct_rank_{lag}", "trade_value_100m", "cross_sectional_rank", lag)
    for lag in _LIQUIDITY_LAGS:
        add("liquidity_size_turnover", f"cap_pct_rank_{lag}", "market_cap_100m", "cross_sectional_rank", lag)
    for lag in _LIQUIDITY_CHANGE_LAGS:
        add("liquidity_size_turnover", f"volume_change_{lag}", "volume", "change_ratio", lag)
    for lag in _LIQUIDITY_CHANGE_LAGS:
        add("liquidity_size_turnover", f"value_change_{lag}", "trade_value_100m", "change_ratio", lag)
    for lag in _LIQUIDITY_CHANGE_LAGS:
        add("liquidity_size_turnover", f"cap_change_{lag}", "market_cap_100m", "change_ratio", lag)
    for lag in _LIQUIDITY_CHANGE_LAGS:
        add("liquidity_size_turnover", f"turnover_change_{lag}", "trade_value_100m", "change_ratio", lag)
    for lag in _LIQUIDITY_CHANGE_LAGS:
        add("liquidity_size_turnover", f"volume_rank_change_{lag}", "volume", "cross_sectional_rank_change", lag)
    for lag in _LIQUIDITY_CHANGE_LAGS:
        add("liquidity_size_turnover", f"value_rank_change_{lag}", "trade_value_100m", "cross_sectional_rank_change", lag)
    for window in _LIQUIDITY_WINDOWS:
        add("liquidity_size_turnover", f"volume_surprise_{window}", "volume", "rolling_surprise", window)
    for window in _LIQUIDITY_WINDOWS:
        add("liquidity_size_turnover", f"value_surprise_{window}", "trade_value_100m", "rolling_surprise", window)
    for window in _LIQUIDITY_WINDOWS:
        add("liquidity_size_turnover", f"volume_zscore_{window}", "volume", "rolling_zscore", window)
    for window in _LIQUIDITY_WINDOWS:
        add("liquidity_size_turnover", f"turnover_zscore_{window}", "trade_value_100m", "rolling_zscore", window)
    for window in _LIQUIDITY_WINDOWS:
        add("liquidity_size_turnover", f"avg_dollar_volume_{window}", "trade_value_100m", "rolling_mean", window)

    # --- 패밀리 4: Investor-flow dynamics (180) -------------------------------------------------
    for source in _FLOW_SOURCES:
        short = source.removesuffix("_netbuy")
        for lag in _FLOW_LAGS:
            add("investor_flow_dynamics", f"{short}_signed_flow_{lag}", source, "lagged_value", lag)
        for lag in _FLOW_LAGS:
            add("investor_flow_dynamics", f"{short}_flow_density_{lag}", source, "flow_density", lag)
        for lag in _FLOW_LAGS:
            add("investor_flow_dynamics", f"{short}_flow_price_int_{lag}", source, "flow_price_interaction", lag)
        for lag in _FLOW_CHANGE_LAGS:
            add("investor_flow_dynamics", f"{short}_flow_change_{lag}", source, "difference", lag)
        for lag in _FLOW_CHANGE_LAGS:
            add(
                "investor_flow_dynamics",
                f"{short}_flow_change_ratio_{lag}",
                source,
                "signed_log_change",
                lag,
            )
        for window in _FLOW_WINDOWS:
            add("investor_flow_dynamics", f"{short}_flow_roll_sum_{window}", source, "rolling_sum", window)
        for window in _FLOW_WINDOWS:
            add("investor_flow_dynamics", f"{short}_flow_roll_mean_{window}", source, "rolling_mean", window)
        for window in _FLOW_WINDOWS:
            add("investor_flow_dynamics", f"{short}_flow_zscore_{window}", source, "rolling_zscore", window)
        for window in _FLOW_STREAK_WINDOWS:
            add("investor_flow_dynamics", f"{short}_flow_pos_streak_{window}", source, "positive_streak", window)

    # --- 패밀리 5: Market/regime context (72) ---------------------------------------------------
    for lag in _INDEX_LAGS:
        add("market_regime_context", f"index_ret_kospi_{lag}", "kospi_pct", "index_return", lag)
    for lag in _INDEX_LAGS:
        add("market_regime_context", f"index_ret_kosdaq_{lag}", "kosdaq_pct", "index_return", lag)
    for window in _INDEX_VOL_WINDOWS:
        add("market_regime_context", f"index_vol_kospi_{window}", "kospi_pct", "index_volatility", window)
    for window in _INDEX_VOL_WINDOWS:
        add("market_regime_context", f"index_vol_kosdaq_{window}", "kosdaq_pct", "index_volatility", window)
    for lag in _INDEX_VOL_CHANGE_LAGS:
        add("market_regime_context", f"vol_change_kospi_{lag}", "v_kospi", "index_vol_change", lag)
    for lag in _INDEX_VOL_CHANGE_LAGS:
        add("market_regime_context", f"vol_change_kosdaq_{lag}", "v_kosdaq", "index_vol_change", lag)
    for lag in _INDEX_LAGS:
        add("market_regime_context", f"relative_return_{lag}", "kospi_pct", "relative_return", lag)
    for lag in _INDEX_LAGS:
        add("market_regime_context", f"market_rank_{lag}", "daily_change_pct", "cross_sectional_rank", lag)
    for window in _VIX_WINDOWS:
        add("market_regime_context", f"vix_regime_kospi_{window}", "v_kospi", "rolling_surprise", window)
    for window in _VIX_WINDOWS:
        add("market_regime_context", f"vix_regime_kosdaq_{window}", "v_kosdaq", "rolling_surprise", window)
    for window in _VIX_WINDOWS:
        add("market_regime_context", f"broad_market_ma_dist_{window}", "kospi_pct", "broad_market_ma_dist", window)

    # --- 패밀리 6: Cross-sectional interactions (72) --------------------------------------------
    for lag in _RANK_LAGS:
        add("cross_sectional_interactions", f"rank_ret_{lag}", "close", "rank_log_return", lag)
    for lag in _RANK_LAGS:
        add("cross_sectional_interactions", f"rank_vol_{lag}", "close", "rank_realised_vol", lag)
    for lag in _RANK_LAGS:
        add("cross_sectional_interactions", f"rank_turnover_{lag}", "trade_value_100m", "rank_turnover", lag)
    for lag in _RANK_LAGS:
        add("cross_sectional_interactions", f"rank_flow_density_{lag}", "foreign_netbuy", "rank_flow_density", lag)
    for lag in _INTERACTION_LAGS:
        add("cross_sectional_interactions", f"int_ret_liquidity_{lag}", "close", "interaction", lag)
    for lag in _INTERACTION_LAGS:
        add("cross_sectional_interactions", f"int_flow_price_{lag}", "foreign_netbuy", "interaction", lag)
    for lag in _INTERACTION_LAGS:
        add("cross_sectional_interactions", f"int_ret_vol_{lag}", "close", "interaction", lag)
    for lag in _INTERACTION_LAGS:
        add("cross_sectional_interactions", f"int_ret_cap_{lag}", "close", "interaction", lag)
    for lag in _INTERACTION_LAGS:
        add("cross_sectional_interactions", f"int_vol_turnover_{lag}", "close", "interaction", lag)
    for lag in _INTERACTION_LAGS:
        add("cross_sectional_interactions", f"int_flow_vol_{lag}", "foreign_netbuy", "interaction", lag)
    for lag in _RATIO_LAGS:
        add("cross_sectional_interactions", f"ratio_ret_vol_{lag}", "close", "interaction_ratio", lag)
    for lag in _RATIO_LAGS:
        add("cross_sectional_interactions", f"ratio_flow_turnover_{lag}", "foreign_netbuy", "interaction_ratio", lag)

    return tuple(entries)


HISTORICAL_CATALOGUE: tuple[dict[str, object], ...] = _build_catalogue()


def _assert_catalogue_counts() -> None:
    """카탈로그 패밀리·총 수 불변량을 fail-closed 로 검증합니다."""
    counts: dict[str, int] = {}
    for entry in HISTORICAL_CATALOGUE:
        family = str(entry["family"])
        counts[family] = counts.get(family, 0) + 1
    for family, expected in _FAMILY_COUNTS.items():
        if counts.get(family, 0) != expected:
            raise ValueError(
                f"catalogue family {family!r} must have exactly {expected} features, "
                f"got {counts.get(family, 0)}"
            )
    if len(HISTORICAL_CATALOGUE) != HISTORICAL_CATALOGUE_COUNT:
        raise ValueError(
            f"catalogue must contain exactly {HISTORICAL_CATALOGUE_COUNT} features, "
            f"got {len(HISTORICAL_CATALOGUE)}"
        )


_assert_catalogue_counts()


def _catalogue_feature_names() -> list[str]:
    return [str(entry["feature_name"]) for entry in HISTORICAL_CATALOGUE]


def _temporal_catalogue_names() -> list[str]:
    """시간축(history_temporal_panel) 카탈로그 컬럼 (배치 단위 계산).

    결정 후보 판넬 변환과 시장 컨텍스트(패밀리 5, 날짜 수준) 변환을 제외합니다.
    """
    return [
        str(entry["feature_name"])
        for entry in HISTORICAL_CATALOGUE
        if str(entry["family"]) != _MARKET_CONTEXT_FAMILY
        and str(entry["transform"]) not in _DECISION_PANEL_TRANSFORMS
    ]


def _market_catalogue_names() -> list[str]:
    """시장 컨텍스트(패밀리 5 비순위, 날짜 수준) 카탈로그 컬럼."""
    return [
        str(entry["feature_name"])
        for entry in HISTORICAL_CATALOGUE
        if str(entry["family"]) == _MARKET_CONTEXT_FAMILY
        and str(entry["transform"]) not in _DECISION_PANEL_TRANSFORMS
    ]


def _decision_catalogue_names() -> list[str]:
    """결정 후보 판넬(decision_candidate_panel) 카탈로그 컬럼."""
    return [
        str(entry["feature_name"])
        for entry in HISTORICAL_CATALOGUE
        if str(entry["transform"]) in _DECISION_PANEL_TRANSFORMS
    ]


def _feature_panel_scope(transform: str) -> str:
    """매니페스트 panel_scope 를 결정적으로 분류합니다."""
    if transform in _DECISION_PANEL_TRANSFORMS:
        return "decision_candidate_panel"
    return "history_temporal_panel"


# --- 유한 안전 산술 헬퍼 --------------------------------------------------------------------


def _safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> pd.Series:
    """벡터 안전 나눗셈: 분모가 0 이거나 NaN 이면 NaN 을 반환합니다."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    if isinstance(numerator, pd.Series):
        return pd.Series(out, index=numerator.index)
    return pd.Series(out)


def _safe_zscore(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> pd.Series:
    """표준화 산술.

    유효한 관측에서 분산이 0이면 ``NaN`` 대신 중립값 0을 반환합니다. 이는
    상수 구간을 데이터 결측으로 잘못 분류하지 않으면서, 유효 관측이 없는
    rolling 구간의 ``NaN``은 그대로 보존합니다.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(num) & np.isfinite(den)
    zero_scale = finite & (den == 0)
    out[zero_scale] = 0.0
    np.divide(num, den, out=out, where=finite & (den != 0))
    if isinstance(numerator, pd.Series):
        return pd.Series(out, index=numerator.index)
    return pd.Series(out)


def _signed_log_change(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    """부호를 보존한 log 변화량으로 0 분모의 흐름을 안정적으로 비교합니다."""
    current_values = current.to_numpy(dtype=np.float64)
    previous_values = previous.to_numpy(dtype=np.float64)
    current_log = np.sign(current_values) * np.log1p(np.abs(current_values))
    previous_log = np.sign(previous_values) * np.log1p(np.abs(previous_values))
    return pd.Series(current_log - previous_log, index=current.index)


def _aggregate_market_dates(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """중복 종목 행을 순서 불변인 날짜별 시장값으로 축약합니다."""
    value_columns = ["kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq"]
    values = frame[[date_col, *value_columns]].copy()
    values[date_col] = pd.to_datetime(values[date_col])
    return (
        values.groupby(date_col, sort=True, as_index=False)[value_columns]
        .median()
        .reset_index(drop=True)
    )


def _sanitize_finite(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """``+/-inf`` 를 ``NaN`` 으로 치환하고 개수를 반환합니다.

    남은 무한대가 있으면 ``ValueError`` 로 fail-closed 합니다.
    """
    numeric_columns = frame.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        return frame, 0
    numeric = frame.loc[:, numeric_columns]
    values = numeric.to_numpy(dtype=np.float64)
    count = int(np.isinf(values).sum())
    if count:
        frame = frame.copy()
        frame.loc[:, numeric_columns] = numeric.where(
            np.isfinite(values) | np.isnan(values), np.nan
        )
    if np.isinf(frame.loc[:, numeric_columns].to_numpy(dtype=np.float64)).any():
        raise ValueError("history feature output still contains infinity after sanitization")
    return frame, count


# --- 벡터화 계산 헬퍼 ----------------------------------------------------------------------


def _shift_series(
    series: pd.Series,
    symbol_labels: pd.Series,
    lag: int,
) -> pd.Series:
    """종목 그룹별 lag 이동 결과를 원본 index 로 반환합니다."""
    return series.groupby(symbol_labels.to_numpy(), sort=False).shift(lag)


def _rolling_series(
    series: pd.Series,
    symbol_labels: pd.Series,
    window: int,
    func: str,
) -> pd.Series:
    """종목 그룹별 rolling 연산 결과를 원본 index 로 반환합니다 (vectorized)."""
    rolled = series.groupby(symbol_labels.to_numpy(), sort=False).rolling(
        window, min_periods=window
    )
    result = getattr(rolled, func)()
    if isinstance(result.index, pd.MultiIndex):
        result = result.droplevel(0)
    return result.reindex(series.index)


def _rolling_corr(
    series: pd.Series,
    other: pd.Series,
    symbol_labels: pd.Series,
    window: int,
) -> pd.Series:
    """종목 그룹별 rolling corr(a, b) 결과를 원본 index 로 반환합니다."""
    out = np.full(len(series), np.nan, dtype=np.float64)
    labels = symbol_labels.to_numpy()
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        base = series.to_numpy(dtype=np.float64)[idx]
        other_vals = other.to_numpy(dtype=np.float64)[idx]
        s = pd.Series(base, index=idx).rolling(window, min_periods=window)
        corr = s.corr(pd.Series(other_vals, index=idx)).to_numpy(dtype=np.float64)
        out[idx] = corr
    return pd.Series(out, index=series.index)


def _ewm_series(
    series: pd.Series,
    symbol_labels: pd.Series,
    span: int,
) -> pd.Series:
    """종목 그룹별 EWM 평균을 원본 index 로 반환합니다."""
    return series.groupby(symbol_labels.to_numpy(), sort=False).transform(
        lambda s: s.ewm(span=span, adjust=False).mean()
    )


def series_ewm_std(series: pd.Series, symbol_labels: pd.Series, span: int) -> pd.Series:
    """종목 그룹별 EWM 표준편차를 원본 index 로 반환합니다."""
    return series.groupby(symbol_labels.to_numpy(), sort=False).transform(
        lambda s: s.ewm(span=span, adjust=False).std()
    )


def _group_rank(
    series: pd.Series,
    group_labels: pd.Series,
    pct: bool = True,
) -> pd.Series:
    """그룹(날짜) 단위 순위를 원본 index 로 반환합니다."""
    return series.groupby(group_labels.to_numpy(), sort=False).rank(
        pct=pct, method="average"
    )


AssignFn = Callable[[str, pd.Series | np.ndarray], None]


# --- 시간축(source) 피처 계산 (배치 단위) ----------------------------------------------------


def _build_temporal_features(
    frame: pd.DataFrame, config: HistoricalFeatureConfig
) -> pd.DataFrame:
    """시간축 패밀리(1,2,3-비순위,4)를 배치 판넬에서 계산합니다.

    ``frame`` 은 ``(symbol, date)`` 로 정렬되고 중복이 제거된 EOD 판넬이어야
    합니다. 반환 DataFrame 의 index 는 ``frame.index`` 와 정렬됩니다.
    """
    symbol_col = config.history_symbol_col
    frame = frame.sort_values([symbol_col, config.history_date_col]).reset_index(drop=True)
    features: dict[str, pd.Series] = {}

    def assign(name: str, values: pd.Series | np.ndarray) -> None:
        arr = values.to_numpy(dtype=np.float64) if isinstance(values, pd.Series) else np.asarray(values, dtype=np.float64)
        features[name] = pd.Series(np.asarray(arr, dtype=np.float32), index=frame.index)

    labels = frame[symbol_col]
    close = frame["close"]
    open_ = frame["open"]
    high = frame["high"]
    low = frame["low"]
    prev_close = frame["prev_close"]
    volume = frame["volume"]
    value = frame["trade_value_100m"]
    cap = frame["market_cap_100m"]
    prev_close_safe = prev_close.replace(0, np.nan)
    open_safe = open_.replace(0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret1 = np.log(_safe_divide(close, _shift_series(close, labels, 1)))

        # --- 패밀리 1: Return, trend, mean reversion -----------------------------------------
        for lag in _RETURN_LAGS:
            assign(f"ret_log_{lag}", np.log(_safe_divide(close, _shift_series(close, labels, lag))))
        for lag in _RETURN_LAGS:
            assign(f"ret_simple_{lag}", _safe_divide(close, _shift_series(close, labels, lag)) - 1.0)
        for window in _MA_WINDOWS:
            ma = _rolling_series(close, labels, window, "mean")
            assign(f"ma_dist_{window}", _safe_divide(close, ma) - 1.0)
        for window in _ZSCORE_WINDOWS:
            ma = _rolling_series(close, labels, window, "mean")
            std = _rolling_series(close, labels, window, "std")
            assign(f"close_zscore_{window}", _safe_zscore(close - ma, std))
        for window in _DRAWDOWN_WINDOWS:
            roll_max = _rolling_series(close, labels, window, "max")
            assign(f"drawdown_{window}", _safe_divide(close, roll_max) - 1.0)
        for window in _VOL_WINDOWS:
            assign(f"realised_vol_{window}", _rolling_series(ret1, labels, window, "std"))
        for window in _VOL_WINDOWS:
            vol = _rolling_series(ret1, labels, window, "std")
            assign(f"vol_scaled_ret_{window}", _safe_divide(ret1, vol))
        for lag in _ACCEL_LAGS:
            assign(f"ret_accel_{lag}", ret1 - _shift_series(ret1, labels, lag))
        for fast, slow in _MA_CROSS_PAIRS:
            ma_fast = _rolling_series(close, labels, fast, "mean")
            ma_slow = _rolling_series(close, labels, slow, "mean")
            assign(f"ma_cross_{fast}_{slow}", _safe_divide(ma_fast, ma_slow) - 1.0)
        daily_change = frame["daily_change_pct"]
        for window in _CUM_WINDOWS:
            assign(f"cum_change_{window}", _rolling_series(daily_change, labels, window, "sum"))
        for window in _EWMA_WINDOWS:
            ewma = _ewm_series(close, labels, window)
            assign(f"ewma_dist_{window}", _safe_divide(close, ewma) - 1.0)
        for window in _EWMA_WINDOWS:
            ewma_mean = _ewm_series(close, labels, window)
            ewma_std = series_ewm_std(close, labels, window)
            assign(f"ewma_zscore_{window}", _safe_zscore(close - ewma_mean, ewma_std))
        for window in _PCT_HIGH_WINDOWS:
            roll_max = _rolling_series(close, labels, window, "max")
            assign(f"pct_of_high_{window}", _safe_divide(close, roll_max) - 1.0)
        for window in _PCT_HIGH_WINDOWS:
            roll_min = _rolling_series(close, labels, window, "min")
            assign(f"pct_of_low_{window}", _safe_divide(close, roll_min) - 1.0)
        change = close - _shift_series(close, labels, 1)
        gain = change.clip(lower=0.0)
        loss = (-change).clip(lower=0.0)
        for window in _RSI_WINDOWS:
            avg_gain = _rolling_series(gain, labels, window, "mean")
            avg_loss = _rolling_series(loss, labels, window, "mean")
            assign(f"rsi_{window}", 100.0 - 100.0 / (1.0 + _safe_divide(avg_gain, avg_loss)))
        for window in _BOLLINGER_WINDOWS:
            ma = _rolling_series(close, labels, window, "mean")
            std = _rolling_series(close, labels, window, "std")
            assign(f"bollinger_upper_dist_{window}", _safe_divide(ma + 2.0 * std, close) - 1.0)
        for window in _BOLLINGER_WINDOWS:
            ma = _rolling_series(close, labels, window, "mean")
            std = _rolling_series(close, labels, window, "std")
            assign(f"bollinger_lower_dist_{window}", _safe_divide(close, ma - 2.0 * std) - 1.0)
        for window in _CUM_WINDOWS:
            cum = _rolling_series(daily_change, labels, window, "sum")
            assign(f"cum_vol_norm_{window}", cum / np.sqrt(window))
        for window in _MA_SLOPE_WINDOWS:
            ma = _rolling_series(close, labels, window, "mean")
            ma_prev = _shift_series(ma, labels, window)
            assign(f"ma_slope_{window}", _safe_divide(ma, ma_prev) - 1.0)
        for fast, slow in _VOL_CHANGE_PAIRS:
            vol_fast = _rolling_series(ret1, labels, fast, "std")
            vol_slow = _rolling_series(ret1, labels, slow, "std")
            assign(f"vol_change_{fast}_{slow}", _safe_divide(vol_fast, vol_slow) - 1.0)
        for window in _TRIX_WINDOWS:
            ema1 = _ewm_series(close, labels, window)
            ema2 = _ewm_series(ema1, labels, window)
            assign(f"trix_{window}", _safe_divide(ema1, ema2) - 1.0)
        for fast, slow in _MA_CROSS_PAIRS:
            ma_fast = _rolling_series(close, labels, fast, "mean")
            ma_slow = _rolling_series(close, labels, slow, "mean")
            assign(f"momentum_osc_{fast}_{slow}", _safe_divide(ma_fast - ma_slow, (ma_fast + ma_slow) / 2.0))
        neg_ret = ret1.clip(upper=0.0)
        for window in _DOWNSIDE_VOL_WINDOWS:
            assign(f"downside_vol_{window}", _rolling_series(neg_ret, labels, window, "std"))
        hl_log = np.log(_safe_divide(high, low.replace(0, np.nan)))
        hl2 = pd.Series(hl_log.to_numpy(dtype=np.float64) * hl_log.to_numpy(dtype=np.float64), index=frame.index)
        for window in _PARKINSON_WINDOWS:
            mean_hl2 = _rolling_series(hl2, labels, window, "mean")
            assign(f"parkinson_vol_{window}", np.sqrt(mean_hl2 / (4.0 * np.log(2.0))))
        for lag in _AUTOCORR_LAGS:
            assign(
                f"ret_autocorr_{lag}",
                _rolling_corr(ret1, _shift_series(ret1, labels, lag), labels, _AUTOCORR_WINDOW),
            )

        # --- 패밀리 2: OHLC range and gap -----------------------------------------------------
        body = close - open_
        rng = high - low
        for lag in _OHLC_LAGS:
            assign(f"candle_body_{lag}", _shift_series(_safe_divide(body, open_safe), labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"candle_range_{lag}", _shift_series(_safe_divide(rng, close), labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"gap_ratio_{lag}", _shift_series(_safe_divide(open_, prev_close_safe) - 1.0, labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"close_location_{lag}", _shift_series(_safe_divide(close - low, rng), labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"upper_shadow_{lag}", _shift_series(_safe_divide(high - np.maximum(open_, close), rng), labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"lower_shadow_{lag}", _shift_series(_safe_divide(np.minimum(open_, close) - low, rng), labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"body_to_range_{lag}", _shift_series(_safe_divide(body, rng), labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"high_close_dist_{lag}", _shift_series(_safe_divide(high, close) - 1.0, labels, lag))
        for lag in _OHLC_LAGS:
            assign(f"low_close_dist_{lag}", _shift_series(_safe_divide(close, low.replace(0, np.nan)) - 1.0, labels, lag))
        for lag in _OHLC_SHORT_LAGS:
            assign(f"intraday_return_{lag}", _shift_series(_safe_divide(close, open_safe) - 1.0, labels, lag))
        for lag in _OHLC_SHORT_LAGS:
            assign(f"overnight_return_{lag}", _shift_series(_safe_divide(open_, prev_close_safe) - 1.0, labels, lag))
        for window in _RANGE_WINDOWS:
            hmax = _rolling_series(high, labels, window, "max")
            assign(f"high_roll_max_dist_{window}", _safe_divide(high, hmax) - 1.0)
        for window in _RANGE_WINDOWS:
            lmin = _rolling_series(low, labels, window, "min")
            assign(f"low_roll_min_dist_{window}", _safe_divide(low, lmin) - 1.0)
        for window in _RANGE_WINDOWS:
            mean_range = _rolling_series(rng, labels, window, "mean")
            assign(f"range_expansion_{window}", _safe_divide(rng, mean_range) - 1.0)
        for window in _RANGE_VOL_WINDOWS:
            assign(f"range_volatility_{window}", _rolling_series(rng, labels, window, "std"))

        # --- 패밀리 3: Liquidity/size/turnover (비순위 부분) ------------------------------------
        turnover = _safe_divide(value, cap)
        for lag in _LIQUIDITY_LAGS:
            assign(f"log_volume_{lag}", _shift_series(np.log1p(np.maximum(volume, 0.0)), labels, lag))
        for lag in _LIQUIDITY_LAGS:
            assign(f"log_value_{lag}", _shift_series(np.log1p(np.maximum(value, 0.0)), labels, lag))
        for lag in _LIQUIDITY_LAGS:
            assign(f"log_cap_{lag}", _shift_series(np.log1p(np.maximum(cap, 0.0)), labels, lag))
        for lag in _LIQUIDITY_LAGS:
            assign(f"turnover_{lag}", _shift_series(turnover, labels, lag))
        for source, prefix in (
            ("volume", "volume"),
            ("trade_value_100m", "value"),
            ("market_cap_100m", "cap"),
        ):
            series = frame[source]
            for lag in _LIQUIDITY_CHANGE_LAGS:
                assign(
                    f"{prefix}_change_{lag}",
                    _safe_divide(series, _shift_series(series, labels, lag)) - 1.0,
                )
        for lag in _LIQUIDITY_CHANGE_LAGS:
            assign(
                f"turnover_change_{lag}",
                _safe_divide(turnover, _shift_series(turnover, labels, lag)) - 1.0,
            )
        for window in _LIQUIDITY_WINDOWS:
            mean_vol = _rolling_series(volume, labels, window, "mean")
            assign(f"volume_surprise_{window}", _safe_divide(volume, mean_vol) - 1.0)
        for window in _LIQUIDITY_WINDOWS:
            mean_val = _rolling_series(value, labels, window, "mean")
            assign(f"value_surprise_{window}", _safe_divide(value, mean_val) - 1.0)
        for window in _LIQUIDITY_WINDOWS:
            mean_vol = _rolling_series(volume, labels, window, "mean")
            std_vol = _rolling_series(volume, labels, window, "std")
            assign(f"volume_zscore_{window}", _safe_zscore(volume - mean_vol, std_vol))
        for window in _LIQUIDITY_WINDOWS:
            mean_to = _rolling_series(turnover, labels, window, "mean")
            std_to = _rolling_series(turnover, labels, window, "std")
            assign(f"turnover_zscore_{window}", _safe_zscore(turnover - mean_to, std_to))
        for window in _LIQUIDITY_WINDOWS:
            assign(f"avg_dollar_volume_{window}", _rolling_series(value, labels, window, "mean"))

        # --- 패밀리 4: Investor-flow dynamics ---------------------------------------------------
        for source in _FLOW_SOURCES:
            short = source.removesuffix("_netbuy")
            flow = frame[source].to_numpy(dtype=np.float64)
            flow_s = pd.Series(flow, index=frame.index)
            for lag in _FLOW_LAGS:
                assign(f"{short}_signed_flow_{lag}", _shift_series(flow_s, labels, lag))
            for lag in _FLOW_LAGS:
                assign(f"{short}_flow_density_{lag}", _shift_series(_safe_divide(flow_s, value), labels, lag))
            for lag in _FLOW_LAGS:
                assign(f"{short}_flow_price_int_{lag}", _shift_series(flow_s * ret1, labels, lag))
            for lag in _FLOW_CHANGE_LAGS:
                assign(f"{short}_flow_change_{lag}", flow_s - _shift_series(flow_s, labels, lag))
            for lag in _FLOW_CHANGE_LAGS:
                assign(
                    f"{short}_flow_change_ratio_{lag}",
                    _signed_log_change(flow_s, _shift_series(flow_s, labels, lag)),
                )
            for window in _FLOW_WINDOWS:
                assign(f"{short}_flow_roll_sum_{window}", _rolling_series(flow_s, labels, window, "sum"))
            for window in _FLOW_WINDOWS:
                assign(f"{short}_flow_roll_mean_{window}", _rolling_series(flow_s, labels, window, "mean"))
            for window in _FLOW_WINDOWS:
                mean_f = _rolling_series(flow_s, labels, window, "mean")
                std_f = _rolling_series(flow_s, labels, window, "std")
                assign(f"{short}_flow_zscore_{window}", _safe_zscore(flow_s - mean_f, std_f))
            pos = (flow_s > 0.0).astype(np.float64)
            for window in _FLOW_STREAK_WINDOWS:
                assign(f"{short}_flow_pos_streak_{window}", _rolling_series(pos, labels, window, "sum"))

    ordered = _temporal_catalogue_names()
    missing = [name for name in ordered if name not in features]
    if missing:
        raise ValueError(f"temporal catalogue features were not computed: {missing[:5]}...")
    return pd.DataFrame(features, index=frame.index, columns=ordered, dtype=np.float32)


# --- 시장 컨텍스트(날짜 수준) 피처 -----------------------------------------------------------


def _build_market_context_frame(
    dates_df: pd.DataFrame, config: HistoricalFeatureConfig
) -> pd.DataFrame:
    """고유 날짜 시리즈에서 패밀리 5 시장 컨텍스트 피처를 계산합니다.

    ``dates_df`` 는 ``date``/``kospi_pct``/``kosdaq_pct``/``v_kospi``/``v_kosdaq``
    컬럼을 가진 고유 거래일 DataFrame 입니다. 결과 DataFrame 은 ``date`` 를
    index 로 가지며 결정 key 의 매칭 EOD 날짜로 매핑할 수 있습니다.
    """
    date_col = config.history_date_col
    ud = dates_df[[date_col, "kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq"]].copy()
    ud[date_col] = pd.to_datetime(ud[date_col])
    # 각 지수 컬럼은 서로 다른 원천 행에서 들어올 수 있고, parquet의 행 순서도
    # 입력 경로마다 다를 수 있다. 중앙값은 NaN을 무시하면서 행 순서에 불변이다.
    ud = _aggregate_market_dates(ud, date_col)
    level_kospi = (1.0 + ud["kospi_pct"].fillna(0.0)).cumprod()

    out: dict[str, pd.Series] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for lag in _INDEX_LAGS:
            out[f"index_ret_kospi_{lag}"] = ud["kospi_pct"].shift(lag)
        for lag in _INDEX_LAGS:
            out[f"index_ret_kosdaq_{lag}"] = ud["kosdaq_pct"].shift(lag)
        for window in _INDEX_VOL_WINDOWS:
            out[f"index_vol_kospi_{window}"] = ud["kospi_pct"].rolling(window, min_periods=window).std()
        for window in _INDEX_VOL_WINDOWS:
            out[f"index_vol_kosdaq_{window}"] = ud["kosdaq_pct"].rolling(window, min_periods=window).std()
        for lag in _INDEX_VOL_CHANGE_LAGS:
            out[f"vol_change_kospi_{lag}"] = _safe_divide(
                ud["v_kospi"], ud["v_kospi"].shift(lag)
            ) - 1.0
        for lag in _INDEX_VOL_CHANGE_LAGS:
            out[f"vol_change_kosdaq_{lag}"] = _safe_divide(
                ud["v_kosdaq"], ud["v_kosdaq"].shift(lag)
            ) - 1.0
        for lag in _INDEX_LAGS:
            out[f"relative_return_{lag}"] = ud["kospi_pct"].shift(lag) - ud["kosdaq_pct"].shift(lag)
        for window in _VIX_WINDOWS:
            out[f"vix_regime_kospi_{window}"] = _safe_divide(
                ud["v_kospi"], ud["v_kospi"].rolling(window, min_periods=window).mean()
            ) - 1.0
        for window in _VIX_WINDOWS:
            out[f"vix_regime_kosdaq_{window}"] = _safe_divide(
                ud["v_kosdaq"], ud["v_kosdaq"].rolling(window, min_periods=window).mean()
            ) - 1.0
        for window in _VIX_WINDOWS:
            out[f"broad_market_ma_dist_{window}"] = _safe_divide(
                level_kospi, level_kospi.rolling(window, min_periods=window).mean()
            ) - 1.0

    frame = pd.DataFrame(out)
    frame.index = ud[date_col].astype("datetime64[ns]").to_numpy()
    frame.index.name = "_market_date"
    return frame


# --- 결정 후보 판넬 횡단면 피처 ---------------------------------------------------------------


def _vol_window_for_lag(lag: int) -> int:
    """결정 판넬 lag 에 매핑되는 realised-vol window 를 반환합니다."""
    for window in _VOL_WINDOWS:
        if window >= lag:
            return window
    return int(_VOL_WINDOWS[-1])


def _build_decision_panel_features(
    panel: pd.DataFrame, config: HistoricalFeatureConfig
) -> pd.DataFrame:
    """결정 후보 판넬에서 횡단면(순위/상호작용) 피처를 계산합니다.

    각 피처는 ``panel`` 의 엄격히 이전(시간축) 값 컬럼을 해당 결정일 그룹 내에서
    순위화해 계산하며, 배치 분할과 무관하게 결정적입니다.
    """
    date_labels = panel[config.decision_date_col]

    def rank_of(column: str) -> pd.Series:
        return _group_rank(panel[column], date_labels)

    def ret_lag(lag: int) -> int:
        return max(lag, 1)

    features: dict[str, pd.Series] = {}
    for lag in _LIQUIDITY_LAGS:
        features[f"volume_pct_rank_{lag}"] = rank_of(f"log_volume_{lag}")
    for lag in _LIQUIDITY_LAGS:
        features[f"value_pct_rank_{lag}"] = rank_of(f"log_value_{lag}")
    for lag in _LIQUIDITY_LAGS:
        features[f"cap_pct_rank_{lag}"] = rank_of(f"log_cap_{lag}")
    rank0_volume = rank_of("log_volume_0")
    rank0_value = rank_of("log_value_0")
    for lag in _LIQUIDITY_CHANGE_LAGS:
        features[f"volume_rank_change_{lag}"] = rank0_volume - rank_of(f"log_volume_{lag}")
    for lag in _LIQUIDITY_CHANGE_LAGS:
        features[f"value_rank_change_{lag}"] = rank0_value - rank_of(f"log_value_{lag}")
    for lag in _INDEX_LAGS:
        features[f"market_rank_{lag}"] = rank_of(f"ret_simple_{ret_lag(lag)}")
    for lag in _RANK_LAGS:
        features[f"rank_ret_{lag}"] = rank_of(f"ret_log_{ret_lag(lag)}")
    for lag in _RANK_LAGS:
        features[f"rank_vol_{lag}"] = rank_of(f"realised_vol_{_vol_window_for_lag(lag)}")
    for lag in _RANK_LAGS:
        features[f"rank_turnover_{lag}"] = rank_of(f"turnover_{lag}")
    for lag in _RANK_LAGS:
        features[f"rank_flow_density_{lag}"] = rank_of(f"foreign_flow_density_{lag}")

    interactions = {
        "int_ret_liquidity": ("ret_log_", "log_value_"),
        "int_flow_price": ("foreign_flow_density_", "ret_log_"),
        "int_ret_vol": ("ret_log_", "realised_vol_"),
        "int_ret_cap": ("ret_log_", "log_cap_"),
        "int_vol_turnover": ("realised_vol_", "turnover_"),
        "int_flow_vol": ("foreign_flow_density_", "realised_vol_"),
    }
    for name, (first_col, second_col) in interactions.items():
        for lag in _INTERACTION_LAGS:
            first = first_col + str(_vol_window_for_lag(lag) if "realised_vol_" in first_col else ret_lag(lag) if "ret_log_" in first_col else lag)
            second = second_col + str(_vol_window_for_lag(lag) if "realised_vol_" in second_col else ret_lag(lag) if "ret_log_" in second_col else lag)
            features[f"{name}_{lag}"] = rank_of(first) * rank_of(second)

    ratios = {
        "ratio_ret_vol": ("ret_log_", "realised_vol_"),
        "ratio_flow_turnover": ("foreign_flow_density_", "turnover_"),
    }
    for name, (first_col, second_col) in ratios.items():
        for lag in _RATIO_LAGS:
            first = first_col + str(_vol_window_for_lag(lag) if "realised_vol_" in first_col else ret_lag(lag) if "ret_log_" in first_col else lag)
            second = second_col + str(_vol_window_for_lag(lag) if "realised_vol_" in second_col else ret_lag(lag) if "ret_log_" in second_col else lag)
            ratio = _safe_divide(rank_of(first), rank_of(second))
            features[f"{name}_{lag}"] = ratio.clip(lower=0.0, upper=20.0)

    ordered = _decision_catalogue_names()
    missing = [name for name in ordered if name not in features]
    if missing:
        raise ValueError(f"decision-panel catalogue features were not computed: {missing[:5]}...")
    return pd.DataFrame(features, index=panel.index, columns=ordered, dtype=np.float32)


# --- 배치 실행/메모리 텔레메트리 --------------------------------------------------------------


def _rss_bytes() -> int:
    """현재 프로세스 RSS (psutil 미사용 시 0)."""
    if not _HAS_PSUTIL:
        return 0
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - telemetry best-effort
        return 0


def _merge_temporal_keys(
    batch_keys: pd.DataFrame,
    hist_batch: pd.DataFrame,
    temporal: pd.DataFrame,
    config: HistoricalFeatureConfig,
) -> pd.DataFrame:
    """배치 종목의 결정 key 를 temporal 피처 판넬과 엄격히 이전으로 조인합니다."""
    symbol_col = config.history_symbol_col
    date_col = config.history_date_col
    dsym = config.decision_symbol_col
    ddate = config.decision_date_col

    # ``_build_temporal_features`` 는 (symbol, date) 정렬 후 index 를 재설정하므로,
    # symbol/date 열도 동일한 정렬 순서로 위치 정렬해 결합합니다.
    hist_sorted = hist_batch.sort_values([symbol_col, date_col]).reset_index(drop=True)
    temporal_panel = temporal.copy()
    temporal_panel[symbol_col] = hist_sorted[symbol_col].to_numpy()
    temporal_panel[date_col] = hist_sorted[date_col].to_numpy()
    temporal_panel = temporal_panel.rename(columns={symbol_col: dsym})

    keys = batch_keys[[dsym, ddate]].copy()
    keys[dsym] = keys[dsym].astype(object)
    keys[ddate] = keys[ddate].astype("datetime64[ns]")
    temporal_panel[dsym] = temporal_panel[dsym].astype(object)
    temporal_panel[date_col] = temporal_panel[date_col].astype("datetime64[ns]")

    keys = keys.sort_values(ddate)
    temporal_panel = temporal_panel.sort_values(date_col)
    merged = pd.merge_asof(
        keys,
        temporal_panel,
        left_on=ddate,
        right_on=date_col,
        by=dsym,
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_history"),
    )
    merged = merged.rename(columns={date_col: "_history_date"})
    if f"{dsym}_history" in merged.columns:
        merged = merged.drop(columns=[f"{dsym}_history"])
    return merged


def _resolve_batches(
    symbols: list[str],
    symbol_row_counts: dict[str, int],
    read_batch: Callable[[Sequence[str]], pd.DataFrame],
    exec_cfg: HistoryFeatureExecutionConfig,
    config: HistoricalFeatureConfig,
) -> tuple[list[list[str]], float]:
    """배치 분할을 결정합니다. 자동 모드는 pilot 측정으로 행당 RSS 를 추정합니다."""
    if exec_cfg.symbols_per_batch is not None:
        n = exec_cfg.symbols_per_batch
        return (
            [symbols[i : i + n] for i in range(0, len(symbols), n)],
            0.0,
        )
    if exec_cfg.memory_budget_bytes is None:
        return [symbols], 0.0

    budget = exec_cfg.memory_budget_bytes
    pilot = symbols[: min(8, len(symbols))]
    rss_before = _rss_bytes()
    pilot_df = read_batch(pilot)
    pilot_rows = len(pilot_df)
    _build_temporal_features(pilot_df, config)
    del pilot_df
    gc.collect()
    rss_after = _rss_bytes()
    delta = max(0, rss_after - rss_before)
    per_row = delta / pilot_rows if pilot_rows > 0 else 0.0
    if per_row <= 0:
        # RSS 가 측정되지 않으면 시간축 float32 행렬 하한(컬럼 수 * 4 바이트)을
        # 보수적 추정치로 사용해 예산 사전 확인을 결정적으로 유지합니다.
        per_row = float(len(_temporal_catalogue_names()) * 4)

    baseline_rss = rss_before
    min_projected = baseline_rss + per_row * min(symbol_row_counts.get(s, 0) for s in symbols)
    if min_projected > budget:
        raise ValueError(
            "memory budget too small for streaming feature build: projected peak "
            f"{min_projected:.0f} bytes (baseline RSS plus a single-symbol batch) exceeds "
            f"memory_budget_bytes={budget}; raise the budget or disable enforcement"
        )
    batches: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0.0
    for symbol in symbols:
        row_bytes = per_row * symbol_row_counts.get(symbol, 0)
        if current and current_bytes + row_bytes > budget:
            batches.append(current)
            current = [symbol]
            current_bytes = row_bytes
        else:
            current.append(symbol)
            current_bytes += row_bytes
    if current:
        batches.append(current)
    return batches, per_row


def _build_panel_core(
    read_batch: Callable[[Sequence[str]], pd.DataFrame],
    symbol_row_counts: dict[str, int],
    market_dates_df: pd.DataFrame,
    keys: pd.DataFrame,
    config: HistoricalFeatureConfig,
    exec_cfg: HistoryFeatureExecutionConfig,
    input_history_rows: int,
) -> pd.DataFrame:
    """배치 단위로 시간축 피처를 계산하고 결정 key 판넬로 축소해 최종 판넬을 반환합니다."""
    dsym = config.decision_symbol_col
    ddate = config.decision_date_col

    started = time.perf_counter()
    symbols = list(keys[dsym].unique())
    if not symbols:
        # 결정 key 가 없으면 빈 판넬을 계약에 맞춰 반환합니다.
        empty_cols = [dsym, ddate, *_catalogue_feature_names()]
        empty = pd.DataFrame(columns=empty_cols)
        empty = empty.astype({dsym: object, ddate: "datetime64[ns]"})
        metrics = HistoryFeatureBuildMetrics(
            input_history_rows=input_history_rows,
            decision_key_rows=0,
            output_rows=0,
            batch_count=0,
            estimated_bytes_per_source_row=0.0,
            peak_rss_bytes=_rss_bytes(),
            elapsed_seconds=float(time.perf_counter() - started),
            nonfinite_to_nan_count=0,
        )
        empty.attrs["history_feature_build_metrics"] = metrics.model_dump()
        return empty
    batches, estimated_bytes_per_source_row = _resolve_batches(
        symbols, symbol_row_counts, read_batch, exec_cfg, config
    )

    parts: list[pd.DataFrame] = []
    peak_rss = _rss_bytes()

    def build_batch_part(batch: Sequence[str]) -> pd.DataFrame:
        hist_batch = read_batch(batch)
        temporal = _build_temporal_features(hist_batch, config)
        batch_keys = keys[keys[dsym].isin(batch)]
        matched = _merge_temporal_keys(batch_keys, hist_batch, temporal, config)
        del hist_batch, temporal
        return matched

    if exec_cfg.n_jobs != 1 and len(batches) > 1:
        # 독립 배치를 병렬 계산 (결과 순서는 ``pool.map`` 이 입력 배치 순서를
        # 유지하므로 concat/sort 이후 최종 판넬은 결정적입니다).
        max_workers = exec_cfg.n_jobs if exec_cfg.n_jobs > 0 else (os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            parts = list(pool.map(build_batch_part, batches))
    else:
        for batch in batches:
            parts.append(build_batch_part(batch))
    gc.collect()
    peak_rss = max(peak_rss, _rss_bytes())
    if (
        exec_cfg.enforce_memory_budget
        and exec_cfg.memory_budget_bytes is not None
        and peak_rss > exec_cfg.memory_budget_bytes
    ):
        raise ValueError(
            f"peak RSS {peak_rss} exceeds memory_budget_bytes "
            f"{exec_cfg.memory_budget_bytes}; streaming build failed closed"
        )

    panel = pd.concat(parts, axis=0).sort_values([ddate, dsym]).reset_index(drop=True)
    del parts
    gc.collect()

    market = _build_market_context_frame(market_dates_df, config)
    panel = panel.join(market, on="_history_date", how="left")
    missing_market = [c for c in _market_catalogue_names() if c not in panel.columns]
    if missing_market:
        raise ValueError(f"market context features were not computed: {missing_market[:5]}...")
    panel, _ = _sanitize_finite(panel)

    decision = _build_decision_panel_features(panel, config)
    panel = pd.concat([panel, decision], axis=1)

    final, nonfinite_count = _sanitize_finite(panel)

    # 실패 폐쇄 불변량: key 중복 없음, 엄격히 이전 매칭.
    if final.duplicated(subset=[dsym, ddate]).any():
        raise ValueError("history feature panel contains duplicate decision keys")
    matched_mask = final["_history_date"].notna()
    if matched_mask.any() and not (
        final.loc[matched_mask, "_history_date"] < final.loc[matched_mask, ddate]
    ).all():
        raise ValueError("history feature panel violates strict-prior date invariant")

    out_cols = [dsym, ddate, *_catalogue_feature_names()]
    output = final[out_cols].copy()
    output = output.sort_values([ddate, dsym]).reset_index(drop=True)
    if output.columns.tolist() != out_cols:
        raise ValueError("history feature panel columns do not match the fixed catalogue")
    if len(output.columns) != HISTORICAL_CATALOGUE_COUNT + 2:
        raise ValueError(
            f"history feature panel must contain exactly {HISTORICAL_CATALOGUE_COUNT} "
            f"features, got {len(output.columns) - 2}"
        )

    elapsed = float(time.perf_counter() - started)
    metrics = HistoryFeatureBuildMetrics(
        input_history_rows=input_history_rows,
        decision_key_rows=len(keys),
        output_rows=len(output),
        batch_count=len(batches),
        estimated_bytes_per_source_row=float(estimated_bytes_per_source_row),
        peak_rss_bytes=int(peak_rss),
        elapsed_seconds=elapsed,
        nonfinite_to_nan_count=int(nonfinite_count),
    )
    output.attrs["history_feature_build_metrics"] = metrics.model_dump()
    return output


def _prepare_keys(decision_keys: pd.DataFrame, config: HistoricalFeatureConfig) -> pd.DataFrame:
    """결정 key 를 정규화/중복 제거하고 종목 사전순으로 정렬합니다."""
    dsym = config.decision_symbol_col
    ddate = config.decision_date_col
    if not {ddate, dsym} <= set(decision_keys.columns):
        raise ValueError(
            f"decision_keys must contain {ddate!r} and {dsym!r}"
        )
    keys = decision_keys.copy()
    keys[dsym] = _zfill_symbols(keys[dsym])
    keys[ddate] = pd.to_datetime(keys[ddate])
    keys = (
        keys.sort_values([dsym, ddate])
        .drop_duplicates(subset=[dsym, ddate], keep="first")
        .reset_index(drop=True)
    )
    keys = keys[[dsym, ddate]]
    keys[dsym] = keys[dsym].astype(object)
    return keys


def _prepare_history_frame(price_history: pd.DataFrame, config: HistoricalFeatureConfig) -> pd.DataFrame:
    """DataFrame 입력을 정규화하고 필요한 원천 컬럼을 검증합니다."""
    symbol_col = config.history_symbol_col
    date_col = config.history_date_col
    missing = [col for col in [symbol_col, date_col, *REQUIRED_HISTORY_COLUMNS] if col not in price_history.columns]
    if missing:
        raise ValueError(f"price_history missing required columns: {missing}")
    hist = price_history.copy()
    hist[symbol_col] = _zfill_symbols(hist[symbol_col])
    hist[date_col] = pd.to_datetime(hist[date_col])
    hist = (
        hist.sort_values([symbol_col, date_col])
        .drop_duplicates(subset=[symbol_col, date_col], keep="first")
        .reset_index(drop=True)
    )
    return hist


def build_causal_history_feature_panel(
    price_history: pd.DataFrame,
    decision_keys: pd.DataFrame,
    config: HistoricalFeatureConfig | None = None,
    execution_config: HistoryFeatureExecutionConfig | None = None,
) -> pd.DataFrame:
    """결정 key 별 causal history 피처 판넬을 반환합니다 (bounded DataFrame 입력).

    Args:
        price_history: EOD 판넬 (``history_date_col``/``history_symbol_col`` 및
            OHLCV/시총/거래대금/수급/지수 컬럼 포함). 메모리 예산 설정 없이
            제한된 판넬을 전달하면 단일 배치로 동작합니다.
        decision_keys: ``decision_symbol_col``/``decision_date_col`` 을 포함한
            결정 key DataFrame. 중복은 첫 행으로 축소됩니다.
        config: 카탈로그/컬럼 이름 설정.
        execution_config: streaming 배치/메모리 예산 설정.

    Returns:
        ``[stock_code, trade_date, <720 카탈로그 피처>]`` 컬럼의 DataFrame.
        이력이 없는 key 의 피처는 ``NaN`` 입니다. 빌드 지표는
        ``attrs["history_feature_build_metrics"]`` 에 저장됩니다.
    """
    config = config or HistoricalFeatureConfig()
    exec_cfg = execution_config or HistoryFeatureExecutionConfig()
    keys = _prepare_keys(decision_keys, config)
    hist = _prepare_history_frame(price_history, config)
    symbol_col = config.history_symbol_col

    symbol_row_counts = hist.groupby(symbol_col, sort=False).size().to_dict()

    def read_batch(symbols: Sequence[str]) -> pd.DataFrame:
        return hist[hist[symbol_col].isin(symbols)]

    market_dates_df = hist[[config.history_date_col, "kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq"]]
    return _build_panel_core(
        read_batch=read_batch,
        symbol_row_counts=symbol_row_counts,
        market_dates_df=market_dates_df,
        keys=keys,
        config=config,
        exec_cfg=exec_cfg,
        input_history_rows=len(hist),
    )


def build_causal_history_feature_panel_from_parquet(
    history_path: str,
    decision_keys: pd.DataFrame,
    config: HistoricalFeatureConfig | None = None,
    execution_config: HistoryFeatureExecutionConfig | None = None,
) -> pd.DataFrame:
    """결정 key 별 causal history 피처 판넬을 Parquet 경로에서 streaming 으로 반환합니다.

    PyArrow 컬럼 투영 + symbol predicate pushdown(row-group pruning)을 사용해
    필요한 원천 컬럼과 해당 배치 종목만 읽습니다. 전체 판넬을 ``pd.read_parquet``
    로 메모리에 올리지 않습니다.
    """
    import pyarrow.parquet as pq

    config = config or HistoricalFeatureConfig()
    exec_cfg = execution_config or HistoryFeatureExecutionConfig()
    keys = _prepare_keys(decision_keys, config)
    symbol_col = config.history_symbol_col
    date_col = config.history_date_col

    parquet_file = pq.ParquetFile(history_path)
    symbol_counts: dict[str, int] = {}
    input_history_rows = 0
    for batch in parquet_file.iter_batches(
        columns=[symbol_col], batch_size=exec_cfg.parquet_batch_rows
    ):
        symbols_batch = (
            batch.column(0).to_pandas().astype("string").str.strip().str.zfill(6)
        )
        input_history_rows += len(symbols_batch)
        for symbol, count in symbols_batch.value_counts(dropna=False).items():
            if pd.isna(symbol):
                continue
            symbol_counts[str(symbol)] = symbol_counts.get(str(symbol), 0) + int(count)

    market_parts: list[pd.DataFrame] = []
    market_columns = [date_col, "kospi_pct", "kosdaq_pct", "v_kospi", "v_kosdaq"]
    for batch in parquet_file.iter_batches(
        columns=market_columns, batch_size=exec_cfg.parquet_batch_rows
    ):
        market_batch = batch.to_pandas()
        market_batch[date_col] = pd.to_datetime(market_batch[date_col])
        market_parts.append(_aggregate_market_dates(market_batch, date_col))
    market_dates_df = (
        pd.concat(market_parts, ignore_index=True)
        .pipe(_aggregate_market_dates, date_col)
    )
    del market_parts, parquet_file

    columns = list(exec_cfg.parquet_columns)
    if date_col not in columns:
        columns.append(date_col)
    if symbol_col not in columns:
        columns.append(symbol_col)

    def read_batch(symbols: Sequence[str]) -> pd.DataFrame:
        table = pq.read_table(
            history_path,
            columns=columns,
            filters=[(symbol_col, "in", list(symbols))],
        )
        df = table.to_pandas()
        df[symbol_col] = _zfill_symbols(df[symbol_col])
        df[date_col] = pd.to_datetime(df[date_col])
        return (
            df.sort_values([symbol_col, date_col])
            .drop_duplicates(subset=[symbol_col, date_col], keep="first")
            .reset_index(drop=True)
        )

    return _build_panel_core(
        read_batch=read_batch,
        symbol_row_counts=symbol_counts,
        market_dates_df=market_dates_df,
        keys=keys,
        config=config,
        exec_cfg=exec_cfg,
        input_history_rows=input_history_rows,
    )


def build_catalogue_manifest() -> pd.DataFrame:
    """720 컬럼 카탈로그의 소스/패밀리/lookback/변환/가용성 매니페스트를 반환합니다."""
    rows = [
        {
            "feature_name": str(entry["feature_name"]),
            "source_column": str(entry["source_field"]),
            "availability_rule": "prior_eod_available_at_decision_time",
            "unit": _catalogue_unit(str(entry["feature_name"])),
            "panel_scope": _feature_panel_scope(str(entry["transform"])),
            "family": str(entry["family"]),
            "lookback": str(entry["lookback"]),
            "transform": str(entry["transform"]),
            "catalogue_version": HISTORICAL_CATALOGUE_VERSION,
        }
        for entry in HISTORICAL_CATALOGUE
    ]
    return pd.DataFrame(rows)


def _catalogue_unit(feature: str) -> str:
    """카탈로그 피처의 단위 라벨을 결정적으로 반환합니다."""
    if feature.endswith("_pct_rank") or feature.startswith(("rank_", "market_rank")):
        return "pct_rank"
    if feature.startswith("log_"):
        return "log_level"
    if feature.endswith("_zscore") or "_zscore_" in feature:
        return "robust_z"
    return "decimal_ratio"


def catalogue_availability_overrides() -> dict[str, dict[str, str]]:
    """feature_manifest 연동용 source/availability 오버라이드 매핑을 반환합니다."""
    return {
        str(entry["feature_name"]): {
            "source_column": str(entry["source_field"]),
            "availability_rule": "prior_eod_available_at_decision_time",
            "panel_scope": _feature_panel_scope(str(entry["transform"])),
        }
        for entry in HISTORICAL_CATALOGUE
    }


def catalogue_quality_metadata() -> dict[str, dict[str, str]]:
    """causal_history_v2 품질 리포트용 피처별 메타데이터를 반환합니다.

    ``family``, ``source_column``, ``transform``, ``lookback``,
    ``availability_rule``, ``panel_scope`` 를 포함하며, 카탈로그에 피처가 정확히
    ``HISTORICAL_CATALOGUE_COUNT`` 개 있는지 fail-closed 로 검증합니다.
    """
    metadata = {
        str(entry["feature_name"]): {
            "family": str(entry["family"]),
            "source_column": str(entry["source_field"]),
            "transform": str(entry["transform"]),
            "lookback": str(entry["lookback"]),
            "availability_rule": "prior_eod_available_at_decision_time",
            "panel_scope": _feature_panel_scope(str(entry["transform"])),
        }
        for entry in HISTORICAL_CATALOGUE
    }
    if len(metadata) != HISTORICAL_CATALOGUE_COUNT:
        raise ValueError(
            f"catalogue quality metadata must contain exactly "
            f"{HISTORICAL_CATALOGUE_COUNT} features, got {len(metadata)}"
        )
    return metadata
