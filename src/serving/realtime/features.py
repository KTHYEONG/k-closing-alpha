"""Snapshot-only feature normalization and derivation for the live serving path.

This module retains the exact column mappings, ``engineer_features``
transformations, and cross-sectional robust-z semantics that the published
bundle was trained against. It never constructs targets, OOF panels,
historical panels, availability-provenance promotion checks, or research-only
feature sets — those live under ``legacy/ml_research/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.research.v3_engine import compute_derived_features
from src.strategy.contract import derive_chg_ratio, tick_cost_bp

# Robust Z-Score ((x - median) / MAD) 횡단면 표준화 대상
_ROBUST_Z_COLUMNS: tuple[str, ...] = (
    "change_rate",
    "buy_price_change_rate",
    "gap_ratio",
    "major_density",
    "prog_dominance",
    "turnover",
    "inst_density",
    "foreign_density",
)

_LOG_AMOUNT_COLUMNS: tuple[str, ...] = (
    "market_cap_100m",
    "trade_value_100m",
    "volume",
    "avg_trade_value",
)

_SIGNED_LOG_COLUMNS: tuple[str, ...] = (
    "inst_net_buy",
    "foreign_net_buy",
    "prog_net_buy",
)

_PCT_RANK_COLUMNS: dict[str, str] = {
    "trade_value_100m": "trade_value_pct_rank",
    "inst_net_buy": "inst_net_buy_pct_rank",
    "foreign_net_buy": "foreign_net_buy_pct_rank",
    "change_rate": "change_rate_pct_rank",
    "major_density": "major_density_pct_rank",
    "prog_dominance": "prog_dominance_pct_rank",
    "gap_ratio": "gap_ratio_pct_rank",
    "turnover": "turnover_pct_rank",
}

# production_calendar_flow 후보: 요일 one-hot 지표 (engineer_features 내에서 생성).
_WEEKDAY_INDICATOR_FEATURES: tuple[str, ...] = (
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
)

# flow_consensus / flow_alignment_direction 의 원천이 되는 단일 소스 밀도 컬럼.
_PRODUCTION_FLOW_SOURCE_COLUMNS: tuple[str, ...] = (
    "inst_density",
    "foreign_density",
    "prog_dominance",
)

SCENARIO_ONE_HOT_FEATURES: tuple[str, ...] = (
    "scenario_is_sangtta",
    "scenario_is_120_breakout",
    "scenario_is_volume_surge",
    "scenario_is_new_high",
    "scenario_is_near_new_high",
    "scenario_is_limitup_next_day",
    "scenario_is_rising_bearish",
    "scenario_other",
)

SCENARIO_CONTEXT_FEATURES: tuple[str, ...] = (
    "scenario_count_for_stock_date",
    "has_sangtta_for_stock_date",
    "is_multi_scenario_stock_date",
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """로그 스케일링, 상대 비율, 횡단면 백분위/robust-z 피처를 생성합니다."""
    df = df.copy()

    prev_close = df["prev_close_price"].replace(0, np.nan)
    # BUG-1: buy_price_change_rate/gap_ratio를 % 단위로 통일 (kospi/kosdaq_change와 동일 스케일)
    df["buy_price_change_rate"] = (df["buy_price"] - df["prev_close_price"]) / prev_close * 100
    df["gap_ratio"] = (df["open_price"] - df["prev_close_price"]) / prev_close * 100
    df["intraday_return"] = (df["close_price"] - df["open_price"]) / df["open_price"].replace(
        0, np.nan
    )

    # FEAT-1: 캔들/가격 파생 피처
    candle_range = (df["high_price"] - df["low_price"]).clip(lower=1)
    df["intraday_range"] = (df["high_price"] - df["low_price"]) / prev_close * 100
    df["close_position"] = (df["close_price"] - df["low_price"]) / candle_range
    body_top = np.maximum(df["open_price"], df["close_price"])
    df["upper_shadow_ratio"] = (df["high_price"] - body_top) / candle_range
    df["body_ratio"] = np.abs(df["close_price"] - df["open_price"]) / candle_range
    df["turnover"] = df["trade_value_100m"] / df["market_cap_100m"].clip(lower=0.01)

    trade_value = df["trade_value_100m"].replace(0, np.nan)
    # BUG-5: 수급 NaN 안전 합산 (fillna(0)) + FEAT-2: 개별 수급 밀도
    if "inst_net_buy" in df.columns:
        df["inst_density"] = df["inst_net_buy"].fillna(0) / trade_value
    if "foreign_net_buy" in df.columns:
        df["foreign_density"] = df["foreign_net_buy"].fillna(0) / trade_value
    inst_part = (
        df["inst_net_buy"].fillna(0)
        if "inst_net_buy" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    foreign_part = (
        df["foreign_net_buy"].fillna(0)
        if "foreign_net_buy" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df["major_density"] = (inst_part + foreign_part) / trade_value
    if "prog_net_buy" in df.columns:
        df["prog_dominance"] = df["prog_net_buy"] / trade_value

    # BUG-6: total_candidate_count NaN → fillna(1)
    df["rank_ratio"] = df["selection_rank"] / df["total_candidate_count"].fillna(1).clip(lower=1)

    market_ref = np.where(
        df["market_type"].astype(str).str.upper().str.contains("KOSDAQ", na=False),
        df["kosdaq_change"],
        df["kospi_change"],
    )
    df["relative_change_rate"] = df["buy_price_change_rate"] - market_ref

    # FEAT-3: KOSPI/KOSDAQ 독립 상대강도 + 섹터 내 상대강도
    df["relative_change_kospi"] = df["change_rate"] - df["kospi_change"]
    df["relative_change_kosdaq"] = df["change_rate"] - df["kosdaq_change"]
    if "theme_sector" in df.columns:
        sector_mean = df.groupby(["trade_date", "theme_sector"])["change_rate"].transform("mean")
        df["sector_relative_change"] = df["change_rate"] - sector_mean

    # FEAT-4: V-KOSPI/V-KOSDAQ 0값을 NaN 처리 후 날짜별 대표값 보간 + 변화율
    for vix_col in ("v_kospi", "v_kosdaq"):
        if vix_col not in df.columns:
            continue
        df[vix_col] = df[vix_col].replace(0, np.nan)
        daily_ref = df.groupby("trade_date")[vix_col].first().ffill().bfill()
        df[vix_col] = df["trade_date"].map(daily_ref)
        daily_change = daily_ref.pct_change().fillna(0)
        df[f"{vix_col}_change"] = df["trade_date"].map(daily_change).fillna(0)

    # BUG-7: log 변환 시 원본 컬럼 유지 + log_ 접두사 파생 컬럼 생성
    for col in _LOG_AMOUNT_COLUMNS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    for col in _SIGNED_LOG_COLUMNS:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    # BUG-4: pct-rank 대상 컬럼 존재 검사 후 생성
    for src_col, dst_col in _PCT_RANK_COLUMNS.items():
        if src_col in df.columns:
            df[dst_col] = df.groupby("trade_date")[src_col].rank(pct=True)

    # P1(ml_internal_panel_enhancement): interaction53 상호작용 피처.
    # 모두 기존 값의 벡터 연산이며, 분모 0 은 NaN 으로 안전 처리 후
    # [-5, 5] 또는 논리적 범위로 클리핑합니다. base40 X 에서는 제외됩니다.
    df["candle_strength"] = (
        (2 * df["close_position"] - 1) * df["body_ratio"] * df["intraday_range"]
    ).clip(-5, 5)
    df["range_efficiency"] = (
        df["intraday_return"].abs() / np.maximum(df["intraday_range"] / 100, 1e-6)
    ).clip(0, 5)
    df["flow_turnover"] = (df["major_density"] * df["turnover"]).clip(0, 5)
    if "major_density_pct_rank" in df.columns and "change_rate_pct_rank" in df.columns:
        df["relative_flow_strength"] = (
            df["major_density_pct_rank"] * df["change_rate_pct_rank"]
        ).clip(0, 1)

    # production_calendar_flow: 캘린더/수급 흐름 연구 후보 피처.
    # 모두 스냅샷 결정 시점 값의 벡터 연산이며, 행 단위 apply / 미래 행 보간은
    # 허용되지 않습니다. 이용 불가능한 수급은 0 으로 간주해 시그널을 만들지 않습니다.
    trade_date = pd.to_datetime(df["trade_date"])
    weekday_index = trade_date.dt.dayofweek
    for offset, name in enumerate(_WEEKDAY_INDICATOR_FEATURES):
        df[name] = (weekday_index == offset).astype("float64")

    flow_matrix = np.column_stack(
        [
            df[name].fillna(0).to_numpy()
            if name in df.columns
            else np.zeros(len(df))
            for name in _PRODUCTION_FLOW_SOURCE_COLUMNS
        ]
    )
    df["flow_consensus"] = np.sign(flow_matrix).sum(axis=1).astype("float64")
    abs_flow = np.abs(flow_matrix).sum(axis=1)
    # 분모(절대 흐름 합)가 0 인 행은 방향 정렬 0.0 으로 처리합니다.
    df["flow_alignment_direction"] = np.divide(
        flow_matrix.sum(axis=1),
        abs_flow,
        out=np.zeros(len(df)),
        where=abs_flow != 0,
    )
    df["friday_selection_rank_pct"] = df["weekday_is_friday"] * (1 - df["rank_ratio"])

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _apply_robust_z(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """횡단면 Robust Z-Score((x - median) / MAD)를 생성하고 [-5, 5]로 클리핑합니다.

    MAD가 0인 그룹은 0-나눗셈 방지를 위해 NaN으로 처리합니다.
    """
    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            continue
        median = df.groupby("trade_date")[col].transform("median")
        mad = (df[col] - median).abs().groupby(df["trade_date"]).transform("median")
        mad = mad.replace(0, np.nan)
        new_cols[f"{col}_z"] = ((df[col] - median) / mad).clip(-5, 5)
    if new_cols:
        df = df.assign(**new_cols)
    return df


def build_topk_ranker_features(
    df: pd.DataFrame, decision_date: pd.Timestamp, *, price_history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Map a live Korean-column snapshot to v3_engine decision-time features.

    Args:
        df: Live daily snapshot with Korean columns and percent-unit indices.
        decision_date: Decision date stamped onto every row.
        price_history: Strictly-past daily rows (HISTORY_REQUIRED_COLUMNS). When
            given, the snapshot is stitched onto it and the v2 cost/history
            features are attached; this additionally requires 시장구분.

    Returns:
        Frame carrying every v3_engine FEATURE_COLS entry, plus
        TOPK_COST_FEATURE_COLS and TOPK_HISTORY_FEATURE_COLS when price_history
        is given. Row order matches df.

    Raises:
        ValueError: Naming every missing required Korean column, or propagated
            from the history stitching.
    """
    # 결측 입력은 플레이스홀더 없이 즉시 차단
    required = (
        "종목코드",
        "종가",
        "전일종가",
        "고가",
        "저가",
        "시가",
        "거래량",
        "거래대금",
        "시가총액",
        "기관_순매수",
        "외국인_순매수",
        "kospi",
        "kosdaq",
        "v_kospi",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    # collect.py 퍼센트 단위를 소수 분율로 환산 (유일한 단위 변환)
    close = pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = pd.to_numeric(df["전일종가"], errors="coerce").to_numpy(dtype=np.float64)
    mapped = pd.DataFrame({
        "symbol": df["종목코드"].astype(str).to_numpy(),
        "date": pd.Timestamp(decision_date),
        "close": pd.to_numeric(df["종가"], errors="coerce").to_numpy(dtype=np.float64),
        "open": pd.to_numeric(df["시가"], errors="coerce").to_numpy(dtype=np.float64),
        "high": pd.to_numeric(df["고가"], errors="coerce").to_numpy(dtype=np.float64),
        "low": pd.to_numeric(df["저가"], errors="coerce").to_numpy(dtype=np.float64),
        "volume": pd.to_numeric(df["거래량"], errors="coerce").to_numpy(dtype=np.float64),
        "tv_clean": pd.to_numeric(df["거래대금"], errors="coerce").to_numpy(dtype=np.float64),
        "mc_clean": pd.to_numeric(df["시가총액"], errors="coerce").to_numpy(dtype=np.float64),
        "inst_netbuy": pd.to_numeric(df["기관_순매수"], errors="coerce").to_numpy(dtype=np.float64),
        "foreign_netbuy": pd.to_numeric(df["외국인_순매수"], errors="coerce").to_numpy(dtype=np.float64),
        "kospi_pct": pd.to_numeric(df["kospi"], errors="coerce").to_numpy(dtype=np.float64) / 100.0,
        "kosdaq_pct": pd.to_numeric(df["kosdaq"], errors="coerce").to_numpy(dtype=np.float64) / 100.0,
        "v_kospi": pd.to_numeric(df["v_kospi"], errors="coerce").to_numpy(dtype=np.float64),
    })
    mapped["chg_ratio"] = np.asarray(derive_chg_ratio(close, prev_close), dtype=np.float64)
    base = compute_derived_features(mapped)
    if price_history is None:
        return base
    from src.ml.topk_history_features import attach_topk_features, stitch_live_panel

    if "시장구분" not in df.columns:
        raise ValueError("missing required columns: ['시장구분'] (needed for tick-cost feature)")
    # 결정일 PIT 호가단위 비용 (collect.flag_cost_aware_admission 과 동일 산식)
    dates = np.full(len(df), np.datetime64(pd.Timestamp(decision_date).strftime("%Y-%m-%d")))
    base["prev_close"] = prev_close
    base["tick_cost_bp"] = tick_cost_bp(close, dates, df["시장구분"].astype(str).to_numpy(dtype=object))
    live_rows = base[["symbol", "open", "close", "prev_close", "volume", "inst_netbuy", "foreign_netbuy"]]
    panel = stitch_live_panel(price_history, live_rows, decision_date)
    return attach_topk_features(base, panel)
