"""Dataset building for champion pipeline."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ml.feature_manifest import build_feature_manifest
from src.ml.scenario_panel import build_scenario_action_panel
from src.processing.schema import RAW_TO_STANDARD_MAP
from src.serving.realtime.features import (
    _ROBUST_Z_COLUMNS,
    _WEEKDAY_INDICATOR_FEATURES,
    _apply_robust_z,
    engineer_features,
)
from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO

logger = logging.getLogger(__name__)

RETURN_UNIT = "decimal_net"
LABEL_THRESHOLDS: dict[str, float] = {"target_good": 0.01, "target_bad": -0.02}
RETURN_CLIP_LOWER = -0.10
RETURN_CLIP_UPPER = 0.10

_ALLOWED_FEATURE_SETS: tuple[str, ...] = (
    "base40",
    "snapshot49",
    "interaction53",
    "production_calendar_flow",
    "close_morning61",
    "close_morning_history",
    "close_morning_sector",
)
_ALLOWED_PANEL_MODES: tuple[str, ...] = ("raw_rows", "scenario_action")

_SNAPSHOT49_FEATURES: tuple[str, ...] = (
    "close_position",
    "body_ratio",
    "upper_shadow_ratio",
    "intraday_range",
    "buy_price_change_rate",
    "gap_ratio",
    "relative_change_rate",
    "buy_price_change_rate_z",
    "gap_ratio_z",
)

_INTERACTION53_FEATURES: tuple[str, ...] = (
    "candle_strength",
    "range_efficiency",
    "flow_turnover",
    "relative_flow_strength",
)

_CLOSE_MORNING61_FEATURES: tuple[str, ...] = ("relative_flow_strength",)

_PRODUCTION_CALENDAR_FLOW_FEATURES: tuple[str, ...] = (
    *_WEEKDAY_INDICATOR_FEATURES,
    "flow_consensus",
    "flow_alignment_direction",
    "flow_turnover",
    "friday_selection_rank_pct",
)

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "prev_close_price",
    "market_cap_100m",
    "trade_value_100m",
    "change_rate",
    "selection_rank",
    "inst_net_buy",
    "foreign_net_buy",
    "prog_net_buy",
    "volume_power",
    "total_candidate_count",
    "avg_trade_value",
    "kospi_change",
    "kosdaq_change",
    "v_kospi",
    "v_kosdaq",
    "volume",
    "buy_price",
    "sell_price",
    "net_return",
)

_CATEGORICAL_COLUMNS: tuple[str, ...] = ("market_type", "theme_sector", "chart_analysis")

_EXCLUDED_FROM_X: set[str] = {
    "trade_date",
    "stock_code",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "prev_close_price",
    "buy_price",
    "sell_price",
    "intraday_return",
    "close_position",
    "body_ratio",
    "upper_shadow_ratio",
    "intraday_range",
    "buy_price_change_rate",
    "gap_ratio",
    "relative_change_rate",
    "buy_price_change_rate_z",
    "gap_ratio_z",
    "relative_change_rate_z",
    "candle_strength",
    "range_efficiency",
    "flow_turnover",
    "relative_flow_strength",
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "flow_consensus",
    "flow_alignment_direction",
    "friday_selection_rank_pct",
    "market_cap_100m",
    "trade_value_100m",
    "volume",
    "avg_trade_value",
    "net_return",
    "target_return",
    "target_rank",
    "target_good",
    "target_bad",
    "realized_vol",
    "sector_cluster_id",
}

_TARGET_NAMES: tuple[str, ...] = ("target_return", "target_rank", "target_good", "target_bad")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼명을 표준 snake_case 식별자로 1:1 매핑 정규화하고 문자열 수치 피처를 정제합니다."""
    df = df.rename(columns=RAW_TO_STANDARD_MAP)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "stock_code" in df.columns:
        df["stock_code"] = (
            df["stock_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        )
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def create_multi_targets(
    df: pd.DataFrame,
    clip_lower: float = -0.10,
    clip_upper: float = 0.10,
) -> pd.DataFrame:
    """회귀/랭킹/분류 3종 타깃 변수를 decimal net 기준으로 생성합니다."""
    if clip_lower >= clip_upper:
        raise ValueError(f"clip_lower {clip_lower} must be < clip_upper {clip_upper}")
    df = df.copy()
    gross_decimal = df["net_return"] / 100.0
    net_of_cost = gross_decimal - ROUND_TRIP_COST_RATIO
    df["target_return"] = net_of_cost.clip(clip_lower, clip_upper)

    def assign_daily_rank(group_df: pd.DataFrame) -> pd.Series:
        n = len(group_df)
        if n < 5:
            ranks = group_df["net_return"].rank(method="first", ascending=True)
            if n == 1:
                return pd.Series(2, index=group_df.index)
            return ((ranks - 1) / (n - 1) * 4).round().astype(int).clip(0, 4)
        return pd.qcut(
            group_df["net_return"].rank(method="first"), q=5, labels=[0, 1, 2, 3, 4]
        ).astype(int)

    # Vectorized per-date quintiles: one C-level rank/size pass, then one
    # qcut per distinct group size (identical to per-group qcut on 1..n).
    first_rank = df.groupby("trade_date", sort=False)["net_return"].rank(
        method="first", ascending=True
    )
    group_size = df.groupby("trade_date", sort=False)["net_return"].transform("size")
    rank_out = np.empty(len(df), dtype=np.int64)
    small_mask = (group_size < 5).to_numpy()
    size_arr = group_size.to_numpy()
    rank_arr = first_rank.to_numpy()
    single_mask = small_mask & (size_arr == 1)
    rank_out[single_mask] = 2
    multi_small = small_mask & (size_arr > 1)
    if multi_small.any():
        rank_out[multi_small] = (
            ((rank_arr[multi_small] - 1) / (size_arr[multi_small] - 1) * 4)
            .round()
            .astype(np.int64)
            .clip(0, 4)
        )
    for n_val in np.unique(size_arr[~small_mask]):
        n_int = int(n_val)
        labels = (
            pd.qcut(
                pd.Series(np.arange(1, n_int + 1, dtype=float)),
                q=5,
                labels=[0, 1, 2, 3, 4],
            )
            .astype(int)
            .to_numpy()
        )
        sel = (~small_mask) & (size_arr == n_val)
        rank_out[sel] = labels[(rank_arr[sel] - 1).astype(np.int64)]
    df["target_rank"] = pd.Series(rank_out, index=df.index, dtype="int64").reindex(df.index)
    df["target_good"] = (net_of_cost >= LABEL_THRESHOLDS["target_good"]).astype(int)
    df["target_bad"] = (net_of_cost <= LABEL_THRESHOLDS["target_bad"]).astype(int)
    df.attrs["return_unit"] = RETURN_UNIT
    df.attrs["label_thresholds"] = dict(LABEL_THRESHOLDS)
    return df


def retarget_with_clip(
    processed_df: pd.DataFrame,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    """target_return 만 clip bounds 로 재계산한 복사본을 반환합니다."""
    if "net_return" not in processed_df.columns or "target_return" not in processed_df.columns:
        raise ValueError("retarget_with_clip requires net_return and target_return columns")
    if clip_lower >= clip_upper:
        raise ValueError(f"clip_lower {clip_lower} must be < clip_upper {clip_upper}")
    out = processed_df.copy()
    gross_decimal = out["net_return"] / 100.0
    net_of_cost = gross_decimal - ROUND_TRIP_COST_RATIO
    out["target_return"] = net_of_cost.clip(clip_lower, clip_upper)
    return out


def _validate_close_morning61_feature(df: pd.DataFrame) -> None:
    """champion 피처 relative_flow_strength 무결성 검증."""
    if "relative_flow_strength" not in df.columns:
        raise ValueError(
            "close_morning61 requires source inputs (change_rate and major "
            "investor flow) to compute relative_flow_strength; missing required "
            "source inputs"
        )
    rfs = df["relative_flow_strength"].to_numpy(dtype=np.float64)
    if not np.isfinite(rfs).all() or not np.logical_and(rfs >= 0.0, rfs <= 1.0).all():
        raise ValueError("relative_flow_strength must be finite within [0, 1]")


def build_ml_dataset(
    trade_log_df: pd.DataFrame,
    theme_df: pd.DataFrame | None = None,
    feature_set: str = "close_morning61",
    panel_mode: str = "scenario_action",
    price_history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str], pd.DataFrame]:
    """매매일지 원본 데이터를 정제하여 (X, targets, cat_features, processed_df)를 반환합니다."""
    if feature_set not in _ALLOWED_FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {list(_ALLOWED_FEATURE_SETS)}, got {feature_set!r}"
        )
    if panel_mode not in _ALLOWED_PANEL_MODES:
        raise ValueError(
            f"panel_mode must be one of {list(_ALLOWED_PANEL_MODES)}, got {panel_mode!r}"
        )
    df = clean_column_names(trade_log_df.copy())

    if theme_df is not None and not theme_df.empty:
        if {"종목코드", "테마"}.issubset(theme_df.columns):
            theme_map = theme_df.set_index("종목코드")["테마"]
            if "theme_sector" in df.columns:
                df["theme_sector"] = df["theme_sector"].fillna(df["stock_code"].map(theme_map))
            else:
                df["theme_sector"] = df["stock_code"].map(theme_map)
        if {"종목코드", "시장구분"}.issubset(theme_df.columns):
            market_map = theme_df.set_index("종목코드")["시장구분"]
            if "market_type" in df.columns:
                df["market_type"] = df["market_type"].fillna(df["stock_code"].map(market_map))
            else:
                df["market_type"] = df["stock_code"].map(market_map)

    if "trade_date" not in df.columns or "net_return" not in df.columns:
        raise ValueError("필수 컬럼(trade_date, net_return)이 데이터에 없습니다.")

    if feature_set == "close_morning61" and "change_rate" not in df.columns:
        raise ValueError(
            "close_morning61 requires the price-change source (change_rate) to "
            "compute relative_flow_strength; missing required source inputs"
        )

    if panel_mode == "scenario_action":
        df, scenario_rejects = build_scenario_action_panel(df)
        df.attrs["scenario_action_rejects"] = scenario_rejects

    df = df.dropna(subset=["net_return"]).copy()
    df = engineer_features(df)
    df = _apply_robust_z(df, _ROBUST_Z_COLUMNS)
    df = create_multi_targets(df)

    for col in _CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    cat_features = [col for col in _CATEGORICAL_COLUMNS if col in df.columns]
    targets = {name: df[name] for name in _TARGET_NAMES}
    feature_cols = [col for col in df.columns if col not in _EXCLUDED_FROM_X]
    if feature_set in ("snapshot49", "interaction53"):
        feature_cols.extend(col for col in _SNAPSHOT49_FEATURES if col in df.columns)
    if feature_set == "interaction53":
        feature_cols.extend(col for col in _INTERACTION53_FEATURES if col in df.columns)
    if feature_set == "production_calendar_flow":
        feature_cols.extend(
            col for col in _PRODUCTION_CALENDAR_FLOW_FEATURES if col in df.columns
        )
    if feature_set == "close_morning61":
        feature_cols.extend(col for col in _SNAPSHOT49_FEATURES if col in df.columns)
        _validate_close_morning61_feature(df)
        feature_cols.extend(
            col for col in _CLOSE_MORNING61_FEATURES if col in df.columns
        )
    if feature_set == "close_morning_history":
        feature_cols.extend(col for col in _SNAPSHOT49_FEATURES if col in df.columns)
        _validate_close_morning61_feature(df)
        feature_cols.extend(col for col in _CLOSE_MORNING61_FEATURES if col in df.columns)
        if price_history_df is None:
            raise ValueError("close_morning_history feature_set requires price_history_df")
        from src.ml.history_features import HISTORY_FEATURE_COLUMNS, attach_history_features

        df = attach_history_features(df, price_history_df, date_col="trade_date", code_col="stock_code")
        feature_cols.extend(col for col in HISTORY_FEATURE_COLUMNS if col in df.columns)
    if feature_set == "close_morning_sector":
        feature_cols.extend(col for col in _SNAPSHOT49_FEATURES if col in df.columns)
        _validate_close_morning61_feature(df)
        feature_cols.extend(col for col in _CLOSE_MORNING61_FEATURES if col in df.columns)
        if price_history_df is None:
            raise ValueError("close_morning_sector feature_set requires price_history_df")
        from src.ml.history_features import HISTORY_FEATURE_COLUMNS, attach_history_features

        df = attach_history_features(df, price_history_df, date_col="trade_date", code_col="stock_code")
        feature_cols.extend(col for col in HISTORY_FEATURE_COLUMNS if col in df.columns)
        from src.ml.sector_features import SECTOR_FEATURE_COLUMNS, attach_sector_features

        df = attach_sector_features(df, price_history_df, date_col="trade_date", code_col="stock_code", change_col="change_rate")
        feature_cols = [c for c in feature_cols if c != "sector_relative_change"]
        feature_cols.extend(col for col in SECTOR_FEATURE_COLUMNS if col in df.columns)
    if panel_mode == "scenario_action":
        feature_cols = [col for col in feature_cols if col != "chart_analysis"]
        cat_features = [col for col in cat_features if col != "chart_analysis"]
    feature_cols = list(dict.fromkeys(feature_cols))
    x_features = df[feature_cols].copy()
    manifest = build_feature_manifest(feature_cols)
    df.attrs["feature_manifest"] = manifest
    df.attrs["feature_set"] = feature_set
    df.attrs["panel_mode"] = panel_mode
    x_features.attrs["feature_manifest"] = manifest
    x_features.attrs["feature_set"] = feature_set
    x_features.attrs["panel_mode"] = panel_mode
    return x_features, targets, cat_features, df
