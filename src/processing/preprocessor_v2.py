"""ML 데이터 전처리 및 피처 엔지니어링 파이프라인 (Parquet 기반).

`docs/specs/ml_data_preprocessing.md` 명세를 구현합니다.
`trade_log.parquet` / `theme.parquet` 원본 컬럼명을 표준 식별자로 정규화하고,
랭킹/회귀/분류 multi-task 학습용 타깃 3종과 횡단면 상대 피처를 생성합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 원본 컬럼명 -> 정규화 컬럼명 1:1 매핑 (스프레드시트 특수문자/단위 제거)
COLUMN_MAP: dict[str, str] = {
    "매수날짜": "trade_date",
    "종목코드": "stock_code",
    "(시가)": "open_price",
    "(고가)": "high_price",
    "(저가)": "low_price",
    "(종가)": "close_price",
    "(전일종가)": "prev_close_price",
    "(시가총액, 억)": "market_cap_100m",
    "(거래대금, 억)": "trade_value_100m",
    "(등락률)": "change_rate",
    "(선정 순위)": "selection_rank",
    "(기관_순매수)": "inst_net_buy",
    "(외국인_순매수)": "foreign_net_buy",
    "(프로그램_순매수)": "prog_net_buy",
    "(체결강도)": "volume_power",
    "(시장구분)": "market_type",
    "(총 종목 수)": "total_candidate_count",
    "(평균 거래대금)": "avg_trade_value",
    "(kospi, %)": "kospi_change",
    "(kosdaq, %)": "kosdaq_change",
    "v_kospi": "v_kospi",
    "v_kosdaq": "v_kosdaq",
    "(거래량)": "volume",
    "(테마/섹터)": "theme_sector",
    "(차트분석)": "chart_analysis",
    "(매수 가격)": "buy_price",
    "(매도 가격)": "sell_price",
    "(수익률, %)": "net_return",
}

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
}

_CATEGORICAL_COLUMNS: tuple[str, ...] = ("market_type", "theme_sector", "chart_analysis")

# 학습 피처 집합(X)에서 완전 격리하는 메타데이터/미래 정보 컬럼
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
    "net_return",
    "target_return",
    "target_rank",
    "target_good",
    "target_bad",
}

_TARGET_NAMES: tuple[str, ...] = ("target_return", "target_rank", "target_good", "target_bad")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼명을 표준 snake_case 식별자로 1:1 매핑 정규화합니다."""
    df = df.rename(columns=COLUMN_MAP)

    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "stock_code" in df.columns:
        df["stock_code"] = (
            df["stock_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        )
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """로그 스케일링, 상대 비율, 횡단면 백분위 피처를 생성합니다."""
    df = df.copy()

    prev_close = df["prev_close_price"].replace(0, np.nan)
    df["buy_price_change_rate"] = (df["buy_price"] - df["prev_close_price"]) / prev_close
    df["gap_ratio"] = (df["open_price"] - df["prev_close_price"]) / prev_close
    df["intraday_return"] = (df["close_price"] - df["open_price"]) / df["open_price"].replace(
        0, np.nan
    )

    trade_value = df["trade_value_100m"].replace(0, np.nan)
    df["major_density"] = (df["inst_net_buy"] + df["foreign_net_buy"]) / trade_value
    df["prog_dominance"] = df["prog_net_buy"] / trade_value
    df["rank_ratio"] = df["selection_rank"] / df["total_candidate_count"].clip(lower=1)

    market_ref = np.where(
        df["market_type"].astype(str).str.upper().str.contains("KOSDAQ", na=False),
        df["kosdaq_change"],
        df["kospi_change"],
    )
    df["relative_change_rate"] = df["buy_price_change_rate"] - market_ref

    for col in _LOG_AMOUNT_COLUMNS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    for col in _SIGNED_LOG_COLUMNS:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    for src_col, dst_col in _PCT_RANK_COLUMNS.items():
        df[dst_col] = df.groupby("trade_date")[src_col].rank(pct=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def create_multi_targets(df: pd.DataFrame) -> pd.DataFrame:
    """회귀/랭킹/분류 3종 타깃 변수를 생성합니다."""
    df = df.copy()
    df["target_return"] = df["net_return"].clip(-10.0, 10.0)

    def assign_daily_rank(group_df: pd.DataFrame) -> pd.Series:
        if len(group_df) < 5:
            ranks = group_df["net_return"].rank(method="min", ascending=True)
            return ((ranks - 1) / len(group_df) * 5).astype(int).clip(0, 4)
        return pd.qcut(
            group_df["net_return"].rank(method="first"), q=5, labels=[0, 1, 2, 3, 4]
        ).astype(int)

    # pandas 2.x의 groupby.apply는 그룹이 단일 날짜뿐일 때 Series가 아닌
    # DataFrame을 반환해 target_rank 할당이 깨집니다. 그룹별 명시 매핑으로
    # 그룹 수와 무관하게 결정적으로 결과를 보장합니다.
    rank_map: dict[int, int] = {}
    for _, group_df in df.groupby("trade_date", sort=False):
        rank_map.update(assign_daily_rank(group_df).to_dict())
    df["target_rank"] = pd.Series(rank_map, dtype="int64").reindex(df.index)
    df["target_good"] = (df["net_return"] >= 1.0).astype(int)
    df["target_bad"] = (df["net_return"] <= -2.0).astype(int)
    return df


def build_ml_dataset(
    trade_log_df: pd.DataFrame, theme_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str], pd.DataFrame]:
    """매매일지 원본 데이터를 정제하여 (X, targets, cat_features, processed_df)를 반환합니다."""
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

    df = df.dropna(subset=["net_return"]).copy()
    df = engineer_features(df)
    df = create_multi_targets(df)

    for col in _CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    cat_features = [col for col in _CATEGORICAL_COLUMNS if col in df.columns]
    targets = {name: df[name] for name in _TARGET_NAMES}
    feature_cols = [col for col in df.columns if col not in _EXCLUDED_FROM_X]
    X = df[feature_cols].copy()
    return X, targets, cat_features, df
