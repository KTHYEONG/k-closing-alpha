"""ML 데이터 전처리 및 피처 엔지니어링 파이프라인 (Parquet 기반).

`docs/specs/ml_data_preprocessing.md` 명세를 구현합니다.
`trade_log.parquet` / `theme.parquet` 원본 컬럼명을 표준 식별자로 정규화하고,
랭킹/회귀/분류 multi-task 학습용 타깃 3종과 횡단면 상대 피처를 생성합니다.

v1 전처리기의 구형 컬럼/매핑 상수(`RENAME_MAP`, `DATE_COL`)는
`src.processing.legacy_mapping`에서 re-export하여 하위 호환성을 보장합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.legacy_mapping import DATE_COL, RENAME_MAP  # noqa: F401  (하위 호환성 re-export)

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
    "major_density": "major_density_pct_rank",
    "prog_dominance": "prog_dominance_pct_rank",
    "gap_ratio": "gap_ratio_pct_rank",
    "turnover": "turnover_pct_rank",
}

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

_CATEGORICAL_COLUMNS: tuple[str, ...] = ("market_type", "theme_sector", "chart_analysis")

# 학습 피처 집합(X)에서 완전 격리하는 메타데이터/미래 정보 컬럼
_EXCLUDED_FROM_X: set[str] = {
    # 메타/식별자
    "trade_date",
    "stock_code",
    # 미래 정보 (매수 시점 미확정 가격)
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "prev_close_price",
    "buy_price",
    "sell_price",
    # Data Leakage 방지
    "intraday_return",
    # 원본 금액/거래량 컬럼 (log_ 접두사 파생 컬럼만 학습 피처로 사용)
    "market_cap_100m",
    "trade_value_100m",
    "volume",
    "avg_trade_value",
    # 타깃 변수
    "net_return",
    "target_return",
    "target_rank",
    "target_good",
    "target_bad",
}

_TARGET_NAMES: tuple[str, ...] = ("target_return", "target_rank", "target_good", "target_bad")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼명을 표준 snake_case 식별자로 1:1 매핑 정규화하고 문자열 수치 피처를 정제합니다."""
    df = df.rename(columns=COLUMN_MAP)

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

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def create_multi_targets(df: pd.DataFrame) -> pd.DataFrame:
    """회귀/랭킹/분류 3종 타깃 변수를 생성합니다."""
    df = df.copy()
    df["target_return"] = df["net_return"].clip(-10.0, 10.0)

    def assign_daily_rank(group_df: pd.DataFrame) -> pd.Series:
        n = len(group_df)
        if n < 5:
            ranks = group_df["net_return"].rank(method="first", ascending=True)
            if n == 1:
                return pd.Series(2, index=group_df.index)  # 단일 종목: 중립 등급
            return ((ranks - 1) / (n - 1) * 4).round().astype(int).clip(0, 4)
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

def _apply_robust_z(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """횡단면 Robust Z-Score((x - median) / MAD)를 생성하고 [-5, 5]로 클리핑합니다.

    MAD가 0인 그룹은 0-나눗셈 방지를 위해 NaN으로 처리합니다.
    """
    for col in columns:
        if col not in df.columns:
            continue
        grouped = df.groupby("trade_date")[col]
        median = grouped.transform("median")
        mad = grouped.transform(lambda x: (x - x.median()).abs().median())
        mad = mad.replace(0, np.nan)
        df[f"{col}_z"] = ((df[col] - median) / mad).clip(-5, 5)
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
    df = _apply_robust_z(df, _ROBUST_Z_COLUMNS)
    df = create_multi_targets(df)

    for col in _CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    cat_features = [col for col in _CATEGORICAL_COLUMNS if col in df.columns]
    targets = {name: df[name] for name in _TARGET_NAMES}
    feature_cols = [col for col in df.columns if col not in _EXCLUDED_FROM_X]
    X = df[feature_cols].copy()
    return X, targets, cat_features, df
