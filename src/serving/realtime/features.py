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

from src.processing.schema import normalize_column_names

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
    for col in columns:
        if col not in df.columns:
            continue
        grouped = df.groupby("trade_date")[col]
        median = grouped.transform("median")
        mad = grouped.transform(lambda x: (x - x.median()).abs().median())
        mad = mad.replace(0, np.nan)
        df[f"{col}_z"] = ((df[col] - median) / mad).clip(-5, 5)
    return df


def build_snapshot_features(
    df: pd.DataFrame,
    decision_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """당일 스냅샷을 로드된 번들의 표준 ML 피처 스키마로 정규화합니다.

    ``normalize_column_names`` 단일 정규화로 일일 CSV 의 한글/괄호 헤더를 표준
    영문 컬럼으로 변환한 뒤 ``engineer_features`` / ``_apply_robust_z`` 를
    적용합니다. 당일 스냅샷에는 존재하지 않는 ``trade_date``(결정 시각)를
    보강합니다. ``buy_price`` 가 없으면 이전 종가 중립값 대신 유한 양수
    ``close_price`` 로 대체합니다(학습 대비 서빙 피처 정합성). 명시적으로 공급된
    ``buy_price`` 는 변경하지 않습니다. 표시용 메타데이터(``종목명`` 등)는
    보존합니다. ``decision_date`` 가 주어지면 ``trade_date`` 로 사용해 결정적
    테스트를 가능하게 하며, 프로덕션은 세션 날짜를 전달합니다.
    """
    work = df.copy()
    work = normalize_column_names(work)
    if "trade_date" not in work.columns:
        work["trade_date"] = (
            decision_date if decision_date is not None else pd.Timestamp.today().normalize()
        )
    if "buy_price" not in work.columns:
        if "close_price" not in work.columns:
            raise ValueError(
                "buy_price is absent and close_price is missing; cannot derive buy_price"
            )
        close_price = pd.to_numeric(work["close_price"], errors="coerce")
        if close_price.isna().any() or not np.isfinite(
            close_price.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "buy_price is absent and close_price is non-finite; cannot derive buy_price"
            )
        if (close_price <= 0.0).any():
            raise ValueError(
                "buy_price is absent and close_price is non-positive; cannot derive buy_price"
            )
        work["buy_price"] = close_price
    work = engineer_features(work)
    return _apply_robust_z(work, _ROBUST_Z_COLUMNS)
