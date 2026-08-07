"""Versioned causal feature catalog (``causal_expanded_v1``).

`docs/specs/ml_feature_engineering_evolution.md` 계약을 구현합니다. 고정
61-feature close-to-morning 패널을 버전화된 인과 피처 카탈로그로 확장하며,
다음 원칙을 지킵니다.

- 단일 원천: ``FeatureDefinition`` 은 결정적 이름/가족/원천 컬럼/룩백/패널 범위/
  단위/이용가능 규칙/벡터 계산을 선언하며, 카탈로그는 600--1000 개의 중복 없는
  후보 이름을 생산합니다.
- 결정 시점 정보만: 운영 시트 캡처(``trade_date``/``stock_code``)의 동일 날짜
  필드는 15:18 캡처 계약 하에 허용되고, 외부 가격/수급 이력은 ``date < trade_date``
  인 strict prior-date 행만 as-of 조인에 사용됩니다 (동일 날짜 EOD/백필 행 제외,
  forward-fill 금지).
- 타깃/미래 정보 금지: ``net_return``/``sell_price``/타깃 파생 컬럼을 생성하지
  않습니다. 동일 날짜 OHLC 는 15:18 운영 시트에서만 허용되며, 외부 EOD 원천의
  완성 OHLC 는 사용하지 않습니다.
- 벡터화: 핫 경로는 NumPy/Pandas 벡터 연산만 사용하고 ``DataFrame.apply`` 는
  금지합니다. 비율 피처는 명시적 0 분모 마스크를 사용해 출력이 ``NaN`` 이거나
  유한하게 만듭니다 (``inf`` 금지).
- fail-closed: 이력이 없거나 선언된 룩백을 평가할 수 없으면 ``ValueError`` 로
  실패합니다. 레거시 피처셋 입력/동작은 변경하지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 인과 카탈로그 후보 수 불변량 (fold-local selector 와 공유).
MIN_CANDIDATES = 600
MAX_CANDIDATES = 1000

SUPPORTED_CATALOG_VERSIONS: tuple[str, ...] = ("causal_expanded_v1",)

# 표준화된 가격 이력 스키마 (`src/backfill/price/normalize.py` 산출물과 동일).
# 캐노니컬 로더/검증기는 이 스키마를 강제합니다.
PRICE_HISTORY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "trade_value_100m",
    "market_cap_100m",
    "daily_change_pct",
)

# 운영 시트 캡처의 필수 결정 시점 소스 컬럼. 없으면 fail-closed.
SNAPSHOT_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "change_rate",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "prev_close_price",
    "market_cap_100m",
    "trade_value_100m",
    "volume",
    "selection_rank",
    "total_candidate_count",
    "inst_net_buy",
    "foreign_net_buy",
    "prog_net_buy",
    "volume_power",
    "avg_trade_value",
    "kospi_change",
    "kosdaq_change",
    "v_kospi",
    "v_kosdaq",
    "buy_price",
)

# 이력의 선택적 수급 컬럼. 없으면 해당 피처는 NaN 으로 남고 future-fill 하지 않습니다.
PRICE_HISTORY_FLOW_COLUMNS: tuple[str, ...] = ("inst_net_buy", "foreign_net_buy", "prog_net_buy")

# 선언된 짧은 룩백 윈도우 그리드.
_LAG_LOOKBACKS: tuple[int, ...] = (2, 3, 5, 7, 9, 10, 12, 14, 15, 20)
_BETA_LOOKBACKS: tuple[int, ...] = (3, 5, 7, 9, 10, 12, 14, 15, 20)
_FLOW_LOOKBACKS: tuple[int, ...] = (2, 3, 5, 7, 9, 10, 12, 14, 15, 20)
_INDICATOR_LOOKBACKS: tuple[int, ...] = (12, 15, 20)
_MA_CROSS_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 5),
    (2, 7),
    (3, 5),
    (3, 10),
    (4, 9),
    (5, 10),
    (5, 20),
    (7, 15),
    (10, 20),
)
_VOL_RATIO_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 5),
    (3, 10),
    (4, 9),
    (5, 10),
    (5, 20),
    (7, 15),
)

# 지연 가격/유동성 상태 통계 (룩백 윈도우 끝이 직전 이력 행).
_LAG_STATS: tuple[str, ...] = (
    "ret",
    "chg",
    "gap",
    "range",
    "vol",
    "dd",
    "vol_surprise",
    "val_surprise",
    "max_ret",
    "min_ret",
    "ret_skew",
    "ret_kurt",
    "close_to_high",
    "close_to_low",
    "range_pos",
    "chg_vol",
    "gap_vol",
    "return_span",
    "ret_abs",
    "volume_z",
    "turnover",
)

_PREV_ROW_FEATURES: tuple[str, ...] = (
    "prev_ret",
    "prev_chg",
    "prev_gap",
    "prev_range",
    "prev_vol_surprise",
    "prev_val_surprise",
    "prev_close_to_high",
    "prev_close_to_low",
)

# 횡단면 베이스 메트릭 (동일 날짜 패널): (표시명, 컨텍스트 컬럼).
# 시장 수준 상수(kospi/kosdaq_change, v_kospi/v_kosdaq)는 횡단면 순위가
# 의미가 없으므로 제외합니다.
_CS_BASE_METRICS: tuple[tuple[str, str], ...] = (
    ("change_rate", "change_rate"),
    ("buy_gap", "_buy_gap"),
    ("gap_ratio", "_gap_ratio"),
    ("range_pct", "_range_pct"),
    ("body_pct", "_body_pct"),
    ("close_position", "_close_position"),
    ("turnover", "_turnover"),
    ("rank_ratio", "_rank_ratio"),
    ("selection_rank", "selection_rank"),
    ("log_market_cap", "_log_market_cap_100m"),
    ("log_trade_value", "_log_trade_value_100m"),
    ("log_volume", "_log_volume"),
    ("log_avg_trade_value", "_log_avg_trade_value"),
    ("log_price_level", "_log_price_level"),
    ("signed_log_inst_flow", "_signed_log_inst_flow"),
    ("signed_log_foreign_flow", "_signed_log_foreign_flow"),
    ("signed_log_prog_flow", "_signed_log_prog_flow"),
    ("inst_density", "_inst_density"),
    ("foreign_density", "_foreign_density"),
    ("major_density", "_major_density"),
    ("prog_dominance", "_prog_dominance"),
    ("volume_power", "volume_power"),
    ("price_level", "close_price"),
    ("open_price", "open_price"),
    ("high_price", "high_price"),
    ("low_price", "low_price"),
    ("buy_price", "buy_price"),
    ("avg_trade_value", "avg_trade_value"),
    ("high_low_spread", "_high_low_spread"),
    ("volume_turn_ratio", "_volume_turn_ratio"),
    ("avg_price", "_avg_price"),
    ("inst_foreign_spread", "_inst_foreign_spread"),
    ("upper_wick", "_upper_wick"),
    ("lower_wick", "_lower_wick"),
)

# 이미 ``cs_rel_*`` 로 중앙값 편차를 선언한 메트릭 (meddev 중복 방지).
_CS_REL_COVERED: frozenset[str] = frozenset(
    {
        "change_rate",
        "log_trade_value",
        "major_density",
        "inst_density",
        "foreign_density",
        "prog_dominance",
        "turnover",
        "log_volume",
        "log_market_cap",
        "log_avg_trade_value",
        "rank_ratio",
        "volume_power",
    }
)

_SECTOR_BASE_METRICS: tuple[tuple[str, str], ...] = (
    ("change_rate", "change_rate"),
    ("buy_gap", "_buy_gap"),
    ("turnover", "_turnover"),
    ("inst_density", "_inst_density"),
    ("foreign_density", "_foreign_density"),
    ("major_density", "_major_density"),
    ("prog_dominance", "_prog_dominance"),
    ("log_trade_value", "_log_trade_value_100m"),
    ("log_market_cap", "_log_market_cap_100m"),
    ("log_volume", "_log_volume"),
    ("volume_power", "volume_power"),
    ("rank_ratio", "_rank_ratio"),
    ("avg_trade_value", "avg_trade_value"),
    ("gap_ratio", "_gap_ratio"),
    ("signed_log_inst_flow", "_signed_log_inst_flow"),
    ("close_position", "_close_position"),
    ("signed_log_foreign_flow", "_signed_log_foreign_flow"),
    ("range_pct", "_range_pct"),
    ("body_pct", "_body_pct"),
)

_MARKET_RELATIVE_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("rel_kospi_change", "change_rate", "kospi_change"),
    ("rel_kosdaq_change", "change_rate", "kosdaq_change"),
    ("rel_market_change", "change_rate", "_market_ref"),
    ("rel_change_rate", "change_rate", "_cs_median_change_rate"),
    ("rel_trade_value", "_log_trade_value_100m", "_cs_median__log_trade_value_100m"),
    ("rel_major_density", "_major_density", "_cs_median__major_density"),
    ("rel_inst_density", "_inst_density", "_cs_median__inst_density"),
    ("rel_foreign_density", "_foreign_density", "_cs_median__foreign_density"),
    ("rel_prog_dominance", "_prog_dominance", "_cs_median__prog_dominance"),
    ("rel_turnover", "_turnover", "_cs_median__turnover"),
    ("rel_log_volume", "_log_volume", "_cs_median__log_volume"),
    ("rel_log_market_cap", "_log_market_cap_100m", "_cs_median__log_market_cap_100m"),
    ("rel_log_avg_trade_value", "_log_avg_trade_value", "_cs_median__log_avg_trade_value"),
    ("rel_rank_ratio", "_rank_ratio", "_cs_median__rank_ratio"),
    ("rel_volume_power", "volume_power", "_cs_median_volume_power"),
    ("rel_v_kospi", "v_kospi", "_cs_median_v_kospi"),
)


@dataclass(frozen=True)
class FeatureDefinition:
    """단일 카탈로그 피처의 결정적 선언입니다."""

    name: str
    family: str
    source_columns: tuple[str, ...]
    lookback_groups: tuple[str, ...]
    panel_scope: str
    unit: str
    availability_rule: str
    calculate: Callable[[pd.DataFrame], pd.Series]


def _select(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    def calc(ff: pd.DataFrame) -> pd.Series:
        if col in ff.columns:
            return ff[col]
        return pd.Series(np.nan, index=ff.index)

    return calc


# 상호작용 부모 이름 -> 실제 컨텍스트 컬럼 매핑 (표시명/내부 컬럼명 정규화).
_INTERACTION_SOURCE_MAP: dict[str, str] = {
    "change_rate": "change_rate",
    "buy_gap": "_buy_gap",
    "gap_ratio": "_gap_ratio",
    "range_pct": "_range_pct",
    "body_pct": "_body_pct",
    "close_position": "_close_position",
    "turnover": "_turnover",
    "log_trade_value": "_log_trade_value_100m",
    "log_volume": "_log_volume",
    "log_market_cap": "_log_market_cap_100m",
    "log_avg_trade_value": "_log_avg_trade_value",
    "major_density": "_major_density",
    "inst_density": "_inst_density",
    "foreign_density": "_foreign_density",
    "prog_dominance": "_prog_dominance",
    "signed_log_inst_flow": "_signed_log_inst_flow",
    "signed_log_foreign_flow": "_signed_log_foreign_flow",
    "rel_kospi_change": "_rel_kospi_change",
    "rel_market_change": "_rel_market_change",
    "v_kospi": "v_kospi",
    "v_kosdaq": "v_kosdaq",
}


def _interact(a: str, b: str) -> Callable[[pd.DataFrame], pd.Series]:
    """선언된 두 부모 피처의 guarded 곱 (둘 중 하나가 비유한이면 NaN)."""

    def calc(ff: pd.DataFrame) -> pd.Series:
        ca = _INTERACTION_SOURCE_MAP.get(a, a)
        cb = _INTERACTION_SOURCE_MAP.get(b, b)
        av = ff[ca].to_numpy(dtype=np.float64)
        bv = ff[cb].to_numpy(dtype=np.float64)
        out = np.full(len(ff), np.nan, dtype=np.float64)
        mask = np.isfinite(av) & np.isfinite(bv)
        out[mask] = av[mask] * bv[mask]
        return pd.Series(out, index=ff.index)

    return calc


def _guard_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.to_numpy(dtype=np.float64)
    den = denominator.to_numpy(dtype=np.float64)
    out = np.full(len(numerator), np.nan, dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return pd.Series(out, index=numerator.index)


# ---------------------------------------------------------------------------
# 캐노니컬 가격 이력 로더/검증기
# ---------------------------------------------------------------------------


def validate_price_history(price_history_df: pd.DataFrame) -> pd.DataFrame:
    """가격 이력을 검증하고 표준 스키마로 정규화해 반환합니다.

    고유 ``(symbol, date)``, 파싱 가능한 날짜, 수치 원천 필드, 그리고
    strict prior-date 사용 규칙이 준수됨을 보장합니다. 위반 시 ``ValueError``.
    """
    if price_history_df is None or price_history_df.empty:
        raise ValueError("price history is required for causal_expanded_v1; got an empty frame")

    missing = [c for c in ("symbol", "date", *PRICE_HISTORY_REQUIRED_COLUMNS) if c not in price_history_df.columns]
    if missing:
        raise ValueError(f"price history is missing required columns: {missing}")

    history = price_history_df.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    if history["date"].isna().any():
        raise ValueError("price history contains unparseable dates")

    history["symbol"] = (
        history["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )

    for col in PRICE_HISTORY_REQUIRED_COLUMNS:
        history[col] = pd.to_numeric(history[col], errors="coerce")
        if history[col].isna().all():
            raise ValueError(f"price history column {col!r} has no parseable numeric values")

    duplicated = history[history.duplicated(subset=["symbol", "date"], keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"price history violates unique (symbol, date): {len(duplicated)} duplicate rows"
        )
    history = history.drop_duplicates(subset=["symbol", "date"], keep="last")
    return history.sort_values(["symbol", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 스냅샷 준비 (동일 날짜 베이스 메트릭)
# ---------------------------------------------------------------------------


def _prepare_snapshot(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """운영 시트 캡처에서 결정 시점 베이스 메트릭을 계산합니다.

    모든 파생 값은 동일 날짜 캡처 필드의 벡터 연산이며, 0 분모는 NaN 으로
    안전 처리합니다.
    """
    if "trade_date" not in snapshot_df.columns or "stock_code" not in snapshot_df.columns:
        raise ValueError("snapshot_df must contain trade_date and stock_code columns")
    missing = [c for c in SNAPSHOT_REQUIRED_SOURCE_COLUMNS if c not in snapshot_df.columns]
    if missing:
        raise ValueError(f"causal_expanded_v1 snapshot is missing required source columns: {missing}")

    snap = snapshot_df.copy()
    snap["stock_code"] = (
        snap["stock_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    def _num(col: str) -> pd.Series:
        return pd.to_numeric(snap[col], errors="coerce")

    snap["_log_market_cap_100m"] = np.log1p(_num("market_cap_100m").clip(lower=0))
    snap["_log_trade_value_100m"] = np.log1p(_num("trade_value_100m").clip(lower=0))
    snap["_log_volume"] = np.log1p(_num("volume").clip(lower=0))
    snap["_log_avg_trade_value"] = np.log1p(_num("avg_trade_value").clip(lower=0))

    for src, dst in (
        ("inst_net_buy", "_signed_log_inst_flow"),
        ("foreign_net_buy", "_signed_log_foreign_flow"),
        ("prog_net_buy", "_signed_log_prog_flow"),
    ):
        raw = _num(src)
        snap[dst] = np.sign(raw) * np.log1p(np.abs(raw))

    trade_value = _num("trade_value_100m")
    snap["_inst_density"] = _guard_div(_num("inst_net_buy").fillna(0), trade_value)
    snap["_foreign_density"] = _guard_div(_num("foreign_net_buy").fillna(0), trade_value)
    snap["_major_density"] = _guard_div(
        _num("inst_net_buy").fillna(0) + _num("foreign_net_buy").fillna(0), trade_value
    )
    snap["_prog_dominance"] = _guard_div(_num("prog_net_buy"), trade_value)
    snap["_turnover"] = _guard_div(_num("trade_value_100m"), _num("market_cap_100m"))

    candidate_count = _num("total_candidate_count").fillna(1).clip(lower=1)
    snap["_rank_ratio"] = _num("selection_rank") / candidate_count

    prev_close = _num("prev_close_price")
    snap["_buy_gap"] = _guard_div(_num("buy_price") - prev_close, prev_close)
    snap["_gap_ratio"] = _guard_div(_num("open_price") - prev_close, prev_close)
    snap["_range_pct"] = _guard_div(_num("high_price") - _num("low_price"), prev_close)
    snap["_body_pct"] = _guard_div(_num("close_price") - _num("open_price"), prev_close)
    candle_range = (_num("high_price") - _num("low_price")).replace(0, np.nan)
    snap["_close_position"] = _guard_div(_num("close_price") - _num("low_price"), candle_range)

    if "market_type" in snap.columns:
        market_ref = np.where(
            snap["market_type"].astype(str).str.upper().str.contains("KOSDAQ", na=False),
            _num("kosdaq_change"),
            _num("kospi_change"),
        )
        snap["_market_ref"] = market_ref
    else:
        snap["_market_ref"] = _num("kospi_change")

    snap["_rel_kospi_change"] = _num("change_rate") - _num("kospi_change")
    snap["_rel_kosdaq_change"] = _num("change_rate") - _num("kosdaq_change")
    snap["_rel_market_change"] = _num("change_rate") - snap["_market_ref"]

    snap["_log_price_level"] = np.log(_num("close_price").clip(lower=1e-9))
    snap["_high_low_spread"] = _guard_div(_num("high_price") - _num("low_price"), _num("close_price"))
    snap["_volume_turn_ratio"] = _guard_div(_num("volume"), _num("market_cap_100m"))
    snap["_avg_price"] = _guard_div(_num("trade_value_100m"), _num("volume"))
    snap["_inst_foreign_spread"] = snap["_inst_density"] - snap["_foreign_density"]
    body_top = np.maximum(_num("open_price"), _num("close_price"))
    body_bottom = np.minimum(_num("open_price"), _num("close_price"))
    snap["_upper_wick"] = _guard_div(_num("high_price") - body_top, prev_close)
    snap["_lower_wick"] = _guard_div(body_bottom - _num("low_price"), prev_close)

    for metric in (
        "change_rate",
        "_log_trade_value_100m",
        "_major_density",
        "_inst_density",
        "_foreign_density",
        "_prog_dominance",
        "_turnover",
        "_log_volume",
        "_log_market_cap_100m",
        "_log_avg_trade_value",
        "_rank_ratio",
        "volume_power",
        "v_kospi",
    ):
        snap[f"_cs_median_{metric}"] = snap.groupby("trade_date")[metric].transform("median")
    return snap


# ---------------------------------------------------------------------------
# 이력 피처 테이블 (per (symbol, date) trailing window)
# ---------------------------------------------------------------------------


def _roll_mean(history: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        history.groupby("symbol", sort=False)[col]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _roll_sum(history: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        history.groupby("symbol", sort=False)[col]
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _roll_std(history: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        history.groupby("symbol", sort=False)[col]
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _roll_max(history: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        history.groupby("symbol", sort=False)[col]
        .rolling(window=window, min_periods=window)
        .max()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _roll_min(history: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        history.groupby("symbol", sort=False)[col]
        .rolling(window=window, min_periods=window)
        .min()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _roll_skew(history: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        history.groupby("symbol", sort=False)[col]
        .rolling(window=window, min_periods=window)
        .skew()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _roll_kurt(history: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        history.groupby("symbol", sort=False)[col]
        .rolling(window=window, min_periods=window)
        .kurt()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _group_shift(history: pd.DataFrame, col: str, periods: int) -> pd.Series:
    return history.groupby("symbol", sort=False)[col].shift(periods)


def _window_ret(history: pd.DataFrame, window: int) -> pd.Series:
    """직전 ``window`` 행의 누적 수익률 (expm1(누적 log-return 차분))."""
    cum = history.groupby("symbol", sort=False)["_log_ret"].cumsum()
    shifted = history.groupby("symbol", sort=False)["_log_ret"].shift(window)
    return np.expm1(cum - shifted)


def _roll_mean_series(series: pd.Series, history: pd.DataFrame, window: int) -> pd.Series:
    tmp = history.copy()
    tmp["_v"] = series.to_numpy(dtype=np.float64)
    return (
        tmp.groupby("symbol", sort=False)["_v"]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(history.index)
    )


def _build_history_feature_table(history: pd.DataFrame) -> pd.DataFrame:
    """이력 행마다 trailing window 통계를 계산한 wide 프레임을 반환합니다.

    모든 윈도우는 해당 행(포함)을 끝으로 하며, as-of 조인 시 마지막 prior 행의
    윈도우 값이 선택됩니다. 룩백을 채우지 못하면 NaN 입니다. ``symbol``/``date``
    컬럼은 as-of 조인을 위해 유지됩니다.
    """
    h = history.reset_index(drop=True).copy()
    h["_ret"] = h.groupby("symbol", sort=False)["close"].pct_change()
    h["_log_ret"] = np.log1p(h["_ret"].fillna(0.0))
    h["_pos"] = (h["_ret"] > 0).astype(np.float64)
    h["_chg"] = pd.to_numeric(h["daily_change_pct"], errors="coerce") / 100.0
    h["_gap"] = _guard_div(
        pd.to_numeric(h["open"], errors="coerce") - pd.to_numeric(h["prev_close"], errors="coerce"),
        pd.to_numeric(h["prev_close"], errors="coerce"),
    )
    h["_range"] = _guard_div(
        pd.to_numeric(h["high"], errors="coerce") - pd.to_numeric(h["low"], errors="coerce"),
        pd.to_numeric(h["close"], errors="coerce"),
    )
    h["_log_vol"] = np.log1p(pd.to_numeric(h["volume"], errors="coerce").clip(lower=0))
    h["_log_val"] = np.log1p(pd.to_numeric(h["trade_value_100m"], errors="coerce").clip(lower=0))
    h["_turnover"] = _guard_div(
        pd.to_numeric(h["trade_value_100m"], errors="coerce"),
        pd.to_numeric(h["market_cap_100m"], errors="coerce"),
    )
    h["_t"] = h.groupby("symbol", sort=False).cumcount().astype(np.float64)
    h["_t2"] = h["_t"] ** 2
    h["_log_close"] = np.log(pd.to_numeric(h["close"], errors="coerce").clip(lower=1e-9))
    h["_t_log_close"] = h["_t"] * h["_log_close"]

    mkt_ret = h["_ret"].groupby(h["date"]).mean()
    h["_mkt_ret"] = h["date"].map(mkt_ret)
    h["_mkt2"] = h["_mkt_ret"] ** 2
    h["_ret_mkt"] = h["_ret"] * h["_mkt_ret"]

    mkt_log = np.log1p(mkt_ret.fillna(0.0))
    mkt_cum = np.expm1(mkt_log.cumsum())

    cols: dict[str, pd.Series] = {"symbol": h["symbol"], "date": h["date"]}

    # ---- 지연 상태 (lagged_state) ----
    for w in _LAG_LOOKBACKS:
        cols[f"hist_ret_{w}d"] = _window_ret(h, w)
        cols[f"hist_chg_{w}d"] = _roll_mean(h, "_chg", w)
        cols[f"hist_gap_{w}d"] = _roll_mean(h, "_gap", w)
        cols[f"hist_range_{w}d"] = _roll_mean(h, "_range", w)
        cols[f"hist_vol_{w}d"] = _roll_std(h, "_ret", w)
        roll_max_close = _roll_max(h, "close", w)
        cols[f"hist_dd_{w}d"] = _guard_div(h["close"] - roll_max_close, roll_max_close)
        roll_mean_vol = _roll_mean(h, "_log_vol", w)
        cols[f"hist_vol_surprise_{w}d"] = h["_log_vol"] - roll_mean_vol
        roll_mean_val = _roll_mean(h, "_log_val", w)
        cols[f"hist_val_surprise_{w}d"] = h["_log_val"] - roll_mean_val
        cols[f"hist_max_ret_{w}d"] = _roll_max(h, "_ret", w)
        cols[f"hist_min_ret_{w}d"] = _roll_min(h, "_ret", w)
        cols[f"hist_ret_skew_{w}d"] = _roll_skew(h, "_ret", w)
        cols[f"hist_ret_kurt_{w}d"] = _roll_kurt(h, "_ret", w)
        roll_max_high = _roll_max(h, "high", w)
        cols[f"hist_close_to_high_{w}d"] = _guard_div(h["close"] - roll_max_high, roll_max_high)
        roll_min_low = _roll_min(h, "low", w)
        cols[f"hist_close_to_low_{w}d"] = _guard_div(h["close"] - roll_min_low, roll_min_low)
        cols[f"hist_range_pos_{w}d"] = _guard_div(
            h["close"] - roll_min_low, roll_max_high - roll_min_low
        )
        cols[f"hist_chg_vol_{w}d"] = _roll_std(h, "_chg", w)
        cols[f"hist_gap_vol_{w}d"] = _roll_std(h, "_gap", w)
        cols[f"hist_return_span_{w}d"] = cols[f"hist_max_ret_{w}d"] - cols[f"hist_min_ret_{w}d"]
        cols[f"hist_ret_abs_{w}d"] = _roll_mean(h, "_ret", w).abs()
        cols[f"hist_volume_z_{w}d"] = _guard_div(
            h["_log_vol"] - roll_mean_vol, _roll_std(h, "_log_vol", w)
        )
        cols[f"hist_turnover_{w}d"] = _roll_mean(h, "_turnover", w)

    # ---- 직전 행 (L=1): as-of 조인 시 마지막 prior 행의 값이 선택됩니다. ----
    cols["hist_prev_ret"] = h["_ret"]
    cols["hist_prev_chg"] = h["_chg"]
    cols["hist_prev_gap"] = h["_gap"]
    cols["hist_prev_range"] = h["_range"]
    cols["hist_prev_vol_surprise"] = h["_log_vol"] - _group_shift(h, "_log_vol", 5)
    cols["hist_prev_val_surprise"] = h["_log_val"] - _group_shift(h, "_log_val", 5)
    roll_max5 = _roll_max(h, "high", 5)
    cols["hist_prev_close_to_high"] = _guard_div(h["close"] - roll_max5, roll_max5)
    roll_min5 = _roll_min(h, "low", 5)
    cols["hist_prev_close_to_low"] = _guard_div(h["close"] - roll_min5, roll_min5)

    # ---- 트렌드/레짐 ----
    for w in _LAG_LOOKBACKS:
        roll_mean_close = _roll_mean(h, "close", w)
        cols[f"hist_ma_spread_{w}d"] = _guard_div(h["close"] - roll_mean_close, roll_mean_close)
        cols[f"hist_atr_norm_{w}d"] = _guard_div(
            _roll_mean(h, "high", w) - _roll_mean(h, "low", w), h["close"]
        )
        cols[f"hist_market_ret_{w}d"] = h["date"].map(mkt_cum.pct_change(w))
        cols[f"hist_market_vol_{w}d"] = h["date"].map(mkt_ret.rolling(w).std())
        cols[f"hist_market_skew_{w}d"] = h["date"].map(mkt_ret.rolling(w).skew())
        cols[f"hist_market_upside_{w}d"] = h["date"].map((mkt_ret > 0).rolling(w).mean())
        cols[f"hist_rel_strength_{w}d"] = cols[f"hist_ret_{w}d"] - cols[f"hist_market_ret_{w}d"]
        cols[f"hist_upside_ratio_{w}d"] = _roll_mean(h, "_pos", w)
        t_mean = _roll_mean(h, "_t", w)
        t2_mean = _roll_mean(h, "_t2", w)
        logc_mean = _roll_mean(h, "_log_close", w)
        tl_mean = _roll_mean(h, "_t_log_close", w)
        var_t = t2_mean - t_mean**2
        cov = tl_mean - t_mean * logc_mean
        cols[f"hist_trend_slope_{w}d"] = _guard_div(cov, var_t)
        cols[f"hist_sharpe_{w}d"] = _guard_div(cols[f"hist_ret_{w}d"], _roll_std(h, "_ret", w))
        cols[f"hist_atr_band_{w}d"] = cols[f"hist_atr_norm_{w}d"] * np.sqrt(w)

    for w in _BETA_LOOKBACKS:
        mean_x = _roll_mean(h, "_ret", w)
        mean_y = _roll_mean(h, "_mkt_ret", w)
        mean_xy = _roll_mean(h, "_ret_mkt", w)
        mean_y2 = _roll_mean(h, "_mkt2", w)
        var_y = mean_y2 - mean_y**2
        cov = mean_xy - mean_x * mean_y
        beta = _guard_div(cov, var_y)
        std_x = _roll_std(h, "_ret", w)
        std_y = _roll_std(h, "_mkt_ret", w)
        cols[f"hist_beta_{w}d"] = beta
        cols[f"hist_corr_{w}d"] = _guard_div(cov, std_x * std_y)
        cols[f"hist_stock_market_rel_vol_{w}d"] = _guard_div(std_x, std_y)

    for short, long in _MA_CROSS_PAIRS:
        cols[f"hist_ma_cross_{short}_{long}d"] = (
            _guard_div(_roll_mean(h, "close", short), _roll_mean(h, "close", long)) - 1
        )
    for short, long in _VOL_RATIO_PAIRS:
        cols[f"hist_vol_ratio_{short}_{long}d"] = (
            _guard_div(_roll_std(h, "_ret", short), _roll_std(h, "_ret", long)) - 1
        )

    # ---- 수급 지속성 (선택적 이력 수급 컬럼) ----
    for flow in PRICE_HISTORY_FLOW_COLUMNS:
        if flow not in h.columns:
            continue
        h[flow] = pd.to_numeric(h[flow], errors="coerce").fillna(0.0)
        f = h[flow]
        tag = flow.replace("_net_buy", "")
        for w in _FLOW_LOOKBACKS:
            cols[f"hist_flow_{tag}_{w}d"] = _roll_sum(h, flow, w)
            cols[f"hist_flow_{tag}_chg_{w}d"] = f - _group_shift(h, flow, w)
            cols[f"hist_flow_{tag}_imb_{w}d"] = _guard_div(
                _roll_sum(h, flow, w), _roll_mean(h, "trade_value_100m", w)
            )
            cols[f"hist_flow_{tag}_std_{w}d"] = _roll_std(h, flow, w)
            cols[f"hist_flow_{tag}_val_ratio_{w}d"] = _guard_div(
                _roll_sum(h, flow, w), _roll_sum(h, "trade_value_100m", w)
            )
            cols[f"hist_flow_{tag}_turn_{w}d"] = _guard_div(
                _roll_sum(h, flow, w), _roll_mean(h, "market_cap_100m", w)
            )

    # ---- 희소 룩백 이용가능 지표 ----
    for w in _INDICATOR_LOOKBACKS:
        cols[f"hist_avail_{w}d"] = cols[f"hist_ret_{w}d"].notna().astype(np.float64)

    return pd.DataFrame(cols, index=h.index)


def _asof_join_history(snap: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """strict prior-date as-of 조인 (동일 날짜 EOD 행 제외, forward-fill 없음).

    ``merge_asof`` 는 on-key 가 전역 단조 증가해야 하므로 날짜 기준 전역 정렬 후
    ``by`` 그룹을 C 레벨 asof 가 처리합니다.
    """
    left = snap.copy()
    left["_row_id"] = np.arange(len(left))
    left = left.sort_values("trade_date")

    hist = wide.copy()
    hist["stock_code"] = hist["symbol"]
    hist = hist.drop(columns=["symbol"])
    hist = hist.sort_values("date")

    merged = pd.merge_asof(
        left,
        hist,
        left_on="trade_date",
        right_on="date",
        by="stock_code",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_row_id").set_index("_row_id")
    merged = merged.drop(columns=["date"])
    merged.index = snap.index
    return merged


# ---------------------------------------------------------------------------
# 카탈로그 빌더
# ---------------------------------------------------------------------------


def _build_sheet_features() -> list[FeatureDefinition]:
    defs: list[FeatureDefinition] = []
    same = ("same_date",)

    def add(name: str, unit: str, col: str | None = None) -> None:
        src = (col or name,)
        defs.append(
            FeatureDefinition(
                name=name,
                family="sheet_level",
                source_columns=src,
                lookback_groups=same,
                panel_scope="candidate_panel",
                unit=unit,
                availability_rule="at_decision_time",
                calculate=_select(col or name),
            )
        )

    for log_name, log_col in (
        ("snap_log_market_cap_100m", "_log_market_cap_100m"),
        ("snap_log_trade_value_100m", "_log_trade_value_100m"),
        ("snap_log_volume", "_log_volume"),
        ("snap_log_avg_trade_value", "_log_avg_trade_value"),
    ):
        add(log_name, "log_won_100m", log_col)
    for name, col in (
        ("snap_signed_log_inst_flow", "_signed_log_inst_flow"),
        ("snap_signed_log_foreign_flow", "_signed_log_foreign_flow"),
        ("snap_signed_log_prog_flow", "_signed_log_prog_flow"),
    ):
        add(name, "signed_log_won", col)
    for name, col, unit in (
        ("snap_turnover", "_turnover", "decimal_ratio"),
        ("snap_rank_ratio", "_rank_ratio", "decimal_ratio"),
        ("snap_selection_rank", "selection_rank", "integer_rank"),
        ("snap_inst_density", "_inst_density", "decimal_ratio"),
        ("snap_foreign_density", "_foreign_density", "decimal_ratio"),
        ("snap_major_density", "_major_density", "decimal_ratio"),
        ("snap_prog_dominance", "_prog_dominance", "decimal_ratio"),
        ("snap_buy_gap", "_buy_gap", "decimal_ratio"),
        ("snap_gap_ratio", "_gap_ratio", "decimal_ratio"),
        ("snap_range_pct", "_range_pct", "decimal_ratio"),
        ("snap_body_pct", "_body_pct", "decimal_ratio"),
        ("snap_close_position", "_close_position", "decimal_ratio"),
        ("snap_change_rate", "change_rate", "percent"),
        ("snap_volume_power", "volume_power", "decimal_ratio"),
        ("snap_kospi_change", "kospi_change", "percent"),
        ("snap_kosdaq_change", "kosdaq_change", "percent"),
        ("snap_v_kospi", "v_kospi", "index_level"),
        ("snap_v_kosdaq", "v_kosdaq", "index_level"),
    ):
        add(name, unit, col)

    def flow_consensus(ff: pd.DataFrame) -> pd.Series:
        flow = np.column_stack(
            [
                ff["_signed_log_inst_flow"].to_numpy(dtype=np.float64),
                ff["_signed_log_foreign_flow"].to_numpy(dtype=np.float64),
                ff["_signed_log_prog_flow"].to_numpy(dtype=np.float64),
            ]
        )
        return pd.Series(np.sign(flow).sum(axis=1).astype(np.float64), index=ff.index)

    def flow_alignment(ff: pd.DataFrame) -> pd.Series:
        flow = np.column_stack(
            [
                ff["_signed_log_inst_flow"].to_numpy(dtype=np.float64),
                ff["_signed_log_foreign_flow"].to_numpy(dtype=np.float64),
                ff["_signed_log_prog_flow"].to_numpy(dtype=np.float64),
            ]
        )
        denom = np.abs(flow).sum(axis=1)
        out = np.zeros(len(ff), dtype=np.float64)
        np.divide(flow.sum(axis=1), denom, out=out, where=denom != 0)
        return pd.Series(out, index=ff.index)

    defs.append(
        FeatureDefinition(
            name="snap_flow_consensus",
            family="sheet_level",
            source_columns=("inst_net_buy", "foreign_net_buy", "prog_net_buy"),
            lookback_groups=same,
            panel_scope="candidate_panel",
            unit="signed_count",
            availability_rule="at_decision_time",
            calculate=flow_consensus,
        )
    )
    defs.append(
        FeatureDefinition(
            name="snap_flow_alignment",
            family="sheet_level",
            source_columns=("inst_net_buy", "foreign_net_buy", "prog_net_buy"),
            lookback_groups=same,
            panel_scope="candidate_panel",
            unit="decimal_ratio",
            availability_rule="at_decision_time",
            calculate=flow_alignment,
        )
    )
    return defs


def _build_cross_section_features() -> list[FeatureDefinition]:
    defs: list[FeatureDefinition] = []
    same = ("same_date",)

    for tag, col in _CS_BASE_METRICS:
        defs.append(
            FeatureDefinition(
                name=f"cs_{tag}_pct_rank",
                family="cross_section",
                source_columns=(col,),
                lookback_groups=same,
                panel_scope="candidate_panel",
                unit="pct_rank",
                availability_rule="at_decision_time",
                calculate=lambda ff, c=col: ff.groupby("trade_date")[c].rank(pct=True),
            )
        )

        def _robust_z(ff: pd.DataFrame, c: str) -> pd.Series:
            grouped = ff.groupby("trade_date")[c]
            med = grouped.transform("median")
            mad = grouped.transform(lambda s: (s - s.median()).abs().median())
            mad = mad.replace(0, np.nan)
            return ((ff[c] - med) / mad).clip(-5, 5)

        defs.append(
            FeatureDefinition(
                name=f"cs_{tag}_robust_z",
                family="cross_section",
                source_columns=(col,),
                lookback_groups=same,
                panel_scope="candidate_panel",
                unit="robust_z",
                availability_rule="at_decision_time",
                calculate=lambda ff, c=col: _robust_z(ff, c),
            )
        )
        if tag not in _CS_REL_COVERED:
            defs.append(
                FeatureDefinition(
                    name=f"cs_{tag}_meddev",
                    family="cross_section",
                    source_columns=(col,),
                    lookback_groups=same,
                    panel_scope="candidate_panel",
                    unit="decimal_ratio",
                    availability_rule="at_decision_time",
                    calculate=lambda ff, c=col: ff[c] - ff.groupby("trade_date")[c].transform("median"),
                )
            )

    def _sector_rank(ff: pd.DataFrame, c: str) -> pd.Series:
        if "theme_sector" not in ff.columns:
            return pd.Series(np.nan, index=ff.index)
        return ff.groupby(["trade_date", "theme_sector"])[c].rank(pct=True)

    for tag, col in _SECTOR_BASE_METRICS:
        defs.append(
            FeatureDefinition(
                name=f"cs_{tag}_sector_pct_rank",
                family="cross_section",
                source_columns=("theme_sector", col),
                lookback_groups=same,
                panel_scope="sector_panel",
                unit="pct_rank",
                availability_rule="at_decision_time",
                calculate=lambda ff, c=col: _sector_rank(ff, c),
            )
        )
        defs.append(
            FeatureDefinition(
                name=f"cs_{tag}_sector_robust_z",
                family="cross_section",
                source_columns=("theme_sector", col),
                lookback_groups=same,
                panel_scope="sector_panel",
                unit="robust_z",
                availability_rule="at_decision_time",
                calculate=lambda ff, c=col: _sector_robust_z(ff, c),
            )
        )

    for tag, col, ref in _MARKET_RELATIVE_FEATURES:
        defs.append(
            FeatureDefinition(
                name=f"cs_{tag}",
                family="cross_section",
                source_columns=(col, ref),
                lookback_groups=same,
                panel_scope="candidate_panel",
                unit="decimal_ratio",
                availability_rule="at_decision_time",
                calculate=lambda ff, c=col, r=ref: ff[c] - ff[r],
            )
        )
    return defs


def _sector_robust_z(ff: pd.DataFrame, c: str) -> pd.Series:
    if "theme_sector" not in ff.columns:
        return pd.Series(np.nan, index=ff.index)
    grouped = ff.groupby(["trade_date", "theme_sector"])[c]
    med = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    mad = mad.replace(0, np.nan)
    return ((ff[c] - med) / mad).clip(-5, 5)


def _build_lagged_state_features() -> list[FeatureDefinition]:
    defs: list[FeatureDefinition] = []

    def add(name: str, unit: str, col: str | None = None, lookback: tuple[str, ...] | None = None) -> None:
        defs.append(
            FeatureDefinition(
                name=name,
                family="lagged_state",
                source_columns=(col or name,),
                lookback_groups=lookback or (f"{name.split('_')[-1]}",),
                panel_scope="stock_history",
                unit=unit,
                availability_rule="prior_date_history_only",
                calculate=_select(col or name),
            )
        )

    for w in _LAG_LOOKBACKS:
        for stat in _LAG_STATS:
            add(f"hist_{stat}_{w}d", "decimal_ratio", f"hist_{stat}_{w}d", (f"{w}d",))
    for stat in _PREV_ROW_FEATURES:
        add(f"hist_{stat}", "decimal_ratio", f"hist_{stat}", ("1d",))
    for w in _INDICATOR_LOOKBACKS:
        add(f"hist_avail_{w}d", "binary_indicator", f"hist_avail_{w}d", (f"{w}d",))
    return defs


def _build_trend_regime_features() -> list[FeatureDefinition]:
    defs: list[FeatureDefinition] = []

    def add(name: str, unit: str, col: str | None = None, lookback: tuple[str, ...] | None = None) -> None:
        defs.append(
            FeatureDefinition(
                name=name,
                family="trend_regime",
                source_columns=(col or name,),
                lookback_groups=lookback or ("declared",),
                panel_scope="stock_history",
                unit=unit,
                availability_rule="prior_date_history_only",
                calculate=_select(col or name),
            )
        )

    for w in _LAG_LOOKBACKS:
        add(f"hist_ma_spread_{w}d", "decimal_ratio", f"hist_ma_spread_{w}d", (f"{w}d",))
        add(f"hist_atr_norm_{w}d", "decimal_ratio", f"hist_atr_norm_{w}d", (f"{w}d",))
        add(f"hist_market_ret_{w}d", "decimal_ratio", f"hist_market_ret_{w}d", (f"{w}d",))
        add(f"hist_market_vol_{w}d", "decimal_ratio", f"hist_market_vol_{w}d", (f"{w}d",))
        add(f"hist_market_skew_{w}d", "decimal_ratio", f"hist_market_skew_{w}d", (f"{w}d",))
        add(f"hist_market_upside_{w}d", "decimal_ratio", f"hist_market_upside_{w}d", (f"{w}d",))
        add(f"hist_rel_strength_{w}d", "decimal_ratio", f"hist_rel_strength_{w}d", (f"{w}d",))
        add(f"hist_upside_ratio_{w}d", "decimal_ratio", f"hist_upside_ratio_{w}d", (f"{w}d",))
        add(f"hist_trend_slope_{w}d", "decimal_ratio", f"hist_trend_slope_{w}d", (f"{w}d",))
        add(f"hist_sharpe_{w}d", "decimal_ratio", f"hist_sharpe_{w}d", (f"{w}d",))
        add(f"hist_atr_band_{w}d", "decimal_ratio", f"hist_atr_band_{w}d", (f"{w}d",))
    for w in _BETA_LOOKBACKS:
        add(f"hist_beta_{w}d", "decimal_ratio", f"hist_beta_{w}d", (f"{w}d",))
        add(f"hist_corr_{w}d", "decimal_ratio", f"hist_corr_{w}d", (f"{w}d",))
        add(f"hist_stock_market_rel_vol_{w}d", "decimal_ratio", f"hist_stock_market_rel_vol_{w}d", (f"{w}d",))
    for short, long in _MA_CROSS_PAIRS:
        add(f"hist_ma_cross_{short}_{long}d", "decimal_ratio", f"hist_ma_cross_{short}_{long}d", (f"{short}d", f"{long}d"))
    for short, long in _VOL_RATIO_PAIRS:
        add(f"hist_vol_ratio_{short}_{long}d", "decimal_ratio", f"hist_vol_ratio_{short}_{long}d", (f"{short}d", f"{long}d"))
    return defs


def _build_flow_persistence_features() -> list[FeatureDefinition]:
    defs: list[FeatureDefinition] = []

    def add(name: str, unit: str, col: str | None = None, lookback: tuple[str, ...] | None = None) -> None:
        defs.append(
            FeatureDefinition(
                name=name,
                family="flow_persistence",
                source_columns=(col or name,),
                lookback_groups=lookback or ("declared",),
                panel_scope="stock_history",
                unit=unit,
                availability_rule="prior_date_history_only",
                calculate=_select(col or name),
            )
        )

    for tag in ("inst", "foreign", "prog"):
        for w in _FLOW_LOOKBACKS:
            add(f"hist_flow_{tag}_{w}d", "won_100m", f"hist_flow_{tag}_{w}d", (f"{w}d",))
            add(f"hist_flow_{tag}_chg_{w}d", "won_100m", f"hist_flow_{tag}_chg_{w}d", (f"{w}d",))
            add(f"hist_flow_{tag}_imb_{w}d", "decimal_ratio", f"hist_flow_{tag}_imb_{w}d", (f"{w}d",))
            add(f"hist_flow_{tag}_std_{w}d", "won_100m", f"hist_flow_{tag}_std_{w}d", (f"{w}d",))
            add(f"hist_flow_{tag}_val_ratio_{w}d", "decimal_ratio", f"hist_flow_{tag}_val_ratio_{w}d", (f"{w}d",))
            add(f"hist_flow_{tag}_turn_{w}d", "decimal_ratio", f"hist_flow_{tag}_turn_{w}d", (f"{w}d",))
    return defs


def _build_interaction_features() -> list[FeatureDefinition]:
    defs: list[FeatureDefinition] = []

    def add(name: str, parents: tuple[str, ...]) -> None:
        availability = "at_decision_time"
        if any(p.startswith(("hist_",)) for p in parents):
            availability = "prior_date_history_only"
        lookback: tuple[str, ...] = ()
        for p in parents:
            if p.startswith(("hist_",)):
                lookback += (p.rsplit("_", 1)[-1],)
        defs.append(
            FeatureDefinition(
                name=name,
                family="interactions",
                source_columns=parents,
                lookback_groups=lookback or ("same_date",),
                panel_scope="candidate_panel",
                unit="decimal_ratio",
                availability_rule=availability,
                calculate=_interact(*parents),
            )
        )

    price_metrics = ("change_rate", "buy_gap", "gap_ratio", "range_pct", "body_pct", "close_position")
    liq_metrics = ("turnover", "log_trade_value", "log_volume", "log_market_cap", "log_avg_trade_value")
    for p in price_metrics:
        for liq in liq_metrics:
            add(f"int_price_{p}_{liq}", (p, liq))

    flow_metrics = ("major_density", "inst_density", "foreign_density", "prog_dominance", "signed_log_inst_flow", "signed_log_foreign_flow")
    momentum = ("change_rate", "hist_ret_5d", "hist_ret_10d", "hist_rel_strength_5d")
    for f in flow_metrics:
        for m in momentum:
            add(f"int_flow_{f}_{m}", (f, m))

    regime = ("hist_market_vol_5d", "hist_market_vol_10d", "hist_market_ret_5d", "v_kospi", "v_kosdaq")
    rel_strength = ("hist_rel_strength_5d", "hist_rel_strength_10d", "rel_kospi_change", "rel_market_change")
    for r in regime:
        for rs in rel_strength:
            add(f"int_regime_{r}_{rs}", (r, rs))

    cs_a = ("turnover", "major_density", "log_trade_value", "inst_density")
    cs_b = ("change_rate", "gap_ratio", "inst_density", "buy_gap")
    for a in cs_a:
        for b in cs_b:
            if a == b:
                continue
            add(f"int_csrank_{a}_{b}", (f"cs_{a}_pct_rank", f"cs_{b}_pct_rank"))

    for pair in (("change_rate", "gap_ratio"), ("change_rate", "buy_gap"), ("gap_ratio", "buy_gap"), ("range_pct", "body_pct")):
        add(f"int_price_{pair[0]}_{pair[1]}", pair)

    price_regime = ("change_rate", "gap_ratio", "range_pct")
    for p in price_regime:
        for r in ("hist_market_vol_5d", "hist_market_ret_5d", "v_kospi"):
            add(f"int_price_regime_{p}_{r}", (p, r))

    return defs


def build_catalog(catalog_version: str = "causal_expanded_v1") -> list[FeatureDefinition]:
    """결정적 카탈로그 피처 정의 목록을 반환합니다 (600--1000 후보)."""
    if catalog_version not in SUPPORTED_CATALOG_VERSIONS:
        raise ValueError(f"unsupported catalog_version: {catalog_version!r}")
    definitions: list[FeatureDefinition] = [
        *_build_sheet_features(),
        *_build_cross_section_features(),
        *_build_lagged_state_features(),
        *_build_trend_regime_features(),
        *_build_flow_persistence_features(),
        *_build_interaction_features(),
    ]
    names = [d.name for d in definitions]
    if len(names) != len(set(names)):
        duplicates = {name for name in names if names.count(name) > 1}
        raise ValueError(f"catalog contains duplicate feature names: {sorted(duplicates)}")
    transform_keys: set[tuple[object, ...]] = set()
    for d in definitions:
        key = (d.family, d.source_columns, d.lookback_groups, d.unit)
        if key in transform_keys:
            raise ValueError(f"catalog contains duplicate source/transform definition: {d.name}")
        transform_keys.add(key)
    return definitions


def catalog_hash(definitions: list[FeatureDefinition], catalog_version: str) -> str:
    """카탈로그 버전과 피처 이름/소스/룩백의 결정적 해시."""
    payload = {
        "version": catalog_version,
        "features": [
            {"name": d.name, "family": d.family, "source": list(d.source_columns), "lookback": list(d.lookback_groups)}
            for d in definitions
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_catalog_manifest(definitions: list[FeatureDefinition]) -> pd.DataFrame:
    """카탈로그 피처에 대한 메타데이터 풍부 매니페스트를 생성합니다."""
    return pd.DataFrame(
        [
            {
                "feature_name": d.name,
                "family": d.family,
                "source_columns": list(d.source_columns),
                "lookback_groups": list(d.lookback_groups),
                "unit": d.unit,
                "panel_scope": d.panel_scope,
                "availability_rule": d.availability_rule,
            }
            for d in definitions
        ]
    )


# ---------------------------------------------------------------------------
# 공개 진입점
# ---------------------------------------------------------------------------


def build_causal_feature_matrix(
    snapshot_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    catalog_version: str = "causal_expanded_v1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """후보 행렬과 메타데이터 매니페스트를 반환합니다.

    Returns:
        (matrix, manifest): matrix 는 ``snapshot_df`` 인덱스에 정렬된
        수치 후보 행렬, manifest 는 카탈로그 메타데이터 DataFrame 입니다.
    """
    if catalog_version not in SUPPORTED_CATALOG_VERSIONS:
        raise ValueError(f"unsupported catalog_version: {catalog_version!r}")
    history = validate_price_history(price_history_df)
    definitions = build_catalog(catalog_version)

    snap = _prepare_snapshot(snapshot_df)
    wide = _build_history_feature_table(history)
    ctx = _asof_join_history(snap, wide)

    # 상호작용 피처가 동일 날짜 부모(시트/횡단면)를 참조할 수 있도록
    # 시트/횡단면 컬럼을 컨텍스트에 먼저 계산합니다.
    same_date_cols: dict[str, pd.Series] = {}
    for definition in definitions:
        if definition.family in ("sheet_level", "cross_section"):
            same_date_cols[definition.name] = definition.calculate(ctx)
    if same_date_cols:
        ctx = pd.concat([ctx, pd.DataFrame(same_date_cols, index=ctx.index)], axis=1)

    columns: dict[str, pd.Series] = {}
    for definition in definitions:
        if definition.name in ctx.columns:
            columns[definition.name] = ctx[definition.name]
        else:
            columns[definition.name] = definition.calculate(ctx)
    matrix = pd.DataFrame(columns, index=snap.index)

    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    if not matrix.columns.is_unique:
        raise ValueError("catalog produced duplicate matrix columns")
    n_candidates = matrix.shape[1]
    if not (MIN_CANDIDATES <= n_candidates <= MAX_CANDIDATES):
        raise ValueError(
            f"catalog produced {n_candidates} candidates; must be within "
            f"[{MIN_CANDIDATES}, {MAX_CANDIDATES}]"
        )

    manifest = build_catalog_manifest(definitions)
    matrix.attrs["catalog_version"] = catalog_version
    matrix.attrs["catalog_hash"] = catalog_hash(definitions, catalog_version)
    matrix.attrs["candidate_count"] = n_candidates
    return matrix, manifest
