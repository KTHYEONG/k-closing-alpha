"""Intraday 정규화 스키마 (KIS/LS 벤더 의미 통합).

KIS 누적 거래대금과 LS 바당 거래대금(백만원 단위)의 단위 충돌,
합성 무거래봉의 공백 채움 의미 반전을 정규화 이름/단위/dtype으로 통일한다.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

LS_VALUE_UNIT_KRW: int = 1_000_000

CANONICAL_BAR_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "symbol",
    "ts_hms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value_krw",
    "has_trade",
    "vendor",
)

CANONICAL_TICK_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "symbol",
    "ts_hms",
    "price",
    "volume",
    "trade_strength",
    "ask1",
    "bid1",
    "truncated",
    "vendor",
)

_BAR_VENDORS: tuple[str, ...] = ("kis", "ls")

_KIS_BAR_REQUIRED: tuple[str, ...] = (
    "stck_cntg_hour",
    "stck_oprc",
    "stck_hgpr",
    "stck_lwpr",
    "stck_prpr",
    "cntg_vol",
    "acml_tr_pbmn",
)

_LS_BAR_REQUIRED: tuple[str, ...] = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "jdiff_vol",
    "value",
)

_KIS_TICK_REQUIRED: tuple[str, ...] = (
    "stck_cntg_hour",
    "stck_prpr",
)

_LS_TICK_REQUIRED: tuple[str, ...] = (
    "time",
    "close",
    "jdiff_vol",
)


def _empty_bar_frame() -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_BAR_COLUMNS})
    return out.astype(
        {
            "snapshot_date": "str",
            "symbol": "str",
            "ts_hms": "int32",
            "open": "int32",
            "high": "int32",
            "low": "int32",
            "close": "int32",
            "volume": "int64",
            "value_krw": "int64",
            "has_trade": "bool",
            "vendor": "str",
        }
    )


def _empty_tick_frame() -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_TICK_COLUMNS})
    return out.astype(
        {
            "snapshot_date": "str",
            "symbol": "str",
            "ts_hms": "int32",
            "price": "int32",
            "volume": "int64",
            "trade_strength": "Float32",
            "ask1": "Int32",
            "bid1": "Int32",
            "truncated": "bool",
            "vendor": "str",
        }
    )


def _check_vendor(vendor: str) -> str:
    if vendor not in ("kis", "ls"):
        raise ValueError(f"Unknown intraday vendor: {vendor!r} (expected one of 'kis', 'ls')")
    return vendor


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], vendor: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required {vendor} source columns: {missing}")


def normalize_bar_frame(df: pd.DataFrame, vendor: str, snapshot_date: str, symbol: str) -> pd.DataFrame:
    """벤더 원천 분봉 프레임을 정규 바 스키마로 변환한다."""
    _check_vendor(vendor)
    if df is None or len(df) == 0:
        return _empty_bar_frame()
    code = str(symbol).zfill(6)

    if vendor == "kis":
        _require_columns(df, _KIS_BAR_REQUIRED, vendor)
        work = pd.DataFrame(
            {
                "ts_hms": pd.to_numeric(df["stck_cntg_hour"].astype(str), errors="coerce"),
                "open": pd.to_numeric(df["stck_oprc"].astype(str), errors="coerce"),
                "high": pd.to_numeric(df["stck_hgpr"].astype(str), errors="coerce"),
                "low": pd.to_numeric(df["stck_lwpr"].astype(str), errors="coerce"),
                "close": pd.to_numeric(df["stck_prpr"].astype(str), errors="coerce"),
                "volume": pd.to_numeric(df["cntg_vol"].astype(str), errors="coerce"),
                "_cum_value": pd.to_numeric(df["acml_tr_pbmn"].astype(str), errors="coerce"),
            }
        )
        work = work.sort_values("ts_hms", kind="stable").reset_index(drop=True)
        cum = work["_cum_value"]
        diff = cum.diff()
        diff.iloc[0] = cum.iloc[0]
        n_negative = int((diff < 0).sum())
        if n_negative:
            logger.warning(
                "[DATA] KIS cumulative value went backwards; clamped %d bar(s) to 0 symbol=%s",
                n_negative,
                code,
            )
        value_krw = diff.clip(lower=0)
    else:
        _require_columns(df, _LS_BAR_REQUIRED, vendor)
        work = pd.DataFrame(
            {
                "ts_hms": pd.to_numeric(df["time"].astype(str), errors="coerce"),
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["jdiff_vol"], errors="coerce"),
            }
        )
        value_krw = pd.to_numeric(df["value"], errors="coerce") * LS_VALUE_UNIT_KRW

    out = pd.DataFrame(
        {
            "snapshot_date": str(snapshot_date),
            "symbol": code,
            "ts_hms": work["ts_hms"],
            "open": work["open"],
            "high": work["high"],
            "low": work["low"],
            "close": work["close"],
            "volume": work["volume"],
            "value_krw": value_krw,
            "has_trade": pd.to_numeric(work["volume"], errors="coerce").fillna(0) > 0,
            "vendor": vendor,
        }
    )
    out = out.astype(
        {
            "snapshot_date": "str",
            "symbol": "str",
            "ts_hms": "int32",
            "open": "int32",
            "high": "int32",
            "low": "int32",
            "close": "int32",
            "volume": "int64",
            "value_krw": "int64",
            "has_trade": "bool",
            "vendor": "str",
        }
    )
    return out[list(CANONICAL_BAR_COLUMNS)]


def normalize_tick_frame(
    df: pd.DataFrame, vendor: str, snapshot_date: str, symbol: str, truncated: bool = False
) -> pd.DataFrame:
    """벤더 원천 틱 프레임을 정규 틱 스키마로 변환한다."""
    _check_vendor(vendor)
    if df is None or len(df) == 0:
        out = _empty_tick_frame()
        return out
    code = str(symbol).zfill(6)

    if vendor == "kis":
        _require_columns(df, _KIS_TICK_REQUIRED, vendor)
        ts_hms = pd.to_numeric(df["stck_cntg_hour"].astype(str), errors="coerce")
        price = pd.to_numeric(df["stck_prpr"].astype(str), errors="coerce")
        vol_src = None
        for key in ("cnqn", "cntg_vol"):
            if key in df.columns:
                vol_src = df[key]
                break
        if vol_src is None:
            raise ValueError("Missing required kis tick volume column: one of ['cnqn', 'cntg_vol']")
        volume = pd.to_numeric(vol_src.astype(str), errors="coerce")
        if "tday_rltv" in df.columns:
            trade_strength: pd.Series = pd.to_numeric(df["tday_rltv"].astype(str), errors="coerce").astype("Float32")
        else:
            trade_strength = pd.Series(pd.NA, index=df.index, dtype="Float32")
        if "askp" in df.columns:
            ask1: pd.Series = pd.to_numeric(df["askp"].astype(str), errors="coerce").astype("Int32")
        else:
            ask1 = pd.Series(pd.NA, index=df.index, dtype="Int32")
        if "bidp" in df.columns:
            bid1: pd.Series = pd.to_numeric(df["bidp"].astype(str), errors="coerce").astype("Int32")
        else:
            bid1 = pd.Series(pd.NA, index=df.index, dtype="Int32")
    else:
        _require_columns(df, _LS_TICK_REQUIRED, vendor)
        ts_hms = pd.to_numeric(df["time"].astype(str), errors="coerce")
        price = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df["jdiff_vol"], errors="coerce")
        trade_strength = pd.Series(pd.NA, index=df.index, dtype="Float32")
        ask1 = pd.Series(pd.NA, index=df.index, dtype="Int32")
        bid1 = pd.Series(pd.NA, index=df.index, dtype="Int32")

    out = pd.DataFrame(
        {
            "snapshot_date": str(snapshot_date),
            "symbol": code,
            "ts_hms": ts_hms,
            "price": price,
            "volume": volume,
            "trade_strength": trade_strength,
            "ask1": ask1,
            "bid1": bid1,
            "truncated": bool(truncated),
            "vendor": vendor,
        }
    )
    out = out.astype(
        {
            "snapshot_date": "str",
            "symbol": "str",
            "ts_hms": "int32",
            "price": "int32",
            "volume": "int64",
            "trade_strength": "Float32",
            "ask1": "Int32",
            "bid1": "Int32",
            "truncated": "bool",
            "vendor": "str",
        }
    )
    return out[list(CANONICAL_TICK_COLUMNS)]


def assert_canonical_bars(df: pd.DataFrame) -> None:
    """정규 바 컬럼 집합/순서가 아니면 offending 컬럼을 명시하며 ValueError."""
    got = list(df.columns)
    if got == list(CANONICAL_BAR_COLUMNS):
        return
    missing = [c for c in CANONICAL_BAR_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in CANONICAL_BAR_COLUMNS]
    raise ValueError(f"Non-canonical bar frame: missing={missing} extra={extra}")


def assert_canonical_ticks(df: pd.DataFrame) -> None:
    """정규 틱 컬럼 집합/순서가 아니면 offending 컬럼을 명시하며 ValueError."""
    got = list(df.columns)
    if got == list(CANONICAL_TICK_COLUMNS):
        return
    missing = [c for c in CANONICAL_TICK_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in CANONICAL_TICK_COLUMNS]
    raise ValueError(f"Non-canonical tick frame: missing={missing} extra={extra}")
