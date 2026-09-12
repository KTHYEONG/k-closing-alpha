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

_BAR_VENDORS: tuple[str, ...] = ("kis", "ls", "kiwoom", "toss")

VENDOR_BUSINESS_DATE_FIELDS: dict[str, str] = {"kis": "stck_bsop_date", "ls": "date", "kiwoom": "cntr_tm", "toss": "timestamp"}

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

_KIWOOM_BAR_REQUIRED: tuple[str, ...] = (
    "cntr_tm",
    "cur_prc",
    "open_pric",
    "high_pric",
    "low_pric",
    "trde_qty",
)

_TOSS_BAR_REQUIRED: tuple[str, ...] = (
    "timestamp",
    "openPrice",
    "highPrice",
    "lowPrice",
    "closePrice",
    "volume",
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


_KIWOOM_TICK_REQUIRED: tuple[str, ...] = (
    "cntr_tm",
    "cur_prc",
    "trde_qty",
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
    if vendor not in ("kis", "ls", "kiwoom"):
        raise ValueError(f"Unknown intraday vendor: {vendor!r} (expected one of 'kis', 'ls', 'kiwoom')")
    return vendor


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], vendor: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required {vendor} source columns: {missing}")


def extract_vendor_business_dates(df: pd.DataFrame, vendor: str) -> pd.Series | None:
    """벤더 원천 프레임에서 YYYYMMDD 문자열 Series를 추출한다."""
    if vendor not in VENDOR_BUSINESS_DATE_FIELDS:
        raise ValueError(f"Unknown intraday vendor: {vendor!r} (expected one of 'kis', 'ls', 'kiwoom')")
    field = VENDOR_BUSINESS_DATE_FIELDS[vendor]
    if field not in df.columns:
        return None
    return df[field].astype(str).str.replace("-", "", regex=False).str[:8]


def filter_to_business_date(df: pd.DataFrame, vendor: str, snapshot_date: str, symbol: str) -> pd.DataFrame:
    """요청 snapshot_date와 벤더 영업일이 일치하는 행만 남긴다."""
    dates = extract_vendor_business_dates(df, vendor)
    if dates is None:
        logger.warning(
            "[DATA] stage=business_date_gate symbol=%s snapshot=%s date_verified=false reason=missing_field rows=%d",
            symbol,
            snapshot_date,
            len(df),
        )
        return df
    target = str(snapshot_date).replace("-", "")[:8]
    mask = dates.astype(str) == target
    if bool(mask.all()):
        return df
    dropped = int((~mask).sum())
    observed = sorted(set(dates[~mask].astype(str).tolist()))
    logger.warning(
        "[DATA] stage=business_date_gate symbol=%s snapshot=%s dropped=%d observed=%s",
        symbol,
        snapshot_date,
        dropped,
        observed,
    )
    return df[mask].copy()


def normalize_bar_frame(df: pd.DataFrame, vendor: str, snapshot_date: str, symbol: str) -> pd.DataFrame:
    """벤더 원천 분봉 프레임을 정규 바 스키마로 변환한다."""
    if vendor not in ("kis", "ls", "kiwoom", "toss"):
        raise ValueError(f"Unknown intraday vendor: {vendor!r} (expected one of 'kis', 'ls', 'kiwoom', 'toss')")
    if df is None or len(df) == 0:
        return _empty_bar_frame()
    df = filter_to_business_date(df, vendor, snapshot_date, symbol)
    if len(df) == 0:
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
    elif vendor == "ls":
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
    elif vendor == "kiwoom":
        _require_columns(df, _KIWOOM_BAR_REQUIRED, vendor)
        work = pd.DataFrame(
            {
                "ts_hms": pd.to_numeric(df["cntr_tm"].astype(str).str[-6:], errors="coerce"),
                "open": pd.to_numeric(df["open_pric"].astype(str), errors="coerce").abs(),
                "high": pd.to_numeric(df["high_pric"].astype(str), errors="coerce").abs(),
                "low": pd.to_numeric(df["low_pric"].astype(str), errors="coerce").abs(),
                "close": pd.to_numeric(df["cur_prc"].astype(str), errors="coerce").abs(),
                "volume": pd.to_numeric(df["trde_qty"].astype(str), errors="coerce").abs(),
            }
        )
        value_krw = work["close"] * work["volume"]
    elif vendor == "toss":
        _require_columns(df, _TOSS_BAR_REQUIRED, vendor)
        hms = df["timestamp"].astype(str).str.split("T").str[1].str.split(".").str[0].str.replace(":", "", regex=False)
        work = pd.DataFrame(
            {
                "ts_hms": pd.to_numeric(hms, errors="coerce"),
                "open": pd.to_numeric(df["openPrice"].astype(str), errors="coerce"),
                "high": pd.to_numeric(df["highPrice"].astype(str), errors="coerce"),
                "low": pd.to_numeric(df["lowPrice"].astype(str), errors="coerce"),
                "close": pd.to_numeric(df["closePrice"].astype(str), errors="coerce"),
                "volume": pd.to_numeric(df["volume"].astype(str), errors="coerce"),
            }
        )
        # Toss 캔들 응답엔 봉당 거래대금 필드가 없다 -- Kiwoom 분기와 동일하게 close*volume으로 근사한다.
        value_krw = work["close"] * work["volume"]

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
    if vendor == "kiwoom":
        out["has_trade"] = out["has_trade"].astype(object)
    return out[list(CANONICAL_BAR_COLUMNS)]



def normalize_tick_frame(
    df: pd.DataFrame, vendor: str, snapshot_date: str, symbol: str, truncated: bool = False
) -> pd.DataFrame:
    """벤더 원천 틱 프레임을 정규 틱 스키마로 변환한다."""
    _check_vendor(vendor)
    if df is None or len(df) == 0:
        out = _empty_tick_frame()
        return out
    df = filter_to_business_date(df, vendor, snapshot_date, symbol)
    if len(df) == 0:
        return _empty_tick_frame()
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
    elif vendor == "kiwoom":
        _require_columns(df, _KIWOOM_TICK_REQUIRED, vendor)
        ts_hms = pd.to_numeric(df["cntr_tm"].astype(str).str[-6:], errors="coerce")
        price = pd.to_numeric(df["cur_prc"].astype(str), errors="coerce").abs()
        volume = pd.to_numeric(df["trde_qty"].astype(str), errors="coerce").abs()
        trade_strength = pd.Series(pd.NA, index=df.index, dtype="Float32")
        ask1 = pd.Series(pd.NA, index=df.index, dtype="Int32")
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
