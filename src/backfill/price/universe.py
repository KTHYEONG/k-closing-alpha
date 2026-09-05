"""종목 후보 우주(universe) 및 날짜 윈도우 결정."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backfill.price.config import DEFAULT_CONFIG, FetchConfig
from src.data.candidate_panel import load_candidate_universe_symbols

logger = logging.getLogger(__name__)


def load_or_build_snapshot(*args, **kwargs) -> pd.DataFrame:
    """종목 후보 우주(symbol/market) 스냅샷을 로컬 DB에서 로드합니다.

    레거시 `src.pipeline.data.load_or_build_snapshot`을 대체하며,
    `table_trade_log`에서 symbol/market 컬럼을 구성합니다.
    """
    from src.data.db_loader import load_trade_log_from_db

    try:
        raw = load_trade_log_from_db()
    except FileNotFoundError:
        return pd.DataFrame(columns=["symbol", "market"])
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["symbol", "market"])

    rename = {}
    for col in raw.columns:
        if "종목코드" in str(col):
            rename[col] = "symbol"
        elif "시장구분" in str(col) or "시장" in str(col):
            rename[col] = "market"
    if "symbol" not in rename.values():
        return pd.DataFrame(columns=["symbol", "market"])
    out = raw.rename(columns=rename)
    if "market" not in out.columns:
        out["market"] = np.nan
    return out[["symbol", "market"]]


def _load_candidate_universe() -> pd.DataFrame:
    raw = load_or_build_snapshot(config=DEFAULT_CONFIG, rebuild=False, sync_gsheet=False)
    try:
        extra = load_candidate_universe_symbols()
    except Exception as exc:
        logger.warning("[DATA] stage=candidate_universe status=fallback_to_trade_log error=%s", exc)
        extra = pd.DataFrame(columns=["symbol", "market"])
    if extra is not None and not extra.empty:
        raw = pd.concat([raw, extra], ignore_index=True, sort=False)
    required = ["symbol", "market"]
    for col in required:
        if col not in raw.columns:
            raw[col] = np.nan
    out = raw[required].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.zfill(6)
    out["market"] = out["market"].astype(str).fillna("UNKNOWN")
    out = out[out["symbol"].str.fullmatch(r"\d{6}", na=False)].copy()
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out


def _build_symbol_windows(
    universe: pd.DataFrame,
    fetch_cfg: FetchConfig,
    symbol_limit: int | None = None,
    include_symbols: set[str] | None = None,
    existing_last_dates: dict[str, pd.Timestamp] | None = None,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    by_symbol = universe[["symbol"]].dropna().drop_duplicates().sort_values("symbol")
    market_hint = (
        universe[["symbol", "market"]]
        .dropna()
        .drop_duplicates(subset=["symbol"], keep="last")
        .set_index("symbol")["market"]
        .to_dict()
    )

    if symbol_limit is not None and symbol_limit > 0:
        by_symbol = by_symbol.head(symbol_limit)
    if include_symbols:
        wanted = {str(s).strip().zfill(6) for s in include_symbols if str(s).strip()}
        by_symbol = by_symbol[by_symbol["symbol"].isin(wanted)].copy()
    else:
        wanted = set()

    rows: list[tuple[str, pd.Timestamp, pd.Timestamp, str]] = []
    start_fixed = pd.Timestamp(fetch_cfg.fixed_start_date)
    end_fixed = min(pd.Timestamp(fetch_cfg.fixed_end_date), pd.Timestamp.today().normalize())
    # 신규 종목은 CLI의 lookback 범위만 요청한다. 기존 종목은 아래의
    # 증분 갱신 경로에서 fixed_start_date와 마지막 저장일을 기준으로 처리한다.
    recent_start = max(
        start_fixed,
        end_fixed - pd.offsets.BDay(max(1, int(fetch_cfg.lookback_trading_days))),
    )
    if start_fixed > end_fixed:
        start_fixed, end_fixed = end_fixed, start_fixed

    existing = existing_last_dates or {}
    overlap_days = max(20, int(fetch_cfg.calendar_buffer_days))

    def _resolve_start(symbol: str) -> pd.Timestamp:
        if fetch_cfg.force_full_history:
            return start_fixed
        last_dt = existing.get(str(symbol))
        if last_dt is None:
            return recent_start
        last_ts = pd.to_datetime(last_dt, errors="coerce")
        if pd.isna(last_ts):
            return start_fixed
        return max(start_fixed, pd.Timestamp(last_ts).normalize() - pd.Timedelta(days=overlap_days))

    for rec in by_symbol.itertuples(index=False):
        symbol = str(rec.symbol)
        start_ts = _resolve_start(symbol)
        if start_ts > end_fixed:
            continue
        rows.append((symbol, start_ts, end_fixed, str(market_hint.get(symbol, "UNKNOWN"))))

    # If a user-requested symbol is missing in snapshot universe, still build a fallback
    # range so manual/backfill test can proceed.
    if wanted:
        existing_symbols = {r[0] for r in rows}
        missing = sorted(wanted - existing_symbols)
        if missing:
            for sym in missing:
                start_ts = _resolve_start(sym)
                if start_ts > end_fixed:
                    continue
                rows.append((sym, start_ts, end_fixed, "UNKNOWN"))
    return rows
