"""Point-in-time per-symbol history and cost features for the top-k cost-aware ranker.

The ranker's label is the overnight return (decision close -> D+1 open), so the
history block centres on each symbol's own overnight-gap record. Every window is
[t-w, t-1] or a value already observed at the 15:20 decision (today's open and
prev_close, today's provisional flows), so research and serving share one function.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.history_features import _group_rolling
from src.ml.research.v3_engine import FEATURE_COLS
from src.strategy.contract import DEFAULT_UNIVERSE, derive_chg_ratio

TOPK_COST_FEATURE_COLS: tuple[str, ...] = ("f_tick_cost", "f_log_close")
TOPK_HISTORY_FEATURE_COLS: tuple[str, ...] = (
    "f_ret5",
    "f_ret20",
    "f_ret60",
    "f_dist_high60",
    "f_upcount20",
    "f_on_mean20",
    "f_on_mean60",
    "f_upnext_on60",
    "f_gap",
    "f_id_mean20",
    "f_inst_cum5",
    "f_foreign_cum5",
)
# 순서 고정: colsample_bytree 가 열 순서에 의존하므로 인증 순서를 그대로 유지한다.
TOPK_FEATURE_COLS_V2: list[str] = [*FEATURE_COLS, *TOPK_COST_FEATURE_COLS, *TOPK_HISTORY_FEATURE_COLS]
HISTORY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "open",
    "close",
    "prev_close",
    "volume",
    "inst_netbuy",
    "foreign_netbuy",
)
# 상승일 판정 = 스크린 진입 하한과 동일한 등락률
UP_DAY_THRESHOLD: float = DEFAULT_UNIVERSE.chg_min
# 조건부 익일갭 평균의 최소 관측 수
MIN_CONDITIONAL_OBS: int = 3
# 60거래일 창 + 1일 시프트를 달력일로 덮는 서빙 조회 폭 (약 135거래일)
HISTORY_LOOKBACK_CALENDAR_DAYS: int = 200
# 설·추석 연휴를 넘는 직전 거래일 탐색 한도 (달력일)
MAX_PREV_TRADING_DAY_LOOKBACK: int = 15
_LIVE_REQUIRED_COLUMNS: tuple[str, ...] = tuple(c for c in HISTORY_REQUIRED_COLUMNS if c != "date")


def _lag_roll(s: pd.Series, labels: pd.Series, window: int, func: str) -> pd.Series:
    # [t-w, t-1] 창: 당일 값을 제외하고 과거만 집계
    lagged = s.groupby(labels.to_numpy(), sort=False).shift(1)
    return _group_rolling(lagged, labels, window, max(3, window // 2), func)


def compute_topk_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute point-in-time per-symbol history features for every panel row.

    Args:
        panel: Daily rows with HISTORY_REQUIRED_COLUMNS; one row per (date, symbol).

    Returns:
        Frame with date, symbol and TOPK_HISTORY_FEATURE_COLS, one row per panel
        row, sorted by (symbol, date). Undefined windows are NaN.

    Raises:
        ValueError: When a required column is missing or (date, symbol) repeats.
    """
    missing = [c for c in HISTORY_REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    p = panel[list(HISTORY_REQUIRED_COLUMNS)].copy()
    p["date"] = pd.to_datetime(p["date"])
    p["symbol"] = p["symbol"].astype(str)
    dup = int(p.duplicated(["date", "symbol"]).sum())
    if dup:
        raise ValueError(f"panel carries {dup} duplicate (date, symbol) rows")
    p = p.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    for c in ("open", "close", "prev_close", "volume", "inst_netbuy", "foreign_netbuy"):
        p[c] = pd.to_numeric(p[c], errors="coerce").astype("float64")
    labels = p["symbol"]
    op = p["open"].to_numpy()
    cl = p["close"].to_numpy()
    pc = p["prev_close"].to_numpy()
    # 등락률은 전일종가 기준으로 재유도 → 액면분할 등 기준가 조정에 강건
    r = pd.Series(derive_chg_ratio(cl, pc), index=p.index)
    lr = np.log1p(r)
    # 시가 0(무체결 정지일)은 -100% 갭이 아니라 결측
    on_ok = np.isfinite(op) & np.isfinite(pc) & (op > 0) & (pc > 0)
    on = pd.Series(np.where(on_ok, op / np.where(pc > 0, pc, 1.0) - 1.0, np.nan), index=p.index)
    id_ok = np.isfinite(op) & np.isfinite(cl) & (op > 0) & (cl > 0)
    idr = pd.Series(np.where(id_ok, cl / np.where(op > 0, op, 1.0) - 1.0, np.nan), index=p.index)
    val = p["close"] * p["volume"]

    def _g(s: pd.Series) -> pd.core.groupby.SeriesGroupBy:
        return s.groupby(labels.to_numpy(), sort=False)

    out = p[["date", "symbol"]].copy()
    out["f_ret5"] = _lag_roll(lr, labels, 5, "sum")
    out["f_ret20"] = _lag_roll(lr, labels, 20, "sum")
    out["f_ret60"] = _lag_roll(lr, labels, 60, "sum")
    # 누적 로그수익 지수의 60일 고점 대비 거리 (시작점 무관)
    idx = _g(lr).cumsum()
    out["f_dist_high60"] = np.expm1(idx - _group_rolling(idx, labels, 60, 20, "max"))
    up = (r >= UP_DAY_THRESHOLD).astype("float64")
    up_sum = _group_rolling(_g(up).shift(1), labels, 20, 5, "sum")
    valid = _lag_roll(lr, labels, 20, "count")
    out["f_upcount20"] = up_sum.where(valid.notna())
    out["f_on_mean20"] = _lag_roll(on, labels, 20, "mean")
    out["f_on_mean60"] = _lag_roll(on, labels, 60, "mean")
    # 과거 상승일(s<=t-1) 다음날 시가갭: s+1<=t 이므로 결정시점에 이미 관측됨
    cond = _g(on).shift(-1).where(r >= UP_DAY_THRESHOLD)
    cond_lag = _g(cond).shift(1)
    cmean = _group_rolling(cond_lag, labels, 60, 1, "mean")
    ccnt = _group_rolling(cond_lag, labels, 60, 1, "count")
    out["f_upnext_on60"] = cmean.where(ccnt >= MIN_CONDITIONAL_OBS)
    out["f_gap"] = on
    out["f_id_mean20"] = _lag_roll(idr, labels, 20, "mean")
    # 5일 순매수 금액 / 5일 거래금액 (당일 포함, 결정시점 잠정치)
    den = _group_rolling(val, labels, 5, 3, "sum").replace(0.0, np.nan)
    out["f_inst_cum5"] = (_group_rolling(p["inst_netbuy"], labels, 5, 3, "sum") / den).clip(-1.0, 1.0)
    out["f_foreign_cum5"] = (_group_rolling(p["foreign_netbuy"], labels, 5, 3, "sum") / den).clip(-1.0, 1.0)
    cols = list(TOPK_HISTORY_FEATURE_COLS)
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    return out[["date", "symbol", *cols]]


def attach_topk_features(cands: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join history features and derive cost features onto candidate rows.

    Args:
        cands: Candidate rows with date, symbol, close and tick_cost_bp.
        panel: Daily panel covering the candidates' history (see compute_topk_history_features).

    Returns:
        Copy of cands (same index and row order, symbol cast to str) carrying
        TOPK_COST_FEATURE_COLS and TOPK_HISTORY_FEATURE_COLS.

    Raises:
        ValueError: When a required candidate column is missing, or propagated
            from compute_topk_history_features.
    """
    missing = [c for c in ("date", "symbol", "close", "tick_cost_bp") if c not in cands.columns]
    if missing:
        raise ValueError(f"cands missing required columns: {missing}")
    hist = compute_topk_history_features(panel)
    out = cands.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    merged = out.merge(hist, on=["date", "symbol"], how="left", validate="many_to_one")
    # left merge 는 행 순서를 보존하므로 원 인덱스를 그대로 복원한다 (sel_mask 위치 정합)
    merged.index = cands.index
    close = pd.to_numeric(merged["close"], errors="coerce").to_numpy(dtype=np.float64)
    merged["f_tick_cost"] = pd.to_numeric(merged["tick_cost_bp"], errors="coerce").astype("float64")
    merged["f_log_close"] = np.log(np.where(close > 0, close, np.nan))
    return merged


def stitch_live_panel(price_history: pd.DataFrame, live_rows: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    """Append the decision-day live rows to strictly-past price history.

    Args:
        price_history: Historical daily rows with HISTORY_REQUIRED_COLUMNS.
        live_rows: Decision-day rows with every required column except date.
        decision_date: Decision date stamped on the live rows.

    Returns:
        Panel with HISTORY_REQUIRED_COLUMNS: history rows dated before
        decision_date followed by the live rows.

    Raises:
        ValueError: When live_rows is empty, repeats a symbol, or either frame
            misses a required column.
    """
    missing = [c for c in _LIVE_REQUIRED_COLUMNS if c not in live_rows.columns]
    if missing:
        raise ValueError(f"live_rows missing required columns: {missing}")
    if live_rows.empty:
        raise ValueError("live_rows is empty; nothing to decide on")
    if live_rows["symbol"].astype(str).duplicated().any():
        raise ValueError("live_rows carries duplicate symbols")
    missing_h = [c for c in HISTORY_REQUIRED_COLUMNS if c not in price_history.columns]
    if missing_h:
        raise ValueError(f"price_history missing required columns: {missing_h}")
    d = pd.Timestamp(decision_date).normalize()
    # 결정일 이후 이력은 PIT 위반이므로 버리고 라이브 행으로 대체
    hist = price_history[pd.to_datetime(price_history["date"]) < d][list(HISTORY_REQUIRED_COLUMNS)].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["symbol"] = hist["symbol"].astype(str)
    live = live_rows[list(_LIVE_REQUIRED_COLUMNS)].copy()
    live["symbol"] = live["symbol"].astype(str)
    live.insert(0, "date", d)
    return pd.concat([hist, live[list(HISTORY_REQUIRED_COLUMNS)]], ignore_index=True)


def resolve_prev_trading_day(
    decision_date: pd.Timestamp,
    is_trading_day: Callable[[pd.Timestamp], bool],
    *,
    max_lookback_days: int = MAX_PREV_TRADING_DAY_LOOKBACK,
) -> pd.Timestamp:
    """Return the latest trading day strictly before decision_date.

    Args:
        decision_date: Decision date (time component ignored).
        is_trading_day: Trading-day oracle; weekends are skipped without a call.
        max_lookback_days: Calendar-day search bound.

    Returns:
        The previous trading day, normalized to midnight.

    Raises:
        ValueError: When no trading day exists within the bound.
    """
    d = pd.Timestamp(decision_date).normalize()
    for k in range(1, int(max_lookback_days) + 1):
        cand = d - pd.Timedelta(days=k)
        if cand.weekday() >= 5:
            continue
        if is_trading_day(cand):
            return cand
    raise ValueError(f"no trading day within {max_lookback_days} calendar days before {d.date()}")


def assert_history_fresh(price_history: pd.DataFrame, prev_trading_day: pd.Timestamp) -> None:
    """Fail closed unless price_history reaches the previous trading day.

    Args:
        price_history: Historical rows with a date column.
        prev_trading_day: Latest trading day before the decision date.

    Returns:
        None when the history is fresh.

    Raises:
        ValueError: When the history is empty or ends before prev_trading_day.
    """
    if price_history.empty:
        raise ValueError("price_history is empty; history features cannot be computed")
    latest = pd.Timestamp(pd.to_datetime(price_history["date"]).max()).normalize()
    if latest < pd.Timestamp(prev_trading_day).normalize():
        raise ValueError(
            f"stale price_history: latest={latest.date()} < prev_trading_day={pd.Timestamp(prev_trading_day).date()}"
        )
    return None


def load_serving_price_history(
    decision_date: pd.Timestamp,
    *,
    path: str | os.PathLike[str] | None = None,
    is_trading_day: Callable[[pd.Timestamp], bool] | None = None,
) -> pd.DataFrame:
    """Read the serving lookback window of price_history and verify freshness.

    Args:
        decision_date: Decision date; rows on or after it are excluded.
        path: Parquet path; None selects settings.PRICE_HISTORY_PARQUET_PATH.
        is_trading_day: Trading-day oracle; None selects is_krx_trading_day,
            whose network failures propagate unchanged.

    Returns:
        HISTORY_REQUIRED_COLUMNS rows dated in
        [decision_date - HISTORY_LOOKBACK_CALENDAR_DAYS, decision_date).

    Raises:
        FileNotFoundError: When the parquet does not exist.
        ValueError: When the window is empty or ends before the previous trading day.
    """
    from src import settings

    src_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if path is None else path)
    if not src_path.exists():
        raise FileNotFoundError(f"price_history not found: {src_path}")
    if is_trading_day is None:
        from src.data.trading_calendar import is_krx_trading_day as is_trading_day
    d = pd.Timestamp(decision_date).normalize()
    start = d - pd.Timedelta(days=HISTORY_LOOKBACK_CALENDAR_DAYS)
    # 열 가지치기 + 날짜 조건 푸시다운으로 546만행 전체 적재 회피
    df = pd.read_parquet(
        src_path, columns=list(HISTORY_REQUIRED_COLUMNS), filters=[("date", ">=", start), ("date", "<", d)]
    )
    prev = resolve_prev_trading_day(d, is_trading_day)
    assert_history_fresh(df, prev)
    return df
