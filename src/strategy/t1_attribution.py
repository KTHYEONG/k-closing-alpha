"""Observe-only evaluation ledger for the full rank pool.

Scores every rank-pool row on the label exit (decision-day close to the next
global trading day open) and adds a TP5%+MOC counterfactual arm. No orders are
placed and model selection is never changed.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.ml.costaware_topk import compute_net_return
from src.ml.exit_policy import simulate_take_profit_exit
from src.ml.retrain_gate import MIN_NAMES_PER_EVAL_DAY
from src.strategy.contract import AA_COST
from src.strategy.growth_shadow import STATUS_PENDING, STATUS_REALIZED, realize_decision_returns

logger = logging.getLogger(__name__)

T1_ATTRIBUTION_PARQUET_NAME: str = "t1_attribution.parquet"
RANK_POOL_PARQUET_NAME: str = "rank_pool_predictions.parquet"
TP_COUNTERFACTUAL_RATIO: float = 0.05
KRX_DAILY_PRICE_LIMIT_RATIO: float = 0.30
STATUS_PRICE_DISCONTINUITY: str = "PRICE_DISCONTINUITY"
DAY_STATUS_SETTLED: str = "SETTLED"
DAY_STATUS_PENDING: str = "PENDING"
POOL_REQUIRED_COLUMNS: tuple[str, ...] = ("decision_date", "symbol", "pred", "tick_cost_bp", "admitted", "selected", "model_version")
PRICE_REQUIRED_COLUMNS: tuple[str, ...] = ("date", "symbol", "open", "high", "close")
REALIZED_COLUMNS: tuple[str, ...] = ("decision_date", "symbol", "model_version", "admitted", "selected", "pred", "rank", "status", "open_gross", "open_net", "tp_gross", "tp_net")
T1_LEDGER_COLUMNS: tuple[str, ...] = (
    "decision_date", "model_version", "day_status",
    "n_pool", "n_pool_realized", "n_admitted", "n_admitted_realized", "n_selected", "n_selected_realized",
    "ic_pool_net", "ic_admitted_net",
    "selected_open_net", "selected_tp_net", "admitted_open_net", "selection_edge_net",
)


def _entry_and_next_bars(
    decision_day: np.ndarray, symbol: np.ndarray, price_history: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ph_symbol = price_history["symbol"].astype(str).str.zfill(6).to_numpy()
    ph_day = pd.to_datetime(price_history["date"]).dt.normalize().to_numpy()
    idx = pd.MultiIndex.from_arrays([ph_symbol, ph_day])
    bars = {c: pd.Series(pd.to_numeric(price_history[c], errors="coerce").astype("float64").to_numpy(), index=idx) for c in ("open", "high", "close")}
    bars = {c: s[~s.index.duplicated(keep="last")] for c, s in bars.items()}
    entry_close = bars["close"].reindex(pd.MultiIndex.from_arrays([symbol, decision_day])).to_numpy(dtype=np.float64)
    # 전역 달력 기준 익일: 심볼별 다음 봉이 아니라 전체 패널의 다음 거래일
    calendar = np.sort(pd.unique(ph_day))
    pos = np.searchsorted(calendar, decision_day, side="right")
    has_next = pos < len(calendar)
    out = {c: np.full(len(symbol), np.nan, dtype=np.float64) for c in ("open", "high", "close")}
    if has_next.any():
        nk = pd.MultiIndex.from_arrays([symbol[has_next], calendar[pos[has_next]]])
        for c in out:
            out[c][has_next] = bars[c].reindex(nk).to_numpy(dtype=np.float64)
    return entry_close, out["open"], out["high"], out["close"]


def realize_pool_exit_arms(pool: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    """Score every pool row on the label exit and the TP counterfactual.

    Args:
        pool: Rank-pool rows with POOL_REQUIRED_COLUMNS.
        price_history: Ingested bars with PRICE_REQUIRED_COLUMNS.

    Returns:
        Per-row frame with REALIZED_COLUMNS in order.

    Raises:
        ValueError: When required columns are missing, (decision_date, symbol)
            rows repeat, or a realized move breaks the price limit.
    """
    miss = [c for c in POOL_REQUIRED_COLUMNS if c not in pool.columns]
    if miss:
        raise ValueError(f"realize_pool_exit_arms pool missing columns: {miss}")
    miss = [c for c in PRICE_REQUIRED_COLUMNS if c not in price_history.columns]
    if miss:
        raise ValueError(f"realize_pool_exit_arms price_history missing columns: {miss}")
    work = pool.reset_index(drop=True)
    symbol = work["symbol"].astype(str).str.zfill(6).to_numpy()
    day = pd.to_datetime(work["decision_date"]).dt.normalize().to_numpy()
    dup = pd.DataFrame({"d": day, "s": symbol}).duplicated()
    if dup.any():
        raise ValueError(f"realize_pool_exit_arms duplicate (decision_date, symbol) rows: {int(dup.sum())}")
    base = realize_decision_returns(work[["decision_date", "symbol", "pred", "tick_cost_bp"]], price_history[["date", "symbol", "open", "close"]])
    status = base["status"].to_numpy(dtype=object).copy()
    open_gross = base["gross_return"].to_numpy(dtype=np.float64).copy()
    open_net = base["net_return"].to_numpy(dtype=np.float64).copy()
    entry, nd_open, nd_high, nd_close = _entry_and_next_bars(day, symbol, price_history)
    tp_gross = np.full(len(work), np.nan, dtype=np.float64)
    ok = (status == STATUS_REALIZED) & np.isfinite(nd_high) & (nd_high > 0) & np.isfinite(nd_close) & (nd_close > 0)
    if ok.any():
        tp_gross[ok] = simulate_take_profit_exit(entry[ok], nd_open[ok], nd_high[ok], nd_close[ok], take_profit_pct=TP_COUNTERFACTUAL_RATIO, fallback="moc")
    tp_net = compute_net_return(pd.DataFrame({"gross_return": tp_gross, "tick_cost_bp": work["tick_cost_bp"].to_numpy(dtype=np.float64), "date": pd.to_datetime(day)}), round_trip_ticks=float(AA_COST.round_trip_ticks))
    tp_net = np.asarray(tp_net, dtype=np.float64)
    realized = status == STATUS_REALIZED
    # 비수정 가격(액면분할 등) 규칙: ±30% 초과 실현 수익은 PRICE_DISCONTINUITY
    disc = realized & ((np.abs(open_gross) > KRX_DAILY_PRICE_LIMIT_RATIO) | (np.abs(np.nan_to_num(tp_gross)) > KRX_DAILY_PRICE_LIMIT_RATIO))
    status[disc] = STATUS_PRICE_DISCONTINUITY
    not_realized = status != STATUS_REALIZED
    for arr in (open_gross, open_net, tp_gross, tp_net):
        arr[not_realized] = np.nan
    return pd.DataFrame({
        "decision_date": pd.to_datetime(day), "symbol": symbol, "model_version": work["model_version"].astype(str).to_numpy(),
        "admitted": work["admitted"].fillna(False).astype(bool).to_numpy(), "selected": work["selected"].fillna(False).astype(bool).to_numpy(),
        "pred": pd.to_numeric(work["pred"], errors="coerce").to_numpy(dtype=np.float64), "rank": base["rank"].to_numpy(dtype="int64"),
        "status": status, "open_gross": open_gross, "open_net": open_net, "tp_gross": tp_gross, "tp_net": tp_net,
    })[list(REALIZED_COLUMNS)]


def _daily_rank_ic(pred: np.ndarray, ret: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(ret)
    if int(ok.sum()) < MIN_NAMES_PER_EVAL_DAY:
        return float("nan")
    value = pd.Series(pred[ok]).rank().corr(pd.Series(ret[ok]).rank())
    return float(value) if np.isfinite(value) else float("nan")


def _basket_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    chosen = values[mask]
    return float(chosen.mean()) if np.isfinite(chosen).all() else float("nan")


def build_attribution_ledger(realized: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized rows into the daily attribution ledger.

    Args:
        realized: Output of realize_pool_exit_arms.

    Returns:
        Daily ledger with T1_LEDGER_COLUMNS in order and a reset index.

    Raises:
        ValueError: When one decision_date mixes model versions.
    """
    rows = []
    for d, g in realized.groupby("decision_date", sort=True):
        versions = sorted(set(g["model_version"].astype(str)))
        if len(versions) != 1:
            raise ValueError(f"build_attribution_ledger decision_date {pd.Timestamp(d).date()} mixes model versions: {versions}")
        st = g["status"].to_numpy(dtype=object)
        rl = st == STATUS_REALIZED
        adm = g["admitted"].to_numpy(dtype=bool)
        sel = g["selected"].to_numpy(dtype=bool)
        pred = g["pred"].to_numpy(dtype=np.float64)
        on = g["open_net"].to_numpy(dtype=np.float64)
        tn = g["tp_net"].to_numpy(dtype=np.float64)
        # PENDING 행이 하나라도 있으면 당일은 미확정: 지표 없이 개수만 기록
        pending = bool((st == STATUS_PENDING).any())
        nan = float("nan")
        row = {"decision_date": pd.Timestamp(d), "model_version": versions[0], "day_status": DAY_STATUS_PENDING if pending else DAY_STATUS_SETTLED,
               "n_pool": len(g), "n_pool_realized": int(rl.sum()), "n_admitted": int(adm.sum()), "n_admitted_realized": int((adm & rl).sum()),
               "n_selected": int(sel.sum()), "n_selected_realized": int((sel & rl).sum())}
        if pending:
            row.update(ic_pool_net=nan, ic_admitted_net=nan, selected_open_net=nan, selected_tp_net=nan, admitted_open_net=nan, selection_edge_net=nan)
        else:
            # 부분 바스켓 금지: 선택 종목이 하나라도 미실현이면 바스켓 평균은 NaN
            s_open = _basket_mean(on, sel)
            a_open = float(np.nanmean(on[adm & rl])) if (adm & rl).any() else nan
            row.update(ic_pool_net=_daily_rank_ic(pred, on), ic_admitted_net=_daily_rank_ic(pred[adm], on[adm]),
                       selected_open_net=s_open, selected_tp_net=_basket_mean(tn, sel), admitted_open_net=a_open, selection_edge_net=s_open - a_open)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(T1_LEDGER_COLUMNS))
    return pd.DataFrame(rows)[list(T1_LEDGER_COLUMNS)].reset_index(drop=True)


def _mean_sd_t(values: np.ndarray) -> tuple[int, float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if n < 2:
        return n, mean, float("nan"), float("nan")
    sd = float(v.std(ddof=1))
    t = mean / sd * math.sqrt(n) if sd > 0 else float("nan")
    return n, mean, sd, t


def summarize_attribution(ledger: pd.DataFrame) -> dict[str, float]:
    """Summarize the ledger into day counts, mean/sd/t and the exit gap.

    Args:
        ledger: Output of build_attribution_ledger.

    Returns:
        Dict with ic_pool_net_days/mean/sd/t, ic_admitted_net_days/mean/sd/t,
        exit_compare_days, selected_open_net_mean_bp, selected_tp_net_mean_bp,
        tp_minus_open_mean_bp and tp_minus_open_t.
    """
    out: dict[str, float] = {}
    for key in ("ic_pool_net", "ic_admitted_net"):
        n, m, sd, t = _mean_sd_t(ledger[key])
        out[f"{key}_days"], out[f"{key}_mean"], out[f"{key}_sd"], out[f"{key}_t"] = n, m, sd, t
    both = ledger["selected_open_net"].to_numpy(dtype=np.float64), ledger["selected_tp_net"].to_numpy(dtype=np.float64)
    ok = np.isfinite(both[0]) & np.isfinite(both[1])
    n, m, sd, t = _mean_sd_t(both[1][ok] - both[0][ok])
    out["exit_compare_days"] = n
    out["selected_open_net_mean_bp"] = float(both[0][ok].mean() * 1e4) if n else float("nan")
    out["selected_tp_net_mean_bp"] = float(both[1][ok].mean() * 1e4) if n else float("nan")
    out["tp_minus_open_mean_bp"] = m * 1e4 if n else float("nan")
    out["tp_minus_open_t"] = t
    return out


def run_t1_attribution(
    pool_path: Path | None = None, price_history_path: Path | None = None, out_path: Path | None = None
) -> int:
    """Recompute the t1 attribution ledger from the full pool store.

    Args:
        pool_path: Rank-pool parquet; defaults to settings.PARQUET_DIR.
        price_history_path: Price history parquet; defaults to settings path.
        out_path: Ledger destination; defaults to settings.PARQUET_DIR.

    Returns:
        Number of ledger rows written, or 0 when the pool store is absent.
    """
    src_path = Path(settings.PARQUET_DIR / RANK_POOL_PARQUET_NAME if pool_path is None else pool_path)
    ph_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if price_history_path is None else price_history_path)
    dest = Path(settings.PARQUET_DIR / T1_ATTRIBUTION_PARQUET_NAME if out_path is None else out_path)
    if not src_path.exists():
        logger.warning("[EVAL] stage=t1_attribution status=NO_POOL path=%s", src_path)
        return 0
    # 야간 전량 재계산: 필요한 컬럼만 읽고 atomic write로 멱등 교체
    pool = pd.read_parquet(src_path, columns=list(POOL_REQUIRED_COLUMNS))
    ph = pd.read_parquet(ph_path, columns=list(PRICE_REQUIRED_COLUMNS))
    ledger = build_attribution_ledger(realize_pool_exit_arms(pool, ph))
    atomic_write_parquet(ledger, dest)
    s = summarize_attribution(ledger)
    logger.info(
        "[EVAL] stage=t1_attribution rows=%d ic_pool_days=%d ic_pool_mean=%.4f ic_pool_t=%.2f ic_admitted_days=%d ic_admitted_mean=%.4f ic_admitted_t=%.2f exit_compare_days=%d selected_open_net_bp=%.2f selected_tp_net_bp=%.2f tp_minus_open_bp=%.2f tp_minus_open_t=%.2f",
        len(ledger), s["ic_pool_net_days"], s["ic_pool_net_mean"], s["ic_pool_net_t"], s["ic_admitted_net_days"], s["ic_admitted_net_mean"], s["ic_admitted_net_t"],
        s["exit_compare_days"], s["selected_open_net_mean_bp"], s["selected_tp_net_mean_bp"], s["tp_minus_open_mean_bp"], s["tp_minus_open_t"],
    )
    return len(ledger)
