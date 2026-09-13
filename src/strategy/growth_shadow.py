"""Growth-shadow ledger for K=2 and trailing-gate forward validation.

The shadow is observe-only: it replays persisted top-3 decisions against the
ingested price history and records what a smaller (K=2) basket and a
trailing-120-day gated variant would have earned. Live selection is untouched
and no orders are ever placed.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.ml.costaware_topk import MIN_TOP_K, compute_net_return
from src.strategy.contract import AA_COST

logger = logging.getLogger(__name__)

SHADOW_K2_TOP_K: int = 2
TRAIL_GATE_WINDOW_DAYS: int = 120
STATUS_REALIZED: str = "REALIZED"
STATUS_PENDING: str = "PENDING"
STATUS_EXIT_UNAVAILABLE: str = "EXIT_UNAVAILABLE"
GROWTH_SHADOW_PARQUET_NAME: str = "growth_shadow.parquet"
LEDGER_COLUMNS: tuple[str, ...] = (
    "decision_date",
    "n_picks",
    "arm_k3_net",
    "arm_k2_net",
    "trail_mean_k3",
    "trail_gate_open",
    "arm_k3_trail_net",
)


def realize_decision_returns(decisions: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    """Realize one-night returns for persisted top-k decisions.

    Args:
        decisions: Persisted picks with decision_date, symbol, pred, tick_cost_bp.
        price_history: Ingested bars with date, symbol, open, close.

    Returns:
        Per-pick frame with decision_date, symbol, pred, rank, gross_return,
        net_return and status columns.

    Raises:
        ValueError: When a required column is missing on either frame.
    """
    # 필수 컬럼 누락은 fail-closed (요구 컬럼명을 메시지에 포함)
    missing_decisions = [c for c in ("decision_date", "symbol", "pred", "tick_cost_bp") if c not in decisions.columns]
    if missing_decisions:
        raise ValueError(f"realize_decision_returns decisions missing columns: {missing_decisions}")
    missing_history = [c for c in ("date", "symbol", "open", "close") if c not in price_history.columns]
    if missing_history:
        raise ValueError(f"realize_decision_returns price_history missing columns: {missing_history}")
    # 심볼 정규화: 숫자형/미패딩 코드와 카테고리 dtype도 6자리로 맞춤
    dec_symbol = decisions["symbol"].astype(str).str.zfill(6)
    ph_symbol = price_history["symbol"].astype(str).str.zfill(6)
    # 일자 정규화: 결정일 자정 기준, 스냅샷 시각이 섞여도 날짜 단위로 비교
    dec_day = pd.to_datetime(decisions["decision_date"]).dt.normalize()
    ph_day = pd.to_datetime(price_history["date"]).dt.normalize()
    # 가격 정규화: float64 수익률 정밀도, 비수치는 NaN으로 전파
    ph_open = pd.to_numeric(price_history["open"], errors="coerce").astype("float64")
    ph_close = pd.to_numeric(price_history["close"], errors="coerce").astype("float64")
    # 진입가: 결정일 확정종가, 청산가: 결정일 이후 첫 날짜의 시가(전역 달력 기준)
    close_by_key = pd.Series(
        ph_close.to_numpy(), index=pd.MultiIndex.from_arrays([ph_symbol.to_numpy(), ph_day.to_numpy()])
    )
    open_by_key = pd.Series(
        ph_open.to_numpy(), index=pd.MultiIndex.from_arrays([ph_symbol.to_numpy(), ph_day.to_numpy()])
    )
    entry_close = (
        close_by_key.reindex(pd.MultiIndex.from_arrays([dec_symbol.to_numpy(), dec_day.to_numpy()])).to_numpy(
            dtype=np.float64
        )
    )
    calendar = np.sort(ph_day.drop_duplicates().to_numpy())
    pos = np.searchsorted(calendar, dec_day.to_numpy(), side="right")
    has_next = pos < len(calendar)
    next_day = np.full(dec_day.shape[0], np.datetime64("NaT", "ns"))
    next_day[has_next] = calendar[pos[has_next]]
    exit_open = np.full(dec_day.shape[0], np.nan, dtype=np.float64)
    if has_next.any():
        exit_open[has_next] = (
            open_by_key.reindex(pd.MultiIndex.from_arrays([dec_symbol.to_numpy()[has_next], next_day[has_next]]))
            .to_numpy(dtype=np.float64)
        )
    # 상태: 다음 날짜 자체가 없으면 PENDING, 진입/청산가 결손이면 EXIT_UNAVAILABLE
    status = np.full(dec_day.shape[0], STATUS_REALIZED, dtype=object)
    status[~has_next] = STATUS_PENDING
    status[has_next & (~(entry_close > 0.0) | ~(exit_open > 0.0))] = STATUS_EXIT_UNAVAILABLE
    # 수익률: 확정종가 매수 → 익일 시가 청산, 비용은 공용 cost 모델에 위임
    gross = np.full(dec_day.shape[0], np.nan, dtype=np.float64)
    is_realized = status == STATUS_REALIZED
    gross[is_realized] = exit_open[is_realized] / entry_close[is_realized] - 1.0
    net = compute_net_return(
        pd.DataFrame(
            {
                "gross_return": gross,
                "tick_cost_bp": decisions["tick_cost_bp"].to_numpy(dtype=np.float64),
                "date": dec_day,
            }
        ),
        round_trip_ticks=float(AA_COST.round_trip_ticks),
    )
    # 순위: 결정일별 pred 내림차순, 동점은 입력 순서 우선
    rank = (
        pd.Series(decisions["pred"].to_numpy())
        .groupby(dec_day.to_numpy())
        .rank(ascending=False, method="first")
        .to_numpy(dtype="int64")
    )
    return pd.DataFrame(
        {
            "decision_date": dec_day,
            "symbol": dec_symbol.to_numpy(),
            "pred": decisions["pred"].to_numpy(),
            "rank": rank,
            "gross_return": gross,
            "net_return": np.asarray(net, dtype=np.float64),
            "status": status,
        }
    )


def build_shadow_ledger(realized: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized picks into the daily K3/K2/trailing-gate ledger.

    Args:
        realized: Output of realize_decision_returns.

    Returns:
        Daily ledger with LEDGER_COLUMNS in order and a reset index.

    Raises:
        ValueError: When a non-pending date does not hold exactly MIN_TOP_K picks.
    """
    # PENDING 날짜는 익일 시세가 없어 실현 불가 → 원장에서 제외
    work = realized.copy()
    work["decision_date"] = pd.to_datetime(work["decision_date"])
    pending_days = work.loc[work["status"] == STATUS_PENDING, "decision_date"].drop_duplicates()
    settled = work[~work["decision_date"].isin(pending_days)].copy()
    # 인증된 바스켓이 아니면 비교 불가 → 날짜를 명시하고 fail-closed
    counts = settled.groupby("decision_date", sort=True).size()
    bad_days = counts[counts != MIN_TOP_K]
    if len(bad_days):
        raise ValueError(
            f"build_shadow_ledger dates with n_picks != MIN_TOP_K ({MIN_TOP_K}): "
            f"{[str(d) for d in bad_days.index]}"
        )
    # K=2 섀도는 인증 top-3 중 pred 상위 2개의 부분집합
    k3_net = settled.groupby("decision_date", sort=True)["net_return"].mean(skipna=False)
    k2_net = (
        settled[settled["rank"] <= SHADOW_K2_TOP_K]
        .groupby("decision_date", sort=True)["net_return"]
        .mean(skipna=False)
        .reindex(k3_net.index)
    )
    # 미실현 픽이 하나라도 있으면 부분 바스켓 없이 양쪽 암 모두 NaN
    all_realized = (
        settled.groupby("decision_date", sort=True)["status"]
        .apply(lambda s: bool((s == STATUS_REALIZED).all()))
        .reindex(k3_net.index)
    )
    arms_ok = all_realized.to_numpy(dtype=bool)
    ledger = pd.DataFrame(
        {
            "decision_date": k3_net.index.to_numpy(),
            "n_picks": counts.reindex(k3_net.index).to_numpy(dtype="int64"),
            "arm_k3_net": np.where(arms_ok, k3_net.to_numpy(dtype=np.float64), np.nan),
            "arm_k2_net": np.where(arms_ok, k2_net.to_numpy(dtype=np.float64), np.nan),
        }
    )
    # 인과적 게이트: D-1 결정분은 D 09:00에 끝나므로 shift(1)만 당일 결정에 사용 가능
    trail = (
        ledger["arm_k3_net"].shift(1).rolling(TRAIL_GATE_WINDOW_DAYS, min_periods=TRAIL_GATE_WINDOW_DAYS).mean()
    )
    gate_open = trail.isna() | (trail > 0.0)
    # 웜업 120일은 게이트 개방, 음수 구간은 현금 0.0
    ledger["trail_mean_k3"] = trail.to_numpy(dtype=np.float64)
    ledger["trail_gate_open"] = gate_open.to_numpy(dtype=bool)
    ledger["arm_k3_trail_net"] = np.where(
        gate_open.to_numpy(), ledger["arm_k3_net"].to_numpy(dtype=np.float64), 0.0
    )
    return ledger[list(LEDGER_COLUMNS)].reset_index(drop=True)


def run_growth_shadow(
    decisions_path: Path | None = None,
    price_history_path: Path | None = None,
    out_path: Path | None = None,
) -> int:
    """Recompute the growth-shadow ledger from persisted decisions.

    Args:
        decisions_path: topk_decisions parquet; defaults to settings.PARQUET_DIR.
        price_history_path: Price history parquet; defaults to settings path.
        out_path: Ledger destination; defaults to settings.PARQUET_DIR.

    Returns:
        Number of ledger rows written, or 0 when no decisions exist yet.
    """
    dec_path = Path(settings.PARQUET_DIR / "topk_decisions.parquet" if decisions_path is None else decisions_path)
    ph_path = Path(settings.PRICE_HISTORY_PARQUET_PATH if price_history_path is None else price_history_path)
    dest = Path(settings.PARQUET_DIR / GROWTH_SHADOW_PARQUET_NAME if out_path is None else out_path)
    # 결정 기록이 없으면 쓸 원장이 없음 → 경고만 남기고 종료
    if not dec_path.exists():
        logger.warning("[PORTFOLIO] stage=growth_shadow status=NO_DECISIONS path=%s", dec_path)
        return 0
    decisions = pd.read_parquet(dec_path)
    price_history = pd.read_parquet(ph_path, columns=["date", "symbol", "open", "close"])
    # 매일 전체 재계산: 야간 배치는 작아 청크 없이 전량 처리
    ledger = build_shadow_ledger(realize_decision_returns(decisions, price_history))
    atomic_write_parquet(ledger, dest)
    arms = {c: ledger[c].to_numpy(dtype=np.float64) for c in ("arm_k3_net", "arm_k2_net", "arm_k3_trail_net")}
    # 유한 행 평균, 없으면 nan (빈 원장에서도 경고 없이 nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k3_mean_bp = float(np.nanmean(arms["arm_k3_net"]) * 10000.0)
        k2_mean_bp = float(np.nanmean(arms["arm_k2_net"]) * 10000.0)
        k3_trail_mean_bp = float(np.nanmean(arms["arm_k3_trail_net"]) * 10000.0)
        gate_open_share = float(np.nanmean(ledger["trail_gate_open"].to_numpy(dtype=bool).astype(np.float64)))
    realized_days = int(np.isfinite(arms["arm_k3_net"]).sum())
    logger.info(
        "[PORTFOLIO] stage=growth_shadow rows=%d realized_days=%d k3_mean_bp=%.2f "
        "k2_mean_bp=%.2f k3_trail_mean_bp=%.2f gate_open_share=%.3f",
        len(ledger),
        realized_days,
        k3_mean_bp,
        k2_mean_bp,
        k3_trail_mean_bp,
        gate_open_share,
    )
    return len(ledger)
