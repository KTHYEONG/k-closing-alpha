"""청산레버(TP5%+MOC) locked-OOS/CPCV(8,2) 재검증 -- 인증 리랭커와 동일 파이프라인 재사용.

exit_policy.evaluate_exit_grid는 이미 cv=CombinatorialPurgedCV를 지원하므로 별도
"현행 표준"에 맞춘 개조가 필요 없다(topk_ranker_research.CPCV_N_GROUPS/CPCV_K_TEST=8/2와
robust_eval.CombinatorialPurgedCV 기본값이 이미 동일). 유일한 공백은 캐시된 OOF가
없다는 것뿐이라, 인증 리랭커(run_topk_ranker_backtest)와 완전히 동일한
build_dual_pool -> attach_pit_net_label -> demean_label_by_date -> split_regime_frames
-> cpcv_score_with_history -> dedupe_cpcv_oof 파이프라인을 그대로 재사용해 OOF를
실제로 재학습 생성한다. run_topk_ranker_backtest/TopKRankerReport 자체는 건드리지
않는다(무변경, 별도 연구 스크립트로만 접근). cost_ratio는 동일 모집단의 실측
tick_cost_bp 평균(bp -> 소수 환산)을 쓴다 -- 별도 상수를 발명하지 않는다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io_utils import atomic_write_parquet
from src.ml.costaware_topk import MIN_TOP_K, assert_screen_constructible
from src.ml.exit_policy import DEFAULT_TAKE_PROFIT_GRID, evaluate_exit_grid, summarize_exit_grid
from src.ml.research.v3_engine import load_and_prepare_price_history
from src.ml.robust_eval import CombinatorialPurgedCV
from src.ml.topk_ranker_research import (
    CPCV_K_TEST,
    CPCV_N_GROUPS,
    RANKER_FEATURE_COLS,
    RANKER_SEEDS,
    TRAIN_POOL_MIN_ROWS,
    assert_unique_date_symbol,
    attach_pit_net_label,
    build_dual_pool,
    cpcv_score_with_history,
    dedupe_cpcv_oof,
    demean_label_by_date,
    split_regime_frames,
)
from src.strategy.contract import DEFAULT_UNIVERSE, KCA_TOPK_COSTAWARE_001, StrategySpec, UniverseSpec

logger = logging.getLogger(__name__)


def build_exit_grid_oof(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    spec: StrategySpec = KCA_TOPK_COSTAWARE_001,
    train_spec: UniverseSpec = DEFAULT_UNIVERSE,
    cv: CombinatorialPurgedCV | None = None,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    min_train_rows: int = TRAIN_POOL_MIN_ROWS,
    train_start: pd.Timestamp | None = None,
    seeds: tuple[int, ...] = RANKER_SEEDS,
) -> tuple[pd.DataFrame, float]:
    """인증 리랭커와 동일 파이프라인으로 청산그리드 재검증용 OOF+실측비용을 만든다.

    Args:
        ph: Prepared price-history panel.
        market_dates: Full trading calendar.
        d_to_idx: Date-to-index lookup for forward exits.
        spec: Strategy specification carrying top_k, select universe and cost.
        train_spec: Wide training screen without the cost cap.
        cv: CPCV splitter override; None lets cpcv_score_with_history default to (8, 2).
        model_params: LightGBM params override.
        huber_delta: Huber alpha for the ranker.
        min_train_rows: Fail-closed floor on per-fold training rows.
        train_start: Training-window start passed to split_regime_frames; None
            selects the panel minimum date (never moves the certification boundary).
        seeds: LightGBM seeds averaged per CPCV fold (same as the served bundle).

    Returns:
        Tuple of (oof_df shaped for exit_policy.evaluate_exit_grid with columns
        trade_date/stock_code/pred/net_return, cost_ratio derived as the mean
        real PIT tick_cost_bp -- over the same selected population -- converted
        from bp to a fraction).

    Raises:
        ValueError: When spec.top_k is below the investable minimum, or any
            downstream fail-closed check (screen constructibility, unique
            (date, symbol) pairs, per-fold minimum training rows) fails.
    """
    if int(spec.top_k) < MIN_TOP_K:
        raise ValueError(f"top_k {spec.top_k} below the minimum investable K {MIN_TOP_K}")
    eff_train_start = pd.Timestamp(pd.to_datetime(ph["date"]).min()) if train_start is None else pd.Timestamp(train_start)
    pool, sel_mask = build_dual_pool(ph, market_dates, d_to_idx, train_spec=train_spec, select_spec=spec.universe)
    constructible_regimes = ("pre_reform", "post_reform") if spec.universe.max_tick_cost_bp is None else ("post_reform",)
    assert_screen_constructible(pool.loc[sel_mask], top_k=int(spec.top_k), regimes=constructible_regimes)
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=spec.cost))
    cert_df, hist_df = split_regime_frames(labeled, train_start=eff_train_start)
    oof = cpcv_score_with_history(
        cert_df, hist_df, RANKER_FEATURE_COLS,
        cv=cv, model_params=model_params, huber_delta=huber_delta,
        min_train_rows=min_train_rows, seeds=seeds,
    )
    sel_oof = oof[sel_mask[oof.index.to_numpy()]]
    dedup = dedupe_cpcv_oof(sel_oof, value_cols=("net_pit", "gross_return", "tick_cost_bp"))
    assert_unique_date_symbol(dedup)
    cost_ratio = float(pd.to_numeric(dedup["tick_cost_bp"], errors="coerce").mean()) / 10000.0
    exit_oof = dedup.rename(columns={"date": "trade_date", "symbol": "stock_code", "net_pit": "net_return"})
    return exit_oof[["trade_date", "stock_code", "pred", "net_return"]].copy(), cost_ratio


def run_exit_grid_revalidation(
    *,
    ph: pd.DataFrame | None = None,
    market_dates: np.ndarray | None = None,
    d_to_idx: dict[pd.Timestamp, int] | None = None,
    export_dir: str | None = None,
    spec: StrategySpec = KCA_TOPK_COSTAWARE_001,
    train_spec: UniverseSpec = DEFAULT_UNIVERSE,
    train_start: pd.Timestamp | None = None,
    min_train_rows: int = TRAIN_POOL_MIN_ROWS,
    take_profit_grid: tuple[float, ...] = DEFAULT_TAKE_PROFIT_GRID,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """청산레버를 인증 리랭커와 동일 CPCV(8,2)+실측비용으로 재검증한다 (1회성 연구 실행).

    Args:
        ph: Prepared price-history panel; None loads settings.PRICE_HISTORY_PARQUET_PATH.
        market_dates: Trading calendar; required together with ph/d_to_idx when overriding.
        d_to_idx: Date-to-index lookup; required together with ph/market_dates when overriding.
        export_dir: When given, writes the per-take-profit grid to
            <export_dir>/exit_grid_revalidation_report.parquet.
        spec: Strategy specification carrying top_k, select universe and cost.
        train_spec: Wide training screen without the cost cap.
        train_start: Training-window start; None selects the panel minimum date.
        min_train_rows: Fail-closed floor on per-fold training rows (passed through to build_exit_grid_oof).
        take_profit_grid: Take-profit levels to score.
        alpha: Two-sided significance threshold for promotion.

    Returns:
        summarize_exit_grid(...) output dict, plus cost_ratio/cv_n_groups/cv_k_test.
    """
    if ph is None or market_dates is None or d_to_idx is None:
        from src import settings

        ph, market_dates, d_to_idx = load_and_prepare_price_history(settings.PRICE_HISTORY_PARQUET_PATH)
    cv = CombinatorialPurgedCV(n_groups=CPCV_N_GROUPS, k_test=CPCV_K_TEST, purge_gap=1, embargo_gap=1)
    oof_df, cost_ratio = build_exit_grid_oof(
        ph, market_dates, d_to_idx, spec=spec, train_spec=train_spec, cv=cv,
        train_start=train_start, min_train_rows=min_train_rows,
    )
    results = evaluate_exit_grid(
        oof_df, ph, cost_ratio=cost_ratio, take_profit_grid=take_profit_grid, cv=cv, alpha=alpha,
    )
    summary = summarize_exit_grid(results)
    summary["cost_ratio"] = cost_ratio
    summary["cv_n_groups"] = CPCV_N_GROUPS
    summary["cv_k_test"] = CPCV_K_TEST
    logger.info(
        "[EVAL] stage=exit_grid_revalidation n_days=%d cost_ratio=%.6f incumbent_mean_net=%.6f best_tp=%s",
        summary["n_days"], cost_ratio, summary["incumbent_mean_net"],
        summary["best"]["take_profit_pct"] if summary["best"] else None,
    )
    if export_dir is not None:
        out_path = Path(export_dir) / "exit_grid_revalidation_report.parquet"
        atomic_write_parquet(pd.DataFrame(summary["grid"]), out_path)
        logger.info("[EVAL] stage=exit_grid_revalidation wrote=%s", out_path)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_exit_grid_revalidation(export_dir="artifacts/models")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
