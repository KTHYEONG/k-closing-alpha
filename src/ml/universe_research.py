"""Reconstructed full-market universe research harness.

Ranker-in-the-loop screen family search: candidate panels are rebuilt from the
full price_history for a family of ScreenConfigs, the same ranker is trained on
each, and every screen is scored as model-free EV vs ranked top-1 net-of-cost
vs CPCV top-1 path win rate. This is a light research record, not a promotion
gate: nothing here is promotable.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.execution import cost_model
from src.ml.buyability import classify_ceiling_entry
from src.ml.dataset import build_ml_dataset
from src.ml.metrics import mean_group_rank_ic
from src.ml.oof import purged_oof_predict
from src.ml.robust_eval import CombinatorialPurgedCV
from src.ml.universe import ScreenConfig, build_universe_panel, screen_baseline_stats
from src.ml.validation import cpcv_path_evidence
from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO

logger = logging.getLogger(__name__)

_VERDICT_MIN_PATH_WIN: float = 0.60

UNIVERSE_TO_RAW_PANEL_MAP: dict[str, str] = {
    "symbol": "종목코드",
    "trade_date": "매수날짜",
    "open": "(시가)",
    "high": "(고가)",
    "low": "(저가)",
    "close": "(종가)",
    "prev_close": "(전일종가)",
    "market_cap_100m": "(시가총액, 억)",
    "trade_value_100m": "(거래대금, 억)",
    "market": "(시장구분)",
    "volume": "(거래량)",
    "inst_netbuy": "(기관_순매수)",
    "foreign_netbuy": "(외국인_순매수)",
    "program_netbuy": "(프로그램_순매수)",
    "v_kospi": "v_kospi",
    "v_kosdaq": "v_kosdaq",
}

DEFAULT_RESEARCH_SCREENS: dict[str, ScreenConfig] = {
    "operator_legacy": ScreenConfig(0.10, None, 100.0, 500.0),
    "liq_only_wide": ScreenConfig(-1.0, None, 100.0, 500.0),
    "liq_only_tight": ScreenConfig(-1.0, None, 300.0, 1000.0),
    "modest_up": ScreenConfig(0.02, 0.10, 100.0, 500.0),
    "strong_up_capped": ScreenConfig(0.08, 0.20, 100.0, 500.0),
    "index_up_liq": ScreenConfig(-1.0, None, 100.0, 500.0, exclude_ceiling=True, require_index_up=True),
}


@dataclass(frozen=True)
class UniverseScreenRecord:
    screen_name: str
    screen: dict[str, Any]
    n_rows: int
    n_days: int
    per_day: float
    model_free_gross_bp: float
    model_free_net_bp: float
    model_free_t_stat: float
    ranked_rank_ic: float
    ranked_top1_gross_bp: float
    ranked_top1_net_bp: float
    cpcv_top1_path_win_rate: float
    cpcv_ic_path_win_rate: float
    cpcv_n_paths: int
    breakeven_cost_bp: float
    cost_ratio_bp: float
    verdict: str

    def to_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _insufficient_record(
    screen_name: str,
    screen_dict: dict[str, Any],
    cost_bp: float,
    *,
    n_rows: int,
    n_days: int,
    per_day: float,
    mf_gross_bp: float,
    mf_net_bp: float,
    mf_t_stat: float,
) -> UniverseScreenRecord:
    nan = float("nan")
    return UniverseScreenRecord(screen_name, screen_dict, n_rows, n_days, per_day, mf_gross_bp, mf_net_bp, mf_t_stat, nan, nan, nan, nan, nan, 0, nan, cost_bp, "insufficient_data")


def build_universe_training_panel(
    price_history_df: pd.DataFrame,
    screen: ScreenConfig,
    theme_df: pd.DataFrame | None = None,
    *,
    feature_set: str = "close_morning61",
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str], pd.DataFrame, dict[str, Any]]:
    """Rebuild a labelled training panel for one screen from full price_history."""
    if price_history_df is None or len(price_history_df) == 0:
        raise ValueError("price_history_df is None or empty: price_history input required")
    panel, prov = build_universe_panel(price_history_df, screen, start_date=start_date, end_date=end_date)
    if len(panel) == 0:
        universe_provenance: dict[str, Any] = {**prov, "screen_name": None, "panel_rows_pre_label": 0}
        empty_processed = pd.DataFrame(columns=["trade_date", "stock_code", "close_price", "prev_close_price", "high_price", "mechanical_gross"])
        return (pd.DataFrame(), {}, [], empty_processed, universe_provenance)
    raw = pd.DataFrame({bracket: panel[universe_col].to_numpy() for universe_col, bracket in UNIVERSE_TO_RAW_PANEL_MAP.items()})
    daily_change = pd.to_numeric(panel["daily_change_pct"], errors="coerce").to_numpy(dtype=np.float64)
    kospi_pct = pd.to_numeric(panel["kospi_pct"], errors="coerce").to_numpy(dtype=np.float64)
    kosdaq_pct = pd.to_numeric(panel["kosdaq_pct"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(panel["close"], errors="coerce").to_numpy(dtype=np.float64)
    gross = pd.to_numeric(panel["mechanical_gross"], errors="coerce").to_numpy(dtype=np.float64)
    raw["(등락률)"] = daily_change * 100.0
    raw["(kospi, %)"] = kospi_pct * 100.0
    raw["(kosdaq, %)"] = kosdaq_pct * 100.0
    raw["(매수 가격)"] = close
    raw["(매도 가격)"] = close * (1.0 + gross)
    raw["(수익률, %)"] = gross * 100.0
    raw["(체결강도)"] = np.nan
    raw["(차트분석)"] = "미분류"
    day_frame = pd.DataFrame({"_day": pd.to_datetime(panel["trade_date"]).to_numpy(), "_tv": pd.to_numeric(panel["trade_value_100m"], errors="coerce").to_numpy(dtype=np.float64)})
    raw["(선정 순위)"] = day_frame.groupby("_day")["_tv"].rank(method="min", ascending=False).to_numpy()
    raw["(총 종목 수)"] = day_frame.groupby("_day")["_tv"].transform("size").to_numpy()
    raw["(평균 거래대금)"] = day_frame.groupby("_day")["_tv"].transform("mean").to_numpy()
    theme_map = theme_df.set_index("종목코드")["테마"] if theme_df is not None and not theme_df.empty and {"종목코드", "테마"}.issubset(theme_df.columns) else None
    codes = panel["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    raw["(테마/섹터)"] = codes.map(theme_map).fillna("기타").to_numpy() if theme_map is not None else "기타"
    x_features, targets, cat_features, processed = build_ml_dataset(raw, theme_df, feature_set=feature_set, panel_mode="scenario_action", price_history_df=price_history_df, scenario_source="auto")
    feature_cols = [c for c in x_features.columns if c not in cat_features]
    proc = processed.copy()
    proc["_join_day"] = pd.to_datetime(proc["trade_date"])
    proc["_join_code"] = proc["stock_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    panel_keys = pd.DataFrame({"_join_day": pd.to_datetime(panel["trade_date"]), "_join_code": panel["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6), "_join_gross": gross}).drop_duplicates(subset=["_join_day", "_join_code"], keep="first")
    proc = proc.merge(panel_keys, on=["_join_day", "_join_code"], how="left")
    proc["mechanical_gross"] = pd.to_numeric(proc["_join_gross"], errors="coerce").to_numpy(dtype=np.float64)
    processed = proc.drop(columns=["_join_day", "_join_code", "_join_gross"])
    return (x_features, targets, feature_cols, processed, {**prov, "screen_name": None, "panel_rows_pre_label": len(raw)})


def _daily_top1_net_bp(oof: pd.DataFrame, group_col: str, gross_col: str, cost_ratio: float, score_col: str = "pred") -> tuple[float, float]:
    """Per-group argmax gross edge in bp: (gross_bp, net_bp). Empty/all-NaN -> (nan, nan)."""
    idx = oof.groupby(group_col, sort=True)[score_col].idxmax()
    gross = pd.to_numeric(oof.loc[idx, gross_col], errors="coerce").to_numpy(dtype=np.float64)
    finite = gross[np.isfinite(gross)]
    mean_gross = float(np.mean(finite)) if finite.size else float("nan")
    return (mean_gross * 1e4, (mean_gross - float(cost_ratio)) * 1e4)


def evaluate_universe_screen(
    price_history_df: pd.DataFrame,
    screen: ScreenConfig,
    *,
    screen_name: str,
    feature_set: str = "close_morning61",
    start_date: str,
    end_date: str,
    oos_reserve_start: str | None = None,
    n_splits: int = 5,
    purge_gap: int = 1,
    cpcv_n_groups: int = 8,
    cpcv_k_test: int = 2,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    cost_ratio: float = ROUND_TRIP_COST_RATIO,
    theme_df: pd.DataFrame | None = None,
) -> UniverseScreenRecord:
    """Train the ranker on one screen panel; score model-free vs ranked vs CPCV."""
    _, _, feature_cols, processed, _prov = build_universe_training_panel(price_history_df, screen, theme_df, feature_set=feature_set, start_date=start_date, end_date=end_date)
    screen_dict = dataclasses.asdict(screen)
    cost = float(cost_ratio)
    cost_bp = cost * 1e4
    processed = processed[~classify_ceiling_entry(processed).to_numpy(bool)]
    processed = processed[pd.to_datetime(processed["trade_date"]) < pd.Timestamp(oos_reserve_start)] if oos_reserve_start is not None else processed
    stats = screen_baseline_stats(processed, group_col="trade_date", gross_col="mechanical_gross", cost_ratio=cost) if len(processed) else None
    n_rows = len(processed)
    n_days = int(processed["trade_date"].nunique()) if n_rows else 0
    per_day = float(n_rows / n_days) if n_days else 0.0
    mf_gross_bp = float(stats["gross_bp"]) if stats is not None else float("nan")
    mf_net_bp = float(stats["net_bp"]) if stats is not None else float("nan")
    mf_t_stat = float(stats["t_stat"]) if stats is not None else float("nan")
    if n_rows == 0 or n_days < int(cpcv_n_groups):
        return _insufficient_record(screen_name, screen_dict, cost_bp, n_rows=n_rows, n_days=n_days, per_day=per_day, mf_gross_bp=mf_gross_bp, mf_net_bp=mf_net_bp, mf_t_stat=mf_t_stat)
    oof = purged_oof_predict(processed, feature_cols, "target_return", "trade_date", n_splits=n_splits, purge_gap=purge_gap, model_params=model_params, huber_delta=float(huber_delta), predict_proba=False)
    oof_merged = oof.copy()
    oof_merged["mechanical_gross"] = processed.loc[oof.index, "mechanical_gross"].to_numpy(dtype=np.float64)
    ranked_rank_ic = float(mean_group_rank_ic(oof_merged, ["trade_date"], "pred", "mechanical_gross", min_group_size=2))
    ranked_top1_gross_bp, ranked_top1_net_bp = _daily_top1_net_bp(oof_merged, "trade_date", "mechanical_gross", cost)
    # embargo_gap=0 is the feasibility floor for small research grids: the default
    # embargo guard needs k*(1+purge+embargo)+1 groups, which a 5-group grid fails.
    cv = CombinatorialPurgedCV(n_groups=int(cpcv_n_groups), k_test=int(cpcv_k_test), purge_gap=int(purge_gap), embargo_gap=0)
    ev = cpcv_path_evidence(processed, feature_cols, "target_return", "mechanical_gross", "trade_date", cv=cv, candidate_params=model_params, control_params=None, huber_delta=float(huber_delta), control_huber_delta=0.9)
    breakeven_bp = float(cost_model.breakeven_cost_bp(oof_merged["mechanical_gross"].to_numpy(dtype=np.float64), oof_merged["trade_date"].to_numpy()))
    top1_win_rate = float(ev["top1_path_win_rate"])
    ic_win_rate = float(ev["ic_path_win_rate"])
    verdict = "clears_cost" if ranked_top1_net_bp > 0.0 and top1_win_rate >= _VERDICT_MIN_PATH_WIN else "below_cost"
    return UniverseScreenRecord(screen_name, screen_dict, n_rows, n_days, per_day, mf_gross_bp, mf_net_bp, mf_t_stat, ranked_rank_ic, ranked_top1_gross_bp, ranked_top1_net_bp, top1_win_rate, ic_win_rate, int(ev["n_path_deltas"]), breakeven_bp, cost_bp, verdict)


def run_universe_screen_grid(
    price_history_df: pd.DataFrame,
    screens: dict[str, ScreenConfig],
    *,
    feature_set: str = "close_morning61",
    start_date: str,
    end_date: str,
    oos_reserve_start: str | None = None,
    n_splits: int = 5,
    purge_gap: int = 1,
    cpcv_n_groups: int = 8,
    cpcv_k_test: int = 2,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    cost_ratio: float = ROUND_TRIP_COST_RATIO,
    theme_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate every screen sequentially; one row per screen, best net first."""
    if not screens:
        raise ValueError("screens must be non-empty")
    records: list[UniverseScreenRecord] = []
    for name, screen in screens.items():
        rec = evaluate_universe_screen(price_history_df, screen, screen_name=name, feature_set=feature_set, start_date=start_date, end_date=end_date, oos_reserve_start=oos_reserve_start, n_splits=n_splits, purge_gap=purge_gap, cpcv_n_groups=cpcv_n_groups, cpcv_k_test=cpcv_k_test, model_params=model_params, huber_delta=huber_delta, cost_ratio=cost_ratio, theme_df=theme_df)
        records.append(rec)
        logger.info("[EVAL] stage=universe_grid screen=%s per_day=%.2f model_free_net_bp=%.2f ranked_top1_net_bp=%.2f cpcv_top1_path_win_rate=%.2f verdict=%s", rec.screen_name, rec.per_day, rec.model_free_net_bp, rec.ranked_top1_net_bp, rec.cpcv_top1_path_win_rate, rec.verdict)
    df = pd.DataFrame([rec.to_row() for rec in records])
    df = df.sort_values("ranked_top1_net_bp", ascending=False, na_position="last").reset_index(drop=True)
    df.attrs.update({"control_screen": "operator_legacy"} if "operator_legacy" in screens else {})
    return df


def main(argv: list[str] | None = None) -> None:
    """Run the screen family grid over full price_history and save the table."""
    parser = argparse.ArgumentParser(description="Universe screen family research harness (ranker-in-the-loop)")
    parser.add_argument("--price-history", default=str(settings.PRICE_HISTORY_PARQUET_PATH))
    parser.add_argument("--theme", default=str(settings.THEME_PARQUET_PATH))
    parser.add_argument("--feature-set", choices=["close_morning61", "close_morning_history", "close_morning_sector"], default="close_morning61")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--oos-reserve-start", default=None)
    parser.add_argument("--hpo-off", action="store_true", help="no Optuna in this harness; LGBM defaults are always used")
    parser.add_argument("--cpcv-n-groups", type=int, default=8)
    parser.add_argument("--cpcv-k-test", type=int, default=2)
    parser.add_argument("--out", default="artifacts/research/universe_grid.parquet")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(args.price_history):
        raise ValueError(f"price_history not found: {args.price_history}")
    price_history_df = pd.read_parquet(args.price_history)
    theme_df = pd.read_parquet(args.theme) if os.path.exists(args.theme) else None
    date_col = "date" if "date" in price_history_df.columns else "trade_date"
    start_date = args.start_date or str(pd.to_datetime(price_history_df[date_col]).min().date())
    end_date = args.end_date or str(pd.to_datetime(price_history_df[date_col]).max().date())
    df = run_universe_screen_grid(price_history_df, DEFAULT_RESEARCH_SCREENS, feature_set=args.feature_set, start_date=start_date, end_date=end_date, oos_reserve_start=args.oos_reserve_start, model_params=None, cpcv_n_groups=int(args.cpcv_n_groups), cpcv_k_test=int(args.cpcv_k_test), theme_df=theme_df)
    atomic_write_parquet(df, Path(args.out))
    print(df.to_string())  # noqa: T201
