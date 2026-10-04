"""Live-parity validation of the decision-time panel against captured 15:20 decision inputs.

On dates where a live decision input exists, the bar-derived panel (built with BARS_ONLY) is pushed
through the exact serving feature path and compared feature-by-feature with the live input's features,
alongside the EOD training-path features as the known-skew reference. The panel is fit for certification
only if it reproduces live features at least as well as the EOD path and the fetch superset recalls the
live rank pool.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.eod_superset import EodSupersetScreen, eod_superset_mask
from src.data.pit1520_panel import (
    PIT1520_PANEL_COLUMNS,
    PanelSourcePolicy,
    Pit1520PanelConfig,
    build_pit1520_panel,
    load_live_decision_input,
    load_regular_bars,
    panel_to_decision_input,
)
from src.ml.topk_contract import RANKER_FEATURE_COLS
from src.ml.topk_history_features import HISTORY_LOOKBACK_CALENDAR_DAYS, HISTORY_REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

COMPARISON_PANEL_VS_LIVE: str = "panel_vs_live"
COMPARISON_EOD_VS_LIVE: str = "eod_vs_live"

_SNAPSHOT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "종목코드",
    "종가",
    "전일종가",
    "고가",
    "저가",
    "시가",
    "거래량",
    "거래대금",
    "시가총액",
    "시장구분",
    "기관_순매수",
    "외국인_순매수",
    "kospi",
    "kosdaq",
    "v_kospi",
)

_EOD_SUPERSET_COLUMNS: tuple[str, ...] = ("symbol", "chg_ratio", "tv_clean", "mc_clean", "volume")

_FRAME_COLUMNS: tuple[str, ...] = (
    "row_type",
    "date",
    "feature",
    "comparison",
    "spearman",
    "median_abs_diff",
    "n_symbols",
    "superset_recall",
    "rank_pool_coverage",
)


@dataclass(frozen=True)
class PanelValidationConfig:
    """Acceptance thresholds for using the panel in certification.

    Attributes:
        min_overlap_days: Minimum dates with both a live input and a bar-derived panel day.
        min_feature_spearman: Minimum across-day median of the per-day cross-sectional Spearman between
            panel and live features, for every non-constant feature.
        require_not_worse_than_eod: Panel-vs-live median Spearman must be >= EOD-vs-live per feature.
        not_worse_tolerance: Absolute slack for that comparison; equal rankings computed through different
            float paths can differ in the last ulp (0.99999999999999989 vs 1.0) and must not fail.
        min_superset_recall: Minimum pooled fraction of live rank-pool symbols whose EOD row passes the
            fetch superset.
        min_rank_pool_coverage: Minimum pooled fraction of live rank-pool symbols present in the bar panel.
        date_constant_features: Features constant within a date; judged by median absolute difference and
            reported, not gated (index EOD fallback is a declared approximation).
    """

    min_overlap_days: int = 5
    min_feature_spearman: float = 0.98
    require_not_worse_than_eod: bool = True
    not_worse_tolerance: float = 1e-9
    min_superset_recall: float = 0.99
    min_rank_pool_coverage: float = 0.98
    date_constant_features: tuple[str, ...] = ("kospi_pct", "kosdaq_pct", "v_kospi")


@dataclass(frozen=True)
class PanelValidationReport:
    """Validation outcome and its evidence frame."""

    verdict: str  # "PASS" | "FAIL" | "INSUFFICIENT_OVERLAP"
    reasons: tuple[str, ...]
    n_overlap_days: int
    frame: pd.DataFrame


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) != len(a):
        a = a[ok]
        b = b[ok]
    if len(a) < 4:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat = spearmanr(a, b).statistic
    rho = float(stat)
    return rho if np.isfinite(rho) else float("nan")


def _median_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if not bool(ok.any()):
        return float("nan")
    return float(np.median(np.abs(a[ok] - b[ok])))


def _nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if not np.isfinite(arr).any():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.nanmedian(arr))


def _history_before(price_history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    missing = [c for c in HISTORY_REQUIRED_COLUMNS if c not in price_history.columns]
    if missing:
        raise ValueError(f"compute_feature_parity price_history missing required columns: {missing}")
    start = day - pd.Timedelta(days=int(HISTORY_LOOKBACK_CALENDAR_DAYS))
    dts = pd.to_datetime(price_history["date"], errors="coerce")
    keep = (dts >= start) & (dts < day)
    return price_history.loc[keep.to_numpy(), list(HISTORY_REQUIRED_COLUMNS)].copy()


def _check_snapshot(frame: pd.DataFrame, *, label: str) -> None:
    missing = [c for c in _SNAPSHOT_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"compute_feature_parity {label} missing required columns: {missing}")


def _check_features(frame: pd.DataFrame, feature_cols: Sequence[str], *, label: str) -> None:
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"compute_feature_parity {label} missing feature columns: {missing}")


def compute_feature_parity(
    *,
    live_inputs: Mapping[str, pd.DataFrame],
    panel: pd.DataFrame,
    eod_pool: pd.DataFrame,
    price_history: pd.DataFrame,
    screen: EodSupersetScreen,
    feature_cols: Sequence[str] = RANKER_FEATURE_COLS,
    config: PanelValidationConfig = PanelValidationConfig(),  # noqa: B008
) -> PanelValidationReport:
    """Compare panel, live and EOD-training features on overlapping dates.

    Args:
        live_inputs: Verified live decision inputs keyed by date (Korean columns).
        panel: BARS_ONLY panel rows covering at least those dates.
        eod_pool: build_dual_pool output (training-path features) covering those dates.
        price_history: Raw price_history rows with HISTORY_REQUIRED_COLUMNS covering
            [first date - HISTORY_LOOKBACK_CALENDAR_DAYS, last date) for history features, plus EOD rows of
            the dates for superset recall.
        screen: Fetch superset whose recall is measured.
        feature_cols: Features compared (default the certified ranker features).
        config: Acceptance thresholds.

    Returns:
        Report whose frame has row_type in {"feature_day", "feature_summary", "day_coverage"} with columns
        row_type, date, feature, comparison ("panel_vs_live" | "eod_vs_live"), spearman, median_abs_diff,
        n_symbols, superset_recall, rank_pool_coverage.

    Raises:
        ValueError: When a live input or panel day misses required columns.
    """
    from src.daily.universe_screen import rank_pool_mask
    from src.serving.realtime.features import build_topk_ranker_features
    from src.strategy.contract import DEFAULT_UNIVERSE

    features = list(feature_cols)
    constant = set(config.date_constant_features)
    missing_panel = [c for c in PIT1520_PANEL_COLUMNS if c not in panel.columns]
    if missing_panel:
        raise ValueError(f"compute_feature_parity panel missing required columns: {missing_panel}")
    missing_pool = [c for c in ("date", "symbol", *_EOD_SUPERSET_COLUMNS) if c not in eod_pool.columns]
    if missing_pool:
        raise ValueError(f"compute_feature_parity eod_pool missing required columns: {missing_pool}")
    _check_features(eod_pool, features, label="eod_pool")
    panel_dates = pd.to_datetime(panel["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    panel_by_date: dict[str, pd.DataFrame] = {}
    for day_label in sorted(set(panel_dates.tolist())):
        day_frame = panel[panel_dates == day_label]
        if len(day_frame):
            panel_by_date[day_label] = day_frame
    overlap = sorted(d for d in live_inputs if d in panel_by_date)
    eod_dates = pd.to_datetime(eod_pool["date"], errors="coerce").dt.normalize()

    day_rows: list[dict[str, Any]] = []
    recall_num = 0
    recall_den = 0
    coverage_num = 0
    coverage_den = 0
    for day_label in overlap:
        day = pd.Timestamp(day_label).normalize()
        live_frame = live_inputs[day_label]
        _check_snapshot(live_frame, label=f"live input {day_label}")
        panel_snapshot = panel_to_decision_input(panel_by_date[day_label])
        _check_snapshot(panel_snapshot, label=f"panel day {day_label}")
        history = _history_before(price_history, day)
        live_pool = live_frame.loc[
            np.asarray(rank_pool_mask(live_frame, decision_date=day, screen=DEFAULT_UNIVERSE), dtype=bool)
        ].reset_index(drop=True)
        panel_pool = panel_snapshot.loc[
            np.asarray(rank_pool_mask(panel_snapshot, decision_date=day, screen=DEFAULT_UNIVERSE), dtype=bool)
        ].reset_index(drop=True)
        live_feat = build_topk_ranker_features(live_pool, day, price_history=history)
        panel_feat = build_topk_ranker_features(panel_pool, day, price_history=history)
        _check_features(live_feat, features, label=f"live features {day_label}")
        _check_features(panel_feat, features, label=f"panel features {day_label}")
        eod_day = eod_pool[eod_dates == day].copy()
        eod_day["symbol"] = eod_day["symbol"].astype(str)
        live_symbols = live_feat["symbol"].astype(str).tolist()
        panel_symbol_set = set(panel_feat["symbol"].astype(str).tolist())
        eod_symbol_set = set(eod_day["symbol"].astype(str).tolist())
        inter_panel = [s for s in live_symbols if s in panel_symbol_set]
        inter_eod = [s for s in live_symbols if s in eod_symbol_set]
        live_matrix = {
            str(s): row for s, row in zip(live_feat["symbol"].astype(str).tolist(), live_feat[features].to_numpy(dtype=np.float64), strict=True)
        }
        panel_matrix = {
            str(s): row for s, row in zip(panel_feat["symbol"].astype(str).tolist(), panel_feat[features].to_numpy(dtype=np.float64), strict=True)
        }
        eod_matrix = {
            str(s): row for s, row in zip(eod_day["symbol"].astype(str).tolist(), eod_day[features].to_numpy(dtype=np.float64), strict=True)
        }
        positions = {name: pos for pos, name in enumerate(features)}
        for feature in features:
            pos = positions[feature]
            for comparison, peers, inter in (
                (COMPARISON_PANEL_VS_LIVE, panel_matrix, inter_panel),
                (COMPARISON_EOD_VS_LIVE, eod_matrix, inter_eod),
            ):
                paired = [
                    (float(live_matrix[s][pos]), float(peers[s][pos]))
                    for s in inter
                    if s in live_matrix and s in peers
                ]
                a = np.array([p[0] for p in paired], dtype=np.float64)
                b = np.array([p[1] for p in paired], dtype=np.float64)
                rho = float("nan") if feature in constant else _spearman(a, b)
                day_rows.append({
                    "row_type": "feature_day",
                    "date": day,
                    "feature": feature,
                    "comparison": comparison,
                    "spearman": rho,
                    "median_abs_diff": _median_abs_diff(a, b),
                    "n_symbols": len(inter),
                    "superset_recall": float("nan"),
                    "rank_pool_coverage": float("nan"),
                })
        live_pool_symbols = {str(s) for s in live_pool["종목코드"].astype(str).str.zfill(6).tolist()} if len(live_pool) else set()
        superset = _superset_on(price_history, day, screen)
        panel_day_symbols = {str(s) for s in panel_by_date[day_label]["symbol"].astype(str).tolist()}
        recall_num += len(live_pool_symbols & superset)
        recall_den += len(live_pool_symbols)
        coverage_num += len(live_pool_symbols & panel_day_symbols)
        coverage_den += len(live_pool_symbols)
        day_rows.append({
            "row_type": "day_coverage",
            "date": day,
            "feature": "",
            "comparison": "",
            "spearman": float("nan"),
            "median_abs_diff": float("nan"),
            "n_symbols": len(live_pool_symbols),
            "superset_recall": float(len(live_pool_symbols & superset) / len(live_pool_symbols)) if live_pool_symbols else float("nan"),
            "rank_pool_coverage": float(len(live_pool_symbols & panel_day_symbols) / len(live_pool_symbols)) if live_pool_symbols else float("nan"),
        })
    summary_medians: dict[tuple[str, str], float] = {}
    for feature in features:
        for comparison in (COMPARISON_PANEL_VS_LIVE, COMPARISON_EOD_VS_LIVE):
            cells = [r for r in day_rows if r["row_type"] == "feature_day" and r["feature"] == feature and r["comparison"] == comparison]
            med_rho = _nanmedian([float(r["spearman"]) for r in cells])
            med_mad = _nanmedian([float(r["median_abs_diff"]) for r in cells])
            summary_medians[(feature, comparison)] = med_rho
            day_rows.append({
                "row_type": "feature_summary",
                "date": pd.NaT,
                "feature": feature,
                "comparison": comparison,
                "spearman": med_rho,
                "median_abs_diff": med_mad,
                "n_symbols": int(sum(int(r["n_symbols"]) for r in cells)),
                "superset_recall": float("nan"),
                "rank_pool_coverage": float("nan"),
            })
    pooled_recall = float(recall_num / recall_den) if recall_den else float("nan")
    pooled_coverage = float(coverage_num / coverage_den) if coverage_den else float("nan")
    frame = pd.DataFrame(day_rows, columns=list(_FRAME_COLUMNS)) if day_rows else pd.DataFrame({c: [] for c in _FRAME_COLUMNS})
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["spearman"] = pd.to_numeric(frame["spearman"], errors="coerce").astype(np.float64)
        frame["median_abs_diff"] = pd.to_numeric(frame["median_abs_diff"], errors="coerce").astype(np.float64)
        frame["n_symbols"] = pd.to_numeric(frame["n_symbols"], errors="coerce").fillna(0).astype(np.int64)
        frame["superset_recall"] = pd.to_numeric(frame["superset_recall"], errors="coerce").astype(np.float64)
        frame["rank_pool_coverage"] = pd.to_numeric(frame["rank_pool_coverage"], errors="coerce").astype(np.float64)

    reasons: list[str] = []
    if len(overlap) < int(config.min_overlap_days):
        reasons.append(
            f"overlap_days {len(overlap)} below min_overlap_days {int(config.min_overlap_days)}"
        )
        return PanelValidationReport(
            verdict="INSUFFICIENT_OVERLAP", reasons=tuple(reasons), n_overlap_days=len(overlap), frame=frame
        )
    for feature in features:
        if feature in constant:
            continue
        med_panel = summary_medians[(feature, COMPARISON_PANEL_VS_LIVE)]
        med_eod = summary_medians[(feature, COMPARISON_EOD_VS_LIVE)]
        if not np.isfinite(med_panel):
            reasons.append(f"feature {feature} panel_vs_live has no measurable overlap (median spearman nan)")
            continue
        if med_panel < float(config.min_feature_spearman):
            reasons.append(
                f"feature {feature} panel_vs_live median spearman {med_panel:.4f} "
                f"below min_feature_spearman {float(config.min_feature_spearman):.4f}"
            )
        if (
            bool(config.require_not_worse_than_eod)
            and np.isfinite(med_eod)
            and med_panel < med_eod - float(config.not_worse_tolerance)
        ):
            reasons.append(
                f"feature {feature} not_worse_than_eod violated: "
                f"panel_vs_live {med_panel:.4f} < eod_vs_live {med_eod:.4f}"
            )
    if not np.isfinite(pooled_recall) or pooled_recall < float(config.min_superset_recall):
        reasons.append(
            f"pooled superset_recall {pooled_recall:.4f} below min_superset_recall {float(config.min_superset_recall):.4f}"
        )
    if not np.isfinite(pooled_coverage) or pooled_coverage < float(config.min_rank_pool_coverage):
        reasons.append(
            f"pooled rank_pool_coverage {pooled_coverage:.4f} below min_rank_pool_coverage {float(config.min_rank_pool_coverage):.4f}"
        )
    verdict = "PASS" if not reasons else "FAIL"
    med_panel_all = [v for (f, c), v in summary_medians.items() if c == COMPARISON_PANEL_VS_LIVE and f not in constant]
    worst_feature = ""
    worst_value = float("nan")
    if med_panel_all:
        ordered = sorted(
            ((f, summary_medians[(f, COMPARISON_PANEL_VS_LIVE)]) for f in features if f not in constant),
            key=lambda kv: (not np.isfinite(kv[1]), kv[1] if np.isfinite(kv[1]) else 0.0),
        )
        worst_feature, worst_value = ordered[0][0], float(ordered[0][1])
    logger.info(
        "[DATA] stage=pit1520_validation verdict=%s overlap_days=%d min_spearman_feature=%s min_spearman=%.4f superset_recall=%.4f rank_pool_coverage=%.4f",
        verdict,
        len(overlap),
        worst_feature,
        worst_value,
        pooled_recall,
        pooled_coverage,
    )
    for reason in reasons:
        logger.info("[DATA] stage=pit1520_validation reason=%s", reason)
    return PanelValidationReport(verdict=verdict, reasons=tuple(reasons), n_overlap_days=len(overlap), frame=frame)


def _superset_on(price_history: pd.DataFrame, day: pd.Timestamp, screen: EodSupersetScreen) -> set[str]:
    # Recall is measured against the fetch superset over the whole market on T, not the training pool:
    # a live rank-pool member outside the 2-10% training screen is still inside the superset by design.
    missing = [c for c in ("date", *_EOD_SUPERSET_COLUMNS) if c not in price_history.columns]
    if missing:
        raise ValueError(f"compute_feature_parity price_history missing superset columns: {missing}")
    dates = pd.to_datetime(price_history["date"], errors="coerce").dt.normalize()
    day_rows = price_history[dates == day]
    if day_rows.empty:
        raise ValueError(f"compute_feature_parity price_history has no EOD rows on overlap day {day.date()}")
    mask = np.asarray(eod_superset_mask(day_rows, screen), dtype=bool)
    return set(day_rows.loc[mask, "symbol"].astype(str).str.zfill(6).tolist())


def _default_live_dates() -> list[str]:
    from src.data.capture_store import resolve_capture_root

    root = resolve_capture_root() / "decision"
    if not root.exists():
        return []
    # Layout: decision/<YYYY-MM-DD>/<run_id>/input.parquet; the date is the grandparent directory.
    return sorted({p.parent.parent.name for p in root.glob("*/*/input.parquet") if p.is_file()})


def _load_validation_inputs(
    start: str | None, end: str | None
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, EodSupersetScreen]:
    from src import settings
    from src.config.collection import CollectionSettings
    from src.data.capture_store import CaptureStore, resolve_capture_root
    from src.data.panel_integrity import prepare_price_panel
    from src.ml.research.v3_engine import load_and_prepare_price_history
    from src.ml.topk_ranker_research import build_dual_pool
    from src.strategy.contract import PRODUCTION_STRATEGY, training_universe

    live_dates = [
        d for d in _default_live_dates() if (start is None or d >= start) and (end is None or d <= end)
    ]
    raw = pd.read_parquet(settings.PRICE_HISTORY_PARQUET_PATH)
    prepared, _prov = prepare_price_panel(raw)
    config = Pit1520PanelConfig()
    store = CaptureStore(resolve_capture_root())
    live_inputs: dict[str, pd.DataFrame] = {}
    for day_label in live_dates:
        loaded = load_live_decision_input(store, day_label, config=config)
        if loaded is not None:
            live_inputs[day_label] = loaded[0]
    screen = EodSupersetScreen.from_profile(CollectionSettings())

    def _load_bars(day: str) -> pd.DataFrame:
        return load_regular_bars(day, config=config)

    def _no_live(day: str) -> tuple[pd.DataFrame, str] | None:
        return None

    if live_dates:
        built = build_pit1520_panel(
            price_history=prepared,
            dates=live_dates,
            bars_loader=_load_bars,
            live_loader=_no_live,
            screen=screen,
            config=config,
            source_policy=PanelSourcePolicy.BARS_ONLY,
        )
        panel = built.panel
    else:
        panel = pd.DataFrame({c: pd.Series(dtype="object") for c in PIT1520_PANEL_COLUMNS})
    ph, market_dates, d_to_idx = load_and_prepare_price_history(settings.PRICE_HISTORY_PARQUET_PATH)
    eod_pool, _sel_mask = build_dual_pool(
        ph,
        market_dates,
        d_to_idx,
        train_spec=training_universe(PRODUCTION_STRATEGY.universe),
        select_spec=PRODUCTION_STRATEGY.universe,
    )
    return live_inputs, panel, eod_pool, prepared, screen


def main(argv: list[str] | None = None) -> None:
    """CLI: validate the bar-derived panel against live decision inputs and persist the evidence.

    Flags:
        --start YYYY-MM-DD, --end YYYY-MM-DD (default: every date with a live decision input)
        --out PATH (default artifacts/research/pit1520_validation.parquet)

    Exit:
        Writes the report, then raises SystemExit(1) unless the verdict is PASS.
    """
    from src.data.io_utils import atomic_write_parquet

    parser = argparse.ArgumentParser(description="Validate the decision-time panel against live 15:20 inputs")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="artifacts/research/pit1520_validation.parquet")
    args = parser.parse_args(argv)
    live_inputs, panel, eod_pool, price_history, screen = _load_validation_inputs(args.start, args.end)
    report = compute_feature_parity(
        live_inputs=live_inputs, panel=panel, eod_pool=eod_pool, price_history=price_history, screen=screen
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(report.frame, out_path)
    if report.verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    from src.utils.cli_logging import configure_cli_logging

    configure_cli_logging()
    main()
