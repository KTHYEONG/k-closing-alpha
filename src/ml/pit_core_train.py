"""PIT-CORE training: persisted 15:20 row store, CORE bundle trainer and the store CLI."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.ml.bundle import build_inline_bundle
from src.ml.pit_core_features import (
    INTRADAY_FEATURES,
    attach_lagged_index,
    core_history_features,
    index_table_from_price_history,
    path_features,
)
from src.ml.research.pit_certification import (
    PitCertificationConfig,
    attach_eod_labels,
    select_usable_panel_days,
)
from src.ml.research.v3_engine import FORWARD_EXIT_MAX_DAYS
from src.ml.retrain_gate import RETRAIN_GATE_EVAL_DAYS
from src.ml.topk_contract import (
    FEATURE_CONTRACT_VERSION_KEY,
    PIT_CORE_CONTRACT_VERSION_KEY,
    PIT_CORE_FEATURE_COLS,
    PIT_CORE_FEATURE_CONTRACT_VERSION,
    RANKER_FEATURE_COLS,
    TOPK_FEATURE_CONTRACT_VERSION,
    build_topk_feature_manifest,
)
from src.ml.topk_ranker_research import (
    CERT_REGIME_START,
    LABEL_MODE,
    TRAIN_POOL_MIN_ROWS,
    attach_pit_net_label,
    build_dual_pool,
    demean_label_by_date,
)
from src.strategy.contract import MIN_TOP_K, PRODUCTION_STRATEGY, StrategySpec, training_universe
from src.strategy.store import StrategyStoreError, require_production_store
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

PIT_STORE_DIRNAME: str = "pit_train"
PIT_STORE_TIERS: tuple[str, ...] = ("exact", "recovered", "reconstructed")
PIT_STORE_COLUMNS: tuple[str, ...] = ("date", "symbol", "tier", *RANKER_FEATURE_COLS, *INTRADAY_FEATURES)
PIT_STORE_MIN_DAY_COVERAGE: float = 0.6
MODEL_RECIPE: str = "pit_core_v1"
HUBER_DELTA: float = 0.9

_EOD_TIER: str = "eod"
_STORE_ROWS_DIRNAME: str = "rows"
_BUILD_DIRNAME: str = "_build"
_BUILD_DONE_DIRNAME: str = "_build_done"


def pit_store_dir() -> Path:
    return Path(settings.HISTORY_DIR) / PIT_STORE_DIRNAME


def _day_key(day: pd.Timestamp) -> str:
    return str(pd.Timestamp(day).strftime("%Y-%m-%d"))


def _day_path(store_dir: Path, day: pd.Timestamp) -> Path:
    return store_dir / _STORE_ROWS_DIRNAME / f"{_day_key(day)}.parquet"


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    return out


def write_store_day(store_dir: Path, day: pd.Timestamp, rows: pd.DataFrame) -> Path:
    """Persist one immutable store day.

    Raises:
        ValueError: Missing columns, rows of another date, duplicate (date, symbol), or a tier outside PIT_STORE_TIERS.
        FileExistsError: The day is already stored.
    """
    missing = [c for c in PIT_STORE_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"store rows missing columns: {missing}")
    work = _normalize_keys(rows[list(PIT_STORE_COLUMNS)])
    target_day = pd.Timestamp(day).normalize()
    if (work["date"] != target_day).any():
        raise ValueError(f"store rows carry dates other than {_day_key(target_day)}")
    if work.duplicated(["date", "symbol"]).any():
        raise ValueError("store rows carry duplicate (date, symbol) keys")
    bad_tiers = sorted(set(work["tier"]) - set(PIT_STORE_TIERS))
    if bad_tiers:
        raise ValueError(f"store rows carry tiers outside PIT_STORE_TIERS: {bad_tiers}")
    path = _day_path(store_dir, target_day)
    if path.exists():
        raise FileExistsError(f"store day already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(work.sort_values("symbol").reset_index(drop=True), path)
    return path


def read_pit_store(store_dir: Path) -> pd.DataFrame:
    """All stored days, sorted by (date, symbol).

    Raises:
        ValueError: The store holds no days.
    """
    files = sorted((store_dir / _STORE_ROWS_DIRNAME).glob("*.parquet"))
    if not files:
        raise ValueError(f"PIT store has no days: {store_dir}")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return _normalize_keys(frame).sort_values(["date", "symbol"]).reset_index(drop=True)


def seed_pit_store(dataset: pd.DataFrame, ext: pd.DataFrame, store_dir: Path) -> int:
    """Write the research dataset's PIT-tier rows, joined with path features, one file per new date.

    Returns:
        Number of days written; dates already in the store are skipped.
    """
    rows = dataset[dataset["tier"].isin(PIT_STORE_TIERS)].drop(columns=list(INTRADAY_FEATURES), errors="ignore")
    rows = _normalize_keys(rows)
    ext_work = _normalize_keys(ext[["date", "symbol", *INTRADAY_FEATURES]])
    joined = rows.merge(ext_work, on=["date", "symbol"], how="left", validate="one_to_one")
    written = 0
    for day, day_rows in joined.groupby("date", sort=True):
        if _day_path(store_dir, day).exists():
            continue
        write_store_day(store_dir, day, day_rows)
        written += 1
    logger.info("[DATA] stage=pit_store seed days_written=%d", written)
    return written


def extend_pit_store(store_dir: Path, *, through: pd.Timestamp) -> list[pd.Timestamp]:  # pragma: no cover - partition I/O
    from src.ml.research.pit_ext_features import _load_day_bars
    from src.ml.research.pit_walkforward import build_dataset, build_research_panel

    last = pd.Timestamp(read_pit_store(store_dir)["date"].max()).normalize()
    start = last + pd.Timedelta(days=1)
    end = pd.Timestamp(through).normalize()
    if start > end:
        return []
    build_dir = store_dir / _BUILD_DIRNAME / uuid.uuid4().hex
    build_research_panel(
        start=_day_key(start),
        end=_day_key(end),
        out_dir=build_dir,
        recovered_dir=None,
        reconstruct_consolidated=True,
    )
    panel = pd.read_parquet(build_dir / "pit1520_panel.parquet")
    panel_days = pd.read_parquet(build_dir / "pit1520_panel_days.parquet")
    usable = select_usable_panel_days(
        panel_days,
        cert_dates=list(pd.to_datetime(panel_days["date"])),
        config=PitCertificationConfig(min_day_coverage=PIT_STORE_MIN_DAY_COVERAGE),
    )
    if not usable:
        # The dataset builder cannot run on an empty panel; an all-unusable range simply adds no days
        # and is retried from the same last stored day next time.
        _retire_build_dir(store_dir, build_dir)
        logger.info("[DATA] stage=pit_store extend days_written=0 reason=no_usable_days through=%s", _day_key(end))
        return []
    dataset = build_dataset(
        panel=panel,
        panel_days=panel_days,
        recovered_keys=frozenset(),
        min_day_coverage=PIT_STORE_MIN_DAY_COVERAGE,
        eod_until=start,
    )
    dataset = _normalize_keys(dataset[dataset["tier"].isin(PIT_STORE_TIERS)])
    research_root = Path(settings.HISTORY_DIR) / "research"
    written: list[pd.Timestamp] = []
    for day, day_rows in dataset.groupby("date", sort=True):
        if day <= last or day > end:
            continue
        bars = _load_day_bars(_day_key(day), research_root)
        if bars.empty:
            continue
        path_cols = path_features(bars).assign(date=day)
        day_rows = day_rows.drop(columns=list(INTRADAY_FEATURES), errors="ignore").merge(
            _normalize_keys(path_cols), on=["date", "symbol"], how="left", validate="one_to_one"
        )
        write_store_day(store_dir, day, day_rows[list(PIT_STORE_COLUMNS)])
        written.append(pd.Timestamp(day))
    _retire_build_dir(store_dir, build_dir)
    logger.info("[DATA] stage=pit_store extend days_written=%d through=%s", len(written), _day_key(end))
    return written


def _retire_build_dir(store_dir: Path, build_dir: Path) -> None:
    done_dir = store_dir / _BUILD_DONE_DIRNAME
    done_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(build_dir), str(done_dir / build_dir.name))


def build_core_training_frame(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    store: pd.DataFrame,
    raw_history: pd.DataFrame,
    spec: StrategySpec = PRODUCTION_STRATEGY,
    label_clip: float,
) -> pd.DataFrame:
    """Assemble the PIT-CORE training frame: EOD rows before the store, then labeled store rows.

    EOD rows keep NaN x_* and h_* by design, matching the validated research view.
    """
    store_norm = _normalize_keys(store)
    store_start = pd.Timestamp(store_norm["date"].min()).normalize()

    pool, _sel_mask = build_dual_pool(
        _pool_source_rows(ph, market_dates, d_to_idx, store_start),
        market_dates,
        d_to_idx,
        train_spec=training_universe(spec.universe),
        select_spec=spec.universe,
    )
    pool = demean_label_by_date(attach_pit_net_label(pool, cost=spec.cost), label_clip=label_clip)
    pool = _normalize_keys(pool)
    eod_cols = ["date", "symbol", *RANKER_FEATURE_COLS, "net_pit", "train_label", "exit_date"]
    eod_rows = pool.loc[pool["date"] < store_start, eod_cols].copy()
    eod_rows["tier"] = _EOD_TIER

    labeled = attach_eod_labels(store_norm[["date", "symbol"]], ph, market_dates, d_to_idx, cost=spec.cost)
    labeled = demean_label_by_date(labeled, label_clip=label_clip)
    labeled = _normalize_keys(labeled)
    store_part = store_norm.merge(
        labeled[["date", "symbol", "net_pit", "train_label", "exit_date"]],
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    history = core_history_features(raw_history)
    store_part = store_part.merge(history, on=["date", "symbol"], how="left", validate="one_to_one")

    frame = pd.concat([eod_rows, store_part], ignore_index=True)
    frame = attach_lagged_index(frame, index_table_from_price_history(ph))
    return frame[["date", "symbol", "tier", *PIT_CORE_FEATURE_COLS, "net_pit", "train_label", "exit_date"]]


def _pool_source_rows(
    ph: pd.DataFrame, market_dates: np.ndarray, d_to_idx: dict[pd.Timestamp, int], store_start: pd.Timestamp
) -> pd.DataFrame:
    # Only pool rows before the store survive, and their features and exits read at most
    # FORWARD_EXIT_MAX_DAYS trading days past the last such date, so later rows are pure peak memory.
    before = int(np.searchsorted(market_dates, np.datetime64(store_start), side="left"))
    if before == 0:
        return ph.iloc[:0]
    cut_idx = min(before - 1 + FORWARD_EXIT_MAX_DAYS, len(market_dates) - 1)
    return ph[ph["date"] <= market_dates[cut_idx]]


def train_core_bundle(
    frame: pd.DataFrame,
    *,
    spec: StrategySpec = PRODUCTION_STRATEGY,
    model_params: Mapping[str, Any],
    seeds: tuple[int, ...],
    min_train_rows: int = TRAIN_POOL_MIN_ROWS,
) -> dict[str, Any]:
    """Fit the CORE bundle on resolved, finitely labeled rows and stamp the production provenance keys.

    Raises:
        ValueError: spec.top_k below MIN_TOP_K, or fewer than min_train_rows fit rows.
        RuntimeError: The trained feature set differs from PIT_CORE_FEATURE_COLS.
    """
    if int(spec.top_k) < MIN_TOP_K:
        raise ValueError(f"top_k {spec.top_k} below the minimum investable K {MIN_TOP_K}")
    store_rows = frame.loc[frame["tier"] != _EOD_TIER, "date"]
    if store_rows.empty:
        raise ValueError("training frame carries no PIT store rows")
    fit = frame[frame["exit_date"].notna() & np.isfinite(frame["train_label"].astype(float))]
    if len(fit) < int(min_train_rows):
        raise ValueError(f"fit rows {len(fit)} below min_train_rows {min_train_rows}")
    eff_params = dict(model_params)
    bundle = build_inline_bundle(
        fit,
        list(PIT_CORE_FEATURE_COLS),
        "train_label",
        "date",
        return_model_params=eff_params,
        huber_delta=HUBER_DELTA,
        seeds=tuple(seeds),
    )
    if list(bundle["feature_cols"]) != list(PIT_CORE_FEATURE_COLS):
        raise RuntimeError("trained feature_cols differ from PIT_CORE_FEATURE_COLS")
    bundle["strategy_id"] = spec.strategy_id
    bundle["top_k"] = int(spec.top_k)
    bundle["train_start"] = str(pd.Timestamp(frame["date"].min()).date())
    bundle["certification_regime_start"] = str(CERT_REGIME_START.date())
    bundle["select_universe"] = _universe_dict(spec)
    bundle["label_mode"] = LABEL_MODE
    bundle["model_params"] = eff_params
    bundle["seeds"] = list(seeds)
    bundle[FEATURE_CONTRACT_VERSION_KEY] = TOPK_FEATURE_CONTRACT_VERSION
    bundle[PIT_CORE_CONTRACT_VERSION_KEY] = PIT_CORE_FEATURE_CONTRACT_VERSION
    bundle["model_recipe"] = MODEL_RECIPE
    bundle["pit_store_last_date"] = str(pd.Timestamp(store_rows.max()).date())
    bundle["feature_manifest"] = build_topk_feature_manifest(bundle["feature_cols"])
    return bundle


def _universe_dict(spec: StrategySpec) -> dict[str, Any]:
    return dataclasses.asdict(spec.universe)


def build_core_gate_eval_frame(frame: pd.DataFrame, *, eval_days: int = RETRAIN_GATE_EVAL_DAYS) -> pd.DataFrame:
    """Non-EOD rows of the most recent eval_days distinct dates."""
    store_rows = frame[frame["tier"] != _EOD_TIER]
    dates = pd.to_datetime(store_rows["date"]).dt.normalize()
    recent = sorted(dates.unique())[-int(eval_days):]
    return store_rows.loc[dates.isin(recent)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    configure_cli_logging()
    try:
        require_production_store()
    except StrategyStoreError as exc:
        logger.error("[SYS] stage=pit_store status=FAIL reason=%s", exc)
        raise SystemExit(1) from exc
    parser = argparse.ArgumentParser(description="PIT-CORE 15:20 row store maintenance")
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed = sub.add_parser("seed", help="seed the store from the validated research dataset")
    seed.add_argument("--dataset", required=True)
    seed.add_argument("--ext", required=True)
    ext = sub.add_parser("extend", help="extend the store from local regular bars")
    ext.add_argument("--through", required=True, help="YYYY-MM-DD")
    args = parser.parse_args(argv)
    store_dir = pit_store_dir()
    if args.cmd == "seed":
        seed_pit_store(pd.read_parquet(args.dataset), pd.read_parquet(args.ext), store_dir)
        return
    written_days = extend_pit_store(store_dir, through=pd.Timestamp(args.through))
    logger.info("[DATA] stage=pit_store extend days_written=%d", len(written_days))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
