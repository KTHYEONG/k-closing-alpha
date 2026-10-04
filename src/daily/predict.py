import functools
import logging
import subprocess
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src import settings
from src.config.base import RANK_POOL_PARQUET_NAME, TOPK_DECISIONS_PARQUET_NAME
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.io_utils import (
    ExistingStoreUnreadableError,
    atomic_write_parquet,
    read_existing_parquet,
    store_write_lock,
)
from src.data.session_calendar import SessionDay, SessionKind, resolve_session_day, trading_session_gate
from src.data.trading_calendar import DAY_HOLIDAY, DAY_WEEKEND, classify_day
from src.tools.run_outcome import RUN_OUTCOME_NO_DECISION, RUN_OUTCOME_OK, record_run_outcome
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

from src.ml.retrain_registry import resolve_code_commit_env
from src.serving.realtime.artifacts import load_model_bundle
from src.utils.display import print_table


def load_daily_snapshot(decision_date: pd.Timestamp, *, available_by: datetime | None = None) -> pd.DataFrame:
    """Read the exact observable decision state instead of a finalized archive view.

    Args:
        decision_date: Requested market date.
        available_by: Explicit aware inference cutoff; None uses the actual call time.

    Returns:
        Verified wide decision input with normalized string security codes and provenance.

    Raises:
        FileNotFoundError: No qualifying new-mode input exists.
        ValueError: Observations, membership, or hashes cannot be certified.
    """
    from src.data.capture_store import CaptureStore

    cutoff = available_by
    if cutoff is None:
        cutoff = datetime.now(ZoneInfo("Asia/Seoul"))
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("available_by must be an aware cutoff")
    frame = CaptureStore(_capture_root()).read_decision(
        decision_date.strftime("%Y-%m-%d"), available_by=cutoff
    )
    if "종목코드" in frame.columns:
        frame["종목코드"] = frame["종목코드"].astype(str).str.zfill(6)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    return frame


def restrict_to_rank_pool(wide: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    """Restrict the snapshot to the training rank pool.

    Args:
        wide: Daily snapshot frame with Korean columns and ``admitted`` flags.
        decision_date: Decision date for point-in-time screen inputs.

    Returns:
        Rank-pool rows with a reset index.

    Raises:
        ValueError: When the snapshot is empty, an admitted row lies outside
            the training rank pool, or the pool itself is empty.
    """
    from src.daily.universe_screen import rank_pool_mask

    if wide.empty:
        raise ValueError("live_rows is empty; nothing to decide on")
    mask = rank_pool_mask(wide, decision_date=pd.Timestamp(decision_date))
    admitted = wide["admitted"].fillna(False).astype(bool).to_numpy()
    outside = wide.loc[admitted & ~mask, "종목코드"].astype(str).tolist()
    if outside:
        raise ValueError(f"admitted rows outside the training rank pool: {outside}")
    pool = wide.loc[mask].reset_index(drop=True)
    logger.info(
        "[DATA] stage=rank_pool n_snapshot=%d n_pool=%d n_admitted=%d",
        len(wide),
        len(pool),
        int(admitted.sum()),
    )
    if pool.empty:
        raise ValueError("rank pool is empty; nothing to decide on")
    return pool


def bundle_model_version(bundle: dict[str, Any]) -> str:
    """Build a deterministic model version string from bundle metadata.

    Args:
        bundle: Model bundle carrying strategy_id, training_cutoff and trained_at.

    Returns:
        Version string in ``strategy_id@training_cutoff@trained_at`` form.
    """
    return f"{bundle.get('strategy_id', 'UNKNOWN')}@{bundle.get('training_cutoff', 'UNKNOWN')}@{bundle.get('trained_at', 'UNKNOWN')}"


def build_rank_pool_frame(
    scored: pd.DataFrame, picks: pd.DataFrame, name_map: dict[str, str], model_version: str
) -> pd.DataFrame:
    """Build the ranked full-pool prediction frame.

    Args:
        scored: Scored full rank pool with symbol and pred columns.
        picks: Selected picks frame with a symbol column.
        name_map: Symbol to name mapping.
        model_version: Model version string to stamp.

    Returns:
        Ranked frame with selected flags, names, model version and 1-based rank.
    """
    out = scored.copy()
    if "symbol" in picks.columns:
        selected = out["symbol"].isin(picks["symbol"])
    else:
        selected = pd.Series(False, index=out.index)
    out["selected"] = selected.to_numpy(dtype=bool)
    out["name"] = out["symbol"].map(name_map)
    out["model_version"] = model_version
    out = out.sort_values("pred", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def persist_rank_pool_predictions(
    decision_date: pd.Timestamp, pool_df: pd.DataFrame, *, code_commit: str
) -> int:
    """Persist the full rank-pool predictions for a decision date.

    The existing-store read, merge and atomic write hold the store write lock, so an overlapping manual
    rerun cannot drop another decision date's pool.

    Args:
        decision_date: Decision date for the pool.
        pool_df: Rank pool frame to persist.
        code_commit: Code commit hash stamped on each row.

    Returns:
        Number of pool rows persisted.

    Raises:
        StoreLockTimeoutError: The store lock was not acquired in time; nothing was written.
    """
    if pool_df is None or len(pool_df) == 0:
        return 0
    out = pool_df.copy()
    out["decision_date"] = pd.Timestamp(decision_date).strftime("%Y-%m-%d")
    out["decided_at"] = pd.Timestamp.now(tz="Asia/Seoul")
    out["code_commit"] = code_commit
    target = settings.PARQUET_DIR / RANK_POOL_PARQUET_NAME
    with store_write_lock(target, purpose="rank-pool"):
        if target.exists():
            existing = pd.read_parquet(target)
            union_cols = sorted(set(existing.columns.tolist()) | set(out.columns.tolist()))
            merged = pd.concat(
                [existing.reindex(columns=union_cols), out.reindex(columns=union_cols)],
                ignore_index=True,
            )
            merged = merged.drop_duplicates(subset=["decision_date", "symbol"], keep="last")
        else:
            merged = out
        atomic_write_parquet(merged, target)
    return len(pool_df)


def resolve_code_commit(run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str:
    """Resolve the current git short commit hash.

    Args:
        run_fn: Subprocess runner (injectable for tests).

    Returns:
        Short commit hash, or ``UNKNOWN`` when git is unavailable.
    """
    env_commit = resolve_code_commit_env()
    if env_commit != "UNKNOWN":
        return env_commit
    try:
        completed = run_fn(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return str(completed.stdout).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("[SYS] stage=code_commit status=UNKNOWN reason=%s", type(exc).__name__)
        return "UNKNOWN"


def validate_index_inputs(wide: pd.DataFrame) -> None:
    """Fail closed when the decision snapshot's market-index inputs are unusable.

    collect.py records every index-fetch failure in 지수_실패 and leaves NaN in
    kospi/kosdaq/v_kospi. The model was certified only on finite index features,
    so scoring such a snapshot would rank on an input regime never seen in
    training; the sleeve must abstain instead.

    Args:
        wide: Decision snapshot (Korean columns) as loaded for inference.

    Raises:
        ValueError: 지수_실패 or v_kospi column is absent; any row's 지수_실패 is
            not boolean False; any v_kospi is non-finite or <= 0. The message names
            the failed check and the offending row count.
    """
    if "지수_실패" not in wide.columns:
        raise ValueError(f"index flag absent: 지수_실패 column missing n_rows={len(wide)}")
    if "v_kospi" not in wide.columns:
        raise ValueError(f"index value absent: v_kospi column missing n_rows={len(wide)}")
    flag = wide["지수_실패"]
    bad_flag = 0
    for value in flag.tolist():
        if isinstance(value, (bool, np.bool_)) and bool(value) is False:
            continue
        bad_flag += 1
    if bad_flag:
        raise ValueError(f"index flag set: 지수_실패 is not False n_bad={bad_flag} n_rows={len(wide)}")
    coerced = pd.to_numeric(wide["v_kospi"], errors="coerce").to_numpy(dtype=np.float64)
    bad_value = int((~np.isfinite(coerced) | (coerced <= 0)).sum())
    if bad_value:
        raise ValueError(f"index value invalid: v_kospi non-finite or <= 0 n_bad={bad_value} n_rows={len(wide)}")


def run_topk_ranker_sleeve(decision_date: pd.Timestamp, *, on_failure: Callable[[Exception], None] | None = None, on_rank_pool: Callable[[pd.DataFrame], None] | None = None) -> pd.DataFrame:
    """자동 top-3 리랭커 슬리브를 실행한다.

    Args:
        decision_date: 리랭커 피처에 찍는 결정 일자.
        on_failure: 삼켜진 예외를 받는 콜백(호출자가 실행 결과를 분류할 수 있게 한다).
        on_rank_pool: 전체 랭크풀 예측을 받는 콜백(영속은 호출자가 담당한다).

    Returns:
        등가중 top-k 선정 결과. 저장소에 기록된 ``admitted`` 컬럼을 그대로
        사용하며 재계산하지 않는다. 번들이 없거나 wide 단면이 비정상이거나
        price_history 가 없거나 직전 거래일까지 갱신되지 않았거나 지수 입력이
        fail-closed 조건(지수_실패/v_kospi)에 걸리면 경고를 남기고 빈 프레임을
        반환한다.
    """
    try:
        from src.ml import topk_history_features
        from src.ml.topk_contract import TOPK_RANKER_BUNDLE_DIR, score_topk_candidates, select_topk_equal_weight
        from src.serving.realtime.features import build_topk_ranker_features
        from src.strategy.contract import MIN_TOP_K

        inference_started_at = datetime.now(ZoneInfo("Asia/Seoul"))
        wide = restrict_to_rank_pool(load_daily_snapshot(decision_date, available_by=inference_started_at), decision_date)
        bundle = load_model_bundle(import_dir=TOPK_RANKER_BUNDLE_DIR)
        validate_index_inputs(wide)
        # 번들이 선언한 피처가 이력 피처를 요구할 때만 price_history 를 읽는다
        price_history = None
        if set(bundle.get("feature_cols", [])) & (set(topk_history_features.TOPK_HISTORY_FEATURE_COLS) | set(topk_history_features.TOPK_FLOW_FEATURE_COLS)):
            price_history = topk_history_features.load_serving_price_history(decision_date)
        features_df = build_topk_ranker_features(wide, decision_date, price_history=price_history)
        features_df["admitted"] = wide["admitted"].to_numpy()
        picks = select_topk_equal_weight(
            features_df, bundle, top_k=int(bundle.get("top_k", MIN_TOP_K))
        )
        name_map = dict(
            wide[["종목코드", "종목명"]].itertuples(index=False, name=None)
        )
        picks["name"] = picks["symbol"].map(name_map)
        model_version = bundle_model_version(bundle)
        picks["model_version"] = model_version
        if "capture_run_id" in wide.columns:
            picks["capture_run_id"] = str(wide["capture_run_id"].iloc[0])
        if "cohort_id" in wide.columns:
            picks["cohort_id"] = str(wide["cohort_id"].iloc[0])
        picks["inference_started_at"] = inference_started_at
        if "feature_available_timestamp" in wide.columns:
            picks["input_available_at"] = pd.to_datetime(wide["feature_available_timestamp"]).max()
        if on_rank_pool is not None:
            pool = build_rank_pool_frame(score_topk_candidates(features_df, bundle), picks, name_map, model_version)
            if "capture_run_id" in picks.columns:
                pool["capture_run_id"] = picks["capture_run_id"].iloc[0]
            if "cohort_id" in picks.columns:
                pool["cohort_id"] = picks["cohort_id"].iloc[0]
            pool["inference_started_at"] = inference_started_at
            if "input_available_at" in picks.columns:
                pool["input_available_at"] = picks["input_available_at"].iloc[0]
            on_rank_pool(pool)
        return picks
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "[ALGO] stage=topk_sleeve status=NO_DECISION date=%s reason=%s: %s",
            pd.Timestamp(decision_date).date(),
            type(exc).__name__,
            exc,
        )
        if on_failure is not None:
            on_failure(exc)
        return pd.DataFrame()


def persist_topk_decision(decision_date: pd.Timestamp, sleeve_df: pd.DataFrame) -> int:
    """Persist the automated top-k decision sleeve to the audit parquet store.

    Read, merge and atomic write run under the store write lock; a lock timeout fails the run closed (the
    decision is not persisted and the unit fails) rather than risking a lost update of the audit trail.

    Args:
        decision_date: Decision date.
        sleeve_df: Selected top-k rows.

    Returns:
        Number of decision rows persisted (0 for an empty sleeve, without locking).

    Raises:
        ExistingStoreUnreadableError: The existing store is unreadable (unchanged contract).
        StoreLockTimeoutError: The store lock was not acquired in time; nothing was written.
    """
    if sleeve_df.empty:
        return 0
    from src.ml.topk_contract import TOPK_RANKER_BUNDLE_DIR

    out = sleeve_df.copy()
    out["decision_date"] = decision_date.strftime("%Y-%m-%d")
    out["decided_at"] = pd.Timestamp.now(tz="Asia/Seoul")
    out["bundle_dir"] = str(TOPK_RANKER_BUNDLE_DIR)
    target = settings.PARQUET_DIR / TOPK_DECISIONS_PARQUET_NAME
    with store_write_lock(target, purpose="topk-decisions"):
        try:
            existing = read_existing_parquet(target)
        except ExistingStoreUnreadableError:
            logger.error(
                "[PORTFOLIO] stage=topk_persist status=ABORT reason=history_unreadable path=%s",
                target,
            )
            raise
        if not existing.empty:
            union_cols = sorted(set(existing.columns.tolist()) | set(out.columns.tolist()))
            merged = pd.concat(
                [existing.reindex(columns=union_cols), out.reindex(columns=union_cols)],
                ignore_index=True,
            )
            merged = merged.drop_duplicates(subset=["decision_date", "symbol"], keep="last")
        else:
            merged = out
        atomic_write_parquet(merged, target)
    return len(sleeve_df)


def load_topk_decision(decision_date: pd.Timestamp) -> pd.DataFrame:
    """Load the persisted top-k decision for a decision date.

    Args:
        decision_date: Decision date to look up.

    Returns:
        Requested-date rows deduplicated by symbol keeping the last write,
        or an empty frame when the store is absent.
    """
    target = settings.PARQUET_DIR / TOPK_DECISIONS_PARQUET_NAME
    if not target.exists():
        return pd.DataFrame()
    df = pd.read_parquet(target)
    out = df[df["decision_date"].astype(str) == pd.Timestamp(decision_date).strftime("%Y-%m-%d")]
    return out.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)


def run_automated_topk_decision(
    decision_date: pd.Timestamp,
    *,
    record_fn: Callable[..., Any] | None = None,
    trading_day_fn: Callable[[str], bool] | None = None,
    session_day_fn: Callable[[date], SessionDay] | None = None,
) -> None:
    """Print the single automated-mode top-3 decision table, if any.

    The session gate runs before any model or snapshot access: only a STANDARD
    session matches the 15:20 decision / 15:30 close geometry the model was
    certified on, so SHIFTED and UNKNOWN dates are NO_DECISION without
    inference, and a calendar closure that the KIS oracle contradicts is
    reported instead of silently skipped.

    Args:
        decision_date: Decision date for the reranker sleeve.
        record_fn: Run outcome recorder (bound to record_run_outcome).
        trading_day_fn: Trading-day oracle consulted on failure and on calendar closures.
        session_day_fn: Session resolver; None uses resolve_session_day.
    """
    date_str = pd.Timestamp(decision_date).strftime("%Y-%m-%d")
    resolve = session_day_fn if session_day_fn is not None else resolve_session_day
    day = resolve(decision_date.date())
    gate = trading_session_gate(day)
    if gate is not None:
        if day.kind is SessionKind.CLOSED:
            trading_day = classify_day(date_str, trading_day_fn)
            if trading_day in (DAY_WEEKEND, DAY_HOLIDAY):
                outcome, reason = RUN_OUTCOME_OK, "non_trading_day"
            else:
                outcome, reason = RUN_OUTCOME_NO_DECISION, "calendar_disagreement"
            metrics: dict[str, Any] = {"n_picks": 0, "day": trading_day, "session": day.kind.value}
        else:
            outcome, reason = RUN_OUTCOME_NO_DECISION, gate
            metrics = {"n_picks": 0, "session": day.kind.value}
        logger.warning("오늘 자동 유니버스 기준 진입 후보 없음(미참여)")
        if record_fn is not None:
            record_fn(outcome, run_date=date_str, reason=reason, metrics=metrics)
        return
    failures: list[Exception] = []
    pools: list[pd.DataFrame] = []
    sleeve_df = run_topk_ranker_sleeve(decision_date, on_failure=failures.append, on_rank_pool=pools.append)
    if failures:
        trading_day = classify_day(date_str, trading_day_fn)
        if trading_day in (DAY_WEEKEND, DAY_HOLIDAY):
            outcome, reason = RUN_OUTCOME_OK, "non_trading_day"
        else:
            outcome, reason = RUN_OUTCOME_NO_DECISION, f"{type(failures[0]).__name__}: {failures[0]}"
        logger.warning("오늘 자동 유니버스 기준 진입 후보 없음(미참여)")
        if record_fn is not None:
            record_fn(outcome, run_date=date_str, reason=reason, metrics={"n_picks": 0, "day": trading_day})
        return
    if sleeve_df.empty:
        logger.warning("오늘 자동 유니버스 기준 진입 후보 없음(미참여)")
        if record_fn is not None:
            record_fn(RUN_OUTCOME_OK, run_date=date_str, reason="admitted_below_top_k", metrics={"n_picks": 0})
        if pools:
            persist_rank_pool_predictions(decision_date, pools[-1], code_commit=resolve_code_commit())
        return
    persist_topk_decision(decision_date, sleeve_df)
    # 분석용 풀 예측 저장 실패가 매매 결정 영속을 막지 않도록 결정 저장 이후에 기록한다
    if pools:
        persist_rank_pool_predictions(decision_date, pools[-1], code_commit=resolve_code_commit())
    if record_fn is not None:
        record_fn(RUN_OUTCOME_OK, run_date=date_str, reason="", metrics={"n_picks": len(sleeve_df)})
    rows = [
        {
            "Code": str(row.get("symbol", "")),
            "Name": row.get("name", ""),
            "Pred": round(float(row.get("pred", 0.0)), 4),
            "Alloc%": round(float(row.get("allocation", 0.0)) * 100.0, 2),
        }
        for _, row in sleeve_df.iterrows()
    ]
    print_table(rows, "Top-3 Cost-Aware Decision (Equal-Weight)")


def main() -> None:
    decision_date = pd.Timestamp.today().normalize()
    run_automated_topk_decision(decision_date, record_fn=functools.partial(record_run_outcome, "predict"))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
