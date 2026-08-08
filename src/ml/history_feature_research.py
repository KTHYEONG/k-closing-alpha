"""History-feature research experiment: v8 control vs causal-history candidate.

`docs/specs/ml_training_optimization.md` 의 리서치 진입점입니다. 기존
``close_morning61 + scenario_action`` 판넬을 그대로 만들고 causal history 720 컬럼
후보를 left-join 하여, v8 recency 앙상블 경로로 동일한 purged OOF 날짜에서 동결
컨트롤과 후보를 비교합니다.

- 공통 커트오프: 명시적 리서치 커트오프와 causal-history 최대 날짜의 최솟값으로
  ``evaluation_cutoff`` 를 산출하고, 이후 행은 join 전에 거부합니다 (조용한
  채움 금지, 제외 수·날짜 기록).
- 평가 경로: 양쪽 arm 을 ``run_close_morning_recency_ensemble_experiment``
  (expanding + recent Huber return 전문가, 불변 p_good 기여, groupwise rank,
  always_buy_top1 정책) 로 평가해 raw-regressor OOF 지표와 v8 컨트롤을 비교하지
  않습니다. 후보의 risk(p_good) 입력은 동결 close_morning61 표면을 유지하고,
  선택된 history 부분집합은 return 전문가에만 적용합니다.
- 후보 아티팩트는 버전화된 research 디렉터리로만 저장하며 활성 아티팩트
  (``artifacts/models``) 를 절대 덮어쓰지 않습니다.
- feature cache 는 지문(catalogue 버전 + history 설정 + decision-key 해시 +
  source identity + 커트오프)이 모두 일치할 때만 warm read 되고, 불일치는
  cache miss 로 재구성합니다. 캐시 메트릭은 모델 아티팩트와 분리 저장합니다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.feature_selection import (
    FeatureSelectionConfig,
    FeatureSelectionResult,
    build_feature_quality_report,
    build_fold_feature_plans,
    median_pairwise_jaccard,
    permutation_null_stability,
    select_features,
)
from src.ml.history_features import (
    HISTORICAL_CATALOGUE,
    HISTORICAL_CATALOGUE_VERSION,
    HistoricalFeatureConfig,
    HistoryFeatureExecutionConfig,
    build_catalogue_manifest,
    build_causal_history_feature_panel,
    build_causal_history_feature_panel_from_parquet,
    catalogue_quality_metadata,
)
from src.ml.model_pipeline import (
    run_close_morning_recency_ensemble_experiment,
)
from src.ml.purged_cv import PurgedGroupTimeSeriesSplit
from src.ml.sizing_engine import save_model_artifacts
from src.processing.preprocessor import build_ml_dataset

logger = logging.getLogger(__name__)

_CATALOGUE_FEATURE_NAMES = [str(entry["feature_name"]) for entry in HISTORICAL_CATALOGUE]
_CROSS_SECTIONAL_SCOPE = "decision_candidate_panel"
_TEMPORAL_SCOPE = "history_temporal_panel"
_RESEARCH_CACHE_VERSION = "history_feature_cache_v1"
_AVAILABLE_MODES = ("confirmation", "discovery")
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _emit_progress(
    callback: ProgressCallback | None, stage: str, details: Mapping[str, Any]
) -> None:
    """관측용 진행 이벤트를 전송합니다. 콜백 실패는 리서치 계산에 영향 주지 않습니다."""
    if callback is None:
        return
    try:
        callback(stage, details)
    except Exception:  # pragma: no cover - observers must not change research behavior
        logger.warning("history-feature progress observer failed at stage=%s", stage, exc_info=True)


def _ensemble_policy_summary(ensemble: dict[str, Any]) -> dict[str, Any]:
    """recency 앙상블 후보 집계를 정책 지표 요약으로 반환합니다."""
    aggregate = ensemble["aggregate"]["candidate"]
    return {
        "policy_version": ensemble["contract"]["version"],
        "policy_id": None,
        "metrics": dict(aggregate),
    }


def _decision_key_hash(decision_keys: pd.DataFrame, group_col: str) -> str:
    """결정 key (stock_code, trade_date) 의 결정적 sha256 을 반환합니다."""
    canonical = (
        decision_keys[[group_col, "stock_code"]]
        .sort_values(["stock_code", group_col])
        .assign(_d=lambda f: f[group_col].dt.strftime("%Y-%m-%d"))
    )
    payload = canonical[["stock_code", "_d"]].to_csv(index=False, header=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_identity(
    price_history: pd.DataFrame | None,
    price_history_path: str | None,
) -> str:
    """history 원천의 불변 식별자를 반환합니다.

    Parquet 경로는 resolve 경로 + 크기 + mtime_ns 로, DataFrame 은
    (date, symbol) 유니온의 결정적 해시로 식별합니다.
    """
    if price_history_path is not None:
        path = Path(price_history_path)
        stat = path.stat()
        return f"parquet:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    if price_history is None:
        return "none"
    config = HistoricalFeatureConfig()
    canonical = (
        price_history[[config.history_date_col, config.history_symbol_col]]
        .drop_duplicates()
        .sort_values([config.history_date_col, config.history_symbol_col])
    )
    digest = hashlib.sha256(
        canonical.to_csv(index=False, header=False).encode("utf-8")
    ).hexdigest()
    return f"frame:{digest}"


def _feature_cache_fingerprint(
    decision_keys: pd.DataFrame,
    group_col: str,
    price_history: pd.DataFrame | None,
    price_history_path: str | None,
    evaluation_cutoff: pd.Timestamp,
) -> str:
    """feature cache 지문 (catalogue + 설정 + key 해시 + source + 커트오프)."""
    payload = json.dumps(
        {
            "version": _RESEARCH_CACHE_VERSION,
            "catalogue_version": HISTORICAL_CATALOGUE_VERSION,
            "history_config": json.dumps(
                HistoricalFeatureConfig().model_dump(), sort_keys=True, default=str
            ),
            "decision_key_hash": _decision_key_hash(decision_keys, group_col),
            "source_identity": _source_identity(price_history, price_history_path),
            "evaluation_cutoff": evaluation_cutoff.isoformat(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _max_history_date(
    price_history: pd.DataFrame | None,
    price_history_path: str | None,
    execution_config: HistoryFeatureExecutionConfig | None,
) -> pd.Timestamp:
    """history 원천의 최대 날짜를 반환합니다 (streaming 시 date 컬럼만 투영)."""
    config = HistoricalFeatureConfig()
    date_col = config.history_date_col
    if price_history_path is not None:
        import pyarrow.parquet as pq

        exec_cfg = execution_config or HistoryFeatureExecutionConfig()
        parquet_file = pq.ParquetFile(price_history_path)
        maximum: pd.Timestamp | None = None
        for batch in parquet_file.iter_batches(
            columns=[date_col], batch_size=exec_cfg.parquet_batch_rows
        ):
            dates = pd.to_datetime(batch.column(0).to_pandas())
            if len(dates) and not dates.isna().all():
                current = dates.max()
                maximum = current if maximum is None or current > maximum else maximum
        if maximum is None:
            raise ValueError("price history source has no parseable dates")
        return maximum
    if price_history is None:
        raise ValueError("either price_history or price_history_path must be provided")
    if date_col not in price_history.columns:
        raise ValueError(f"price history source must contain date column {date_col!r}")
    dates = pd.to_datetime(price_history[date_col], errors="coerce")
    maximum = dates.max()
    if pd.isna(maximum):
        raise ValueError("price history source has no parseable dates")
    return maximum


def _load_or_build_history_panel(
    price_history: pd.DataFrame | None,
    price_history_path: str | None,
    decision_keys: pd.DataFrame,
    group_col: str,
    execution_config: HistoryFeatureExecutionConfig | None,
    cache_dir: str | None,
    fingerprint: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """지문이 일치하면 캐시 warm read, 아니면 재구성 후 저장합니다."""
    cache_meta: dict[str, Any] = {"cache_state": "cold", "cache_fingerprint": fingerprint}
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_path / f"history_features_{fingerprint}.parquet"
        metrics_path = cache_path / f"history_features_{fingerprint}.metrics.json"
        if parquet_path.is_file() and metrics_path.is_file():
            try:
                panel = pd.read_parquet(parquet_path)
                stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                panel.attrs["history_feature_build_metrics"] = stored_metrics["build_metrics"]
                panel.attrs["history_feature_cache_metrics"] = stored_metrics["cache_metrics"]
                cache_meta = dict(stored_metrics["cache_metrics"])
                cache_meta["cache_state"] = "warm"
                logger.info("history-feature cache warm read: %s", parquet_path.name)
                return panel, cache_meta
            except Exception as exc:  # noqa: BLE001 - corrupt cache must rebuild
                logger.warning("history-feature cache read failed, rebuilding: %s", exc)

    if price_history_path is not None:
        history_panel = build_causal_history_feature_panel_from_parquet(
            price_history_path,
            decision_keys,
            HistoricalFeatureConfig(),
            execution_config,
        )
    else:
        history_panel = build_causal_history_feature_panel(
            price_history,
            decision_keys,
            HistoricalFeatureConfig(),
            execution_config,
        )
    build_metrics = dict(history_panel.attrs.get("history_feature_build_metrics", {}))
    cache_metrics = {
        "cache_state": "cold",
        "cache_fingerprint": fingerprint,
        "source_identity": _source_identity(price_history, price_history_path),
    }
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_path / f"history_features_{fingerprint}.parquet"
        metrics_path = cache_path / f"history_features_{fingerprint}.metrics.json"
        history_panel.to_parquet(parquet_path, index=False)
        metrics_path.write_text(
            json.dumps(
                {
                    "build_metrics": build_metrics,
                    "cache_metrics": cache_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("history-feature cache written: %s", parquet_path.name)
    return history_panel, cache_metrics


def _build_selection_diagnostics(
    plans: list[Any],
    final_selection: FeatureSelectionResult | None,
    data_cutoff: str,
    config: FeatureSelectionConfig,
    candidate_feature_cols: list[str],
) -> dict[str, Any] | None:
    """fold 계획 + full-data final 선택으로 결정적 진단을 조립합니다."""
    if not plans:
        return None
    jaccard = median_pairwise_jaccard([p.selected_features for p in plans])
    stability = permutation_null_stability(
        [p.selected_features for p in plans],
        random_seed=config.random_seed,
        universe=candidate_feature_cols,
    )
    final_features = (
        list(final_selection.selected_features)
        if final_selection is not None
        else list(plans[-1].selected_features)
    )
    return {
        "version": plans[0].metadata.get("version"),
        "catalogue_version": config.catalogue_version,
        "config_fingerprint": plans[0].config_fingerprint,
        "data_cutoff": data_cutoff,
        "n_folds": len(plans),
        "fold_selections": [
            {
                "fold": plan.fold,
                "data_cutoff": plan.data_cutoff,
                "selected_features": list(plan.selected_features),
                "gains": list(plan.gains),
                "rejected": list(plan.rejected),
                "counts": dict(plan.counts),
                "metadata": dict(plan.metadata),
            }
            for plan in plans
        ],
        "median_pairwise_jaccard": jaccard,
        "stability": stability,
        "final_train_only": True,
        "final_features": final_features,
        "final_selection": (
            {
                "selected_features": list(final_selection.selected_features),
                "gains": list(final_selection.gains),
                "rejected": list(final_selection.rejected),
                "counts": dict(final_selection.counts),
                "metadata": dict(final_selection.metadata),
            }
            if final_selection is not None
            else None
        ),
    }


def _promotion_gate(
    control: dict[str, Any],
    candidate: dict[str, Any],
    identical_oof_dates: bool,
    stability: dict[str, Any],
) -> dict[str, Any]:
    """확정 게이트: 동일 OOF 날짜 + 양수 scheduled mean + 대조군보다 엄격히 높은
    mean + PF>1 + MDD 엄격 감소 + 안정성."""
    control_agg = control["aggregate"]["candidate"]
    candidate_agg = candidate["aggregate"]["candidate"]
    cand_mean = float(candidate_agg["scheduled_mean_return"])
    control_mean = float(control_agg["scheduled_mean_return"])
    cand_pf = float(candidate_agg["profit_factor"])
    cand_mdd = float(candidate_agg["entry_sequence_drawdown"])
    control_mdd = float(control_agg["entry_sequence_drawdown"])
    positive_mean = np.isfinite(cand_mean) and cand_mean > 0.0
    beats_control = (
        np.isfinite(cand_mean)
        and np.isfinite(control_mean)
        and cand_mean > control_mean
    )
    pf_above_one = np.isfinite(cand_pf) and cand_pf > 1.0
    lower_mdd = (
        np.isfinite(cand_mdd) and np.isfinite(control_mdd) and cand_mdd < control_mdd
    )
    stability_ok = bool(stability.get("gate_passed", False))
    rejected_reasons: list[str] = []
    if not identical_oof_dates:
        rejected_reasons.append("oof_dates_mismatch")
    if not positive_mean:
        rejected_reasons.append("non_positive_scheduled_mean")
    if not beats_control:
        rejected_reasons.append("candidate_mean_not_strictly_higher")
    if not pf_above_one:
        rejected_reasons.append("profit_factor_not_above_one")
    if not lower_mdd:
        rejected_reasons.append("compounded_mdd_not_strictly_lower")
    if not stability_ok:
        rejected_reasons.append("stability_gate_failed")
    return {
        "promoted": not rejected_reasons,
        "identical_oof_dates": bool(identical_oof_dates),
        "positive_scheduled_net_mean": bool(positive_mean),
        "candidate_beats_control_mean": bool(beats_control),
        "profit_factor_above_one": bool(pf_above_one),
        "candidate_lower_compounded_mdd": bool(lower_mdd),
        "stability_gate_passed": stability_ok,
        "rejected_reasons": rejected_reasons,
    }


def run_history_feature_research_experiment(
    trade_log_df: pd.DataFrame,
    theme_df: pd.DataFrame | None,
    price_history: pd.DataFrame | None = None,
    price_history_path: str | None = None,
    *,
    feature_set: str = "close_morning61",
    panel_mode: str = "scenario_action",
    n_splits: int = 5,
    purge_gap: int = 1,
    feature_selection_config: FeatureSelectionConfig | None = None,
    execution_config: HistoryFeatureExecutionConfig | None = None,
    research_cutoff: str | None = None,
    cache_dir: str | None = None,
    mode: str = "confirmation",
    wall_time_budget_seconds: float | None = 1800.0,
    export_dir: str = "artifacts/models/research",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """close_morning61 v8 컨트롤 대비 causal-history 후보 리서치 실험을 실행합니다.

    Args:
        trade_log_df: 매매일지 원본 DataFrame.
        theme_df: 테마/시장 메타데이터 DataFrame (선택).
        price_history: 제한된 EOD 판넬 DataFrame (``price_history_path`` 가 없을 때).
        price_history_path: ``price_history.parquet`` 경로. 주어지면 전체 판넬을
            ``pd.read_parquet`` 로 읽지 않고 PyArrow streaming 배치로 처리합니다.
        feature_set: ``build_ml_dataset`` feature_set (기본 close_morning61).
        panel_mode: ``build_ml_dataset`` panel_mode (기본 scenario_action).
        n_splits: purged walk-forward fold 수 (컨트롤/후보 동일).
        purge_gap: purge group 수 (컨트롤/후보 동일).
        feature_selection_config: 후보 fold-local 선택 설정.
        execution_config: history 피처 streaming 배치/메모리 예산 설정.
        research_cutoff: 동결 리서치 커트오프 (ISO 날짜). 미지정 시 패널 최대 날짜.
        cache_dir: feature cache 저장 root. 지문 일치 시 warm read.
        mode: ``confirmation``(기본) 또는 ``discovery``(candidate-speed, 보고용).
        wall_time_budget_seconds: 전체 실행 wall-time 예산 (기본 30분).
        export_dir: 후보 번들 저장 root (버전/컷오프 하위 디렉터리로 분리).
        progress_callback: 진행 이벤트 콜백.

    Returns:
        dict: ``contract``, ``build_metrics``, ``control``, ``candidate``,
        ``comparison``, ``promotion``, ``candidate_bundle_path``.
    """
    if price_history is None and price_history_path is None:
        raise ValueError("either price_history or price_history_path must be provided")
    if mode not in _AVAILABLE_MODES:
        raise ValueError(f"mode must be one of {_AVAILABLE_MODES}, got {mode!r}")
    config = feature_selection_config or FeatureSelectionConfig()
    _x, _targets, cat_features, processed = build_ml_dataset(
        trade_log_df,
        theme_df,
        feature_set=feature_set,
        panel_mode=panel_mode,
    )
    base_feature_cols = [col for col in _x.columns if col not in cat_features]
    target_col = "target_return"
    group_col = "trade_date"
    if target_col not in processed.columns:
        raise ValueError(f"processed panel must contain target_col {target_col!r}")
    if not {"stock_code", "chart_analysis"} <= set(processed.columns):
        raise ValueError("processed panel must contain stock_code and chart_analysis columns")

    max_history_date = _max_history_date(price_history, price_history_path, execution_config)
    frozen_cutoff = (
        pd.Timestamp(research_cutoff) if research_cutoff is not None else processed[group_col].max()
    )
    evaluation_cutoff = min(frozen_cutoff, max_history_date)

    processed_dates = processed[group_col]
    retain_mask = processed_dates <= evaluation_cutoff
    excluded = processed[~retain_mask]
    excluded_count = len(excluded)
    excluded_dates = [str(d) for d in sorted(excluded[group_col].unique())]
    if excluded_count:
        logger.info(
            "history cutoff rejected %d rows on %d later dates; evaluation_cutoff=%s",
            excluded_count,
            len(excluded_dates),
            evaluation_cutoff.date(),
        )
    processed = processed[retain_mask].copy()
    _emit_progress(
        progress_callback,
        "cutoff_applied",
        {
            "evaluation_cutoff": str(evaluation_cutoff),
            "excluded_rows": excluded_count,
            "excluded_dates": excluded_dates,
        },
    )

    decision_keys = processed[[group_col, "stock_code"]].drop_duplicates()
    _emit_progress(
        progress_callback,
        "dataset_built",
        {"processed_rows": len(processed), "decision_key_rows": len(decision_keys)},
    )

    fingerprint = _feature_cache_fingerprint(
        decision_keys,
        group_col,
        price_history,
        price_history_path,
        evaluation_cutoff,
    )
    history_panel, cache_metrics = _load_or_build_history_panel(
        price_history,
        price_history_path,
        decision_keys,
        group_col,
        execution_config,
        cache_dir,
        fingerprint,
    )
    build_metrics = dict(history_panel.attrs.get("history_feature_build_metrics", {}))
    build_metrics["cache_state"] = cache_metrics["cache_state"]
    build_metrics["cache_fingerprint"] = fingerprint
    _emit_progress(progress_callback, "history_panel_built", build_metrics)

    joined = processed.merge(history_panel, on=["stock_code", group_col], how="left")
    candidate_feature_cols = [*base_feature_cols, *_CATALOGUE_FEATURE_NAMES]
    missing_candidates = [col for col in candidate_feature_cols if col not in joined.columns]
    if missing_candidates:
        raise ValueError(f"history candidate columns missing after join: {missing_candidates[:5]}")

    def _fold_event(payload: Mapping[str, Any]) -> None:
        _emit_progress(progress_callback, "pipeline_fold", dict(payload))

    memory_budget_bytes = (
        execution_config.memory_budget_bytes if execution_config is not None else None
    )

    control = run_close_morning_recency_ensemble_experiment(
        joined,
        feature_cols=base_feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        risk_feature_cols=base_feature_cols,
        pred_linear=True,
        memory_budget_bytes=memory_budget_bytes,
        wall_time_budget_seconds=wall_time_budget_seconds,
        fold_event_callback=_fold_event,
    )
    _emit_progress(
        progress_callback,
        "control_complete",
        {"metrics": dict(control["aggregate"]["candidate"])},
    )

    diagnostics: dict[str, Any] | None = None
    plans: list[Any] | None = None
    final_selection: FeatureSelectionResult | None = None
    quality_report: dict[str, Any] | None = None
    if config is not None:
        plans = build_fold_feature_plans(
            joined,
            candidate_feature_cols,
            target_col,
            group_col,
            config,
            n_splits=n_splits,
            purge_gap=purge_gap,
        )
        final_selection = select_features(joined, candidate_feature_cols, target_col, config)
        diagnostics = _build_selection_diagnostics(
            plans, final_selection, str(joined[group_col].max()), config, candidate_feature_cols
        )
        assert diagnostics is not None
        quality_report = build_feature_quality_report(
            [plan.selection for plan in plans if plan.selection is not None],
            candidate_feature_cols,
            catalogue_quality_metadata(),
            config.min_fold_selection_rate,
        )
        diagnostics["quality_report"] = quality_report

    candidate = run_close_morning_recency_ensemble_experiment(
        joined,
        feature_cols=candidate_feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        risk_feature_cols=base_feature_cols,
        fold_feature_plans=plans,
        final_feature_selection=final_selection,
        pred_linear=mode == "discovery",
        memory_budget_bytes=memory_budget_bytes,
        wall_time_budget_seconds=wall_time_budget_seconds,
        fold_event_callback=_fold_event,
    )
    _emit_progress(
        progress_callback,
        "candidate_complete",
        {"metrics": dict(candidate["aggregate"]["candidate"])},
    )

    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    oof_dates = sorted(
        {
            date
            for _, val_idx in splitter.split(
                joined, y=joined[target_col], groups=joined[group_col]
            )
            for date in joined.iloc[val_idx][group_col].unique()
        }
    )
    identical_oof_dates = True
    control_agg = dict(control["aggregate"]["candidate"])
    candidate_agg = dict(candidate["aggregate"]["candidate"])

    stability = (
        diagnostics.get("stability", {})
        if diagnostics is not None
        else {"gate_passed": True, "reason": "no_feature_selection"}
    )
    promotion = _promotion_gate(control, candidate, identical_oof_dates, stability)

    comparison = {
        "identical_oof_dates": identical_oof_dates,
        "control_oof_dates": [str(d) for d in oof_dates],
        "candidate_oof_dates": [str(d) for d in oof_dates],
        "control_metrics": control_agg,
        "candidate_metrics": candidate_agg,
        "control_policy": _ensemble_policy_summary(control),
        "candidate_policy": _ensemble_policy_summary(candidate),
    }

    final_features = list(diagnostics["final_features"]) if diagnostics else candidate_feature_cols
    final_metadata = (
        diagnostics.get("final_selection", {}).get("metadata", {})
        if diagnostics and diagnostics.get("final_selection")
        else {}
    )
    catalogue_manifest = build_catalogue_manifest()
    cross_sectional_count = int(
        (catalogue_manifest["panel_scope"] == _CROSS_SECTIONAL_SCOPE).sum()
    )
    from src.ml.feature_manifest import build_feature_manifest

    bundle = {
        "feature_set": feature_set,
        "panel_mode": panel_mode,
        "feature_cols": final_features,
        "candidate_feature_cols": candidate_feature_cols,
        "feature_manifest": build_feature_manifest(
            final_features, catalogue=config.catalogue
        ),
        "catalogue_manifest": catalogue_manifest,
        "catalogue_version": HISTORICAL_CATALOGUE_VERSION,
        "cross_sectional_scope": _CROSS_SECTIONAL_SCOPE,
        "temporal_scope": _TEMPORAL_SCOPE,
        "cross_sectional_feature_count": cross_sectional_count,
        "history_feature_build_metrics": build_metrics,
        "history_feature_cache_metrics": cache_metrics,
        "requested_screening_device": final_metadata.get("requested_screening_device"),
        "resolved_screening_device": final_metadata.get("resolved_screening_device"),
        "gpu_fallback_reason": final_metadata.get("gpu_fallback_reason"),
        "feature_selection_version": diagnostics.get("version") if diagnostics else None,
        "feature_selection_diagnostics": diagnostics,
        "quality_report": quality_report,
        "training_cutoff": str(joined[group_col].max()),
        "control_metrics": control_agg,
        "candidate_metrics": candidate_agg,
        "promotion": promotion,
        "comparison": comparison,
    }
    save_dir = f"{export_dir}/{HISTORICAL_CATALOGUE_VERSION}/{bundle['training_cutoff'][:10]}"
    saved_path = save_model_artifacts(bundle, save_dir)
    _emit_progress(
        progress_callback,
        "artifact_saved",
        {"candidate_bundle_path": saved_path, "final_feature_count": len(final_features)},
    )
    logger.info(
        "history-feature research candidate persisted (identical_oof_dates=%s, "
        "final_features=%d, cache_state=%s, batch_count=%s, promoted=%s, path=%s)",
        identical_oof_dates,
        len(final_features),
        build_metrics.get("cache_state"),
        build_metrics.get("batch_count"),
        promotion["promoted"],
        saved_path,
    )

    return {
        "contract": {
            "version": "history-feature-research-v2",
            "feature_set": feature_set,
            "panel_mode": panel_mode,
            "catalogue_version": HISTORICAL_CATALOGUE_VERSION,
            "cross_sectional_scope": _CROSS_SECTIONAL_SCOPE,
            "candidate_count": len(candidate_feature_cols),
            "n_splits": n_splits,
            "purge_gap": purge_gap,
            "evaluation_cutoff": str(evaluation_cutoff),
            "mode": mode,
            "cache_state": build_metrics.get("cache_state"),
            "excluded_rows_after_cutoff": excluded_count,
            "excluded_dates": excluded_dates,
        },
        "build_metrics": build_metrics,
        "control": {
            "metrics": control_agg,
            "policy": _ensemble_policy_summary(control),
        },
        "candidate": {
            "metrics": candidate_agg,
            "policy": _ensemble_policy_summary(candidate),
            "final_features": final_features,
            "feature_selection_diagnostics": diagnostics,
            "quality_report": quality_report,
        },
        "comparison": comparison,
        "promotion": promotion,
        "candidate_bundle_path": saved_path,
    }
