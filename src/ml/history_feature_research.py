"""History-feature research experiment: close_morning61 control vs causal-history candidate.

`docs/specs/ml_feature_selection_pipeline.md` 의 리서치 진입점입니다. 기존
``close_morning61 + scenario_action`` 판넬을 그대로 만들고 causal history 720 컬럼
후보를 left-join 하여, 동일한 purged OOF 날짜에서 동결 컨트롤과 후보를 비교합니다.

- 컨트롤: ``close_morning61`` 수치 피처만 사용하는 기존 ``run_model_pipeline``.
- 후보: 컨트롤 피처 + 720 history 후보에 fold-local ``feature_selection_config``
  적용.
- 후보 아티팩트는 버전화된 research 디렉터리로만 저장하며 활성 아티팩트
  (``artifacts/models``) 를 절대 덮어쓰지 않습니다.
- ``train_and_save_real_model_bundle`` 기본값은 변경하지 않습니다.

승격 조건은 리서치 전용이며, 고정 홀드아웃/paper 기간에서 positive scheduled net
mean, 0.20% 왕복 비용 후 profit factor > 1, 컨트롤 대비 낮은 compounded MDD 를
충족해야 합니다 (본 모듈은 결과 비교를 산출하며 승격 결정은 하지 않습니다).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from src.ml.feature_selection import FeatureSelectionConfig
from src.ml.history_features import (
    HISTORICAL_CATALOGUE,
    HISTORICAL_CATALOGUE_VERSION,
    HistoricalFeatureConfig,
    HistoryFeatureExecutionConfig,
    build_catalogue_manifest,
    build_causal_history_feature_panel,
    build_causal_history_feature_panel_from_parquet,
)
from src.ml.model_pipeline import run_model_pipeline
from src.ml.sizing_engine import save_model_artifacts
from src.processing.preprocessor import build_ml_dataset

logger = logging.getLogger(__name__)

_CATALOGUE_FEATURE_NAMES = [str(entry["feature_name"]) for entry in HISTORICAL_CATALOGUE]
_CROSS_SECTIONAL_SCOPE = "decision_candidate_panel"
_TEMPORAL_SCOPE = "history_temporal_panel"
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


def _policy_summary(result: dict[str, Any]) -> dict[str, Any]:
    """번들용 정책 지표 요약을 반환합니다 (정책 부재 시 빈 dict)."""
    metadata = result.get("policy_metadata")
    if not metadata:
        return {}
    metrics = dict(metadata.get("policy_metrics", {}))
    return {
        "policy_version": metadata.get("policy_version"),
        "policy_id": metadata.get("policy_id"),
        "metrics": metrics,
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
    export_dir: str = "artifacts/models/research",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """close_morning61 컨트롤 대비 causal-history 후보 리서치 실험을 실행합니다.

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
        export_dir: 후보 번들 저장 root (버전/컷오프 하위 디렉터리로 분리).

    Returns:
        dict: ``contract``, ``control``, ``candidate``, ``comparison``,
        ``candidate_bundle_path`` (후보 저장 경로).
    """
    if price_history is None and price_history_path is None:
        raise ValueError("either price_history or price_history_path must be provided")
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

    decision_keys = processed[[group_col, "stock_code"]].drop_duplicates()
    _emit_progress(
        progress_callback,
        "dataset_built",
        {"processed_rows": len(processed), "decision_key_rows": len(decision_keys)},
    )
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
    _emit_progress(progress_callback, "history_panel_built", build_metrics)
    joined = processed.merge(history_panel, on=["stock_code", group_col], how="left")
    candidate_feature_cols = [*base_feature_cols, *_CATALOGUE_FEATURE_NAMES]
    missing_candidates = [col for col in candidate_feature_cols if col not in joined.columns]
    if missing_candidates:
        raise ValueError(f"history candidate columns missing after join: {missing_candidates[:5]}")

    control = run_model_pipeline(
        joined,
        feature_cols=base_feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
    )
    _emit_progress(progress_callback, "control_complete", {"metrics": dict(control["metrics"])})
    candidate = run_model_pipeline(
        joined,
        feature_cols=candidate_feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
        feature_selection_config=config,
    )
    _emit_progress(
        progress_callback,
        "candidate_complete",
        {"metrics": dict(candidate["metrics"])},
    )

    control_oof = control["oof_predictions"]
    candidate_oof = candidate["oof_predictions"]
    control_dates = sorted(control_oof[group_col].unique())
    candidate_dates = sorted(candidate_oof[group_col].unique())
    identical_oof_dates = control_dates == candidate_dates

    comparison = {
        "identical_oof_dates": identical_oof_dates,
        "control_oof_dates": [str(d) for d in control_dates],
        "candidate_oof_dates": [str(d) for d in candidate_dates],
        "control_metrics": dict(control["metrics"]),
        "candidate_metrics": dict(candidate["metrics"]),
        "control_backtest": control["backtest_eval"],
        "candidate_backtest": candidate["backtest_eval"],
        "control_policy": _policy_summary(control),
        "candidate_policy": _policy_summary(candidate),
    }

    diagnostics = candidate.get("feature_selection_diagnostics")
    final_features = list(diagnostics["final_features"]) if diagnostics else candidate_feature_cols
    final_metadata = (
        diagnostics.get("final_selection", {}).get("metadata", {})
        if diagnostics
        else {}
    )
    catalogue_manifest = build_catalogue_manifest()
    cross_sectional_count = int(
        (catalogue_manifest["panel_scope"] == _CROSS_SECTIONAL_SCOPE).sum()
    )
    bundle = {
        "feature_set": feature_set,
        "panel_mode": panel_mode,
        "feature_cols": final_features,
        "candidate_feature_cols": candidate_feature_cols,
        "feature_manifest": candidate["feature_manifest"],
        "catalogue_manifest": catalogue_manifest,
        "catalogue_version": HISTORICAL_CATALOGUE_VERSION,
        "cross_sectional_scope": _CROSS_SECTIONAL_SCOPE,
        "temporal_scope": _TEMPORAL_SCOPE,
        "cross_sectional_feature_count": cross_sectional_count,
        "history_feature_build_metrics": build_metrics,
        "requested_screening_device": final_metadata.get("requested_screening_device"),
        "resolved_screening_device": final_metadata.get("resolved_screening_device"),
        "gpu_fallback_reason": final_metadata.get("gpu_fallback_reason"),
        "feature_selection_version": diagnostics.get("version") if diagnostics else None,
        "feature_selection_diagnostics": diagnostics,
        "training_cutoff": str(joined[group_col].max()),
        "control_metrics": dict(control["metrics"]),
        "candidate_metrics": dict(candidate["metrics"]),
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
        "final_features=%d, batch_count=%s, path=%s)",
        identical_oof_dates,
        len(final_features),
        build_metrics.get("batch_count"),
        saved_path,
    )

    return {
        "contract": {
            "version": "history-feature-research-v1",
            "feature_set": feature_set,
            "panel_mode": panel_mode,
            "catalogue_version": HISTORICAL_CATALOGUE_VERSION,
            "cross_sectional_scope": _CROSS_SECTIONAL_SCOPE,
            "candidate_count": len(candidate_feature_cols),
            "n_splits": n_splits,
            "purge_gap": purge_gap,
            "evaluation_cutoff": str(joined[group_col].max()),
        },
        "build_metrics": build_metrics,
        "control": {
            "metrics": dict(control["metrics"]),
            "policy": _policy_summary(control),
        },
        "candidate": {
            "metrics": dict(candidate["metrics"]),
            "policy": _policy_summary(candidate),
            "final_features": final_features,
            "feature_selection_diagnostics": diagnostics,
        },
        "comparison": comparison,
        "candidate_bundle_path": saved_path,
    }
