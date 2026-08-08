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
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

try:
    import psutil
except Exception:  # pragma: no cover - optional telemetry dependency
    psutil = None

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
from src.ml.sizing_engine import save_model_artifacts
from src.processing.preprocessor import build_ml_dataset

logger = logging.getLogger(__name__)

_CATALOGUE_FEATURE_NAMES = [str(entry["feature_name"]) for entry in HISTORICAL_CATALOGUE]
_CROSS_SECTIONAL_SCOPE = "decision_candidate_panel"
_TEMPORAL_SCOPE = "history_temporal_panel"
_RESEARCH_CACHE_VERSION = "history_feature_cache_v1"
_AVAILABLE_MODES = ("confirmation", "discovery")
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _normalize_oof_dates(dates: np.ndarray) -> tuple[bool, list[str]]:
    """실제 scheduled-return 날짜 배열을 표준화합니다.

    각 날짜를 파싱해 유일·오름차순 문자열 목록으로 정규화하고, 파싱 불가·중복·
    빈 배열이면 ``(False, [])`` 를 반환합니다. 표준화는 재현 가능하도록 시간대가
    보존된 ISO 문자열을 사용합니다.
    """
    if dates is None or len(dates) == 0:
        return False, []
    series = pd.Series(np.asarray(dates, dtype=object))
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        return False, []
    if parsed.duplicated().any():
        return False, []
    normalized = sorted(d.isoformat() for d in parsed.drop_duplicates())
    return True, normalized


def validate_research_oof_alignment(
    control_dates: np.ndarray, candidate_dates: np.ndarray
) -> tuple[bool, list[str], list[str]]:
    """컨트롤/후보 실험의 실제 OOF 날짜 배열 동일성을 검증합니다.

    두 실험의 실제 scheduled-return 날짜 배열을 각각 표준화한 뒤 비교해, 누락·
    추가·재배열·중복·파싱 불가가 하나라도 있으면 ``(False, [], [])`` 를 반환합니다.
    표준화된 두 날짜 목록과 일치 여부는 리서치 번들 비교 페이로드에 그대로 보존됩니다.
    """
    control_ok, control_norm = _normalize_oof_dates(control_dates)
    candidate_ok, candidate_norm = _normalize_oof_dates(candidate_dates)
    if not control_ok or not candidate_ok:
        return False, [], []
    return control_norm == candidate_norm, control_norm, candidate_norm


def _oof_date_mismatch_diagnostic(
    control_norm: list[str], candidate_norm: list[str]
) -> str:
    """표준화 날짜 목록의 결정적 불일치 진단 문자열을 반환합니다."""
    if control_norm == candidate_norm:
        return "oof_dates_identical"
    control_set = set(control_norm)
    candidate_set = set(candidate_norm)
    missing = sorted(control_set - candidate_set)
    additional = sorted(candidate_set - control_set)
    parts: list[str] = []
    if missing:
        parts.append(f"missing={missing[:5]}")
    if additional:
        parts.append(f"additional={additional[:5]}")
    return "oof_date_mismatch(" + ", ".join(parts) + ")"


def _resolve_research_selection_config(
    feature_selection_config: FeatureSelectionConfig | None,
) -> FeatureSelectionConfig:
    """리서치 후보 선택 설정을 결정합니다.

    호출자가 설정을 생략하면 ``permutation_fwer`` 리서치 기본값을, 명시적으로
    전달하면 그대로(레거시 ``fixed_cap`` 포함) 반환합니다.
    """
    if feature_selection_config is not None:
        return feature_selection_config
    return FeatureSelectionConfig(selection_rule="permutation_fwer")


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


def _rss_bytes() -> int:
    """현재 프로세스 RSS (psutil 미사용 시 0)."""
    if psutil is None:
        return 0
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - telemetry best-effort
        return 0


class _Lifecycle:
    """연구 실험 수명주기 이벤트를 단조 stage_id 로 방출합니다.

    이벤트는 append-only 이며 ``status`` 는 ``started``/``completed``/``failed``
    중 하나입니다. 실패는 진행 중 stage 에 대해 예외 진단을 포함한 ``failed``
    이벤트를 남긴 뒤 예외를 다시 던집니다. observer 는 비권위적입니다.
    """

    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._run_started = time.perf_counter()
        self._stage_id = 0

    def emit(
        self,
        stage: str,
        status: str,
        *,
        elapsed_seconds: float | None = None,
        fold: int | None = None,
        **details: Any,
    ) -> None:
        self._stage_id += 1
        payload: dict[str, Any] = {
            "status": status,
            "stage_id": self._stage_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_seconds": (
                round(elapsed_seconds, 4)
                if elapsed_seconds is not None
                else round(time.perf_counter() - self._run_started, 4)
            ),
            "current_rss_bytes": _rss_bytes(),
            "peak_rss_bytes": _rss_bytes(),
        }
        if fold is not None:
            payload["fold"] = int(fold)
        payload.update(details)
        _emit_progress(self._callback, stage, payload)

    def stage(
        self, stage: str, *, fold: int | None = None, **started_details: Any
    ) -> _StageEvent:
        """stage 시작/완료/실패 이벤트를 감싸는 context manager 를 반환합니다."""
        return _StageEvent(self, stage, fold=fold, started_details=started_details)


class _StageEvent:
    """수명주기 stage 를 감싸 started → completed/failed 이벤트를 방출합니다."""

    def __init__(
        self,
        lifecycle: _Lifecycle,
        stage: str,
        *,
        fold: int | None = None,
        started_details: Mapping[str, Any],
    ) -> None:
        self._lifecycle = lifecycle
        self._stage = stage
        self._fold = fold
        self._started = time.perf_counter()
        self._peak_rss = _rss_bytes()
        self._completed = False
        lifecycle.emit(stage, "started", fold=fold, **dict(started_details))

    def __enter__(self) -> _StageEvent:
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> Literal[False]:
        if exc is not None:
            self.failed(exc)
        else:
            self.completed()
        return False

    def _finalize(self, **details: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "elapsed_seconds": time.perf_counter() - self._started,
            "peak_rss_bytes": max(self._peak_rss, _rss_bytes()),
        }
        if self._fold is not None:
            payload["fold"] = self._fold
        payload.update(details)
        return payload

    def completed(self, **details: Any) -> None:
        if self._completed:
            return
        self._completed = True
        self._lifecycle.emit(self._stage, "completed", **self._finalize(**details))

    def failed(self, exc: BaseException, **details: Any) -> None:
        if self._completed:
            return
        self._completed = True
        self._lifecycle.emit(
            self._stage,
            "failed",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            **self._finalize(**details),
        )


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
    batch_observer: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """지문이 일치하면 캐시 warm read, 아니면 재구성 후 원자적으로 저장합니다.

    캐시는 panel parquet + metrics JSON 한 쌍으로만 warm 이며, 누락·지문 불일치·
    읽기 실패는 항상 cold 재구성으로 이어지고 ``cache_reason`` 에 원인을 남깁니다.
    저장은 임시 형제 파일에 먼저 쓰고 검증한 뒤 ``os.replace`` 로 원자 교체합니다.
    """
    cache_meta: dict[str, Any] = {
        "cache_state": "cold",
        "cache_fingerprint": fingerprint,
        "cache_reason": "no_cache_dir",
    }
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_path / f"history_features_{fingerprint}.parquet"
        metrics_path = cache_path / f"history_features_{fingerprint}.metrics.json"
        if parquet_path.is_file() and metrics_path.is_file():
            try:
                panel = pd.read_parquet(parquet_path)
                stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                stored_cache = stored_metrics.get("cache_metrics", {})
                if stored_cache.get("cache_fingerprint") != fingerprint:
                    raise ValueError(
                        f"stored fingerprint {stored_cache.get('cache_fingerprint')} "
                        f"does not match {fingerprint}"
                    )
                panel.attrs["history_feature_build_metrics"] = stored_metrics["build_metrics"]
                panel.attrs["history_feature_cache_metrics"] = stored_cache
                cache_meta = dict(stored_cache)
                cache_meta["cache_state"] = "warm"
                cache_meta["cache_reason"] = "warm_fingerprint_match"
                logger.info("history-feature cache warm read: %s", parquet_path.name)
                return panel, cache_meta
            except Exception as exc:  # noqa: BLE001 - corrupt cache must rebuild
                logger.warning("history-feature cache read failed, rebuilding: %s", exc)
                cache_meta["cache_reason"] = f"cache_unreadable:{type(exc).__name__}"
        else:
            missing = [
                path.name
                for path in (parquet_path, metrics_path)
                if not path.is_file()
            ]
            cache_meta["cache_reason"] = "incomplete_cache_pair:" + ",".join(missing)

    if price_history_path is not None:
        history_panel = build_causal_history_feature_panel_from_parquet(
            price_history_path,
            decision_keys,
            HistoricalFeatureConfig(),
            execution_config,
            batch_observer=batch_observer,
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
        "cache_reason": cache_meta["cache_reason"],
    }
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_path / f"history_features_{fingerprint}.parquet"
        metrics_path = cache_path / f"history_features_{fingerprint}.metrics.json"
        temp_parquet = Path(f"{parquet_path}.tmp")
        temp_metrics = Path(f"{metrics_path}.tmp")
        try:
            history_panel.to_parquet(temp_parquet, index=False)
            temp_metrics.write_text(
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
            _validate_cache_pair(temp_parquet, temp_metrics, fingerprint)
            os.replace(temp_parquet, parquet_path)
            os.replace(temp_metrics, metrics_path)
        finally:
            temp_parquet.unlink(missing_ok=True)
            temp_metrics.unlink(missing_ok=True)
        logger.info("history-feature cache written (atomic): %s", parquet_path.name)
    return history_panel, cache_metrics

def _validate_cache_pair(
    parquet_path: Path, metrics_path: Path, fingerprint: str
) -> None:
    """임시 캐시 쌍을 원자 교체 전에 검증합니다.

    실패하면 예외를 던져 출판을 중단합니다. 검증은 고정 카탈로그 컬럼 계약과
    저장된 지문 일치를 확인해, 손상된 쌍이 warm 캐시로 취급되는 것을 방지합니다.
    """
    panel = pd.read_parquet(parquet_path)
    out_cols = [
        "stock_code",
        "trade_date",
        *[str(entry["feature_name"]) for entry in HISTORICAL_CATALOGUE],
    ]
    if panel.columns.tolist() != out_cols:
        raise ValueError("cache panel columns do not match the fixed catalogue")
    stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if stored_metrics.get("cache_metrics", {}).get("cache_fingerprint") != fingerprint:
        raise ValueError("cache metrics fingerprint does not match")
    if "build_metrics" not in stored_metrics:
        raise ValueError("cache metrics missing build_metrics")


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
        "final_selection_provenance": "full_training_selection",
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
    availability_promotable: bool = True,
) -> dict[str, Any]:
    """확정 게이트: 가용성 증명 + 동일 OOF 날짜 + 양수 scheduled mean + 대조군보다
    엄격히 높은 mean + PF>1 + MDD 엄격 감소 + 안정성."""
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
    if not availability_promotable:
        rejected_reasons.append("availability_manifest_non_promotable")
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
        "availability_manifest_promotable": bool(availability_promotable),
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
    observation_time_col: str | None = None,
    decision_time_col: str | None = None,
    execution_time_col: str | None = None,
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
        observation_time_col / decision_time_col / execution_time_col: 원산지
            증명 타임스탬프 컬럼명. 모두 주어지면 결정 시점 가용성 증명이 강제되고
            승격 게이트의 ``availability_manifest_promotable`` 로 반영됩니다.
            주어지지 않으면 (레거시) discovery-only 승격 불가 상태로 기록됩니다.

    Returns:
        dict: ``contract``, ``build_metrics``, ``control``, ``candidate``,
        ``comparison``, ``promotion``, ``candidate_bundle_path``.
    """
    if price_history is None and price_history_path is None:
        raise ValueError("either price_history or price_history_path must be provided")
    if mode not in _AVAILABLE_MODES:
        raise ValueError(f"mode must be one of {_AVAILABLE_MODES}, got {mode!r}")
    lifecycle = _Lifecycle(progress_callback)
    config = _resolve_research_selection_config(feature_selection_config)
    _x, _targets, cat_features, processed = build_ml_dataset(
        trade_log_df,
        theme_df,
        feature_set=feature_set,
        panel_mode=panel_mode,
        observation_time_col=observation_time_col,
        decision_time_col=decision_time_col,
        execution_time_col=execution_time_col,
        provenance_mode=mode,
    )
    base_feature_candidates = [col for col in _x.columns if col not in cat_features]
    base_feature_cols = []
    base_feature_exclusions: dict[str, str] = {}
    for column in base_feature_candidates:
        values = pd.to_numeric(_x[column], errors="coerce").to_numpy(dtype=np.float64)
        missing_rate = float((~np.isfinite(values)).mean())
        if missing_rate > config.missing_rate_threshold:
            base_feature_exclusions[column] = "source_missing_rate"
        else:
            base_feature_cols.append(column)
    if not base_feature_cols:
        raise ValueError("all baseline features failed the source missing-rate gate")
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
    with lifecycle.stage(
        "history_panel_build",
        source="parquet" if price_history_path is not None else "dataframe",
        evaluation_cutoff=str(evaluation_cutoff),
        fingerprint=fingerprint,
    ) as panel_stage:
        history_panel, cache_metrics = _load_or_build_history_panel(
            price_history,
            price_history_path,
            decision_keys,
            group_col,
            execution_config,
            cache_dir,
            fingerprint,
            batch_observer=progress_callback,
        )
    build_metrics = dict(history_panel.attrs.get("history_feature_build_metrics", {}))
    build_metrics["cache_state"] = cache_metrics["cache_state"]
    build_metrics["cache_fingerprint"] = fingerprint
    build_metrics["cache_reason"] = cache_metrics.get("cache_reason")
    panel_stage.completed(
        cache_state=cache_metrics["cache_state"],
        cache_reason=cache_metrics.get("cache_reason"),
        batch_count=build_metrics.get("batch_count"),
        decision_key_rows=build_metrics.get("decision_key_rows"),
        output_rows=build_metrics.get("output_rows"),
    )
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

    with lifecycle.stage("control", feature_cols=len(base_feature_cols)) as control_stage:
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
    control_stage.completed(n_folds=n_splits)
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
        with lifecycle.stage(
            "selection_plan", n_splits=n_splits, candidate_features=len(candidate_feature_cols)
        ) as plan_stage:
            plans = build_fold_feature_plans(
                joined,
                candidate_feature_cols,
                target_col,
                group_col,
                config,
                n_splits=n_splits,
                purge_gap=purge_gap,
            )
        plan_stage.completed(fold_count=len(plans))
        for _plan in plans:
            lifecycle.emit(
                "selection_plan",
                "completed",
                fold=_plan.fold,
                selected_features=len(_plan.selected_features),
                data_cutoff=_plan.data_cutoff,
            )
        with lifecycle.stage("final_selection") as final_stage:
            final_selection = select_features(
                joined, candidate_feature_cols, target_col, config, group_col=group_col
            )
        final_stage.completed(selected_features=len(final_selection.selected_features))
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

    with lifecycle.stage("candidate", feature_cols=len(candidate_feature_cols)) as candidate_stage:
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
    candidate_stage.completed(n_folds=n_splits)
    _emit_progress(
        progress_callback,
        "candidate_complete",
        {"metrics": dict(candidate["aggregate"]["candidate"])},
    )

    control_agg = dict(control["aggregate"]["candidate"])
    candidate_agg = dict(candidate["aggregate"]["candidate"])

    identical_oof_dates, control_oof_dates, candidate_oof_dates = (
        validate_research_oof_alignment(
            control["baseline_oof_dates"], candidate["candidate_oof_dates"]
        )
    )

    stability = (
        diagnostics.get("stability", {})
        if diagnostics is not None
        else {"gate_passed": True, "reason": "no_feature_selection"}
    )
    availability_provenance = processed.attrs.get(
        "availability_provenance", {"promotable": False, "mode": "discovery"}
    )
    availability_promotable = bool(availability_provenance.get("promotable", False))
    promotion = _promotion_gate(
        control,
        candidate,
        identical_oof_dates,
        stability,
        availability_promotable=availability_promotable,
    )

    comparison = {
        "identical_oof_dates": identical_oof_dates,
        "control_oof_dates": control_oof_dates,
        "candidate_oof_dates": candidate_oof_dates,
        "oof_date_mismatch": _oof_date_mismatch_diagnostic(
            control_oof_dates, candidate_oof_dates
        ),
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
        "baseline_feature_cols": base_feature_cols,
        "baseline_feature_exclusions": base_feature_exclusions,
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
        "availability_provenance": availability_provenance,
        "control_metrics": control_agg,
        "candidate_metrics": candidate_agg,
        "promotion": promotion,
        "comparison": comparison,
    }
    save_dir = f"{export_dir}/{HISTORICAL_CATALOGUE_VERSION}/{bundle['training_cutoff'][:10]}"
    with lifecycle.stage("export", save_dir=save_dir) as export_stage:
        saved_path = save_model_artifacts(bundle, save_dir)
    export_stage.completed(
        candidate_bundle_path=saved_path, final_feature_count=len(final_features)
    )
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
            "baseline_feature_count": len(base_feature_cols),
            "baseline_feature_exclusions": base_feature_exclusions,
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
        "availability_provenance": availability_provenance,
        "candidate_bundle_path": saved_path,
    }
