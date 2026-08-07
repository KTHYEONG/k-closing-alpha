"""모델 번들 검증/재학습 경계 서비스."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import pandas as pd

from src import settings
from src.ml.feature_selection import FeatureSelectionConfig
from src.ml.model_pipeline import _calibrate_oof_policy
from src.ml.single_stock_policy import SingleStockPolicy
from src.ml.sizing_engine import (
    _CLOSE_MORNING_RERANKER_CONFIG,
    _train_inline_bundle,
    save_model_artifacts,
)
from src.processing.preprocessor import build_ml_dataset
from src.utils.display import Colors

logger = logging.getLogger(__name__)


def _load_single_stock_policy(models_bundle: dict[str, Any]) -> SingleStockPolicy | None:
    """모델 번들에 영속화된 ``SingleStockPolicy`` 상태를 반환합니다.

    유효한 정책 상태가 없으면 ``None`` 을 반환하며, 호출부가 조용한 Top-N 폴백
    대신 명시적 ``ABSTAIN``(``missing_validated_policy``)을 산출하게 합니다.
    """
    raw = models_bundle.get("single_stock_policy")
    if raw is None:
        return None
    if isinstance(raw, SingleStockPolicy):
        return raw
    if isinstance(raw, dict):
        return SingleStockPolicy(**raw)
    logger.info(
        f"{Colors.YELLOW}[Warning] 인식할 수 없는 single_stock_policy 상태입니다. "
        f"ABSTAIN(missing_validated_policy) 으로 결정합니다.{Colors.RESET}"
    )
    return None


# 모델 번들 검증 상수: 단위 테스트 픽스처가 생성하는 더미 피처 패턴 및
# 실데이터 학습 번들이 반드시 포함해야 하는 모델 키 목록.
_DUMMY_FEATURE_PATTERN = re.compile(r"^f\d+$")
_MODEL_BUNDLE_KEYS = ("rank_model", "quantile_models", "calibrators")

# research 후보 피처셋: 활성 아티팩트와 분리된 버전화 하위 디렉터리에 저장합니다.
# 검증된 champion close_morning61 이 기본 후보이며, 승격 전까지 활성 아티팩트를
# 덮어쓰지 않도록 cutoff 로 버전화된 후보 경로를 사용합니다.
_CANDIDATE_FEATURE_SET = "close_morning61"
_RESEARCH_CANDIDATE_FEATURE_SETS: frozenset[str] = frozenset(
    {"close_morning61", "causal_expanded_v1"}
)


def _candidate_export_dir(
    export_dir: str, feature_set: str, bundle: dict[str, Any]
) -> str:
    """후보 피처셋 번들은 버전화된 하위 디렉터리에 저장할 경로를 반환합니다.

    활성 아티팩트는 ``export_dir`` 루트의 ``sizing_pipeline_bundle.joblib`` 이므로,
    후보(``close_morning61``, ``causal_expanded_v1``)는 훈련 데이터 cutoff 날짜
    (YYYY-MM-DD)로 버전화된 별도 디렉터리에 기록해 활성 아티팩트를 덮어쓰지
    않습니다. 다른 feature_set(활성 아티팩트 재학습)은 기존대로 루트 경로를
    반환합니다.
    """
    if feature_set not in _RESEARCH_CANDIDATE_FEATURE_SETS:
        return export_dir
    version = str(bundle.get("training_cutoff", ""))[:10] or "candidate"
    return os.path.join(export_dir, f"{feature_set}_{version}")


def _is_dummy_feature_cols(feature_cols: list[str]) -> bool:
    """더미 테스트 피처(예: ['f1', 'f2', 'f3']) 여부를 결정적으로 판별합니다."""
    return any(_DUMMY_FEATURE_PATTERN.match(col) for col in feature_cols)


def _is_valid_real_bundle(models_bundle: dict[str, Any]) -> bool:
    """저장된 번들이 실데이터 학습 산출물인지 검증합니다.

    실데이터 번들은 비어있지 않은 수치 피처 목록과 rank/quantile/calibrator
    모델 키를 모두 포함해야 합니다. 더미 피처만으로 구성된 번들은 무효로
    간주합니다.
    """
    feature_cols = list(models_bundle.get("feature_cols", []))
    if not feature_cols or _is_dummy_feature_cols(feature_cols):
        return False
    return all(key in models_bundle for key in _MODEL_BUNDLE_KEYS)


def train_and_save_real_model_bundle(
    export_dir: str = "artifacts/models",
    trade_log_path: str | os.PathLike[str] | None = None,
    theme_path: str | os.PathLike[str] | None = None,
    feature_set: str = "close_morning61",
    panel_mode: str = "scenario_action",
    price_history_path: str | os.PathLike[str] | None = None,
    feature_selection_config: FeatureSelectionConfig | None = None,
) -> dict[str, Any]:
    """``trade_log.parquet`` 실데이터로 표준 ML 번들을 학습·저장한 뒤 반환합니다.

    기본값은 검증된 champion ``close_morning61`` 피처셋 + ``scenario_action``
    패널 모드입니다. 훈련 절차:

    1. ``build_ml_dataset`` 로 ``close_morning61`` 시나리오 행동 패널을 구성합니다.
    2. ``run_model_pipeline`` 으로 동일 수치 피처 + 행동 메타데이터로 purged OOF
       예측과 ``SingleStockPolicy`` 를 생성합니다.
    3. ``_train_inline_bundle`` 로 전체 이력 최종 추론 모델을 학습합니다.
    4. ``single_stock_policy.model_dump()`` 와 함께 보정 cutoff, 정책 버전,
       피처셋 이름, compact OOF 정책 지표를 번들에 영속화합니다.
       ``close_morning61 + scenario_action`` 은 reranker 결정 스코어
       (``decision_score``) OOF 정책을 사용하며 ``oof_score_col`` 와
       ``daily_score_col`` 모두 ``"decision_score"`` 로 기록하고
       ``decision_score_config`` 를 영속화합니다. 그 외 번들은 기존
       ``pred``/``rank_score`` 매핑을 유지합니다.
    5. 버전화된 후보 아티팩트만 저장하며 활성 아티팩트를 자동으로 대체하지
       않습니다. 정책 상태가 없으면 ``ABSTAIN(missing_validated_policy)`` 로
       명시 처리되고 Top-N 폴백은 없습니다.

    ``build_ml_dataset`` 산출 feature_cols 에서 범주형 문자열 컬럼
    (``market_type``, ``theme_sector``, ``chart_analysis``)을 제외한 수치
    피처만 LightGBM 학습에 사용해 Booster 구성 오류를 방지합니다.
    """
    trade_log_path = str(trade_log_path or settings.TRADE_LOG_PARQUET_PATH)
    theme_path = str(theme_path or settings.THEME_PARQUET_PATH)
    trade_log_df = pd.read_parquet(trade_log_path)
    theme_df = pd.read_parquet(theme_path) if os.path.exists(theme_path) else None
    price_history_df = None
    if price_history_path is not None:
        price_history_path = str(price_history_path)
        price_history_df = (
            pd.read_parquet(price_history_path) if os.path.exists(price_history_path) else None
        )
    X, targets, cat_features, processed = build_ml_dataset(
        trade_log_df,
        theme_df,
        feature_set=feature_set,
        panel_mode=panel_mode,
        price_history_df=price_history_df,
    )
    feature_cols = [col for col in X.columns if col not in cat_features]
    target_col = "target_return"
    group_col = "trade_date"
    reranker = feature_set == "close_morning61" and panel_mode == "scenario_action"
    policy, policy_metadata = _calibrate_oof_policy(
        processed,
        feature_cols,
        target_col,
        group_col,
        n_splits=5,
        purge_gap=1,
        reranker=reranker,
        feature_selection_config=feature_selection_config,
    )
    bundle = _train_inline_bundle(
        processed[[*feature_cols, target_col, group_col]],
        feature_cols,
        target_col,
        group_col,
        feature_selection_config=feature_selection_config,
        catalog_metadata=(
            processed.attrs.get("feature_manifest") if feature_set == "causal_expanded_v1" else None
        ),
    )
    bundle["feature_set"] = feature_set
    bundle["panel_mode"] = panel_mode
    bundle["single_stock_policy"] = policy.model_dump() if policy is not None else None
    bundle["policy_metadata"] = policy_metadata
    if reranker:
        bundle["decision_score_config"] = dict(_CLOSE_MORNING_RERANKER_CONFIG)
        bundle["oof_score_col"] = "decision_score"
        bundle["daily_score_col"] = "decision_score"
    else:
        bundle["oof_score_col"] = "pred"
        bundle["daily_score_col"] = "rank_score"
    save_dir = _candidate_export_dir(export_dir, feature_set, bundle)
    save_model_artifacts(bundle, save_dir)
    logger.info(
        f"{Colors.GREEN}실데이터 모델 번들 재학습·저장 완료: "
        f"feature_set={feature_set} panel_mode={panel_mode} "
        f"feature_cols={len(feature_cols)}개 "
        f"policy={'serialized' if policy is not None else 'absent(ABSTAIN)'} "
        f"(save_dir={save_dir}){Colors.RESET}"
    )
    return bundle


def ensure_valid_model_bundle(models_bundle: dict[str, Any]) -> dict[str, Any]:
    """더미/무효 모델 번들을 감지하면 실데이터로 재학습한 번들로 대체합니다."""
    if _is_valid_real_bundle(models_bundle):
        return models_bundle
    logger.info(
        f"{Colors.YELLOW}[Warning] 모델 번들이 더미 테스트 피처이거나 무효합니다. "
        f"trade_log.parquet 로 실데이터 재학습을 시작합니다.{Colors.RESET}"
    )
    return train_and_save_real_model_bundle()
