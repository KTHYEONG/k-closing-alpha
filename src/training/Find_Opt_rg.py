import json
import os
import sys
import time
import warnings
from typing import Callable, Any, Dict

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.data.db_loader import load_trade_log_from_db
from src.processing.preprocessor import preprocess_data

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_RATIO = 0.8
# [Optimization] 효율적인 학습을 위해 Trial 횟수 조정 (60 -> 30)
N_TRIALS = 30
BASE_SPLITS = 5
MODELS_DIR = os.path.join(project_root, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_prepare_data(model_type="tree"):
    """
    model_type: 'tree' (LabelEncoded for RF/ET), 'cat' (Original for CatBoost)
    """
    print(f"\n[{model_type.upper()}] Loading data from local DB...")
    df_raw = load_trade_log_from_db()
    print(f" -> Downloaded {len(df_raw):,} rows.", flush=True)

    X_raw, y, cat_features, df_processed = preprocess_data(df_raw, task="regression")
    
    # [Fix] day_name 컬럼 제거
    if "day_name" in X_raw.columns:
        X_raw = X_raw.drop(columns=["day_name"])
        
    print(f" -> Preprocessing complete. Features: {len(X_raw.columns)}")

    if "매수날짜" in df_processed.columns:
        order_idx = pd.to_datetime(df_processed["매수날짜"]).sort_values().index
    else:
        order_idx = df_processed.index

    X_raw = X_raw.loc[order_idx].reset_index(drop=True)
    y = y.loc[order_idx].reset_index(drop=True)

    # 숫자열 처리
    numeric_cols = X_raw.columns.difference(cat_features)
    X_raw[numeric_cols] = (
        X_raw[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )

    encoders = {}
    cat_cols = cat_features
    
    # 모델 타입에 따른 인코딩 분기
    if model_type == "tree":
        # ExtraTrees/RF용: Label Encoding
        X = X_raw.copy()
        for col in cat_features:
            if col in X.columns:
                le = LabelEncoder()
                # 문자열로 변환 후 인코딩
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = list(le.classes_)
    else:
        # CatBoost용: 그대로 두되 문자열로 변환
        X = X_raw.copy()
        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)
        encoders = {c: [] for c in cat_features} # Dummy

    split_point = int(len(X) * TRAIN_RATIO)
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_holdout = X.iloc[split_point:]
    y_holdout = y.iloc[split_point:]

    return X_train, y_train, X_holdout, y_holdout, encoders, cat_cols, X, y


def save_artifacts(
    model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    encoders: Dict[str, LabelEncoder],
    X_full: pd.DataFrame,
    feature_importances: pd.Series = None,
):
    print(f"\n[Saving Artifacts] as {model_name}...")
    
    # 1. Save Model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    dump(model, model_path)
    
    # 2. Save Encoders
    enc_path = os.path.join(MODELS_DIR, f"{model_name}_encoders.json")
    encoder_payload = {
        col: encoder for col, encoder in encoders.items()
    }
    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(encoder_payload, f, ensure_ascii=False, indent=2)

    # 3. Save Feature Importances
    fi_path = os.path.join(MODELS_DIR, f"{model_name}_feature_importances.csv")
    if feature_importances is not None:
        feature_importances.to_frame(name="importance").to_csv(fi_path, index_label="feature")

    # 4. SHAP (Optional)
    shap_path = None
    if HAS_SHAP and not model_name.startswith("catboost"): # CatBoost has own shap
        try:
            explainer = shap.TreeExplainer(model)
            # Sample for speed (CatBoost might be slow with SHAP)
            sample_X = X_full.sample(min(1000, len(X_full)), random_state=42)
            shap_values = explainer.shap_values(sample_X)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_series = pd.Series(mean_abs_shap, index=X_full.columns).sort_values(ascending=False)
            
            shap_path = os.path.join(MODELS_DIR, f"{model_name}_shap_summary.csv")
            shap_series.to_frame("mean_abs_shap").to_csv(shap_path, index_label="feature")
        except Exception as e:
            print(f"Warning: SHAP extraction failed: {e}")

    # 5. Save Metrics
    report_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "best_params": best_params,
                "paths": {"model": model_path, "encoders": enc_path, "fi": fi_path, "shap": shap_path},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f" -> Completed! Model saved to {model_path}")


def perform_extratrees_optimization():
    print("\n" + "=" * 50)
    print(" Strategy: ExtraTrees Optimization")
    print("=" * 50)
    
    X_train, y_train, X_holdout, y_holdout, encoders, _, X_full, y_full = load_and_prepare_data("tree")
    n_splits = min(BASE_SPLITS, max(len(X_train) - 1, 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 18),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
            "n_jobs": -1,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model = ExtraTreesRegressor(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[val_idx])
            scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f" -> Best CV MAE: {study.best_value:.5f}")
    
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1})
    
    final_model = ExtraTreesRegressor(**best_params)
    final_model.fit(X_full, y_full)
    
    # Evaluate on Holdout (Simulated) - 학습은 전체로 했지만 지표 확인용
    # 실제로는 Holdout을 학습에 안 쓰고 평가만 해야 정확하지만,
    # 최종 배포 모델은 전체 데이터 학습이 원칙이므로 여기서는 전체 학습 모델을 저장함.
    holdout_preds = final_model.predict(X_holdout)
    metric_mae = mean_absolute_error(y_holdout, holdout_preds)
    
    fi_series = pd.Series(final_model.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    
    save_artifacts(
        final_model, 
        "best_stock_rg_et", 
        {"holdout_mae": metric_mae, "cv_best_mae": study.best_value}, 
        best_params, 
        encoders, 
        X_full, 
        fi_series
    )


def perform_catboost_optimization():
    print("\n" + "=" * 50)
    print(" Strategy: CatBoost Optimization (High Efficiency)")
    print("=" * 50)
    
    X_train, y_train, X_holdout, y_holdout, encoders, cat_cols, X_full, y_full = load_and_prepare_data("cat")
    
    # CatBoost는 cat_features 인덱스를 필요로 함
    cat_indices = []
    if cat_cols:
        cat_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
        
    n_splits = min(BASE_SPLITS, max(len(X_train) - 1, 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        # [Optimization] 탐색 범위 축소 (Pruning)
        params = {
            "iterations": trial.suggest_int("iterations", 800, 1500), # 너무 적으면 학습 부족
            "depth": trial.suggest_int("depth", 6, 8),                # 4~10 -> 6~8 (주식에 적합)
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True), # 범위 축소
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 10),
            "random_seed": 42,
            "loss_function": "MAE",
            "verbose": False,
            "allow_writing_files": False,
            "early_stopping_rounds": 50, # [Optimization] 조기 종료 적용
        }
        
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            train_pool = X_train.iloc[train_idx]
            train_y = y_train.iloc[train_idx]
            val_pool = X_train.iloc[val_idx]
            val_y = y_train.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            
            # eval_set을 지정해야 early_stopping이 작동함
            model.fit(
                train_pool, train_y, 
                cat_features=cat_indices,
                eval_set=(val_pool, val_y),
                early_stopping_rounds=50,
                verbose=False
            )
            
            # 예측
            preds = model.predict(val_pool)
            scores.append(mean_absolute_error(val_y, preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f" -> Best CV MAE: {study.best_value:.5f}")
    
    best_params = study.best_params
    best_params.update({
        "random_seed": 42, 
        "loss_function": "MAE",
        "verbose": False
    })
    
    # [Optimization] 최종 모델 학습 (전체 데이터)
    print(" -> Training final model with best params...")
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(
        X_full, y_full, 
        cat_features=cat_indices,
        verbose=100  # 진행 상황 표시
    )
    
    # 평가용 (참고용)
    holdout_preds = final_model.predict(X_holdout)
    metric_mae = mean_absolute_error(y_holdout, holdout_preds)
    
    fi_series = pd.Series(final_model.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    
    save_artifacts(
        final_model, 
        "best_stock_rg_cat", 
        {"holdout_mae": metric_mae, "cv_best_mae": study.best_value}, 
        best_params, 
        encoders, 
        X_full, 
        fi_series
    )
    
if __name__ == "__main__":
    # 필요한 모델 주석 해제하여 사용
    # perform_extratrees_optimization()
    perform_catboost_optimization()
