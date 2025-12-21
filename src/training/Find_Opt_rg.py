import json
import os
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor
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
N_TRIALS = 60
BASE_SPLITS = 5

GROWTH_THRESHOLD = 5000
EXPANDED_THRESHOLD = 10000

print("\n[ExtraTrees tuner] Loading data from local DB...")
df_raw = load_trade_log_from_db()

print(f" -> Downloaded {len(df_raw):,} rows from sheets.", flush=True)

X_raw, y, cat_features, df_processed = preprocess_data(df_raw, task="regression")
print(f" -> Preprocessing complete. Features: {len(X_raw.columns)}")

if "매수날짜" in df_processed.columns:
    order_idx = pd.to_datetime(df_processed["매수날짜"]).sort_values().index
else:
    order_idx = df_processed.index

X_raw = X_raw.loc[order_idx].reset_index(drop=True)
y = y.loc[order_idx].reset_index(drop=True)

# 숫자열 확인
numeric_cols = X_raw.columns.difference(cat_features)
X_raw[numeric_cols] = (
    X_raw[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
)

X = X_raw.copy()
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

dataset_size = len(y)
if dataset_size < GROWTH_THRESHOLD:
    dataset_regime = "small"
elif dataset_size < EXPANDED_THRESHOLD:
    dataset_regime = "growing"
else:
    dataset_regime = "expanded"

split_point = max(1, int(len(X) * TRAIN_RATIO))
split_point = min(split_point, len(X) - 1)

X_train = X.iloc[:split_point].reset_index(drop=True)
y_train = y.iloc[:split_point].reset_index(drop=True)
X_holdout = X.iloc[split_point:].reset_index(drop=True)
y_holdout = y.iloc[split_point:].reset_index(drop=True)

n_splits = min(BASE_SPLITS, max(len(X_train) - 1, 2))
tscv = TimeSeriesSplit(n_splits=n_splits)

print(
    f"Dataset size: {dataset_size} | Regime: {dataset_regime} | "
    f"Train: {len(X_train)} | Holdout: {len(X_holdout)} | CV folds: {n_splits}"
)


def objective_extra_trees(trial: optuna.Trial) -> float:
    print(f"[Optuna] Trial {trial.number} started.", flush=True)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 18),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1,
    }

    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        model = ExtraTreesRegressor(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        mae = mean_absolute_error(y_train.iloc[val_idx], preds)
        scores.append(mae)
        print(
            f"  [Trial {trial.number}] Fold {fold_idx + 1}/{n_splits} MAE: {mae:.6f}",
            flush=True,
        )

        trial.report(float(np.mean(scores)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores)) if scores else float("inf")


start_time = time.time()
sampler = TPESampler(seed=42)
print(f"\n[ExtraTrees tuner] Starting Optuna search ({N_TRIALS} trials)...")
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective_extra_trees, n_trials=N_TRIALS)

print(f"Best CV MAE: {study.best_value:.6f}")

best_params = {
    **study.best_params,
    "random_state": 42,
    "n_jobs": -1,
}

print("\n[ExtraTrees tuner] Training with best parameters on training window...")
final_model = ExtraTreesRegressor(**best_params)
final_model.fit(X_train, y_train)
print(" -> Finished training. Evaluating on holdout set...", flush=True)

holdout_preds = final_model.predict(X_holdout)
holdout_mae = mean_absolute_error(y_holdout, holdout_preds)
# 일부 sklearn 버전은 squared 인자를 지원하지 않아 수동으로 RMSE를 계산
holdout_rmse = mean_squared_error(y_holdout, holdout_preds) ** 0.5
holdout_r2 = r2_score(y_holdout, holdout_preds)
holdout_median_ae = median_absolute_error(y_holdout, holdout_preds)
holdout_explained_var = explained_variance_score(y_holdout, holdout_preds)
epsilon = 1e-6
holdout_mape = float(
    np.mean(
        np.abs(
            (y_holdout.values - holdout_preds)
            / np.clip(np.abs(y_holdout.values), epsilon, None)
        )
    )
    * 100
)

metrics = {
    "holdout_mae": holdout_mae,
    "holdout_mape": holdout_mape,
    "holdout_median_ae": holdout_median_ae,
    "holdout_explained_variance": holdout_explained_var,
    "holdout_rmse": holdout_rmse,
    "holdout_r2": holdout_r2,
    "cv_best_mae": study.best_value,
    "dataset_size": dataset_size,
    "dataset_regime": dataset_regime,
    "train_size": len(X_train),
    "holdout_size": len(X_holdout),
}

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

deployment_model = ExtraTreesRegressor(**best_params)
deployment_model.fit(X, y)
print(" -> Deployment model trained on full dataset.", flush=True)

model_path = os.path.join(models_dir, "best_stock_rg.joblib")
dump(deployment_model, model_path)
enc_path = os.path.join(models_dir, "best_stock_rg_label_encoders.json")
encoder_payload = {
    col: encoder.classes_.tolist() for col, encoder in label_encoders.items()
}
with open(enc_path, "w", encoding="utf-8") as f:
    json.dump(encoder_payload, f, ensure_ascii=False, indent=2)
print(f" -> Saved label encoders to: {enc_path}", flush=True)

feature_importances = pd.Series(
    deployment_model.feature_importances_, index=X.columns
).sort_values(ascending=False)
fi_path = os.path.join(models_dir, "best_stock_rg_feature_importances.csv")
feature_importances.to_frame(name="importance").to_csv(fi_path, index_label="feature")
print(f" -> Saved feature importances to: {fi_path}", flush=True)

# --- SHAP summary ---
shap_path = None
if HAS_SHAP:
    try:
        explainer = shap.TreeExplainer(deployment_model)
        sample_size = min(1000, len(X))
        shap_sample = X.sample(n=sample_size, random_state=42) if sample_size > 0 else X
        shap_values = explainer.shap_values(shap_sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_series = pd.Series(mean_abs_shap, index=shap_sample.columns).sort_values(
            ascending=False
        )
        shap_path = os.path.join(models_dir, "best_stock_rg_shap_summary.csv")
        shap_series.to_frame(name="mean_abs_shap").to_csv(
            shap_path, index_label="feature"
        )
        print(f" -> Saved SHAP summary to: {shap_path}", flush=True)
    except Exception as exc:
        print(f"Warning: SHAP export failed: {exc}", flush=True)
else:
    print(" -> SHAP not installed; skipping SHAP summary export.", flush=True)

report_path = os.path.join(models_dir, "best_stock_rg_metrics.json")
with open(report_path, "w", encoding="utf-8") as f:
    payload = {
        "metrics": metrics,
        "best_params": best_params,
        "artifacts": {
            "model": model_path,
            "metrics_report": report_path,
            "feature_importances": fi_path,
            "shap_summary": shap_path,
            "label_encoders": enc_path,
        },
    }
    json.dump(payload, f, ensure_ascii=False, indent=2)

elapsed_min = (time.time() - start_time) / 60

print(f"\nHoldout MAE: {metrics['holdout_mae']:.6f}")
print(f"Saved model to: {model_path}")
print(f"Saved metrics to: {report_path}")
print(f"Elapsed minutes: {elapsed_min:.2f}")
