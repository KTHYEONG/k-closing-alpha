import json
import os
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.data.gsheet_loader import load_and_combine_sheets
from src.processing.preprocessor import preprocess_data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

GOOGLE_SHEET_NAME = "Stock"
WORKSHEET_NAMES = ["Trade", "Trade2"]
TRAIN_RATIO = 0.8
USE_GPU = False
N_TRIALS = 60
BASE_SPLITS = 4

GROWTH_THRESHOLD = 5000
EXPANDED_THRESHOLD = 10000

print("\n[CatBoost tuner] Loading data from Google Sheets...")
df_raw = load_and_combine_sheets(GOOGLE_SHEET_NAME, WORKSHEET_NAMES)

X, y, cat_features, df_processed = preprocess_data(df_raw)

if "매수일자" in df_processed.columns:
    order_idx = pd.to_datetime(df_processed["매수일자"]).sort_values().index
else:
    order_idx = df_processed.index

X = X.loc[order_idx].reset_index(drop=True)
y = y.loc[order_idx].reset_index(drop=True)
df_processed = df_processed.loc[order_idx].reset_index(drop=True)

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


def objective_cat(trial: optuna.Trial) -> float:
    if dataset_regime == "growing" or dataset_regime == "expanded":
        iteration_range = (900, 1800)
        depth_range = (6, 10)
        l2_leaf_reg_range = (2, 15)
        min_data_in_leaf_range = (20, 120)
        bagging_temp_high = 1.5
    else:
        iteration_range = (500, 1400)
        depth_range = (4, 8)
        l2_leaf_reg_range = (1, 10)
        min_data_in_leaf_range = (10, 80)
        bagging_temp_high = 1.0

    bootstrap_candidates = ["Bayesian", "Bernoulli"]
    if USE_GPU:
        bootstrap_candidates.append("Poisson")

    bootstrap_type = trial.suggest_categorical("bootstrap_type", bootstrap_candidates)
    subsample = None
    if bootstrap_type in ["Bernoulli", "Poisson"]:
        subsample = trial.suggest_float("subsample", 0.6, 1.0)

    bagging_temperature = None
    if bootstrap_type == "Bayesian":
        bagging_temperature = trial.suggest_float(
            "bagging_temperature", 0.0, bagging_temp_high
        )

    use_rsm = not USE_GPU
    params = {
        "iterations": trial.suggest_int(
            "iterations", iteration_range[0], iteration_range[1]
        ),
        "depth": trial.suggest_int("depth", depth_range[0], depth_range[1]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", l2_leaf_reg_range[0], l2_leaf_reg_range[1]
        ),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 8
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", min_data_in_leaf_range[0], min_data_in_leaf_range[1]
        ),
        "bootstrap_type": bootstrap_type,
        "auto_class_weights": "Balanced",
        "cat_features": cat_features,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "task_type": "GPU" if USE_GPU else "CPU",
        "verbose": False,
        "random_seed": 42,
        "allow_writing_files": False,
    }
    if subsample is not None:
        params["subsample"] = subsample
    if bagging_temperature is not None:
        params["bagging_temperature"] = bagging_temperature
    if use_rsm:
        params["rsm"] = trial.suggest_float("rsm", 0.6, 1.0)

    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_t,
            y_t,
            eval_set=[(X_v, y_v)],
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=False,
        )

        preds = model.predict_proba(X_v)[:, 1]
        try:
            fold_auc = roc_auc_score(y_v, preds)
            scores.append(fold_auc)
        except ValueError:
            continue

        trial.report(float(np.mean(scores)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores)) if scores else 0.0


start_time = time.time()
sampler = TPESampler(seed=42)
print(f"\n[CatBoost tuner] Starting Optuna search ({N_TRIALS} trials)...")
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective_cat, n_trials=N_TRIALS)

print(f"Best CV ROC-AUC: {study.best_value:.4f}")

best_params = {
    **study.best_params,
    "auto_class_weights": "Balanced",
    "cat_features": cat_features,
    "verbose": False,
    "random_seed": 42,
    "allow_writing_files": False,
    "task_type": "GPU" if USE_GPU else "CPU",
    "loss_function": "Logloss",
}

print("\n[CatBoost tuner] Training with best parameters on training window...")
final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_holdout, y_holdout)],
    use_best_model=True,
    verbose=False,
)

probs = final_model.predict_proba(X_holdout)[:, 1]
preds = (probs >= 0.5).astype(int)
try:
    holdout_auc = roc_auc_score(y_holdout, probs)
except ValueError:
    holdout_auc = float("nan")

metrics = {
    "holdout_auc": holdout_auc,
    "holdout_accuracy": accuracy_score(y_holdout, preds),
    "holdout_precision": precision_score(y_holdout, preds, zero_division=0),
    "holdout_recall": recall_score(y_holdout, preds, zero_division=0),
    "holdout_f1": f1_score(y_holdout, preds, zero_division=0),
    "cv_best_auc": study.best_value,
    "dataset_size": dataset_size,
    "dataset_regime": dataset_regime,
    "train_size": len(X_train),
    "holdout_size": len(X_holdout),
}

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

deployment_model = CatBoostClassifier(**best_params)
deployment_model.fit(X, y, verbose=False)

model_path = os.path.join(models_dir, "best_stock_cl.cbm")
deployment_model.save_model(model_path)

report_path = os.path.join(models_dir, "best_stock_cl_metrics.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(
        {"metrics": metrics, "best_params": best_params},
        f,
        ensure_ascii=False,
        indent=2,
    )

elapsed_min = (time.time() - start_time) / 60

print(f"\nHoldout AUC: {metrics['holdout_auc']:.4f}")
print(f"Saved model to: {model_path}")
print(f"Saved metrics to: {report_path}")
print(f"Elapsed minutes: {elapsed_min:.2f}")
