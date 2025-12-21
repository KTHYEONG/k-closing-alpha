import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.data.db_loader import load_trade_log_from_db
from src.processing.preprocessor import preprocess_data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

GROWTH_THRESHOLD = 5000
EXPANDED_THRESHOLD = 10000

HOLDOUT_RATIOS = {
    "small": 0.20,
    "growing": 0.18,
    "expanded": 0.15,
}
MIN_HOLDOUT = 200

print("\n[Step 1/5] Loading data from local DB...")
df_raw = load_trade_log_from_db()

X_cat, y, cat_features, df_processed = preprocess_data(df_raw.copy())

if "매수일자" in df_processed.columns:
    order_idx = pd.to_datetime(df_processed["매수일자"]).sort_values().index
else:
    order_idx = df_processed.index

X_cat = X_cat.loc[order_idx].reset_index(drop=True)
y = y.loc[order_idx].reset_index(drop=True)
df_processed = df_processed.loc[order_idx].reset_index(drop=True)

# 숫자열 확인
numeric_cols = X_cat.columns.difference(cat_features)
X_cat[numeric_cols] = (
    X_cat[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
)

X_encoded = X_cat.copy()
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

DATASET_SIZE = len(y)
if DATASET_SIZE < GROWTH_THRESHOLD:
    dataset_regime = "small"
elif DATASET_SIZE < EXPANDED_THRESHOLD:
    dataset_regime = "growing"
else:
    dataset_regime = "expanded"

if dataset_regime == "small":
    n_splits = 5
    n_trials = 70
elif dataset_regime == "growing":
    n_splits = 6
    n_trials = 90
else:
    n_splits = 8
    n_trials = 120

holdout_ratio = HOLDOUT_RATIOS[dataset_regime]
holdout_size = max(int(DATASET_SIZE * holdout_ratio), MIN_HOLDOUT)
holdout_size = min(holdout_size, int(DATASET_SIZE * 0.3))
holdout_size = min(holdout_size, DATASET_SIZE - 1)

train_size = DATASET_SIZE - holdout_size
if train_size <= n_splits:
    train_size = max(DATASET_SIZE - 1, n_splits + 1)
    holdout_size = DATASET_SIZE - train_size

X_train_cat = X_cat.iloc[:train_size].reset_index(drop=True)
X_holdout_cat = X_cat.iloc[train_size:].reset_index(drop=True)
X_train_encoded = X_encoded.iloc[:train_size].reset_index(drop=True)
X_holdout_encoded = X_encoded.iloc[train_size:].reset_index(drop=True)
y_train = y.iloc[:train_size].reset_index(drop=True)
y_holdout = y.iloc[train_size:].reset_index(drop=True)

class_counts = y_train.value_counts()
if len(class_counts) == 2 and class_counts.iloc[1] > 0:
    scale_pos_weight_raw = class_counts.iloc[0] / class_counts.iloc[1]
else:
    scale_pos_weight_raw = 1.0
scale_pos_weight = float(np.clip(scale_pos_weight_raw, 0.5, 4.0))

n_splits = min(n_splits, max(len(X_train_cat) - 1, 2))
tscv = TimeSeriesSplit(n_splits=n_splits)
sampler = TPESampler(seed=42)
pruner = HyperbandPruner(min_resource=1, max_resource=n_splits, reduction_factor=3)

print(
    f"Dataset size: {DATASET_SIZE} | Regime: {dataset_regime} | "
    f"CV folds: {n_splits} | Holdout: {holdout_size} | scale_pos_weight: {scale_pos_weight:.2f}"
)


def objective_xgb(trial: optuna.Trial) -> float:
    if dataset_regime == "growing" or dataset_regime == "expanded":
        n_estimators_range = (700, 1700)
        max_depth_range = (4, 10)
        min_child_weight_range = (4, 12)
        gamma_high = 5.0
        reg_alpha_range = (0.0, 3.0)
        reg_lambda_range = (1.0, 16.0)
    else:
        n_estimators_range = (400, 900)
        max_depth_range = (3, 7)
        min_child_weight_range = (2, 8)
        gamma_high = 3.0
        reg_alpha_range = (0.0, 2.0)
        reg_lambda_range = (1.0, 15.0)

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", n_estimators_range[0], n_estimators_range[1]
        ),
        "max_depth": trial.suggest_int(
            "max_depth", max_depth_range[0], max_depth_range[1]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", min_child_weight_range[0], min_child_weight_range[1]
        ),
        "gamma": trial.suggest_float("gamma", 0.0, gamma_high),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", reg_alpha_range[0], reg_alpha_range[1]
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", reg_lambda_range[0], reg_lambda_range[1]
        ),
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "logloss",
        "tree_method": "hist",
    }

    scores = []
    for step, (train_idx, val_idx) in enumerate(tscv.split(X_train_encoded)):
        X_t, X_v = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = XGBClassifier(**params)
        model.fit(X_t, y_t)
        preds = model.predict_proba(X_v)[:, 1]
        score = roc_auc_score(y_v, preds)
        scores.append(score)
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def objective_lgbm(trial: optuna.Trial) -> float:
    if dataset_regime == "expanded":
        max_depth = trial.suggest_int("max_depth", 4, 10)
        num_leaves = trial.suggest_int("num_leaves", 63, 255)
        min_child_samples = trial.suggest_int("min_child_samples", 40, 200)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.6, 0.95)
        feature_fraction = trial.suggest_float("feature_fraction", 0.6, 0.95)
        subsample = bagging_fraction
        colsample = feature_fraction
    else:
        max_depth = trial.suggest_int("max_depth", 3, 8)
        num_leaves = trial.suggest_int("num_leaves", 31, 160)
        min_child_samples = trial.suggest_int("min_child_samples", 20, 140)
        subsample = trial.suggest_float("subsample", 0.7, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.7, 1.0)
        bagging_fraction = None
        feature_fraction = None

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "max_depth": max_depth,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "num_leaves": num_leaves,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "min_child_samples": min_child_samples,
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
        "device": "cpu",
    }

    if bagging_fraction is not None:
        params["bagging_fraction"] = bagging_fraction
    if feature_fraction is not None:
        params["feature_fraction"] = feature_fraction

    scores = cross_val_score(
        LGBMClassifier(**params),
        X_train_encoded,
        y_train,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return float(scores.mean())


def objective_cat(trial: optuna.Trial) -> float:
    if dataset_regime == "growing" or dataset_regime == "expanded":
        iteration_range = (900, 2000)
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

    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli"]
    )
    subsample = None
    if bootstrap_type == "Bernoulli":
        subsample = trial.suggest_float("subsample", 0.6, 1.0)

    bagging_temperature = None
    if bootstrap_type == "Bayesian":
        bagging_temperature = trial.suggest_float(
            "bagging_temperature", 0.0, bagging_temp_high
        )

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
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "auto_class_weights": "Balanced",
        "cat_features": cat_features,
        "verbose": False,
        "random_seed": 42,
        "allow_writing_files": False,
        "task_type": "CPU",
    }
    if subsample is not None:
        params["subsample"] = subsample
    if bagging_temperature is not None:
        params["bagging_temperature"] = bagging_temperature

    scores = []
    for step, (train_idx, val_idx) in enumerate(tscv.split(X_train_cat)):
        X_t, X_v = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostClassifier(**params)
        model.fit(
            X_t,
            y_t,
            eval_set=[(X_v, y_v)],
            early_stopping_rounds=75,
            verbose=False,
        )
        preds = model.predict_proba(X_v)[:, 1]
        try:
            score = roc_auc_score(y_v, preds)
            scores.append(score)
        except ValueError:
            continue
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores)) if scores else 0.0


def objective_rf(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_int("max_depth", 4, 22),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    scores = cross_val_score(
        RandomForestClassifier(**params),
        X_train_encoded,
        y_train,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return float(scores.mean())


def objective_et(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1400),
        "max_depth": trial.suggest_int("max_depth", 4, 26),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    scores = cross_val_score(
        ExtraTreesClassifier(**params),
        X_train_encoded,
        y_train,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    return float(scores.mean())


def evaluate_holdout(model, X_data, y_true) -> dict:
    metrics = {}
    probs = model.predict_proba(X_data)[:, 1]
    preds = (probs >= 0.5).astype(int)
    try:
        metrics["holdout_auc"] = roc_auc_score(y_true, probs)
    except ValueError:
        metrics["holdout_auc"] = float("nan")
    metrics["holdout_accuracy"] = accuracy_score(y_true, preds)
    metrics["holdout_precision"] = precision_score(y_true, preds, zero_division=0)
    metrics["holdout_recall"] = recall_score(y_true, preds, zero_division=0)
    metrics["holdout_f1"] = f1_score(y_true, preds, zero_division=0)
    return metrics


print("\n[Step 2/5] Optimizing XGBoost with pruning...")
study_xgb = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study_xgb.optimize(objective_xgb, n_trials=n_trials)

print("\n[Step 3/5] Optimizing LightGBM with pruning...")
study_lgbm = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study_lgbm.optimize(objective_lgbm, n_trials=n_trials)

print("\n[Step 4/5] Optimizing CatBoost with pruning...")
study_cat = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study_cat.optimize(objective_cat, n_trials=n_trials)

print("\n[Step 5/5] Optimizing RandomForest and ExtraTrees...")
study_rf = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study_rf.optimize(objective_rf, n_trials=max(int(n_trials * 0.7), 40))

study_et = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study_et.optimize(objective_et, n_trials=max(int(n_trials * 0.7), 40))

cv_results = {
    "XGBoost": study_xgb.best_value,
    "LightGBM": study_lgbm.best_value,
    "CatBoost": study_cat.best_value,
    "RandomForest": study_rf.best_value,
    "ExtraTrees": study_et.best_value,
}

print("\nCross-validation best ROC-AUC per model:")
for name, score in cv_results.items():
    print(f" - {name}: {score:.4f}")


def merged_params(best_params: dict, fixed: dict) -> dict:
    combined = {**best_params}
    combined.update(fixed)
    return combined


xgb_fixed = {
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
}
lgbm_fixed = {
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
    "device": "cpu",
}
cat_fixed = {
    "auto_class_weights": "Balanced",
    "cat_features": cat_features,
    "verbose": False,
    "random_seed": 42,
    "allow_writing_files": False,
    "task_type": "CPU",
}
rf_fixed = {"class_weight": "balanced", "n_jobs": -1, "random_state": 42}
et_fixed = {"class_weight": "balanced", "n_jobs": -1, "random_state": 42}

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

holdout_scores = {}
trained_models = {}

print(
    "\n[Holdout evaluation] Training each best model on training window and evaluating on holdout..."
)

# XGBoost
xgb_model = XGBClassifier(**merged_params(study_xgb.best_params, xgb_fixed))
xgb_model.fit(X_train_encoded, y_train)
trained_models["XGBoost"] = xgb_model
holdout_scores["XGBoost"] = evaluate_holdout(xgb_model, X_holdout_encoded, y_holdout)

# LightGBM
lgbm_model = LGBMClassifier(**merged_params(study_lgbm.best_params, lgbm_fixed))
lgbm_model.fit(X_train_encoded, y_train)
trained_models["LightGBM"] = lgbm_model
holdout_scores["LightGBM"] = evaluate_holdout(lgbm_model, X_holdout_encoded, y_holdout)

# CatBoost
cat_model = CatBoostClassifier(**merged_params(study_cat.best_params, cat_fixed))
cat_model.fit(
    X_train_cat,
    y_train,
    eval_set=[(X_holdout_cat, y_holdout)],
    use_best_model=True,
    verbose=False,
)
trained_models["CatBoost"] = cat_model
holdout_scores["CatBoost"] = evaluate_holdout(cat_model, X_holdout_cat, y_holdout)

# RandomForest
rf_model = RandomForestClassifier(**merged_params(study_rf.best_params, rf_fixed))
rf_model.fit(X_train_encoded, y_train)
trained_models["RandomForest"] = rf_model
holdout_scores["RandomForest"] = evaluate_holdout(
    rf_model, X_holdout_encoded, y_holdout
)

# ExtraTrees
et_model = ExtraTreesClassifier(**merged_params(study_et.best_params, et_fixed))
et_model.fit(X_train_encoded, y_train)
trained_models["ExtraTrees"] = et_model
holdout_scores["ExtraTrees"] = evaluate_holdout(et_model, X_holdout_encoded, y_holdout)

for name, metrics in holdout_scores.items():
    print(
        f" - {name}: AUC={metrics['holdout_auc']:.4f}, "
        f"Precision={metrics['holdout_precision']:.4f}, Recall={metrics['holdout_recall']:.4f}"
    )


def _selection_auc(name: str) -> float:
    auc_val = holdout_scores[name].get("holdout_auc", float("-inf"))
    if isinstance(auc_val, float) and np.isnan(auc_val):
        return float("-inf")
    return auc_val


best_model_name = max(holdout_scores, key=_selection_auc)
best_model = trained_models[best_model_name]

if best_model_name == "CatBoost":
    best_model_path = os.path.join(models_dir, "best_model.cbm")
    best_model.save_model(best_model_path)
else:
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)

report = {
    "dataset_size": DATASET_SIZE,
    "dataset_regime": dataset_regime,
    "train_size": train_size,
    "holdout_size": holdout_size,
    "scale_pos_weight": scale_pos_weight,
    "cv_best_scores": cv_results,
    "holdout_metrics": holdout_scores,
    "best_model": best_model_name,
    "best_model_path": best_model_path,
    "best_params": {
        "XGBoost": merged_params(study_xgb.best_params, xgb_fixed),
        "LightGBM": merged_params(study_lgbm.best_params, lgbm_fixed),
        "CatBoost": merged_params(study_cat.best_params, cat_fixed),
        "RandomForest": merged_params(study_rf.best_params, rf_fixed),
        "ExtraTrees": merged_params(study_et.best_params, et_fixed),
    },
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

report_path = os.path.join(models_dir, "model_selection_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(
    f"\nBest model: {best_model_name} "
    f"(CV AUC={cv_results[best_model_name]:.4f}, "
    f"Holdout AUC={holdout_scores[best_model_name]['holdout_auc']:.4f})"
)
print(f"Saved best model to: {best_model_path}")
print(f"Saved selection report to: {report_path}")
