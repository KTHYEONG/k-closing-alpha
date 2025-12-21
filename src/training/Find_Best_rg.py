import os
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.data.db_loader import load_trade_log_from_db
from src.processing.preprocessor import preprocess_data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

# ==========================================
# [최적화 설정] CPU 코어 할당 전략
# ==========================================
N_JOBS_OPTUNA = 6  # 동시에 실행할 Trial 개수 (P코어 개수에 맞춤)
N_JOBS_MODEL = 1  # 각 모델은 단일 스레드만 사용 (오버헤드 방지)


def load_regression_data():
    """Load & preprocess data for regression using shared preprocessor."""
    print("\n[Step 1/5] Loading data from local DB...", flush=True)
    start_time = time.time()

    df_raw = load_trade_log_from_db()

    elapsed = time.time() - start_time
    print(f" -> Data loaded successfully in {elapsed:.2f} seconds.", flush=True)

    print(" -> Preprocessing data...", flush=True)
    X_cat, y, cat_features, df_processed = preprocess_data(
        df_raw.copy(), task="regression"
    )

    order_idx = df_processed.index
    X_cat = X_cat.loc[order_idx]
    y = y.loc[order_idx]

    # 숫자열 확인
    numeric_cols = X_cat.columns.difference(cat_features)
    X_cat[numeric_cols] = (
        X_cat[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )

    # 범주형 라벨 인코딩 버전 (LightGBM/XGBoost 등 숫자형 입력용)
    X_encoded = X_cat.copy()
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

    return X_cat, X_encoded, y, cat_features, label_encoders


print("\n[Step 2/5] Preparing data...", flush=True)
X_cat, X_encoded, y, cat_features, label_encoders = load_regression_data()

# Holdout 세트 분리
DATASET_SIZE = len(y)
holdout_ratio = 0.2  # 20%를 Holdout으로 사용
holdout_size = max(int(DATASET_SIZE * holdout_ratio), 50)  # 최소 50개 확보
train_size = DATASET_SIZE - holdout_size

X_train_cat, X_holdout_cat = X_cat.iloc[:train_size], X_cat.iloc[train_size:]
X_train_encoded, X_holdout_encoded = (
    X_encoded.iloc[:train_size],
    X_encoded.iloc[train_size:],
)
y_train, y_holdout = y.iloc[:train_size], y.iloc[train_size:]

# CatBoost에 사용할 카테고리 컬럼 인덱스
cat_feature_indices = [X_cat.columns.get_loc(c) for c in cat_features]

n_splits = min(5, max(2, len(y_train) - 1))
tscv = TimeSeriesSplit(n_splits=n_splits)
sampler = TPESampler(seed=42)
pruner = HyperbandPruner(min_resource=1, max_resource=n_splits, reduction_factor=3)

print(
    f"Dataset size: {DATASET_SIZE} | Train: {train_size} | Holdout: {holdout_size} | "
    f"CV folds: {n_splits} | Features: {len(X_cat.columns)}",
    flush=True,
)


# ===============
# Objective funcs
# ===============
def objective_rf(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "n_jobs": N_JOBS_MODEL,
        "random_state": 42,
    }
    scores = []
    for train_idx, val_idx in tscv.split(X_train_encoded):
        model = RandomForestRegressor(**params)
        model.fit(X_train_encoded.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train_encoded.iloc[val_idx])
        scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
    return float(np.mean(scores))


def objective_et(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "n_jobs": N_JOBS_MODEL,
        "random_state": 42,
    }
    scores = []
    for train_idx, val_idx in tscv.split(X_train_encoded):
        model = ExtraTreesRegressor(**params)
        model.fit(X_train_encoded.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train_encoded.iloc[val_idx])
        scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
    return float(np.mean(scores))


def objective_cat(trial: optuna.Trial) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1200),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "loss_function": "MAE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": N_JOBS_MODEL,
    }
    scores = []
    for step, (train_idx, val_idx) in enumerate(tscv.split(X_train_cat)):
        model = CatBoostRegressor(**params)
        model.fit(
            X_train_cat.iloc[train_idx],
            y_train.iloc[train_idx],
            cat_features=cat_feature_indices,
            eval_set=[(X_train_cat.iloc[val_idx], y_train.iloc[val_idx])],
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=False,
        )
        preds = model.predict(X_train_cat.iloc[val_idx])
        score = mean_absolute_error(y_train.iloc[val_idx], preds)
        scores.append(score)

        trial.report(np.mean(scores), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def objective_ngb(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "minibatch_frac": trial.suggest_float("minibatch_frac", 0.5, 1.0),
        "random_state": 42,
        "verbose": False,
    }
    scores = []
    for step, (train_idx, val_idx) in enumerate(tscv.split(X_train_encoded)):

        model = NGBRegressor(Dist=Normal, Score=LogScore, **params)
        model.fit(
            X_train_encoded.iloc[train_idx],
            y_train.iloc[train_idx],
            X_val=X_train_encoded.iloc[val_idx],
            Y_val=y_train.iloc[val_idx],
            early_stopping_rounds=30,
        )
        preds = model.predict(X_train_encoded.iloc[val_idx])
        scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))

        # Pruning
        trial.report(np.mean(scores), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


models = {
    "RandomForest": (objective_rf, 40),
    "ExtraTrees": (objective_et, 40),
    "CatBoost": (objective_cat, 50),
    "NGBoost": (objective_ngb, 25),
}

cv_results = {}
studies = {}

print(
    "\n[Step 3/5] Running model comparison (Optuna with Parallel Trials)...", flush=True
)
print(
    f" -> Parallel Jobs: {N_JOBS_OPTUNA} (Optuna) | Model Threads: {N_JOBS_MODEL}",
    flush=True,
)

for name, (objective, n_trials) in models.items():
    print(f"\n[{name}] Searching... ({n_trials} trials)", flush=True)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # [최적화] n_jobs=N_JOBS_OPTUNA (병렬 Trial 실행)
    study.optimize(
        objective, n_trials=n_trials, n_jobs=N_JOBS_OPTUNA, show_progress_bar=True
    )

    cv_results[name] = study.best_value
    studies[name] = study
    print(f" -> Best CV MAE: {study.best_value:.5f}", flush=True)

print("\n[Step 4/5] Evaluating best models on Holdout set...", flush=True)
holdout_scores = {}

# 최종 평가 시에는 단일 모델을 빠르게 학습하기 위해 n_jobs=-1 사용
FINAL_N_JOBS = -1

for name, study in studies.items():
    params = study.best_params

    # 학습된 파라미터에서 thread 관련 설정이 있다면 제거 (충돌 방지)
    if "thread_count" in params:
        del params["thread_count"]
    if "n_jobs" in params:
        del params["n_jobs"]

    if name == "RandomForest":
        model = RandomForestRegressor(n_jobs=FINAL_N_JOBS, random_state=42, **params)
        model.fit(X_train_encoded, y_train)
        preds = model.predict(X_holdout_encoded)
    elif name == "ExtraTrees":
        model = ExtraTreesRegressor(n_jobs=FINAL_N_JOBS, random_state=42, **params)
        model.fit(X_train_encoded, y_train)
        preds = model.predict(X_holdout_encoded)
    elif name == "CatBoost":
        model = CatBoostRegressor(
            loss_function="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=FINAL_N_JOBS,
            **params,
        )
        model.fit(X_train_cat, y_train, cat_features=cat_feature_indices, verbose=False)
        preds = model.predict(X_holdout_cat)
    elif name == "NGBoost":
        model = NGBRegressor(
            Dist=Normal, Score=LogScore, random_state=42, verbose=False, **params
        )
        model.fit(X_train_encoded, y_train)
        preds = model.predict(X_holdout_encoded)

    holdout_scores[name] = mean_absolute_error(y_holdout, preds)
    print(
        f" - {name}: Holdout MAE = {holdout_scores[name]:.5f} (CV MAE = {cv_results[name]:.5f})",
        flush=True,
    )


print("\n[Step 5/5] Ranking models by Holdout MAE...", flush=True)
ranked = sorted(holdout_scores.items(), key=lambda x: x[1])
for rank, (name, score) in enumerate(ranked, 1):
    print(f" {rank}. {name}: MAE={score:.5f}", flush=True)

best_model_name, best_mae = ranked[0]

print("\n--- Best Model Summary ---", flush=True)
print(f"Best Model: {best_model_name}", flush=True)
print(f"Holdout MAE: {best_mae:.5f}", flush=True)
print(f"CV MAE: {cv_results[best_model_name]:.5f}", flush=True)
