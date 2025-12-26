"""
Model Validation Script (Regression)
- DB 기반 과거 매매 기록으로 모델 성능을 검증합니다.
- Model_Performance_Validator.py와 동일한 로직을 사용하되, 검증 지표를 더 상세히 출력합니다.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from joblib import load

# 프로젝트 루트 경로 설정
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.processing.preprocessor import preprocess_data
from src import settings
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

# 설정
DB_PATH = os.path.join(project_root, "data", "stock.db")
MODEL_PATH = str(settings.MODEL_PATH)
TRAIN_RATIO = 0.8
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]


def load_all_trade_logs():
    """DB에서 모든 매매 기록을 불러옵니다."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {DB_PATH}")
        
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM table_trade_log"
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ DB에서 {len(df)}건의 매매 기록 로드 완료")
    return df


def summarize_thresholds(pred_scores, actual_returns, thresholds):
    """임계값별 성과를 요약합니다."""
    rows = []
    actual_positive = (actual_returns > 0).astype(float)
    
    for th in thresholds:
        mask = pred_scores >= th
        count = int(mask.sum())
        
        if count == 0:
            rows.append({
                "Threshold": th,
                "Count": 0,
                "Avg_Pred": np.nan,
                "Avg_Return": np.nan,
                "Median_Return": np.nan,
                "Win_Rate": np.nan,
            })
            continue
        
        rows.append({
            "Threshold": th,
            "Count": count,
            "Avg_Pred": float(np.mean(pred_scores[mask])),
            "Avg_Return": float(np.mean(actual_returns[mask])),
            "Median_Return": float(np.median(actual_returns[mask])),
            "Win_Rate": float(np.mean(actual_positive[mask])),
        })
    
    return pd.DataFrame(rows)


def print_distribution_stats(label, values):
    """분포 통계를 출력합니다."""
    series = pd.Series(values)
    print(f"\n=== {label} 분포 요약 ===")
    print(f"평균: {series.mean():.4f} | 중앙값: {series.median():.4f} | 표준편차: {series.std():.4f}")
    
    quantiles = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for q, val in quantiles.items():
        print(f"  q={q:.2f}: {val:.4f}")


def sign_accuracy(preds, actual):
    """방향성 정확도를 계산합니다 (부호 일치율)."""
    mask = actual != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.sign(preds[mask]) == np.sign(actual[mask])))


def main():
    print("=" * 70)
    print(" 모델 검증 (Regression) - 과거 재현 검증 (Retro-Validation)")
    print(" Data Leakage 방지: Train 데이터만으로 학습한 모델로 평가")
    print("=" * 70)
    
    # Step 1: DB에서 데이터 로드
    print("\n[Step 1/6] DB에서 매매 기록 로드 중...")
    df_raw = load_all_trade_logs()
    
    # Step 2: 전처리 (최신 preprocessor.py 사용)
    print("\n[Step 2/6] 데이터 전처리 중...")
    X, y, cat_features, df_processed = preprocess_data(df_raw, task="regression")
    
    # day_name 컬럼 제거 (최신 preprocessor는 weekday만 사용)
    if "day_name" in X.columns:
        X = X.drop(columns=["day_name"])
        print(f"[Info] 'day_name' 컬럼 제거됨 (weekday 사용)")
    
    # Step 3: 저장된 모델의 파라미터 로드 (모델 자체는 로드하지 않음!)
    print("\n[Step 3/6] 저장된 모델의 파라미터 로드 중...")
    
    metrics_path = str(MODEL_PATH).replace(".joblib", "_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics 파일을 찾을 수 없습니다: {metrics_path}")
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        best_params = saved_data.get("best_params", {})
    
    if not best_params:
        raise ValueError("저장된 파라미터를 찾을 수 없습니다.")
    
    print(f"✅ 파라미터 로드 완료")
    print(f"   주요 설정: depth={best_params.get('depth')}, "
          f"learning_rate={best_params.get('learning_rate'):.4f}, "
          f"iterations={best_params.get('iterations')}")
    
    # Step 4: 피처 매칭 및 범주형 처리
    print("\n[Step 4/6] 피처 준비 중...")
    
    # 범주형 변수를 문자열로 변환 (CatBoost 호환)
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
    
    # 날짜순 정렬 (시계열 분할을 위해)
    if "매수날짜" in df_processed.columns:
        order_idx = pd.to_datetime(df_processed["매수날짜"], errors='coerce').sort_values().index
        X = X.loc[order_idx].reset_index(drop=True)
        y = y.loc[order_idx].reset_index(drop=True)
    
    # Train/Holdout 분할 (시계열 기준)
    split_point = max(1, int(len(X) * TRAIN_RATIO))
    split_point = min(split_point, len(X) - 1)
    
    X_train = X.iloc[:split_point].reset_index(drop=True)
    y_train = y.iloc[:split_point].reset_index(drop=True)
    X_holdout = X.iloc[split_point:].reset_index(drop=True)
    y_holdout = y.iloc[split_point:].reset_index(drop=True)
    
    print(f"✅ Train: {len(X_train)}건 | Holdout: {len(X_holdout)}건 (비율: {TRAIN_RATIO:.2f})")
    
    # Step 5: 과거 재현 - Train 데이터로만 새 모델 학습
    print("\n[Step 5/6] 과거 재현 모델 학습 중 (Train 데이터만 사용)...")
    
    # CatBoost cat_features 인덱스
    cat_indices = []
    if cat_features:
        cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    
    # 저장된 파라미터로 새 모델 생성
    from catboost import CatBoostRegressor
    best_params_clean = best_params.copy()
    best_params_clean["allow_writing_files"] = False
    retro_model = CatBoostRegressor(**best_params_clean)
    
    # Train 데이터로만 학습 (Holdout은 절대 보여주지 않음!)
    retro_model.fit(
        X_train, y_train,
        cat_features=cat_indices,
        verbose=100
    )
    
    print(f"✅ 과거 재현 모델 학습 완료")
    
    # Step 6: Holdout 평가 (공정한 평가)
    print("\n[Step 6/6] Holdout 성능 평가 중...")
    preds = retro_model.predict(X_holdout)
    
    # 기본 회귀 지표
    epsilon = 1e-6
    metrics = {
        "MAE": mean_absolute_error(y_holdout, preds),
        "Median_AE": median_absolute_error(y_holdout, preds),
        "RMSE": mean_squared_error(y_holdout, preds) ** 0.5,
        "R2": r2_score(y_holdout, preds),
        "Explained_Var": explained_variance_score(y_holdout, preds),
        "MAPE (%)": float(
            np.mean(
                np.abs(
                    (y_holdout.values - preds)
                    / np.clip(np.abs(y_holdout.values), epsilon, None)
                )
            )
            * 100
        ),
        "Correlation": float(np.corrcoef(y_holdout, preds)[0, 1]),
        "Sign_Accuracy": sign_accuracy(preds, y_holdout.values),
    }
    
    print("\n" + "=" * 70)
    print(" Holdout 성능 지표 (Regression) - 과거 재현 검증")
    print(" ⚠️ 이 결과는 Data Leakage가 없는 진짜 성능입니다.")
    print("=" * 70)
    for key, value in metrics.items():
        print(f"{key:>20}: {value:.6f}")
    
    # 분포 통계
    print_distribution_stats("예측값 (Pred)", preds)
    print_distribution_stats("실제 수익률 (Actual)", y_holdout.values)
    
    # 잔차 분석
    residuals = y_holdout.values - preds
    print("\n=== 잔차 (Actual - Pred) 요약 ===")
    print(f"{'mean':>15}: {float(np.mean(residuals)):.4f}")
    print(f"{'std':>15}: {float(np.std(residuals)):.4f}")
    print(f"{'abs_mean':>15}: {float(np.mean(np.abs(residuals))):.4f}")
    print(f"{'abs_median':>15}: {float(np.median(np.abs(residuals))):.4f}")
    
    # 임계값별 성과
    threshold_df = summarize_thresholds(preds, y_holdout.values, thresholds=THRESHOLDS)
    if not threshold_df.empty:
        print("\n=== 임계값별 실적 비교 ===")
        print(
            threshold_df.to_string(
                index=False,
                formatters={
                    "Avg_Pred": "{:.4f}".format,
                    "Avg_Return": "{:.4f}".format,
                    "Median_Return": "{:.4f}".format,
                    "Win_Rate": "{:.2%}".format,
                },
            )
        )
    
    print("\n" + "=" * 70)
    print(" 검증 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
