"""
Threshold Optimizer (Retro-Validation)
- 과거 재현 검증 방식으로 최적의 의사결정 임계값을 찾습니다.
- Data Leakage 방지: 80% 데이터로 모델을 재학습한 뒤, 20% Holdout으로 임계값을 최적화합니다.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostRegressor

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from src.data.db_loader import load_trade_log_from_db
from src.processing.preprocessor import preprocess_data
from src import settings

# 설정
MODEL_PATH = str(settings.MODEL_PATH)
METRICS_PATH = MODEL_PATH.replace(".joblib", "_metrics.json")
TRAIN_RATIO = 0.8

# 시뮬레이션 비중 가중치
WEIGHTS = {
    "Reduce": 0.0,      # 비중 축소 (또는 매수 금지)
    "Neutral": 1.0,     # 기본 비중
    "Expand": 1.5,      # 비중 확대
    "Max_Expand": 2.0   # 공격적 투자
}

def optimize_thresholds():
    print("\n" + "="*60)
    print(" Threshold Optimizer (Retro-Validation)")
    print(" Data Leakage 방지: Train 데이터로만 학습 후 Holdout 최적화")
    print("="*60)
    
    # 1. 파라미터 로드
    print("\n[Step 1/5] 저장된 모델 파라미터 로드 중...")
    if not os.path.exists(METRICS_PATH):
        print(f"Error: Metrics 파일을 찾을 수 없습니다: {METRICS_PATH}")
        print("먼저 Find_Opt_rg.py를 실행하여 모델을 학습하세요.")
        return
        
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        best_params = saved_data.get("best_params", {})
    
    if not best_params:
        print("Error: 저장된 파라미터를 찾을 수 없습니다.")
        return
    
    print(f"✅ 파라미터 로드 완료")
    
    # 2. 데이터 로드 및 전처리
    print("\n[Step 2/5] DB에서 데이터 로드 및 전처리 중...")
    df_raw = load_trade_log_from_db()
    X, y, cat_features, df_processed = preprocess_data(df_raw, task="regression")
    
    # day_name 컬럼 제거
    if "day_name" in X.columns:
        X = X.drop(columns=["day_name"])
        
    # 범주형 변수 처리
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
            
    # 날짜순 정렬
    if "매수날짜" in df_processed.columns:
        order_idx = pd.to_datetime(df_processed["매수날짜"], errors='coerce').sort_values().index
        X = X.loc[order_idx].reset_index(drop=True)
        y = y.loc[order_idx].reset_index(drop=True)
    
    # Train/Holdout 분할
    split_point = max(1, int(len(X) * TRAIN_RATIO))
    
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_holdout = X.iloc[split_point:]
    y_holdout = y.iloc[split_point:]
    
    print(f"✅ 데이터 분할: Train {len(X_train)}건 | Holdout {len(X_holdout)}건")
    
    # 3. 모델 재학습 (과거 재현)
    print("\n[Step 3/5] 과거 재현 모델 학습 중 (Train 데이터만 사용)...")
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    
    best_params_clean = best_params.copy()
    best_params_clean["allow_writing_files"] = False
    retro_model = CatBoostRegressor(**best_params_clean)
    retro_model.fit(X_train, y_train, cat_features=cat_indices, verbose=False)
    
    print(f"✅ 모델 학습 완료")
    
    # 4. Holdout 예측
    print("\n[Step 4/5] Holdout 데이터 예측 중...")
    preds = retro_model.predict(X_holdout)
    actuals = y_holdout.values
    
    print(f"✅ 예측 완료")
    print(f"   예측값 범위: Min={preds.min():.4f}, Max={preds.max():.4f}, Mean={preds.mean():.4f}")
    
    # 거래 비용 및 리스크 패널티 설정
    COST_RATE = 0.003  # 수수료+세금+슬리피지 (0.3%)
    RISK_PENALTY = 1.5 # 손실에 대한 패널티 가중치 (1.5배)
    
    # 5. Optuna 최적화
    print("\n[Step 5/5] 최적 임계값 탐색 중...")
    
    def objective(trial):
        # 경계값(Threshold) 3개를 찾습니다. (t1 < t2 < t3)
        low = float(np.percentile(preds, 5))
        high = float(np.percentile(preds, 95))
        
        t1 = trial.suggest_float("t1", low, high)
        t2 = trial.suggest_float("t2", t1 + 0.05, high + 0.05)
        t3 = trial.suggest_float("t3", t2 + 0.05, high + 0.1)

        # 각 구간별 마스크 생성
        mask_reduce = preds < t1
        mask_neutral = (preds >= t1) & (preds < t2)
        mask_expand = (preds >= t2) & (preds < t3)
        mask_max = preds >= t3
        
        # Real-World PnL Simulation
        def calculate_adjusted_pnl(mask, weight):
            if not np.any(mask) or weight == 0:
                return 0.0
            
            # 비용 차감 후 수익률
            net_returns = actuals[mask] - COST_RATE
            
            # 손익 가중치 적용 (수익은 1배, 손실은 1.5배 반영)
            weighted_returns = np.where(net_returns > 0, net_returns, net_returns * RISK_PENALTY)
            
            # 최종 비중 곱하기
            return np.sum(weighted_returns * weight)

        pnl = calculate_adjusted_pnl(mask_reduce, WEIGHTS["Reduce"]) + \
              calculate_adjusted_pnl(mask_neutral, WEIGHTS["Neutral"]) + \
              calculate_adjusted_pnl(mask_expand, WEIGHTS["Expand"]) + \
              calculate_adjusted_pnl(mask_max, WEIGHTS["Max_Expand"])
              
        return pnl

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # 결과 리포트
    best_t1 = study.best_params["t1"]
    best_t2 = study.best_params["t2"]
    best_t3 = study.best_params["t3"]
    
    print("\n" + "="*60)
    print(" ✅ 최적 임계값 탐색 완료 (Data Leakage 없음)")
    print("="*60)
    print(f"현재 Bins (수동):     [-inf, 0.4, 0.5, 0.6, inf]")
    print(f"최적 Bins (AI 추천): [-inf, {best_t1:.4f}, {best_t2:.4f}, {best_t3:.4f}, inf]")
    print("-" * 60)
    print(f"Reduce     (~ {best_t1:.2f})")
    print(f"Neutral    ({best_t1:.2f} ~ {best_t2:.2f})")
    print(f"Expand     ({best_t2:.2f} ~ {best_t3:.2f})")
    print(f"Max_Expand ({best_t3:.2f} ~)")
    print("-" * 60)
    print(f"최적화 점수 (PnL Sum): {study.best_value:.4f}")
    print("="*60)
    
    # 등급별 통계 출력
    print("\n[참고] 최적 임계값 적용 시 등급별 성과 (Holdout 기준)")
    result_df = pd.DataFrame({
        "pred": preds,
        "actual": actuals
    })
    
    bins = [-np.inf, best_t1, best_t2, best_t3, np.inf]
    labels = ["Reduce", "Neutral", "Expand", "Max_Expand"]
    result_df["decision"] = pd.cut(result_df["pred"], bins=bins, labels=labels)
    
    stats = result_df.groupby("decision", observed=True).agg({
        "actual": ["count", "mean", lambda x: (x > 0).mean()]
    })
    stats.columns = ["Count", "Avg_Return", "Win_Rate"]
    print(stats)
    
    print(f"\n[Guide] Daily_Pos_AI.py의 pd.cut bins 부분을 위 '최적 Bins' 값으로 교체하세요.")

if __name__ == "__main__":
    optimize_thresholds()
