"""
Model Performance Validator (Regression)
- 과거 재현 검증(Retro-Validation) 방식으로 모델의 진짜 실력을 검증합니다.
- 80% 학습 / 20% 평가 방식을 사용하여 Data Leakage를 방지합니다.
- 실험 결과를 data/experiment_history.csv에 기록합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import json
from joblib import load

# 프로젝트 루트 경로 설정
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src import settings
from src.utils.display import Colors
from src.processing.preprocessor import preprocess_data
from catboost import CatBoostRegressor

# 설정
DB_PATH = os.path.join(project_root, "data", "stock.db")
MODEL_PATH = str(settings.MODEL_PATH)
METRICS_PATH = MODEL_PATH.replace(".joblib", "_metrics.json")
TRAIN_RATIO = 0.8

def load_all_trade_logs():
    """DB에서 모든 매매 기록을 불러옵니다."""
    if not os.path.exists(DB_PATH):
        print(f"{Colors.RED}DB 파일을 찾을 수 없습니다: {DB_PATH}{Colors.RESET}")
        sys.exit(1)
        
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM table_trade_log"
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"{Colors.CYAN}✅ 매매 기록 로드 완료: 총 {len(df)}건{Colors.RESET}")
    return df

def main():
    print(f"\n{Colors.GREEN}=== 과거 재현 검증 기반 성능 평가 시작 (Retro-Validation) ==={Colors.RESET}")
    print(f"공정한 평가를 위해 {int(TRAIN_RATIO*100)}% 학습 후 나머지 {int((1-TRAIN_RATIO)*100)}% 데이터를 평가합니다.\n")
    
    # 1. 파라미터 로드
    if not os.path.exists(METRICS_PATH):
        print(f"{Colors.RED}Metrics 파일이 없습니다. 먼저 Find_Opt_rg.py를 실행하세요.{Colors.RESET}")
        return
        
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        best_params = saved_data.get("best_params", {})
    
    # 2. 데이터 로드 및 전처리
    df_raw = load_all_trade_logs()
    X, y, cat_features, df_processed = preprocess_data(df_raw, task="regression")
    
    # day_name 컬럼 제거
    if "day_name" in X.columns:
        X = X.drop(columns=["day_name"])
        
    # 범주형 변수 처리
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)
            
    # 날짜순 정렬 및 분할
    if "매수날짜" in df_processed.columns:
        order_idx = pd.to_datetime(df_processed["매수날짜"], errors='coerce').sort_values().index
        X = X.loc[order_idx].reset_index(drop=True)
        y = y.loc[order_idx].reset_index(drop=True)
        df_eval_base = df_processed.loc[order_idx].reset_index(drop=True)
    else:
        df_eval_base = df_processed.copy()

    split_point = max(1, int(len(X) * TRAIN_RATIO))
    
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_holdout = X.iloc[split_point:]
    y_holdout = y.iloc[split_point:]
    df_holdout_base = df_eval_base.iloc[split_point:]

    print(f"📊 데이터 분할: Train {len(X_train)}건 | Holdout {len(X_holdout)}건")
    
    # 3. 모델 재학습 (과거 재현)
    print(f"\n{Colors.YELLOW}모델 재학습 중... (Train 데이터만 사용){Colors.RESET}")
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    
    best_params_clean = best_params.copy()
    best_params_clean["allow_writing_files"] = False
    retro_model = CatBoostRegressor(**best_params_clean)
    retro_model.fit(X_train, y_train, cat_features=cat_indices, verbose=False)
    
    # 4. 예측 수행 (처음 보는 Holdout 데이터 대상)
    print(f"{Colors.CYAN}Holdout 데이터 평가 중...{Colors.RESET}")
    preds = retro_model.predict(X_holdout)
    
    # 5. 결과 분석용 데이터프레임 생성
    result_df = df_holdout_base.copy()
    result_df["AI_Score"] = preds
    result_df["실제_수익률"] = y_holdout.values
    
    # 6. 의사결정 등급 매기기
    # Daily_Pos_AI.py와 동일한 bins 적용
    bins = [-np.inf, 0.4, 0.5, 0.6, np.inf]
    labels = ["Reduce", "Neutral", "Expand", "Max_Expand"]
    
    result_df["AI_Decision"] = pd.cut(
        result_df["AI_Score"],
        bins=bins,
        labels=labels
    ).astype(str)
    
    # 7. 그룹별 실적 통계
    print(f"\n{Colors.GREEN}=== [Holdout] AI 검증 결과 리포트 (진짜 실력) ==={Colors.RESET}")
    
    group_stats = result_df.groupby("AI_Decision", observed=True).agg({
        "AI_Score": ["count", "mean"],
        "실제_수익률": ["mean", lambda x: (x > 0).mean()]
    })
    group_stats.columns = ["count", "Avg_Pred", "Avg_Return", "Win_Rate"]
    
    print(group_stats)
    
    # 8. 종목별 리스트 (Top 5 / Bottom 5)
    display_cols = ["매수날짜", "종목명", "AI_Score", "AI_Decision", "실제_수익률"]
    existing_cols = [c for c in display_cols if c in result_df.columns]
    
    print(f"\n{Colors.YELLOW}=== [Holdout] AI 추천 상위 5 종목 ==={Colors.RESET}")
    print(result_df.nlargest(5, "AI_Score")[existing_cols].to_string(index=False))
    
    print(f"\n{Colors.RED}=== [Holdout] AI 추천 하위 5 종목 ==={Colors.RESET}")
    print(result_df.nsmallest(5, "AI_Score")[existing_cols].to_string(index=False))
    
    # 9. 실험 결과 로깅
    save_experiment_log(group_stats, bins, len(result_df))

def save_experiment_log(group_stats, bins, total_samples):
    """실험 결과를 CSV 파일에 저장합니다."""
    import csv
    from datetime import datetime
    
    log_dir = os.path.join(project_root, "data")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, "experiment_history.csv")
    
    def get_stat(decision, col):
        try:
            val = group_stats.loc[decision, col]
            return round(float(val), 4)
        except (KeyError, TypeError):
            return ""

    thresholds_str = "/".join(map(str, bins[1:-1]))
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": f"{os.path.basename(MODEL_PATH)} (Retro)",
        "Thresholds": thresholds_str,
        "Total_Samples": total_samples,
        "Max_Expand_Ret": get_stat("Max_Expand", "Avg_Return"),
        "Max_Expand_Win": get_stat("Max_Expand", "Win_Rate"),
        "Expand_Ret": get_stat("Expand", "Avg_Return"),
        "Expand_Win": get_stat("Expand", "Win_Rate"),
        "Neutral_Ret": get_stat("Neutral", "Avg_Return"),
        "Neutral_Win": get_stat("Neutral", "Win_Rate"),
        "Reduce_Ret": get_stat("Reduce", "Avg_Return"),
        "Reduce_Win": get_stat("Reduce", "Win_Rate"),
        "Note": "Data Leakage Removal"
    }
    
    file_exists = os.path.exists(log_path)
    try:
        with open(log_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)
        print(f"\n{Colors.CYAN}✅ 실험 결과가 저장되었습니다: {log_path}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}[Error] 로그 저장 실패: {e}{Colors.RESET}")

if __name__ == "__main__":
    main()
