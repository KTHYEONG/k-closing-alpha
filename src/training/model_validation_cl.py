import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import CalibrationDisplay
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.data.gsheet_loader import load_and_combine_sheets
from src.processing.preprocessor import preprocess_data

# ===========================
# 0. 설정
# ===========================
# Google Sheet 인증 키는 환경변수(GSPREAD_KEY_PATH)에서 주입된다고 가정
GOOGLE_SHEET_NAME = "Stock"
WORKSHEET_NAMES = ["Trade", "Trade2"]
TRAIN_RATIO = 0.8
MODEL_PATH = os.path.join(project_root, "models", "best_stock_ai.cbm")

# ===========================
# 1. 데이터 로드 및 전처리
# ===========================
# 데이터 로드
df_raw = load_and_combine_sheets(GOOGLE_SHEET_NAME, WORKSHEET_NAMES)

# 데이터 전처리 (학습 코드와 완전히 동일한 로직 사용)
X, y, _, df_processed = preprocess_data(df_raw)

# ===========================
# 2. 검증 데이터 분리 (Out-of-Time Testing)
# ===========================
# 전체 데이터의 앞쪽 80%를 학습, 뒤쪽 20%를 검증용으로 분리 (시계열 기준)
if "매수날짜" not in df_processed.columns:
    raise KeyError("전처리된 데이터에 '매수날짜' 컬럼이 필요합니다.")

sorted_idx = df_processed["매수날짜"].sort_values().index
X = X.iloc[sorted_idx].reset_index(drop=True)
y = y.iloc[sorted_idx].reset_index(drop=True)
df_processed = df_processed.iloc[sorted_idx].reset_index(drop=True)

if len(X) < 5:
    print("⚠️ 데이터가 너무 적어 분할할 수 없습니다. 전체 데이터로 테스트합니다.")
    split_point = 0
else:
    split_point = max(1, int(len(X) * TRAIN_RATIO))
    split_point = min(split_point, len(X) - 1)

X_train = X.iloc[:split_point]
y_train = y.iloc[:split_point]
X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]

dates_train = df_processed["매수날짜"].iloc[:split_point]
dates_test = df_processed["매수날짜"].iloc[split_point:]

print(
    f"📖 학습 데이터 구간: {dates_train.min().date()} ~ {dates_train.max().date()} ({len(X_train)}개)"
)
print(
    f"🧪 검증 데이터 구간: {dates_test.min().date()} ~ {dates_test.max().date()} ({len(X_test)}개)"
)
print(f"🧪 검증 샘플 수: {len(X_test)}")

# ===========================
# 3. 저장된 모델 로드 및 예측
# ===========================
if not os.path.exists(MODEL_PATH):
    print(f"Error: 학습된 모델 파일('{MODEL_PATH}')이 없습니다.")
    sys.exit(1)

model = CatBoostClassifier()
model.load_model(MODEL_PATH)
print(f"✅ 모델 로드 완료: {MODEL_PATH}")

# 확률 예측 (Class 1일 확률)
probs = model.predict_proba(X_test)[:, 1]
# 클래스 예측 (기본 임계값 0.5 사용 시)
preds = model.predict(X_test)

# ===========================
# 4. 상세 신뢰도 지표 출력
# ===========================
print("\n📊 [Reliability Report]")
try:
    auc_val = roc_auc_score(y_test, probs)
    print(f"AUC Score: {auc_val:.4f}")
except ValueError:
    print("AUC Score: N/A (한 가지 클래스만 존재함)")

print("-" * 30)
print(classification_report(y_test, preds, target_names=["Lose(0)", "Win(1)"]))

# 트레이딩 관점 중요 지표
precision = precision_score(y_test, preds, zero_division=0)
print(f"🔥 정밀도 (Precision): {precision:.4f}")
print("   -> 모델이 '사라'고 했을 때 실제로 이긴 비율입니다. (가장 중요)")

# ===========================
# 5. 임계값(Threshold) 시뮬레이션
# ===========================
# 모델이 확신할 때만(예: 확률 0.6 이상) 진입했을 때의 성능 확인
thresholds = [0.5, 0.6, 0.7, 0.8]
print("\n🎯 [Threshold Simulation]")
print(f"{'Threshold':<10} | {'Win Rate (Precision)':<20} | {'Trade Count':<12}")
print("-" * 50)

for th in thresholds:
    high_conf_idx = probs >= th
    if sum(high_conf_idx) > 0:
        real_win_rate = np.mean(y_test[high_conf_idx])
        trade_count = sum(high_conf_idx)
        print(f"{th:<10} | {real_win_rate:.2%}             | {trade_count:<12}")
    else:
        print(f"{th:<10} | N/A (No Trades)      | 0")

# ===========================
# 6. 시각화: Calibration Curve (확률 신뢰도)
# ===========================
try:
    plt.figure(figsize=(10, 5))
    disp = CalibrationDisplay.from_predictions(y_test, probs, n_bins=10, name="Model")
    disp.ax_.plot(
        [0, 1], [0, 1], linestyle="--", color="black", label="Perfectly Calibrated"
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual Win Rate)")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend()
    plt.show()
except Exception as e:
    print(f"\n⚠️ 시각화 오류 (데이터 부족 등): {e}")

# ===========================
# 7. 시각화: 확률 분포 (Probability Distribution)
# ===========================
try:
    plt.figure(figsize=(10, 5))
    sns.histplot(
        probs[y_test == 0], color="red", alpha=0.3, label="Actual Lose", kde=True
    )
    sns.histplot(
        probs[y_test == 1], color="blue", alpha=0.3, label="Actual Win", kde=True
    )
    plt.title("Prediction Probability Distribution by Class")
    plt.xlabel("Predicted Probability (Win)")
    plt.legend()
    plt.show()
except Exception as e:
    pass
