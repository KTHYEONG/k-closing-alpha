import json
import os
import sys
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.data.gsheet_loader import load_and_combine_sheets
from src.processing.preprocessor import preprocess_data


GOOGLE_SHEET_NAME = "Stock"
WORKSHEET_NAMES = ["Trade", "Trade2"]
TRAIN_RATIO = 0.8
MODEL_PATH = os.path.join(project_root, "models", "best_stock_rg.cbm")
ENCODER_PATH = os.path.join(project_root, "models", "best_stock_rg_label_encoders.json")
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]


def load_label_encoder_map(path):
    if not os.path.exists(path):
        print(f"[Warn] Label encoder file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    encoder_map = {}
    for col, classes in raw.items():
        mapping = {str(cls): idx for idx, cls in enumerate(classes)}
        unknown_idx = mapping.get("Unknown", len(mapping))
        encoder_map[col] = {"mapping": mapping, "unknown": unknown_idx}
    print(f"[Info] Loaded encoders for: {list(encoder_map.keys())}")
    return encoder_map


def encode_categoricals(df, cat_features, encoder_map):
    df = df.copy()
    for col in cat_features:
        if col not in df.columns:
            continue
        if encoder_map and col in encoder_map:
            mapping = encoder_map[col]["mapping"]
            unknown_idx = encoder_map[col]["unknown"]
            df[col] = (
                df[col]
                .astype(str)
                .apply(lambda val: mapping.get(val, unknown_idx))
                .astype(float)
            )
        else:
            df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)
    return df


def summarize_thresholds(pred_scores, actual_returns, thresholds):
    rows = []
    actual_positive = (actual_returns > 0).astype(float)
    for th in thresholds:
        mask = pred_scores >= th
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "Threshold": th,
                    "Count": 0,
                    "Avg_Pred": np.nan,
                    "Avg_Return": np.nan,
                    "Median_Return": np.nan,
                    "Positive_Rate": np.nan,
                }
            )
            continue
        rows.append(
            {
                "Threshold": th,
                "Count": count,
                "Avg_Pred": float(np.mean(pred_scores[mask])),
                "Avg_Return": float(np.mean(actual_returns[mask])),
                "Median_Return": float(np.median(actual_returns[mask])),
                "Positive_Rate": float(np.mean(actual_positive[mask])),
            }
        )
    return pd.DataFrame(rows)


def print_distribution_stats(label, values):
    series = pd.Series(values)
    print(f"\n=== {label} 분포 요약 ===")
    print(
        f"평균: {series.mean():.4f} | 중앙값: {series.median():.4f} | 표준편차: {series.std():.4f}"
    )
    quantiles = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for q, val in quantiles.items():
        print(f"  q={q:.2f}: {val:.4f}")


def compute_residual_summary(preds, actual):
    residuals = actual - preds
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "abs_mean": float(np.mean(np.abs(residuals))),
        "abs_median": float(np.median(np.abs(residuals))),
    }


def sign_accuracy(preds, actual):
    mask = actual != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.sign(preds[mask]) == np.sign(actual[mask])))


def build_prediction_bins(preds, actual, n_bins=5):
    df = pd.DataFrame({"pred": preds, "actual": actual})
    df["residual"] = df["actual"] - df["pred"]
    df["bin"] = pd.qcut(df["pred"], q=n_bins, labels=False, duplicates="drop")
    grouped = (
        df.groupby("bin")
        .agg(
            count=("pred", "size"),
            pred_mean=("pred", "mean"),
            actual_mean=("actual", "mean"),
            residual_mean=("residual", "mean"),
            abs_residual_mean=("residual", lambda x: np.mean(np.abs(x))),
        )
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(int)
    return grouped


def main():
    print("[Step 1/4] Loading Google Sheet data...")
    df_raw = load_and_combine_sheets(GOOGLE_SHEET_NAME, WORKSHEET_NAMES)

    print("[Step 2/4] Preprocessing data for regression task...")
    X_raw, y, cat_features, df_processed = preprocess_data(df_raw, task="regression")

    if "매수날짜" in df_processed.columns:
        order_idx = pd.to_datetime(df_processed["매수날짜"]).sort_values().index
    else:
        order_idx = df_processed.index

    X_raw = X_raw.loc[order_idx].reset_index(drop=True)
    y = y.loc[order_idx].reset_index(drop=True)

    print(f"[Info] Samples: {len(X_raw)} | Features: {len(X_raw.columns)}")

    encoder_map = load_label_encoder_map(ENCODER_PATH)
    X = encode_categoricals(X_raw, cat_features, encoder_map)

    split_point = max(1, int(len(X) * TRAIN_RATIO))
    split_point = min(split_point, len(X) - 1)

    X_train = X.iloc[:split_point].reset_index(drop=True)
    y_train = y.iloc[:split_point].reset_index(drop=True)
    X_holdout = X.iloc[split_point:].reset_index(drop=True)
    y_holdout = y.iloc[split_point:].reset_index(drop=True)

    print(
        f"[Info] Train size: {len(X_train)} | Holdout size: {len(X_holdout)} (ratio {TRAIN_RATIO:.2f})"
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print("[Step 3/4] Loading ExtraTrees model...")
    model = load(MODEL_PATH)

    print("[Step 4/4] Evaluating on holdout window...")
    preds = model.predict(X_holdout)
    score_series = pd.Series(preds, name="pred_score")
    actual_series = pd.Series(y_holdout.values, name="actual_return")

    epsilon = 1e-6
    metrics = {
        "mae": mean_absolute_error(y_holdout, preds),
        "median_ae": median_absolute_error(y_holdout, preds),
        "rmse": mean_squared_error(y_holdout, preds) ** 0.5,
        "r2": r2_score(y_holdout, preds),
        "explained_variance": explained_variance_score(y_holdout, preds),
        "mape": float(
            np.mean(
                np.abs(
                    (y_holdout.values - preds)
                    / np.clip(np.abs(y_holdout.values), epsilon, None)
                )
            )
            * 100
        ),
        "corr": float(np.corrcoef(y_holdout, preds)[0, 1]),
        "sign_accuracy": sign_accuracy(preds, y_holdout.values),
    }

    print("\n=== Holdout Metrics (Regression) ===")
    for key, value in metrics.items():
        print(f"{key:>20}: {value:.6f}")

    print("\n=== Decision Reference (Threshold Table) ===")
    print_distribution_stats("예측값", score_series)
    print_distribution_stats("실제 수익률", actual_series)

    residual_info = compute_residual_summary(preds, y_holdout.values)
    print("\n=== 잔차(Actual - Pred) 요약 ===")
    for key, value in residual_info.items():
        print(f"{key:>15}: {value:.4f}")

    bucket_df = build_prediction_bins(preds, y_holdout.values, n_bins=5)
    if not bucket_df.empty:
        print("\n=== 예측 구간별 성과 비교 (q-cut) ===")
        print(
            bucket_df.to_string(
                index=False,
                formatters={
                    "pred_mean": "{:.4f}".format,
                    "actual_mean": "{:.4f}".format,
                    "residual_mean": "{:.4f}".format,
                    "abs_residual_mean": "{:.4f}".format,
                },
            )
        )

    threshold_df = summarize_thresholds(
        score_series.values, y_holdout.values, thresholds=THRESHOLDS
    )
    if not threshold_df.empty:
        print("\n=== 사용자 정의 임계값별 실적 비교 ===")
        print(
            threshold_df.to_string(
                index=False,
                formatters={
                    "Avg_Pred": "{:.4f}".format,
                    "Avg_Return": "{:.4f}".format,
                    "Median_Return": "{:.4f}".format,
                    "Positive_Rate": "{:.2%}".format,
                },
            )
        )




if __name__ == "__main__":
    main()
