import pandas as pd
import numpy as np

# 컬럼 이름 상수로 관리
DATE_COL = "매수날짜"
TARGET_CLASSIFICATION = "Win"
TARGET_REGRESSION = "수익률"

RENAME_MAP = {
    "(매수날짜)": DATE_COL,
    "(종목코드)": "종목코드",
    "(시가총액, 억)": "시가총액",
    "(거래대금, 억)": "거래대금",
    "(등락률)": "등락률",
    "(선정 순위)": "선정순위",
    "(기관_순매수)": "기관_순매수",
    "(외국인_순매수)": "외국인_순매수",
    "(테마/섹터)": "테마_섹터",
    "(차트통과)": "차트통과",
    "(차트분석)": "차트분석",
    "(매수 가격)": "매수가격",
    "(매도 가격)": "매도가격",
    "(총 종목 수)": "총_종목수",
    "(평균 거래대금)": "평균_거래대금",
    "(수익률, %)": TARGET_REGRESSION,
    "(Win)": TARGET_CLASSIFICATION,
    "(kospi, %)": "kospi",
    "(kosdaq, %)": "kosdaq",
    "(시장구분)": "시장구분",
    "(수익 구간)": "수익_구간",
    "(중요 손실 지표)": "중요_손실_지표",
    "(프로그램_순매수)": "프로그램_순매수",
    "(체결강도)": "체결강도",
}


def preprocess_data(df, task="classification", target_col=None):
    """
    주식 데이터프레임을 받아 전처리하고, 특성(X), 타겟(y), 범주형 특성 리스트, 전처리된 df를 반환합니다.
    task: "classification"(기본)일 때는 Win을, "regression"일 때는 수익률을 타깃으로 사용합니다.
    target_col: 타깃 컬럼을 직접 지정하려면 사용합니다.
    """
    print("⚙️ 데이터 전처리 중...")

    df.rename(columns=RENAME_MAP, inplace=True)
    df.replace("", np.nan, inplace=True)

    # (차트분석)과 (차트통과) 결합 로직 추가
    if "차트분석" in df.columns and "차트통과" in df.columns:
        # 결측치 처리 후 결합 (예: '신고가_0', '상따_1')
        df["차트분석"] = (
            df["차트분석"].fillna("Unknown").astype(str)
            + "_"
            + df["차트통과"].fillna("N").astype(str)
        )
        df.drop(columns=["차트통과"], inplace=True)

    task = task.lower()
    if task not in {"classification", "regression"}:
        raise ValueError("task는 'classification' 또는 'regression'이어야 합니다.")

    if target_col:
        label_col = target_col
    else:
        label_col = (
            TARGET_CLASSIFICATION if task == "classification" else TARGET_REGRESSION
        )

    missing = [col for col in [label_col, DATE_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Error: 필수 컬럼({missing})이 데이터에 없습니다.")

    df.dropna(subset=[label_col, DATE_COL], inplace=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    percent_cols = ["kospi", "kosdaq"]
    for col in percent_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace("%", "").astype(float)

    if task == "classification":
        df[label_col] = df[label_col].astype(int)
    else:
        # 회귀용 타깃(수익률) 문자열에 포함된 %/콤마를 제거 후 숫자로 변환
        df[label_col] = (
            df[label_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df.dropna(subset=[label_col], inplace=True)

    if "기관_순매수" in df.columns:
        df["기관_순매수"] = df["기관_순매수"].fillna(0)
    if "외국인_순매수" in df.columns:
        df["외국인_순매수"] = df["외국인_순매수"].fillna(0)

    for col in ["프로그램_순매수", "체결강도"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 연속형 컬럼의 결측을 중앙값으로 보정
    numeric_cols = [c for c in df.columns if c not in [DATE_COL, "종목코드", label_col]]
    for col in numeric_cols:
        if df[col].dtype != object and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # 날짜 관련 컬럼 처리
    if "day_name" in df.columns:
        df["day_name"] = df["day_name"].fillna("Unknown").astype(str)
    if "day_of_month" in df.columns:
        df["day_of_month"] = pd.to_numeric(df["day_of_month"], errors="coerce")
        if df["day_of_month"].isnull().any():
            df["day_of_month"].fillna(df["day_of_month"].median(), inplace=True)
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        if df["month"].isnull().any():
            df["month"].fillna(df["month"].median(), inplace=True)

    # 시계열 정렬
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 특성 선택
    exclude_cols = {
        DATE_COL,
        TARGET_CLASSIFICATION,
        TARGET_REGRESSION,
        "종목코드",
        "매수가격",
        "매도가격",
        "수익_구간",
        "중요_손실_지표",
    }
    exclude_cols.add(label_col)  # 현재 task의 타겟 컬럼도 제외
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    cat_features_candidates = ["테마_섹터", "차트분석", "시장구분", "day_name"]
    cat_features = [col for col in cat_features_candidates if col in feature_cols]

    X = df[feature_cols].copy()
    y = df[label_col]

    # 범주형 특성 처리
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    print(
        f"✅ 전처리 완료! (총 {len(X)}개 샘플, {len(feature_cols)}개 특성, task={task}, target={label_col})"
    )

    return X, y, cat_features, df
