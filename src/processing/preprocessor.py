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
    "(v-kospi)": "v_kospi",
    "(v-kosdaq)": "v_kosdaq",
}


def preprocess_data(df, task="classification", target_col=None):
    """
    주식 데이터프레임을 받아 전처리하고, 특성(X), 타겟(y), 범주형 특성 리스트, 전처리된 df를 반환합니다.
    """
    print("⚙️ 데이터 전처리 및 영업일 기준 고도화 중...")

    df.rename(columns=RENAME_MAP, inplace=True)
    df.replace("", np.nan, inplace=True)

    # (차트분석)과 (차트통과) 결합 로직
    if "차트분석" in df.columns and "차트통과" in df.columns:
        df["차트분석"] = (
            df["차트분석"].fillna("Unknown").astype(str)
            + "_"
            + df["차트통과"].fillna("N").astype(str)
        )
        df.drop(columns=["차트통과"], inplace=True)

    # -------------------------------------------------------------------------
    # [최우선 작업] 날짜 변환 및 정렬
    # -------------------------------------------------------------------------
    if DATE_COL not in df.columns:
        raise ValueError(f"Error: 필수 컬럼({DATE_COL})이 데이터에 없습니다.")
    
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # 날짜순 정렬 (shift 연산 및 시계열 데이터 무결성 보장)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # [날짜 피처 고도화] 영업일 및 리스크 기반
    # -------------------------------------------------------------------------
    def _apply_advanced_date_features(df):
        df = df.copy()
        dt_series = df[DATE_COL]
        
        # 1. 주기성 인코딩 (Month, Day) - 실제 월 일수(days_in_month) 반영
        df["month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)
        df["day_sin"] = np.sin(2 * np.pi * dt_series.dt.day / dt_series.dt.days_in_month)
        df["day_cos"] = np.cos(2 * np.pi * dt_series.dt.day / dt_series.dt.days_in_month)

        # 2. 영업일 기준 특징 (unique_dates를 사용하여 종목 중복 영향 배제)
        unique_dates = pd.Series(sorted(dt_series.unique()))
        
        # 2-1. 오버나이트 리스크 (다음 거래일까지의 날짜 차이)
        date_risk_map = {
            date: diff for date, diff in zip(unique_dates, unique_dates.diff().dt.days.shift(-1).fillna(1))
        }
        df["date_diff"] = dt_series.map(date_risk_map)

        # 2-2. 실제 영업일 월말 (다음 데이터에서 월이 바뀌는지 체크)
        is_month_end_map = {
            date: int(month != next_month) for date, month, next_month in zip(
                unique_dates, unique_dates.dt.month, unique_dates.dt.month.shift(-1).fillna(unique_dates.dt.month.iloc[-1])
            )
        }
        df["is_trading_month_end"] = dt_series.map(is_month_end_map)

        # 3. 요일 (Categorical)
        df["weekday"] = dt_series.dt.dayofweek
        
        return df

    df = _apply_advanced_date_features(df)

    # 나머지 수치형 전처리 로직 (기존 구현 함수들 호출)
    # -------------------------------------------------------------------------
    def _apply_log_scaling(df):
        cols = ["시가총액", "거래대금", "평균_거래대금", "총_종목수", "선정순위"]
        for col in [c for c in cols if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = np.log1p(df[col].clip(lower=0))
        return df

    def _apply_signed_log_scaling(df):
        cols = ["기관_순매수", "외국인_순매수", "프로그램_순매수"]
        for col in [c for c in cols if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
        return df

    def _apply_custom_ratios(df):
        calc_cols = ["등락률", "kospi", "kosdaq"]
        for col in [c for c in calc_cols if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "체결강도" in df.columns:
            df["체결강도"] = pd.to_numeric(df["체결강도"], errors="coerce").fillna(100)
            df["체결강도"] = np.log(df["체결강도"].clip(lower=1) / 100.0)
        
        if "선정순위" in df.columns and "총_종목수" in df.columns:
            rank_raw = np.expm1(df["선정순위"]) 
            total_raw = np.expm1(df["총_종목수"])
            df["선정순위_상대"] = rank_raw / total_raw.clip(lower=1)

        if "등락률" in df.columns:
            if "시장구분" in df.columns:
                market_ref = np.where(df["시장구분"].str.contains("KOSDAQ", na=False, case=False), 
                                      df["kosdaq"], df["kospi"])
            else:
                market_ref = df["kospi"]
            df["상대_등락률"] = df["등락률"] - market_ref
            df["방어_강도"] = np.where(market_ref < 0, df["상대_등락률"], 0)

        if "거래대금" in df.columns and "평균_거래대금" in df.columns:
            raw_trade = np.expm1(df["거래대금"])
            raw_avg_trade = np.expm1(df["평균_거래대금"])
            df["상대_거래대금"] = np.log((raw_trade + 1) / (raw_avg_trade + 1).clip(lower=1))

        # 5. 수급 질적 분석 (중복 제거 및 주도성 분류)
        buy_cols = ["기관_순매수", "외국인_순매수", "프로그램_순매수"]
        if all(c in df.columns for c in buy_cols) and "거래대금" in df.columns:
            # Signed Log 복원
            raw_inst = np.sign(df["기관_순매수"]) * np.expm1(np.abs(df["기관_순매수"]))
            raw_fore = np.sign(df["외국인_순매수"]) * np.expm1(np.abs(df["외국인_순매수"]))
            raw_prog = np.sign(df["프로그램_순매수"]) * np.expm1(np.abs(df["프로그램_순매수"]))
            raw_trade = np.expm1(df["거래대금"]).clip(lower=1)
            
            # 5-1. 메이저 밀도 (순수 주체 합산: 기관 + 외국인)
            df["메이저_밀도"] = (raw_inst + raw_fore) / raw_trade
            
            # 5-2. 프로그램 주도성 (매매 방식의 기계적 강도)
            df["프로그램_주도성"] = raw_prog / raw_trade
            
        return df

    def _convert_remaining_numeric(df):
        cols = ["등락률", "kospi", "kosdaq", "day_of_month", "month", "체결강도", "v_kospi", "v_kosdaq"]
        for col in [c for c in cols if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    df = _apply_log_scaling(df)
    df = _apply_signed_log_scaling(df)
    df = _apply_custom_ratios(df)
    df = _convert_remaining_numeric(df)

    def _apply_vkospi_advanced(df):
        if "v_kospi" not in df.columns:
            return df
            
        # 1. 값이 0인 경우(결측) NaN 처리 후 날짜별 대표값(평균) 추출
        # 종목 데이터이므로 같은 날짜면 v_kospi 값은 동일해야 함
        # _convert_remaining_numeric에서 이미 0으로 채워졌을 수 있음
        temp_df = df[[DATE_COL, "v_kospi"]].copy()
        temp_df["v_kospi"] = temp_df["v_kospi"].replace(0, np.nan)
        
        daily_ref = temp_df.groupby(DATE_COL)["v_kospi"].mean()
        
        # 2. 결측치 보간 (Forward Fill -> Backward Fill)
        # 과거 데이터가 비어있으면 앞의 데이터로, 앞이 없으면 뒤에서 가져옴
        daily_ref = daily_ref.fillna(method='ffill').fillna(method='bfill')
        
        # 3. 변화율(Change) 피처 생성
        daily_change = daily_ref.pct_change().fillna(0)
        
        # 4. 원본 DF에 적용 (Map)
        # v_kospi 자체도 보간된 값으로 업데이트 (안정성 확보)
        df["v_kospi"] = df[DATE_COL].map(daily_ref).fillna(0)
        df["v_kospi_change"] = df[DATE_COL].map(daily_change).fillna(0)
        
        return df

    df = _apply_vkospi_advanced(df)

    def _apply_vkosdaq_advanced(df):
        """V-KOSDAQ 결측치 처리 및 변화율 피처 생성"""
        if "v_kosdaq" not in df.columns:
            return df
            
        # 0을 NaN으로 변환 후 보간
        df["v_kosdaq"] = df["v_kosdaq"].replace(0, np.nan)
        df["v_kosdaq"] = df["v_kosdaq"].fillna(method='ffill').fillna(method='bfill')
        df["v_kosdaq"] = df["v_kosdaq"].fillna(0)

        # 변화율 피처 생성
        df["v_kosdaq_change"] = df["v_kosdaq"].pct_change().fillna(0)
        df["v_kosdaq_change"] = df["v_kosdaq_change"].replace([np.inf, -np.inf], 0)
        
        return df

    df = _apply_vkosdaq_advanced(df)

    # -------------------------------------------------------------------------
    # 타겟 및 특징 선택
    # -------------------------------------------------------------------------
    task = task.lower()
    label_col = target_col if target_col else (TARGET_CLASSIFICATION if task == "classification" else TARGET_REGRESSION)
    df.dropna(subset=[label_col], inplace=True)

    exclude_cols = {
        DATE_COL, TARGET_CLASSIFICATION, TARGET_REGRESSION, 
        "종목코드", "매수가격", "매도가격", "수익_구간", "중요_손실_지표", label_col,
        # [사용자 요청] 영향력이 미미하거나 메타데이터인 피처 제외
        "real_trade", "체결강도", "방어_강도", "kosdaq", "kospi", "is_trading_month_end"
    }
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    cat_features_candidates = ["테마_섹터", "차트분석", "시장구분", "weekday"] # weekday로 교체
    cat_features = [col for col in cat_features_candidates if col in feature_cols]

    X = df[feature_cols].copy()
    y = df[label_col]
    
    # -------------------------------------------------------------------------
    # [Data Type Conversion] 타겟을 숫자형으로 변환
    # -------------------------------------------------------------------------
    # DB에서 불러온 데이터가 문자열일 수 있으므로 명시적으로 숫자형 변환
    y = pd.to_numeric(y, errors='coerce')
    
    # NaN 제거 (변환 실패한 행)
    valid_idx = y.notna()
    if not valid_idx.all():
        n_invalid = (~valid_idx).sum()
        print(f"⚠️ [Warning] 타겟 변환 실패로 {n_invalid}건 제거됨")
        X = X[valid_idx]
        y = y[valid_idx]
        df = df[valid_idx]
    
    # -------------------------------------------------------------------------
    # [Target Clipping] 극단적인 수익률 제한 (Regression 전용)
    # -------------------------------------------------------------------------
    # 모델이 "30% 대박"을 쫓다가 함정에 빠지는 것을 방지하기 위해,
    # 학습 타겟의 상한/하한을 설정하여 "안정적인 수익 패턴"을 학습하도록 유도합니다.
    if task == "regression":
        TARGET_LOWER_BOUND = -10.0  # 하한: -10% (큰 손실은 -10%로 간주)
        TARGET_UPPER_BOUND = 10.0   # 상한: +10% (큰 수익은 +10%로 간주)
        
        original_y = y.copy()
        y = y.clip(lower=TARGET_LOWER_BOUND, upper=TARGET_UPPER_BOUND)
        
        # 클리핑 통계 출력
        n_clipped_lower = (original_y < TARGET_LOWER_BOUND).sum()
        n_clipped_upper = (original_y > TARGET_UPPER_BOUND).sum()
        
        if n_clipped_lower > 0 or n_clipped_upper > 0:
            print(f"📊 [Target Clipping] 극단값 제한 적용:")
            print(f"   - 하한 초과 ({TARGET_LOWER_BOUND}% 미만): {n_clipped_lower}건")
            print(f"   - 상한 초과 ({TARGET_UPPER_BOUND}% 초과): {n_clipped_upper}건")
            print(f"   → 모델이 '안정적 수익 패턴'에 집중하도록 유도합니다.")
    
    if task == "classification": 
        y = y.astype(int)

    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    print(f"✅ 전처리 완료! (총 {len(X)}개 샘플, {len(feature_cols)}개 특성, task={task})")
    return X, y, cat_features, df

