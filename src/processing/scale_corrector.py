"""주식 가격 데이터 스케일 불일치 탐지 및 보정 모듈.

수동으로 스프레드시트에 기입한 OHLC 가격 데이터와 API(pykrx/KIS)로 수집한 수정주가 데이터 간의
스케일 불일치(액면분할/병합, 단위 차이 등)를 자동으로 탐지하고 보정하는 기능을 제공합니다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

# 로깅 설정
logger = logging.getLogger(__name__)


def find_closest_valid_scale(ratio: float, tolerance: float = 0.35) -> float:
    """비율이 정해진 유효 스케일 팩터 후보군 중 하나에 허용 오차 이내로 수렴하는지 확인합니다.

    특히 1.0(정상) 팩터에 대해서는 상/하한가(30~35%) 수준의 변동률인 [0.65, 1.53] 범위를
    정상 범위로 넓게 인정하여 오진(False Positive)을 원천 차단합니다.
    그 외의 분할/병합 팩터(0.1, 10, 100 등)에 대해서는 유연한 오차(35%) 내에 수렴하는 경우만 판정합니다.
    자연스러운 주가 변동성(반토막, 2배, 5배 등) 범위와 겹치는 팩터(0.2, 0.5, 2.0, 5.0)는
    오폭을 방지하기 위해 후보군에서 명확히 배제되었습니다.

    Args:
        ratio: API 가격 또는 종가 대비 수동 가격/매매가 비율 (기준 종가 / 매매 가격)
        tolerance: 스케일 팩터 대비 허용 오차율 (기본값: 0.35)

    Returns:
        float: 수렴된 유효 스케일 팩터 (스케일 정상/수렴 실패 시 1.0 반환)

    Time Complexity: O(1) - 후보군이 고정 크기이므로 상수 시간 소요.
    Space Complexity: O(1) - 추가 메모리 할당이 최소화됨.

    """
    if ratio <= 0:
        return 1.0

    # 1.0 (정상 스케일)은 35% 오차율을 넓게 인정
    if (1.0 - 0.35) <= ratio <= (1.0 + 0.53):
        return 1.0

    # 액면분할/병합 또는 10배/100배 기입 오류 등의 스케일 팩터 후보군
    # 주가 2배, 5배 등의 자연스러운 변동률(0.2, 0.5, 2.0, 5.0)은 완벽히 제외
    factors = [0.001, 0.01, 0.04, 0.05, 0.1, 10.0, 20.0, 25.0, 100.0, 1000.0]

    for f in factors:
        lower_bound = f * (1.0 - tolerance)
        upper_bound = f * (1.0 + tolerance)
        if lower_bound <= ratio <= upper_bound:
            return f

    return 1.0


def detect_price_scale_mismatch(
    df: pd.DataFrame,
    api_price_provider: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None,
    threshold_lower: float = 0.65,
    threshold_upper: float = 1.53,
    max_workers: int = 4,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """로컬 데이터의 종가를 기준으로 매수가격 및 매도가격의 단독 스케일 불일치를 탐지합니다.

    매수 가격과 매도 가격을 세트(Set)로 취급하여 매수 가격 스케일이 정상이면 매도 가격 검사도
    안전하게 생략하고 정상 판정함으로써 주가 급등락에 따른 매도가격 오진을 완벽히 방지합니다.
    또한 보정 후 가격이 최소 주가(100원) 미만으로 내려가는 극단적 오진은 자동 배제됩니다.

    Args:
        df: 전처리 대상 데이터프레임 (종목코드, 매수날짜, 종가, 매수가격, 매도가격 필수 포함)
        api_price_provider: 미사용 (기존 인터페이스 하위 호환용)
        threshold_lower: 미사용 (find_closest_valid_scale 내부에서 정밀 제어)
        threshold_upper: 미사용 (find_closest_valid_scale 내부에서 정밀 제어)
        max_workers: 미사용

    Returns:
        dict[tuple[str, pd.Timestamp], dict[str, float]]:
            스케일 조정이 필요한 (종목코드, 매수날짜)와 {컬럼명: 스케일팩터} 매핑 딕셔너리

    Time Complexity: O(N) - N은 매매일지 행 수. 각 행에 대해 상수 시간 연산 수행.
    Space Complexity: O(M) - M은 탐지된 스케일 오류 개수. 오류 매핑 정보만 임시 저장.

    """
    mismatched_symbols: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}

    required_cols = ["종목코드", "매수날짜", "종가"]
    
    # 구글 시트 직접 동기화 직후 괄호 포함 헤더 대응을 위한 표준화
    column_mapping = {
        "(종목코드)": "종목코드",
        "(매수날짜)": "매수날짜",
        "(종가)": "종가",
        "(매수 가격)": "매수가격",
        "매수 가격": "매수가격",
        "(매도 가격)": "매도가격",
        "매도 가격": "매도가격"
    }
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={k: v for k, v in column_mapping.items() if k in df_copy.columns})
    
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    if missing_cols:
        logger.error("필수 컬럼이 누락되었습니다: %s", missing_cols)
        return mismatched_symbols

    # 종목코드 표준화 및 날짜 전처리
    df_copy["종목코드"] = df_copy["종목코드"].astype(str).str.strip().str.zfill(6)
    df_copy["매수날짜"] = pd.to_datetime(df_copy["매수날짜"]).dt.normalize()

    # 가격 컬럼 파싱 및 수치 정형화
    for col in ["종가", "매수가격", "매도가격"]:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0)
        else:
            df_copy[col] = 0.0

    logger.info("📈 [단독 보정 검사] 로컬 종가 대비 매매 가격 스케일 정밀 진단 시작...")

    for _, row in df_copy.iterrows():
        symbol = str(row["종목코드"])
        date = row["매수날짜"]
        ref_price = float(row["종가"])
        buy_price = float(row["매수가격"])
        sell_price = float(row["매도가격"])

        if pd.isna(date) or not symbol or ref_price <= 0:
            continue

        column_scales: dict[str, float] = {}

        # 1. 매수가격이 존재하는 경우: 매수가격을 지배적인 기준으로 삼고 동기화 진단
        if buy_price > 0:
            ratio_buy = ref_price / buy_price
            scale_buy = find_closest_valid_scale(ratio_buy)
            if scale_buy != 1.0:
                # 안전장치: 보정된 매수가격이 상식적인 최소 금액(100원) 이상 확보될 때만 승인
                if round(buy_price * scale_buy) >= 100:
                    logger.warning(
                        "⚠️ [매수가 스케일 오류 감지] %s (%s) | 기준 종가: %.2f vs 매수가: %.2f | 적용 스케일: %.4f",
                        symbol, date.strftime("%Y-%m-%d"), ref_price, buy_price, scale_buy
                    )
                    # 매수 가격과 매도 가격은 작성 스케일 단위가 동일하므로 세트로 일괄 적용
                    column_scales["매수가격"] = scale_buy
                    if sell_price > 0:
                        # 매도가 보정 결과도 100원 미만으로 떨어지는지 2중 안전 검사
                        if round(sell_price * scale_buy) >= 100:
                            column_scales["매도가격"] = scale_buy
            else:
                # 매수가격이 정상이면 매도가격도 정상 스케일로 기입되었다고 신뢰하여 단독 검사를 완전히 생략
                pass

        # 2. 매수가격이 없고 매도가격만 단독으로 기입된 특수 경우에 한해서만 매도가격 단독 체크
        elif sell_price > 0:
            ratio_sell = ref_price / sell_price
            scale_sell = find_closest_valid_scale(ratio_sell)
            if scale_sell != 1.0:
                # 안전장치: 보정된 매도가격이 최소 100원 이상 확보될 때만 승인
                if round(sell_price * scale_sell) >= 100:
                    logger.warning(
                        "⚠️ [매도가 단독 오류 감지] %s (%s) | 기준 종가: %.2f vs 매도가: %.2f | 적용 스케일: %.4f",
                        symbol, date.strftime("%Y-%m-%d"), ref_price, sell_price, scale_sell
                    )
                    column_scales["매도가격"] = scale_sell

        if column_scales:
            if (symbol, date) in mismatched_symbols:
                mismatched_symbols[(symbol, date)].update(column_scales)
            else:
                mismatched_symbols[(symbol, date)] = column_scales

    logger.info("📊 [단독 보정 진단 완료] 총 %d건의 매매 가격 스케일 오류를 진단하였습니다.", len(mismatched_symbols))
    return mismatched_symbols


def apply_scale_correction(
    df: pd.DataFrame,
    mismatched_symbols: dict[tuple[str, pd.Timestamp], dict[str, float]],
    price_cols: list[str] | None = None
) -> pd.DataFrame:
    """탐지된 스케일 불일치 종목들의 특정 날짜 매매 가격 컬럼을 보정합니다.

    OHLCV 시장 가격 데이터는 철저히 배제하고, 오직 매수가격과 매도가격만 안전 가드레일을 
    적용하여 동기화 보정함으로써 데이터 왜곡 및 오폭을 원천 방지합니다.

    Args:
        df: 원본 데이터프레임
        mismatched_symbols: 스케일 조정이 필요한 (종목코드, 매수날짜)와 {컬럼명: 스케일팩터} 매핑 딕셔너리
        price_cols: 미사용 (하위 호환성 유지)

    Returns:
        pd.DataFrame: 가격 스케일이 올바르게 보정된 데이터프레임

    Time Complexity: O(N) - N은 데이터프레임 행 수. 각 행에 대해 상수 시간 내 보정 연산 수행.
    Space Complexity: O(N) - 수정된 데이터프레임의 사본 반환.

    """
    column_mapping = {
        "(매수 가격)": "매수가격",
        "매수 가격": "매수가격",
        "(매도 가격)": "매도가격",
        "매도 가격": "매도가격"
    }

    corrected_df = df.copy()
    corrected_df = corrected_df.rename(columns={k: v for k, v in column_mapping.items() if k in corrected_df.columns})

    if "_temp_date" not in corrected_df.columns:
        corrected_df["_temp_date"] = pd.to_datetime(corrected_df["매수날짜"]).dt.normalize()

    # Pandas Arrow String 형변환 에러 방지를 위해 미리 수치형 형변환 수행
    for col in ["매수가격", "매도가격"]:
        if col in corrected_df.columns:
            corrected_df[col] = pd.to_numeric(corrected_df[col], errors="coerce").fillna(0.0)

    # 행별로 종가(종가 컬럼)는 대표 스케일 판단의 기준으로만 읽기 위해 로드
    temp_df = df.copy()
    close_col_name = "종가"
    if "(종가)" in temp_df.columns:
        close_col_name = "(종가)"
    if close_col_name in temp_df.columns:
        temp_df[close_col_name] = pd.to_numeric(temp_df[close_col_name], errors="coerce").fillna(0.0)
    else:
        temp_df[close_col_name] = 0.0

    for (symbol, date), column_scales in mismatched_symbols.items():
        rows_mask = (corrected_df["종목코드"] == symbol) & (corrected_df["_temp_date"] == date)
        if not rows_mask.any():
            continue

        for idx, row in corrected_df[rows_mask].iterrows():
            ref_price = float(temp_df.loc[idx, close_col_name]) if idx in temp_df.index else 0.0
            if ref_price <= 0:
                continue

            # 1. 행별 대표 스케일 팩터 산출 (매수가격/매도가격 기반 세트 동기화)
            representative_scale = 1.0
            
            # 매수가격 기준 진단
            if "매수가격" in corrected_df.columns:
                buy_val = float(row["매수가격"])
                if buy_val > 0:
                    scale_buy = find_closest_valid_scale(ref_price / buy_val)
                    if scale_buy != 1.0:
                        representative_scale = scale_buy

            # 매수가격이 없는 특수 경우에만 매도가격 기준 진단
            if representative_scale == 1.0 and "매도가격" in corrected_df.columns:
                sell_val = float(row["매도가격"])
                if sell_val > 0:
                    scale_sell = find_closest_valid_scale(ref_price / sell_val)
                    if scale_sell != 1.0:
                        representative_scale = scale_sell

            # 진단 불가 시 딕셔너리에 등록된 기본값 차용
            if representative_scale == 1.0:
                valid_factors = [f for f in column_scales.values() if f != 1.0]
                if valid_factors:
                    representative_scale = valid_factors[0]
                else:
                    representative_scale = column_scales.get("종가", 1.0)

            if representative_scale == 1.0:
                continue

            # 2. 동기식 대표 스케일 일괄 적용 (OHLCV를 철저히 배제하고 매수가격/매도가격만 안전하게 보정)
            for col in ["매수가격", "매도가격"]:
                if col not in corrected_df.columns:
                    continue

                raw_val = float(row[col])
                if raw_val <= 0:
                    continue

                # 안전장치: 보정 후 가격이 극단적으로 낮아지는 현상 (예: 100원 미만) 방지
                corrected_val = round(raw_val * representative_scale)
                if corrected_val < 100:
                    logger.warning(
                        "⚠️ [보정 스킵] 종목 %s (%s) %s 보정 후 가격(%.2f)이 100원 미만으로 산출되어 스케일 오진으로 간주하고 취소합니다.",
                        symbol, date.strftime("%Y-%m-%d"), col, corrected_val
                    )
                    continue

                corrected_df.at[idx, col] = corrected_val
                logger.info(
                    "✅ 종목 %s (%s) %s 컬럼 일괄 보정 완료 (값: %.2f -> %.2f, 스케일: %.4f 적용)",
                    symbol, date.strftime("%Y-%m-%d"), col, raw_val, corrected_val, representative_scale
                )

    corrected_df = corrected_df.drop(columns=["_temp_date"])
    return corrected_df

