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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def find_closest_valid_scale(ratio: float, tolerance: float = 0.05) -> float:
    """비율이 정해진 유효 스케일 팩터 후보군 중 하나에 허용 오차 이내로 수렴하는지 확인합니다.

    특히 1.0(정상) 팩터에 대해서는 상/하한가(30~35%) 수준의 변동률인 [0.65, 1.53] 범위를
    정상 범위로 넓게 인정하여 오진(False Positive)을 원천 차단합니다.
    그 외의 분할/병합 팩터(0.1, 10, 100 등)에 대해서는 엄격한 오차(5%) 내에 수렴하는 경우만 판정합니다.

    Args:
        ratio: API 가격 또는 종가 대비 수동 가격/매매가 비율 (기준 종가 / 매매 가격)
        tolerance: 스케일 팩터 대비 허용 오차율 (기본값: 0.05)

    Returns:
        float: 수렴된 유효 스케일 팩터 (스케일 정상/수렴 실패 시 1.0 반환)

    """
    if ratio <= 0:
        return 1.0

    # 1. 1.0(정상) 범위 체크: 상하한가 30~35% 수준을 반영하여 0.65 ~ 1.53은 정상 수용
    if 0.65 <= ratio <= 1.53:
        return 1.0

    # 2. 검증할 스케일 팩터 후보군 (1.0 제외)
    factors = [0.001, 0.01, 0.1, 0.2, 0.5, 2.0, 5.0, 10.0, 100.0, 1000.0]
    best_factor = 1.0
    min_error = float("inf")

    for f in factors:
        error = abs(ratio - f) / f
        if error < min_error:
            min_error = error
            best_factor = f

    # 상대 오차율이 tolerance(5%) 이내인 경우만 스케일 불일치로 승인
    if min_error <= tolerance:
        return best_factor

    return 1.0


def detect_price_scale_mismatch(
    df: pd.DataFrame,
    api_price_provider: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None,
    threshold_lower: float = 0.65,
    threshold_upper: float = 1.53,
    max_workers: int = 4,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """로컬 데이터의 종가를 기준으로 매수가격 및 매도가격의 단독 스케일 불일치를 탐지합니다.

    API 조회를 제거하고 로컬 가격 대조만 수행하므로 매우 신속하고 예외 상황에 강건합니다.
    api_price_provider 및 max_workers 등은 기존 시그니처 호환성을 위해 유지하되, 무시됩니다.

    Args:
        df: 전처리 대상 데이터프레임 (종목코드, 매수날짜, 종가, 매수가격, 매도가격 필수 포함)
        api_price_provider: 미사용 (기존 인터페이스 하위 호환용)
        threshold_lower: 미사용 (find_closest_valid_scale 내부에서 정밀 제어)
        threshold_upper: 미사용 (find_closest_valid_scale 내부에서 정밀 제어)
        max_workers: 미사용

    Returns:
        dict[tuple[str, pd.Timestamp], dict[str, float]]:
            스케일 조정이 필요한 (종목코드, 매수날짜)와 {컬럼명: 스케일팩터} 매핑 딕셔너리

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

        # 1. 매수가격 단독 불일치 체크
        if buy_price > 0:
            ratio_buy = ref_price / buy_price
            scale_buy = find_closest_valid_scale(ratio_buy)
            if scale_buy != 1.0:
                logger.warning(
                    "⚠️ [매수가 단독 오류] %s (%s) | 기준 종가: %.2f vs 매수가: %.2f | 적용 스케일: %.4f",
                    symbol, date.strftime("%Y-%m-%d"), ref_price, buy_price, scale_buy
                )
                column_scales["매수가격"] = scale_buy

        # 2. 매도가격 단독 불일치 체크
        if sell_price > 0:
            ratio_sell = ref_price / sell_price
            scale_sell = find_closest_valid_scale(ratio_sell)
            if scale_sell != 1.0:
                logger.warning(
                    "⚠️ [매도가 단독 오류] %s (%s) | 기준 종가: %.2f vs 매도가: %.2f | 적용 스케일: %.4f",
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

    Args:
        df: 원본 데이터프레임
        mismatched_symbols: 스케일 조정이 필요한 (종목코드, 매수날짜)와 {컬럼명: 스케일팩터} 매핑 딕셔너리
        price_cols: 미사용 (하위 호환성 유지)

    Returns:
        pd.DataFrame: 가격 스케일이 올바르게 보정된 데이터프레임

    """
    column_mapping = {
        "(시가)": "시가",
        "(고가)": "고가",
        "(저가)": "저가",
        "(종가)": "종가",
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
    for col in ["시가", "고가", "저가", "종가", "매수가격", "매도가격"]:
        if col in corrected_df.columns:
            corrected_df[col] = pd.to_numeric(corrected_df[col], errors="coerce").fillna(0.0)

    for (symbol, date), column_scales in mismatched_symbols.items():
        rows_mask = (corrected_df["종목코드"] == symbol) & (corrected_df["_temp_date"] == date)
        if not rows_mask.any():
            continue

        for idx, row in corrected_df[rows_mask].iterrows():
            ref_price = float(row["종가"])
            if ref_price <= 0:
                continue

            for col, default_scale_factor in column_scales.items():
                if col not in corrected_df.columns:
                    continue

                raw_val = float(row[col])
                if raw_val <= 0:
                    continue

                # 개별 행별 가격 편차 오폭 방지를 위한 실시간 스케일 검증
                if col in ["매수가격", "매도가격"]:
                    ratio = ref_price / raw_val
                    scale_factor = find_closest_valid_scale(ratio)
                else:
                    scale_factor = default_scale_factor

                if scale_factor != 1.0:
                    corrected_df.at[idx, col] = round(raw_val * scale_factor)
                    logger.info(
                        "✅ 종목 %s (%s) %s 컬럼 개별 보정 완료 (값: %.2f -> %.2f, 스케일: %.4f 적용)",
                        symbol, date.strftime("%Y-%m-%d"), col, raw_val, raw_val * scale_factor, scale_factor
                    )

    corrected_df = corrected_df.drop(columns=["_temp_date"])
    return corrected_df

