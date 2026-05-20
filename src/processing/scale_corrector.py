"""주식 가격 데이터 스케일 불일치 탐지 및 보정 모듈.

수동으로 스프레드시트에 기입한 OHLC 가격 데이터와 API(pykrx/KIS)로 수집한 수정주가 데이터 간의
스케일 불일치(액면분할/병합, 단위 차이 등)를 자동으로 탐지하고 보정하는 기능을 제공합니다.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def detect_price_scale_mismatch(
    df: pd.DataFrame,
    api_price_provider,
    threshold_lower: float = 0.9,
    threshold_upper: float = 1.1,
    max_workers: int = 4,
) -> Dict[str, float]:
    """수동 기입 가격과 API 가격 간의 스케일 불일치를 병렬(Multi-threading)로 탐지합니다.

    Args:
        df: 전처리 대상 데이터프레임 (종목코드, 매수날짜, 시가, 고가, 저가, 종가 컬럼 필수 포함)
        api_price_provider: 종목코드, 시작일, 종료일을 받아 API 가격 데이터프레임(date, close)을 반환하는 함수
        threshold_lower: 정상 스케일 비율 하한선 (기본값: 0.9)
        threshold_upper: 정상 스케일 비율 상한선 (기본값: 1.1)
        max_workers: 병렬 조회용 스레드 개수 (기본값: 4, KRX API 레이트 리밋 방지용 안정적 한도)

    Returns:
        Dict[str, float]: 스케일 조정이 필요한 종목코드와 스케일 팩터(API 가격 / 수동 가격) 매핑 딕셔너리
    """
    mismatched_symbols: Dict[str, float] = {}

    required_cols = ["종목코드", "매수날짜", "시가", "고가", "저가", "종가"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("필수 컬럼이 누락되었습니다: %s", missing_cols)
        return mismatched_symbols

    # 고유 종목 목록 추출
    symbols = df["종목코드"].dropna().unique()
    logger.info("📈 스케일 분석 시작: 종목 %d개 (스레드: %d)", len(symbols), max_workers)

    def _check_single_symbol(symbol: str) -> tuple[str, float | None]:
        symbol_df = df[df["종목코드"] == symbol].copy()
        if symbol_df.empty:
            return symbol, None

        # 수동 가격 결측 및 Zero 필터링
        symbol_df["종가"] = pd.to_numeric(symbol_df["종가"], errors="coerce")
        valid_manual = symbol_df[symbol_df["종가"] > 0]
        if valid_manual.empty:
            return symbol, None

        start_date = pd.to_datetime(valid_manual["매수날짜"].min())
        end_date = pd.to_datetime(valid_manual["매수날짜"].max())

        # API를 통한 실 가격 데이터 조회
        try:
            api_df = api_price_provider(symbol, start_date, end_date)
            if api_df is None or api_df.empty:
                return symbol, None
            
            # API 가격 컬럼 표준화
            api_df = api_df.copy()
            api_df["date"] = pd.to_datetime(api_df["date"])
            api_df["close"] = pd.to_numeric(api_df["close"], errors="coerce")
            api_df = api_df.dropna(subset=["date", "close"])
        except Exception as e:
            logger.debug("종목 %s API 조회 오류: %s", symbol, e)
            return symbol, None

        # 수동 가격과 API 가격 날짜 기준 머지
        merged = pd.merge(
            valid_manual[["매수날짜", "종가"]].rename(columns={"매수날짜": "date", "종가": "manual_close"}),
            api_df[["date", "close"]].rename(columns={"close": "api_close"}),
            on="date",
            how="inner"
        )

        if len(merged) < 1:
            return symbol, None

        # 비율 계산 (API 종가 / 수동 종가)
        merged["ratio"] = merged["api_close"] / merged["manual_close"]
        median_ratio = float(merged["ratio"].median())

        # 스케일 불일치 여부 판단
        if median_ratio < threshold_lower or median_ratio > threshold_upper:
            # 매수날짜 문자열 생성 (단일 일자는 하나만 표현, 여러 일자는 범위 형식으로 표현)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            date_str = start_str if start_str == end_str else f"{start_str} ~ {end_str}"

            logger.debug(
                "⚠️ 스케일 오류: %s (%s) | 비율: %.4f",
                symbol, date_str, median_ratio
            )
            return symbol, median_ratio
        return symbol, None

    # ThreadPoolExecutor 병렬 검사 및 강제 종료 예외 처리
    from concurrent.futures import ThreadPoolExecutor
    import time

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(_check_single_symbol, sym): sym for sym in symbols}

    try:
        # Windows 터미널에서 Ctrl+C 즉각 반응을 위해 0.1초씩 쉬면서 폴링
        while any(not fut.done() for fut in futures):
            time.sleep(0.1)

        # 결과 수집
        for future, symbol in futures.items():
            if future.done() and not future.cancelled():
                try:
                    _, ratio = future.result()
                    if ratio is not None:
                        mismatched_symbols[symbol] = ratio
                except Exception as e:
                    logger.debug("종목 %s 처리 중 오류: %s", symbol, e)
    except KeyboardInterrupt:
        logger.warning("\n🛑 사용자에 의해 강제 종료 요청됨. 병렬 처리를 즉시 취소 및 중단합니다...")
        for fut in futures:
            fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise KeyboardInterrupt
    finally:
        executor.shutdown(wait=False)

    return mismatched_symbols


def apply_scale_correction(
    df: pd.DataFrame,
    mismatched_symbols: Dict[str, float],
    price_cols: List[str] = None
) -> pd.DataFrame:
    """탐지된 스케일 불일치 종목들에 대해 가격 컬럼을 보정합니다.

    Args:
        df: 원본 데이터프레임
        mismatched_symbols: 스케일 조정이 필요한 종목코드와 스케일 팩터 매핑 딕셔너리
        price_cols: 보정할 가격 관련 컬럼 목록 (기본값: 시가, 고가, 저가, 종가, 매수가격, 매도가격)

    Returns:
        pd.DataFrame: 가격 스케일이 올바르게 보정된 데이터프레임
    """
    if price_cols is None:
        price_cols = ["시가", "고가", "저가", "종가", "매수가격", "매도가격"]

    corrected_df = df.copy()

    for symbol, scale_factor in mismatched_symbols.items():
        symbol_mask = corrected_df["종목코드"] == symbol
        if not symbol_mask.any():
            continue

        # 스케일 팩터 적용 (가격 컬럼에 스케일 비율 곱해주기)
        for col in price_cols:
            if col in corrected_df.columns:
                corrected_df.loc[symbol_mask, col] = (
                    pd.to_numeric(corrected_df.loc[symbol_mask, col], errors="coerce") * scale_factor
                )
        
        # EMA 등의 기술적 지표도 보정이 필요한 경우
        ema_cols = ["ema5", "ema10", "ema20"]
        for col in ema_cols:
            if col in corrected_df.columns:
                corrected_df.loc[symbol_mask, col] = (
                    pd.to_numeric(corrected_df.loc[symbol_mask, col], errors="coerce") * scale_factor
                )

        logger.debug("✅ 종목 %s 보정 완료 (스케일 팩터: %.4f 적용)", symbol, scale_factor)

    return corrected_df
