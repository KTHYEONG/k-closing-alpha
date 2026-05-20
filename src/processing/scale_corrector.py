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


def detect_price_scale_mismatch(
    df: pd.DataFrame,
    api_price_provider: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame],
    threshold_lower: float = 0.65,
    threshold_upper: float = 1.50,
    max_workers: int = 4,
) -> dict[tuple[str, pd.Timestamp], float]:
    """수동 기입 가격과 API 가격 간의 스케일 불일치를 (종목코드, 매수날짜) 튜플 단위로 병렬 탐지합니다.

    Args:
        df: 전처리 대상 데이터프레임 (종목코드, 매수날짜, 시가, 고가, 저가, 종가 컬럼 필수 포함)
        api_price_provider: 종목코드, 시작일, 종료일을 받아 API 가격 데이터프레임(date, close)을 반환하는 함수
        threshold_lower: 정상 스케일 비율 하한선 (기본값: 0.65)
        threshold_upper: 정상 스케일 비율 상한선 (기본값: 1.50)
        max_workers: 병렬 조회용 스레드 개수 (기본값: 4)

    Returns:
        dict[tuple[str, pd.Timestamp], float]: 스케일 조정이 필요한 (종목코드, 매수날짜) 와 스케일 팩터 매핑 딕셔너리

    """
    mismatched_symbols: dict[tuple[str, pd.Timestamp], float] = {}

    required_cols = ["종목코드", "매수날짜", "시가", "고가", "저가", "종가"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("필수 컬럼이 누락되었습니다: %s", missing_cols)
        return mismatched_symbols

    # 1. 로컬 데이터 대조를 통한 스케일 의심 종목 1차 사전 필터링 (Heuristics Pre-filtering)
    # 수동 기입 가격(매수가격/매도가격 등)과 수정주가(시가/종가 등)의 동일 행 비율이 임계치를 벗어나는 종목만 표적 선별
    df_copy = df.copy()
    
    # 구글 시트 직접 동기화 직후 괄호 포함 헤더 대응을 위한 표준화
    column_mapping = {
        "(종목코드)": "종목코드",
        "(매수날짜)": "매수날짜",
        "(시가)": "시가",
        "(고가)": "고가",
        "(저가)": "저가",
        "(종가)": "종가",
        "(매수 가격)": "매수가격",
        "매수 가격": "매수가격",
        "(매도 가격)": "매도가격",
        "매도 가격": "매도가격"
    }
    df_copy = df_copy.rename(columns={k: v for k, v in column_mapping.items() if k in df_copy.columns})
    
    df_copy["종가"] = pd.to_numeric(df_copy["종가"], errors="coerce")
    
    # 매수가격 컬럼이 존재할 시 대조 수행
    compare_col = "매수가격" if "매수가격" in df_copy.columns else "종가"
    df_copy[compare_col] = pd.to_numeric(df_copy[compare_col], errors="coerce")
    
    valid_rows = df_copy[(df_copy["종가"] > 0) & (df_copy[compare_col] > 0)].copy()
    valid_rows["ratio"] = valid_rows[compare_col] / valid_rows["종가"]
    
    # 스케일 불일치 하한선 및 상한선 기준 의심 필터 적용
    suspect_rows = valid_rows[(valid_rows["ratio"] < threshold_lower) | (valid_rows["ratio"] > threshold_upper)]
    symbols = suspect_rows["종목코드"].dropna().unique()
    
    total_raw_symbols = len(df_copy["종목코드"].dropna().unique())
    logger.info(
        "📈 [1차 로컬 필터링] 전체 %d개 종목 중 스케일 오차 의심 종목 %d개 선별 완료 (API 호출 %.1f%% 절감)",
        total_raw_symbols,
        len(symbols),
        (1 - len(symbols) / total_raw_symbols) * 100 if total_raw_symbols > 0 else 0
    )
    logger.info("📈 스케일 분석 시작: 종목 %d개 (스레드: %d)", len(symbols), max_workers)

    def _check_single_symbol(symbol: str) -> tuple[str, list[tuple[str, pd.Timestamp, float]]]:
        symbol_df = df[df["종목코드"] == symbol].copy()
        if symbol_df.empty:
            return symbol, []

        # 수동 가격 결측 및 Zero 필터링
        symbol_df["종가"] = pd.to_numeric(symbol_df["종가"], errors="coerce")
        valid_manual = symbol_df[symbol_df["종가"] > 0].copy()
        if valid_manual.empty:
            return symbol, []

        # 매수날짜를 Timestamp 형태로 표준화 및 정규화(normalize)
        valid_manual["매수날짜"] = pd.to_datetime(valid_manual["매수날짜"]).dt.normalize()
        unique_dates = valid_manual["매수날짜"].dropna().unique()
        if len(unique_dates) == 0:
            return symbol, []

        # API 호출 최소화를 위해 최소 매수날짜부터 최대 매수날짜까지 전체 기간을 단 1회의 쿼리로 일괄 조회 (세그먼트 분할 오버헤드 제거)
        unique_dates = sorted(unique_dates)
        seg_start = unique_dates[0]
        seg_end = unique_dates[-1]

        try:
            combined_api_df = api_price_provider(symbol, seg_start, seg_end)
        except Exception as e:
            logger.debug("종목 %s API 조회 중 오류 (%s ~ %s): %s", symbol, seg_start, seg_end, e)
            return symbol, []

        if combined_api_df is None or combined_api_df.empty:
            return symbol, []
        combined_api_df["date"] = pd.to_datetime(combined_api_df["date"]).dt.normalize()
        combined_api_df["close"] = pd.to_numeric(combined_api_df["close"], errors="coerce")
        combined_api_df = combined_api_df.dropna(subset=["date", "close"])

        # 수동 가격과 API 가격 날짜 기준 머지
        merged = pd.merge(
            valid_manual[["매수날짜", "종가"]].rename(columns={"매수날짜": "date", "종가": "manual_close"}),
            combined_api_df[["date", "close"]].rename(columns={"close": "api_close"}),
            on="date",
            how="inner"
        )

        if merged.empty:
            return symbol, []

        mismatches: list[tuple[str, pd.Timestamp, float]] = []
        for _, row in merged.iterrows():
            manual_close = float(row["manual_close"])
            api_close = float(row["api_close"])
            if manual_close <= 0 or api_close <= 0:
                continue

            ratio = api_close / manual_close
            if ratio < threshold_lower or ratio > threshold_upper:
                d = row["date"]
                logger.debug(
                    "⚠️ 스케일 오류 감지: %s (%s) | API: %.2f vs 수동: %.2f | 비율: %.4f",
                    symbol, d.strftime("%Y-%m-%d"), api_close, manual_close, ratio
                )
                mismatches.append((symbol, d, ratio))

        return symbol, mismatches

    # ThreadPoolExecutor 병렬 검사 및 강제 종료 예외 처리
    import time
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(_check_single_symbol, sym): sym for sym in symbols}

    total_symbols = len(symbols)
    completed_count = 0
    start_time = time.time()

    try:
        # Windows 터미널에서 Ctrl+C 즉각 반응을 위해 0.1초씩 쉬면서 폴링하며 실시간 진행률 로깅
        while futures:
            done_futures = [fut for fut in futures if fut.done()]
            for fut in done_futures:
                symbol = futures.pop(fut)
                completed_count += 1

                # 10개 단위 또는 최종 완료 시 진행 상황 로그 출력
                if completed_count % 10 == 0 or completed_count == total_symbols:
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_symbols - completed_count) / speed if speed > 0 else 0
                    logger.info(
                        "⏳ 스케일 분석 진행 중: %d/%d (%d%%) 완료 | 속도: %.1f종목/초 | 남은시간: %d초",
                        completed_count,
                        total_symbols,
                        int(completed_count / total_symbols * 100),
                        speed,
                        int(eta),
                    )

                if not fut.cancelled():
                    try:
                        _, mismatch_list = fut.result()
                        if mismatch_list:
                            for sym, dt, ratio in mismatch_list:
                                mismatched_symbols[(sym, dt)] = ratio
                    except Exception as e:
                        logger.debug("종목 %s 처리 중 오류: %s", symbol, e)

            if futures:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.warning("\n🛑 사용자에 의해 강제 종료 요청됨. 프로세스를 즉시 강제 중단합니다...")
        for fut in futures:
            fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        import os
        os._exit(1)
    finally:
        executor.shutdown(wait=False)

    return mismatched_symbols


def apply_scale_correction(
    df: pd.DataFrame,
    mismatched_symbols: dict[tuple[str, pd.Timestamp], float],
    price_cols: list[str] | None = None
) -> pd.DataFrame:
    """탐지된 스케일 불일치 종목들의 특정 날짜 가격 컬럼을 보정합니다.

    Args:
        df: 원본 데이터프레임
        mismatched_symbols: 스케일 조정이 필요한 (종목코드, 매수날짜)와 스케일 팩터 매핑 딕셔너리
        price_cols: 보정할 가격 관련 컬럼 목록 (기본값: 시가, 고가, 저가, 종가, 매수가격, 매도가격)

    Returns:
        pd.DataFrame: 가격 스케일이 올바르게 보정된 데이터프레임

    """
    if price_cols is None:
        price_cols = ["시가", "고가", "저가", "종가", "매수가격", "매도가격"]

    corrected_df = df.copy()
    if "_temp_date" not in corrected_df.columns:
        corrected_df["_temp_date"] = pd.to_datetime(corrected_df["매수날짜"]).dt.normalize()

    for (symbol, date), scale_factor in mismatched_symbols.items():
        symbol_mask = (corrected_df["종목코드"] == symbol) & (corrected_df["_temp_date"] == date)
        if not symbol_mask.any():
            continue

        # 스케일 팩터 적용 (가격 컬럼에 스케일 비율 곱해주기)
        for col in price_cols:
            if col in corrected_df.columns:
                corrected_df.loc[symbol_mask, col] = (
                     pd.to_numeric(corrected_df.loc[symbol_mask, col], errors="coerce") * scale_factor
                )
        
        logger.debug(
            "✅ 종목 %s (%s) 보정 완료 (스케일 팩터: %.4f 적용)",
            symbol, date.strftime("%Y-%m-%d"), scale_factor
        )

    corrected_df = corrected_df.drop(columns=["_temp_date"])
    return corrected_df
