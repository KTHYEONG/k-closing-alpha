import logging
import os
import sqlite3

import numpy as np
import pandas as pd

from src import settings
from src.data.gsheet_loader import load_and_combine_sheets

logger = logging.getLogger(__name__)

# ==========================================
# [설정] 구글 시트 및 DB 정보
# ==========================================
DB_PATH = str(settings.STOCK_DB_PATH)
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
TRADE_WORKSHEETS = settings.TRADE_WORKSHEETS


def filter_valid_rows(df, required_keywords=None):
    """
    데이터프레임에서 '종목코드' 관련 열이 비어있는 행만 제거합니다.
    """
    if df is None or df.empty:
        return df

    if required_keywords is None:
        required_keywords = ["(종목코드)", "종목코드", "코드"]

    target_col = None
    for col in df.columns:
        if any(keyword in str(col) for keyword in required_keywords):
            target_col = col
            break

    if not target_col:
        logger.warning("'종목코드' 관련 컬럼을 찾지 못해 필터링을 건너뜁니다. (컬럼목록: %s)", list(df.columns))
        return df

    df[target_col] = df[target_col].replace(r"^\s*$", np.nan, regex=True)

    initial_len = len(df)
    df_valid = df.dropna(subset=[target_col])
    final_len = len(df_valid)

    if initial_len != final_len:
        logger.info("'%s' 값이 없는 %d개 행을 삭제했습니다.", target_col, initial_len - final_len)

    if target_col != "종목코드":
        df_valid = df_valid.rename(columns={target_col: "종목코드"})

    return df_valid


def sync_trade_log(conn):
    """매매일지 (Trade + Trade2) 동기화"""
    logger.info("[1/2] 매매일지(%s) 로드 중...", TRADE_WORKSHEETS)
    df_trade = load_and_combine_sheets(GOOGLE_SHEET_NAME, TRADE_WORKSHEETS)

    if df_trade is not None and not df_trade.empty:
        df_trade = filter_valid_rows(df_trade)

        if not df_trade.empty:
            if "매수날짜" in df_trade.columns:
                df_trade["매수날짜"] = pd.to_datetime(
                    df_trade["매수날짜"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
            elif "(매수날짜)" in df_trade.columns:
                df_trade["매수날짜"] = pd.to_datetime(
                    df_trade["(매수날짜)"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                df_trade = df_trade.drop(columns=["(매수날짜)"])

            if "종목코드" in df_trade.columns:
                df_trade["종목코드"] = df_trade["종목코드"].astype(str).str.zfill(6)
            
            # v-kospi 컬럼명 정리 (preprocessor.py와 일관성 유지)
            if "(v-kospi)" in df_trade.columns:
                df_trade = df_trade.rename(columns={"(v-kospi)": "v_kospi"})
                logger.info("v-kospi 컬럼 포함됨")

            # v-kosdaq 컬럼명 정리
            if "(v-kosdaq)" in df_trade.columns:
                df_trade = df_trade.rename(columns={"(v-kosdaq)": "v_kosdaq"})
                logger.info("v-kosdaq 컬럼 포함됨")

            if "매수날짜" in df_trade.columns:
                cols = df_trade.columns.tolist()
                cols.remove("매수날짜")
                cols.insert(0, "매수날짜")
                df_trade = df_trade[cols]

            df_trade.to_sql("table_trade_log", conn, if_exists="replace", index=False)
            logger.info("table_trade_log 저장 완료: %d행", len(df_trade))
            
            # Parquet 백업/학습용 저장
            try:
                from src.data.parquet_loader import save_trade_log_to_parquet
                save_trade_log_to_parquet(df_trade)
            except Exception as e:
                logger.error("Parquet 저장 중 오류 발생: %s", e)
        else:
            logger.warning("유효한 데이터가 없습니다.")
    else:
        logger.warning("매매일지 데이터를 가져오지 못했습니다.")


def sync_gsheet_data():
    """구글 시트 데이터 전체 동기화 (Parquet & SQLite 저장).

    테마/섹터 동기화는 코드_테마_DB 시트 폐지(수동 기입 -> theme_resolver 자동화)로
    제거됨 -- 매매일지(Trade/Trade2)만 남은 시트 소스.
    """
    logger.info("구글 시트 데이터 전체 동기화 시작 (Parquet & SQLite)...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        sync_trade_log(conn)
    except Exception as e:
        logger.error("동기화 중 오류 발생: %s", e)
    finally:
        conn.close()
        logger.info("동기화 종료")


# 레거시 명칭 호환성 alias
sync_gsheet_to_sqlite = sync_gsheet_data


if __name__ == "__main__":
    sync_gsheet_data()
