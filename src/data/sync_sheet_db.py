import logging
import os
import sqlite3
import pandas as pd
import numpy as np

from src import settings
from src.data.gsheet_loader import load_and_combine_sheets, load_data_from_gsheet

logger = logging.getLogger(__name__)

# ==========================================
# [설정] 구글 시트 및 DB 정보
# ==========================================
DB_PATH = str(settings.STOCK_DB_PATH)
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
TRADE_WORKSHEETS = settings.TRADE_WORKSHEETS
THEME_WORKSHEET = settings.THEME_WORKSHEET_NAME


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
        else:
            logger.warning("유효한 데이터가 없습니다.")
    else:
        logger.warning("매매일지 데이터를 가져오지 못했습니다.")


def sync_theme_only(conn=None):
    """테마 정보 (코드_테마_DB)만 동기화"""
    should_close = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        should_close = True

    logger.info("테마정보(%s) 동기화 중...", THEME_WORKSHEET)
    try:
        key_path = str(settings.GOOGLE_KEY_PATH)
        # 1. 전체 데이터 로드 (API 호출 최소화)
        all_values = load_data_from_gsheet(key_path, GOOGLE_SHEET_NAME, THEME_WORKSHEET)
        # load_data_from_gsheet가 내부적으로 df를 반환하므로, 
        # API 레벨에서의 최적화는 gsheet_loader의 append_stocks_to_gsheet와 유사하게 
        # 직접 gspread를 호출하는 것이 좋지만, 기존 구조를 유지하며 효율적으로 처리합니다.

        if all_values is not None and not all_values.empty:
            df_theme = filter_valid_rows(all_values)

            if not df_theme.empty:
                if "종목코드" in df_theme.columns:
                    df_theme["종목코드"] = (
                        df_theme["종목코드"]
                        .astype(str)
                        .str.strip()
                        .str.split(".")
                        .str[0]
                        .str.zfill(6)
                    )
                
                if "테마/섹터" in df_theme.columns:
                    df_theme = df_theme.rename(columns={"테마/섹터": "테마"})

                df_theme.to_sql("table_theme", conn, if_exists="replace", index=False)
                logger.info("table_theme 저장 완료: %d행", len(df_theme))
            else:
                logger.warning("유효한 테마 데이터가 없습니다.")
        else:
            logger.warning("테마 데이터를 가져오지 못했습니다.")
    finally:
        if should_close:
            conn.close()


def sync_gsheet_to_sqlite():
    logger.info("구글 시트 데이터 전체 동기화 시작...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        sync_trade_log(conn)
        sync_theme_only(conn)
    except Exception as e:
        logger.error("동기화 중 오류 발생: %s", e)
    finally:
        conn.close()
        logger.info("동기화 종료")


if __name__ == "__main__":
    sync_gsheet_to_sqlite()
