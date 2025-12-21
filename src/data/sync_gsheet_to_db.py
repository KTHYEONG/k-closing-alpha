import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 1. 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

# 2. gsheet_loader 모듈 임포트
try:
    from src.data.gsheet_loader import load_and_combine_sheets, load_data_from_gsheet
except ImportError:
    print("Error: 'src.data.gsheet_loader'를 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit(1)

# .env 로드
load_dotenv(os.path.join(project_root, ".env"))

# ==========================================
# [설정] 구글 시트 및 DB 정보
# ==========================================
DB_PATH = os.path.join(project_root, "data", "stock.db")
GOOGLE_SHEET_NAME = "Stock"
TRADE_WORKSHEETS = ["Trade", "Trade2"]
THEME_WORKSHEET = "코드_테마_DB"


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
        print(
            f"    ⚠️ '종목코드' 관련 컬럼을 찾지 못해 필터링을 건너뜁니다. (컬럼목록: {list(df.columns)})"
        )
        return df

    df[target_col] = df[target_col].replace(r"^\s*$", np.nan, regex=True)

    initial_len = len(df)
    df_valid = df.dropna(subset=[target_col])
    final_len = len(df_valid)

    if initial_len != final_len:
        print(
            f"    🧹 '{target_col}' 값이 없는 {initial_len - final_len}개 행을 삭제했습니다."
        )

    if target_col != "종목코드":
        df_valid = df_valid.rename(columns={target_col: "종목코드"})

    return df_valid


def sync_gsheet_to_sqlite():
    print(f"🚀 구글 시트 데이터 동기화 시작 (매수날짜 1열 고정)...")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    print(f" -> DB 연결 성공: {DB_PATH}")

    try:
        # ==================================================
        # 1. 매매일지 (Trade + Trade2) 동기화
        # ==================================================
        print(f" -> [1/2] 매매일지({TRADE_WORKSHEETS}) 로드 중...")

        df_trade = load_and_combine_sheets(GOOGLE_SHEET_NAME, TRADE_WORKSHEETS)

        if df_trade is not None and not df_trade.empty:
            df_trade = filter_valid_rows(df_trade)

            if not df_trade.empty:
                # 1) 날짜 포맷팅 및 컬럼명 통일
                if "매수날짜" in df_trade.columns:
                    df_trade["매수날짜"] = pd.to_datetime(
                        df_trade["매수날짜"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                elif "(매수날짜)" in df_trade.columns:
                    df_trade["매수날짜"] = pd.to_datetime(
                        df_trade["(매수날짜)"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                    df_trade = df_trade.drop(columns=["(매수날짜)"])

                # 2) 종목코드 6자리 맞추기
                if "종목코드" in df_trade.columns:
                    df_trade["종목코드"] = df_trade["종목코드"].astype(str).str.zfill(6)

                # 컬럼 재정렬: '매수날짜'를 무조건 맨 앞으로 이동
                if "매수날짜" in df_trade.columns:
                    cols = df_trade.columns.tolist()
                    cols.remove("매수날짜")
                    cols.insert(0, "매수날짜")  # 0번 인덱스(맨 앞)에 삽입
                    df_trade = df_trade[cols]

                # DB 저장
                df_trade.to_sql(
                    "table_trade_log", conn, if_exists="replace", index=False
                )
                print(
                    f"    ✅ table_trade_log 저장 완료: {len(df_trade)}행 (매수날짜 1열 정렬됨)"
                )
            else:
                print("    ⚠️ 유효한 데이터가 없습니다 (모든 행에 종목코드가 없음).")
        else:
            print("    ⚠️ 매매일지 데이터를 가져오지 못했습니다.")

        # ==================================================
        # 2. 테마 정보 (코드_테마_DB) 동기화
        # ==================================================
        print(f" -> [2/2] 테마정보({THEME_WORKSHEET}) 로드 중...")

        key_path = os.getenv("GSPREAD_KEY_PATH")
        if key_path and not os.path.isabs(key_path):
            key_path = os.path.join(project_root, key_path)

        df_theme = load_data_from_gsheet(key_path, GOOGLE_SHEET_NAME, THEME_WORKSHEET)

        if df_theme is not None and not df_theme.empty:
            df_theme = filter_valid_rows(df_theme)

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
                
                # 컬럼명 통일: "테마/섹터"를 "테마"로 변경하여 DB 저장 (특수문자 방지)
                if "테마/섹터" in df_theme.columns:
                    df_theme = df_theme.rename(columns={"테마/섹터": "테마"})

                df_theme.to_sql("table_theme", conn, if_exists="replace", index=False)
                print(f"    ✅ table_theme 저장 완료: {len(df_theme)}행")
            else:
                print("    ⚠️ 유효한 테마 데이터가 없습니다.")
        else:
            print("    ⚠️ 테마 데이터를 가져오지 못했습니다.")

    except Exception as e:
        print(f"❌ 동기화 중 오류 발생: {e}")
    finally:
        conn.close()
        print("🏁 동기화 종료")


if __name__ == "__main__":
    sync_gsheet_to_sqlite()
