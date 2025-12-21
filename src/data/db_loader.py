import sqlite3
import pandas as pd
import os
import sys

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
DB_PATH = os.path.join(project_root, "data", "stock.db")


def get_db_connection():
    """DB 연결 객체 반환"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def load_trade_log_from_db():
    """
    [학습용] 매매일지(table_trade_log) 데이터를 불러옵니다.
    """
    conn = get_db_connection()
    try:
        query = "SELECT * FROM table_trade_log ORDER BY 매수날짜"
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def load_theme_from_db():
    """
    [공통] 종목별 테마 정보(table_theme)를 불러와 딕셔너리로 반환합니다.
    Return: {'005930': '반도체', ...}
    """
    conn = get_db_connection()
    try:
        query = 'SELECT "종목코드", "테마" FROM table_theme'
        df = pd.read_sql(query, conn)

        # 종목코드 6자리 포맷팅 (혹시 모를 에러 방지)
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)

        # 딕셔너리로 변환
        theme_map = dict(zip(df["종목코드"], df["테마"]))
        return theme_map
    except Exception as e:
        print(f"[Warning] 테마 로드 실패: {e}")
        return {}
    finally:
        conn.close()


def load_condition_data_from_db(date=None, limit=None):
    """
    [분석용] 조건검색 결과(table_condition)를 불러옵니다.
    date: 특정 날짜('YYYY-MM-DD') 지정 시 해당 날짜 데이터만 로드
    limit: 최근 N개 행만 로드 (None이면 전체)
    """
    conn = get_db_connection()
    try:
        if date:
            query = f"SELECT * FROM table_condition WHERE 스냅샷_날짜 LIKE '{date}%'"
        else:
            # 날짜 지정 없으면 전체 로드 (혹은 최근 데이터)
            query = "SELECT * FROM table_condition"

        if limit:
            query += f" ORDER BY 스냅샷_날짜 DESC LIMIT {limit}"

        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()
