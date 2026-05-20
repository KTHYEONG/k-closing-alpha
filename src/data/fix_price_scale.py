"""구글 스프레드시트 및 로컬 DB 내 가격 데이터 스케일 불일치 해결 스크립트.

수동으로 기입한 가격과 pykrx API를 통해 받은 실제 수정주가의 불일치를 감지하고,
사용자 승인 하에 자동으로 올바른 스케일로 수정(보정)합니다.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Windows 터미널 한글 인코딩 에러 방지 (이모지 및 UTF-8 강제 출력)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: S110
    pass


import gspread
import pandas as pd

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import settings
from src.processing.scale_corrector import apply_scale_correction, detect_price_scale_mismatch

try:
    from pykrx import stock
except ImportError:
    stock = None


def pykrx_price_provider(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """pykrx를 이용하여 해당 종목의 실 거래 종가를 조회합니다. (Rate Limit 우회 및 안정적 재시도 탑재)

    Args:
        symbol: 종목코드 (6자리)
        start_date: 시작일
        end_date: 종료일

    Returns:
        pd.DataFrame: date, close 컬럼이 포함된 데이터프레임

    """
    if stock is None:
        raise RuntimeError("pykrx 패키지가 설치되어 있지 않습니다.")

    # 조회 기간 버퍼 추가 (영업일이 아닌 날 매수한 경우 직전 영업일 가격 확보를 위함)
    buffer_start = (start_date - pd.Timedelta(days=5)).strftime("%Y%m%d")
    buffer_end = (end_date + pd.Timedelta(days=5)).strftime("%Y%m%d")

    # KRX 서버 차단 예방을 위한 호출 딜레이 추가
    import time
    time.sleep(0.3)

    # 3회 재시도 루프 (Exponential Backoff)
    max_retries = 3
    df = None
    for attempt in range(max_retries):
        try:
            df = stock.get_market_ohlcv_by_date(buffer_start, buffer_end, symbol)
            if df is not None and not df.empty:
                break
        except Exception:
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(1.0 * (attempt + 1))

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df["date"] = df.index

    # '종가' 컬럼명 매핑
    close_col = None
    for c in ["종가", "Close", "close"]:
        if c in df.columns:
            close_col = c
            break

    if close_col is None:
        close_col = list(df.columns)[3] if len(df.columns) >= 4 else df.columns[0]

    df = df.rename(columns={close_col: "close"})
    return df[["date", "close"]].reset_index(drop=True)


def _connect_gsheet() -> gspread.Client:
    from oauth2client.service_account import ServiceAccountCredentials

    key_path = str(settings.GOOGLE_KEY_PATH)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
    client: gspread.Client = gspread.authorize(creds)
    return client


def update_gsheet_scales(sh: gspread.Spreadsheet, sheet_name: str, mismatched_symbols: dict[tuple[str, pd.Timestamp], float]) -> None:
    """구글 스프레드시트에 기입된 가격 스케일 오류 값을 일괄 업데이트합니다.

    Args:
        sh: gspread 스프레드시트 객체
        sheet_name: 워크시트 이름 (예: Trade, Trade2)
        mismatched_symbols: 탐지된 {(종목코드, 매수날짜): 스케일비율} 매핑

    """
    ws = sh.worksheet(sheet_name)
    all_values = ws.get_all_values()
    if not all_values:
        return

    headers = all_values[0]

    # 컬럼 인덱스 찾기
    from src.data.backfill_trade_sheets import _resolve_columns
    cols = _resolve_columns(headers, sheet_name=sheet_name)

    symbol_idx = cols.get("symbol")
    date_idx = cols.get("date")

    if symbol_idx is None or date_idx is None:
        print(f"   ⚠️ {sheet_name} 시트에서 필수 컬럼(종목코드/날짜)을 찾지 못했습니다.")
        return

    # 명시적 타입 좁히기
    s_idx: int = symbol_idx
    d_idx: int = date_idx

    # 업데이트 대상 가격 컬럼들의 인덱스 매핑
    target_price_keys = ["open", "high", "low", "close", "prev_close", "buy_price", "sell_price"]
    target_col_idxs: dict[str, int] = {}
    for k in target_price_keys:
        val = cols.get(k)
        if val is not None:
            target_col_idxs[k] = val

    cells_to_update: list[gspread.Cell] = []
    updated_rows_count = 0

    # 검사 중 로그 제거 (간소화)
    for row_idx, row in enumerate(all_values[1:], start=2):
        if s_idx >= len(row) or d_idx >= len(row):
            continue

        symbol = str(row[s_idx]).strip().zfill(6)
        raw_date = str(row[d_idx]).strip()
        if not raw_date:
            continue

        try:
            normalized_date = pd.to_datetime(raw_date).normalize()
        except (ValueError, TypeError):
            continue

        key = (symbol, normalized_date)
        if key not in mismatched_symbols:
            continue

        scale_factor = mismatched_symbols[key]
        row_updated = False

        for key_name, col_idx in target_col_idxs.items():
            if col_idx >= len(row):
                continue

            raw_val = str(row[col_idx]).strip()
            if not raw_val:
                continue

            try:
                # 숫자 파싱 후 스케일 비율 곱해주기
                num_val = float(raw_val.replace(",", ""))
                if num_val == 0:
                    continue

                corrected_val = round(num_val * scale_factor, 2)
                if key_name in ["open", "high", "low", "close", "prev_close", "buy_price", "sell_price"]:
                    corrected_val = round(corrected_val)

                cells_to_update.append(gspread.Cell(row=row_idx, col=col_idx + 1, value=str(corrected_val)))
                row_updated = True
            except ValueError:
                continue

        if row_updated:
            updated_rows_count += 1

    if not cells_to_update:
        print(f"   ✅ {sheet_name} 시트에 수정할 스케일 데이터가 없습니다.")
        return

    print(f" 🌐 {sheet_name} 시트: {updated_rows_count}개 행 보정용 {len(cells_to_update)}개 셀 업데이트 중...")
    
    # API 한도 제한 회피를 위한 배치 업데이트 (1000개 단위)
    batch_size = 1000
    for i in range(0, len(cells_to_update), batch_size):
        ws.update_cells(cells_to_update[i : i + batch_size], value_input_option="RAW")  # type: ignore[arg-type]
        time.sleep(1.0) # Rate limit 방지

    print(f"   ✅ {sheet_name} 완료!")


def main() -> None:
    parser = argparse.ArgumentParser(description="수동 기입 가격과 API 가격 간의 스케일 불일치 탐지 및 수정 도구")
    parser.add_argument("--db-check", action="store_true", help="로컬 SQLite 데이터베이스만 검사")
    parser.add_argument("--apply", action="store_true", help="실제 보정된 값을 구글 시트와 로컬 DB에 적용")
    parser.add_argument("--threshold-lower", type=float, default=0.65, help="스케일 오류 탐지 하한선 (default: 0.65)")
    parser.add_argument("--threshold-upper", type=float, default=1.50, help="스케일 오류 탐지 상한선 (default: 1.50)")
    args = parser.parse_args()

    db_path = str(settings.STOCK_DB_PATH)
    if not os.path.exists(db_path):
        print(f"❌ DB 파일을 찾을 수 없습니다: {db_path}")
        sys.exit(1)

    print(f"🔍 로컬 DB ({db_path}) 로드 중...")
    conn = sqlite3.connect(db_path)
    
    try:
        # DB에서 매매일지 정보 읽기
        df = pd.read_sql("SELECT * FROM table_trade_log", conn)
    except Exception as e:
        print(f"❌ DB 조회 실패 (먼저 sync_gsheet_to_db.py를 실행하여 동기화해주세요): {e}")
        conn.close()
        sys.exit(1)

    if df.empty:
        print("⚠️ table_trade_log 데이터가 비어 있습니다.")
        conn.close()
        sys.exit(1)

    # 컬럼 이름 전처리 매핑 적용
    from src.processing.preprocessor import RENAME_MAP
    df = df.rename(columns=RENAME_MAP)
    
    # 종목코드 표준화
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.strip().str.zfill(6)

    # 날짜 컬럼 파싱
    if "매수날짜" in df.columns:
        df["매수날짜"] = pd.to_datetime(df["매수날짜"])

    print("📈 pykrx API와 스프레드시트 가격 비교 분석 시작...")
    # 스케일 불일치 종목 탐지
    mismatched = detect_price_scale_mismatch(
        df,
        pykrx_price_provider,
        threshold_lower=args.threshold_lower,
        threshold_upper=args.threshold_upper,
        max_workers=2
    )

    if not mismatched:
        print("✨ 분석 완료: 모든 종목의 가격 데이터 스케일이 정상 범위 내에 있습니다.")
        conn.close()
        sys.exit(0)

    print(f"\n⚠️ 스케일 불일치 데이터 {len(mismatched)}건 탐지됨:")
    mismatched_details = []
    for (symbol, date), ratio in mismatched.items():
        mismatched_details.append(f"{symbol}({date.strftime('%Y-%m-%d')}, 비율:{ratio:.2f})")
    print("  -> " + ", ".join(mismatched_details) + "\n")

    if not args.apply:
        print("💡 팁: 실제 시트 및 DB에 자동 반영하려면 '--apply' 옵션을 붙여 다시 실행하십시오.")
        conn.close()
        sys.exit(0)

    # 보정 가격 적용
    print("⚙️ 데이터베이스 내 가격 데이터 스케일 보정 적용 중...")
    df_corrected = apply_scale_correction(df, mismatched)
    
    # DB 업데이트를 위한 컬럼명 원복
    INVERSE_RENAME_MAP = {v: k for k, v in RENAME_MAP.items()}
    df_db_ready = df_corrected.rename(columns=INVERSE_RENAME_MAP)
    df_db_ready["매수날짜"] = df_db_ready["매수날짜"].dt.strftime("%Y-%m-%d")

    # DB에 보정 결과 저장
    df_db_ready.to_sql("table_trade_log", conn, if_exists="replace", index=False)
    print("✅ 데이터베이스 table_trade_log 스케일 보정 업데이트 완료!")
    conn.close()

    if args.db_check:
        print("🏁 DB 체크 모드로 구글 시트 갱신은 건너뜁니다.")
        sys.exit(0)

    # 구글 스프레드시트 갱신
    print(f"🌐 구글 스프레드시트 '{settings.GOOGLE_SHEET_NAME}'에 접속 중...")
    try:
        sh = _connect_gsheet().open(settings.GOOGLE_SHEET_NAME)
        for sheet_name in settings.TRADE_WORKSHEETS:
            update_gsheet_scales(sh, sheet_name, mismatched)
        print("🏁 모든 작업이 정상 종료되었습니다.")
    except Exception as e:
        print(f"❌ 구글 시트 업데이트 중 오류 발생: {e}")
        print("💡 로컬 DB는 보정 완료되었으나 구글 시트 자동 갱신에 실패했습니다. API 키 및 권한 설정을 확인하세요.")


if __name__ == "__main__":
    main()
