"""condition_history_종가매매.xlsx 데이터 정제 및 백필(Backfill) 모듈.

- ``min_date``(2025-12-29) 이전 스냅샷 행을 제거합니다.
- ``restore_start_date``(2026-05-20) 이후 미수집된 주가/수급/지수 컬럼을
  KIS REST API(주가/수급/시가총액) 및 Naver Finance(지수 등락률/종목 마스터)로 복원합니다.
- 과거 일자별 체결강도는 수집 원천이 없으므로 NaN으로 유지합니다.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import requests

from src import settings
from src.backfill.backfill_price import (
    FetchConfig,
    _fetch_kis_daily_ohlcv,
    _fetch_program_history_by_date,
)
from src.sync.fetcher_investor import get_investor_trade_daily

# ---------------------------------------------------------------------------
# 상수 & 스키마
# ---------------------------------------------------------------------------
SNAPSHOT_DATE = "스냅샷_날짜"
STOCK_NAME = "종목명"
STOCK_CODE = "종목코드"
MARKET = "시장구분"

KRW_PER_EOR = 100_000_000.0  # 1억 원
KRW_MILLION_PER_EOR = 100.0  # 1억 = 100백만 원

_NAVER_SISE_ROW = re.compile(r'\["(\d{8})",([^\]]*)\]')

COLUMN_ORDER = [
    SNAPSHOT_DATE,
    STOCK_NAME,
    STOCK_CODE,
    "시가",
    "고가",
    "저가",
    "종가",
    "전일종가",
    "등락률",
    "체결강도",
    MARKET,
    "시가총액(억)",
    "거래대금(억)",
    "순위",
    "기관_순매수(억)",
    "외국인_순매수(억)",
    "프로그램_순매수(억)",
    "전체종목수",
    "평균거래대금(억)",
    "KOSPI등락률",
    "KOSDAQ등락률",
    "(v-kospi)",
    "(v-kosdaq)",
    "(거래량)",
]

PRICE_COLUMNS = ["시가", "고가", "저가", "종가", "전일종가"]
FLOW_EOR_COLUMNS = ["기관_순매수(억)", "외국인_순매수(억)", "프로그램_순매수(억)"]
TWO_DECIMAL_COLUMNS = [
    "등락률",
    "시가총액(억)",
    "거래대금(억)",
    "평균거래대금(억)",
    "KOSPI등락률",
    "KOSDAQ등락률",
    *FLOW_EOR_COLUMNS,
]

# 주가 블록(종목 코드 기반)에서 한국 컬럼으로 매핑
_OHLCV_TO_KOR = {
    "open": "시가",
    "high": "고가",
    "low": "저가",
    "close": "종가",
    "prev_close": "전일종가",
    "change_pct": "등락률",
    "market_cap_eok": "시가총액(억)",
    "trade_value_eok": "거래대금(억)",
    "volume": "(거래량)",
    "inst_netbuy_eok": "기관_순매수(억)",
    "foreign_netbuy_eok": "외국인_순매수(억)",
    "program_netbuy_eok": "프로그램_순매수(억)",
}

_OHLCV_BLOCK_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "change_pct",
    "market_cap_eok",
    "trade_value_eok",
    "volume",
]


class ConditionHistoryDataSource(Protocol):
    """백필 데이터 원천 인터페이스.

    실환경 구현은 :class:`LiveConditionHistoryDataSource`, 테스트에서는
    네트워크 없이 동작하는 페이크 구현으로 대체할 수 있다.
    """

    def resolve_stock(self, name: str) -> tuple[str, str] | None:
        """종목 한글명으로 (종목코드, 시장구분) 매핑. 미발견 시 None."""
        ...

    def fetch_ohlcv(
        self, code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """종목 일별 OHLCV/거래대금/시가총액.

        반환 컬럼: date, open, high, low, close, volume,
        trade_value_krw, market_cap_krw
        """
        ...

    def fetch_investor_flow(self, code: str, dates: list[str]) -> pd.DataFrame:
        """종목별 기관/외국인 순매수(억).

        반환 컬럼: date, inst_netbuy_eok, foreign_netbuy_eok
        """
        ...

    def fetch_program_flow(self, code: str, dates: list[str]) -> pd.DataFrame:
        """종목별 프로그램 순매수(억).

        반환 컬럼: date, program_netbuy_eok
        """
        ...

    def fetch_index_returns(self, dates: list[str]) -> pd.DataFrame:
        """일자별 KOSPI/KOSDAQ 등락률(%).

        반환 컬럼: date, kospi_pct, kosdaq_pct
        """
        ...


@dataclass(frozen=True)
class ConditionHistoryConfig:
    min_date: str = "2025-12-29"
    restore_start_date: str = "2026-05-20"
    ema_lookback_calendar_days: int = 90
    workers: int = 4
    collect_flows: bool = True
    collect_index: bool = True
    output_dir: Path | None = None


def _to_ymd(value: Any) -> str | None:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime("%Y%m%d")


def _parse_naver_sise_close(text: str) -> pd.DataFrame:
    """Naver siseJson 응답 텍스트에서 일자/종가 DataFrame을 추출합니다.

    응답은 헤더 행(단일 인용부)과 데이터 행(이중 인용부)이 섞인 비표준 JSON이므로
    정규식으로 데이터 행을 추출합니다.
    """
    rows: list[dict[str, Any]] = []
    for match in _NAVER_SISE_ROW.finditer(text):
        date_str = match.group(1)
        fields = [f.strip().strip('"').strip("'") for f in match.group(2).split(",")]
        if len(fields) < 4:
            continue
        dt = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        close = pd.to_numeric(fields[3], errors="coerce")
        if pd.notna(dt) and pd.notna(close):
            rows.append({"date": dt, "close": float(close)})
    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame(rows)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


class LiveConditionHistoryDataSource:
    """KIS REST API / Naver Finance 기반 실환경 데이터 원천."""

    def __init__(
        self,
        fetch_cfg: FetchConfig | None = None,
        request_sleep_sec: float = 0.05,
    ) -> None:
        self.fetch_cfg = fetch_cfg or FetchConfig()
        self.request_sleep_sec = max(0.0, float(request_sleep_sec))
        self._name_cache: dict[str, tuple[str, str] | None] = {}

    def resolve_stock(self, name: str) -> tuple[str, str] | None:
        key = str(name).strip()
        if key in self._name_cache:
            return self._name_cache[key]
        result = self._resolve_stock_remote(key)
        self._name_cache[key] = result
        return result

    def _resolve_stock_remote(self, name: str) -> tuple[str, str] | None:
        try:
            time.sleep(self.request_sleep_sec)
            resp = requests.get(
                "https://ac.stock.naver.com/ac",
                params={"q": name, "st": "code", "target": "stock"},
                timeout=10,
            )
            payload = resp.json()
        except Exception:
            return None
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip() != name:
                continue
            if item.get("category") != "stock":
                continue
            code = str(item.get("code", "")).strip()
            if not code.isdigit():
                continue
            market = str(item.get("typeCode", "")).strip().upper()
            if market not in {"KOSPI", "KOSDAQ"}:
                continue
            return code.zfill(6), market
        return None

    def fetch_ohlcv(
        self, code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        raw = _fetch_kis_daily_ohlcv(code, start, end, self.fetch_cfg)
        empty = pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_value_krw",
                "market_cap_krw",
            ]
        )
        if raw is None or raw.empty:
            return empty
        out = raw.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        out = out.drop_duplicates(subset=["date"], keep="last")
        keep = ["date", "open", "high", "low", "close", "volume", "trade_value_krw", "market_cap_krw"]
        missing = [c for c in keep if c not in out.columns]
        for col in missing:
            out[col] = np.nan
        return out[keep]

    def fetch_investor_flow(self, code: str, dates: list[str]) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["date", "inst_netbuy_eok", "foreign_netbuy_eok"])
        if not dates:
            return empty
        try:
            inv = get_investor_trade_daily(
                code, min(dates), max(dates), target_dates=dates
            )
        except Exception:
            return empty
        if inv is None or inv.empty or "inst_netbuy" not in inv.columns:
            return empty
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(inv["date"], errors="coerce"),
                "inst_netbuy_eok": pd.to_numeric(inv["inst_netbuy"], errors="coerce")
                / KRW_MILLION_PER_EOR,
                "foreign_netbuy_eok": pd.to_numeric(inv["foreign_netbuy"], errors="coerce")
                / KRW_MILLION_PER_EOR,
            }
        )
        out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        return out

    def fetch_program_flow(self, code: str, dates: list[str]) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["date", "program_netbuy_eok"])
        if not dates:
            return empty
        start_ts = pd.to_datetime(min(dates), format="%Y%m%d", errors="coerce")
        end_ts = pd.to_datetime(max(dates), format="%Y%m%d", errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            return empty
        try:
            raw = _fetch_program_history_by_date(
                code, start_ts, end_ts, self.fetch_cfg, target_dates=dates
            )
        except Exception:
            return empty
        if raw is None or raw.empty:
            return empty
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["date"], errors="coerce"),
                "program_netbuy_eok": pd.to_numeric(
                    raw["program_netbuy"], errors="coerce"
                )
                / KRW_PER_EOR,
            }
        )
        out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        return out

    def fetch_index_returns(self, dates: list[str]) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["date", "kospi_pct", "kosdaq_pct"])
        if not dates:
            return empty
        start_ts = pd.to_datetime(min(dates), format="%Y%m%d", errors="coerce")
        end_ts = pd.to_datetime(max(dates), format="%Y%m%d", errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            return empty
        start = (start_ts - pd.Timedelta(days=5)).strftime("%Y%m%d")
        end = end_ts.strftime("%Y%m%d")
        wanted = set(dates)
        frames: list[pd.DataFrame] = []
        for symbol, out_col in (("KOSPI", "kospi_pct"), ("KOSDAQ", "kosdaq_pct")):
            try:
                time.sleep(self.request_sleep_sec)
                resp = requests.get(
                    "https://api.finance.naver.com/siseJson.naver",
                    params={
                        "symbol": symbol,
                        "requestType": 1,
                        "startTime": start,
                        "endTime": end,
                        "timeframe": "day",
                    },
                    timeout=10,
                )
                close_df = _parse_naver_sise_close(resp.text)
            except Exception:
                close_df = pd.DataFrame(columns=["date", "close"])
            if close_df.empty:
                frames.append(pd.DataFrame(columns=["date", out_col]))
                continue
            close = pd.to_numeric(close_df["close"], errors="coerce")
            close_df[out_col] = close.pct_change() * 100.0
            close_df = close_df[close_df["date"].dt.strftime("%Y%m%d").isin(wanted)]
            frames.append(close_df[["date", out_col]].copy())
        kospi, kosdaq = frames[0], frames[1]
        if kospi.empty and kosdaq.empty:
            return empty
        out = kospi
        if not kosdaq.empty:
            out = out.merge(kosdaq, on="date", how="outer")
        return out.sort_values("date").reset_index(drop=True)


class ConditionHistoryBackfiller:
    """엑셀 정제 + 미수집 컬럼 복원 파이프라인."""

    def __init__(
        self,
        data_source: ConditionHistoryDataSource,
        config: ConditionHistoryConfig | None = None,
    ) -> None:
        self.data_source = data_source
        self.config = config or ConditionHistoryConfig()

    # ------------------------------------------------------------------
    # 정제 (Filtering)
    # ------------------------------------------------------------------
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if SNAPSHOT_DATE not in df.columns:
            raise ValueError(f"입력 데이터에 '{SNAPSHOT_DATE}' 컬럼이 없습니다.")
        out = df.copy()
        out[SNAPSHOT_DATE] = pd.to_datetime(out[SNAPSHOT_DATE], errors="coerce")
        min_dt = pd.Timestamp(self.config.min_date)
        out = out[out[SNAPSHOT_DATE] >= min_dt].copy()
        return out.dropna(subset=[SNAPSHOT_DATE])

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self.filter(df)
        vol_col = "(거래량)"
        vol_missing = cleaned[vol_col].isna() if vol_col in cleaned.columns else pd.Series(True, index=cleaned.index)
        restore_mask = (cleaned[SNAPSHOT_DATE] >= pd.Timestamp(self.config.restore_start_date)) | vol_missing
        unchanged = cleaned[~restore_mask]
        restore = cleaned[restore_mask].copy()
        if not restore.empty:
            restore = self._enrich(restore)
        result = pd.concat([unchanged, restore], ignore_index=True)
        result = result.reindex(columns=COLUMN_ORDER)
        result = result.sort_values(SNAPSHOT_DATE).reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # 복원 (Backfill)
    # ------------------------------------------------------------------
    def _resolve_names(self, names: list[str]) -> dict[str, tuple[str, str] | None]:
        unique = sorted({str(n).strip() for n in names if pd.notna(n) and str(n).strip()})
        cache: dict[str, tuple[str, str] | None] = {}

        def _resolve(name: str) -> None:
            if name not in cache:
                cache[name] = self.data_source.resolve_stock(name)

        with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as ex:
            list(ex.map(_resolve, unique))
        return cache

    def _enrich(self, restore: pd.DataFrame) -> pd.DataFrame:
        out = restore.copy()
        out[STOCK_CODE] = out[STOCK_CODE].astype(object)
        out["체결강도"] = np.nan

        # 1. 종목명 -> 종목코드/시장구분 매핑
        resolutions = self._resolve_names(list(out[STOCK_NAME].dropna().unique()))
        code_by_name = {n: r[0] for n, r in resolutions.items() if r is not None}
        market_by_name = {n: r[1] for n, r in resolutions.items() if r is not None}
        out[STOCK_CODE] = (
            out[STOCK_CODE]
            .fillna(out[STOCK_NAME].map(code_by_name))
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
        out[MARKET] = out[MARKET].fillna(out[STOCK_NAME].map(market_by_name))

        # 2. 종목별 주가/수급 블록 병합
        out["_ymd"] = out[SNAPSHOT_DATE].dt.strftime("%Y%m%d")
        end_ts = pd.Timestamp(out[SNAPSHOT_DATE].max())
        start_ts = end_ts - pd.Timedelta(days=self.config.ema_lookback_calendar_days)

        coded = out[out[STOCK_CODE].str.len() > 0]
        block_frames: list[pd.DataFrame] = []
        if not coded.empty:
            by_code = (
                coded.groupby(STOCK_CODE)["_ymd"]
                .apply(lambda s: sorted(set(s)))
                .to_dict()
            )
            code_items = sorted(by_code.items())
            with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as ex:
                futures = {
                    ex.submit(self._fetch_code_block, code, ymd_list, start_ts, end_ts): code
                    for code, ymd_list in code_items
                }
                for fut in as_completed(futures):
                    code = futures[fut]
                    try:
                        block = fut.result()
                        if block is not None and not block.empty:
                            block_frames.append(block)
                    except Exception as exc:  # pragma: no cover - network/runtime
                        print(f"[warn] condition backfill fetch failed code={code}: {exc}")

            if block_frames:
                blocks = pd.concat(block_frames, ignore_index=True)
                out = out.merge(
                    blocks,
                    how="left",
                    left_on=[STOCK_CODE, "_ymd"],
                    right_on=["code", "_ymd"],
                )

        # 3. 주가 블록 컬럼 -> 한국 컬럼 채움 (기존 값 우선 유지)
        for src, dst in _OHLCV_TO_KOR.items():
            if src in out.columns:
                out[dst] = out[dst].fillna(pd.to_numeric(out[src], errors="coerce"))
        out = out.drop(columns=[c for c in _OHLCV_TO_KOR if c in out.columns], errors="ignore")

        # 4. 일자별 집계 (전체종목수/평균거래대금/순위)
        out["전체종목수"] = out.groupby(SNAPSHOT_DATE)[STOCK_NAME].transform("count")
        out["평균거래대금(억)"] = out.groupby(SNAPSHOT_DATE)["거래대금(억)"].transform("mean")
        out["순위"] = out.groupby(SNAPSHOT_DATE)["거래대금(억)"].rank(
            ascending=False, method="first"
        )

        # 5. 지수 등락률
        if self.config.collect_index:
            try:
                index_df = self.data_source.fetch_index_returns(sorted(out["_ymd"].unique()))
            except Exception as exc:  # pragma: no cover - network/runtime
                print(f"[warn] condition backfill index fetch failed: {exc}")
                index_df = pd.DataFrame(columns=["date", "kospi_pct", "kosdaq_pct"])
            if index_df is not None and not index_df.empty:
                index_df = index_df.copy()
                index_df["_ymd"] = index_df["date"].dt.strftime("%Y%m%d")
                out = out.drop(columns=["KOSPI등락률", "KOSDAQ등락률"], errors="ignore")
                out = out.merge(
                    index_df[["_ymd", "kospi_pct", "kosdaq_pct"]],
                    on="_ymd",
                    how="left",
                )
                out = out.rename(
                    columns={
                        "kospi_pct": "KOSPI등락률",
                        "kosdaq_pct": "KOSDAQ등락률",
                    }
                )

        # 6. 타입/자릿수 정리
        for col in PRICE_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)
        for col in TWO_DECIMAL_COLUMNS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
        if "전체종목수" in out.columns:
            out["전체종목수"] = pd.to_numeric(out["전체종목수"], errors="coerce")

        out = out.drop(columns=["_ymd"], errors="ignore")
        return out

    def _fetch_code_block(
        self,
        code: str,
        target_ymds: list[str],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        target = set(target_ymds)
        frames: list[pd.DataFrame] = []

        ohlcv = self.data_source.fetch_ohlcv(code, start_ts, end_ts)
        if ohlcv is not None and not ohlcv.empty:
            hist = (
                ohlcv.sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .copy()
            )
            hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
            hist["prev_close"] = hist["close"].shift(1)
            hist["change_pct"] = (hist["close"] - hist["prev_close"]) / hist["prev_close"] * 100.0
            hist["market_cap_eok"] = (
                pd.to_numeric(hist["market_cap_krw"], errors="coerce") / KRW_PER_EOR
            )
            hist["trade_value_eok"] = (
                pd.to_numeric(hist["trade_value_krw"], errors="coerce") / KRW_PER_EOR
            )
            hist["_ymd"] = hist["date"].dt.strftime("%Y%m%d")
            part = hist[hist["_ymd"].isin(target)].copy()
            if not part.empty:
                part["code"] = code
                frames.append(part[["code", "_ymd", *_OHLCV_BLOCK_COLUMNS]])

        if self.config.collect_flows:
            inv = self.data_source.fetch_investor_flow(code, target_ymds)
            if inv is not None and not inv.empty:
                inv = inv.copy()
                inv["_ymd"] = inv["date"].dt.strftime("%Y%m%d")
                inv["code"] = code
                frames.append(
                    inv[["code", "_ymd", "inst_netbuy_eok", "foreign_netbuy_eok"]]
                )
            prog = self.data_source.fetch_program_flow(code, target_ymds)
            if prog is not None and not prog.empty:
                prog = prog.copy()
                prog["_ymd"] = prog["date"].dt.strftime("%Y%m%d")
                prog["code"] = code
                frames.append(prog[["code", "_ymd", "program_netbuy_eok"]])

        if not frames:
            return pd.DataFrame()
        merged = frames[0]
        for other in frames[1:]:
            merged = merged.merge(other, on=["code", "_ymd"], how="outer")
        return merged


def save_cleaned(df: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """정제/복원 완료 DataFrame을 CSV + Parquet 로 저장합니다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if SNAPSHOT_DATE in out.columns:
        out[SNAPSHOT_DATE] = pd.to_datetime(out[SNAPSHOT_DATE], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    if STOCK_CODE in out.columns:
        out[STOCK_CODE] = out[STOCK_CODE].fillna("").astype(str).str.zfill(6)

    csv_path = output_dir / "condition_history_cleaned.csv"
    parquet_path = output_dir / "condition_history_cleaned.parquet"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        out.to_parquet(parquet_path, index=False)
    except Exception as exc:
        print(f"[warn] condition history parquet save failed: {exc}")
    return [csv_path, parquet_path]


def run_condition_history_backfill(
    excel_path: str,
    min_date: str = "2025-12-29",
    *,
    data_source: ConditionHistoryDataSource | None = None,
    config: ConditionHistoryConfig | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """condition_history_종가매매.xlsx 정제 및 미수집 데이터 백필을 실행합니다."""
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"입력 엑셀 파일이 존재하지 않습니다: {excel_path}")

    cfg = config or ConditionHistoryConfig(min_date=min_date)
    source = data_source or LiveConditionHistoryDataSource()
    df = pd.read_excel(path)

    backfiller = ConditionHistoryBackfiller(source, cfg)
    result = backfiller.run(df)

    if save_output:
        out_dir = cfg.output_dir or settings.DATA_DIR / "history"
        save_cleaned(result, out_dir)
        print(f"[backfill] condition history saved rows={len(result)} dir={out_dir}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="condition_history_종가매매.xlsx 정제 및 미수집 데이터 백필."
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default="condition_history_종가매매.xlsx",
        help="입력 엑셀 파일 경로",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default="2025-12-29",
        help="이 날짜 이전 스냅샷 행 제거 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="정제 결과 저장 디렉토리 (기본: data/history)",
    )
    parser.add_argument("--workers", type=int, default=4, help="병렬 수집 워커 수")
    parser.add_argument("--no-flows", action="store_true", help="수급(기관/외국인/프로그램) 수집 생략")
    parser.add_argument("--no-save", action="store_true", help="결과 파일 저장 생략")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ConditionHistoryConfig(
        min_date=args.min_date,
        workers=max(1, int(args.workers)),
        collect_flows=not bool(args.no_flows),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    result = run_condition_history_backfill(
        args.excel_path,
        config=cfg,
        save_output=not bool(args.no_save),
    )
    print(f"[backfill] condition history done rows={len(result)}")


if __name__ == "__main__":
    main()
