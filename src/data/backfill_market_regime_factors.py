"""일자 단위 마켓 레짐 팩터(VIX, KRX breadth)를 수집/저장하는 백필 스크립트.

VIX(^VIX) 종가와 KRX 상승/하락 종목 집계를 date-level parquet 캐시에 upsert 저장한다.
KRX breadth는 KRX OpenAPI(KRX_OPENAPI_KEY)를 우선 사용하고, 실패 시 pykrx로 fallback 한다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import DEFAULT_CONFIG, PipelineConfig
from src.pipeline.data import load_or_build_snapshot

try:
    from pykrx import stock as krx_stock
except ImportError:  # pragma: no cover - dependency availability differs by env
    krx_stock = None


@dataclass(frozen=True)
class MarketFactorFetchConfig:
    retries: int = 3
    retry_sleep_sec: float = 1.0
    timeout_sec: int = 20
    krx_request_sleep_sec: float = 0.03
    krx_openapi_base_urls: Tuple[str, ...] = (
        "https://data-dbg.krx.co.kr",
        "http://data-dbg.krx.co.kr",
    )
    krx_openapi_endpoints: Tuple[str, ...] = (
        "/svc/apis/sto/stk_bydd_trd",
        "/svc/apis/sto/ksq_bydd_trd",
    )


def _get_env_value(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    if raw is not None and str(raw).strip():
        return str(raw).strip().strip('"').strip("'")

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return default
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            txt = line.strip()
            if not txt or txt.startswith("#") or "=" not in txt:
                continue
            k, v = txt.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        return default
    return default


def _get_env_csv(key: str) -> List[str]:
    txt = _get_env_value(key, "")
    if not txt:
        return []
    return [x.strip() for x in txt.split(",") if x.strip()]


def _parse_date_arg(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None or not str(value).strip():
        return None
    txt = str(value).strip()
    if len(txt) == 8 and txt.isdigit():
        return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(txt, errors="coerce")


def _resolve_range(
    start: Optional[str],
    end: Optional[str],
    use_snapshot_range: bool,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = _parse_date_arg(start)
    end_ts = _parse_date_arg(end)

    if use_snapshot_range and (start_ts is None or end_ts is None):
        snap = load_or_build_snapshot(config=config, rebuild=False, sync_gsheet=False)
        if not snap.empty and "date" in snap.columns:
            d = pd.to_datetime(snap["date"], errors="coerce").dropna()
            if not d.empty:
                if start_ts is None:
                    start_ts = pd.Timestamp(d.min()).normalize()
                if end_ts is None:
                    end_ts = pd.Timestamp(d.max()).normalize()

    if start_ts is None:
        start_ts = pd.Timestamp("2010-01-01")
    if end_ts is None:
        end_ts = pd.Timestamp.today().normalize()

    start_ts = pd.Timestamp(start_ts).normalize()
    end_ts = pd.Timestamp(end_ts).normalize()
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    return start_ts, end_ts


def _download_vix_yfinance(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: MarketFactorFetchConfig,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError(
            "yfinance is required. Install with: python -m pip install yfinance"
        ) from exc

    end_exclusive = end + pd.Timedelta(days=1)
    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(cfg.retries))):
        try:
            data = yf.download(
                "^VIX",
                start=start.strftime("%Y-%m-%d"),
                end=end_exclusive.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=int(cfg.timeout_sec),
            )
            if data is None or data.empty:
                return pd.DataFrame(columns=["date", "vix_close", "source_vix"])

            out = data.copy()
            if isinstance(out.columns, pd.MultiIndex):
                out.columns = [str(c[0]) for c in out.columns]
            out = out.reset_index()
            if "Date" not in out.columns or "Close" not in out.columns:
                return pd.DataFrame(columns=["date", "vix_close", "source_vix"])

            view = pd.DataFrame(
                {
                    "date": pd.to_datetime(out["Date"], errors="coerce"),
                    "vix_close": pd.to_numeric(out["Close"], errors="coerce"),
                    "source_vix": "yfinance:^VIX",
                }
            )
            view = view.dropna(subset=["date", "vix_close"]).sort_values("date")
            view = view.drop_duplicates(subset=["date"], keep="last")
            return view
        except Exception as exc:
            last_err = exc
            time.sleep(float(cfg.retry_sleep_sec) * (attempt + 1))
    raise RuntimeError(f"yfinance VIX download failed: {last_err}")


def _normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(map(str, df.columns))
    col_map = {_normalize_col_name(c): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        normalized = _normalize_col_name(c)
        if normalized in col_map:
            return col_map[normalized]
    return None


def _pick_change_col(df: pd.DataFrame) -> Optional[str]:
    exact = _find_col(
        df,
        [
            "등락률",
            "등락",
            "Change",
            "change",
            "PRDY_CTRT",
            "CMPPREVDD_PRC",
            "UPDN_RT",
            "FLUC_RT",
            "TDD_CMPR",
        ],
    )
    if exact is not None:
        return exact
    for c in map(str, df.columns):
        lowered = _normalize_col_name(c)
        if (
            ("등락" in c)
            or ("change" in lowered)
            or ("cmpprev" in lowered)
            or ("ctrt" in lowered)
            or ("fluc" in lowered)
            or ("updn" in lowered)
        ):
            return c
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _extract_price_change_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    work = df.copy()
    change_col = _pick_change_col(work)
    if change_col is not None:
        return _to_numeric(work[change_col])

    close_col = _find_col(
        work,
        ["종가", "Close", "TDD_CLSPRC", "CLSPRC", "close", "tdd_clsprc"],
    ) or (list(work.columns)[3] if len(work.columns) > 3 else None)
    open_col = _find_col(
        work,
        ["시가", "Open", "TDD_OPNPRC", "OPNPRC", "open", "tdd_opnprc"],
    ) or (list(work.columns)[0] if len(work.columns) > 0 else None)
    if close_col is not None and open_col is not None:
        close = _to_numeric(work[close_col])
        open_ = _to_numeric(work[open_col])
        return close - open_

    prev_close_col = _find_col(
        work,
        ["전일종가", "PRDY_CLSPRC", "PREV_CLSPRC", "BF_CLSPRC", "prdy_clsprc"],
    )
    if close_col is not None and prev_close_col is not None:
        close = _to_numeric(work[close_col])
        prev = _to_numeric(work[prev_close_col])
        return close - prev
    return pd.Series(dtype=float)


def _extract_outblock_rows(payload: object) -> List[Dict[str, object]]:
    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("OutBlock_1"), list):
        return [r for r in payload["OutBlock_1"] if isinstance(r, dict)]

    if isinstance(payload.get("output"), list):
        return [r for r in payload["output"] if isinstance(r, dict)]

    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return [r for r in v if isinstance(r, dict)]
    return []


def _safe_get_krx_openapi_day(
    *,
    date_ymd: str,
    endpoint: str,
    auth_key: str,
    cfg: MarketFactorFetchConfig,
    base_urls: List[str],
) -> Tuple[pd.DataFrame, bool]:
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("requests is required for KRX OpenAPI calls.") from exc

    last_err: Optional[Exception] = None
    unauthorized = False
    for base_url in [b.rstrip("/") for b in base_urls if str(b).strip()]:
        url = f"{base_url}{endpoint}"
        for attempt in range(max(1, int(cfg.retries))):
            try:
                time.sleep(max(0.0, float(cfg.krx_request_sleep_sec)))
                resp = requests.get(
                    url,
                    params={"basDd": date_ymd},
                    headers={"AUTH_KEY": auth_key.strip()},
                    timeout=int(cfg.timeout_sec),
                )
                if resp.status_code != 200:
                    body_head = (resp.text or "")[:240]
                    if resp.status_code in (401, 403):
                        unauthorized = True
                        return pd.DataFrame(), True
                    raise RuntimeError(f"http_status={resp.status_code} body={body_head}")
                payload = resp.json()
                resp_code = str(payload.get("respCode", "")).strip() if isinstance(payload, dict) else ""
                if resp_code in {"401", "403"}:
                    unauthorized = True
                    return pd.DataFrame(), True
                rows = _extract_outblock_rows(payload)
                if not rows:
                    return pd.DataFrame(), unauthorized
                return pd.DataFrame(rows), unauthorized
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_err = exc
                time.sleep(float(cfg.retry_sleep_sec) * (attempt + 1))
    if last_err is not None:
        print(f"[warn] KRX OpenAPI fetch failed date={date_ymd} endpoint={endpoint}: {last_err}")
    return pd.DataFrame(), unauthorized


def _fetch_krx_breadth_openapi(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: MarketFactorFetchConfig,
    auth_key: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    dates = pd.date_range(start=start, end=end, freq="B")
    env_base_urls = _get_env_csv("KRX_OPENAPI_BASE_URLS")
    env_single_base = _get_env_value("KRX_OPENAPI_BASE_URL", "")
    if env_single_base:
        env_base_urls = [env_single_base]
    base_urls = env_base_urls or [b for b in cfg.krx_openapi_base_urls if str(b).strip()]

    env_endpoints = _get_env_csv("KRX_OPENAPI_ENDPOINTS")
    endpoints = env_endpoints or [e for e in cfg.krx_openapi_endpoints if str(e).strip()]
    disabled_endpoints: Set[str] = set()

    for dt in dates:
        date_ymd = dt.strftime("%Y%m%d")
        parts: List[pd.DataFrame] = []
        called_endpoints: List[str] = []
        for endpoint in endpoints:
            if endpoint in disabled_endpoints:
                continue
            day_df, unauthorized = _safe_get_krx_openapi_day(
                date_ymd=date_ymd,
                endpoint=endpoint,
                auth_key=auth_key,
                cfg=cfg,
                base_urls=base_urls,
            )
            if unauthorized:
                disabled_endpoints.add(endpoint)
                print(f"[warn] KRX OpenAPI unauthorized endpoint disabled: {endpoint}")
                continue
            if day_df is None or day_df.empty:
                continue
            parts.append(day_df)
            called_endpoints.append(endpoint.rsplit("/", 1)[-1])

        if not parts:
            continue

        day_all = pd.concat(parts, axis=0, ignore_index=True)
        chg = _extract_price_change_series(day_all)
        chg = chg.replace([np.inf, -np.inf], np.nan).dropna()
        if chg.empty:
            print(f"[warn] KRX OpenAPI no usable change column date={date_ymd}")
            continue

        adv_count = int((chg > 0).sum())
        dec_count = int((chg < 0).sum())
        den = adv_count + dec_count
        adv_ratio = float(adv_count / den) if den > 0 else np.nan
        rows.append(
            {
                "date": pd.Timestamp(dt).normalize(),
                "adv_count": adv_count,
                "dec_count": dec_count,
                "adv_ratio": adv_ratio,
                "source_breadth": f"krx_openapi:{'+'.join(called_endpoints)}",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "adv_count", "dec_count", "adv_ratio", "source_breadth"])
    out = pd.DataFrame(rows).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def _safe_get_krx_day(date_ymd: str, cfg: MarketFactorFetchConfig) -> pd.DataFrame:
    if krx_stock is None:
        return pd.DataFrame()
    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(cfg.retries))):
        try:
            time.sleep(max(0.0, float(cfg.krx_request_sleep_sec)))
            return krx_stock.get_market_ohlcv_by_ticker(date_ymd, market="ALL")
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_err = exc
            time.sleep(float(cfg.retry_sleep_sec) * (attempt + 1))
    print(f"[warn] KRX breadth fetch failed date={date_ymd}: {last_err}")
    return pd.DataFrame()


def _safe_get_krx_change_day(date_ymd: str, cfg: MarketFactorFetchConfig) -> pd.DataFrame:
    if krx_stock is None:
        return pd.DataFrame()
    markets = ["ALL", "KOSPI", "KOSDAQ", "KONEX"]
    frames: List[pd.DataFrame] = []
    for market in markets:
        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(cfg.retries))):
            try:
                time.sleep(max(0.0, float(cfg.krx_request_sleep_sec)))
                part = krx_stock.get_market_price_change_by_ticker(date_ymd, date_ymd, market=market)
                if part is None or part.empty:
                    break
                frames.append(part.copy())
                break
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                last_err = exc
                time.sleep(float(cfg.retry_sleep_sec) * (attempt + 1))
        if last_err is not None and market == "ALL":
            print(f"[warn] KRX change fallback failed date={date_ymd}: {last_err}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0)
    out = out[~out.index.duplicated(keep="last")]
    return out


def _fetch_krx_breadth(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: MarketFactorFetchConfig,
) -> pd.DataFrame:
    auth_key = _get_env_value("KRX_OPENAPI_KEY", "")
    if auth_key:
        openapi_df = _fetch_krx_breadth_openapi(
            start=start,
            end=end,
            cfg=cfg,
            auth_key=auth_key,
        )
        if not openapi_df.empty:
            return openapi_df
        print("[warn] KRX OpenAPI breadth collection returned empty; fallback to pykrx.")
    else:
        print("[warn] KRX_OPENAPI_KEY is not set; fallback to pykrx breadth collection.")

    if krx_stock is None:
        print("[warn] pykrx is not installed; KRX breadth collection skipped.")
        return pd.DataFrame(columns=["date", "adv_count", "dec_count", "adv_ratio", "source_breadth"])

    rows: List[Dict[str, object]] = []
    dates = pd.date_range(start=start, end=end, freq="B")
    for dt in dates:
        date_ymd = dt.strftime("%Y%m%d")
        day_ohlcv = _safe_get_krx_day(date_ymd, cfg=cfg)
        chg = pd.Series(dtype=float)
        source = ""

        if day_ohlcv is not None and not day_ohlcv.empty:
            chg = _extract_price_change_series(day_ohlcv)
            if not chg.empty:
                source = "pykrx:ohlcv"

        if chg.empty:
            day_chg = _safe_get_krx_change_day(date_ymd, cfg=cfg)
            if day_chg is None or day_chg.empty:
                continue
            chg = _extract_price_change_series(day_chg)
            if not chg.empty:
                source = "pykrx:price_change_by_ticker"
            else:
                continue

        chg = chg.replace([np.inf, -np.inf], np.nan).dropna()
        if chg.empty:
            continue

        adv_count = int((chg > 0).sum())
        dec_count = int((chg < 0).sum())
        den = adv_count + dec_count
        adv_ratio = float(adv_count / den) if den > 0 else np.nan
        rows.append(
            {
                "date": pd.Timestamp(dt).normalize(),
                "adv_count": adv_count,
                "dec_count": dec_count,
                "adv_ratio": adv_ratio,
                "source_breadth": source,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "adv_count", "dec_count", "adv_ratio", "source_breadth"])
    out = pd.DataFrame(rows).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def _merge_factor_frames(vix_df: pd.DataFrame, breadth_df: pd.DataFrame) -> pd.DataFrame:
    if vix_df.empty and breadth_df.empty:
        return pd.DataFrame(
            columns=["date", "vix_close", "source_vix", "adv_count", "dec_count", "adv_ratio", "source_breadth"]
        )
    if vix_df.empty:
        return breadth_df.copy()
    if breadth_df.empty:
        return vix_df.copy()
    out = vix_df.merge(breadth_df, on="date", how="outer")
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


def _upsert_parquet(df_new: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        if path.exists() and path.is_dir():
            path = path / "market_factors_daily.parquet"
        else:
            raise ValueError(
                f"Output path must be a parquet file path (.parquet): {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if df_new is None or df_new.empty:
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                print(f"[warn] existing parquet read failed, returning empty: {path} ({exc})")
                return pd.DataFrame(
                    columns=["date", "vix_close", "source_vix", "adv_count", "dec_count", "adv_ratio", "source_breadth"]
                )
        return pd.DataFrame(
            columns=["date", "vix_close", "source_vix", "adv_count", "dec_count", "adv_ratio", "source_breadth"]
        )

    incoming = df_new.copy()
    incoming["date"] = pd.to_datetime(incoming["date"], errors="coerce")
    incoming = incoming.dropna(subset=["date"]).sort_values("date")
    incoming = incoming.drop_duplicates(subset=["date"], keep="last")

    if path.exists():
        try:
            old = pd.read_parquet(path)
            old["date"] = pd.to_datetime(old["date"], errors="coerce")
            old = old.dropna(subset=["date"]).sort_values("date")
            old = old.drop_duplicates(subset=["date"], keep="last")
        except Exception as exc:
            print(f"[warn] existing parquet read failed, rebuilding from incoming only: {path} ({exc})")
            old = pd.DataFrame(columns=incoming.columns)
    else:
        old = pd.DataFrame(columns=incoming.columns)

    old_idx = old.set_index("date")
    new_idx = incoming.set_index("date")
    # Prefer incoming non-null values while preserving old values for missing cells.
    out_idx = new_idx.combine_first(old_idx)
    out = out_idx.reset_index().rename(columns={"index": "date"})
    if "vix_close" in out.columns:
        out["vix_close"] = pd.to_numeric(out["vix_close"], errors="coerce")
    for col in ["adv_count", "dec_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "adv_ratio" in out.columns:
        out["adv_ratio"] = pd.to_numeric(out["adv_ratio"], errors="coerce")

    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out.to_parquet(path, index=False)
    return out


def _normalize_output_path(path: Path) -> Path:
    if path.suffix.lower() == ".parquet":
        return path
    if path.exists() and path.is_dir():
        return path / "market_factors_daily.parquet"
    raise ValueError(f"Output path must be a parquet file path (.parquet): {path}")


def run_backfill_market_regime_factors(
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_snapshot_range: bool = True,
    out_path: Optional[Path] = None,
    collect_vix: bool = True,
    collect_krx_breadth: bool = True,
    config: PipelineConfig = DEFAULT_CONFIG,
    fetch_cfg: MarketFactorFetchConfig = MarketFactorFetchConfig(),
) -> pd.DataFrame:
    start_ts, end_ts = _resolve_range(
        start=start,
        end=end,
        use_snapshot_range=use_snapshot_range,
        config=config,
    )

    vix_df = (
        _download_vix_yfinance(start=start_ts, end=end_ts, cfg=fetch_cfg)
        if collect_vix
        else pd.DataFrame(columns=["date", "vix_close", "source_vix"])
    )
    breadth_df = (
        _fetch_krx_breadth(start=start_ts, end=end_ts, cfg=fetch_cfg)
        if collect_krx_breadth
        else pd.DataFrame(columns=["date", "adv_count", "dec_count", "adv_ratio", "source_breadth"])
    )
    merged_new = _merge_factor_frames(vix_df=vix_df, breadth_df=breadth_df)

    out_target = _normalize_output_path(out_path or config.market_factors_path)
    merged = _upsert_parquet(merged_new, out_target)
    min_date = pd.to_datetime(merged.get("date"), errors="coerce").min() if "date" in merged.columns else pd.NaT
    max_date = pd.to_datetime(merged.get("date"), errors="coerce").max() if "date" in merged.columns else pd.NaT
    range_txt = "N/A ~ N/A"
    if pd.notna(min_date) and pd.notna(max_date):
        range_txt = f"{min_date.date()} ~ {max_date.date()}"
    print(
        "[market-regime] saved "
        f"rows={len(merged)} date_range=({range_txt}) "
        f"path={out_target}"
    )
    return merged


def run_backfill_market_factors(
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_snapshot_range: bool = True,
    out_path: Optional[Path] = None,
    config: PipelineConfig = DEFAULT_CONFIG,
    fetch_cfg: MarketFactorFetchConfig = MarketFactorFetchConfig(),
) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return run_backfill_market_regime_factors(
        start=start,
        end=end,
        use_snapshot_range=use_snapshot_range,
        out_path=out_path,
        collect_vix=True,
        collect_krx_breadth=True,
        config=config,
        fetch_cfg=fetch_cfg,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill date-level market regime factors (VIX + KRX breadth via OpenAPI->pykrx fallback)."
    )
    parser.add_argument("--start", type=str, default=None, help="start date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="end date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument(
        "--no-snapshot-range",
        action="store_true",
        help="do not use snapshot date range fallback",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output parquet path (default: config.market_factors_path)",
    )
    parser.add_argument("--retries", type=int, default=3, help="download retries")
    parser.add_argument("--retry-sleep-sec", type=float, default=1.0, help="retry backoff base seconds")
    parser.add_argument("--timeout-sec", type=int, default=20, help="yfinance request timeout seconds")
    parser.add_argument("--krx-request-sleep-sec", type=float, default=0.03, help="sleep seconds between KRX calls")
    parser.add_argument("--no-vix", action="store_true", help="skip VIX collection")
    parser.add_argument("--no-krx-breadth", action="store_true", help="skip KRX breadth collection")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fetch_cfg = MarketFactorFetchConfig(
        retries=max(1, int(args.retries)),
        retry_sleep_sec=max(0.1, float(args.retry_sleep_sec)),
        timeout_sec=max(5, int(args.timeout_sec)),
        krx_request_sleep_sec=max(0.0, float(args.krx_request_sleep_sec)),
    )
    out_path = Path(args.out) if args.out else None
    run_backfill_market_regime_factors(
        start=args.start,
        end=args.end,
        use_snapshot_range=not bool(args.no_snapshot_range),
        out_path=out_path,
        collect_vix=not bool(args.no_vix),
        collect_krx_breadth=not bool(args.no_krx_breadth),
        config=DEFAULT_CONFIG,
        fetch_cfg=fetch_cfg,
    )


if __name__ == "__main__":
    main()
