"""종목별 과거 가격 백필 호환성 퍼사드 모듈.

구현은 ``src.backfill.price.*`` 로 분리되었으며, 이 모듈은 공개 심볼과 CLI
``main`` 만 재-export 합니다. 중복 구현이 없고 마이그레이션 기간 동안 기존
import 경로를 보장합니다.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.backfill.price.config import (
    DEFAULT_CONFIG,
    KRW_100M,
    FetchConfig,
    PipelineConfig,
    _effective_kis_sleep_sec,
    _ensure_kis_semaphore,
    _kis_slot,
)
from src.backfill.price.factors import (
    _merge_index_returns,
    compute_vkospi_proxy,
)
from src.backfill.price.normalize import (
    _find_col,
    _normalize_investor_flow,
    _normalize_symbol_history,
    _subtract_flow_frames,
    _sum_present_cols,
    _to_ymd,
)
from src.backfill.price.runner import (
    fetch_one_symbol,
    preview_windows,
    run_backfill,
)
from src.backfill.price.sources import (
    _fetch_index_returns,
    _fetch_investor_history_by_date,
    _fetch_kis_daily_ohlcv,
    _fetch_program_history_by_date,
    _fetch_vkospi_proxy,
    _kis_sync_client,
    _resolve_investor_history_func,
    _resolve_program_history_func,
    _safe_get_market_cap_by_date,
    _safe_get_market_ohlcv_by_date,
    _safe_get_trading_value_by_date,
)
from src.backfill.price.universe import (
    _build_symbol_windows,
    _load_candidate_universe,
    load_or_build_snapshot,
)

logger = logging.getLogger(__name__)

try:
    from pykrx import stock
except ImportError:  # pragma: no cover - dependency availability differs by env
    stock = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill per-symbol historical bars for rolling-feature support."
    )
    parser.add_argument("--lookback-days", type=int, default=40, help="target trading-day lookback")
    parser.add_argument("--start-date", type=str, default="", help="full-history start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="", help="full-history end date (YYYY-MM-DD)")
    parser.add_argument("--skip-flows", action="store_true", help="skip investor/program flow APIs")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers")
    parser.add_argument(
        "--kis-rest-rps",
        type=float,
        default=20.0,
        help="KIS REST nominal limit (req/s).",
    )
    parser.add_argument(
        "--kis-safety-ratio",
        type=float,
        default=0.6,
        help="KIS safety ratio in (0,1]. effective_rps = kis_rest_rps * ratio.",
    )
    parser.add_argument(
        "--kis-max-parallel",
        type=int,
        default=1,
        help="Maximum parallel KIS sections in this process.",
    )
    parser.add_argument(
        "--pykrx-rps",
        type=float,
        default=8.0,
        help="Process-wide pykrx request limit (requests/sec).",
    )
    parser.add_argument("--limit-symbols", type=int, default=None, help="debug symbol cap")
    parser.add_argument("--symbols", type=str, default="", help="comma separated symbol list, e.g. 005930,000660")
    parser.add_argument("--dry-run", action="store_true", help="print fetch windows only, no API calls")
    parser.add_argument(
        "--parquet-out",
        type=str,
        default="data/history/price_history.parquet",
        help="output parquet path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    parquet_out = Path(args.parquet_out) if str(args.parquet_out).strip() else None
    if parquet_out is None:
        raise ValueError("parquet output path is required.")

    include_symbols = None
    if str(args.symbols).strip():
        include_symbols = {
            s.strip().zfill(6)
            for s in str(args.symbols).split(",")
            if s.strip()
        }
        if not include_symbols:
            include_symbols = None

    if args.dry_run:
        plan = preview_windows(
            lookback_trading_days=args.lookback_days,
            symbol_limit=args.limit_symbols,
            include_symbols=include_symbols,
            parquet_out=parquet_out,
        )
        if plan.empty:
            logger.info("[backfill] status=no_matching_symbols mode=dry_run")
            return
        logger.info("[backfill] mode=dry_run symbols=%d", len(plan))
        logger.info("[backfill] mode=dry_run_plan\\n%s", plan.to_string(index=False))
        return

    if stock is None:
        raise RuntimeError("pykrx is required. Install with: python -m pip install pykrx")

    run_backfill(
        lookback_trading_days=args.lookback_days,
        max_workers=args.workers,
        kis_rest_limit_per_sec=args.kis_rest_rps,
        kis_rest_safety_ratio=args.kis_safety_ratio,
        kis_max_parallel_calls=args.kis_max_parallel,
        pykrx_requests_per_sec=args.pykrx_rps,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        include_flows=not args.skip_flows,
        symbol_limit=args.limit_symbols,
        include_symbols=include_symbols,
        parquet_out=parquet_out,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "KRW_100M",
    "FetchConfig",
    "PipelineConfig",
    "_build_symbol_windows",
    "_effective_kis_sleep_sec",
    "_ensure_kis_semaphore",
    "_fetch_index_returns",
    "_fetch_investor_history_by_date",
    "_fetch_kis_daily_ohlcv",
    "_fetch_program_history_by_date",
    "_fetch_vkospi_proxy",
    "_find_col",
    "_kis_slot",
    "_kis_sync_client",
    "_load_candidate_universe",
    "_merge_index_returns",
    "_normalize_investor_flow",
    "_normalize_symbol_history",
    "_resolve_investor_history_func",
    "_resolve_program_history_func",
    "_safe_get_market_cap_by_date",
    "_safe_get_market_ohlcv_by_date",
    "_safe_get_trading_value_by_date",
    "_subtract_flow_frames",
    "_sum_present_cols",
    "_to_ymd",
    "compute_vkospi_proxy",
    "fetch_one_symbol",
    "load_or_build_snapshot",
    "main",
    "preview_windows",
    "run_backfill",
]


if __name__ == "__main__":  # pragma: no cover
    main()
