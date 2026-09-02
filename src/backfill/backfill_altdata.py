"""CLI entrypoint for historical alt-data backfill (shorting/fundamental/investor/derivatives/disclosure).

Implemented per docs/specs/ml_altdata_backfill_contract.json.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import settings
from src.backfill.altdata.config import AltDataFetchConfig, _ALTDATA_PANELS
from src.backfill.altdata.runner import run_altdata_backfill

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Parse alt-data backfill arguments and dispatch to the collection runner.

    Args:
        argv: CLI 인자 목록.
    """
    parser = argparse.ArgumentParser(description="Backfill historical alt-data panels.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="panel name (repeatable or comma-separated, default all)",
    )
    parser.add_argument("--start", type=str, default="2016-01-01", help="start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="", help="end date YYYY-MM-DD")
    parser.add_argument("--out-dir", type=str, default=str(settings.ALTDATA_DIR), help="output directory")
    parser.add_argument(
        "--dart-key",
        type=str,
        default=(settings.OPENDART_API_KEY or settings.DART_API_KEY),
        help="OpenDART API key",
    )
    parser.add_argument(
        "--krx-key", type=str, default=settings.KRX_OPENAPI_KEY, help="KRX Open API AUTH_KEY"
    )
    parser.add_argument("--pykrx-rps", type=float, default=6.0, help="pykrx requests per sec")
    parser.add_argument("--dart-rps", type=float, default=8.0, help="DART requests per sec")
    parser.add_argument("--krx-rps", type=float, default=4.0, help="KRX Open API requests per sec")
    parser.add_argument(
        "--trade-log-universe",
        action="store_true",
        help="filter to symbols ever present in trade_log.parquet",
    )
    args = parser.parse_args(argv)

    # Resolve sources
    raw_sources: list[str] = []
    if args.source:
        for item in args.source:
            for part in str(item).split(","):
                p = part.strip()
                if p:
                    raw_sources.append(p)
    if not raw_sources:
        sources: tuple[str, ...] = tuple(_ALTDATA_PANELS.keys())
    else:
        # Validate against registry
        for s in raw_sources:
            if s not in _ALTDATA_PANELS:
                parser.error(f"invalid source '{s}', allowed: {list(_ALTDATA_PANELS.keys())}")
        sources = tuple(raw_sources)

    start = pd.Timestamp(args.start) if str(args.start).strip() else pd.Timestamp("2016-01-01")
    end_raw = str(args.end).strip()
    if end_raw:
        end = pd.Timestamp(end_raw)
    else:
        # Use today as derived from settings? Use pandas today
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else settings.ALTDATA_DIR

    dart_key = str(args.dart_key or "").strip()
    krx_key = str(args.krx_key or "").strip()
    pykrx_rps = float(args.pykrx_rps)
    dart_rps = float(args.dart_rps)
    krx_rps = float(args.krx_rps)

    universe_symbols: frozenset[str] | None = None
    if args.trade_log_universe:
        try:
            trade_path = settings.TRADE_LOG_PARQUET_PATH
            if trade_path.exists():
                df = pd.read_parquet(trade_path)
                # Find stock_code column
                col = None
                for cand in ["stock_code", "종목코드", "symbol", "code"]:
                    if cand in df.columns:
                        col = cand
                        break
                if col is not None:
                    syms = set(df[col].astype(str).str.strip().str.zfill(6).tolist())
                    syms = {s for s in syms if s and s != "000000" and s.lower() != "nan"}
                    if syms:
                        universe_symbols = frozenset(syms)
        except Exception as exc:
            logger.warning("[DATA] stage=altdata_universe status=FAIL error=%s", exc)

    cfg = AltDataFetchConfig(
        start=start,
        end=end,
        out_dir=out_dir,
        sources=sources,
        dart_api_key=dart_key,
        krx_api_key=krx_key,
        pykrx_requests_per_sec=pykrx_rps,
        dart_requests_per_sec=dart_rps,
        krx_requests_per_sec=krx_rps,
        universe_symbols=universe_symbols,
    )
    manifest = run_altdata_backfill(cfg)
    for panel, info in manifest.get("panels", {}).items():
        logger.info(
            "[DATA] panel=%s status=%s rows=%s first_date=%s last_date=%s",
            panel,
            info.get("status"),
            info.get("rows"),
            info.get("first_date"),
            info.get("last_date"),
        )


if __name__ == "__main__":
    main()
