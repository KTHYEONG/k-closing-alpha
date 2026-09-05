"""Daily materialization of the restored ML training panel (additive artifact)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from src import settings
from src.data.candidate_panel import (
    _PANEL_FLOAT32_COLUMNS,
    build_restored_trade_log,
    check_price_history_freshness,
)
from src.data.parquet_codec import (
    INTRADAY_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    atomic_write_parquet,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Materialize the restored trade-log panel to a new parquet artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Materialize ML training panel")
    parser.add_argument("--trade-log", default=str(settings.TRADE_LOG_PARQUET_PATH))
    parser.add_argument("--theme", default=str(settings.THEME_PARQUET_PATH))
    parser.add_argument("--price-history", default=str(settings.PRICE_HISTORY_PARQUET_PATH))
    parser.add_argument(
        "--condition-history",
        default=str(settings.HISTORY_DIR / "condition_history_cleaned.parquet"),
    )
    parser.add_argument("--out", default=str(settings.PARQUET_DIR / "ml_training_panel.parquet"))
    args = parser.parse_args(argv)

    trade_log_df = pd.read_parquet(args.trade_log)
    price_history_df = pd.read_parquet(args.price_history)
    theme_df = pd.read_parquet(args.theme) if os.path.exists(args.theme) else None

    freshness = check_price_history_freshness(price_history_df)
    if freshness.get("is_stale"):
        logger.warning(
            "[DATA] stage=ml_panel_freshness status=stale max_date=%s staleness_days=%s threshold_days=%s",
            freshness.get("max_date"),
            freshness.get("staleness_days"),
            5,
        )

    result = build_restored_trade_log(
        trade_log_df,
        price_history_df,
        condition_history_path=Path(args.condition_history),
        theme_df=theme_df,
    )

    out_path = Path(args.out)
    # R8 dtype discipline: sheet-synced executed rows arrive as str while
    # reconstructed rows are float32/float64 -- coerce to the panel contract
    # before the parquet write so the artifact never mixes object columns.
    for col in _PANEL_FLOAT32_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")
    for col in ("(수익률, %)", "(매수 가격)", "(매도 가격)"):
        if col in result.columns:
            # 레거시 시트 값은 "5.95%" 형태 문자열이 섞여 있어 % 제거 없이
            # to_numeric 하면 실행 매매 92%가 NaN으로 유실된다 (clean_column_names와 동일 처리).
            if result[col].dtype == object or isinstance(result[col].dtype, pd.StringDtype):
                result[col] = (
                    result[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
    atomic_write_parquet(
        result,
        out_path,
        compression=INTRADAY_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )

    prov = result.attrs.get("panel_restoration", {})
    logger.info(
        "[DATA] stage=ml_panel_materialize rows=%s restored_rows=%s restored_dates=%s execution_offset_pct=%s out=%s",
        len(result),
        prov.get("restored_rows"),
        prov.get("restored_dates"),
        prov.get("execution_offset_pct"),
        str(out_path),
    )

    try:
        # Executed rows in the restored frame are exactly the trade-log rows,
        # so the latest executed trade_date is the trade log's max 매수날짜.
        dates = pd.to_datetime(trade_log_df["매수날짜"], errors="coerce")
        latest = dates.max()
        day_rows = int((dates == latest).sum())
        status = "nominal" if day_rows <= 3 else "elevated"
        logger.info(
            "[DATA] stage=ml_panel_daily_volume date=%s rows=%s status=%s",
            str(pd.Timestamp(latest).date()),
            day_rows,
            status,
        )
    except Exception:  # pragma: no cover
        logger.info("[DATA] stage=ml_panel_daily_volume date=unknown rows=unknown status=unknown")


if __name__ == "__main__":  # pragma: no cover
    main()
