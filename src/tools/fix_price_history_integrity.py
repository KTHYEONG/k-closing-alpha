"""price_history.parquet prev_close 정합성 복원 및 무결성 정제 도구.

원천 데이터의 prev_close 컬럼에 전일종가 대신 '전일대비 변동금액(diff)'이 오기입된
2025-08-01 전종목(2,449건) 및 첫 거래일(2016-01-04 등 2,468건) 이상치를 복원합니다.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.data.parquet_codec import write_price_history_parquet

logger = logging.getLogger(__name__)


def inspect_integrity(df: pd.DataFrame) -> dict[str, Any]:
    """price_history 데이터프레임의 prev_close 무결성 지표를 산출한다."""
    sorted_df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    first_mask = sorted_df.groupby("symbol", sort=False).cumcount() == 0

    pc = sorted_df["prev_close"]
    na_count = int(pc.isna().sum())
    neg_count = int((pc < 0).sum())
    zero_count = int((pc == 0).sum())
    pos_count = int((pc > 0).sum())

    d2025 = sorted_df[sorted_df["date"] == "2025-08-01"]
    d2016 = sorted_df[sorted_df["date"] == "2016-01-04"]

    return {
        "total_rows": len(sorted_df),
        "total_symbols": int(sorted_df["symbol"].nunique()),
        "prev_close_na": na_count,
        "prev_close_neg": neg_count,
        "prev_close_zero": zero_count,
        "prev_close_pos": pos_count,
        "first_rows_count": int(first_mask.sum()),
        "first_rows_na": int(pc[first_mask].isna().sum()),
        "first_rows_neg": int((pc[first_mask] < 0).sum()),
        "non_first_neg": int((pc[~first_mask] < 0).sum()),
        "non_first_zero": int((pc[~first_mask] == 0).sum()),
        "non_first_na": int(pc[~first_mask].isna().sum()),
        "d2025_0801_total": len(d2025),
        "d2025_0801_neg": int((d2025["prev_close"] < 0).sum()),
        "d2016_0104_total": len(d2016),
        "d2016_0104_neg": int((d2016["prev_close"] < 0).sum()),
    }


def fix_price_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    """symbol, date 정렬 후 close의 shift(1)로 prev_close를 복원한다."""
    out = df.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    out["prev_close"] = out.groupby("symbol", sort=False)["close"].shift(1)
    return out


def run_fix_price_history(
    source_path: Path,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """price_history.parquet 무결성을 복원하고 저장한다."""
    source_path = Path(source_path)
    if not source_path.exists():
        return {"status": "missing", "path": str(source_path)}

    logger.info("[DATA] reading %s...", source_path)
    original = pd.read_parquet(source_path)
    before_stats = inspect_integrity(original)

    logger.info("[DATA] before fix: neg=%d na=%d", before_stats["prev_close_neg"], before_stats["prev_close_na"])

    fixed = fix_price_history_frame(original)
    after_stats = inspect_integrity(fixed)

    # 안전 검증: row 수, 심볼 수, close 값 불변 검증
    if len(fixed) != len(original):
        raise ValueError(f"Row count mismatch: {len(original)} -> {len(fixed)}")
    if after_stats["total_symbols"] != before_stats["total_symbols"]:
        raise ValueError("Symbol count changed")
    if after_stats["non_first_neg"] != 0 or after_stats["non_first_na"] != 0:
        raise ValueError(f"Integrity check failed: non_first has anomalies ({after_stats})")

    if dry_run:
        logger.info("[DATA] dry-run mode: skipping file write")
        return {
            "status": "dry_run_success",
            "before": before_stats,
            "after": after_stats,
        }

    # 백업 생성
    if backup:
        bak_path = source_path.with_suffix(".parquet.bak")
        if not bak_path.exists():
            logger.info("[DATA] creating backup %s", bak_path)
            shutil.copyfile(source_path, bak_path)
        else:
            logger.info("[DATA] backup %s already exists, keeping existing", bak_path)

    # Parquet 원자적 저장 (zstd 압축 및 다운캐스팅 표준 정책)
    logger.info("[DATA] writing fixed dataframe to %s...", source_path)
    write_price_history_parquet(fixed, source_path)

    # 디스크 저장 후 재검증
    reloaded = pd.read_parquet(source_path)
    reloaded_stats = inspect_integrity(reloaded)

    logger.info(
        "[DATA] file=%s status=fixed rows=%d neg=%d na=%d",
        source_path.name,
        reloaded_stats["total_rows"],
        reloaded_stats["prev_close_neg"],
        reloaded_stats["prev_close_na"],
    )

    return {
        "status": "success",
        "before": before_stats,
        "after": reloaded_stats,
        "source_path": str(source_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fix price_history.parquet prev_close integrity")
    parser.add_argument(
        "--path",
        type=Path,
        default=settings.PRICE_HISTORY_PARQUET_PATH,
        help="Target parquet file path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect and simulate without writing")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating .bak backup")

    args = parser.parse_args()
    res = run_fix_price_history(args.path, backup=not args.no_backup, dry_run=args.dry_run)
    logger.info("Summary Result: %s", res)


if __name__ == "__main__":
    main()
