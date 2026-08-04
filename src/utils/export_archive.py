"""export_archive.py
------------------
archive.db / archive.parquet → TSV/CSV 변환 유틸리티 (copy-paste 최적화).

Parquet 아카이브가 존재하면 우선 로드하고, 없으면 SQLite fallback.
컬럼 순서는 ARCHIVE_COLUMN_ORDER (26컬럼) 고정.

Usage:
    uv run python src/utils/export_archive.py [--date YYYY-MM-DD] [--out PATH] [--format tsv|csv]

Example:
    uv run python src/utils/export_archive.py --format tsv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

from src.daily.archive import ARCHIVE_COLUMN_ORDER, fetch_archive_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 기본 경로 상수 ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "exports"


# ── 핵심 함수 ─────────────────────────────────────────────────────────────────
def export_archive_snapshot(
    out_path: Path | str,
    *,
    date: str | None = None,
    fmt: Literal["tsv", "csv"] = "tsv",
    include_header: bool = True,
) -> Path:
    """archive.parquet (또는 archive.db fallback) → TSV/CSV 파일 변환.

    Parquet 존재 시 우선 로드, 없으면 SQLite fallback.
    컬럼 순서: ARCHIVE_COLUMN_ORDER (26컬럼) 고정.
    date=None → 최신 날짜 자동 선택.

    Args:
        out_path: 저장할 출력 파일 경로.
        date: 대상 날짜 (YYYY-MM-DD). None이면 최신 스냅샷.
        fmt: 출력 포맷 (tsv / csv).
        include_header: 헤더 행 포함 여부.

    Returns:
        저장된 파일 경로.

    Raises:
        ValueError: 지정한 날짜의 스냅샷이 아카이브에 없을 때.
    """
    out_path = Path(out_path)
    df = fetch_archive_snapshot(snapshot_date=date)

    if df.empty:
        raise ValueError(
            f"지정한 날짜에 해당하는 스냅샷이 아카이브에 없습니다: {date or '(최신)'}"
        )

    sep = "\t" if fmt == "tsv" else ","
    encoding = "utf-8-sig" if fmt == "csv" else "utf-8"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=ARCHIVE_COLUMN_ORDER).to_csv(
        out_path,
        sep=sep,
        index=False,
        header=include_header,
        encoding=encoding,
    )
    logger.info("저장 완료: %d rows -> %s", len(df), out_path)
    return out_path


# ── CLI 진입점 ────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="archive.db / archive.parquet → TSV/CSV 변환 (copy-paste 최적화)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="날짜 필터 (YYYY-MM-DD). 생략 시 최신 날짜.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 파일 경로. 생략 시 data/exports/archive_{date}.{ext}",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["tsv", "csv"],
        default="tsv",
        help="출력 포맷",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.date:
        snapshot_date = args.date
    else:
        latest = fetch_archive_snapshot()
        if latest.empty:
            raise ValueError("아카이브에 조회 가능한 스냅샷이 없습니다.")
        snapshot_date = str(latest["스냅샷_날짜"].astype(str).iloc[0])

    fmt: Literal["tsv", "csv"] = args.format
    ext = fmt
    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_OUTPUT_DIR / f"archive_{snapshot_date}.{ext}"
    )

    export_archive_snapshot(out_path, date=snapshot_date, fmt=fmt)


if __name__ == "__main__":
    main()
