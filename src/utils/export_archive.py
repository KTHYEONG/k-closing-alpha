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
    month: str | None = None,
    fmt: Literal["tsv", "csv"] = "tsv",
    include_header: bool = True,
) -> Path:
    """archive.parquet (또는 archive.db fallback) → TSV/CSV 파일 변환.

    Parquet 존재 시 우선 로드, 없으면 SQLite fallback.
    컬럼 순서: ARCHIVE_COLUMN_ORDER (26컬럼) 고정.
    date 및 month가 None이면 최신월 자동 선택.

    Args:
        out_path: 저장할 출력 파일 경로.
        date: 대상 날짜 (YYYY-MM-DD).
        month: 대상 월 (YYYY-MM).
        fmt: 출력 포맷 (tsv / csv).
        include_header: 헤더 행 포함 여부.

    Returns:
        저장된 파일 경로.

    Raises:
        ValueError: 지정한 조건의 스냅샷이 아카이브에 없을 때.
    """
    out_path = Path(out_path)
    df = fetch_archive_snapshot(snapshot_date=date, month=month)

    if df.empty:
        target_str = date or month or "(최신월)"
        raise ValueError(
            f"지정한 조건에 해당하는 스냅샷이 아카이브에 없습니다: {target_str}"
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


def export_all_months(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    fmt: Literal["tsv", "csv"] = "tsv",
) -> list[Path]:
    """아카이브의 모든 과거 데이터를 월별(YYYY-MM) 개별 파일로 각각 내보냅니다."""
    out_dir = Path(output_dir)
    df_all = fetch_archive_snapshot(all_rows=True)

    if df_all.empty or "스냅샷_날짜" not in df_all.columns:
        raise ValueError("아카이브에 조회 가능한 스냅샷이 없습니다.")

    # YYYY-MM 추출
    df_all["_month"] = df_all["스냅샷_날짜"].astype(str).str.slice(0, 7)
    created_files: list[Path] = []

    for month_str, group in df_all.groupby("_month"):
        group_clean = group.drop(columns=["_month"])
        out_path = out_dir / f"archive_{month_str}.{fmt}"
        sep = "\t" if fmt == "tsv" else ","
        encoding = "utf-8-sig" if fmt == "csv" else "utf-8"

        out_dir.mkdir(parents=True, exist_ok=True)
        group_clean.reindex(columns=ARCHIVE_COLUMN_ORDER).to_csv(
            out_path,
            sep=sep,
            index=False,
            header=True,
            encoding=encoding,
        )
        logger.info("월별 내보내기 완료: %d rows -> %s", len(group_clean), out_path)
        created_files.append(out_path)

    return created_files


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
        help="특정 날짜 필터 (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="특정 월 필터 (YYYY-MM). 생략 시 모든 월을 각각 파일로 내보냄.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 파일 경로/디렉토리. (단일 날짜/월 지정 시 파일 경로, 기본 실행 시 출력 디렉토리)",
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
    fmt: Literal["tsv", "csv"] = args.format

    if args.date:
        out_path = (
            Path(args.out)
            if args.out
            else DEFAULT_OUTPUT_DIR / f"archive_{args.date}.{fmt}"
        )
        export_archive_snapshot(out_path, date=args.date, fmt=fmt)
    elif args.month:
        out_path = (
            Path(args.out)
            if args.out
            else DEFAULT_OUTPUT_DIR / f"archive_{args.month}.{fmt}"
        )
        export_archive_snapshot(out_path, month=args.month, fmt=fmt)
    else:
        # 기본 옵션: 아카이브 내 모든 데이터의 모든 월을 개별 TSV 파일로 각각 출력
        output_dir = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR
        export_all_months(output_dir=output_dir, fmt=fmt)


if __name__ == "__main__":
    main()
