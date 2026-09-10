"""WSL 부팅 시 1회 실행되는 당일 결손 가시화 감사.

누락 단계가 있어도 비정상 종료하지 않는다(부팅 감사용 가시화 전용).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import settings
from src.daily.archive import fetch_archive_snapshot
from src.data.intraday_store import intraday_partition_path
from src.processing.schema import CLOSE_CONFIRMED_COL

logger = logging.getLogger(__name__)


def audit_daily_completeness(snapshot_date: str) -> dict[str, bool]:
    """해당 일자의 단계별 산출물 존재 여부를 bool 딕셔너리로 반환한다.

    존재 판정은 파일/행 존재만으로 하며 값 검증은 하지 않는다.
    """
    frame = fetch_archive_snapshot(snapshot_date)
    archive_ok = frame is not None and len(frame) > 0
    minute_bars_ok = intraday_partition_path(1, snapshot_date, "regular").exists()
    store = Path(settings.PAPER_DIR) / "decisions.parquet"
    decisions = pd.read_parquet(store) if store.exists() else pd.DataFrame()
    decision_ok = (
        (not decisions.empty)
        and "decision_date" in decisions.columns
        and bool((decisions["decision_date"].astype(str) == snapshot_date).any())
    )
    close_confirmed_ok = (
        archive_ok
        and CLOSE_CONFIRMED_COL in frame.columns
        and bool(frame[CLOSE_CONFIRMED_COL].fillna(False).astype(bool).any())
    )
    return {
        "archive": bool(archive_ok),
        "minute_bars": bool(minute_bars_ok),
        "decision": bool(decision_ok),
        "close_confirmed": bool(close_confirmed_ok),
    }


def main() -> None:  # pragma: no cover - CLI entry; logic covered via audit_daily_completeness scenarios
    parser = argparse.ArgumentParser(description="Daily completeness audit (boot-time visibility only)")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today)")
    args = parser.parse_args()
    snapshot_date: str = args.date or pd.Timestamp.today().strftime("%Y-%m-%d")
    result = audit_daily_completeness(snapshot_date)
    missing = sorted(step for step, ok in result.items() if not ok)
    if missing:
        logger.warning("[DATA] stage=daily_audit status=MISSING steps=%s", ",".join(missing))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
