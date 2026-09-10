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
from src.data.trading_calendar import is_krx_trading_day
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


def audit_or_skip(snapshot_date: str) -> dict[str, bool] | None:
    """비거래일에는 감사를 건너뛰고, 거래일에는 기존 감사를 그대로 수행한다.

    휴장일에는 산출물이 없어 MISSING 오탐만 찍히므로 감사 자체를 수행하지
    않는다. 거래일 판정 장애(네트워크/인증)는 휴장일로 오판하지 않고 그대로
    전파한다.
    """
    if not is_krx_trading_day(snapshot_date):
        logger.info("[DATA] stage=daily_audit status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return None
    return audit_daily_completeness(snapshot_date)


def main() -> None:  # pragma: no cover - CLI entry; logic covered via audit_daily_completeness scenarios
    parser = argparse.ArgumentParser(description="Daily completeness audit (boot-time visibility only)")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today)")
    args = parser.parse_args()
    snapshot_date: str = args.date or pd.Timestamp.today().strftime("%Y-%m-%d")
    result = audit_or_skip(snapshot_date)
    if result is None:
        return
    missing = sorted(step for step, ok in result.items() if not ok)
    if missing:
        logger.warning("[DATA] stage=daily_audit status=MISSING steps=%s", ",".join(missing))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
