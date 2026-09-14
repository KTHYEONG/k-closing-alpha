"""매 평일 EOD 이후 스케줄 실행되는 자동화 점검 + 일일 요약 발송.

평일마다 정확히 한 통의 요약(정상/경고/휴장일)을 보낸다. 요약이 오지 않는 것
자체가 스케줄러 중단 신호가 되도록 누락이 없어도 침묵하지 않는다. 누락 단계가
있어도 비정상 종료하지 않는다(가시화 전용).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

import aiohttp
import pandas as pd

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.daily.archive import fetch_archive_snapshot
from src.daily.archive_intraday import resolve_previous_archive_date
from src.data.intraday_store import intraday_partition_path
from src.data.trading_calendar import is_kis_trading_day
from src.processing.schema import CLOSE_CONFIRMED_COL
from src.tools.alerts import dispatch_digest

logger = logging.getLogger(__name__)

DAY_WEEKEND: str = "weekend"
DAY_HOLIDAY: str = "holiday"
DAY_TRADING: str = "trading"
# 달력 조회 장애: 휴장일로 단정하지 않고 감사를 수행한다(장애 조기 발견 우선)
DAY_UNKNOWN: str = "unknown"
AUDIT_STEPS: tuple[str, ...] = (
    "archive",
    "close_confirmed",
    "decision",
    "paper_entry",
    "minute_bars",
    "price_history_fresh",
)
SYSTEMCTL_TIMEOUT_SEC: int = 30


def _column_dates(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path)
    if frame.empty or column not in frame.columns:
        return set()
    return {str(value)[:10] for value in frame[column].dropna().tolist()}


def _entry_fill_dates(path: Path) -> set[str]:
    if not path.exists():
        return set()
    fills = pd.read_parquet(path)
    if fills.empty or not {"side", "decision_date"}.issubset(fills.columns):
        return set()
    buys = fills.loc[fills["side"].astype(str) == "buy", "decision_date"]
    return {str(value)[:10] for value in buys.dropna().tolist()}


def _price_history_fresh(snapshot_date: str) -> bool:
    # 감사가 당일 야간 적재(21:30) 전에 돌 수 있으므로 직전 아카이브 영업일까지 적재됐는지로 판정한다
    previous = resolve_previous_archive_date(snapshot_date)
    if previous is None:
        return True
    path = Path(settings.PRICE_HISTORY_PARQUET_PATH)
    if not path.exists():
        return False
    dates = pd.read_parquet(path, columns=["date"])["date"]
    if dates.empty:
        return False
    return str(pd.Timestamp(dates.max()).date()) >= previous


def audit_daily_completeness(snapshot_date: str) -> dict[str, bool]:
    """해당 일자의 단계별 산출물 존재 여부를 AUDIT_STEPS 키 bool 딕셔너리로 반환한다.

    존재 판정은 파일/행 존재만으로 하며 값 검증은 하지 않는다. 결정과 페이퍼 진입은
    결정이 없던 날의 명시적 '결정 없음' 기록도 수행된 것으로 인정한다.

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD, KST).

    Returns:
        AUDIT_STEPS 각 단계의 수행 여부.
    """
    frame = fetch_archive_snapshot(snapshot_date)
    archive_ok = frame is not None and len(frame) > 0
    close_confirmed_ok = (
        archive_ok
        and CLOSE_CONFIRMED_COL in frame.columns
        and bool(frame[CLOSE_CONFIRMED_COL].fillna(False).astype(bool).any())
    )
    topk_dates = _column_dates(Path(settings.PARQUET_DIR) / "topk_decisions.parquet", "decision_date")
    no_decision_dates = _column_dates(Path(settings.PAPER_DIR) / "decisions.parquet", "decision_date")
    entry_dates = _entry_fill_dates(Path(settings.PAPER_DIR) / "fills.parquet")
    return {
        "archive": bool(archive_ok),
        "close_confirmed": bool(close_confirmed_ok),
        "decision": snapshot_date in topk_dates or snapshot_date in no_decision_dates,
        "paper_entry": snapshot_date in entry_dates or snapshot_date in no_decision_dates,
        "minute_bars": bool(intraday_partition_path(1, snapshot_date, "regular").exists()),
        "price_history_fresh": _price_history_fresh(snapshot_date),
    }


def _kis_trading_day(snapshot_date: str) -> bool:  # pragma: no cover - live KIS boundary
    async def _run() -> bool:
        client = KisApiClient(**kis_data_client_kwargs())  # type: ignore[no-untyped-call]
        async with client.create_session() as session:
            await client.ensure_token(session)
            return await is_kis_trading_day(client, session, snapshot_date)

    return asyncio.run(_run())


def classify_day(snapshot_date: str, trading_day_fn: Callable[[str], bool] | None = None) -> str:
    """점검 대상일을 주말/휴장일/거래일/미상 중 하나로 분류한다.

    KIS 지수 일별시세는 당일 게시되므로 휴장일 판정에 쓴다(KRX 공식 지수는 1일 이상
    지연 게시되어 당일 휴장일 판정에 쓸 수 없다). 조회 장애는 미상으로 낮춰 감사를
    막지 않는다.

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD).
        trading_day_fn: 거래일 오라클. None이면 KIS 실조회를 사용한다.

    Returns:
        DAY_WEEKEND, DAY_HOLIDAY, DAY_TRADING, DAY_UNKNOWN 중 하나.
    """
    if pd.Timestamp(snapshot_date).weekday() >= 5:
        return DAY_WEEKEND
    oracle = trading_day_fn if trading_day_fn is not None else _kis_trading_day
    try:
        return DAY_TRADING if oracle(snapshot_date) else DAY_HOLIDAY
    except (RuntimeError, OSError, aiohttp.ClientError) as exc:
        logger.warning("[DATA] stage=daily_audit calendar_lookup=FAIL reason=%s", type(exc).__name__)
        return DAY_UNKNOWN


def list_failed_kca_units(run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[str]:
    """systemd 유저 매니저에서 failed 상태인 kca-* 유닛 이름을 정렬해 반환한다.

    systemctl 자체를 실행할 수 없으면 빈 목록으로 숨기지 않고 표식 문자열을 반환해
    요약에 드러나게 한다.

    Args:
        run_fn: subprocess.run 호환 실행기(테스트 주입용).

    Returns:
        실패 유닛 이름 목록 또는 조회 불가 표식 1건.
    """
    try:
        result = run_fn(
            ["systemctl", "--user", "list-units", "--failed", "--plain", "--no-legend", "kca-*"],
            capture_output=True,
            text=True,
            timeout=SYSTEMCTL_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"<systemctl unavailable: {type(exc).__name__}>"]
    return sorted(line.split()[0] for line in result.stdout.splitlines() if line.strip())


def build_digest(
    snapshot_date: str, day_kind: str, result: dict[str, bool] | None, failed_units: list[str]
) -> tuple[str, str]:
    """일일 요약의 (제목, 본문)을 만든다.

    Args:
        snapshot_date: 점검 대상일.
        day_kind: classify_day 결과(주말 제외).
        result: audit_daily_completeness 결과. 휴장일에만 None을 허용한다.
        failed_units: list_failed_kca_units 결과.

    Returns:
        (제목, 본문) 튜플.

    Raises:
        ValueError: 휴장일이 아닌데 result가 None인 경우.
    """
    lines = [f"date={snapshot_date}", f"day={day_kind}"]
    missing: list[str] = []
    if day_kind != DAY_HOLIDAY:
        if result is None:
            raise ValueError(f"audit result required for day_kind={day_kind!r}")
        missing = [step for step in AUDIT_STEPS if not result.get(step, False)]
        lines += [f"{step}={'OK' if result.get(step, False) else 'MISSING'}" for step in AUDIT_STEPS]
    lines.append(f"failed_units={','.join(failed_units) if failed_units else 'none'}")
    if not missing and not failed_units:
        label = "휴장일 SKIP" if day_kind == DAY_HOLIDAY else "일일점검 OK"
        return f"[KCA] {snapshot_date} {label}", "\n".join(lines)
    problems = []
    if missing:
        problems.append(f"누락 {','.join(missing)}")
    if failed_units:
        problems.append(f"실패유닛 {','.join(failed_units)}")
    return f"[KCA] {snapshot_date} 일일점검 경고: {' / '.join(problems)}", "\n".join(lines)


def run_daily_audit(
    snapshot_date: str,
    *,
    trading_day_fn: Callable[[str], bool] | None = None,
    failed_units_fn: Callable[[], list[str]] = list_failed_kca_units,
    dispatch_fn: Callable[[str, str], dict[str, bool]] = dispatch_digest,
) -> str | None:
    """평일 1회 점검 후 요약을 발송한다. 주말이면 아무것도 보내지 않는다.

    Args:
        snapshot_date: 점검 대상일(YYYY-MM-DD, KST).
        trading_day_fn: 거래일 오라클 주입(테스트용).
        failed_units_fn: 실패 유닛 조회 주입(테스트용).
        dispatch_fn: 요약 발송 주입(테스트용).

    Returns:
        발송한 요약 제목. 주말이면 None.
    """
    day_kind = classify_day(snapshot_date, trading_day_fn)
    if day_kind == DAY_WEEKEND:
        logger.info("[DATA] stage=daily_audit status=SKIP reason=weekend date=%s", snapshot_date)
        return None
    result = None if day_kind == DAY_HOLIDAY else audit_daily_completeness(snapshot_date)
    subject, body = build_digest(snapshot_date, day_kind, result, failed_units_fn())
    logger.info("[DATA] stage=daily_audit day=%s subject=%s", day_kind, subject)
    dispatch_fn(subject, body)
    return subject


def main() -> None:  # pragma: no cover - CLI entry; logic covered via run_daily_audit scenarios
    parser = argparse.ArgumentParser(description="Daily automation audit + digest (every weekday after EOD)")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today in KST)")
    args = parser.parse_args()
    snapshot_date: str = args.date or pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d")
    run_daily_audit(snapshot_date)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
