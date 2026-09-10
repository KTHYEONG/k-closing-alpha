"""저녁 1회 실행: 당일 워치리스트 정규세션+NXT 애프터마켓 1분봉 아카이브."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src import settings
from src.api.kis.client import KisApiClient
from src.api.kiwoom.client import KiwoomApiClient
from src.api.ls.client import LsApiClient
from src.backfill.intraday.collector import (
    collect_intraday_bars,
    collect_intraday_trade_ticks,
    collect_krx_aftermarket_bars,
    collect_nxt_aftermarket_bars,
    collect_nxt_premarket_bars,
)
from src.config.market_session import (
    DEFAULT_BAR_INTERVAL_MINUTES,
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_REGULAR,
)
from src.daily import archive
from src.data.intraday_store import write_intraday_partition, write_tick_partition

logger = logging.getLogger(__name__)

def _today_watchlist_codes(snapshot_date: str) -> list[str]:
    try:
        df = archive.fetch_archive_snapshot(snapshot_date=snapshot_date)
    except Exception as e:
        logger.warning("Watchlist fetch failed date=%s: %s", snapshot_date, e)
        return []
    if df is None or df.empty or "종목코드" not in df.columns:
        return []
    return df["종목코드"].astype(str).str.zfill(6).dropna().unique().tolist()


def resolve_previous_archive_date(snapshot_date: str) -> str | None:
    """아카이브 날짜 인덱스에서 snapshot_date 직전 영업일을 찾는다."""
    try:
        df = archive.fetch_archive_snapshot(all_rows=True)
    except Exception as e:
        logger.warning("[DATA] Previous archive date lookup failed date=%s: %s", snapshot_date, e)
        return None
    if df is None or df.empty or "스냅샷_날짜" not in df.columns:
        return None
    dates = sorted({str(d) for d in df["스냅샷_날짜"].astype(str).tolist() if str(d) < str(snapshot_date)})
    return dates[-1] if dates else None


def _archive_target_codes(snapshot_date: str) -> list[str]:
    """당일 + 직전 아카이브 영업일 워치리스트의 중복 제거 합집합.

    _today_watchlist_codes/resolve_previous_archive_date는 자체적으로 조회 실패를
    흡수해 빈 결과를 반환하므로(never raise), 여기서는 별도 예외 처리가 필요 없다.
    """
    today = _today_watchlist_codes(snapshot_date)
    prev_date = resolve_previous_archive_date(snapshot_date)
    codes: list[str] = list(today)
    if prev_date is not None:
        for code in _today_watchlist_codes(prev_date):
            if code not in codes:
                codes.append(code)
    return codes


def run_intraday_archive(snapshot_date: str | None = None, bar_interval_minutes: int = DEFAULT_BAR_INTERVAL_MINUTES) -> tuple[int, int, int]:
    """당일 워치리스트 정규세션+NXT 애프터마켓 1분봉+정규세션 틱 체결을 세 파티션에 각각 저장. (정규행수, 애프터행수, 틱행수) 반환."""
    snap_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    codes = _archive_target_codes(snap_date)
    if not codes:
        return (0, 0, 0)

    async def _run() -> tuple[int, int, int]:
        client = KisApiClient()
        ls_client = LsApiClient() if getattr(settings, 'LS_APP_KEY', None) else None
        kiwoom_client = KiwoomApiClient() if (getattr(settings, 'KIWOM_APP_KEY', None) or getattr(settings, 'KIWOOM_APP_KEY', None)) else None
        async with client.create_session() as session:
            await client.ensure_token(session)
            bars = await collect_intraday_bars(client, session, codes, snap_date, bar_interval_minutes, ls_client=ls_client)
            nxt_after = await collect_nxt_aftermarket_bars(client, session, codes, snap_date, bar_interval_minutes, kiwoom_client=kiwoom_client)
            nxt_pre = await collect_nxt_premarket_bars(client, session, codes, snap_date, bar_interval_minutes, kiwoom_client=kiwoom_client)
            krx_after = await collect_krx_aftermarket_bars(client, session, codes, snap_date, bar_interval_minutes)
            n_bars = write_intraday_partition(bars, bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
            n_nxt_after = write_intraday_partition(nxt_after, bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_AFTERMARKET)
            n_nxt_pre = write_intraday_partition(nxt_pre, bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_PREMARKET)
            n_krx_after = write_intraday_partition(krx_after, bar_interval_minutes, snap_date, INTRADAY_SESSION_KRX_AFTERMARKET)
            logger.info("[DATA] stage=krx_aftermarket date=%s rows=%d", snap_date, n_krx_after)
            n_nxt = n_nxt_after + n_nxt_pre
            ticks = await collect_intraday_trade_ticks(client, session, codes, snap_date, ls_client=ls_client, kiwoom_client=kiwoom_client)
            n_ticks = write_tick_partition(ticks, snap_date, INTRADAY_SESSION_REGULAR)
            return (n_bars, n_nxt, n_ticks)

    return asyncio.run(_run())


def main() -> None:
    import sys

    from src.utils.display import Colors

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    target_codes = _archive_target_codes(target_date)
    logger.info(
        "🚀 [Intraday 아카이브 시작] 대상일: %s, 대상 종목: %d개, 저장소: %s",
        target_date,
        len(target_codes),
        settings.HISTORY_DIR,
    )
    bars_rows, nxt_rows, tick_rows = run_intraday_archive(snapshot_date=target_date)

    box_top = "━" * 60
    divider = "─" * 60
    logger.info(f"\n{Colors.BOLD}{box_top}{Colors.RESET}")
    logger.info(f" {Colors.GREEN}{Colors.BOLD}📦 [Intraday 분봉/틱 아카이브 완료]{Colors.RESET} (기준일: {target_date})")
    logger.info(f"{Colors.BOLD}{divider}{Colors.RESET}")
    logger.info(f"   • 대상 종목수 : {Colors.CYAN}{len(target_codes):>5}{Colors.RESET} 종목 (당일 + 직전 영업일 워치리스트)")
    logger.info(f"   • 정규 세션   : {Colors.GREEN}{bars_rows:>5,}{Colors.RESET} 행 (1분봉)")
    logger.info(f"   • NXT 세션    : {Colors.GREEN}{nxt_rows:>5,}{Colors.RESET} 행 (프리/애프터마켓)")
    logger.info(f"   • 체결 틱     : {Colors.GREEN}{tick_rows:>5,}{Colors.RESET} 행 (정규장 틱 데이터)")
    logger.info(f"   • 저장 경로   : {settings.HISTORY_DIR}")
    logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")


if __name__ == "__main__":
    main()
