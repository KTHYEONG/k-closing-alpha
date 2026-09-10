"""KIS 종가 단일가매매(15:20~15:30 KST) 구간 API 거동 라이브 관측 스크립트.

목적
----
`decision_timing_and_store_unification` 설계는 "15:20~15:30은 가격이 동결된
구간이라 그 안 어디서 스냅샷을 떠도 동일 결과가 나온다"는 가정 위에 서 있다.
이 가정이 맞는지는 실측 1분봉(사후 기록)으로는 검증할 수 없고, 그 10분 창이
실제로 열려 있을 때 KIS API를 짧은 간격으로 반복 호출해봐야만 확인된다.

이 스크립트는 두 TR을 동시에 폴링한다:
  1) FHKST01010100 (주식현재가 시세, get_current_price)      -> stck_prpr
  2) FHKST01010200 (주식현재가 호가/예상체결, get_orderbook_snapshot)
     -> 예상체결가/예상체결량 계열 필드(정확한 필드명은 실측으로 확정)

확인하려는 것
--------------
  (a) 15:20~15:29:5x 구간에서 stck_prpr(현재가)가 매 호출마다 바뀌는가,
      아니면 15:20 직전 마지막 체결가에 그대로 얼어붙어 있는가?
  (b) 예상체결가(antc_cnpr 등)가 호가 잔량 변화에 따라 시간에 걸쳐 움직이는가?
  (c) 15:30:00~15:30:1x 사이에 값이 실제 종가로 스냅(snap)되는 순간을
      포착할 수 있는가(있다면 그 지연이 얼마인가)?

사용법
------
    uv run python -m src.tools.observe_closing_auction_price_behavior

내일(거래일) 15:19:30경 실행해서 15:30:30까지 자동으로 폴링하고 종료한다.
날짜/시각/종목 리스트는 아래 상수만 바꾸면 된다. 원시 응답 전체를 JSONL로
남기므로, 실행 후 그 JSONL을 다시 분석 스크립트로 돌려 결론을 낸다(이 파일은
관측 전용이며 여기서 결론을 내리지 않는다).

이 파일은 scratch/가 아닌 src/tools/에 둔다 -- scratch/는 /sync 시 정리 대상이라
이 관측 결과를 나중에 다시 참고하려면(또는 다음 관측에 재사용하려면) 영속 위치가
필요하다.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp

from src import settings
from src.api.kis.client import KisApiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

# 오늘 archive에 실제 잡힌 워치리스트에서 유동성 상위 소수만 뽑아 API 호출을
# 최소화한다(계좌/레이트리밋 부담 방지). 필요시 이 리스트만 갈아끼우면 된다.
WATCHLIST_CODES: list[str] = [
    "005930",  # 삼성전자 (KRX+NXT 유동성 최상위, 기준 종목)
    "000660",  # SK하이닉스
]

POLL_START_HHMMSS = "152000"  # 관측 시작 (연속거래 종료 직후)
POLL_END_HHMMSS = "153030"  # 관측 종료 (단일가 체결 확정 이후 30초까지)
POLL_INTERVAL_SEC = 5.0

OUT_PATH = settings.DATA_DIR / "diagnostics" / "closing_auction_observation.jsonl"

APP_KEY = settings.KIS_API_CONFIG["app_key"]
APP_SECRET = settings.KIS_API_CONFIG["app_secret"]
ACCOUNT_ID = settings.KIS_API_CONFIG.get("account_id", "")
HTS_ID = settings.KIS_API_CONFIG.get("hts_id")
TOKEN_FILE = str(settings.TOKEN_FILE)


def _now_kst() -> datetime:
    return datetime.now(KST)


def _hhmmss(dt: datetime) -> str:
    return dt.strftime("%H%M%S")


def _find_antc_fields(node: object, path: str = "") -> dict[str, object]:
    """예상체결(antc_*) 관련 필드를 응답 어디에 있든(output1/output2/기타) 재귀적으로 찾는다.

    실제 KIS 응답에서 예상체결가/예상체결량이 output1에 있는지 output2에
    있는지(또는 아예 이 TR에 없는지) 사전에 확신할 수 없으므로, 특정 블록만
    보고 "없다"고 단정하지 않기 위해 응답 트리 전체를 훑는다(orderbook_store.py가
    output1만 저장하는 기존 프로덕션 버그와 같은 함정을 관측 단계에서부터 피한다).
    """
    found: dict[str, object] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else str(key)
            if "antc" in str(key).lower():
                found[key_path] = value
            found.update(_find_antc_fields(value, key_path))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.update(_find_antc_fields(item, f"{path}[{i}]"))
    return found


async def _poll_once(client: KisApiClient, session: aiohttp.ClientSession, code: str) -> dict[str, object]:  # pragma: no cover - live network call, manually verified
    """한 종목에 대해 현재가 + 호가/예상체결을 동시에 조회해 원시 응답을 그대로 반환."""
    price_task = client.get_current_price(session, code)  # type: ignore[no-untyped-call]
    book_task = client.get_orderbook_snapshot(session, code, market_div_code="J")
    price_res, book_res = await asyncio.gather(price_task, book_task, return_exceptions=True)

    def _safe(res: object) -> dict[str, object] | str:
        if isinstance(res, Exception):
            return f"EXC:{type(res).__name__}:{res}"
        return res  # type: ignore[return-value]

    return {"code": code, "current_price": _safe(price_res), "orderbook": _safe(book_res)}


async def main() -> None:  # pragma: no cover - live market-hours orchestration loop, manual CLI use only
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("관측 대상: %s | 구간 %s~%s KST | 간격 %.1fs | 기록: %s",
                WATCHLIST_CODES, POLL_START_HHMMSS, POLL_END_HHMMSS, POLL_INTERVAL_SEC, OUT_PATH)

    now = _now_kst()
    if _hhmmss(now) < POLL_START_HHMMSS:
        wait_sec = (
            datetime.strptime(POLL_START_HHMMSS, "%H%M%S")
            .replace(year=now.year, month=now.month, day=now.day, tzinfo=KST)
            - now
        ).total_seconds()
        if 0 < wait_sec <= 3600:
            logger.info("관측 시작까지 %.0f초 대기...", wait_sec)
            await asyncio.sleep(wait_sec)
        elif wait_sec > 3600:
            logger.warning(
                "관측 시작까지 %.0f분 남음 -- 스크립트를 15:19~15:20경 다시 실행하세요.", wait_sec / 60
            )
            return

    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = KisApiClient(APP_KEY, APP_SECRET, ACCOUNT_ID, HTS_ID, token_file=TOKEN_FILE)  # type: ignore[no-untyped-call]
        await client.ensure_token(session)

        n_polls = 0
        with OUT_PATH.open("a", encoding="utf-8") as fh:
            while True:
                loop_start = _now_kst()
                hhmmss = _hhmmss(loop_start)
                if hhmmss > POLL_END_HHMMSS:
                    break

                results = await asyncio.gather(
                    *(_poll_once(client, session, code) for code in WATCHLIST_CODES)
                )
                capture_ts = _now_kst()
                record = {
                    "capture_ts": capture_ts.isoformat(),
                    "capture_hhmmss": _hhmmss(capture_ts),
                    "results": results,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                n_polls += 1

                for r in results:
                    cp = r["current_price"]
                    prpr = cp.get("output", {}).get("stck_prpr") if isinstance(cp, dict) else None
                    ob = r["orderbook"]
                    antc_fields = _find_antc_fields(ob) if isinstance(ob, dict) else {}
                    logger.info(
                        "[%s] code=%s stck_prpr=%s antc_fields=%s",
                        _hhmmss(capture_ts), r["code"], prpr, antc_fields or "(없음 -- output1/output2 어디에도 antc_* 키 없음)",
                    )

                elapsed = (_now_kst() - loop_start).total_seconds()
                sleep_left = max(0.0, POLL_INTERVAL_SEC - elapsed)
                await asyncio.sleep(sleep_left)

        logger.info("관측 완료: %d회 폴링, 기록: %s", n_polls, OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    asyncio.run(main())
