"""KIS 실시간 체결틱 웹소켓 클라이언트 (페이퍼 체결 오라클 수신 전용).

주문 전송 TR(TTTC*)을 어떤 형태로도 참조하지 않는다. H0STCNT0 실체결
프린트만을 구독해 페이퍼 체결 판정의 오라클로 흘려보낸다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import aiohttp

from src import settings

logger = logging.getLogger(__name__)

# 실전 실시간 도메인. 실증 확인: H0STCNT0 SUBSCRIBE SUCCESS.
KIS_WS_URL: str = "ws://ops.koreainvestment.com:21000"
KIS_WS_MAX_SUBSCRIPTIONS: int = 41


async def issue_approval_key(
    session: aiohttp.ClientSession, app_key: str, app_secret: str, base_url: str | None = None
) -> str:
    """웹소켓 접속용 approval_key를 발급한다. 응답에 키가 없으면 ValueError."""
    url = f"{base_url or settings.KIS_BASE_URL}/oauth2/Approval"
    body = {"grant_type": "client_credentials", "appkey": app_key, "secretkey": app_secret}
    async with session.post(url, json=body) as resp:
        data = await resp.json()
    key = data.get("approval_key") if isinstance(data, dict) else None
    if not key:
        raise ValueError(f"approval_key missing in Approval response: {data}")
    return str(key)


def parse_realtime_frame(raw: str) -> list[tuple[str, str, int]]:
    """H0STCNT0 데이터 프레임을 (symbol, hhmmss, price) 리스트로 파싱한다.

    JSON 제어 프레임(구독응답/PINGPONG)과 암호화 프레임('1|')은 빈 리스트를
    반환한다. 필드 수 부족·비수치 가격은 해당 건만 건너뛰되 WARNING을 남긴다.
    """
    text = raw.strip()
    if text.startswith("{") or text.startswith("1|"):
        return []
    parts = text.split("|")
    assert len(parts) >= 4, f"not an H0STCNT0 frame: {text[:64]}"
    assert parts[0] == "0", f"not an H0STCNT0 frame: {text[:64]}"
    assert parts[1] == "H0STCNT0", f"not an H0STCNT0 frame: {text[:64]}"
    count = int(parts[2])
    fields = parts[3:]
    prints: list[tuple[str, str, int]] = []
    for i in range(count):
        chunk = fields[i * 41 : (i + 1) * 41]
        if len(chunk) < 3:
            logger.warning("[DATA] stage=ws_parse status=MALFORMED detail=short_fields idx=%d", i)
            continue
        try:
            price = int(chunk[2])
        except ValueError:
            logger.warning("[DATA] stage=ws_parse status=MALFORMED detail=bad_price symbol=%s", chunk[0])
            continue
        prints.append((chunk[0], chunk[1], price))
    return prints


class KisWebSocketClient:
    """H0STCNT0 실체결 프린트 구독 클라이언트. 주문 TR은 절대 참조하지 않는다."""

    def __init__(self, approval_key: str, ws_url: str | None = None) -> None:
        self._approval_key = approval_key
        self._ws_url = ws_url or KIS_WS_URL

    async def stream(
        self, session: aiohttp.ClientSession, codes: list[str]
    ) -> AsyncIterator[tuple[str, str, int]]:
        """codes 종목의 (symbol, hhmmss, price) 프린트를 흘려보낸다."""
        if not codes:
            raise ValueError("codes must not be empty")
        if len(codes) > KIS_WS_MAX_SUBSCRIPTIONS:
            raise ValueError(f"subscription limit exceeded: {len(codes)} > {KIS_WS_MAX_SUBSCRIPTIONS}")
        async with session.ws_connect(self._ws_url) as ws:  # pragma: no cover - live KIS websocket, probe-verified
            for code in codes:
                await ws.send_json(
                    {
                        "header": {
                            "approval_key": self._approval_key,
                            "custtype": "P",
                            "tr_type": "1",
                            "content-type": "utf-8",
                        },
                        "body": {"input": {"tr_id": "H0STCNT0", "tr_key": code}},
                    }
                )
            async for msg in ws:
                for symbol, hhmmss, price in parse_realtime_frame(msg.data):
                    yield symbol, hhmmss, price
