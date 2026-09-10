from __future__ import annotations


def test_parse_realtime_frame_extracts_prints_and_skips_control() -> None:
    from src.api.kis.ws_client import parse_realtime_frame

    # Given: 실측 확인된 H0STCNT0 데이터 프레임 형태 (0|TR|건수|본문)
    body = "|".join(["005930", "151900", "70500"] + ["0"] * 38)
    raw = f"0|H0STCNT0|001|{body}"

    # When
    prints = parse_realtime_frame(raw)

    # Then
    assert prints == [("005930", "151900", 70500)]

    # And: 구독 성공 응답(JSON 제어 프레임)은 체결이 아니다
    ctrl = '{"header":{"tr_id":"H0STCNT0"},"body":{"rt_cd":"0","msg1":"SUBSCRIBE SUCCESS"}}'
    assert parse_realtime_frame(ctrl) == []

    # And: 암호화 프레임은 파싱하지 않는다
    assert parse_realtime_frame(f"1|H0STCNT0|001|{body}") == []


def test_issue_approval_key_returns_key_and_fails_closed() -> None:
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from src.api.kis.ws_client import issue_approval_key

    def _session_returning(payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.json = AsyncMock(return_value=payload)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post = MagicMock(return_value=ctx)
        return session

    # When: 정상 응답
    ok = _session_returning({"approval_key": "k" * 36})
    key = asyncio.run(issue_approval_key(ok, "app", "secret"))

    # Then
    assert key == "k" * 36

    # And: 키 없는 응답은 fail-closed
    bad = _session_returning({"error_description": "nope"})
    with pytest.raises(ValueError, match="approval_key"):
        asyncio.run(issue_approval_key(bad, "app", "secret"))


def test_ws_client_rejects_empty_and_oversized_subscription() -> None:
    import asyncio
    from unittest.mock import MagicMock

    import pytest

    from src.api.kis.ws_client import KIS_WS_MAX_SUBSCRIPTIONS, KisWebSocketClient

    client = KisWebSocketClient(approval_key="k" * 36)

    async def _drain(codes: list[str]) -> None:
        async for _ in client.stream(MagicMock(), codes):
            break

    # When / Then: 빈 목록
    with pytest.raises(ValueError, match="codes"):
        asyncio.run(_drain([]))

    # And: 구독 한도 초과
    too_many = [f"{i:06d}" for i in range(KIS_WS_MAX_SUBSCRIPTIONS + 1)]
    with pytest.raises(ValueError, match="subscription"):
        asyncio.run(_drain(too_many))


def test_ws_client_module_never_references_order_transmission_trs() -> None:
    from pathlib import Path

    # Given: 실주문 전송 TR 목록
    order_trs = ("TTTC0802U", "TTTC0801U", "TTTC0803U")
    paper_modules = (
        Path("src/api/kis/ws_client.py"),
        Path("src/execution/paper_broker.py"),
        Path("src/daily/paper_trade.py"),
    )

    # When / Then: 어떤 페이퍼 모듈도 주문 TR을 참조하지 않는다
    for path in paper_modules:
        text = path.read_text(encoding="utf-8")
        for tr in order_trs:
            assert tr not in text, f"{path} must not reference order TR {tr}"


def test_parse_realtime_frame_skips_malformed_records(caplog) -> None:
    import logging

    from src.api.kis.ws_client import parse_realtime_frame

    good = "|".join(["005930", "151900", "70500"] + ["0"] * 38)
    bad_price = "|".join(["000660", "151901", "N/A"] + ["0"] * 38)

    # Given: 정상 1건 + 비수치 가격 1건이 한 프레임에 담겨 온다
    with caplog.at_level(logging.WARNING):
        prints = parse_realtime_frame(f"0|H0STCNT0|002|{good}|{bad_price}")

    # Then: 정상 건만 살아남고, 건너뛴 사실은 로그로 남는다
    assert prints == [("005930", "151900", 70500)]
    assert caplog.records

    # And: 필드 수가 모자란 프레임은 전량 스킵되며 예외를 던지지 않는다
    assert parse_realtime_frame("0|H0STCNT0|001|005930|151900") == []
