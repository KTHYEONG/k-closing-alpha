"""마켓 레짐 백필 진입점의 생존 여부를 검증하는 회귀 가드."""

from __future__ import annotations


def test_market_regime_entrypoint_survives_while_legacy_wrapper_is_gone() -> None:
    from src.backfill import backfill_regime

    # 살아남아야 하는 진입점 (동일 모듈 내부 호출로 실사용 중)
    assert callable(backfill_regime.run_backfill_market_regime_factors)
    assert callable(backfill_regime.main)
    # 이번 단계에서 삭제된 하위 호환 래퍼
    assert not hasattr(backfill_regime, "run_backfill_market_factors")
