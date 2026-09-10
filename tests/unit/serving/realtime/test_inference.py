"""realtime inference의 실현변동성 기본값이 strategy.contract 단일 정의로 통합되었는지 검증합니다."""

from __future__ import annotations


def test_inference_realized_vol_default_comes_from_contract() -> None:
    from src.serving.realtime import inference
    from src.strategy import contract

    assert contract.DEFAULT_REALIZED_VOL == 0.02
    assert not hasattr(inference, "_DEFAULT_REALIZED_VOL")
