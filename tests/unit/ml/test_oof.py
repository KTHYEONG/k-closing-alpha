"""oof 라벨 임계값이 strategy.contract 단일 정의로 통합되었는지 검증합니다."""

from __future__ import annotations


def test_oof_label_thresholds_come_from_contract() -> None:
    from src.ml import oof
    from src.strategy import contract

    assert contract.LABEL_GOOD_THRESHOLD == 0.01
    assert contract.LABEL_BAD_THRESHOLD == -0.02
    assert not hasattr(oof, "_GOOD_THRESHOLD")
    assert not hasattr(oof, "_BAD_THRESHOLD")
    assert oof.LABEL_GOOD_THRESHOLD == contract.LABEL_GOOD_THRESHOLD
    assert oof.LABEL_BAD_THRESHOLD == contract.LABEL_BAD_THRESHOLD
