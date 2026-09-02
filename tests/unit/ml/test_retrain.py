"""retrain CLI 의 --feature-set 인자 배선 계약."""
from __future__ import annotations

import pytest

from src.ml.retrain import main


def test_retrain_rejects_unknown_feature_set(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--feature-set", "bogus"])
    assert exc.value.code != 0
    assert "feature-set" in capsys.readouterr().err
