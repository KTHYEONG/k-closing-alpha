"""Every deciding or trading entry point refuses the public example store before doing work."""

from __future__ import annotations

import pytest

from src.strategy import store


@pytest.fixture
def example_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "strategy_values", lambda: store.load_strategy_values(store.EXAMPLE_STORE_ROOT))


def test_roster_capture_refuses_example_store(example_store: None) -> None:
    from src.daily import roster_capture

    with pytest.raises(SystemExit) as exc:
        roster_capture.main(["--date", "2026-09-09"])
    assert exc.value.code == 1


def test_retrain_refuses_example_store(example_store: None) -> None:
    from src.ml import retrain

    with pytest.raises(SystemExit) as exc:
        retrain.main([])
    assert exc.value.code == 1


def test_deploy_preflight_refuses_example_store(example_store: None) -> None:
    from src.tools import deploy_preflight

    with pytest.raises(SystemExit) as exc:
        deploy_preflight.main([])
    assert exc.value.code == 1
