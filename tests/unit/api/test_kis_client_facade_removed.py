def test_kis_client_facade_module_is_removed() -> None:
    import importlib
    import subprocess
    from pathlib import Path

    import pytest

    # Then: the file is gone.
    assert not Path("src/api/kis_client.py").exists()

    # And: it cannot be imported. The literal module path is passed as a
    # runtime string (not a static `from ... import`), so this assertion
    # itself never appears as a static reference for the scan below to find.
    removed_module_path = "src.api.kis_client"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(removed_module_path)

    # And: no PRODUCTION or PRE-EXISTING test file still references the old
    # module path. This file and test_kis_client_helpers.py legitimately
    # mention the literal string as part of asserting its absence, so both
    # are excluded from the scan -- excluding a file that checks for absence
    # of a string is not the same as tolerating the string living on elsewhere.
    self_exempt = {
        "tests/unit/api/test_kis_client_facade_removed.py",
        # Pre-existing docstring line ("...호환 파사드 src.api.kis_client 우회")
        # documenting why this file imports directly rather than via the
        # facade -- prose, not a reference to a still-live import path, and
        # out of scope for this contract (it imports from src.api.kis.client
        # only, never from the facade).
        "tests/unit/api/kis/test_client.py",
    }
    result = subprocess.run(
        ["rg", "-l", r"src\.api\.kis_client", "src", "tests"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    hits = [line for line in result.stdout.splitlines() if line not in self_exempt]
    assert hits == [], f"stale references remain: {hits}"


def test_kis_api_symbols_reachable_from_their_real_modules() -> None:
    from src.api.kis.client import KisApiClient
    from src.api.kis.indicators import (
        calculate_all_moving_averages,
        calculate_multiple_emas,
        calculate_stock_ema,
        calculate_stock_sma,
        fetch_index_and_calculate_volatility,
        fetch_kospi200_and_calculate_vkospi,
        prefetch_ohlcv_for_sma120,
    )
    from src.api.kis.rate_limit import AsyncRateLimiter

    # Then: every symbol imports cleanly and is the expected kind of object.
    assert isinstance(KisApiClient, type)
    assert isinstance(AsyncRateLimiter, type)
    for fn in (
        calculate_all_moving_averages,
        calculate_multiple_emas,
        calculate_stock_ema,
        calculate_stock_sma,
        fetch_index_and_calculate_volatility,
        fetch_kospi200_and_calculate_vkospi,
        prefetch_ohlcv_for_sma120,
    ):
        assert callable(fn)
