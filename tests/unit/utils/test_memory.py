"""release_free_memory is a value-preserving best-effort call."""

from __future__ import annotations

import ctypes.util

import numpy as np

from src.utils import memory


def test_release_keeps_live_values() -> None:
    live = np.arange(1_000_000, dtype=np.float64)
    expected = float(live.sum())

    memory.release_free_memory()

    assert float(live.sum()) == expected


def test_release_without_libc_is_a_noop(monkeypatch) -> None:
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: None)

    memory.release_free_memory()


def test_release_with_unloadable_libc_is_a_noop(monkeypatch) -> None:
    def broken(_name: str) -> None:
        raise OSError("not loadable")

    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libc-missing.so")
    monkeypatch.setattr(memory.ctypes, "CDLL", broken)

    memory.release_free_memory()
