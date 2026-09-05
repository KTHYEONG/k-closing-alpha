"""Alt-data backfill 패키지."""

from __future__ import annotations

from typing import Any

__all__ = ["AltDataFetchConfig", "run_altdata_backfill"]


def __getattr__(name: str) -> Any:
    # 지연 re-export: 패키지 import 시점에 runner의 무거운 의존성
    # (pandas/numpy C 확장)을 끌어오지 않는다. coverage의 소스-해석
    # 단계(find_spec + sys.modules 복원)가 이미 로드된 네이티브 확장을
    # 등록 해제 상태로 남겨 이후 pandas import를 깨뜨리는 것을 방지한다.
    if name == "AltDataFetchConfig":
        from src.backfill.altdata.config import AltDataFetchConfig

        return AltDataFetchConfig
    if name == "run_altdata_backfill":
        from src.backfill.altdata.runner import run_altdata_backfill

        return run_altdata_backfill
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
