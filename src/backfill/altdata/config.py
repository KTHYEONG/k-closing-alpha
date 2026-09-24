"""Alt-data backfill 설정 및 패널 레지스트리."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.backfill.altdata.dart_keys import DartKeyPool

_ALTDATA_PANELS: dict[str, dict[str, Any]] = {
    "shorting": {
        "filename": "shorting.parquet",
        "key_cols": ("date", "symbol"),
        "availability_rule": "eod_release_next_decision",
        "level": "symbol",
    },
    "derivatives_basis": {
        "filename": "derivatives_basis.parquet",
        "key_cols": ("date",),
        "availability_rule": "eod_release_next_decision",
        "level": "market",
    },
    "disclosure": {
        "filename": "disclosure.parquet",
        "key_cols": ("date", "symbol"),
        "availability_rule": "eod_release_next_decision",
        "level": "symbol",
    },
    "credit_balance": {
        "filename": "credit_balance.parquet",
        "key_cols": ("date", "symbol"),
        "availability_rule": "eod_release_next_decision",
        "level": "symbol",
    },
    "program_trade_daily": {
        "filename": "program_trade_daily.parquet",
        "key_cols": ("date", "symbol"),
        "availability_rule": "eod_release_next_decision",
        "level": "symbol",
    },
}


@dataclass(frozen=True)
class AltDataFetchConfig:
    """Alt-data 수집 설정.

    Attributes:
        start: 수집 시작일.
        end: 수집 종료일.
        out_dir: 출력 디렉토리.
        sources: 수집할 패널 이름 튜플.
        markets: 대상 시장 튜플.
        universe_symbols: 필터링할 종목 코드 집합.
        pykrx_requests_per_sec: pykrx 호출 제한.
        dart_requests_per_sec: DART 호출 제한.
        retries: 재시도 횟수.
        retry_sleep_sec: 재시도 대기 시간.
        dart_api_key: DART 인증 키.
        dart_key_pool: Ordered OpenDART key pool. Single-key shorthand
            ``dart_api_key`` is used only when ``dart_key_pool is None``
            (CLI ``backfill_altdata`` keeps using it unchanged).
        page_count: 페이지 당 레코드 수.
        extra_client_kwargs: 병렬 호출용 추가 KIS 데이터 키.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    out_dir: Path
    sources: tuple[str, ...] = ("shorting", "derivatives_basis", "disclosure", "credit_balance", "program_trade_daily")
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ")
    universe_symbols: frozenset[str] | None = None
    pykrx_requests_per_sec: float = 6.0
    dart_requests_per_sec: float = 8.0
    krx_requests_per_sec: float = 4.0
    retries: int = 4
    retry_sleep_sec: float = 1.0
    dart_api_key: str = ""
    dart_key_pool: DartKeyPool | None = None
    krx_api_key: str = ""
    page_count: int = 100
    extra_client_kwargs: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        # Coerce start/end via pd.Timestamp
        s = pd.Timestamp(self.start)
        e = pd.Timestamp(self.end)
        object.__setattr__(self, "start", s)
        object.__setattr__(self, "end", e)
        if not (s < e):
            raise ValueError("start must be < end")
        # out_dir must be Path
        if not isinstance(self.out_dir, Path):
            object.__setattr__(self, "out_dir", Path(self.out_dir))
            # also validate after conversion - if still not Path raise
            if not isinstance(self.out_dir, Path):
                raise ValueError("out_dir must be a Path")
        # Validate sources
        if not self.sources or len(self.sources) == 0:
            raise ValueError("sources must be non-empty")
        for src in self.sources:
            if src not in _ALTDATA_PANELS:
                raise ValueError(f"source '{src}' is not in allowed panels")
        # Validate markets
        allowed_markets = {"KOSPI", "KOSDAQ"}
        if not self.markets or len(self.markets) == 0:
            raise ValueError("markets must be non-empty")
        for m in self.markets:
            if m not in allowed_markets:
                raise ValueError(f"market '{m}' is not allowed")
        # Validate rates
        if not (float(self.pykrx_requests_per_sec) > 0):
            raise ValueError("pykrx_requests_per_sec must be > 0")
        if not (float(self.dart_requests_per_sec) > 0):
            raise ValueError("dart_requests_per_sec must be > 0")
        if not (float(self.krx_requests_per_sec) > 0):
            raise ValueError("krx_requests_per_sec must be > 0")
        if not (int(self.retries) >= 1):
            raise ValueError("retries must be >= 1")
        if not (float(self.retry_sleep_sec) >= 0):
            raise ValueError("retry_sleep_sec must be >= 0")
        if not (1 <= int(self.page_count) <= 100):
            raise ValueError("page_count must be in [1, 100]")
        for item in self.extra_client_kwargs:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"extra_client_kwargs entry {item!r} must be a (app_key, app_secret, hts_id) tuple with non-empty app_key/app_secret")
            app_key, app_secret, _hts_id = item
            if not isinstance(app_key, str) or not app_key or not isinstance(app_secret, str) or not app_secret:
                raise ValueError(f"extra_client_kwargs entry {item!r} must be a (app_key, app_secret, hts_id) tuple with non-empty app_key/app_secret")
        # Validate universe_symbols
        if self.universe_symbols is not None:
            if not isinstance(self.universe_symbols, frozenset):
                # allow set input but coerce? spec says must be frozenset
                # we keep as-is for validation, but require frozenset
                raise ValueError("universe_symbols must be a frozenset")
            if len(self.universe_symbols) == 0:
                raise ValueError("universe_symbols must be non-empty")
            for sym in self.universe_symbols:
                if not isinstance(sym, str) or len(sym) != 6 or not sym.isdigit():
                    raise ValueError(f"universe_symbols entry '{sym}' must be 6-digit string")


def dart_pool_for(cfg: AltDataFetchConfig) -> DartKeyPool:
    """Return the configured pool or a single-key fallback pool.

    Args:
        cfg: Alt-data fetch configuration.

    Returns:
        Configured pool when set, otherwise a pool holding one credential
        labeled ``DEFAULT`` from ``cfg.dart_api_key``.
    """
    if cfg.dart_key_pool is not None:
        return cfg.dart_key_pool
    from src.backfill.altdata.dart_keys import DartCredential, DartKeyPool

    return DartKeyPool([DartCredential(label="DEFAULT", key=cfg.dart_api_key)])
