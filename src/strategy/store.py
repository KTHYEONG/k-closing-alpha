"""Chosen strategy values, read from the private strategy store.

The public tree carries the code and methodology only. The chosen values live in a store root
selected by ``KCA_STRATEGY_STORE``; without it the public example store is used and flagged, and
every deciding or trading path refuses it through ``require_production_store``.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

logger = logging.getLogger(__name__)

_STORE_ENV: Final = "KCA_STRATEGY_STORE"
_SCHEMA: Final = "kca.strategy_values.v1"
_PROJECT_DIRNAME: Final = "k-closing-alpha"
_REVISION_FILE: Final = "STORE_REVISION"
EXAMPLE_STORE_ROOT: Final[Path] = Path(__file__).resolve().parents[2] / "config" / "strategy_store.example"

_RANKER_PARAM_TYPES: Final[Mapping[str, type]] = MappingProxyType(
    {
        "n_estimators": int,
        "learning_rate": float,
        "num_leaves": int,
        "min_child_samples": int,
        "subsample": float,
        "subsample_freq": int,
        "colsample_bytree": float,
        "reg_lambda": float,
    }
)
_CHAMPION_PARAM_TYPES: Final[Mapping[str, type]] = _RANKER_PARAM_TYPES


class StrategyStoreError(ValueError):
    """Strategy store is missing, malformed, or is the public example store on a production path."""


@dataclass(frozen=True, slots=True)
class StrategyValues:
    universe_chg_min: float
    universe_chg_max: float
    universe_min_trade_value_100m: float
    universe_min_market_cap_100m: float
    ranker_model_params: Mapping[str, Any]
    ranker_seeds: tuple[int, ...]
    label_clip: float
    champion_model_params: Mapping[str, Any]
    production_calendar_flow_features: tuple[str, ...]
    example: bool
    root: Path
    revision: str | None


def strategy_store_root() -> Path:
    raw = os.environ.get(_STORE_ENV, "").strip()
    if not raw:
        return EXAMPLE_STORE_ROOT
    root = Path(raw)
    if not root.is_absolute():
        raise StrategyStoreError(f"{_STORE_ENV} must be an absolute path")
    return root


def store_revision(root: Path) -> str | None:
    try:
        text = (root / _REVISION_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _required(table: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise StrategyStoreError(f"strategy store missing key: {where}.{key}")
    return table[key]


def _table(table: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = _required(table, key, where)
    if not isinstance(value, dict):
        raise StrategyStoreError(f"strategy store key must be a table: {where}.{key}")
    return value


def _text(table: Mapping[str, Any], key: str, where: str) -> str:
    value = _required(table, key, where)
    if not isinstance(value, str) or not value:
        raise StrategyStoreError(f"strategy store key must be a non-empty string: {where}.{key}")
    return value


def _boolean(table: Mapping[str, Any], key: str, where: str) -> bool:
    value = _required(table, key, where)
    if not isinstance(value, bool):
        raise StrategyStoreError(f"strategy store key must be a boolean: {where}.{key}")
    return value


def _finite(table: Mapping[str, Any], key: str, where: str) -> float:
    value = _required(table, key, where)
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        raise StrategyStoreError(f"strategy store key must be a finite float: {where}.{key}")
    return value


def _seeds(table: Mapping[str, Any], key: str, where: str) -> tuple[int, ...]:
    value = _required(table, key, where)
    if not isinstance(value, list) or not value:
        raise StrategyStoreError(f"strategy store key must be a non-empty array: {where}.{key}")
    if any(isinstance(s, bool) or not isinstance(s, int) for s in value):
        raise StrategyStoreError(f"strategy store key must contain only integers: {where}.{key}")
    if len(set(value)) != len(value):
        raise StrategyStoreError(f"strategy store key contains duplicate seeds: {where}.{key}")
    return tuple(value)


def _model_params(table: Mapping[str, Any], key: str, where: str, types: Mapping[str, type]) -> Mapping[str, Any]:
    raw = _table(table, key, where)
    where_full = f"{where}.{key}"
    if set(raw) != set(types):
        missing = sorted(set(types) - set(raw))
        extra = sorted(set(raw) - set(types))
        raise StrategyStoreError(
            f"strategy store model_params key set mismatch: {where_full} missing={missing} unexpected={extra}"
        )
    params: dict[str, Any] = {}
    for name, kind in types.items():
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, kind):
            raise StrategyStoreError(f"strategy store key has the wrong type: {where_full}.{name}")
        params[name] = value
    return MappingProxyType(params)


def _features(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = _required(table, key, where)
    if not isinstance(value, list) or not value or not all(isinstance(f, str) and f for f in value):
        raise StrategyStoreError(f"strategy store key must be a non-empty array of names: {where}.{key}")
    if len(set(value)) != len(value):
        raise StrategyStoreError(f"strategy store key contains duplicate names: {where}.{key}")
    return tuple(value)


def load_strategy_values(root: Path | None = None) -> StrategyValues:
    store_root = strategy_store_root() if root is None else root
    path = store_root / _PROJECT_DIRNAME / "params.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise StrategyStoreError(f"strategy store params unreadable: {path}") from exc
    meta = _table(raw, "meta", "root")
    if _text(meta, "schema", "meta") != _SCHEMA:
        raise StrategyStoreError("strategy store meta.schema is not supported")
    for key in ("version", "created_at", "source", "validated_by"):
        _text(meta, key, "meta")
    example = _boolean(meta, "example", "meta")
    universe = _table(raw, "universe", "root")
    ranker = _table(raw, "ranker", "root")
    champion = _table(raw, "champion", "root")
    features = _table(raw, "features", "root")
    return StrategyValues(
        universe_chg_min=_finite(universe, "chg_min", "universe"),
        universe_chg_max=_finite(universe, "chg_max", "universe"),
        universe_min_trade_value_100m=_finite(universe, "min_trade_value_100m", "universe"),
        universe_min_market_cap_100m=_finite(universe, "min_market_cap_100m", "universe"),
        ranker_model_params=_model_params(ranker, "model_params", "ranker", _RANKER_PARAM_TYPES),
        ranker_seeds=_seeds(ranker, "seeds", "ranker"),
        label_clip=_finite(ranker, "label_clip", "ranker"),
        champion_model_params=_model_params(champion, "model_params", "champion", _CHAMPION_PARAM_TYPES),
        production_calendar_flow_features=_features(features, "production_calendar_flow", "features"),
        example=example,
        root=store_root,
        revision=store_revision(store_root),
    )


@functools.cache
def strategy_values() -> StrategyValues:
    return load_strategy_values(strategy_store_root())


def require_production_store() -> StrategyValues:
    values = strategy_values()
    if values.example:
        raise StrategyStoreError(
            f"private strategy store required; public example store active (revision={values.revision})"
        )
    logger.info("[SYS] strategy_store revision=%s", values.revision)
    return values
