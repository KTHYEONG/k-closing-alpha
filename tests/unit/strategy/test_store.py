from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from src.strategy import store
from src.strategy.store import (
    EXAMPLE_STORE_ROOT,
    StrategyStoreError,
    load_strategy_values,
    require_production_store,
    strategy_store_root,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "strategy_store"
PARAMS = "k-closing-alpha/params.toml"


def _copy_store(tmp_path: Path, source: Path = FIXTURE_ROOT) -> Path:
    root = tmp_path / "store"
    shutil.copytree(source, root)
    return root


def _rewrite(root: Path, old: str, new: str) -> None:
    path = root / PARAMS
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_loads_fixture_values_with_revision(tmp_path: Path) -> None:
    root = _copy_store(tmp_path)
    (root / "STORE_REVISION").write_text("rev-7\n", encoding="utf-8")

    values = load_strategy_values(root)

    assert values.example is False
    assert values.revision == "rev-7"
    assert values.universe_chg_min == 0.013
    assert values.ranker_seeds == (21, 22)
    assert values.label_clip == 0.06
    assert values.ranker_model_params["n_estimators"] == 123
    assert values.champion_model_params["num_leaves"] == 9
    assert values.production_calendar_flow_features[0] == "placeholder_flow_a"
    with pytest.raises(TypeError):
        values.ranker_model_params["n_estimators"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "key_line",
    [
        "version = ",
        "example = ",
        "chg_min = ",
        "chg_max = ",
        "min_trade_value_100m = ",
        "min_market_cap_100m = ",
        "seeds = ",
        "label_clip = ",
        "production_calendar_flow = ",
        "n_estimators = ",
        "reg_lambda = ",
    ],
)
def test_missing_key_fails_closed_and_names_key_not_value(tmp_path: Path, key_line: str) -> None:
    root = _copy_store(tmp_path)
    path = root / PARAMS
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if not line.startswith(key_line)
    ]
    assert len(lines) < len(path.read_text(encoding="utf-8").splitlines(keepends=True))
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(StrategyStoreError) as exc:
        load_strategy_values(root)
    assert key_line.split(" =")[0] in str(exc.value)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("chg_min = 0.013", "chg_min = true"),
        ("min_trade_value_100m = 150.0", "min_trade_value_100m = 150"),
        ("seeds = [21, 22]", "seeds = []"),
        ("seeds = [21, 22]", "seeds = [21, 21]"),
        ("seeds = [21, 22]", "seeds = [21, true]"),
        ("n_estimators = 123", "n_estimators = 123.0"),
        ("subsample = 0.7", "subsample = 1"),
        ("num_leaves = 9", "num_leaves = 9\nextra_param = 1"),
        ("example = false", 'example = "no"'),
        (
            'production_calendar_flow = ["placeholder_flow_a", "placeholder_flow_b", "placeholder_flow_c", "placeholder_flow_d"]',
            'production_calendar_flow = ["placeholder_flow_a", "placeholder_flow_a"]',
        ),
        ('schema = "kca.strategy_values.v1"', 'schema = "kca.strategy_values.v0"'),
    ],
)
def test_type_violations_fail_closed(tmp_path: Path, old: str, new: str) -> None:
    root = _copy_store(tmp_path)
    _rewrite(root, old, new)

    with pytest.raises(StrategyStoreError) as exc:
        load_strategy_values(root)
    assert "placeholder" not in str(exc.value)


def test_model_params_key_set_must_match_exactly(tmp_path: Path) -> None:
    root = _copy_store(tmp_path)
    _rewrite(root, "reg_lambda = 1.5", "reg_lambda = 1.5\nunknown_knob = 2")

    with pytest.raises(StrategyStoreError, match=r"model_params key set mismatch: champion\.model_params"):
        load_strategy_values(root)


def test_relative_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KCA_STRATEGY_STORE", "relative/store")

    with pytest.raises(StrategyStoreError, match="absolute"):
        strategy_store_root()


def test_unset_env_uses_example_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KCA_STRATEGY_STORE", raising=False)

    assert strategy_store_root() == EXAMPLE_STORE_ROOT
    assert load_strategy_values(EXAMPLE_STORE_ROOT).example is True


def test_env_store_never_falls_back_to_example(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty-store"
    empty.mkdir()
    monkeypatch.setenv("KCA_STRATEGY_STORE", str(empty))

    with pytest.raises(StrategyStoreError, match="unreadable"):
        load_strategy_values(strategy_store_root())


def test_require_production_store_rejects_example(monkeypatch: pytest.MonkeyPatch) -> None:
    example_values = load_strategy_values(EXAMPLE_STORE_ROOT)
    monkeypatch.setattr(store, "strategy_values", lambda: example_values)

    with pytest.raises(StrategyStoreError, match="example store active") as exc:
        require_production_store()
    assert re.search(r"revision=None", str(exc.value))


def test_require_production_store_accepts_private_store(monkeypatch: pytest.MonkeyPatch) -> None:
    private_values = load_strategy_values(FIXTURE_ROOT)
    monkeypatch.setattr(store, "strategy_values", lambda: private_values)

    assert require_production_store() is private_values


def test_public_tree_has_no_research_results() -> None:
    repo = Path(__file__).resolve().parents[3]

    assert not (repo / "docs" / "research" / "closing_strategy_matrix.md").exists()
    assert not (repo / "artifacts" / "research" / "costaware_topk_report.parquet").exists()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('version = "fixture"', "version = 1", "non-empty string: meta.version"),
        ("production_calendar_flow = [", "production_calendar_flow = [1, ", "non-empty array of names"),
    ],
)
def test_structural_violations_fail_closed(tmp_path: Path, old: str, new: str, message: str) -> None:
    root = _copy_store(tmp_path)
    path = root / "k-closing-alpha" / "params.toml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(StrategyStoreError, match=message):
        load_strategy_values(root)


def test_non_table_section_fails_closed(tmp_path: Path) -> None:
    root = _copy_store(tmp_path)
    path = root / "k-closing-alpha" / "params.toml"
    text = path.read_text(encoding="utf-8").replace("[universe]\n", "[universe_unused]\n", 1)
    path.write_text("universe = 3\n" + text, encoding="utf-8")
    with pytest.raises(StrategyStoreError, match=r"must be a table: root\.universe"):
        load_strategy_values(root)
