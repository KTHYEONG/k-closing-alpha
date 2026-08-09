"""Read-only bundle validation contract for the live serving path."""

from __future__ import annotations

import pytest
from joblib import dump

from src.serving.realtime.artifacts import load_model_bundle


def test_load_model_bundle_returns_feature_cols(tmp_path) -> None:
    bundle = {"feature_cols": ["f1", "f2"], "rank_model": "r", "quantile_models": {}, "calibrators": {}}
    dump(bundle, tmp_path / "sizing_pipeline_bundle.joblib")
    loaded = load_model_bundle(str(tmp_path))
    assert loaded["feature_cols"] == ["f1", "f2"]


def test_load_model_bundle_raises_when_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_bundle(str(tmp_path / "no_such_dir"))


def test_load_model_bundle_raises_on_empty_feature_cols(tmp_path) -> None:
    dump({"feature_cols": [], "rank_model": "r", "quantile_models": {}, "calibrators": {}}, tmp_path / "sizing_pipeline_bundle.joblib")
    with pytest.raises(ValueError, match="feature_cols"):
        load_model_bundle(str(tmp_path))


def test_load_model_bundle_raises_on_missing_model_keys(tmp_path) -> None:
    dump({"feature_cols": ["f1"]}, tmp_path / "sizing_pipeline_bundle.joblib")
    with pytest.raises(ValueError, match="missing required model keys"):
        load_model_bundle(str(tmp_path))
