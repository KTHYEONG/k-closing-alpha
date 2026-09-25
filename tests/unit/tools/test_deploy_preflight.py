from __future__ import annotations

import dataclasses


def _compatible_bundle() -> dict:
    from src.ml.costaware_topk import MIN_TOP_K
    from src.ml.topk_ranker_research import RANKER_FEATURE_COLS
    from src.strategy.contract import COST_AWARE_UNIVERSE

    return {
        "select_universe": dataclasses.asdict(COST_AWARE_UNIVERSE),
        "feature_cols": list(RANKER_FEATURE_COLS[:5]),
        "top_k": MIN_TOP_K,
    }


def test_check_bundle_serving_compat_accepts_compatible_bundle() -> None:
    from src.tools.deploy_preflight import check_bundle_serving_compat

    assert check_bundle_serving_compat(_compatible_bundle()) == []


def test_check_bundle_serving_compat_reports_screen_drift() -> None:
    from src.strategy.contract import COST_AWARE_UNIVERSE
    from src.tools.deploy_preflight import check_bundle_serving_compat

    bundle = _compatible_bundle()
    drifted = dict(bundle["select_universe"])
    drifted["exclude_non_screenable_class"] = not COST_AWARE_UNIVERSE.exclude_non_screenable_class
    bundle["select_universe"] = drifted

    issues = check_bundle_serving_compat(bundle)

    assert len(issues) == 1
    assert "exclude_non_screenable_class" in issues[0]


def test_check_bundle_serving_compat_reports_unknown_feature() -> None:
    from src.tools.deploy_preflight import check_bundle_serving_compat

    bundle = _compatible_bundle()
    bundle["feature_cols"] = [*bundle["feature_cols"], "f_future"]

    issues = check_bundle_serving_compat(bundle)

    assert len(issues) == 1
    assert "f_future" in issues[0]


def test_check_bundle_serving_compat_reports_empty_feature_cols() -> None:
    from src.tools.deploy_preflight import check_bundle_serving_compat

    bundle = _compatible_bundle()
    bundle["feature_cols"] = []

    issues = check_bundle_serving_compat(bundle)

    assert len(issues) == 1
    assert "feature_cols" in issues[0]


def test_check_bundle_serving_compat_reports_all_issues_together() -> None:
    from src.strategy.contract import COST_AWARE_UNIVERSE
    from src.tools.deploy_preflight import check_bundle_serving_compat

    bundle = _compatible_bundle()
    drifted = dict(bundle["select_universe"])
    drifted["exclude_non_screenable_class"] = not COST_AWARE_UNIVERSE.exclude_non_screenable_class
    bundle["select_universe"] = drifted
    bundle["feature_cols"] = [*bundle["feature_cols"], "f_future"]
    bundle["top_k"] = 5

    issues = check_bundle_serving_compat(bundle)

    assert len(issues) == 3


def test_main_fails_when_bundle_missing(tmp_path) -> None:
    import pytest

    from src.tools.deploy_preflight import main

    with pytest.raises(SystemExit) as exc:
        main(["--bundle-dir", str(tmp_path / "empty")])

    assert exc.value.code == 1


def test_main_fails_on_incompatible_bundle(tmp_path, caplog) -> None:
    import logging

    import pytest
    from joblib import dump

    from src.tools.deploy_preflight import main

    bundle = _compatible_bundle()
    bundle["top_k"] = 5
    bundle.update({"rank_model": None, "quantile_models": None, "calibrators": None})
    dump(bundle, tmp_path / "sizing_pipeline_bundle.joblib")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        main(["--bundle-dir", str(tmp_path)])

    assert exc.value.code == 1
    assert any("stage=deploy_preflight status=FAIL" in rec.message for rec in caplog.records)


def test_main_writes_nothing_and_reports_ok(tmp_path, caplog) -> None:
    import logging
    import os

    from joblib import dump

    from src.tools.deploy_preflight import main

    bundle = _compatible_bundle()
    bundle.update({"rank_model": None, "quantile_models": None, "calibrators": None})
    target = tmp_path / "sizing_pipeline_bundle.joblib"
    dump(bundle, target)
    before = {path: os.stat(path) for path in sorted(tmp_path.iterdir())}

    with caplog.at_level(logging.INFO):
        assert main(["--bundle-dir", str(tmp_path)]) is None

    after = {path: os.stat(path) for path in sorted(tmp_path.iterdir())}
    assert [path.name for path in after] == [path.name for path in before]
    assert all(after[path].st_mtime_ns == before[path].st_mtime_ns for path in after)
    assert any("stage=deploy_preflight status=OK" in rec.message for rec in caplog.records)
