"""Invariant guards for the top-k train/serve contract module."""

from __future__ import annotations

import ast
import dataclasses
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_removed_shim_names_are_gone() -> None:
    import src.ml.topk_contract as contract
    import src.ml.topk_history_features as history_features
    import src.ml.topk_ranker_research as research
    from src.ml.research import v3_engine

    for name in (
        "TOPK_RANKER_BUNDLE_DIR",
        "assert_bundle_screen_parity",
        "save_production_bundle",
        "score_topk_candidates",
        "select_topk_equal_weight",
    ):
        assert not hasattr(research, name), name
    for name in ("FEATURE_COLS", "compute_derived_features"):
        assert not hasattr(v3_engine, name), name
    for name in (
        "TOPK_COST_FEATURE_COLS",
        "TOPK_FLOW_FEATURE_COLS",
        "TOPK_HISTORY_FEATURE_COLS",
        "TOPK_FEATURE_COLS_V2",
    ):
        assert getattr(history_features, name) is getattr(contract, name)


def test_research_module_keeps_the_contract_bindings_it_uses() -> None:
    import src.ml.topk_contract as contract
    import src.ml.topk_ranker_research as research

    for name in (
        "RANKER_FEATURE_COLS",
        "compute_derived_features",
        "select_topk_by_score",
        "build_topk_feature_manifest",
    ):
        assert getattr(research, name) is getattr(contract, name), name


def test_contract_module_is_a_leaf() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    code = "import sys, src.ml.topk_contract; print(sorted(sys.modules))"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo_root)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "src.ml.topk_ranker_research" not in proc.stdout
    assert "src.ml.research" not in proc.stdout
    assert "scipy" not in proc.stdout
    assert "lightgbm" not in proc.stdout
    assert "sklearn" not in proc.stdout


def test_serving_callers_import_no_research_module() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for rel in (
        "src/daily/predict.py",
        "src/serving/realtime/features.py",
        "src/tools/deploy_preflight.py",
    ):
        tree = ast.parse((repo_root / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "src.ml.topk_ranker_research"
                    assert not alias.name.startswith("src.ml.research")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module != "src.ml.topk_ranker_research"
                assert not module.startswith("src.ml.research")


def test_ranker_feature_order_preserved() -> None:
    import src.ml.topk_contract as contract

    expected = [
        *contract.FEATURE_COLS,
        *contract.TOPK_FLOW_FEATURE_COLS,
        *contract.TOPK_COST_FEATURE_COLS,
        *contract.TOPK_HISTORY_FEATURE_COLS,
    ]
    assert list(contract.RANKER_FEATURE_COLS) == expected
    assert len(contract.RANKER_FEATURE_COLS) == 27
    assert len(set(contract.RANKER_FEATURE_COLS)) == 27


class _ReturnStub:
    def predict(self, features: pd.DataFrame):
        return features["f1"].to_numpy(dtype=float) + features["f2"].to_numpy(dtype=float)


class _QuantileStub:
    def __init__(self, offset: float) -> None:
        self._offset = offset

    def predict(self, features: pd.DataFrame):
        base = features["f1"].to_numpy(dtype=float) + features["f2"].to_numpy(dtype=float)
        return base + self._offset


def test_selection_behavior_unchanged_through_new_path() -> None:
    import src.ml.topk_contract as contract
    from src.strategy.contract import MIN_TOP_K, PRODUCTION_STRATEGY

    rows = [
        {
            "date": pd.Timestamp(day),
            "symbol": f"{idx:06d}",
            "f1": float(idx),
            "f2": float(idx % 2),
            "admitted": True,
        }
        for day in ("2026-09-01", "2026-09-02")
        for idx in range(5)
    ]
    frame = pd.DataFrame(rows)
    bundle = {
        "return_model": _ReturnStub(),
        "quantile_models": {
            "pred_q10": _QuantileStub(-1.0),
            "pred_q50": _QuantileStub(0.0),
            "pred_q90": _QuantileStub(1.0),
        },
        "calibrators": {"p_good": 0.7, "p_bad": 0.3},
        "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe),
        "feature_cols": ["f1", "f2"],
    }
    new_picks = contract.select_topk_equal_weight(frame, bundle, top_k=MIN_TOP_K)
    assert len(new_picks) == 2 * MIN_TOP_K
    for _date, group in new_picks.groupby("date", sort=False):
        assert len(group) == MIN_TOP_K
        assert group["pred"].is_monotonic_decreasing
        assert (group["allocation"] == 1.0 / MIN_TOP_K).all()


EXPECTED_SPECS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "1": (
        ("chg_ratio", "decimal_return", "decision_snapshot"),
        ("log_tv", "log1p_100m_krw", "decision_snapshot"),
        ("log_mc", "log1p_100m_krw", "decision_snapshot"),
        ("body_ratio", "decimal_ratio", "decision_snapshot"),
        ("upper_shadow_ratio", "decimal_ratio", "decision_snapshot"),
        ("intraday_range", "decimal_ratio", "decision_snapshot"),
        ("kospi_pct", "decimal_return", "decision_snapshot"),
        ("kosdaq_pct", "decimal_return", "decision_snapshot"),
        ("v_kospi", "index_level", "decision_snapshot"),
        ("tv_rank", "pct_rank", "decision_snapshot"),
        ("chg_rank", "pct_rank", "decision_snapshot"),
        ("inst_density", "decimal_ratio", "prior_confirmed"),
        ("inst_rank", "pct_rank", "prior_confirmed"),
        ("f_tick_cost", "basis_points", "decision_snapshot"),
        ("f_log_close", "log_krw", "decision_snapshot"),
        ("f_ret5", "log_return_sum", "prior_confirmed"),
        ("f_ret20", "log_return_sum", "prior_confirmed"),
        ("f_ret60", "log_return_sum", "prior_confirmed"),
        ("f_dist_high60", "decimal_return", "snapshot_and_prior"),
        ("f_upcount20", "count", "prior_confirmed"),
        ("f_on_mean20", "decimal_return", "prior_confirmed"),
        ("f_on_mean60", "decimal_return", "prior_confirmed"),
        ("f_upnext_on60", "decimal_return", "snapshot_and_prior"),
        ("f_gap", "decimal_return", "decision_snapshot"),
        ("f_id_mean20", "decimal_return", "prior_confirmed"),
        ("f_inst_cum5", "decimal_ratio", "prior_confirmed"),
        ("f_foreign_cum5", "decimal_ratio", "prior_confirmed"),
    ),
}

GOLDEN: dict[str, dict[str, list[float]]] = {
    "1": {
        "chg_ratio": [-0.027828593033113247, -0.04223204034995531, -0.01506734376727381, -0.04815720438371385, 0.059102918661399384, 0.09806195835227194],
        "log_tv": [6.6319702987612885, 6.3730768785757785, 6.872091594297928, 5.662926057048779, 7.256547480097302, 8.72812180749052],
        "log_mc": [9.810863540328976, 9.299956086531292, 9.232969014653262, 9.402639382286129, 9.719819231534958, 10.945419905761522],
        "body_ratio": [-0.7046560164398341, -0.7629441500131021, -0.5921050144991354, -0.7860611480582195, 0.743996404628032, 0.7678822855200275],
        "upper_shadow_ratio": [0.02758210402311381, 0.13711223429208264, 0.35109472932083885, 0.07861713405305797, 0.2407113922960421, 0.1273136619407026],
        "intraday_range": [0.06816100253190692, 0.0493245411172964, 0.04773547068524214, 0.07151984416420654, 0.0551660506881034, 0.08783417679075468],
        "kospi_pct": [0.002633778825832136, 0.0026181146182244286, -0.008031037522962201, -0.008708497059945919, -0.005057629920102463, -0.00396837245904304],
        "kosdaq_pct": [0.008252671870025322, -0.002322162089517135, 0.00465329740353834, 0.007266496191783716, -0.00010281761427138922, -0.016031341898309036],
        "v_kospi": [18.0, 18.0, 18.0, 18.0, 18.0, 18.0],
        "tv_rank": [0.5, 0.3333333333333333, 0.6666666666666666, 0.16666666666666666, 0.8333333333333334, 1.0],
        "chg_rank": [0.5, 0.3333333333333333, 0.6666666666666666, 0.16666666666666666, 0.8333333333333334, 1.0],
        "inst_density": [0.0009279703771588841, -0.0004814039872881781, 0.0005823222046454264, -0.00014135830211152662, -0.00029657145761971245, 0.00011303963207401361],
        "inst_rank": [0.8333333333333334, 0.3333333333333333, 1.0, 0.5, 0.16666666666666666, 0.6666666666666666],
        "f_tick_cost": [5.485546798806645, 9.143660621000903, 9.777210492462082, 8.251282081583092, 6.008447021674399, 17.63892823315951],
        "f_log_close": [12.113393779359539, 11.6024497470992, 11.535456340321764, 11.705141965863355, 12.022344241863786, 13.247987359982899],
        "f_ret5": [0.03619609756469825, 0.1697224672502095, 0.10913074364254874, 0.20363297064375552, 0.025642293547346123, 0.21187326238072096],
        "f_ret20": [0.2313460824038294, 0.4277134708105594, 0.18699005633536558, 0.26142881656795547, -0.12253478410738125, 0.6734895886111779],
        "f_ret60": [0.6654464667918756, 0.7822752268916561, 0.15522656833072387, 0.594830548419665, 0.26918987682242035, 1.2489003690420872],
        "f_dist_high60": [-0.07948951676936408, -0.08570830550635863, -0.028043123233459425, -0.04815720438371386, -0.16059166265658625, 0.0],
        "f_upcount20": [10.0, 11.0, 8.0, 10.0, 9.0, 14.0],
        "f_on_mean20": [0.00836818820522996, 0.0005417188612571233, -0.0005017154051530859, -0.004632917547947446, -0.001963225437799815, 6.495102662332775e-05],
        "f_on_mean60": [0.004452412940180628, -0.0017895199832610841, -0.0025779498857869234, -0.0026121773826959477, -0.00412570817602298, -0.0023405789587967853],
        "f_upnext_on60": [0.0022431328240652723, 0.0027880603553082544, -0.0013990837106594592, -0.005956608112640965, -0.005808698125608025, -0.0006949130205982999],
        "f_gap": [0.018864858479979985, -0.006189440908836219, 0.012771198189547084, 0.005354417961279889, 0.01563379390716446, 0.024001732844036727],
        "f_id_mean20": [0.004768116019135971, 0.023337348342733887, 0.011439772320488284, 0.019916886820084644, -0.0026153729830674264, 0.03574374228715249],
        "f_inst_cum5": [0.00014493833910244372, -0.00038544984837704136, 0.0006867457437414573, 0.00016444102929174474, -7.225041568386142e-05, 2.887884391958843e-05],
        "f_foreign_cum5": [0.00026858449415710937, 7.790479878673832e-05, -0.00013159495760108728, -0.00020188506673776918, 7.901739899203083e-05, -6.0325194280503864e-05],
    },
}


def _deterministic_panel() -> pd.DataFrame:
    from src.data.panel_integrity import prepare_price_panel

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=90)
    rows = []
    for i in range(6):
        market = "KOSPI" if i % 2 == 0 else "KOSDAQ"
        prev = 50000.0 + i * 10000.0
        for d in dates:
            chg = float(rng.uniform(-0.08, 0.10))
            close = prev * (1.0 + chg)
            open_ = prev * (1.0 + float(rng.uniform(-0.03, 0.03)))
            high = max(open_, close) * (1.0 + float(rng.uniform(0.0, 0.02)))
            low = min(open_, close) * (1.0 - float(rng.uniform(0.0, 0.02)))
            volume = float(rng.integers(100000, 2000000))
            trade_value_100m = close * volume / 1e8 * float(rng.uniform(0.9, 1.1))
            market_cap_100m = close * 1e7 / 1e8
            rows.append({
                "date": d,
                "symbol": f"{i:06d}",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "prev_close": prev,
                "volume": volume,
                "market_cap_100m": market_cap_100m,
                "trade_value_100m": trade_value_100m,
                "market": market,
                "daily_change_pct": chg,
                "inst_netbuy": float(rng.integers(-10**8, 10**8)),
                "foreign_netbuy": float(rng.integers(-10**8, 10**8)),
                "program_netbuy": 0.0,
                "kospi_pct": float(rng.normal(0, 0.005)),
                "kosdaq_pct": float(rng.normal(0, 0.006)),
                "v_kospi": 18.0,
                "v_kosdaq": 22.0,
            })
            prev = close
    ph, _prov = prepare_price_panel(pd.DataFrame(rows))
    return ph


def test_declared_table_pinned_per_version() -> None:
    import src.ml.topk_contract as contract

    projected = tuple((s.name, s.unit.value, s.timing.value) for s in contract.TOPK_FEATURE_SPECS)
    assert projected == EXPECTED_SPECS[contract.TOPK_FEATURE_CONTRACT_VERSION]


def test_declared_table_matches_ranker_feature_order() -> None:
    import src.ml.topk_contract as contract

    names = [s.name for s in contract.TOPK_FEATURE_SPECS]
    assert names == list(contract.RANKER_FEATURE_COLS)
    assert len(set(names)) == len(names)


def test_feature_definition_golden_pin() -> None:
    import src.ml.topk_contract as contract
    from src.ml.topk_history_features import attach_lagged_flow_features, attach_topk_features

    ph = _deterministic_panel()
    t = pd.Timestamp(pd.to_datetime(ph["date"]).max())
    day = ph[pd.to_datetime(ph["date"]) == t].copy().sort_values("symbol").reset_index(drop=True)
    cands = contract.compute_derived_features(day)
    cands = attach_lagged_flow_features(cands, ph)
    frame = attach_topk_features(cands, ph).sort_values("symbol").reset_index(drop=True)
    pin = GOLDEN[contract.TOPK_FEATURE_CONTRACT_VERSION]
    assert set(pin) == set(contract.RANKER_FEATURE_COLS)
    for name in contract.RANKER_FEATURE_COLS:
        assert np.allclose(
            frame[name].to_numpy(dtype=np.float64),
            np.asarray(pin[name], dtype=np.float64),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ), name


def test_manifest_uses_declared_units() -> None:
    import src.ml.topk_contract as contract

    manifest = contract.build_topk_feature_manifest(contract.RANKER_FEATURE_COLS)
    assert list(manifest["feature_name"]) == list(contract.RANKER_FEATURE_COLS)
    assert len(manifest) == 27
    by_name = {row["feature_name"]: row for _, row in manifest.iterrows()}
    assert by_name["intraday_range"]["unit"] == "decimal_ratio"
    assert by_name["f_tick_cost"]["unit"] == "basis_points"
    assert by_name["inst_density"]["availability_rule"] == "prior_confirmed"
    assert by_name["inst_rank"]["availability_rule"] == "prior_confirmed"


def test_manifest_rejects_unknown_feature() -> None:
    import src.ml.topk_contract as contract

    with pytest.raises(ValueError, match="not_a_feature"):
        contract.build_topk_feature_manifest(["chg_ratio", "not_a_feature"])
    with pytest.raises(TypeError):
        contract.build_topk_feature_manifest("chg_ratio")


def test_keyless_bundle_resolves_to_legacy_baseline() -> None:
    import src.ml.topk_contract as contract

    assert contract.bundle_feature_contract_version({}) == contract.LEGACY_FEATURE_CONTRACT_VERSION
    assert contract.feature_contract_issue({}) is None


def test_keyless_bundle_rejected_after_bump(monkeypatch) -> None:
    import src.ml.topk_contract as contract

    monkeypatch.setattr(contract, "TOPK_FEATURE_CONTRACT_VERSION", "2")
    issue = contract.feature_contract_issue({})
    assert issue is not None
    assert "missing" in issue
    assert "2" in issue


def test_mismatched_stamp_rejected() -> None:
    import src.ml.topk_contract as contract

    bundle = {contract.FEATURE_CONTRACT_VERSION_KEY: "0"}
    with pytest.raises(ValueError, match="0"):
        contract.assert_feature_contract(bundle, source="x")
    with pytest.raises(ValueError, match="1") as excinfo:
        contract.assert_feature_contract(bundle, source="x")
    assert "1" in str(excinfo.value)


@pytest.mark.parametrize("stamp", [1, "", None])
def test_malformed_stamp_rejected_without_coercion(stamp) -> None:
    import src.ml.topk_contract as contract

    bundle = {contract.FEATURE_CONTRACT_VERSION_KEY: stamp}
    with pytest.raises(ValueError, match="non-empty"):
        contract.bundle_feature_contract_version(bundle)
    assert contract.feature_contract_issue(bundle) is not None


def test_legacy_acceptance_is_logged(caplog) -> None:
    import src.ml.topk_contract as contract

    with caplog.at_level(logging.WARNING):
        assert contract.assert_feature_contract({}, source="dir") is None
    assert len([r for r in caplog.records if "legacy_assumed" in r.getMessage()]) == 1
    record = next(r for r in caplog.records if "legacy_assumed" in r.getMessage())
    assert "[ALGO]" in record.getMessage()
    assert "dir" in record.getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            contract.assert_feature_contract(
                {contract.FEATURE_CONTRACT_VERSION_KEY: "1"}, source="dir"
            )
            is None
        )
    assert [r for r in caplog.records if "legacy_assumed" in r.getMessage()] == []


def _parity_bundle(screen: dict, strategy_id: str | None = "KCA-TOPK-COSTAWARE-001") -> dict:
    bundle: dict = {"select_universe": dict(screen)}
    if strategy_id is not None:
        bundle["strategy_id"] = strategy_id
    return bundle


def test_production_certified_bundle_passes_parity_silently(caplog) -> None:
    import dataclasses
    import logging

    import src.ml.topk_contract as contract
    from src.strategy.contract import PRODUCTION_STRATEGY

    bundle = _parity_bundle(
        dataclasses.asdict(PRODUCTION_STRATEGY.universe), "KCA-TOPK-COSTAWARE-002")
    with caplog.at_level(logging.WARNING):
        assert contract.assert_bundle_screen_parity(bundle) is None
    assert [r for r in caplog.records if "GRANDFATHERED" in r.getMessage()] == []


def test_live_001_bundle_is_grandfathered(caplog) -> None:
    import dataclasses
    import logging

    import src.ml.topk_contract as contract
    from src.strategy.contract import COST_AWARE_UNIVERSE

    bundle = _parity_bundle(dataclasses.asdict(COST_AWARE_UNIVERSE))
    with caplog.at_level(logging.WARNING):
        assert contract.assert_bundle_screen_parity(bundle) is None
    records = [r for r in caplog.records if "GRANDFATHERED" in r.getMessage()]
    assert len(records) == 1
    assert "status=GRANDFATHERED" in records[0].getMessage()


def test_keyless_001_bundle_is_grandfathered() -> None:
    import dataclasses

    import src.ml.topk_contract as contract
    from src.strategy.contract import COST_AWARE_UNIVERSE

    screen = dataclasses.asdict(COST_AWARE_UNIVERSE)
    screen.pop("exclude_non_screenable_class")
    assert contract.assert_bundle_screen_parity(_parity_bundle(screen)) is None


def test_grandfather_is_scoped_to_strategy_id() -> None:
    import dataclasses

    import pytest

    import src.ml.topk_contract as contract
    from src.strategy.contract import COST_AWARE_UNIVERSE

    screen = dataclasses.asdict(COST_AWARE_UNIVERSE)
    with pytest.raises(ValueError, match="exclude_non_screenable_class"):
        contract.assert_bundle_screen_parity(_parity_bundle(screen, None))
    with pytest.raises(ValueError, match="exclude_non_screenable_class"):
        contract.assert_bundle_screen_parity(_parity_bundle(screen, "KCA-TOPK-CAPFREE-001"))


def test_grandfather_never_hides_other_drift() -> None:
    import dataclasses

    import pytest

    import src.ml.topk_contract as contract
    from src.strategy.contract import COST_AWARE_UNIVERSE

    screen = dataclasses.asdict(COST_AWARE_UNIVERSE)
    screen["max_tick_cost_bp"] = 7.5
    with pytest.raises(ValueError, match=r"max_tick_cost_bp.*exclude_non_screenable_class"):
        contract.assert_bundle_screen_parity(_parity_bundle(screen))


def test_parity_rejects_reverse_class_direction() -> None:
    import dataclasses

    import pytest

    import src.ml.topk_contract as contract
    from src.strategy.contract import COST_AWARE_UNIVERSE, PRODUCTION_STRATEGY

    bundle = _parity_bundle(dataclasses.asdict(PRODUCTION_STRATEGY.universe))
    with pytest.raises(ValueError, match="exclude_non_screenable_class"):
        contract.assert_bundle_screen_parity(bundle, spec=COST_AWARE_UNIVERSE)


def test_certified_screen_normalizes_omissions_and_numerics() -> None:
    import dataclasses

    import src.ml.topk_contract as contract
    from src.strategy.contract import COST_AWARE_UNIVERSE

    joblib_style = dataclasses.asdict(COST_AWARE_UNIVERSE)
    joblib_style["min_trade_value_100m"] = 100
    joblib_style["min_market_cap_100m"] = 500
    del joblib_style["exclude_non_screenable_class"]
    assert contract.certified_screen({"select_universe": joblib_style}) == contract.certified_screen(
        {"select_universe": dataclasses.asdict(COST_AWARE_UNIVERSE)})


def test_certified_screen_rejects_non_dict_and_unknown_fields() -> None:
    import pytest

    import src.ml.topk_contract as contract

    with pytest.raises(ValueError, match="not certified"):
        contract.certified_screen({})
    with pytest.raises(ValueError, match="unknown to live screen"):
        contract.certified_screen({"select_universe": {"future_field": True}})
