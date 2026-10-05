"""Decision-time certification report contract tests."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.ml.pit_report import (
    PIT_CERTIFICATION_BUNDLE_KEY,
    AugmentationSummary,
    PairedDelta,
    PitHaircutReport,
    PitReportStatus,
    bundle_pit_certification,
    load_pit_haircut_report,
    pit_certification_metadata,
    save_pit_haircut_report,
)


def _full_report() -> PitHaircutReport:
    return PitHaircutReport(
        generated_at="2026-10-04T00:00:00+09:00",
        strategy_id="KCA-TOPK-COSTAWARE-001",
        strategy_fingerprint="abc123",
        top_k=3,
        select_universe={"chg_min": 0.02, "max_tick_cost_bp": 12.0},
        feature_contract_version="1",
        model_params={"n_estimators": 10},
        seeds=(1,),
        status=PitReportStatus.OK,
        panel_date_min="2023-02-01",
        panel_date_max="2023-07-01",
        n_usable_days=100,
        n_paired_days=90,
        n_live_days=80,
        n_eod_index_days=10,
        mean_net_bp={"eod_full": 10.0, "eod_matched": 9.0, "pit_feature": 8.0, "pit_native": 7.0},
        haircut=PairedDelta(delta=3.0, ci_low=float("nan"), ci_high=1.0, p_value=0.04, n_days=90),
        coverage_component=PairedDelta(delta=1.0, ci_low=0.5, ci_high=1.5, p_value=0.01, n_days=90),
        feature_component=PairedDelta(delta=1.0, ci_low=0.5, ci_high=1.5, p_value=0.01, n_days=90),
        selection_component=PairedDelta(delta=1.0, ci_low=0.5, ci_high=1.5, p_value=0.01, n_days=90),
        pit_native_vs_zero=PairedDelta(delta=7.0, ci_low=5.0, ci_high=9.0, p_value=0.001, n_days=90),
        rank_ic_mean={"eod_full": 0.05, "eod_matched": 0.04, "pit_feature": 0.03, "pit_native": 0.02},
        rank_ic_haircut=PairedDelta(delta=0.03, ci_low=0.01, ci_high=0.05, p_value=0.02, n_days=90),
        pick_overlap_mean={"pit_feature": 0.9, "pit_native": 0.8},
        haircut_by_index_basis={"live_1520": 3.0, "eod_fallback": 2.5},
        augmentation=AugmentationSummary(
            status="OK",
            improvement_bp=PairedDelta(delta=0.5, ci_low=0.1, ci_high=0.9, p_value=0.03, n_days=90),
            declared_trials=1,
            alpha=0.05,
            verdict="REJECT",
        ),
    )


def _daily() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01", "2023-02-01"]),
        "arm": ["eod_full", "pit_native"],
        "topk_net_bp": [10.0, 7.0],
        "rank_ic": [0.05, 0.02],
        "n_pool": [20, 20],
        "overlap_vs_eod": [1.0, 0.8],
        "index_basis": ["live_1520", "live_1520"],
    })


def test_pit_report_round_trip_is_lossless(tmp_path) -> None:
    report = _full_report()
    daily = _daily()

    json_path, parquet_path = save_pit_haircut_report(report, daily, out_dir=tmp_path)
    assert json_path.exists() and parquet_path.exists()
    loaded = load_pit_haircut_report(tmp_path)

    assert loaded is not None
    assert loaded.status == report.status
    assert loaded.augmentation is not None and loaded.augmentation.verdict == "REJECT"
    assert np.isnan(loaded.haircut.ci_low) and loaded.haircut.ci_high == 1.0
    assert loaded.seeds == (1,)
    assert loaded.mean_net_bp == report.mean_net_bp
    reloaded_daily = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(reloaded_daily, daily.reset_index(drop=True))


def test_pit_report_missing_is_none_corrupt_is_error(tmp_path) -> None:
    import pytest

    assert load_pit_haircut_report(tmp_path) is None

    (tmp_path / "pit_haircut_report.json").write_text('{"schema_version": 1,', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_pit_haircut_report(tmp_path)

    (tmp_path / "pit_haircut_report.json").write_text(
        json.dumps({"schema_version": 99, "status": "OK"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_pit_haircut_report(tmp_path)


def test_pit_certification_metadata_missing_report() -> None:
    meta = pit_certification_metadata(None, gate_mode="advisory", gate_status="MISSING", gate_reasons=("no report",))

    assert meta["status"] == "MISSING"
    assert meta["gate_mode"] == "advisory"
    assert meta["gate_status"] == "MISSING"
    assert meta["gate_reasons"] == ["no report"]
    json.dumps(meta)


def test_bundle_pit_certification_backward_compatible() -> None:
    import pytest

    assert bundle_pit_certification({}) is None
    assert bundle_pit_certification({PIT_CERTIFICATION_BUNDLE_KEY: {"status": "OK"}}) == {"status": "OK"}
    with pytest.raises(ValueError, match="mapping"):
        bundle_pit_certification({PIT_CERTIFICATION_BUNDLE_KEY: "OK"})


def _payload_of(report: PitHaircutReport) -> dict:
    import tempfile
    from pathlib import Path

    from src.ml.pit_report import PIT_HAIRCUT_REPORT_FILENAME, save_pit_haircut_report

    tmp = Path(tempfile.mkdtemp())
    save_pit_haircut_report(report, _daily(), out_dir=tmp)
    return json.loads((tmp / PIT_HAIRCUT_REPORT_FILENAME).read_text(encoding="utf-8"))


def test_pit_report_round_trip_without_augmentation_and_metadata(tmp_path) -> None:
    import dataclasses

    report = dataclasses.replace(_full_report(), augmentation=None)
    daily = _daily()
    saved = save_pit_haircut_report(report, daily, out_dir=tmp_path)
    assert all(p.exists() for p in saved)
    loaded = load_pit_haircut_report(tmp_path)

    assert loaded is not None and loaded.augmentation is None
    meta = pit_certification_metadata(
        loaded, gate_mode="enforce", gate_status="PASS", gate_reasons=())
    assert meta["status"] == "OK"
    assert meta["gate_mode"] == "enforce"
    assert meta["haircut"]["delta"] == 3.0
    assert meta["mean_net_bp"]["pit_native"] == 7.0
    json.dumps(meta)


def test_pit_report_rejects_malformed_payloads(tmp_path) -> None:
    import pytest

    base = _payload_of(_full_report())

    def _write(payload) -> None:
        (tmp_path / "pit_haircut_report.json").write_text(json.dumps(payload), encoding="utf-8")

    _write(dict(base, haircut=dict(base["haircut"], delta="x")))
    with pytest.raises(ValueError, match="number or null"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, haircut=5))
    with pytest.raises(ValueError, match="must be a mapping"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, haircut=dict(base["haircut"], extra=1)))
    with pytest.raises(ValueError, match="unknown keys"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, haircut=dict(base["haircut"], n_days="x")))
    with pytest.raises(ValueError, match="n_days"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, augmentation=5))
    with pytest.raises(ValueError, match="mapping or null"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, augmentation={"status": "OK", "bogus": 1}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, mean_net_bp=5))
    with pytest.raises(ValueError, match="must be a mapping"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, extra_top_key=1))
    with pytest.raises(ValueError, match="unknown keys"):
        load_pit_haircut_report(tmp_path)

    slim = {"schema_version": 1, "status": "OK"}
    _write(slim)
    with pytest.raises(ValueError, match="missing keys"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, status="BOGUS"))
    with pytest.raises(ValueError, match="status"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, select_universe=[1]))
    with pytest.raises(ValueError, match="select_universe"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, model_params=[1]))
    with pytest.raises(ValueError, match="model_params"):
        load_pit_haircut_report(tmp_path)

    _write(dict(base, seeds="1"))
    with pytest.raises(ValueError, match="seeds"):
        load_pit_haircut_report(tmp_path)

    (tmp_path / "pit_haircut_report.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_pit_haircut_report(tmp_path)


def test_pit_report_refuses_partial_persistence(tmp_path) -> None:
    import dataclasses

    import pytest

    with pytest.raises(ValueError, match="unpopulated"):
        save_pit_haircut_report(PitHaircutReport(), _daily(), out_dir=tmp_path)

    partial = dataclasses.replace(
        _full_report(), select_universe={}, model_params={}, mean_net_bp={})
    object.__setattr__(partial, "haircut", None)
    with pytest.raises(ValueError, match="haircut"):
        save_pit_haircut_report(partial, _daily(), out_dir=tmp_path)

    partial2 = dataclasses.replace(_full_report())
    object.__setattr__(partial2, "rank_ic_mean", None)
    with pytest.raises(ValueError, match="unpopulated"):
        save_pit_haircut_report(partial2, _daily(), out_dir=tmp_path)


def _recon_cert(**overrides):
    from src.ml.pit_report import CalibrationStability, ReconstructionCertification

    base: dict = {
        "generated_at": "2026-10-05T00:00:00+09:00",
        "exact_dir": "exact",
        "recon_dir": "recon",
        "paired_days": ("2026-03-02", "2026-03-03"),
        "dropped_days": ("2026-03-04",),
        "coverage_improvement": PairedDelta(delta=4.0, ci_low=2.0, ci_high=6.0, p_value=0.001, n_days=60),
        "reconstruction_feature": PairedDelta(delta=0.2, ci_low=-1.0, ci_high=1.4, p_value=0.6, n_days=60),
        "stability": CalibrationStability(passed=True, median_rel_err=0.03, n_scored=40, detail=""),
        "coverage_by_year_and_basis": {"exact": {"2026": {"live_1520": 0.95}}},
        "gate_verdict": "ADOPT",
        "gate_reasons": (),
    }
    base.update(overrides)
    return ReconstructionCertification(**base)


def test_reconstruction_certification_round_trip(tmp_path) -> None:
    import pytest

    from src.ml.pit_report import (
        RECONSTRUCTION_CERTIFICATION_FILENAME,
        load_reconstruction_certification,
        save_reconstruction_certification,
    )

    cert = _recon_cert()
    path = save_reconstruction_certification(cert, out_path=tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    assert path.exists()
    loaded = load_reconstruction_certification(path)
    assert loaded == cert
    assert loaded.stability is not None and loaded.stability.passed is True

    with pytest.raises(FileNotFoundError, match="not found"):
        load_reconstruction_certification(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_reconstruction_certification(bad)
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_reconstruction_certification(bad)


def test_reconstruction_certification_rejects_malformed_payloads(tmp_path) -> None:
    import dataclasses
    import json as json_lib

    import pytest

    from src.ml.pit_report import (
        RECONSTRUCTION_CERTIFICATION_FILENAME,
        load_reconstruction_certification,
        save_reconstruction_certification,
    )

    def _write(payload) -> None:
        (tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME).write_text(
            json_lib.dumps(payload), encoding="utf-8")

    def _payload(cert) -> dict:
        save_reconstruction_certification(cert, out_path=tmp_path / "ok.json")
        return json_lib.loads((tmp_path / "ok.json").read_text(encoding="utf-8"))

    base = _payload(_recon_cert())
    _write(dict(base, extra=1))
    with pytest.raises(ValueError, match="unknown keys"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    _write(dict(base, schema_version=99))
    with pytest.raises(ValueError, match="schema_version"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    slim = {k: v for k, v in base.items() if k != "gate_verdict"}
    _write(slim)
    with pytest.raises(ValueError, match="missing keys"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    _write(dict(base, paired_days="2026-03-02"))
    with pytest.raises(ValueError, match="must be a sequence"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    _write(dict(base, coverage_by_year_and_basis=[]))
    with pytest.raises(ValueError, match="must be a mapping"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    _write(dict(base, gate_verdict="MAYBE"))
    with pytest.raises(ValueError, match="gate_verdict"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    _write(dict(base, stability={"passed": True, "bogus": 1}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    _write(dict(base, stability="yes"))
    with pytest.raises(ValueError, match="mapping or null"):
        load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME)
    nulled = dict(base, stability=None)
    _write(nulled)
    assert load_reconstruction_certification(tmp_path / RECONSTRUCTION_CERTIFICATION_FILENAME).stability is None

    with pytest.raises(ValueError, match="unpopulated"):
        save_reconstruction_certification(
            dataclasses.replace(_recon_cert(), coverage_improvement=None), out_path=tmp_path / "x.json")
    with pytest.raises(ValueError, match="unpopulated"):
        save_reconstruction_certification(
            dataclasses.replace(_recon_cert(), coverage_by_year_and_basis=None), out_path=tmp_path / "y.json")
