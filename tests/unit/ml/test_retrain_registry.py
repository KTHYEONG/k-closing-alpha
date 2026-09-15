from __future__ import annotations


def test_bundle_sha256_prefix_hashes_file_bytes(tmp_path) -> None:
    import hashlib

    import pytest

    from src.ml.retrain_registry import BUNDLE_SHA_PREFIX_LEN, bundle_sha256_prefix

    path = tmp_path / "sizing_pipeline_bundle.joblib"
    path.write_bytes(b"bundle-bytes" * 1000)

    # When
    digest = bundle_sha256_prefix(path)

    # Then
    assert BUNDLE_SHA_PREFIX_LEN == 12
    assert digest == hashlib.sha256(b"bundle-bytes" * 1000).hexdigest()[:12]
    with pytest.raises(FileNotFoundError):
        bundle_sha256_prefix(tmp_path / "missing.joblib")


def test_resolve_code_commit_env_reads_build_arg_or_unknown() -> None:
    from src.ml.retrain_registry import resolve_code_commit_env

    assert resolve_code_commit_env({"KCA_CODE_COMMIT": " f7b15ea0c1d2 "}) == "f7b15ea0c1d2"
    assert resolve_code_commit_env({"KCA_CODE_COMMIT": ""}) == "UNKNOWN"
    assert resolve_code_commit_env({}) == "UNKNOWN"


def test_build_retrain_record_captures_provenance_and_rejects_unknown_outcome(tmp_path) -> None:
    import hashlib

    import pandas as pd
    import pytest

    from src.ml.retrain_registry import RETRAIN_OUTCOME_REJECTED, build_retrain_record

    path = tmp_path / "rejected" / "sizing_pipeline_bundle.joblib"
    path.parent.mkdir()
    path.write_bytes(b"candidate")
    bundle = {
        "strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": pd.Timestamp("2026-09-18"), "train_start": "2016-01-04",
        "trained_at": "2026-09-19T22:05:00+09:00", "feature_cols": ["f1", "f2", "f3"],
    }
    attempted = pd.Timestamp("2026-09-19 22:00:07", tz="Asia/Seoul")

    # When
    record = build_retrain_record(
        outcome=RETRAIN_OUTCOME_REJECTED, bundle=bundle, bundle_path=path, agreement=0.81,
        reasons=("prediction agreement 0.810 below 0.950",), attempted_at=attempted, code_commit="abc123",
    )

    # Then
    assert record == {
        "attempted_at": "2026-09-19T22:00:07+09:00",
        "outcome": "REJECTED",
        "strategy_id": "KCA-TOPK-COSTAWARE-001",
        "training_cutoff": "2026-09-18 00:00:00",
        "train_start": "2016-01-04",
        "trained_at": "2026-09-19T22:05:00+09:00",
        "n_features": 3,
        "agreement": 0.81,
        "reasons": ["prediction agreement 0.810 below 0.950"],
        "bundle_path": str(path),
        "bundle_sha": hashlib.sha256(b"candidate").hexdigest()[:12],
        "code_commit": "abc123",
    }

    # And: 미정의 결과는 fail-closed
    with pytest.raises(ValueError, match="unknown retrain outcome"):
        build_retrain_record(outcome="MAYBE", bundle=bundle, bundle_path=path, agreement=None, reasons=(), attempted_at=attempted, code_commit="abc123")


def test_append_retrain_record_appends_one_json_line_per_attempt(tmp_path) -> None:
    import json

    from src.ml.retrain_registry import append_retrain_record

    registry = tmp_path / "topk_ranker" / "retrain_registry.jsonl"

    # When
    append_retrain_record(registry, {"outcome": "PROMOTED", "agreement": 0.987654321})
    append_retrain_record(registry, {"outcome": "REJECTED", "reasons": ["한글 사유"]})

    # Then
    lines = registry.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"outcome": "PROMOTED", "agreement": 0.987654321},
        {"outcome": "REJECTED", "reasons": ["한글 사유"]},
    ]
    assert "한글 사유" in lines[1]
