import hashlib
import json
from pathlib import Path


def _profile(slots=("5",)):
    from src.config.collection import CollectionSettings

    return CollectionSettings(COLLECTION_RESEARCH_SLOTS=slots)


def _write_consumer(path: Path, text: str) -> str:
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _write_attestation(path: Path, owners: list, *, host_id="host-1", verified_at="2026-09-17T10:00:00+09:00") -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "host_id": host_id,
        "verified_at": verified_at,
        "credential_owners": owners,
    }))


def _owner(slot: str, digests: dict, *, exclusive: bool = True) -> dict:
    return {
        "slot": slot,
        "key_id": "abc123",
        "owner": "k-closing-alpha",
        "purpose": "research",
        "exclusive": exclusive,
        "allowed_rest_roles": ["batch"],
        "verified_consumer_config_sha256": digests,
    }


def _valid_setup(tmp_path, *, slot="5", consumer_text="krx-pool-v1"):
    consumer = tmp_path / "krx.conf"
    digest = _write_consumer(consumer, consumer_text)
    attestation = tmp_path / "ownership.json"
    _write_attestation(attestation, [_owner(f"DATA_{slot}", {"krx": digest})])
    return attestation, {"krx": consumer}


def test_check_activation_accepts_consistent_evidence(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)

    assert check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs=consumers,
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    ) == ()


def test_check_activation_rejects_changed_consumer_hash(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)
    (tmp_path / "krx.conf").write_text("krx-pool-v2")

    reasons = check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs=consumers,
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )

    assert reasons and "fresh attestation" in reasons[0]


def test_check_activation_rejects_unverified_default_slot(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation = tmp_path / "ownership.json"
    _write_attestation(attestation, [])
    consumer = tmp_path / "krx.conf"
    consumer.write_text("x")

    reasons = check_collection_activation(
        _profile(slots=("5",)), ownership_path=attestation, consumer_configs={"krx": consumer},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )

    assert reasons and "DATA_5" in reasons[0]


def test_check_activation_rejects_excess_memory(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)

    reasons = check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs=consumers,
        measured_peak_rss_mib=2048, measured_max_sweep_seconds=30.0,
    )

    assert reasons and "memory" in reasons[0]


def test_check_activation_rejects_slow_sweep(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)

    reasons = check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs=consumers,
        measured_peak_rss_mib=512, measured_max_sweep_seconds=120.0,
    )

    assert reasons and "interval" in reasons[0]


def test_check_activation_rejects_missing_consumer_file(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)
    missing = tmp_path / "absent.conf"

    reasons = check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs={"krx": missing},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )

    assert reasons and "unreadable" in reasons[0]


def test_check_activation_performs_no_mutation(tmp_path, monkeypatch) -> None:
    import subprocess

    import urllib.request

    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)

    def _forbid(*args, **kwargs):
        raise AssertionError("read-only checker must not mutate")

    monkeypatch.setattr(subprocess, "run", _forbid)
    monkeypatch.setattr(urllib.request, "urlopen", _forbid)

    assert check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs=consumers,
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    ) == ()


def test_check_activation_raises_on_invalid_measurements(tmp_path) -> None:
    import pytest

    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)

    with pytest.raises(ValueError, match="measured evidence"):
        check_collection_activation(
            _profile(), ownership_path=attestation, consumer_configs=consumers,
            measured_peak_rss_mib=0, measured_max_sweep_seconds=30.0,
        )
    with pytest.raises(ValueError, match="measured evidence"):
        check_collection_activation(
            _profile(), ownership_path=attestation, consumer_configs=consumers,
            measured_peak_rss_mib=512, measured_max_sweep_seconds=float("nan"),
        )


def test_check_activation_rejects_unreadable_attestation(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    consumer = tmp_path / "krx.conf"
    consumer.write_text("x")

    reasons = check_collection_activation(
        _profile(), ownership_path=tmp_path / "absent.json", consumer_configs={"krx": consumer},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )

    assert reasons and "fresh verification" in reasons[0]


def test_check_activation_rejects_malformed_attestation(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    consumer = tmp_path / "krx.conf"
    consumer.write_text("x")
    bad_schema = tmp_path / "bad.json"
    bad_schema.write_text(json.dumps({"schema_version": 999}))
    naive_time = tmp_path / "naive.json"
    _write_attestation(naive_time, [], verified_at="2026-09-17T10:00:00")

    assert check_collection_activation(
        _profile(), ownership_path=bad_schema, consumer_configs={"krx": consumer},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )
    assert check_collection_activation(
        _profile(), ownership_path=naive_time, consumer_configs={"krx": consumer},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )


def test_check_activation_rejects_empty_slots(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)

    reasons = check_collection_activation(
        _profile(slots=()), ownership_path=attestation, consumer_configs=consumers,
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )

    assert reasons and "slots" in reasons[0]


def test_check_activation_rejects_incomplete_and_conflicting_evidence(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    consumer = tmp_path / "krx.conf"
    digest = _write_consumer(consumer, "v1")
    incomplete = tmp_path / "incomplete.json"
    _write_attestation(incomplete, [_owner("DATA_5", {})])
    conflict = tmp_path / "conflict.json"
    other = _write_consumer(tmp_path / "other.conf", "other")
    _write_attestation(conflict, [_owner("DATA_5", {"krx": digest}), _owner("DATA_6", {"krx": other})])

    assert check_collection_activation(
        _profile(), ownership_path=incomplete, consumer_configs={"krx": consumer},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )
    conflict_reasons = check_collection_activation(
        _profile(slots=("5", "6")), ownership_path=conflict, consumer_configs={"krx": consumer},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )
    assert conflict_reasons and "conflicting" in conflict_reasons[0]


def test_check_activation_rejects_inventory_mismatch(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    attestation, consumers = _valid_setup(tmp_path)
    extra = tmp_path / "extra.conf"
    extra.write_text("extra")

    reasons = check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs={**consumers, "extra": extra},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )

    assert reasons and "inventory" in reasons[0]


def test_check_activation_never_emits_credential_contents(tmp_path) -> None:
    from src.tools.collection_activation import check_collection_activation

    secret = "super-secret-pool-token-xyz"
    attestation, _ = _valid_setup(tmp_path, consumer_text=secret)
    (tmp_path / "krx.conf").write_text("rotated-secret-value")
    consumer = tmp_path / "krx.conf"

    reasons = check_collection_activation(
        _profile(), ownership_path=attestation, consumer_configs={"krx": consumer},
        measured_peak_rss_mib=512, measured_max_sweep_seconds=30.0,
    )

    assert reasons
    assert secret not in reasons[0]
    assert "rotated-secret-value" not in reasons[0]
