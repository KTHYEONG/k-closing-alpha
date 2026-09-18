"""Research-key ownership attestation generator invariant guards."""

from __future__ import annotations

import json

import pytest


def test_build_ownership_document_pins_slot_key_id_and_consumer_digest(tmp_path) -> None:
    from src.api.kis.key_pool import kis_key_id
    from src.tools.generate_research_key_ownership import build_ownership_document

    consumer = tmp_path / "consumer.py"
    consumer.write_text("print('hi')\n", encoding="utf-8")

    document = build_ownership_document(
        host_id="or-vps",
        slots=("2", "3"),
        env={"KIS_DATA_2_APP_KEY": "abc", "KIS_DATA_3_APP_KEY": "xyz"},
        consumer_configs={"auction_capture": consumer},
    )

    assert document["schema_version"] == 1
    assert document["host_id"] == "or-vps"
    owners = {item["slot"]: item for item in document["credential_owners"]}
    assert set(owners) == {"DATA_2", "DATA_3"}
    assert owners["DATA_2"]["key_id"] == kis_key_id("abc")
    assert owners["DATA_3"]["key_id"] == kis_key_id("xyz")
    for entry in owners.values():
        assert entry["owner"] == "k-closing-alpha"
        assert entry["purpose"] == "research"
        assert entry["exclusive"] is True
        assert entry["verified_consumer_config_sha256"]["auction_capture"]


def test_build_ownership_document_rejects_missing_app_key(tmp_path) -> None:
    from src.tools.generate_research_key_ownership import build_ownership_document

    consumer = tmp_path / "consumer.py"
    consumer.write_text("print('hi')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing KIS_DATA_2_APP_KEY"):
        build_ownership_document(
            host_id="or-vps",
            slots=("2",),
            env={},
            consumer_configs={"auction_capture": consumer},
        )


def test_generated_document_satisfies_resolve_research_credentials(tmp_path) -> None:
    from src.api.kis.key_pool import resolve_research_credentials
    from src.tools.generate_research_key_ownership import build_ownership_document

    consumer = tmp_path / "consumer.py"
    consumer.write_text("print('hi')\n", encoding="utf-8")
    document = build_ownership_document(
        host_id="or-vps",
        slots=("2", "3"),
        env={"KIS_DATA_2_APP_KEY": "abc", "KIS_DATA_3_APP_KEY": "xyz"},
        consumer_configs={"auction_capture": consumer},
    )
    ownership_path = tmp_path / "ownership.json"
    ownership_path.write_text(json.dumps(document), encoding="utf-8")

    env = {
        "KIS_DATA_SLOTS": "1,2,3,4,5",
        "KIS_DATA_2_APP_KEY": "abc",
        "KIS_DATA_2_APP_SECRET": "abc-secret",
        "KIS_DATA_3_APP_KEY": "xyz",
        "KIS_DATA_3_APP_SECRET": "xyz-secret",
    }

    creds = resolve_research_credentials(env, slots=("2", "3"), ownership_path=ownership_path)

    assert {cred.slot for cred in creds} == {"DATA_2", "DATA_3"}


def test_main_writes_ownership_json_from_env_file(tmp_path) -> None:
    from src.tools.generate_research_key_ownership import main

    env_file = tmp_path / "kis-data.env"
    env_file.write_text("KIS_DATA_2_APP_KEY=abc\nKIS_DATA_3_APP_KEY=xyz\n", encoding="utf-8")
    out_path = tmp_path / "ownership.json"

    rc = main(["--host-id", "or-vps", "--slot", "2", "--slot", "3", "--env-file", str(env_file), "--out", str(out_path)])

    assert rc == 0
    document = json.loads(out_path.read_text(encoding="utf-8"))
    assert {item["slot"] for item in document["credential_owners"]} == {"DATA_2", "DATA_3"}
