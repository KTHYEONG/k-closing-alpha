"""Auto-generated from contract: kis_key_pool."""

from __future__ import annotations

import pytest

def test_parse_host_data_credentials_returns_host_slots_in_order() -> None:
    from src.api.kis.key_pool import KisCredential, parse_host_data_credentials

    # Given: 풀 1,2,3 중 이 호스트는 2,1 순서로 사용, 슬롯 3은 시크릿 미배포
    env = {
        "KIS_DATA_SLOTS": "1,2,3",
        "KIS_HOST_DATA_SLOTS": "2,1",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2",
        "KIS_TRADE_APP_KEY": "trade",
    }

    # When
    creds = parse_host_data_credentials(env)

    # Then
    assert creds == (
        KisCredential(slot="DATA_2", app_key="key2", app_secret="sec2", hts_id=""),
        KisCredential(slot="DATA_1", app_key="key1", app_secret="sec1", hts_id="hts1"),
    )


def test_parse_host_data_credentials_legacy_mode_returns_empty_when_slots_unset() -> None:
    from src.api.kis.key_pool import parse_host_data_credentials

    assert parse_host_data_credentials({}) == ()
    assert parse_host_data_credentials({"KIS_DATA_SLOTS": "  ", "KIS_DATA_APP_KEY": "legacy"}) == ()


def test_parse_host_data_credentials_requires_host_slots_when_pool_defined() -> None:
    import pytest

    from src.api.kis.key_pool import parse_host_data_credentials

    base = {"KIS_DATA_SLOTS": "1", "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1"}

    with pytest.raises(ValueError, match="KIS_HOST_DATA_SLOTS"):
        parse_host_data_credentials(base)
    with pytest.raises(ValueError, match="KIS_HOST_DATA_SLOTS"):
        parse_host_data_credentials({**base, "KIS_HOST_DATA_SLOTS": " "})


def test_parse_host_data_credentials_rejects_malformed_or_foreign_slots() -> None:
    import pytest

    from src.api.kis.key_pool import parse_host_data_credentials

    keys = {
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2",
    }

    # 풀 밖 슬롯
    with pytest.raises(ValueError, match="KIS_DATA_SLOTS"):
        parse_host_data_credentials({**keys, "KIS_DATA_SLOTS": "1,2", "KIS_HOST_DATA_SLOTS": "4"})
    # 비정수 토큰
    with pytest.raises(ValueError, match="invalid data slot"):
        parse_host_data_credentials({**keys, "KIS_DATA_SLOTS": "1,a", "KIS_HOST_DATA_SLOTS": "1"})
    # 0 은 슬롯 번호가 아님
    with pytest.raises(ValueError, match="invalid data slot"):
        parse_host_data_credentials({**keys, "KIS_DATA_SLOTS": "0,1", "KIS_HOST_DATA_SLOTS": "1"})
    # 중복
    with pytest.raises(ValueError, match="duplicate"):
        parse_host_data_credentials({**keys, "KIS_DATA_SLOTS": "1,2", "KIS_HOST_DATA_SLOTS": "1,1"})


def test_parse_host_data_credentials_rejects_missing_secret() -> None:
    import pytest

    from src.api.kis.key_pool import parse_host_data_credentials

    with pytest.raises(ValueError, match="missing credentials"):
        parse_host_data_credentials({
            "KIS_DATA_SLOTS": "1", "KIS_HOST_DATA_SLOTS": "1",
            "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "",
        })
    with pytest.raises(ValueError, match="missing credentials"):
        parse_host_data_credentials({
            "KIS_DATA_SLOTS": "1", "KIS_HOST_DATA_SLOTS": "1",
            "KIS_DATA_1_APP_SECRET": "sec1",
        })


def test_parse_host_data_credentials_rejects_trade_key_reuse_and_duplicates() -> None:
    import pytest

    from src.api.kis.key_pool import parse_host_data_credentials

    pool = {
        "KIS_DATA_SLOTS": "1,2", "KIS_HOST_DATA_SLOTS": "1,2",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2",
    }

    with pytest.raises(ValueError, match="collides"):
        parse_host_data_credentials({**pool, "KIS_TRADE_APP_KEY": "key2"})
    with pytest.raises(ValueError, match="collides"):
        parse_host_data_credentials({**pool, "KIS_APP_KEY": "key1"})
    with pytest.raises(ValueError, match="duplicate"):
        parse_host_data_credentials({**pool, "KIS_DATA_2_APP_KEY": "key1"})
    # 공백 TRADE 키는 비교 대상 아님
    assert len(parse_host_data_credentials({**pool, "KIS_TRADE_APP_KEY": ""})) == 2


def test_select_data_credential_maps_roles_and_rejects_unknown() -> None:
    import pytest

    from src.api.kis.key_pool import KisCredential, select_data_credential

    c1 = KisCredential(slot="DATA_1", app_key="key1", app_secret="sec1", hts_id="")
    c2 = KisCredential(slot="DATA_2", app_key="key2", app_secret="sec2", hts_id="")

    assert select_data_credential((c1, c2), "decision") is c1
    assert select_data_credential((c1, c2), "batch") is c2
    assert select_data_credential((c1,), "batch") is c1
    with pytest.raises(ValueError, match="unknown data role"):
        select_data_credential((c1, c2), "trade")
    with pytest.raises(ValueError, match="no data credentials"):
        select_data_credential((), "decision")


def test_token_cache_path_is_stable_hash_without_raw_key(tmp_path) -> None:
    import hashlib

    from src.api.kis.key_pool import kis_key_id, token_cache_path

    expected_id = hashlib.sha256(b"PSabc").hexdigest()[:12]

    path = token_cache_path("PSabc", tmp_path)

    assert kis_key_id("PSabc") == expected_id
    assert path == tmp_path / f"token_{expected_id}.json"
    assert "PSabc" not in str(path)
    assert token_cache_path("PSabc", tmp_path) == path
    assert token_cache_path("PSxyz", tmp_path) != path


def test_load_kis_env_process_env_overrides_dotenv(tmp_path, monkeypatch) -> None:
    from src.api.kis.key_pool import load_kis_env

    env_file = tmp_path / ".env"
    env_file.write_text("KIS_DATA_SLOTS=1\nKIS_HOST_DATA_SLOTS=1\n", encoding="utf-8")
    monkeypatch.delenv("KIS_DATA_SLOTS", raising=False)
    monkeypatch.setenv("KIS_HOST_DATA_SLOTS", "2")

    env = load_kis_env(env_file)

    assert env["KIS_DATA_SLOTS"] == "1"
    assert env["KIS_HOST_DATA_SLOTS"] == "2"
    missing = load_kis_env(tmp_path / "missing.env")
    assert "KIS_DATA_SLOTS" not in missing
    assert missing["KIS_HOST_DATA_SLOTS"] == "2"


def test_resolve_host_data_credentials_fails_closed_without_pool_env() -> None:
    import pytest

    from src.api.kis.key_pool import KisCredential, resolve_host_data_credentials

    # Given/When/Then: 풀 정의가 없으면 레거시 단일키로 대체하지 않고 즉시 실패한다
    with pytest.raises(ValueError, match="KIS host data slots are not configured"):
        resolve_host_data_credentials({})

    # And: 풀이 선언되면 그대로 해석한다
    pooled = resolve_host_data_credentials({
        "KIS_DATA_SLOTS": "3", "KIS_HOST_DATA_SLOTS": "3",
        "KIS_DATA_3_APP_KEY": "key3", "KIS_DATA_3_APP_SECRET": "sec3",
    })
    assert pooled == (KisCredential(slot="DATA_3", app_key="key3", app_secret="sec3", hts_id=""),)


def test_read_token_issued_date_reads_existing_and_handles_missing_or_corrupt(tmp_path) -> None:
    import json

    from src.api.kis.key_pool import read_token_issued_date

    # Given: 존재하지 않는 파일
    missing = tmp_path / "token_missing.json"

    # Then
    assert read_token_issued_date(missing) is None

    # And: 정상 캐시 파일
    valid = tmp_path / "token_valid.json"
    valid.write_text(
        json.dumps({"access_token": "x", "issued_at": "2026-09-16T07:05:01+09:00", "app_key": "k"}),
        encoding="utf-8",
    )
    assert read_token_issued_date(valid) == "2026-09-16"

    # And: 손상된 JSON
    corrupt = tmp_path / "token_corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert read_token_issued_date(corrupt) is None

    # And: issued_at 필드 자체가 없는 파일
    no_field = tmp_path / "token_no_field.json"
    no_field.write_text(json.dumps({"access_token": "x"}), encoding="utf-8")
    assert read_token_issued_date(no_field) is None

    # And: dict가 아닌 JSON 페이로드
    not_dict = tmp_path / "token_not_dict.json"
    not_dict.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert read_token_issued_date(not_dict) is None



def test_resolve_host_issued_credentials_includes_declared_non_pool_keys() -> None:
    import pytest

    from src.api.kis.key_pool import (
        HOST_ISSUED_KEY_SPECS,
        KisCredential,
        resolve_host_issued_credentials,
    )

    # Given: 풀 슬롯 1개 + 선언된 비풀 키(PRIMARY)
    env = {
        "KIS_DATA_SLOTS": "1", "KIS_HOST_DATA_SLOTS": "1",
        "KIS_DATA_1_APP_KEY": "pool1", "KIS_DATA_1_APP_SECRET": "psec1",
        "KIS_APP_KEY": "primary", "KIS_APP_SECRET": "psecret", "KIS_HTS_ID": "phts",
    }

    # When
    creds = resolve_host_issued_credentials(env)

    # Then: 슬롯 뒤에 선언 키가 붙고 슬롯명이 로그/감사 라벨로 쓰인다
    assert creds == (
        KisCredential(slot="DATA_1", app_key="pool1", app_secret="psec1", hts_id=""),
        KisCredential(slot="PRIMARY", app_key="primary", app_secret="psecret", hts_id="phts"),
    )
    assert [spec.name for spec in HOST_ISSUED_KEY_SPECS] == ["PRIMARY"]

    # And: 선언된 키의 자격증명이 비면 fail-closed
    with pytest.raises(ValueError, match="missing credentials for host key PRIMARY"):
        resolve_host_issued_credentials({
            "KIS_DATA_SLOTS": "1", "KIS_HOST_DATA_SLOTS": "1",
            "KIS_DATA_1_APP_KEY": "pool1", "KIS_DATA_1_APP_SECRET": "psec1",
        })


def test_parse_decision_shard_credentials_returns_empty_when_unset() -> None:
    from src.api.kis.key_pool import parse_decision_shard_credentials

    env = {
        "KIS_DATA_SLOTS": "1,2",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2", "KIS_DATA_2_HTS_ID": "hts2",
    }
    assert parse_decision_shard_credentials(env) == ()


def test_parse_decision_shard_credentials_returns_empty_when_pool_undeclared() -> None:
    from src.api.kis.key_pool import parse_decision_shard_credentials

    env = {"KIS_DECISION_SHARD_SLOTS": "1,5"}
    assert parse_decision_shard_credentials(env) == ()


def test_parse_decision_shard_credentials_resolves_named_slots_in_order() -> None:
    from src.api.kis.key_pool import KisCredential, parse_decision_shard_credentials

    env = {
        "KIS_DATA_SLOTS": "1,2,3,4,5",
        "KIS_DECISION_SHARD_SLOTS": "1,5",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_5_APP_KEY": "key5", "KIS_DATA_5_APP_SECRET": "sec5", "KIS_DATA_5_HTS_ID": "hts5",
    }
    result = parse_decision_shard_credentials(env)
    assert result == (
        KisCredential(slot="DATA_1", app_key="key1", app_secret="sec1", hts_id="hts1"),
        KisCredential(slot="DATA_5", app_key="key5", app_secret="sec5", hts_id="hts5"),
    )


def test_parse_decision_shard_credentials_rejects_slot_outside_pool() -> None:
    from src.api.kis.key_pool import parse_decision_shard_credentials

    env = {
        "KIS_DATA_SLOTS": "1,2",
        "KIS_DECISION_SHARD_SLOTS": "1,5",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
    }
    with pytest.raises(ValueError, match="DATA_5 is not in KIS_DATA_SLOTS pool"):
        parse_decision_shard_credentials(env)


def test_parse_decision_shard_credentials_rejects_missing_secret() -> None:
    from src.api.kis.key_pool import parse_decision_shard_credentials

    env = {
        "KIS_DATA_SLOTS": "1,5",
        "KIS_DECISION_SHARD_SLOTS": "1,5",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_5_APP_KEY": "key5", "KIS_DATA_5_APP_SECRET": "",
    }
    with pytest.raises(ValueError, match="missing credentials for slot DATA_5"):
        parse_decision_shard_credentials(env)


def test_parse_decision_shard_credentials_rejects_duplicate_app_key() -> None:
    from src.api.kis.key_pool import parse_decision_shard_credentials

    env = {
        "KIS_DATA_SLOTS": "1,5",
        "KIS_DECISION_SHARD_SLOTS": "1,5",
        "KIS_DATA_1_APP_KEY": "same", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_5_APP_KEY": "same", "KIS_DATA_5_APP_SECRET": "sec5", "KIS_DATA_5_HTS_ID": "hts5",
    }
    with pytest.raises(ValueError, match="duplicate app_key in decision shard slot DATA_5"):
        parse_decision_shard_credentials(env)


def test_resolve_decision_shard_credentials_falls_back_to_single_decision_key() -> None:
    from src.api.kis.key_pool import (
        resolve_decision_shard_credentials,
        resolve_host_data_credentials,
        select_data_credential,
    )

    env = {
        "KIS_DATA_SLOTS": "1,4",
        "KIS_HOST_DATA_SLOTS": "1,4",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_4_APP_KEY": "key4", "KIS_DATA_4_APP_SECRET": "sec4", "KIS_DATA_4_HTS_ID": "hts4",
    }
    expected = (select_data_credential(resolve_host_data_credentials(env), "decision"),)
    assert resolve_decision_shard_credentials(env) == expected
    assert expected[0].slot == "DATA_1"


def test_resolve_decision_shard_credentials_returns_configured_shards() -> None:
    from src.api.kis.key_pool import KisCredential, resolve_decision_shard_credentials

    env = {
        "KIS_DATA_SLOTS": "1,2,3,4,5",
        "KIS_HOST_DATA_SLOTS": "1,2,3,4",
        "KIS_DECISION_SHARD_SLOTS": "1,5",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2", "KIS_DATA_2_HTS_ID": "hts2",
        "KIS_DATA_3_APP_KEY": "key3", "KIS_DATA_3_APP_SECRET": "sec3", "KIS_DATA_3_HTS_ID": "hts3",
        "KIS_DATA_4_APP_KEY": "key4", "KIS_DATA_4_APP_SECRET": "sec4", "KIS_DATA_4_HTS_ID": "hts4",
        "KIS_DATA_5_APP_KEY": "key5", "KIS_DATA_5_APP_SECRET": "sec5", "KIS_DATA_5_HTS_ID": "hts5",
    }
    result = resolve_decision_shard_credentials(env)
    assert result == (
        KisCredential(slot="DATA_1", app_key="key1", app_secret="sec1", hts_id="hts1"),
        KisCredential(slot="DATA_5", app_key="key5", app_secret="sec5", hts_id="hts5"),
    )


def test_resolve_decision_shard_credentials_rejects_misordered_shard_slots() -> None:
    from src.api.kis.key_pool import resolve_decision_shard_credentials

    env = {
        "KIS_DATA_SLOTS": "1,2,3,4,5",
        "KIS_HOST_DATA_SLOTS": "1,2,3,4",
        "KIS_DECISION_SHARD_SLOTS": "5,1",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2", "KIS_DATA_2_HTS_ID": "hts2",
        "KIS_DATA_3_APP_KEY": "key3", "KIS_DATA_3_APP_SECRET": "sec3", "KIS_DATA_3_HTS_ID": "hts3",
        "KIS_DATA_4_APP_KEY": "key4", "KIS_DATA_4_APP_SECRET": "sec4", "KIS_DATA_4_HTS_ID": "hts4",
        "KIS_DATA_5_APP_KEY": "key5", "KIS_DATA_5_APP_SECRET": "sec5", "KIS_DATA_5_HTS_ID": "hts5",
    }
    with pytest.raises(ValueError, match="KIS_DECISION_SHARD_SLOTS must lead with"):
        resolve_decision_shard_credentials(env)


def test_resolve_host_issued_credentials_includes_decision_shard_extra_slot() -> None:
    from src.api.kis.key_pool import resolve_host_issued_credentials

    env = {
        "KIS_DATA_SLOTS": "1,2,3,4,5",
        "KIS_HOST_DATA_SLOTS": "1,2,3,4",
        "KIS_DECISION_SHARD_SLOTS": "1,5",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2", "KIS_DATA_2_HTS_ID": "hts2",
        "KIS_DATA_3_APP_KEY": "key3", "KIS_DATA_3_APP_SECRET": "sec3", "KIS_DATA_3_HTS_ID": "hts3",
        "KIS_DATA_4_APP_KEY": "key4", "KIS_DATA_4_APP_SECRET": "sec4", "KIS_DATA_4_HTS_ID": "hts4",
        "KIS_DATA_5_APP_KEY": "key5", "KIS_DATA_5_APP_SECRET": "sec5", "KIS_DATA_5_HTS_ID": "hts5",
        "KIS_APP_KEY": "primary", "KIS_APP_SECRET": "psec", "KIS_HTS_ID": "phts",
    }
    slots = [c.slot for c in resolve_host_issued_credentials(env)]
    assert slots == ["DATA_1", "DATA_2", "DATA_3", "DATA_4", "DATA_5", "PRIMARY"]


def test_resolve_host_issued_credentials_dedupes_when_shard_slot_already_a_host_slot() -> None:
    from src.api.kis.key_pool import resolve_host_issued_credentials

    env = {
        "KIS_DATA_SLOTS": "1,4",
        "KIS_HOST_DATA_SLOTS": "1,4",
        "KIS_DECISION_SHARD_SLOTS": "1,4",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_4_APP_KEY": "key4", "KIS_DATA_4_APP_SECRET": "sec4", "KIS_DATA_4_HTS_ID": "hts4",
        "KIS_APP_KEY": "primary", "KIS_APP_SECRET": "psec", "KIS_HTS_ID": "phts",
    }
    slots = [c.slot for c in resolve_host_issued_credentials(env)]
    assert slots == ["DATA_1", "DATA_4", "PRIMARY"]

