def test_build_runtime_fragment_emits_declared_keys_in_canonical_order(tmp_path) -> None:
    from src.tools.provision_env import RUNTIME_ENV_SPEC, build_runtime_fragment

    source = tmp_path / ".quant.env"
    assignments = [
        "# comment",
        "",
        "export LIVE_ALERT_GMAIL_USER=alert@example.com",
        "export LIVE_ALERT_GMAIL_APP_PASSWORD=alert-pass",
        "export ALERT_GMAIL_TO=ops@example.com",
        "export KIS_APP_KEY=kis-key",
        'export KIS_APP_SECRET="kis-secret"',
        "export KIS_ACCOUNT_ID=12345678",
        "export KIS_HTS_ID=hts-id",
        "export KIWOM_APP_KEY=kiwoom-key",
        "export KIWOM_SECRET_KEY=kiwoom-secret",
        "export LS_APP_KEY=ls-key",
        "export LS_APP_SECRET=ls-secret",
        "export KRX_OPENAPI_KEY=krx-key",
        "export TOSS_APP_KEY=toss-key",
        "TOSS_APP_SECRET='toss-secret'",
        "export OPENDART_API_KEY=dart-key",
    ]
    source.write_text("\n".join(assignments) + "\n", encoding="utf-8")

    fragment = build_runtime_fragment(source)

    lines = fragment.splitlines()
    required = [key.target for key in RUNTIME_ENV_SPEC if not key.optional]
    assert [line.partition("=")[0] for line in lines] == required
    assert len(lines) == len(required)
    assert fragment.endswith("\n")
    assert "export " not in fragment
    assert "KIS_APP_SECRET=kis-secret" in lines
    assert "TOSS_APP_SECRET=toss-secret" in lines
    assert "ALERT_GMAIL_USER=alert@example.com" in lines
    assert "ALERT_GMAIL_APP_PASSWORD=alert-pass" in lines
    assert "LIVE_ALERT_GMAIL_USER" not in fragment
    assert "KIS_ACCOUNT_ID=12345678" in lines


def test_runtime_env_spec_excludes_legacy_data_single_key_fields() -> None:
    from src.tools.provision_env import RUNTIME_ENV_SPEC

    targets = [key.target for key in RUNTIME_ENV_SPEC]
    sources = [source for key in RUNTIME_ENV_SPEC for source in key.sources]

    assert len(targets) == 16
    assert len(set(targets)) == 16
    for forbidden in ("KIS_DATA_APP_KEY", "KIS_DATA_APP_SECRET", "KIS_DATA_HTS_ID"):
        assert forbidden not in targets
        assert forbidden not in sources


def test_runtime_env_spec_excludes_shared_keypool_keys() -> None:
    from src.tools.provision_env import RUNTIME_ENV_SPEC

    targets = [key.target for key in RUNTIME_ENV_SPEC]
    sources = [source for key in RUNTIME_ENV_SPEC for source in key.sources]

    assert len(targets) == 16
    assert len(set(targets)) == 16
    for forbidden in ("KIS_DATA_SLOTS", "KIS_HOST_DATA_SLOTS"):
        assert forbidden not in targets
        assert forbidden not in sources
    for slot in range(1, 6):
        for field in ("APP_KEY", "APP_SECRET", "HTS_ID", "ACCOUNT_NO", "ACCOUNT_PRODUCT_CODE"):
            assert f"KIS_DATA_{slot}_{field}" not in targets
            assert f"KIS_DATA_{slot}_{field}" not in sources


def test_build_runtime_fragment_accepts_kis_account_no_alias_for_account_id(tmp_path) -> None:
    from src.tools.provision_env import RUNTIME_ENV_SPEC, build_runtime_fragment

    source = tmp_path / ".quant.env"
    lines = [
        f"{key.target}=value-{index}"
        for index, key in enumerate(RUNTIME_ENV_SPEC)
        if key.target != "KIS_ACCOUNT_ID"
    ]
    lines.append("KIS_ACCOUNT_NO=87654321")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fragment = build_runtime_fragment(source)

    assert "KIS_ACCOUNT_ID=87654321" in fragment.splitlines()
    assert "KIS_ACCOUNT_NO=" not in fragment


def test_build_runtime_fragment_rejects_missing_required_key(tmp_path) -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, RUNTIME_ENV_SPEC, build_runtime_fragment

    source = tmp_path / ".quant.env"
    lines = [
        f"{key.target}=value-{index}"
        for index, key in enumerate(RUNTIME_ENV_SPEC)
        if key.target != "KIS_ACCOUNT_ID"
    ]
    lines.append("KIS_ACCOUNT_ID=")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ProvisioningError, match="KIS_ACCOUNT_ID"):
        build_runtime_fragment(source)


def test_build_runtime_fragment_never_leaks_undeclared_keys(tmp_path) -> None:
    from src.tools.provision_env import RUNTIME_ENV_SPEC, build_runtime_fragment

    source = tmp_path / ".quant.env"
    lines = [f"{key.target}=value-{index}" for index, key in enumerate(RUNTIME_ENV_SPEC)]
    lines.extend(
        [
            "KIS_DATA_SLOTS=1,2,3,4,5",
            "KIS_DATA_1_APP_SECRET=pool-secret",
            "KIS_TRADE_APP_KEY=trade-key",
            "BINANCE_SECRET_KEY=binance-secret",
            "LIVE_ARTIFACT_KEY=artifact-key",
        ]
    )
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fragment = build_runtime_fragment(source)

    for forbidden in (
        "KIS_DATA_SLOTS",
        "KIS_DATA_1_APP_SECRET",
        "pool-secret",
        "KIS_TRADE_APP_KEY",
        "trade-key",
        "BINANCE_SECRET_KEY",
        "binance-secret",
        "LIVE_ARTIFACT_KEY",
        "artifact-key",
    ):
        assert forbidden not in fragment


def test_install_runtime_fragment_sends_values_only_on_ssh_stdin(monkeypatch) -> None:
    import subprocess

    import src.tools.provision_env as provisioning

    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(provisioning.subprocess, "run", fake_run)
    fragment = "KIS_APP_SECRET=secret-value\n"

    provisioning.install_runtime_fragment("or-vps", fragment)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "ssh"
    assert args[1] == "or-vps"
    # 단일 문자열 인자: ssh가 argv[2:]를 공백으로 이어붙여 원격 셸에 전달하므로
    # 여러 인자로 나누면 개행 포함 스크립트가 재분리되어 환경변수가 유출된다.
    assert len(args) == 3
    assert args[2].startswith("bash -c ")
    assert kwargs["input"] == fragment
    assert kwargs["check"] is True
    assert kwargs["text"] is True
    assert "shell" not in kwargs
    assert all("secret-value" not in part for part in args)
    remote_script = args[2]
    assert provisioning.REMOTE_RUNTIME_ENV_PATH == "/home/ubuntu/quant-secrets/k-closing-alpha.env"
    assert provisioning.REMOTE_RUNTIME_ENV_PATH in remote_script
    assert "set -euo pipefail" in remote_script
    assert "chmod 0700" in remote_script
    assert "chmod 600" in remote_script
    assert "chown ubuntu:ubuntu" in remote_script
    assert "mv -f" in remote_script


def test_parse_workstation_assignments_normalizes_and_rejects_duplicates(tmp_path) -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, parse_workstation_assignments

    accepted = frozenset({"ALPHA", "BETA", "GAMMA", "DELTA"})
    source = tmp_path / ".quant.env"
    source.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "no_assignment_line",
                "export ALPHA=one",
                '  BETA = "two"  ',
                "GAMMA='three'",
                "DELTA=",
                "OUT_OF_SCOPE=first",
                "OUT_OF_SCOPE=second",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_workstation_assignments(source, accepted)

    assert parsed == {"ALPHA": "one", "BETA": "two", "GAMMA": "three"}

    duplicate = tmp_path / "dup.env"
    duplicate.write_text("ALPHA=one\nexport ALPHA=two\n", encoding="utf-8")

    with pytest.raises(ProvisioningError, match="ALPHA"):
        parse_workstation_assignments(duplicate, accepted)


def test_parse_workstation_assignments_resolves_shell_variable_reference(tmp_path) -> None:
    """실측 회귀: KIS_APP_KEY=$KIS_TRADE_APP_KEY 리터럴 텍스트가 그대로 배포돼
    KIS_APP_KEY 가 빈 문자열로 주입된 2026-09-16 프로덕션 장애 재발 방지."""
    from src.tools.provision_env import parse_workstation_assignments

    accepted = frozenset({"KIS_APP_KEY", "KIS_ACCOUNT_ID"})
    source = tmp_path / ".quant.env"
    source.write_text(
        "\n".join(
            [
                "export KIS_TRADE_APP_KEY=trade-key-value",
                "export KIS_APP_KEY=$KIS_TRADE_APP_KEY",
                "export KIS_TRADE_ACCOUNT_NO=trade-account",
                "export KIS_ACCOUNT_ID=${KIS_TRADE_ACCOUNT_NO}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_workstation_assignments(source, accepted)

    assert parsed == {"KIS_APP_KEY": "trade-key-value", "KIS_ACCOUNT_ID": "trade-account"}
    assert "$" not in parsed["KIS_APP_KEY"]


def test_parse_workstation_assignments_drops_unresolvable_reference_as_missing(tmp_path) -> None:
    """참조 대상이 파일에 없으면 리터럴 텍스트를 흘리지 않고 빈 값(=미존재)으로 취급한다."""
    from src.tools.provision_env import parse_workstation_assignments

    accepted = frozenset({"KIS_APP_KEY"})
    source = tmp_path / ".quant.env"
    source.write_text("export KIS_APP_KEY=$UNDEFINED_ELSEWHERE\n", encoding="utf-8")

    parsed = parse_workstation_assignments(source, accepted)

    assert "KIS_APP_KEY" not in parsed


def test_parse_workstation_assignments_rejects_circular_variable_reference(tmp_path) -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, parse_workstation_assignments

    accepted = frozenset({"ALPHA"})
    source = tmp_path / ".quant.env"
    source.write_text("export ALPHA=$BETA\nexport BETA=$ALPHA\n", encoding="utf-8")

    with pytest.raises(ProvisioningError, match="circular"):
        parse_workstation_assignments(source, accepted)


def test_parse_workstation_assignments_rejects_unreadable_source(tmp_path) -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, parse_workstation_assignments

    missing = tmp_path / "missing.env"

    with pytest.raises(ProvisioningError, match="missing"):
        parse_workstation_assignments(missing, frozenset({"ALPHA"}))


def test_main_builds_before_install_and_dry_run_installs_nothing(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import src.tools.provision_env as cli

    events: list[str] = []
    _stub_remote(monkeypatch, cli, env_text="", commit="abc")
    monkeypatch.setattr(cli, "build_runtime_fragment", lambda path: events.append("build") or "KIS_APP_SECRET=secret-value\n")
    monkeypatch.setattr(cli, "install_runtime_fragment", lambda host, fragment: events.append(f"install:{host}"))
    source = tmp_path / ".quant.env"
    source.write_text("KIS_APP_SECRET=secret-value\n", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        assert cli.main(["--host", "or-vps", "--source", str(source)]) == 0

    assert events == ["build", "install:or-vps"]
    assert "secret-value" not in caplog.text

    events.clear()

    def fail_install(host: str, fragment: str) -> None:
        raise AssertionError("dry-run must not install")

    monkeypatch.setattr(cli, "install_runtime_fragment", fail_install)
    assert cli.main(["--host", "or-vps", "--source", str(source), "--dry-run"]) == 0
    assert events == ["build"]


def _full_source_lines(exclude: set[str] | None = None) -> list[str]:
    from src.tools.provision_env import RUNTIME_ENV_SPEC

    exclude = exclude or set()
    return [f"{key.target}=value-{i}" for i, key in enumerate(RUNTIME_ENV_SPEC) if key.target not in exclude]


def test_build_runtime_fragment_optional_key_absent_is_omitted(tmp_path) -> None:
    """Optional key absent is omitted."""
    from src.tools.provision_env import build_runtime_fragment

    source = tmp_path / ".quant.env"
    source.write_text("\n".join(_full_source_lines(exclude={"OPENDART_API_KEY_2"})) + "\n", encoding="utf-8")
    fragment = build_runtime_fragment(source)
    assert "OPENDART_API_KEY_2" not in fragment


def test_build_runtime_fragment_optional_key_present_in_spec_order(tmp_path) -> None:
    """Optional key present is emitted in spec order."""
    from src.tools.provision_env import RUNTIME_ENV_SPEC, build_runtime_fragment

    source = tmp_path / ".quant.env"
    source.write_text("\n".join(_full_source_lines()) + "\n", encoding="utf-8")
    fragment = build_runtime_fragment(source)
    lines = fragment.splitlines()
    targets = [line.partition("=")[0] for line in lines]
    assert targets == [key.target for key in RUNTIME_ENV_SPEC]
    assert targets.index("OPENDART_API_KEY_2") == targets.index("OPENDART_API_KEY") + 1


def test_build_runtime_fragment_required_key_absent_fails_closed(tmp_path) -> None:
    """Required key absent still fails closed."""
    import pytest

    from src.tools.provision_env import ProvisioningError, build_runtime_fragment

    source = tmp_path / ".quant.env"
    source.write_text("\n".join(_full_source_lines(exclude={"OPENDART_API_KEY"})) + "\n", encoding="utf-8")
    with pytest.raises(ProvisioningError):
        build_runtime_fragment(source)


def test_merge_remote_env_preserves_foreign_keys_and_replaces_managed(monkeypatch) -> None:
    from src.tools.provision_env import VPS_SELECTORS, merge_remote_env

    fragment = "KIS_APP_KEY=new-key\nOPENDART_API_KEY=dart-new\n"
    remote = "\n".join(
        [
            "# operator note",
            "KIS_APP_KEY=stale-key",
            "OPENDART_API_KEY_2=removed-locally-value",
            "KIS_DECISION_SHARD_SLOTS=9,9",
            "SOME_HOST_ONLY_FLAG=keep-me",
            "SOME_HOST_ONLY_FLAG=keep-me-latest",
            "",
        ]
    )

    merged, preserved = merge_remote_env(fragment, remote)

    lines = merged.splitlines()
    assert lines[:2] == fragment.splitlines()
    assert lines[2 : 2 + len(VPS_SELECTORS)] == [f"{name}={value}" for name, value in VPS_SELECTORS]
    assert lines[-1] == "SOME_HOST_ONLY_FLAG=keep-me-latest"
    assert preserved == ("SOME_HOST_ONLY_FLAG",)
    assert "stale-key" not in merged
    assert "removed-locally-value" not in merged
    assert "9,9" not in merged
    assert len(lines) == len({line.partition("=")[0] for line in lines})
    assert merged.endswith("\n")


def test_merge_remote_env_on_absent_remote_emits_fragment_and_selectors_only() -> None:
    from src.tools.provision_env import VPS_SELECTORS, merge_remote_env

    merged, preserved = merge_remote_env("KIS_APP_KEY=k\n", "")

    assert preserved == ()
    assert merged.splitlines() == ["KIS_APP_KEY=k", *[f"{n}={v}" for n, v in VPS_SELECTORS]]


def test_vps_selectors_form_a_valid_runtime_configuration() -> None:
    """Selectors must satisfy the project's own parsers against the documented pool."""
    from src.api.kis.key_pool import resolve_decision_shard_credentials, resolve_research_credentials
    from src.config.collection import CollectionSettings
    from src.tools.provision_env import VPS_SELECTORS

    env = dict(VPS_SELECTORS)
    env.update({"KIS_DATA_SLOTS": "1,2,3,4,5", "KIS_HOST_DATA_SLOTS": "1,2,3,4", "KIS_APP_KEY": "primary"})
    for slot in range(1, 6):
        env[f"KIS_DATA_{slot}_APP_KEY"] = f"key-{slot}"
        env[f"KIS_DATA_{slot}_APP_SECRET"] = f"secret-{slot}"

    profile = CollectionSettings(
        _env_file=None,
        COLLECTION_AUCTION_ENABLED=env["COLLECTION_AUCTION_ENABLED"],
        COLLECTION_ALTDATA_ENABLED=env["COLLECTION_ALTDATA_ENABLED"],
        COLLECTION_RESEARCH_SLOTS=env["COLLECTION_RESEARCH_SLOTS"],
    )
    research = resolve_research_credentials(env, slots=profile.COLLECTION_RESEARCH_SLOTS)
    shards = resolve_decision_shard_credentials(env)

    extras = resolve_research_credentials(env, slots=CollectionSettings(
        _env_file=None, COLLECTION_ALTDATA_EXTRA_SLOTS=env["COLLECTION_ALTDATA_EXTRA_SLOTS"]
    ).COLLECTION_ALTDATA_EXTRA_SLOTS)

    assert profile.COLLECTION_AUCTION_ENABLED and profile.COLLECTION_ALTDATA_ENABLED
    assert len(research) >= 1
    assert len(shards) >= 2
    assert len(extras) >= 1
    shard_slots = {c.slot for c in shards}
    assert {c.slot for c in research}.isdisjoint(shard_slots)


def test_intraday_kca_slots_never_share_the_krx_snapshot_slot() -> None:
    """krx snapshot REST owns its slot 08:00-15:39; kca intraday users must avoid it."""
    from src.tools.provision_env import KRX_SNAPSHOT_DATA_SLOT, VPS_SELECTORS

    env = dict(VPS_SELECTORS)
    decision_lead = "1"  # KIS_HOST_DATA_SLOTS 선두 = 결정 역할
    intraday = {decision_lead, *env["KIS_DECISION_SHARD_SLOTS"].split(","), *env["COLLECTION_RESEARCH_SLOTS"].split(",")}
    assert KRX_SNAPSHOT_DATA_SLOT not in intraday


def _stub_remote(monkeypatch, cli, *, env_text: str, commit: str, local: str | None = None, validate=None) -> None:
    monkeypatch.setattr(cli, "read_remote_state", lambda host: cli.RemoteState(env_text=env_text, kis_data_text="", image_commit=commit))
    monkeypatch.setattr(cli, "local_code_commit", lambda repo: commit if local is None else local)
    monkeypatch.setattr(cli, "validate_runtime_env", validate or (lambda env_text, kis_text: None))


def _kis_data_text() -> str:
    lines = ["KIS_DATA_SLOTS=1,2,3,4,5", "KIS_HOST_DATA_SLOTS=1,2,3,4"]
    for slot in range(1, 6):
        lines += [f"KIS_DATA_{slot}_APP_KEY=key-{slot}", f"KIS_DATA_{slot}_APP_SECRET=secret-{slot}", f"KIS_DATA_{slot}_HTS_ID=hts"]
    return "\n".join(lines) + "\n"


def _valid_env_text() -> str:
    from src.tools.provision_env import VPS_SELECTORS

    base = ["KIS_APP_KEY=primary", "KIS_APP_SECRET=primary-secret", "OPENDART_API_KEY=dart-1"]
    return "\n".join(base + [f"{n}={v}" for n, v in VPS_SELECTORS]) + "\n"


def test_validate_runtime_env_accepts_the_declared_host_configuration(monkeypatch) -> None:
    from src.tools.provision_env import validate_runtime_env

    # 워크스테이션 환경변수가 검증 결과를 바꾸면 안 된다(대상 파일만 반영).
    monkeypatch.setenv("COLLECTION_AUCTION_ENABLED", "false")
    validate_runtime_env(_valid_env_text(), _kis_data_text())


def test_validate_runtime_env_rejects_silently_disabled_collection() -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, validate_runtime_env

    text = "\n".join(line for line in _valid_env_text().splitlines() if not line.startswith("COLLECTION_ALTDATA_ENABLED")) + "\n"
    with pytest.raises(ProvisioningError, match="COLLECTION_ALTDATA_ENABLED"):
        validate_runtime_env(text, _kis_data_text())


def test_validate_runtime_env_rejects_unresolvable_slots_without_leaking_values() -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, validate_runtime_env

    kis = "\n".join(line for line in _kis_data_text().splitlines() if not line.startswith("KIS_DATA_5_")) + "\n"
    with pytest.raises(ProvisioningError, match="DATA_5") as exc_info:
        validate_runtime_env(_valid_env_text(), kis)
    assert "secret-" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_validate_runtime_env_rejects_unparsable_collection_value() -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, validate_runtime_env

    text = _valid_env_text().replace("COLLECTION_RESEARCH_SLOTS=3,4", "COLLECTION_RESEARCH_SLOTS=DATA_3")
    with pytest.raises(ProvisioningError, match="COLLECTION_RESEARCH_SLOTS"):
        validate_runtime_env(text, _kis_data_text())


def test_validate_runtime_env_requires_a_dart_key_when_altdata_enabled() -> None:
    import pytest

    from src.tools.provision_env import ProvisioningError, validate_runtime_env

    text = _valid_env_text().replace("OPENDART_API_KEY=dart-1\n", "")
    with pytest.raises(ProvisioningError, match="OpenDART"):
        validate_runtime_env(text, _kis_data_text())


def test_diff_env_names_reports_names_only() -> None:
    from src.tools.provision_env import diff_env_names

    added, removed, changed = diff_env_names("A=1\nB=2\nC=3\n", "A=1\nB=20\nD=4\n")

    assert (added, removed, changed) == (("D",), ("C",), ("B",))


def test_main_refuses_removal_without_flag_and_never_installs(tmp_path, monkeypatch) -> None:
    import pytest

    import src.tools.provision_env as cli

    _stub_remote(monkeypatch, cli, env_text="OPENDART_API_KEY_2=old\n", commit="abc")
    monkeypatch.setattr(cli, "build_runtime_fragment", lambda path: "KIS_APP_KEY=k\n")
    monkeypatch.setattr(cli, "install_runtime_fragment", lambda host, text: pytest.fail("must not install"))

    with pytest.raises(cli.ProvisioningError, match="OPENDART_API_KEY_2"):
        cli.main(["--source", str(tmp_path / "x")])

    installed: list[str] = []
    monkeypatch.setattr(cli, "install_runtime_fragment", lambda host, text: installed.append(text))
    assert cli.main(["--source", str(tmp_path / "x"), "--allow-remove"]) == 0
    assert "OPENDART_API_KEY_2" not in installed[0]


def test_main_refuses_version_skew_even_on_dry_run(tmp_path, monkeypatch) -> None:
    import pytest

    import src.tools.provision_env as cli

    monkeypatch.setattr(cli, "build_runtime_fragment", lambda path: "KIS_APP_KEY=k\n")
    monkeypatch.setattr(cli, "install_runtime_fragment", lambda host, text: pytest.fail("must not install"))
    for local in ("new-commit", ""):
        _stub_remote(monkeypatch, cli, env_text="", commit="old-commit", local=local)
        with pytest.raises(cli.ProvisioningError, match="deploy first"):
            cli.main(["--source", str(tmp_path / "x"), "--dry-run"])
    assert cli.main(["--source", str(tmp_path / "x"), "--dry-run", "--allow-version-skew"]) == 0


def test_main_validation_failure_blocks_install(tmp_path, monkeypatch) -> None:
    import pytest

    import src.tools.provision_env as cli

    def _invalid(env_text: str, kis_text: str) -> None:
        raise cli.ProvisioningError("collection settings invalid: COLLECTION_RESEARCH_SLOTS")

    _stub_remote(monkeypatch, cli, env_text="", commit="abc", validate=_invalid)
    monkeypatch.setattr(cli, "build_runtime_fragment", lambda path: "KIS_APP_KEY=k\n")
    monkeypatch.setattr(cli, "install_runtime_fragment", lambda host, text: pytest.fail("must not install"))
    with pytest.raises(cli.ProvisioningError, match="invalid"):
        cli.main(["--source", str(tmp_path / "x")])


def test_main_logs_names_but_no_values(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import src.tools.provision_env as cli

    installed: list[str] = []
    _stub_remote(monkeypatch, cli, env_text="HOST_ONLY=host-secret-value\nKIS_APP_KEY=old-secret\n", commit="abc")
    monkeypatch.setattr(cli, "build_runtime_fragment", lambda path: "KIS_APP_KEY=new-secret\n")
    monkeypatch.setattr(cli, "install_runtime_fragment", lambda host, text: installed.append(text))

    with caplog.at_level(logging.INFO):
        assert cli.main(["--source", str(tmp_path / "x")]) == 0

    assert "HOST_ONLY=host-secret-value" in installed[0]
    assert "changed=KIS_APP_KEY" in caplog.text and "preserved=HOST_ONLY" in caplog.text
    for secret in ("host-secret-value", "old-secret", "new-secret"):
        assert secret not in caplog.text


def test_read_remote_state_splits_sections_and_fails_closed(monkeypatch) -> None:
    import subprocess

    import pytest

    import src.tools.provision_env as provisioning

    sentinel = provisioning._SECTION_SENTINEL
    stdout = f"A=1\n\n{sentinel}\nKIS_DATA_SLOTS=1\n\n{sentinel}\nabc123\n"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(provisioning.subprocess, "run", fake_run)
    state = provisioning.read_remote_state("or-vps")
    assert state.env_text.strip() == "A=1"
    assert state.kis_data_text.strip() == "KIS_DATA_SLOTS=1"
    assert state.image_commit == "abc123"
    assert calls[0][:2] == ["ssh", "or-vps"] and len(calls[0]) == 3

    monkeypatch.setattr(provisioning.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, "garbage", ""))
    with pytest.raises(provisioning.ProvisioningError, match="layout"):
        provisioning.read_remote_state("or-vps")

    def failing_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(255, args)

    monkeypatch.setattr(provisioning.subprocess, "run", failing_run)
    with pytest.raises(provisioning.ProvisioningError, match="refusing blind install"):
        provisioning.read_remote_state("or-vps")


def test_local_code_commit_treats_dirty_src_as_unknown(monkeypatch, tmp_path) -> None:
    import subprocess

    import src.tools.provision_env as provisioning

    def fake_run(dirty: str):
        def _run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            out = "abc123\n" if "rev-parse" in args else dirty
            return subprocess.CompletedProcess(args, 0, out, "")

        return _run

    monkeypatch.setattr(provisioning.subprocess, "run", fake_run(""))
    assert provisioning.local_code_commit(tmp_path) == "abc123"
    monkeypatch.setattr(provisioning.subprocess, "run", fake_run(" M src/x.py\n"))
    assert provisioning.local_code_commit(tmp_path) == ""

    def boom(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("no git")

    monkeypatch.setattr(provisioning.subprocess, "run", boom)
    assert provisioning.local_code_commit(tmp_path) == ""
