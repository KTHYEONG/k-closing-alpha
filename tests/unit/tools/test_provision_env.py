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
    assert [line.partition("=")[0] for line in lines] == [key.target for key in RUNTIME_ENV_SPEC]
    assert len(lines) == 15
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

    assert len(targets) == 15
    assert len(set(targets)) == 15
    for forbidden in ("KIS_DATA_APP_KEY", "KIS_DATA_APP_SECRET", "KIS_DATA_HTS_ID"):
        assert forbidden not in targets
        assert forbidden not in sources


def test_runtime_env_spec_excludes_shared_keypool_keys() -> None:
    from src.tools.provision_env import RUNTIME_ENV_SPEC

    targets = [key.target for key in RUNTIME_ENV_SPEC]
    sources = [source for key in RUNTIME_ENV_SPEC for source in key.sources]

    assert len(targets) == 15
    assert len(set(targets)) == 15
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
