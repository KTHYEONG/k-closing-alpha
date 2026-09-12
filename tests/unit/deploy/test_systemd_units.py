def test_no_unit_hardcodes_home_kth_path() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    offenders = [
        p.name for p in sorted(root.glob("kca-*.service"))
        if "/home/kth" in p.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_every_timer_file_uses_h_specifier_in_its_service() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    for svc in sorted(root.glob("kca-*.service")):
        text = svc.read_text(encoding="utf-8")
        assert "WorkingDirectory=%h/k-closing-alpha" in text, svc.name
        assert "%h/.local/bin/" in text, svc.name


def test_install_script_enables_every_existing_timer() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    timers = sorted(p.name for p in (base / "systemd").glob("kca-*.timer"))
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    missing = [t for t in timers if t not in install_text]

    assert missing == [], f"install_systemd.sh 가 활성화하지 않는 타이머: {missing}"


def test_daily_audit_timer_exists_and_targets_service() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    timer = (root / "kca-daily-audit.timer").read_text(encoding="utf-8")
    service = (root / "kca-daily-audit.service").read_text(encoding="utf-8")

    assert "Unit=kca-daily-audit.service" in timer
    assert "OnCalendar=" in timer
    assert "WantedBy=default.target" not in service


def test_backup_service_syncs_data_and_artifacts_to_gdrive() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup.service").read_text(encoding="utf-8")

    assert "rclone sync" in text
    assert "gdrive:quant-lake/live/k-closing-alpha/data" in text
    assert "gdrive:quant-lake/live/k-closing-alpha/artifacts" in text


def test_backup_runs_after_archive_and_audit() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup.service").read_text(encoding="utf-8")
    after_line = next(line for line in text.splitlines() if line.startswith("After="))

    assert "kca-archive-intraday.service" in after_line
    assert "kca-daily-audit.service" in after_line


def test_backup_timer_schedule_is_after_daily_audit_timer() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    def _first_time(name: str) -> str:
        text = (root / name).read_text(encoding="utf-8")
        match = re.search(r"OnCalendar=.*?(\d{2}:\d{2}:\d{2})", text)
        assert match is not None, name
        return match.group(1)

    audit_time = _first_time("kca-daily-audit.timer")
    backup_time = _first_time("kca-backup.timer")

    assert backup_time > audit_time


def test_backup_uses_backup_dir_instead_of_bare_sync() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup.service").read_text(encoding="utf-8")

    assert "--backup-dir gdrive:quant-lake/live/k-closing-alpha/_deleted/data/" in text
    assert "--backup-dir gdrive:quant-lake/live/k-closing-alpha/_deleted/artifacts/" in text


def test_backup_prune_removes_deleted_snapshots_older_than_30_days() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup-prune.service").read_text(encoding="utf-8")

    assert "rclone delete --min-age 30d gdrive:quant-lake/live/k-closing-alpha/_deleted" in text
    assert "rclone rmdirs gdrive:quant-lake/live/k-closing-alpha/_deleted" in text


def test_backup_prune_timer_exists_and_targets_service() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    timer = (root / "kca-backup-prune.timer").read_text(encoding="utf-8")
    service = (root / "kca-backup-prune.service").read_text(encoding="utf-8")

    assert "Unit=kca-backup-prune.service" in timer
    assert "OnCalendar=" in timer
    assert "After=kca-backup.service" in service


def test_alert_template_service_and_critical_path_onfailure_hooks() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    # Given/When: 알림 템플릿 유닛
    template = (root / "kca-alert@.service").read_text(encoding="utf-8")

    # Then
    assert "WorkingDirectory=%h/k-closing-alpha" in template
    assert "%h/.local/bin/" in template
    assert "src.tools.alerts" in template
    assert "--unit %i" in template

    # And: 결정경로 5개 서비스 전부 OnFailure 훅 보유
    for name in ("kca-collect", "kca-predict", "kca-finalize-close", "kca-paper-entry", "kca-paper-exit"):
        text = (root / f"{name}.service").read_text(encoding="utf-8")
        assert "OnFailure=kca-alert@%n.service" in text, name


def test_retrain_timer_exists_and_install_script_enables_it() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    timer = (base / "systemd" / "kca-retrain.timer").read_text(encoding="utf-8")
    service = (base / "systemd" / "kca-retrain.service").read_text(encoding="utf-8")
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    # Then: 타이머가 서비스를 정확히 겨냥
    assert "Unit=kca-retrain.service" in timer
    assert "OnCalendar=" in timer
    assert "Persistent=true" in timer

    # And: 서비스가 재학습 커맨드 + 실패 얼러트 훅을 가짐
    assert "src.ml.retrain --train-ranker-bundle" in service
    assert "OnFailure=kca-alert@%n.service" in service

    # And: install_systemd.sh 가 이 타이머를 활성화 목록에 포함
    assert "kca-retrain.timer" in install_text
