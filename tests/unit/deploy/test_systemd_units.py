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
