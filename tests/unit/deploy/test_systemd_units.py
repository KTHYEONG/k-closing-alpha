


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
    containerized = {"kca-retrain.service"}

    for svc in sorted(root.glob("kca-*.service")):
        text = svc.read_text(encoding="utf-8")
        assert "WorkingDirectory=%h/k-closing-alpha" in text, svc.name
        if svc.name in containerized:
            assert "%h/k-closing-alpha/data" in text, svc.name
        else:
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


def test_retrain_service_runs_containerized_with_measured_resource_limits() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-retrain.service").read_text(encoding="utf-8")

    assert "docker run --rm" in text
    assert "--memory=6g" in text
    assert "--cpus=1.8" in text
    assert "OnFailure=kca-alert@%n.service" in text
    assert "src.ml.retrain --train-ranker-bundle" in text


def test_code_sync_timer_exists_and_install_script_enables_it() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    timer = (base / "systemd" / "kca-code-sync.timer").read_text(encoding="utf-8")
    service = (base / "systemd" / "kca-code-sync.service").read_text(encoding="utf-8")
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    assert "Unit=kca-code-sync.service" in timer
    assert "OnCalendar=Mon..Fri" in timer
    assert "src.tools.code_sync" in service
    assert "OnFailure=kca-alert@%n.service" in service
    assert "kca-code-sync.timer" in install_text


def test_backup_service_copies_data_and_artifacts_to_gdrive_without_deleting() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup.service").read_text(encoding="utf-8")
    exec_lines = [line for line in text.splitlines() if line.startswith("ExecStart=")]

    assert len(exec_lines) == 2
    assert all("rclone copy" in line for line in exec_lines)
    assert "rclone sync" not in text
    assert "--exclude" not in text
    assert "gdrive:quant-lake/live/k-closing-alpha/data" in text
    assert "gdrive:quant-lake/live/k-closing-alpha/artifacts" in text


def test_backup_prune_service_runs_dated_directory_pruner() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup-prune.service").read_text(encoding="utf-8")

    assert "src.tools.backup_prune" in text
    assert "--min-age" not in text
    assert "After=kca-backup.service" in text


def test_every_kca_service_pins_kst_timezone() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    services = sorted(root.glob("kca-*.service"))

    assert services
    offenders = [p.name for p in services if "Environment=TZ=Asia/Seoul" not in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_every_kca_service_except_alert_template_has_failure_alert() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    services = [p for p in sorted(root.glob("kca-*.service")) if p.name != "kca-alert@.service"]

    missing = [p.name for p in services if "OnFailure=kca-alert@%n.service" not in p.read_text(encoding="utf-8")]
    assert missing == []
    # 알림 템플릿이 자기 자신을 OnFailure로 부르면 실패 루프가 된다
    assert "OnFailure=" not in (root / "kca-alert@.service").read_text(encoding="utf-8")


def test_decision_path_timers_fire_with_one_second_accuracy() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    for name in ("kca-collect.timer", "kca-predict.timer", "kca-finalize-close.timer", "kca-paper-exit.timer"):
        assert "AccuracySec=1s" in (root / name).read_text(encoding="utf-8"), name


def test_code_sync_runs_before_morning_ingest_and_paper_exit() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    def _first_time(name: str) -> str:
        text = (root / name).read_text(encoding="utf-8")
        match = re.search(r"OnCalendar=.*?(\d{2}:\d{2}:\d{2})", text)
        assert match is not None, name
        return match.group(1)

    ingest_text = (root / "kca-price-ingest.timer").read_text(encoding="utf-8")
    ingest_times = sorted(re.findall(r"OnCalendar=.*?(\d{2}:\d{2}:\d{2})", ingest_text))

    assert _first_time("kca-code-sync.timer") < ingest_times[0]
    assert _first_time("kca-code-sync.timer") < _first_time("kca-paper-exit.timer")


def test_backup_runs_after_evening_price_ingest() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    def _first_time(name: str) -> str:
        text = (root / name).read_text(encoding="utf-8")
        match = re.search(r"OnCalendar=.*?(\d{2}:\d{2}:\d{2})", text)
        assert match is not None, name
        return match.group(1)

    ingest_text = (root / "kca-price-ingest.timer").read_text(encoding="utf-8")
    latest_ingest = max(re.findall(r"OnCalendar=.*?(\d{2}:\d{2}:\d{2})", ingest_text))
    service = (root / "kca-backup.service").read_text(encoding="utf-8")
    after_line = next(line for line in service.splitlines() if line.startswith("After="))

    assert _first_time("kca-backup.timer") > latest_ingest
    assert "kca-price-ingest.service" in after_line


def test_daily_audit_waits_for_intraday_archive() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-daily-audit.service").read_text(encoding="utf-8")
    after_line = next(line for line in text.splitlines() if line.startswith("After="))

    assert "kca-archive-intraday.service" in after_line


def test_finalize_close_always_hands_off_to_paper_entry() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    # Given
    lines = (root / "kca-finalize-close.service").read_text(encoding="utf-8").splitlines()

    # Then: 확정 성공/실패와 무관하게 페이퍼 진입이 기동되어 픽별 종결 레코드를 남긴다
    assert "ExecStopPost=/usr/bin/systemctl --user start --no-block kca-paper-entry.service" in lines
    assert not any(line.startswith("OnSuccess=") for line in lines)
    assert "OnFailure=kca-alert@%n.service" in lines




def test_paper_exit_timer_fires_after_open_auction_and_catches_up() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-paper-exit.timer").read_text(encoding="utf-8").splitlines()

    # Then: 시가단일가(09:00:00) 체결 후 발화, 놓치면 재개 직후 이어받는다
    assert "Persistent=true" in lines
    assert "AccuracySec=1s" in lines
    assert "OnCalendar=Mon..Fri 09:01:00 Asia/Seoul" in lines


def test_paper_exit_service_timeout_covers_quote_retry_budget_only() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-paper-exit.service").read_text(encoding="utf-8").splitlines()

    assert "TimeoutStartSec=15min" in lines
    assert not any(line.startswith("TimeoutStartSec=") and line.endswith("h") for line in lines)


def test_paper_entry_backstop_timer_fires_after_finalize_deadline_and_catches_up() -> None:
    import pathlib
    import re

    from src.config.market_session import CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-paper-entry.timer").read_text(encoding="utf-8")
    lines = text.splitlines()

    # Then: 종가확정 데드라인 이후에만 발화(미확정 종가를 진입가로 오인 방지)
    assert "OnCalendar=Mon..Fri 15:34:00 Asia/Seoul" in lines
    match = re.search(r"OnCalendar=Mon\.\.Fri (\d{2}):(\d{2}):(\d{2}) Asia/Seoul", text)
    assert match is not None
    fire_hhmmss = "".join(match.groups())
    assert fire_hhmmss > CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS
    assert "AccuracySec=1s" in lines
    assert "Persistent=true" in lines
    assert "Unit=kca-paper-entry.service" in lines
    assert "WantedBy=timers.target" in lines


def test_install_script_enables_paper_entry_backstop_alongside_finalize_close() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    # Then: ExecStopPost 체이닝이 조용히 끊겨도 독립 타이머가 진입 시도를 보증한다
    assert "kca-paper-entry.timer" in install_text
    assert "kca-finalize-close.timer" in install_text


def test_finalize_close_still_hands_off_to_paper_entry_via_execstoppost() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-finalize-close.service").read_text(encoding="utf-8").splitlines()

    # Then: 빠른 경로(ExecStopPost)는 그대로 유지 — 독립 타이머는 보증 경로일 뿐 대체가 아니다
    assert "ExecStopPost=/usr/bin/systemctl --user start --no-block kca-paper-entry.service" in lines
