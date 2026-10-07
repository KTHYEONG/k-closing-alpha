


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
    containerized = {
        "kca-retrain.service",
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-altdata-capture.service",
        "kca-auction-close.service",
        "kca-auction-open.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
        "kca-tape-sweep.service",
    }

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

    # 연구용(무주문) 수집기는 RSS/소요시간 실측과 감사 없이 상시 가동시키지 않는다는
    # 설계 결정에 따라 install_systemd.sh가 의도적으로 자동 활성화하지 않는다.
    optional_manual_activation = {
        "kca-altdata-capture.timer",
        "kca-auction-close.timer",
        "kca-auction-open.timer",
    }

    missing = [t for t in timers if t not in install_text and t not in optional_manual_activation]

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


def test_backup_service_runs_offsite_runner_under_shared_drive_lock() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup.service").read_text(encoding="utf-8")
    exec_lines = [line for line in text.splitlines() if line.startswith("ExecStart=")]

    assert len(exec_lines) == 1
    assert "/usr/bin/flock -w" in exec_lines[0]
    assert "%t/quant-gdrive.lock" in exec_lines[0]
    assert "src.tools.offsite_backup" in exec_lines[0]
    assert "TimeoutStartSec=" in text
    assert "rclone" not in text


def test_backup_prune_service_holds_shared_drive_lock() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup-prune.service").read_text(encoding="utf-8")

    assert "ExecStart=/usr/bin/flock -w 7200 %t/quant-gdrive.lock" in text
    assert "src.tools.backup_prune" in text
    assert "TimeoutStartSec=" in text


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


def test_alert_template_bounds_its_runtime() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-alert@.service").read_text(encoding="utf-8").splitlines()

    assert "TimeoutStartSec=5min" in lines
    exec_line = next(line for line in lines if line.startswith("ExecStart="))
    assert "--unit %i" in exec_line


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


def test_containerized_units_use_shared_image_and_new_env_file() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    containerized = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
    )
    for name in containerized:
        text = (root / name).read_text(encoding="utf-8")
        assert "docker run --rm" in text, name
        assert "ghcr.io/kthyeong/k-closing-alpha:latest" in text, name
        assert "--env-file %h/quant-secrets/k-closing-alpha.env" in text, name
        assert "%h/k-closing-alpha/.env" not in text, name


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


def test_decision_role_units_forward_env_into_container() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    decision = (
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
    )
    for name in decision:
        text = (root / name).read_text(encoding="utf-8")
        assert "Environment=KIS_DATA_ROLE=decision" in text, name
        exec_line = next(line for line in text.splitlines() if line.startswith("ExecStart=") and "docker run" in line)
        assert "-e KIS_DATA_ROLE=decision" in exec_line, name

    predict_text = (root / "kca-predict.service").read_text(encoding="utf-8")
    assert "-e KIS_DATA_ROLE=decision" not in predict_text


def test_kis_cache_mounted_only_for_units_using_kis_client() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    needs_kis_cache = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
        "kca-auction-open.service",
        "kca-auction-close.service",
        "kca-altdata-capture.service",
        "kca-tape-sweep.service",
    )
    mount = "-v %h/.cache/kis:/app/.cache/kis"
    for name in needs_kis_cache:
        assert mount in (root / name).read_text(encoding="utf-8"), name

    assert mount not in (root / "kca-retrain.service").read_text(encoding="utf-8")


def test_kis_using_containerized_units_forward_key_pool_env_and_cache() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    shared_env = "--env-file %h/quant-secrets/k-closing-alpha.env"
    pool_env = "--env-file %h/quant-secrets/kis-data.env"
    mount = "-v %h/.cache/kis:/app/.cache/kis"
    kis_units = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
    )

    for name in kis_units:
        text = (root / name).read_text(encoding="utf-8")
        exec_line = next(line for line in text.splitlines() if line.startswith("ExecStart=") and "docker run" in line)
        # 드롭인 EnvironmentFile은 docker 클라이언트에만 적용되므로 컨테이너엔 --env-file로 명시해야 한다
        assert pool_env in exec_line, name
        assert exec_line.index(shared_env) < exec_line.index(pool_env), name
        # 풀 키를 받고도 캐시를 못 보면 매 실행 재발급이 되어 1일1토큰 불변식이 깨진다
        assert mount in exec_line, name

    retrain_text = (root / "kca-retrain.service").read_text(encoding="utf-8")
    assert pool_env not in retrain_text
    assert mount not in retrain_text


def test_daily_audit_unit_loads_key_pool_env_for_token_coverage() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-daily-audit.service").read_text(encoding="utf-8")

    # 풀 env가 없으면 list_stale_kis_tokens가 키를 하나도 해석하지 못해
    # 토큰 커버리지 감사가 조용히 무의미해진다
    assert "EnvironmentFile=%h/quant-secrets/k-closing-alpha.env" in text
    assert "EnvironmentFile=%h/quant-secrets/kis-data.env" in text
    assert "src.tools.daily_audit" in text


def test_containerized_units_preserve_data_and_artifacts_mounts() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    containerized = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
    )
    for name in containerized:
        text = (root / name).read_text(encoding="utf-8")
        assert "-v %h/k-closing-alpha/data:/app/data" in text, name
        assert "-v %h/k-closing-alpha/artifacts:/app/artifacts" in text, name


def test_containerized_units_have_no_docker_pull_before_run() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    offenders = [
        p.name
        for p in sorted(root.glob("kca-*.service"))
        if "docker pull" in p.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_containerized_units_bound_json_file_logs() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    checked = 0
    for svc in sorted(root.glob("kca-*.service")):
        lines = svc.read_text(encoding="utf-8").splitlines()
        exec_lines = [line for line in lines if line.startswith("ExecStart=") and "docker run --rm" in line]
        for exec_line in exec_lines:
            assert "--log-opt max-size=10m" in exec_line, svc.name
            assert "--log-opt max-file=3" in exec_line, svc.name
            checked += 1

    assert checked > 0


def test_containerized_units_have_no_unmeasured_resource_caps() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    containerized = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
    )
    for name in containerized:
        text = (root / name).read_text(encoding="utf-8")
        assert "--memory=" not in text, name
        assert "--cpus=" not in text, name

    retrain_text = (root / "kca-retrain.service").read_text(encoding="utf-8")
    assert "--memory=6g" in retrain_text
    assert "--cpus=1.8" in retrain_text


def test_host_bound_units_remain_bare_metal() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    host_bound = (
        "kca-backup.service",
        "kca-backup-prune.service",
        "kca-daily-audit.service",
        "kca-alert@.service",
    )
    for name in host_bound:
        text = (root / name).read_text(encoding="utf-8")
        assert "docker run" not in text, name
        assert "%h/.local/bin/" in text, name


def test_retrain_service_uses_shared_runtime_env_file_not_legacy_dotenv() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-retrain.service").read_text(encoding="utf-8")

    assert "--env-file %h/quant-secrets/k-closing-alpha.env" in text
    assert "%h/k-closing-alpha/.env" not in text


def test_code_sync_unit_retired_in_favor_of_ci_deploy() -> None:
    """실무 정리: 호스트 pytest 재실행형 code-sync는 CI(deploy.yml)의 단일
    커밋 수렴(uv run python -m src.tools.code_sync --sha)으로 대체됐다.
    운영 시크릿을 물고 호스트에서 전체스위트를 재실행하는 경로가 격리
    결함에 취약해(실측: 2026-09-16~21) 배포를 며칠간 조용히 막았다."""
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    assert not (base / "systemd" / "kca-code-sync.service").exists()
    assert not (base / "systemd" / "kca-code-sync.timer").exists()
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")
    assert "kca-code-sync.timer" not in install_text


def test_after_ordering_preserved_across_containerization() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    finalize_lines = (root / "kca-finalize-close.service").read_text(encoding="utf-8").splitlines()
    after_line = next(line for line in finalize_lines if line.startswith("After="))
    assert "kca-collect.service" in after_line
    assert "kca-predict.service" in after_line
    assert "ExecStopPost=/usr/bin/systemctl --user start --no-block kca-paper-entry.service" in finalize_lines

    audit_after = next(
        line for line in (root / "kca-daily-audit.service").read_text(encoding="utf-8").splitlines()
        if line.startswith("After=")
    )
    assert "kca-archive-intraday.service" in audit_after

    backup_after = next(
        line for line in (root / "kca-backup.service").read_text(encoding="utf-8").splitlines()
        if line.startswith("After=")
    )
    assert "kca-archive-intraday.service" in backup_after
    assert "kca-daily-audit.service" in backup_after
    assert "kca-price-ingest.service" in backup_after

    assert "After=kca-backup.service" in (root / "kca-backup-prune.service").read_text(encoding="utf-8")


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
def test_kis_token_warmup_timer_precedes_first_kis_job() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    timer = (root / "kca-kis-token-warmup.timer").read_text(encoding="utf-8")
    service = (root / "kca-kis-token-warmup.service").read_text(encoding="utf-8")

    assert "OnCalendar=Mon..Fri 07:05:00 Asia/Seoul" in timer
    assert "Unit=kca-kis-token-warmup.service" in timer
    assert "Persistent=true" in timer
    assert "docker run --rm" in service
    assert "src.tools.kis_token_warmup" in service
    assert "-v %h/.cache/kis:/app/.cache/kis" in service
    assert "--env-file %h/quant-secrets/kis-data.env" in service
    assert "OnFailure=kca-alert@%n.service" in service


def test_kis_token_warmup_forwards_shared_key_pool_env_into_container() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    service = (root / "kca-kis-token-warmup.service").read_text(encoding="utf-8")
    exec_line = next(line for line in service.splitlines() if line.startswith("ExecStart=") and "docker run" in line)

    # Then: 키풀 원본 자격증명(kis-data.env)은 systemd 드롭인이 아니라 docker run 자체에
    # 명시돼야 컨테이너 프로세스로 전달된다(드롭인의 EnvironmentFile은 host 'docker'
    # 클라이언트 프로세스에만 적용되고 컨테이너 내부로 자동 전파되지 않는다).
    assert "--env-file %h/quant-secrets/kis-data.env" in exec_line
    assert exec_line.index("--env-file %h/quant-secrets/k-closing-alpha.env") < exec_line.index(
        "--env-file %h/quant-secrets/kis-data.env"
    )
    assert "Type=oneshot" in service


def test_decision_window_services_pin_decision_data_role() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    decision = {
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
    }

    for svc in sorted(root.glob("kca-*.service")):
        text = svc.read_text(encoding="utf-8")
        assert ("Environment=KIS_DATA_ROLE=decision" in text) == (svc.name in decision), svc.name


def test_every_kca_service_loads_shared_runtime_env_file() -> None:
    from pathlib import Path

    services = sorted(Path("deploy/systemd").glob("kca-*.service"))
    assert services, "no kca service units found"

    expected = "EnvironmentFile=%h/quant-secrets/k-closing-alpha.env"
    offenders = [p.name for p in services if expected not in p.read_text(encoding="utf-8")]
    assert offenders == [], offenders

    optional = [
        p.name
        for p in services
        if "EnvironmentFile=-" in p.read_text(encoding="utf-8")
    ]
    assert optional == [], optional

    hardcoded = [
        p.name for p in services if "/home/ubuntu" in p.read_text(encoding="utf-8")
    ]
    assert hardcoded == [], hardcoded


def test_daily_audit_and_backup_normalize_ownership_before_reading_container_writes() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    expected_chown_line = {
        # daily_audit는 data/artifacts(파케이 산출물)뿐 아니라 list_stale_kis_tokens가
        # 읽는 ~/.cache/kis 토큰 캐시도 컨테이너가 root로 남기므로 함께 정규화해야 한다.
        # systemctl --user list-units(호스트 systemd 세션) 조회가 필요해 Docker화할 수
        # 없다 -- 실측: 컨테이너 안엔 systemctl이 없어 실패유닛 점검이
        # "<systemctl unavailable: FileNotFoundError>"로 조용히 무력화됐다.
        "kca-daily-audit.service": (
            "ExecStartPre=/usr/bin/sudo /usr/bin/chown -R ubuntu:ubuntu "
            "%h/k-closing-alpha/data %h/k-closing-alpha/artifacts %h/.cache/kis"
        ),
        "kca-backup.service": (
            "ExecStartPre=/usr/bin/sudo /usr/bin/chown -R ubuntu:ubuntu "
            "%h/k-closing-alpha/data %h/k-closing-alpha/artifacts"
        ),
    }

    for name, chown_line in expected_chown_line.items():
        lines = (root / name).read_text(encoding="utf-8").splitlines()
        assert chown_line in lines, name
        chown_idx = lines.index(chown_line)
        exec_start_idx = next(i for i, line in enumerate(lines) if line.startswith("ExecStart="))
        assert chown_idx < exec_start_idx, name

    # 컨테이너 유닛 자체는 이미 root로 쓰는 쪽이므로 이 정규화 훅이 필요 없다
    collect_text = (root / "kca-collect.service").read_text(encoding="utf-8")
    assert "ExecStartPre=/usr/bin/sudo /usr/bin/chown" not in collect_text


def test_backup_prune_timer_runs_every_weekday() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-backup-prune.timer").read_text(encoding="utf-8")

    assert "OnCalendar=Mon..Fri 21:00:00 Asia/Seoul" in text
    assert "*-*-01" not in text
    assert "Persistent=true" in text


def test_paper_entry_orders_after_finalize() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-paper-entry.service").read_text(encoding="utf-8").splitlines()

    # Then: 15:34 백스톱이 아직 도는 finalize를 기다린다(양쪽 발화의 15:45 재기동 경합은 원장 락이 제거)
    assert "After=kca-finalize-close.service" in lines


def test_paper_exit_timeout_covers_max_wait_plus_quote_budget() -> None:
    import pathlib
    import re

    from src.config.market_session import PAPER_EXIT_MAX_PRESTART_WAIT_SECONDS
    from src.daily.paper_trade import (
        PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS,
        PAPER_EXIT_OPEN_QUOTE_RETRY_SECONDS,
    )

    lines = (
        pathlib.Path(__file__).resolve().parents[3]
        / "deploy"
        / "systemd"
        / "kca-paper-exit.service"
    ).read_text(encoding="utf-8").splitlines()
    match = next(re.search(r"TimeoutStartSec=(\d+)min", line) for line in lines if "TimeoutStartSec=" in line)
    timeout_seconds = int(match.group(1)) * 60

    # Then: 최대 사전대기 + 재조회 간격 + 시도당 1회 조회+차트 왕복(관대한 30초 가정)도 15분 안에 든다
    per_attempt_round_seconds = 30.0
    budget = (
        PAPER_EXIT_MAX_PRESTART_WAIT_SECONDS
        + PAPER_EXIT_OPEN_QUOTE_RETRY_SECONDS * (PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS - 1)
        + per_attempt_round_seconds * PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS
    )
    assert budget < timeout_seconds


def test_archive_ready_constants_match_timer_schedules() -> None:
    import pathlib
    import re

    from src.config.market_session import (
        ARCHIVE_AFTERMARKET_READY_HHMMSS,
        ARCHIVE_REGULAR_READY_HHMMSS,
    )

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    def _oncalendar_hhmmss(name: str) -> str:
        text = (root / name).read_text(encoding="utf-8")
        match = re.search(r"OnCalendar=.*?(\d{2}):(\d{2}):(\d{2})", text)
        assert match is not None, name
        assert "Persistent=true" in text.splitlines(), name
        return "".join(match.groups())

    assert _oncalendar_hhmmss("kca-archive-intraday-regular.timer") == ARCHIVE_REGULAR_READY_HHMMSS
    assert _oncalendar_hhmmss("kca-archive-intraday.timer") == ARCHIVE_AFTERMARKET_READY_HHMMSS


def test_warmup_unit_retries_without_per_attempt_alerts() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-kis-token-warmup.service").read_text(encoding="utf-8")
    lines = text.splitlines()

    assert "Restart=on-failure" in lines
    assert "RestartMode=direct" in lines
    assert "RestartSec=5min" in lines
    assert "StartLimitBurst=4" in lines
    assert "StartLimitIntervalSec=2h" in lines
    assert "Type=oneshot" in lines
    assert "-v %h/.cache/kis:/app/.cache/kis" in text
    assert "OnFailure=kca-alert@%n.service" in text


def test_warmup_retry_budget_finishes_before_first_consumer() -> None:
    import pathlib
    import re
    from datetime import datetime

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    service = (root / "kca-kis-token-warmup.service").read_text(encoding="utf-8")
    timer = (root / "kca-kis-token-warmup.timer").read_text(encoding="utf-8")

    burst = int(re.search(r"StartLimitBurst=(\d+)", service).group(1))  # type: ignore[union-attr]
    restart_min = int(re.search(r"RestartSec=(\d+)min", service).group(1))  # type: ignore[union-attr]
    attempt_min = _parse_systemd_duration(
        next(line for line in service.splitlines() if line.startswith("TimeoutStartSec=")).split("=", 1)[1]
    ) // 60
    start = datetime.strptime(re.search(r"OnCalendar=\S+ (\d{2}:\d{2}:\d{2})", timer).group(1), "%H:%M:%S")  # type: ignore[union-attr]

    # 매 시도는 TimeoutStartSec 안에 끝나거나 실패하므로 전체 예산은 시도 상한과 재시도 간격의 합이다
    end_minute = start.hour * 60 + start.minute + burst * attempt_min + (burst - 1) * restart_min
    assert end_minute < 8 * 60


def test_core_snapshot_unit_holds_shared_drive_lock_and_alerts() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    service = (base / "systemd" / "kca-core-snapshot.service").read_text(encoding="utf-8")
    timer = (base / "systemd" / "kca-core-snapshot.timer").read_text(encoding="utf-8")
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    assert "/usr/bin/flock -w 7200 %t/quant-gdrive.lock" in service
    assert "src.tools.core_snapshot" in service
    assert "OnFailure=kca-alert@%n.service" in service
    assert "Persistent=true" in timer
    assert "OnCalendar=Sun 10:00:00 Asia/Seoul" in timer
    assert "kca-core-snapshot.timer" in install_text


def test_containers_run_as_host_user_with_tmp_caches() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    services = [p for p in sorted(root.glob("kca-*.service")) if "docker run --rm" in p.read_text(encoding="utf-8")]
    assert services
    for svc in services:
        text = svc.read_text(encoding="utf-8")
        exec_line = next(line for line in text.splitlines() if line.startswith("ExecStart=") and "docker run" in line)
        assert "--user %U:%G" in exec_line, svc.name
        assert "-e HOME=/tmp" in exec_line, svc.name
        assert "-e UV_CACHE_DIR=/tmp/uv-cache" in exec_line, svc.name


def test_kis_cache_mount_follows_non_root_home() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    needs_kis_cache = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
        "kca-auction-open.service",
        "kca-auction-close.service",
        "kca-altdata-capture.service",
        "kca-tape-sweep.service",
    )
    for name in needs_kis_cache:
        text = (root / name).read_text(encoding="utf-8")
        assert "-v %h/.cache/kis:/app/.cache/kis" in text, name
        assert "-e KIS_TOKEN_CACHE_DIR=/app/.cache/kis" in text, name
    for svc in sorted(root.glob("kca-*.service")):
        assert "/root/.cache/kis" not in svc.read_text(encoding="utf-8"), svc.name


def test_timer_descriptions_state_real_schedule() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    for timer in sorted(root.glob("kca-*.timer")):
        text = timer.read_text(encoding="utf-8")
        cals = re.findall(r"OnCalendar=\S+ (\d{2}):(\d{2}):\d{2}", text)
        if len(cals) != 1:
            continue
        desc_match = re.search(r"Description=.*?(\d{2}):(\d{2})", text)
        if desc_match is None:
            continue
        assert (desc_match.group(1), desc_match.group(2)) == (cals[0][0], cals[0][1]), timer.name


def test_altdata_timer_fires_on_weekdays_only() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-altdata-capture.timer").read_text(encoding="utf-8").splitlines()
    assert "OnCalendar=Mon..Fri 21:35:00 Asia/Seoul" in lines
    assert "Persistent=false" in lines


def _parse_systemd_duration(raw: str) -> int:
    """Parse a systemd TimeoutStartSec value into seconds (bare numbers are seconds)."""
    import re

    value = raw.strip()
    if value.isdigit():
        return int(value)
    if re.fullmatch(r"(?:\d+(?:h|min|s))+", value) is None:
        raise ValueError(f"non-finite or unsupported duration: {value!r}")
    total = 0
    for amount, unit in re.findall(r"(\d+)(h|min|s)", value):
        total += int(amount) * {"h": 3600, "min": 60, "s": 1}[unit]
    if total <= 0:
        raise ValueError(f"non-finite or unsupported duration: {value!r}")
    return total


def _service_timeout_seconds(root, name: str) -> int:
    text = (root / name).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("TimeoutStartSec=")]
    assert len(lines) == 1, name
    return _parse_systemd_duration(lines[0].split("=", 1)[1])


def _timer_first_seconds(root, name: str) -> int:
    import re

    text = (root / name).read_text(encoding="utf-8")
    match = re.search(r"OnCalendar=.*?(\d{2}):(\d{2}):(\d{2})", text)
    assert match is not None, name
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def test_every_kca_service_has_finite_start_timeout() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    services = sorted(root.glob("kca-*.service"))

    assert services
    for svc in services:
        text = svc.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.startswith("TimeoutStartSec=")]

        assert len(lines) == 1, svc.name
        assert _parse_systemd_duration(lines[0].split("=", 1)[1]) > 0, svc.name


def test_decision_chain_timeouts_end_before_next_stage() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    collect_end = _timer_first_seconds(root, "kca-collect.timer") + _service_timeout_seconds(root, "kca-collect.service")
    assert collect_end < _timer_first_seconds(root, "kca-finalize-close.timer")

    predict_end = _timer_first_seconds(root, "kca-predict.timer") + _service_timeout_seconds(root, "kca-predict.service")
    assert predict_end < 15 * 3600 + 30 * 60


def test_archive_regular_timeout_ends_before_aftermarket_archive() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    regular_end = _timer_first_seconds(root, "kca-archive-intraday-regular.timer") + _service_timeout_seconds(
        root, "kca-archive-intraday-regular.service"
    )

    assert regular_end <= _timer_first_seconds(root, "kca-archive-intraday.timer")


def test_containerized_units_have_unique_names_and_reaper() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    names: dict[str, str] = {}
    checked = 0
    for svc in sorted(root.glob("kca-*.service")):
        lines = svc.read_text(encoding="utf-8").splitlines()
        exec_lines = [line for line in lines if line.startswith("ExecStart=") and "docker run" in line]
        if not exec_lines:
            continue
        stem = svc.name.removeprefix("kca-").removesuffix(".service")
        match = re.search(r"--name (\S+)", exec_lines[0])

        assert match is not None, svc.name
        assert match.group(1) == f"kca-{stem}", svc.name
        assert f"ExecStopPost=-/usr/bin/docker rm -f kca-{stem}" in lines, svc.name
        assert match.group(1) not in names, svc.name
        names[match.group(1)] = svc.name
        checked += 1

    assert checked > 0


def test_finalize_close_reaper_precedes_paper_entry_handoff() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-finalize-close.service").read_text(encoding="utf-8").splitlines()

    reaper_idx = lines.index("ExecStopPost=-/usr/bin/docker rm -f kca-finalize-close")
    handoff_idx = lines.index("ExecStopPost=/usr/bin/systemctl --user start --no-block kca-paper-entry.service")

    assert reaper_idx < handoff_idx


def test_budget_lines_are_commented() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    budgeted = (
        "kca-collect.service",
        "kca-predict.service",
        "kca-auction-close.service",
        "kca-finalize-close.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-auction-open.service",
        "kca-archive-intraday-regular.service",
        "kca-archive-intraday.service",
        "kca-price-ingest.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-daily-audit.service",
        "kca-retrain.service",
        "kca-altdata-capture.service",
        "kca-backup.service",
        "kca-backup-prune.service",
        "kca-core-snapshot.service",
        "kca-extended-backfill.service",
        "kca-aftermarket-book.service",
        "kca-tape-sweep.service",
    )
    for name in budgeted:
        lines = (root / name).read_text(encoding="utf-8").splitlines()
        idx = next(i for i, line in enumerate(lines) if line.startswith("TimeoutStartSec="))

        assert lines[idx - 1].startswith("#"), name


def test_offsite_verify_unit_holds_drive_lock_and_alerts() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-offsite-verify.service").read_text(encoding="utf-8")

    assert "/usr/bin/flock -w 7200 %t/quant-gdrive.lock" in text
    assert "src.tools.capture_offsite verify" in text
    assert "OnFailure=kca-alert@%n.service" in text
    assert "TimeoutStartSec=2h" in text
    assert "docker run" not in text


def test_offsite_verify_timer_runs_after_core_snapshot() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"

    def _sunday_seconds(name: str) -> int:
        text = (root / name).read_text(encoding="utf-8")
        assert "OnCalendar=Sun" in text, name
        match = re.search(r"OnCalendar=.*?(\d{2}):(\d{2}):(\d{2})", text)
        assert match is not None, name
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))

    timer = (root / "kca-offsite-verify.timer").read_text(encoding="utf-8")
    assert "OnCalendar=Sun 11:00:00 Asia/Seoul" in timer
    assert "Persistent=true" in timer
    assert "Unit=kca-offsite-verify.service" in timer
    assert _sunday_seconds("kca-offsite-verify.timer") > _sunday_seconds("kca-core-snapshot.timer")

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    assert "kca-offsite-verify.timer" in (base / "install_systemd.sh").read_text(encoding="utf-8")


def test_aftermarket_book_unit_is_bounded_alerting_and_ordered() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    root = base / "systemd"
    service = (root / "kca-aftermarket-book.service").read_text(encoding="utf-8")
    timer = (root / "kca-aftermarket-book.timer").read_text(encoding="utf-8")
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    # Then: 장시간 저녁 수집에 맞는 상한, 실패 알림, 15:40 KST 1초 정밀 발화
    assert "TimeoutStartSec=4h40min" in service.splitlines()
    assert "OnFailure=kca-alert@%n.service" in service
    assert "OnCalendar=Mon..Fri 15:40:00 Asia/Seoul" in timer.splitlines()
    assert "AccuracySec=1s" in timer.splitlines()
    assert "Persistent=false" in timer.splitlines()

    # And: 감사·백업이 저녁 수집을 기다리고 설치 스크립트가 타이머를 활성화
    for name in ("kca-daily-audit.service", "kca-backup.service"):
        after_line = next(
            line for line in (root / name).read_text(encoding="utf-8").splitlines() if line.startswith("After=")
        )
        assert "kca-aftermarket-book.service" in after_line, name
    assert "kca-aftermarket-book.timer" in install_text


def test_aftermarket_book_service_keeps_shared_container_contract() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    text = (root / "kca-aftermarket-book.service").read_text(encoding="utf-8")
    exec_line = next(line for line in text.splitlines() if line.startswith("ExecStart=") and "docker run" in line)

    # Then: 공유 이미지·env 파일·KIS 캐시·데이터/아티팩트 마운트, 리소스 상한 없음
    assert "ghcr.io/kthyeong/k-closing-alpha:latest" in exec_line
    assert "--env-file %h/quant-secrets/k-closing-alpha.env" in exec_line
    assert "--env-file %h/quant-secrets/kis-data.env" in exec_line
    assert "-v %h/.cache/kis:/app/.cache/kis" in exec_line
    assert "-v %h/k-closing-alpha/data:/app/data" in exec_line
    assert "-v %h/k-closing-alpha/artifacts:/app/artifacts" in exec_line
    assert "src.daily.aftermarket_book" in exec_line
    assert "--memory=" not in text
    assert "--cpus=" not in text


def test_tape_sweep_unit_runs_after_evening_archive_outside_archive_slots() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    root = base / "systemd"
    service = (root / "kca-tape-sweep.service").read_text(encoding="utf-8")
    timer = (root / "kca-tape-sweep.timer").read_text(encoding="utf-8")
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    # Then: 20:05 아카이브·23:05 백필 슬롯 밖인 20:35 발화, 1시간 상한, 저녁 아카이브 이후 순서
    assert "OnCalendar=Mon..Fri 20:35:00 Asia/Seoul" in timer.splitlines()
    assert "Persistent=true" in timer.splitlines()
    assert "Unit=kca-tape-sweep.service" in timer
    assert "TimeoutStartSec=3600" in service.splitlines()
    assert "Type=oneshot" in service
    assert "OnFailure=kca-alert@%n.service" in service
    assert "After=kca-archive-intraday.service" in service
    assert "src.daily.tick_tape_sweep" in service

    # And: Kiwoom-only 작업이라 KIS 키풀 env를 물지 않는다 (admission 공유 마운트는 유지)
    assert "--env-file %h/quant-secrets/kis-data.env" not in service
    assert "-v %h/.cache/kis:/app/.cache/kis" in service
    assert "-e KIS_TOKEN_CACHE_DIR=/app/.cache/kis" in service
    assert "-e BROKER_ADMISSION_CLASS=bulk" in service

    # And: 설치 스크립트가 타이머를 활성화하고 실패 스캔(kca-*)에 자동 포함된다
    assert "kca-tape-sweep.timer" in install_text


def test_backup_backstop_timeout_fits_shared_lock_window() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-backup.service").read_text(encoding="utf-8").splitlines()
    timeout = next(line for line in lines if line.startswith("TimeoutStartSec=")).split("=", 1)[1]

    # krx-host-backup이 최대 7200초 대기하므로 kca는 3시간 안에 락을 풀어야 한다
    assert _parse_systemd_duration(timeout) <= 3 * 3600


def test_broker_units_mount_shared_admission_dir() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    broker_docker_units = (
        "kca-collect.service",
        "kca-predict.service",
        "kca-auction-open.service",
        "kca-auction-close.service",
        "kca-finalize-close.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-kis-token-warmup.service",
        "kca-kiwoom-token-rotate.service",
        "kca-aftermarket-book.service",
        "kca-archive-intraday-regular.service",
        "kca-archive-intraday.service",
        "kca-price-ingest.service",
        "kca-altdata-capture.service",
        "kca-extended-backfill.service",
        "kca-tape-sweep.service",
    )
    for name in broker_docker_units:
        text = (root / name).read_text(encoding="utf-8")
        exec_line = next(line for line in text.splitlines() if line.startswith("ExecStart=") and "docker run" in line)
        assert "-v %h/.cache/kis:/app/.cache/kis" in exec_line, name
        assert "-e KIS_TOKEN_CACHE_DIR=/app/.cache/kis" in exec_line, name


def test_every_broker_unit_declares_its_admission_class() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    expected = {
        "kca-collect.service": "critical",
        "kca-predict.service": "critical",
        "kca-auction-open.service": "critical",
        "kca-auction-close.service": "critical",
        "kca-finalize-close.service": "critical",
        "kca-paper-entry.service": "critical",
        "kca-paper-exit.service": "critical",
        "kca-kis-token-warmup.service": "critical",
        "kca-kiwoom-token-rotate.service": "critical",
        "kca-aftermarket-book.service": "standard",
        "kca-archive-intraday-regular.service": "standard",
        "kca-archive-intraday.service": "standard",
        "kca-price-ingest.service": "standard",
        "kca-altdata-capture.service": "standard",
        "kca-daily-audit.service": "standard",
        "kca-extended-backfill.service": "bulk",
        "kca-tape-sweep.service": "bulk",
    }
    for name, cls in expected.items():
        text = (root / name).read_text(encoding="utf-8")
        match = re.search(r"BROKER_ADMISSION_CLASS=([A-Za-z]+)", text)
        assert match is not None, name
        assert match.group(1) == cls, name


def test_non_broker_units_stay_unmounted() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    for name in ("kca-retrain.service", "kca-backup.service", "kca-backup-prune.service", "kca-core-snapshot.service", "kca-offsite-verify.service"):
        text = (root / name).read_text(encoding="utf-8")
        assert "-v %h/.cache/kis:/app/.cache/kis" not in text, name
        assert "BROKER_ADMISSION_CLASS=" not in text, name
    assert "BROKER_ADMISSION_CLASS=" not in (root / "kca-alert@.service").read_text(encoding="utf-8")


def test_install_script_creates_host_admission_marker() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    text = (base / "install_systemd.sh").read_text(encoding="utf-8")
    assert "${HOME}/.cache/kis/.host-admission" in text
    assert "${HOME}/.cache/kis" in text


def test_kiwoom_token_rotate_unit_moves_expiry_out_of_decision_window() -> None:
    import pathlib

    base = pathlib.Path(__file__).resolve().parents[3] / "deploy"
    root = base / "systemd"
    timer = (root / "kca-kiwoom-token-rotate.timer").read_text(encoding="utf-8")
    service = (root / "kca-kiwoom-token-rotate.service").read_text(encoding="utf-8")
    install_text = (base / "install_systemd.sh").read_text(encoding="utf-8")

    assert "OnCalendar=Mon..Fri 07:10:00 Asia/Seoul" in timer
    assert "Persistent=true" in timer
    assert "AccuracySec=1s" in timer
    assert "Unit=kca-kiwoom-token-rotate.service" in timer
    assert "src.tools.kiwoom_token_rotate" in service
    assert "-v %h/.cache/kis:/app/.cache/kis" in service
    assert "-e BROKER_ADMISSION_CLASS=critical" in service
    assert "OnFailure=kca-alert@%n.service" in service
    assert "kca-kiwoom-token-rotate.timer" in install_text


def test_p5_reconcile_unit_tolerates_busy_lock() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    service = (root / "kca-audit-reconcile.service").read_text(encoding="utf-8")
    timer = (root / "kca-audit-reconcile.timer").read_text(encoding="utf-8")

    assert "flock -w 30 -E 75 %t/kca-audit.lock" in service
    assert "SuccessExitStatus=75" in service.splitlines()

    def _svc_seconds() -> int:
        line = next(line for line in service.splitlines() if line.startswith("TimeoutStartSec="))
        return _parse_systemd_duration(line.split("=", 1)[1])

    assert _svc_seconds() == 9 * 60

    cals = [line for line in timer.splitlines() if line.startswith("OnCalendar=")]
    assert len(cals) > 2
    assert "Persistent=true" not in timer.splitlines()
    assert "OnFailure=kca-alert@%n.service" in service


def test_p5_all_repository_timers_parse_for_sentinel() -> None:
    import pathlib

    from src.tools.ops_sentinel import load_job_schedules

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    schedules = load_job_schedules(root)
    by_unit = {s.unit: s for s in schedules}
    assert "kca-audit-reconcile.service" in by_unit
    assert len(by_unit["kca-audit-reconcile.service"].slots) > 2


def test_p5_reconcile_tick_chowns_only_its_paths() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    lines = (root / "kca-audit-reconcile.service").read_text(encoding="utf-8").splitlines()
    chown = next(line for line in lines if "/usr/bin/chown" in line)
    targets = chown.split("ubuntu:ubuntu", 1)[1].split()
    assert "%h/k-closing-alpha/data" not in targets
    assert all(t.startswith(("%h/k-closing-alpha/data/", "%h/.cache/kis")) for t in targets), targets
