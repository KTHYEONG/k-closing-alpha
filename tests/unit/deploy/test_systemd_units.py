


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
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
        "kca-altdata-capture.service",
        "kca-auction-close.service",
        "kca-auction-open.service",
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
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
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
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
    )
    mount = "-v %h/.cache/kis:/root/.cache/kis"
    for name in needs_kis_cache:
        assert mount in (root / name).read_text(encoding="utf-8"), name

    assert mount not in (root / "kca-retrain.service").read_text(encoding="utf-8")


def test_kis_using_containerized_units_forward_key_pool_env_and_cache() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    shared_env = "--env-file %h/quant-secrets/k-closing-alpha.env"
    pool_env = "--env-file %h/quant-secrets/kis-data.env"
    mount = "-v %h/.cache/kis:/root/.cache/kis"
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
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
    )
    for name in containerized:
        text = (root / name).read_text(encoding="utf-8")
        assert "-v %h/k-closing-alpha/data:/app/data" in text, name
        assert "-v %h/k-closing-alpha/artifacts:/app/artifacts" in text, name


def test_containerized_units_have_no_docker_pull_before_run() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    containerized = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
    )
    for name in containerized:
        assert "docker pull" not in (root / name).read_text(encoding="utf-8"), name

    assert "docker pull" in (root / "kca-retrain.service").read_text(encoding="utf-8")


def test_containerized_units_have_no_unmeasured_resource_caps() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    containerized = (
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-collect.service",
        "kca-finalize-close.service",
        "kca-kis-token-warmup.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-predict.service",
        "kca-price-ingest.service",
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
    assert "-v %h/.cache/kis:/root/.cache/kis" in service
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
