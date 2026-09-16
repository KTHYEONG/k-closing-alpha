def test_alert_settings_default_empty_and_env_override(monkeypatch) -> None:
    from src.config.alerts import AlertSettings

    # Given: 실제 운영 호스트(예: code-sync 유닛의 EnvironmentFile)에 이미 설정된
    # 알림 채널 값이 이 프로세스에 새어들지 않도록 격리한다.
    for var in ("ALERT_WEBHOOK_URL", "ALERT_GMAIL_USER", "ALERT_GMAIL_APP_PASSWORD", "ALERT_GMAIL_TO"):
        monkeypatch.delenv(var, raising=False)

    # Given / When: 환경변수 미설정
    default = AlertSettings(_env_file=None)

    # Then: 모든 채널이 빈 문자열 기본값 -> 미설정이 예외가 아닌 정상 상태
    assert default.ALERT_WEBHOOK_URL == ""
    assert default.ALERT_GMAIL_USER == ""
    assert default.ALERT_GMAIL_APP_PASSWORD == ""
    assert default.ALERT_GMAIL_TO == ""

    # Given: 환경변수 오버라이드
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/x")
    monkeypatch.setenv("ALERT_GMAIL_TO", "ops@example.com")

    # When
    overridden = AlertSettings(_env_file=None)

    # Then
    assert overridden.ALERT_WEBHOOK_URL == "https://hooks.example.com/x"
    assert overridden.ALERT_GMAIL_TO == "ops@example.com"
