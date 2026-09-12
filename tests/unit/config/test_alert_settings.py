def test_alert_settings_default_empty_and_env_override(monkeypatch) -> None:
    from src.config.alerts import AlertSettings

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
