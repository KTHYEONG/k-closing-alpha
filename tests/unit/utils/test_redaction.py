from __future__ import annotations

import pytest

from src.utils.redaction import REDACTION_MASK, redact_secrets


def test_redact_secrets_exact_configured_value_masked_anywhere() -> None:
    secret = "S3cr3t+/Value=="
    out = redact_secrets(f"401 for url: https://x/api?id=1&q={secret}", {secret})
    assert secret not in out
    assert REDACTION_MASK in out
    assert "https://x/api?id=1" in out


def test_redact_secrets_longest_secret_masked_whole() -> None:
    out = redact_secrets("err ABCDEFGHIJKLMNOP end", {"ABCDEFGH", "ABCDEFGHIJKLMNOP"})
    assert "IJKLMNOP" not in out
    assert "ABCDEFGH" not in out


def test_redact_secrets_short_and_empty_values_ignored() -> None:
    assert redact_secrets("abc def", {"", "   ", "abc"}) == "abc def"


@pytest.mark.parametrize(
    ("line", "value", "kept"),
    [
        ("appkey=AAA111", "AAA111", "appkey"),
        ("APP_SECRET: BBB222", "BBB222", "APP_SECRET"),
        ('{"appSecret": "CCC333"}', "CCC333", "appSecret"),
        ("{'secretkey': 'DDD444'}", "DDD444", "secretkey"),
        ("AUTH_KEY=EEE555", "EEE555", "AUTH_KEY"),
        ("access_token=FFF666", "FFF666", "access_token"),
        ("?crtfc_key=GGG777&page=1", "GGG777", "&page=1"),
        ("KIS_DATA_1_APP_SECRET=HHH888", "HHH888", "KIS_DATA_1_APP_SECRET"),
        ("OPENDART_API_KEY_2=III999", "III999", "OPENDART_API_KEY_2"),
    ],
)
def test_redact_secrets_key_value_shapes_masked_case_insensitive(line: str, value: str, kept: str) -> None:
    out = redact_secrets(line)
    assert value not in out
    assert kept in out


@pytest.mark.parametrize(
    ("text", "value", "kept"),
    [
        (
            "requests.exceptions.HTTPError: 401 for url: "
            "https://opendart.fss.or.kr/api/list.json?crtfc_key=ABCDEF0123456789&page_no=1",
            "ABCDEF0123456789",
            "page_no=1",
        ),
        ('metrics={"err": "appkey=ABCDEFGH12345"}', "ABCDEFGH12345", '"err"'),
        ("error='upstream ?crtfc_key=KEYVALUE99 failed'", "KEYVALUE99", "error="),
        ("GET https://data.krx.co.kr/svc?AUTH_KEY=KRXSECRET77 ok", "KRXSECRET77", "https://data.krx.co.kr/svc?"),
    ],
)
def test_redact_secrets_credential_nested_in_non_credential_value_masked(text: str, value: str, kept: str) -> None:
    out = redact_secrets(text)
    assert value not in out
    assert kept in out


def test_redact_secrets_quoted_value_with_escaped_quote_masked_whole() -> None:
    out = redact_secrets('{"appSecret": "abc\\"defTAILSECRET"}')
    assert "TAILSECRET" not in out
    assert "abc" not in out


def test_redact_secrets_key_at_line_end_does_not_consume_next_line() -> None:
    text = "token:\nnext line stays"
    assert redact_secrets(text) == text


def test_redact_secrets_authorization_bearer_masked_with_scheme_kept() -> None:
    out = redact_secrets("authorization: Bearer eyJhbGciOi.payload.sig\nheader Bearer eyJxyz")
    assert "eyJhbGciOi" not in out
    assert "eyJxyz" not in out
    assert out.count("Bearer ***") == 2
    assert "authorization" in out


def test_redact_secrets_non_credential_keys_untouched() -> None:
    text = "key_id=abc123def456 app_key_var=KIS_APP_KEY token_file=/tmp/t.json stale_kis_tokens=DATA_1 at=15:20:00"
    assert redact_secrets(text) == text


def test_redact_secrets_is_idempotent_and_line_preserving() -> None:
    text = (
        "line1 appkey=AAA111\n"
        "url: https://x/?crtfc_key=ZZZ999&a=1\n"
        "Authorization: Bearer tok.abc.def\n"
        "plain text configured-secret-0001"
    )
    once = redact_secrets(text, {"configured-secret-0001"})
    assert redact_secrets(once, {"configured-secret-0001"}) == once
    assert once.count("\n") == text.count("\n")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"Authorization": "Bearer tok.abc.def"}', '{"Authorization": "Bearer ***"}'),
        ("{'authorization': 'bearer tok.abc.def'}", "{'authorization': 'bearer ***'}"),
        ("authorization=rawtoken123", "authorization=***"),
        ('{"authorization": "Basic dXNlcjpwYXNz"}', '{"authorization": "***"}'),
        ("Authorization: Bearer ***", "Authorization: Bearer ***"),
    ],
)
def test_redact_secrets_authorization_value_shapes(text: str, expected: str) -> None:
    assert redact_secrets(text) == expected


def test_redact_secrets_empty_text_returned_unchanged() -> None:
    assert redact_secrets("", {"configured-secret-0001"}) == ""


def test_redact_secrets_bearer_at_line_end_does_not_consume_next_line() -> None:
    text = "scheme Bearer\nnext line stays"
    assert redact_secrets(text) == text


@pytest.mark.parametrize(
    ("text", "value", "kept"),
    [
        ('"body": "{\\"appsecret\\": \\"NESTEDSECRET123\\"}"', "NESTEDSECRET123", '\\"appsecret\\"'),
        ('{\\"appkey\\":\\"K1234567\\",\\"x\\":1}', "K1234567", '\\"x\\":1'),
    ],
)
def test_redact_secrets_escaped_json_pairs_masked(text: str, value: str, kept: str) -> None:
    out = redact_secrets(text)
    assert value not in out
    assert kept in out
    assert redact_secrets(out) == out


def test_redact_secrets_escaped_json_non_credential_untouched() -> None:
    text = '"body": "{\\"msg\\": \\"ok\\"}"'
    assert redact_secrets(text) == text


def test_redact_secrets_unquoted_value_with_backslash_masked_whole() -> None:
    out = redact_secrets("appsecret=ABC\\DEFTAILSECRET end")
    assert "TAILSECRET" not in out
    assert "end" in out
