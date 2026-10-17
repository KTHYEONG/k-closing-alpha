"""Invariant guards for the pure broker auth-rejection registry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from src.api.auth_rejection import (
    AUTH_REJECTION_CODES,
    KIWOOM_AUTH_EXPIRED_RETURN_CODE,
    is_auth_rejected,
    is_kis_data_auth_rejected,
)

_VENDORS = ["kis", "ls", "dbsec", "toss", "kiwoom"]


def _legacy_is_auth_rejected(vendor: str, status: int, body: dict[str, Any]) -> bool:
    """Predicate recorded verbatim from HEAD 654fe1b, used as the parity oracle."""
    if vendor == "kis":
        return str(body.get("msg_cd", "")) in {"EGW00121", "EGW00123"}
    if vendor == "ls":
        if status == 401:
            return True
        return str(body.get("rsp_cd", "")) in {"IGW00101", "IGW00102", "IGW00123"}
    if vendor == "dbsec":
        if status == 401:
            return True
        return str(body.get("rsp_cd", "")) in {"IGW00121", "IGW00123"}
    if vendor == "toss":
        if status == 401:
            return True
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("code", "")) in {"invalid-token"}
        return False
    if vendor == "kiwoom":
        return body.get("return_code") == 3 and "8005" in str(body.get("return_msg", ""))
    raise AssertionError(f"unknown vendor {vendor!r}")


def _bodies_for(vendor: str) -> list[dict[str, Any]]:
    if vendor == "kis":
        return [
            {"msg_cd": "EGW00121"},
            {"msg_cd": "EGW00123"},
            {"msg_cd": "EGW00201"},
            {"msg_cd": "OTHER"},
            {"msg_cd": None},
            {"rt_cd": "1"},
            {"msg_cd": "EGW00123", "rt_cd": "1"},
            {},
        ]
    if vendor in ("ls", "dbsec"):
        codes = ["IGW00101", "IGW00102", "IGW00123"] if vendor == "ls" else ["IGW00121", "IGW00123"]
        return (
            [{"rsp_cd": code} for code in codes]
            + [{"rsp_cd": "IGW00201"}, {"rsp_cd": "OTHER"}, {"rsp_cd": None}, {"msg_cd": "EGW00123"}, {}]
        )
    if vendor == "toss":
        return [
            {"error": {"code": "invalid-token"}},
            {"error": {"code": "other"}},
            {"error": {"code": None}},
            {"error": "invalid-token"},
            {"error": {"code": "invalid-token"}, "rsp_cd": "IGW00123"},
            {"error": {"other": "x"}},
            {},
        ]
    return [
        {"return_code": KIWOOM_AUTH_EXPIRED_RETURN_CODE, "return_msg": "token expired 8005"},
        {"return_code": KIWOOM_AUTH_EXPIRED_RETURN_CODE, "return_msg": "no code here"},
        {"return_code": "3", "return_msg": "8005"},
        {"return_code": 0, "return_msg": "8005"},
        {"return_code": KIWOOM_AUTH_EXPIRED_RETURN_CODE},
        {"return_msg": "8005"},
        {},
    ]


def test_registry_codes_exact() -> None:
    assert dict(AUTH_REJECTION_CODES) == {
        "kis": frozenset({"EGW00121", "EGW00123"}),
        "ls": frozenset({"IGW00101", "IGW00102", "IGW00123"}),
        "dbsec": frozenset({"IGW00121", "IGW00123"}),
        "toss": frozenset({"invalid-token"}),
        "kiwoom": frozenset({"8005"}),
    }
    assert isinstance(AUTH_REJECTION_CODES, MappingProxyType)
    with pytest.raises(TypeError):
        AUTH_REJECTION_CODES["kis"] = frozenset()  # type: ignore[index]


def test_parity_with_legacy_predicates() -> None:
    for vendor in _VENDORS:
        for status in (200, 401, 403):
            for body in _bodies_for(vendor):
                assert is_auth_rejected(vendor, status, body) is _legacy_is_auth_rejected(
                    vendor, status, body
                ), (vendor, status, body)


def test_kiwoom_401_not_rejection() -> None:
    assert is_auth_rejected("kiwoom", 401, {"return_code": 0}) is False


def test_kiwoom_compound_rule() -> None:
    assert is_auth_rejected("kiwoom", 200, {"return_code": 3, "return_msg": "expired:8005"}) is True
    assert is_auth_rejected("kiwoom", 200, {"return_code": 3, "return_msg": "x"}) is False
    assert is_auth_rejected("kiwoom", 200, {"return_code": "3", "return_msg": "expired:8005"}) is False


def test_kis_lenient_superset() -> None:
    for code in AUTH_REJECTION_CODES["kis"]:
        body = {"rt_cd": "1", "msg_cd": code}
        assert is_auth_rejected("kis", 200, body) is True
        assert is_kis_data_auth_rejected(body) is True
    assert is_kis_data_auth_rejected({"rt_cd": "1", "msg1": "기간이 만료된 token"}) is True
    assert is_auth_rejected("kis", 200, {"rt_cd": "1", "msg1": "기간이 만료된 token"}) is False
    assert is_kis_data_auth_rejected({"rt_cd": "1", "msg1": "만료된 토큰"}) is True
    assert is_kis_data_auth_rejected({"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "busy"}) is False
    assert is_kis_data_auth_rejected({"rt_cd": "0", "msg_cd": "EGW00123"}) is False


def test_non_dict_never_raises() -> None:
    for body in ([], None, "x"):
        for vendor in _VENDORS:
            assert is_auth_rejected(vendor, 200, body) is False
            assert is_auth_rejected(vendor, 401, body) is (vendor in ("ls", "dbsec", "toss"))
        assert is_kis_data_auth_rejected(body) is False


def test_unknown_vendor_is_not_a_rejection() -> None:
    assert is_auth_rejected("unknown", 401, {}) is False  # type: ignore[arg-type]
