"""Pure auth-rejection classification shared by every broker client and the order verdict path."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

BrokerVendor = Literal["kis", "ls", "dbsec", "toss", "kiwoom"]

AUTH_REJECTION_CODES: Final[Mapping[BrokerVendor, frozenset[str]]] = MappingProxyType(
    {
        "kis": frozenset({"EGW00121", "EGW00123"}),
        "ls": frozenset({"IGW00101", "IGW00102", "IGW00123"}),
        "dbsec": frozenset({"IGW00121", "IGW00123"}),
        "toss": frozenset({"invalid-token"}),
        "kiwoom": frozenset({"8005"}),
    }
)

KIWOOM_AUTH_EXPIRED_RETURN_CODE: Final[int] = 3


def is_auth_rejected(vendor: BrokerVendor, status: int, body: object) -> bool:
    """Return True when (status, decoded body) is the vendor's token rejection; never raises."""
    codes = AUTH_REJECTION_CODES.get(vendor)
    if codes is None:
        return False
    if vendor == "kis":
        return isinstance(body, dict) and str(body.get("msg_cd", "")) in codes
    if vendor == "kiwoom":
        if not isinstance(body, dict) or body.get("return_code") != KIWOOM_AUTH_EXPIRED_RETURN_CODE:
            return False
        return_msg = str(body.get("return_msg", ""))
        return any(code in return_msg for code in codes)
    if status == 401:
        return True
    if not isinstance(body, dict):
        return False
    if vendor == "toss":
        error = body.get("error")
        return isinstance(error, dict) and str(error.get("code", "")) in codes
    return str(body.get("rsp_cd", "")) in codes


def is_kis_data_auth_rejected(body: object) -> bool:
    """Return True for the KIS data-client lenient rejection (codes or token text in msg1); never raises."""
    if not isinstance(body, dict) or body.get("rt_cd") == "0":
        return False
    if str(body.get("msg_cd", "")) in AUTH_REJECTION_CODES["kis"]:
        return True
    msg1 = str(body.get("msg1", ""))
    return "token" in msg1.lower() or "토큰" in msg1


__all__ = [
    "AUTH_REJECTION_CODES",
    "KIWOOM_AUTH_EXPIRED_RETURN_CODE",
    "BrokerVendor",
    "is_auth_rejected",
    "is_kis_data_auth_rejected",
]
