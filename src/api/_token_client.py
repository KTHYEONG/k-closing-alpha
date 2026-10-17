"""Host-shared token lifecycle shared by the LS, Toss and DB Securities REST clients."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, Literal

from src.api._transport import TransportResponse
from src.api.auth_rejection import is_auth_rejected
from src.api.host_broker_contract import IssueThrottlePolicy, TokenIssueThrottled
from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
from src.config import settings

if TYPE_CHECKING:
    import aiohttp

__all__ = ["SharedTokenClientMixin"]


class SharedTokenClientMixin:
    """LS/Toss/DB host-shared token lifecycle bound to one vendor id."""

    _TOKEN_VENDOR: ClassVar[Literal["ls", "toss", "dbsec"]]
    _TOKEN_EXPIRY_MARGIN_SECONDS: ClassVar[float]
    _TOKEN_ISSUE_THROTTLE_POLICY: ClassVar[IssueThrottlePolicy]
    _TOKEN_THROTTLED_ERROR: ClassVar[type[TokenIssueThrottled]] = TokenIssueThrottled
    _TOKEN_THROTTLED_MESSAGE: ClassVar[str | None] = None
    app_key: str | None
    token: str | None

    async def _issue_token(self, session: aiohttp.ClientSession) -> IssuedToken:
        """Issue one vendor token; raise on vendor failure."""
        raise NotImplementedError

    def _token_store(self) -> SharedTokenStore:
        """Return the host-shared store for (vendor, app_key)."""
        return SharedTokenStore(
            shared_token_path(self._TOKEN_VENDOR, self.app_key or ""),
            vendor=self._TOKEN_VENDOR,
            lock_timeout_seconds=float(settings.BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS),
            expiry_margin_seconds=self._TOKEN_EXPIRY_MARGIN_SECONDS,
            issue_throttle_policy=self._TOKEN_ISSUE_THROTTLE_POLICY,
            clock=lambda: datetime.now(UTC),
        )

    async def ensure_token(self, session: aiohttp.ClientSession) -> str:
        """Adopt a usable host token or issue once; raise TokenStoreLockTimeout, TokenIssueThrottled, or issuance errors."""
        try:
            record = await self._token_store().get_or_issue(lambda: self._issue_token(session))
        except TokenIssueThrottled as exc:
            if isinstance(exc, self._TOKEN_THROTTLED_ERROR):
                raise
            raise self._TOKEN_THROTTLED_ERROR(self._TOKEN_THROTTLED_MESSAGE or str(exc)) from exc
        self.token = record.access_token
        return self.token

    async def _refresh_rejected_token(self, session: aiohttp.ClientSession, rejected_token: str) -> None:
        """CAS-replace the token that was sent; raise TokenStoreLockTimeout, TokenIssueThrottled, or issuance errors.

        ``replace_rejected`` adopts a peer-rotated usable token unchanged and issues only when the stored token is
        the one that was rejected. Passing the token that was sent, rather than ``self.token`` at handling time,
        prevents a second issuance when another task already rotated it. Toss additionally invalidates the previous
        token the moment a new one is issued, so a redundant issuance would revoke the token a peer just obtained.
        """
        try:
            record = await self._token_store().replace_rejected(rejected_token, lambda: self._issue_token(session))
        except TokenIssueThrottled as exc:
            if isinstance(exc, self._TOKEN_THROTTLED_ERROR):
                raise
            raise self._TOKEN_THROTTLED_ERROR(self._TOKEN_THROTTLED_MESSAGE or str(exc)) from exc
        self.token = record.access_token

    def _auth_rejected(self, response: TransportResponse) -> bool:
        """Return is_auth_rejected(_TOKEN_VENDOR, response.status, response.body)."""
        return is_auth_rejected(self._TOKEN_VENDOR, response.status, response.body)
