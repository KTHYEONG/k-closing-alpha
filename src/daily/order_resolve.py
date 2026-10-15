"""Manual operator resolution for stuck UNKNOWN intents."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from src.api.kis.trading import BrokerReadError, KisTradingClient, TradingDisabledError, resolve_trading_binding
from src.config import settings
from src.daily.order_session import SessionBrokerPort
from src.execution.order_contract import BrokerOrder, normalize_order_no
from src.execution.order_gateway import match_candidates
from src.execution.order_journal import IntentStatus, JournalEventKind, OrderJournal
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolveReport:
    """Outcome of one operator resolve attempt."""

    intent_id: str
    action: str
    applied: bool
    reason_code: str
    candidates: tuple[str, ...]


def _refused(intent_id: str, reason_code: str, candidates: tuple[str, ...] = ()) -> ResolveReport:
    return ResolveReport(intent_id=intent_id, action="refused", applied=False,
                         reason_code=reason_code, candidates=candidates)


async def resolve_unknown_intent(
    intent_id: str,
    *,
    order_no: str | None,
    not_found: bool,
    operator: str,
    reason: str,
    apply: bool = False,
    broker: SessionBrokerPort | None = None,
    journal: OrderJournal | None = None,
) -> ResolveReport:
    """Validate against the broker's daily list and (when apply) append one OPERATOR_RESOLVE event."""
    has_order = order_no is not None and str(order_no).strip() != ""
    want_not_found = bool(not_found)
    if has_order == want_not_found:
        return _refused(intent_id, "bad_arguments")
    if not isinstance(operator, str) or not operator.strip():
        return _refused(intent_id, "bad_arguments")
    if not isinstance(reason, str) or not reason.strip():
        return _refused(intent_id, "bad_arguments")
    try:
        binding = resolve_trading_binding()
    except TradingDisabledError:
        return _refused(intent_id, "trading_disabled")
    active_journal = journal if journal is not None else OrderJournal(
        settings.ORDER_JOURNAL_DIR, binding.env,
        lock_timeout_seconds=settings.ORDER_JOURNAL_LOCK_TIMEOUT_SECONDS)
    active_broker: SessionBrokerPort | Any = broker
    if active_broker is None:
        timeout = (
            settings.ORDER_MOCK_HTTP_TIMEOUT_SECONDS
            if binding.env == "mock"
            else settings.ORDER_HTTP_TIMEOUT_SECONDS
        )
        active_broker = KisTradingClient(binding, http_timeout_seconds=timeout)
    run_id = f"operator-resolve:{intent_id}"
    clean_operator = operator.strip()
    clean_reason = reason.strip()

    async def _execute(sess: Any) -> ResolveReport:
        with active_journal.exclusive():
            states = active_journal.states()
            state = states.get(intent_id)
            if state is None or state.intent is None:
                return _refused(intent_id, "not_unknown")
            allowed = (IntentStatus.UNKNOWN,) if want_not_found else (IntentStatus.UNKNOWN, IntentStatus.NOT_FOUND)
            if state.status not in allowed:
                return _refused(intent_id, "not_unknown")
            intent = state.intent
            target_no = normalize_order_no(order_no) if order_no is not None else ""
            if has_order and not target_no:
                return _refused(intent_id, "bad_arguments")
            bound = {normalize_order_no(st.broker_order_no) for st in states.values() if st.broker_order_no}
            try:
                await active_broker.ensure_token(sess)
                snapshot = await active_broker.fetch_daily_orders(sess, intent.trading_date)
            except BrokerReadError as exc:
                logger.warning("[EXEC] stage=operator_resolve intent_id=%s outcome=refused reason=broker_unreadable "
                               "detail=%s", intent_id, exc)
                return _refused(intent_id, "broker_unreadable")
            if has_order:
                named = [o for o in snapshot if normalize_order_no(o.order_no) == target_no]
                # Operator evidence overrides the baseline, never the one-order-one-intent binding.
                matched = match_candidates(intent, named, bound)
                if len(matched) != 1 or not matched[0].org_no.strip():
                    return _refused(intent_id, "order_mismatch")
                target: BrokerOrder = matched[0]
                if apply:
                    active_journal.append(
                        JournalEventKind.OPERATOR_RESOLVE, intent_id,
                        {"resolution": "adopt", "operator": clean_operator, "reason": clean_reason,
                         "broker_order_no": target.order_no, "broker_org_no": target.org_no},
                        run_id=run_id)
                    logger.info("[EXEC] stage=operator_resolve intent_id=%s action=adopt applied=true", intent_id)
                return ResolveReport(intent_id=intent_id, action="adopt", applied=apply,
                                     reason_code="ok", candidates=(target_no,))
            found = match_candidates(intent, snapshot, bound, baseline=state.baseline_order_nos)
            if found:
                numbers = tuple(sorted({normalize_order_no(o.order_no) for o in found}))
                return _refused(intent_id, "candidate_exists", numbers)
            if apply:
                active_journal.append(
                    JournalEventKind.OPERATOR_RESOLVE, intent_id,
                    {"resolution": "not_found", "operator": clean_operator, "reason": clean_reason},
                    run_id=run_id)
                logger.info("[EXEC] stage=operator_resolve intent_id=%s action=not_found applied=true", intent_id)
            return ResolveReport(intent_id=intent_id, action="not_found", applied=apply,
                                 reason_code="ok", candidates=())

    if hasattr(active_broker, "create_session"):
        async with active_broker.create_session() as sess:
            return await _execute(sess)
    return await _execute(None)


def main() -> None:
    """CLI entry: --intent-id, --order-no | --not-found, --operator, --reason, --apply."""
    parser = argparse.ArgumentParser(description="Manually resolve a stuck UNKNOWN intent")
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--order-no", default=None)
    parser.add_argument("--not-found", action="store_true")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--i-understand-live", action="store_true")
    args = parser.parse_args()
    configure_cli_logging()
    try:
        binding = resolve_trading_binding()
    except TradingDisabledError:
        logger.warning("[EXEC] stage=operator_resolve intent_id=%s outcome=refused reason=trading_disabled",
                       args.intent_id)
        raise SystemExit(2) from None
    if binding.env == "live" and not args.i_understand_live:
        logger.warning("[EXEC] stage=operator_resolve intent_id=%s outcome=refused reason=live_confirmation_required",
                       args.intent_id)
        raise SystemExit(2)
    report = asyncio.run(resolve_unknown_intent(
        args.intent_id, order_no=args.order_no, not_found=args.not_found,
        operator=args.operator, reason=args.reason, apply=args.apply))
    logger.info("[EXEC] stage=operator_resolve intent_id=%s action=%s applied=%s reason=%s",
                report.intent_id, report.action, report.applied, report.reason_code)
    if report.action == "refused":
        raise SystemExit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
