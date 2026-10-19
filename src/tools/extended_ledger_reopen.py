"""Manual reopen of in-retention EXHAUSTED extended-backfill ledger keys.

Dry-run by default; ``--apply`` rewrites matched keys to FAILED. Never calls a vendor.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from src.backfill.intraday.extended_session_backfill import (
    ExtendedBackfillLedger,
    ReopenSelector,
)
from src.config.collection import CollectionSettings
from src.data.capture_contracts import SEOUL

logger = logging.getLogger(__name__)
KEY_DISPLAY_LIMIT: int = 10


def _parse_day(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("expected YYYY-MM-DD")
        return parsed
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid YYYY-MM-DD date: {value!r}") from None


def _parse_instant(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid ISO datetime: {value!r}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=SEOUL)
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reopen EXHAUSTED extended-backfill ledger keys (dry-run by default).")
    parser.add_argument("--ledger", default=None, help="Ledger parquet path (default the history-tree location).")
    parser.add_argument("--session", dest="sessions", action="append", default=[], help="Session filter (repeatable).")
    parser.add_argument("--date-from", default=None, type=_parse_day, help="Inclusive snapshot_date lower bound.")
    parser.add_argument("--date-to", default=None, type=_parse_day, help="Inclusive snapshot_date upper bound.")
    parser.add_argument("--reason-prefix", default=None, help="Reason prefix matched after stripping exhausted: markers.")
    parser.add_argument("--symbol", dest="symbols", action="append", default=[], help="Symbol filter (repeatable).")
    parser.add_argument("--attempted-since", default=None, type=_parse_instant, help="ISO date or datetime lower bound.")
    parser.add_argument("--as-of", default=None, type=_parse_day, help="Retention window date YYYY-MM-DD (default KST today).")
    parser.add_argument("--apply", action="store_true", help="Persist the reopen; without it, dry-run only.")
    parser.add_argument("--backup-dir", default=None, help="Backup directory (default the ledger directory).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Reopen in-retention EXHAUSTED ledger keys as retryable; dry-run unless --apply; returns 0 on success, 2 on invalid arguments."""
    args = _build_parser().parse_args(argv)
    if args.date_from is not None and args.date_to is not None and args.date_from > args.date_to:
        print("error: --date-from is after --date-to", flush=True)  # noqa: T201 - CLI plan output is the operator contract
        return 2
    narrowed = bool(
        args.sessions or args.symbols or args.date_from is not None or args.date_to is not None
        or bool(args.reason_prefix) or args.attempted_since is not None
    )
    if args.apply and not narrowed:
        print("error: --apply requires at least one narrowing filter", flush=True)  # noqa: T201 - CLI plan output is the operator contract
        return 2
    ledger = ExtendedBackfillLedger(Path(args.ledger) if args.ledger else ExtendedBackfillLedger().path)
    selector = ReopenSelector(
        sessions=frozenset(str(s) for s in args.sessions),
        date_from=args.date_from.isoformat() if args.date_from is not None else None,
        date_to=args.date_to.isoformat() if args.date_to is not None else None,
        reason_prefix=args.reason_prefix,
        symbols=frozenset(str(s).zfill(6) for s in args.symbols),
        attempted_since=args.attempted_since,
    )
    as_of = args.as_of if args.as_of is not None else datetime.now(SEOUL).date()
    plan = ExtendedBackfillLedger(ledger.path).reopen_exhausted(
        selector,
        as_of=as_of,
        retention_days=int(CollectionSettings().COLLECTION_KIS_MINUTE_RETENTION_DAYS),
        apply=bool(args.apply),
        run_id=f"reopen-{uuid.uuid4().hex[:8]}",
        attempted_at=datetime.now(SEOUL),
        backup_dir=Path(args.backup_dir) if args.backup_dir else None,
    )
    reasons = ",".join(f"{name}={plan.by_reason_class[name]}" for name in sorted(plan.by_reason_class)) or "-"
    if args.apply:
        logger.info(
            "[DATA] stage=extended_ledger_reopen mode=APPLY matched=%d skipped_out_of_retention=%d reopened=%d reasons=%s backup=%s",
            plan.matched,
            plan.skipped_out_of_retention,
            plan.reopened,
            reasons,
            plan.backup_path,
        )
        shown = list(plan.keys[:KEY_DISPLAY_LIMIT])
        omitted = len(plan.keys) - len(shown)
        print(f"matched={plan.matched} skipped_out_of_retention={plan.skipped_out_of_retention} reopened={plan.reopened} reasons={reasons}")  # noqa: T201 - CLI plan output is the operator contract
        for day, session, symbol in shown:
            print(f"{day}/{session}/{symbol}")  # noqa: T201 - CLI plan output is the operator contract
        if omitted > 0:
            print(f"외 {omitted}건")  # noqa: T201 - CLI plan output is the operator contract
        if plan.backup_path is not None:
            print(f"backup={plan.backup_path}")  # noqa: T201 - CLI plan output is the operator contract
        return 0
    logger.info(
        "[DATA] stage=extended_ledger_reopen mode=DRY_RUN matched=%d skipped_out_of_retention=%d reasons=%s",
        plan.matched,
        plan.skipped_out_of_retention,
        reasons,
    )
    shown = list(plan.keys[:KEY_DISPLAY_LIMIT])
    omitted = len(plan.keys) - len(shown)
    print(f"matched={plan.matched} skipped_out_of_retention={plan.skipped_out_of_retention} reasons={reasons}")  # noqa: T201 - CLI plan output is the operator contract
    for day, session, symbol in shown:
        print(f"{day}/{session}/{symbol}")  # noqa: T201 - CLI plan output is the operator contract
    if omitted > 0:
        print(f"외 {omitted}건")  # noqa: T201 - CLI plan output is the operator contract
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry; logic covered via unit scenarios
    from src.utils.cli_logging import configure_cli_logging

    configure_cli_logging()
    raise SystemExit(main())
