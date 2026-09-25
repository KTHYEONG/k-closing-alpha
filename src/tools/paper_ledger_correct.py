"""Operator CLI for auditable paper-ledger corrections (no market access)."""

from __future__ import annotations

import argparse
import logging

from src import settings
from src.execution.paper_broker import PaperLedger, refresh_trade_ledgers

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable paper-ledger corrections (no market access)")
    sub = parser.add_subparsers(dest="command", required=True)
    void = sub.add_parser("void-fill", help="Void one recorded fill via a VOID_FILL correction")
    void.add_argument("--order-id", required=True)
    void.add_argument("--reason", required=True)
    void.add_argument("--evidence", required=True)
    void.add_argument("--operator", required=True)
    void.add_argument("--as-of", required=True)
    note = sub.add_parser("note", help="Record a NOTE correction with no economic effect")
    note.add_argument("--targets", required=True, help="Comma-separated order ids")
    note.add_argument("--reason", required=True)
    note.add_argument("--evidence", required=True)
    note.add_argument("--operator", required=True)
    note.add_argument("--as-of", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Operator CLI for auditable paper-ledger corrections (no market access).

    Subcommands:
        void-fill --order-id ID --reason R --evidence E --operator O --as-of YYYY-MM-DD
        note --targets ID[,ID...] --reason R --evidence E --operator O --as-of YYYY-MM-DD

    Both subcommands hold PaperLedger.exclusive for the whole operation and run
    refresh_trade_ledgers for --as-of after a VOID_FILL.

    Raises:
        SystemExit: argparse errors; ValueError from the ledger propagates as exit 1.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    ledger = PaperLedger()
    try:
        with ledger.exclusive():
            if args.command == "void-fill":
                row = ledger.void_fill(
                    args.order_id,
                    reason=args.reason,
                    evidence=args.evidence,
                    operator=args.operator,
                    as_of_date=args.as_of,
                )
                refresh_trade_ledgers(ledger, settings.PAPER_SEED_CAPITAL, args.as_of)
            else:
                targets = tuple(t.strip() for t in str(args.targets).split(",") if t.strip())
                row = ledger.record_note(
                    target_order_ids=targets,
                    reason=args.reason,
                    evidence=args.evidence,
                    operator=args.operator,
                    as_of_date=args.as_of,
                )
        logger.info(
            "[PORTFOLIO] stage=paper_ledger_correction action=%s correction_id=%s targets=%s operator=%s",
            row["action"],
            row["correction_id"],
            row["target_order_ids"],
            row["operator"],
        )
    except ValueError as exc:
        logger.error("paper_ledger_correct: error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
