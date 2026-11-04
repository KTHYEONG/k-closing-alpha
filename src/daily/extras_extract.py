"""Nightly extraction of decision-run raw captures into research extras (research only, no broker calls)."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

from src.data.capture_contracts import SEOUL
from src.data.capture_store import CaptureStore, resolve_capture_root
from src.data.decision_extras import (
    DECISION_EXTRAS_DATASET,
    INVESTOR_DECISION_DATASET,
    build_decision_extras,
    decision_run_id,
    investor_series_rows,
    kis_price_rows,
    kiwoom_scan_rows,
)
from src.data.research_store import write_research_partition
from src.tools.alerts import redact_for_egress
from src.tools.run_outcome import (
    RUN_OUTCOME_DEGRADED,
    RUN_OUTCOME_OK,
    RUN_OUTCOME_SKIPPED,
    record_run_outcome,
)
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

_JOB = "decision_extras"


def _now() -> datetime:
    return datetime.now(SEOUL)


def extract_decision_extras(snapshot_date: str, *, store: CaptureStore, now: datetime) -> tuple[int, int]:
    """Write the decision extras and investor series partitions of one date from its raw decision run.

    Args:
        snapshot_date: ISO trading date whose decision run is extracted.
        store: Owner-local capture store holding the raw evidence.
        now: Aware cutoff; only decision runs completed by then qualify.

    Returns:
        Row counts of the extras partition and the investor series partition.

    Raises:
        FileNotFoundError: No qualifying decision run exists for the date.
        ValueError: Raw evidence is corrupt or the decision manifest is inconsistent.
    """
    run_id = decision_run_id(store, snapshot_date, available_by=now)
    cohort = store.read_cohort(snapshot_date, available_by=now).eligible_symbols
    scan = kiwoom_scan_rows(store.iter_run_envelopes(snapshot_date, run_id))
    price = kis_price_rows(store.iter_run_envelopes(snapshot_date, run_id))
    investor = investor_series_rows(store.iter_run_envelopes(snapshot_date, run_id))
    extras = build_decision_extras(cohort, scan, price, run_id)
    n_extras = write_research_partition(extras, DECISION_EXTRAS_DATASET, snapshot_date, overwrite=True)
    n_series = write_research_partition(investor, INVESTOR_DECISION_DATASET, snapshot_date, overwrite=True)
    return n_extras, n_series


def _requested_days(parser: argparse.ArgumentParser, args: argparse.Namespace, today: date) -> list[date]:
    if args.start is None:
        return [args.date or today]
    if args.end < args.start:
        parser.error("--to precedes --from")
    return [args.start + timedelta(days=offset) for offset in range((args.end - args.start).days + 1)]


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python -m src.daily.extras_extract [--date YYYY-MM-DD | --from D --to D]``."""
    parser = argparse.ArgumentParser(description="KCA decision-run extras extraction from raw capture (research, no orders)")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--from", dest="start", type=date.fromisoformat, default=None)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)
    if args.date is not None and (args.start is not None or args.end is not None):
        parser.error("--date cannot be combined with --from/--to")
    if (args.start is None) != (args.end is None):
        parser.error("--from and --to must be given together")

    now = _now()
    days = _requested_days(parser, args, now.date())
    store = CaptureStore(resolve_capture_root())
    counts = {"ok": 0, "raw_absent": 0, "no_decision_run": 0, "invalid": 0}
    rows = {"extras_rows": 0, "series_rows": 0}
    for day in days:
        snapshot_date = day.isoformat()
        if not (store.root / "raw" / snapshot_date).is_dir():
            counts["raw_absent"] += 1
            logger.info("[DATA] stage=decision_extras status=SKIP reason=raw_absent date=%s", snapshot_date)
            continue
        try:
            n_extras, n_series = extract_decision_extras(snapshot_date, store=store, now=now)
        except FileNotFoundError:
            counts["no_decision_run"] += 1
            logger.info("[DATA] stage=decision_extras status=SKIP reason=no_decision_run date=%s", snapshot_date)
            continue
        except ValueError as exc:
            counts["invalid"] += 1
            reason = redact_for_egress(str(exc)) or type(exc).__name__
            logger.warning("[DATA] stage=decision_extras status=FAILED reason=%s date=%s", reason, snapshot_date)
            continue
        counts["ok"] += 1
        rows["extras_rows"] += n_extras
        rows["series_rows"] += n_series
        logger.info(
            "[DATA] stage=decision_extras status=OK date=%s extras=%d series=%d",
            snapshot_date,
            n_extras,
            n_series,
        )

    run_date = days[-1].isoformat()
    metrics = {**counts, **rows}
    if counts["invalid"]:
        record_run_outcome(_JOB, RUN_OUTCOME_DEGRADED, run_date=run_date, reason="invalid_raw_evidence", metrics=metrics)
    elif counts["ok"]:
        record_run_outcome(_JOB, RUN_OUTCOME_OK, run_date=run_date, metrics=metrics)
    else:
        record_run_outcome(_JOB, RUN_OUTCOME_SKIPPED, run_date=run_date, reason="no_decision_input", metrics=metrics)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
