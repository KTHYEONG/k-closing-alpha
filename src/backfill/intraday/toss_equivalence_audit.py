"""Random-sample equivalence audit of gate-accepted Toss bars against stored KIS bars.

Measures how often a gate-ACCEPTED Toss day differs from KIS on a random
sample of KIS-stored symbol-days. This is the false-accept rate of the volume
gate and the evidence required before the `toss:toss-candles -> KRX` route is
trusted for training data. It writes nothing to the production store: Toss
evidence goes to the explicit `evidence_store` (a temporary root) and no
partition or ledger is touched.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backfill.intraday.toss_regular import (
    TossBasisVerdict,
    acquire_toss_regular_bars,
    toss_basis_verdict,
)
from src.config.collection import CollectionSettings
from src.config.market_session import INTRADAY_SESSION_REGULAR
from src.data.capture_contracts import SEOUL, CaptureStatus
from src.data.capture_store import CaptureStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_FALSE_ACCEPT_SHARE: float = 0.01
_HEARTBEAT_EVERY_SAMPLES: int = 25
_CALLS_PER_SAMPLE: int = 2

_KIS_FULL_SESSION_FIRST_HHMMSS: int = 90000
_KIS_FULL_SESSION_LAST_HHMMSS: int = 151900


@dataclass(frozen=True)
class EquivalenceMetrics:
    snapshot_date: str
    symbol: str
    accepted: bool
    gate_reason: str
    volume_ratio: float | None
    kis_bars: int
    toss_bars: int
    kis_only_bars: int
    toss_only_bars: int
    ohlc_exact_share: float | None
    volume_exact_share: float | None


@dataclass(frozen=True)
class EquivalenceReport:
    n_symbol_days: int
    accepted_share: float
    consolidated_share: float
    accepted_mismatch_share: float
    accepted_kis_only_bars: int
    volume_ratio_quantiles: dict[str, float]
    metrics: tuple[EquivalenceMetrics, ...]


def _hhmmss_to_seconds(hms: int) -> int:
    hms = int(hms)
    return (hms // 10000) * 3600 + ((hms // 100) % 100) * 60 + (hms % 100)


def _seconds_to_hhmmss(seconds: int) -> int:
    seconds = int(seconds)
    return (seconds // 3600) * 10000 + ((seconds % 3600) // 60) * 100 + (seconds % 60)


def _shift_minutes(hms: int, delta_minutes: int) -> int:
    return _seconds_to_hhmmss(_hhmmss_to_seconds(int(hms)) + int(delta_minutes) * 60)


def _labels(frame: pd.DataFrame) -> list[int]:
    if frame is None or frame.empty or "ts_hms" not in frame.columns:
        return []
    values = pd.to_numeric(frame["ts_hms"], errors="coerce").dropna().astype(np.int64).tolist()
    return [int(v) for v in values]


def _identity(frame: pd.DataFrame) -> tuple[str, str]:
    day, symbol = "", ""
    if frame is not None and not frame.empty:
        if "snapshot_date" in frame.columns and len(frame):
            day = str(frame["snapshot_date"].iloc[0])
        if "symbol" in frame.columns and len(frame):
            symbol = str(frame["symbol"].iloc[0])
    return day, symbol


def compare_toss_to_kis(
    kis_bars: pd.DataFrame, toss_bars: pd.DataFrame, *, verdict: TossBasisVerdict
) -> EquivalenceMetrics:
    """Compare one symbol-day of Toss bars with the stored KIS bars of the same day after aligning the END-stamped Toss label to the KIS start stamp (label minus one minute). Bars present only in Toss are zero-volume carry minutes and are reported, not penalized; bars present only in KIS are a defect of the Toss series and are reported separately."""
    kis_labels = _labels(kis_bars)
    toss_labels = _labels(toss_bars)
    toss_as_start = [_shift_minutes(int(v), -1) for v in toss_labels]
    kis_set = set(kis_labels)
    aligned = sorted(kis_set & set(toss_as_start))
    kis_only = len(kis_set - set(toss_as_start))
    toss_only = len(set(toss_as_start) - kis_set)
    day, symbol = _identity(kis_bars)
    if not day or not symbol:
        fallback_day, fallback_symbol = _identity(toss_bars)
        day = day or fallback_day
        symbol = symbol or fallback_symbol
    kis_vol = (
        float(pd.to_numeric(kis_bars["volume"], errors="coerce").fillna(0).sum())
        if kis_bars is not None and not kis_bars.empty and "volume" in kis_bars.columns
        else 0.0
    )
    toss_vol = (
        float(pd.to_numeric(toss_bars["volume"], errors="coerce").fillna(0).sum())
        if toss_bars is not None and not toss_bars.empty and "volume" in toss_bars.columns
        else 0.0
    )
    volume_ratio = (toss_vol / kis_vol) if kis_vol > 0 else None
    if not aligned:
        return EquivalenceMetrics(
            snapshot_date=day,
            symbol=symbol,
            accepted=bool(verdict.accepted),
            gate_reason=str(verdict.reason),
            volume_ratio=volume_ratio,
            kis_bars=len(kis_labels),
            toss_bars=len(toss_labels),
            kis_only_bars=kis_only,
            toss_only_bars=toss_only,
            ohlc_exact_share=None,
            volume_exact_share=None,
        )
    kis_by_label: dict[int, pd.Series] = {}
    if kis_bars is not None and not kis_bars.empty:
        for _, row in kis_bars.iterrows():
            try:
                kis_by_label[int(float(str(row["ts_hms"])))] = row
            except (ValueError, TypeError):
                continue
    toss_by_start: dict[int, pd.Series] = {}
    if toss_bars is not None and not toss_bars.empty:
        for _, row in toss_bars.iterrows():
            try:
                toss_by_start[_shift_minutes(int(float(str(row["ts_hms"]))), -1)] = row
            except (ValueError, TypeError):
                continue
    ohlc_hits = 0
    vol_hits = 0
    for label in aligned:
        left = kis_by_label.get(label)
        right = toss_by_start.get(label)
        if left is None or right is None:
            continue
        try:
            left_ohlc = [float(left[c]) for c in ("open", "high", "low", "close")]
            right_ohlc = [float(right[c]) for c in ("open", "high", "low", "close")]
        except (ValueError, TypeError, KeyError):
            continue
        if all(a == b for a, b in zip(left_ohlc, right_ohlc, strict=True)):
            ohlc_hits += 1
        try:
            if float(left["volume"]) == float(right["volume"]):
                vol_hits += 1
        except (ValueError, TypeError, KeyError):
            pass
    denom = len(aligned)
    return EquivalenceMetrics(
        snapshot_date=day,
        symbol=symbol,
        accepted=bool(verdict.accepted),
        gate_reason=str(verdict.reason),
        volume_ratio=volume_ratio,
        kis_bars=len(kis_labels),
        toss_bars=len(toss_labels),
        kis_only_bars=kis_only,
        toss_only_bars=toss_only,
        ohlc_exact_share=float(ohlc_hits / denom),
        volume_exact_share=float(vol_hits / denom),
    )


def sample_kis_symbol_days(
    *,
    start: str,
    end: str,
    n: int,
    seed: int,
    stored_loader: Callable[[str], pd.DataFrame],
    calendar: Sequence[str],
) -> list[tuple[str, str]]:
    """Draw a seeded sample of stored vendor-`kis` full-session symbol-days.

    Only symbol-days whose stored bars are all vendor `kis` and span the full
    regular session are eligible. Sampling uses a seeded generator, so the same
    `(seed, calendar, stored data)` gives the same sample.
    """
    eligible: list[tuple[str, str]] = []
    for day in sorted({str(d) for d in calendar}):
        if day < str(start) or day > str(end):
            continue
        frame = stored_loader(day)
        if frame is None or frame.empty:
            continue
        if not {"symbol", "vendor", "ts_hms"} <= set(frame.columns):
            continue
        for symbol, group in frame.groupby(frame["symbol"].astype(str)):
            vendors = {str(v) for v in group["vendor"].astype(str).tolist()}
            if vendors != {"kis"}:
                continue
            stamps = pd.to_numeric(group["ts_hms"], errors="coerce").dropna()
            if stamps.empty:
                continue
            if int(stamps.min()) <= _KIS_FULL_SESSION_FIRST_HHMMSS and int(stamps.max()) >= _KIS_FULL_SESSION_LAST_HHMMSS:
                eligible.append((day, str(symbol)))
    eligible = sorted(set(eligible))
    if n <= 0 or not eligible:
        return []
    if n >= len(eligible):
        return eligible
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(len(eligible), size=int(n), replace=False)
    return sorted([eligible[int(i)] for i in picked])


def _gate_thresholds(profile: CollectionSettings) -> tuple[float, float]:
    from src.backfill.intraday.toss_regular import TOSS_BASIS_RATIO_MIN, TOSS_BASIS_RATIO_TOLERANCE

    ratio_min = getattr(profile, "COLLECTION_TOSS_BASIS_VOLUME_RATIO_MIN", TOSS_BASIS_RATIO_MIN)
    tolerance = getattr(profile, "COLLECTION_TOSS_BASIS_VOLUME_TOLERANCE", TOSS_BASIS_RATIO_TOLERANCE)
    return float(ratio_min), float(tolerance)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def run_equivalence_audit(
    *,
    samples: Sequence[tuple[str, str]],
    client: Any,
    session: Any,
    profile: CollectionSettings,
    eod_volumes: Mapping[tuple[str, str], float],
    kis_loader: Callable[..., pd.DataFrame],
    evidence_store: CaptureStore,
    run_id: str,
) -> EquivalenceReport:
    """Measure on a random sample of KIS-stored symbol-days how often a gate-ACCEPTED Toss day differs from KIS. This is the false-accept rate of the volume gate and the evidence required before the `toss:toss-candles -> KRX` route is trusted for training data. It writes nothing to the production store: Toss evidence goes to the explicit `evidence_store` (a temporary root) and no partition or ledger is touched."""
    if not str(run_id).strip():
        raise ValueError("run_id must be nonempty")
    if evidence_store is None:
        raise ValueError("evidence_store must be an explicit temporary CaptureStore")
    ordered = [(str(day), str(symbol)) for day, symbol in samples]
    ratio_min, tolerance = _gate_thresholds(profile)
    collected: list[EquivalenceMetrics] = []
    for index, (day, symbol) in enumerate(ordered, start=1):
        eod = eod_volumes.get((day, symbol))
        toss_frame, entry = await acquire_toss_regular_bars(
            client,
            session,
            symbol,
            day,
            eod_volume=eod,
            profile=profile,
            capture_store=evidence_store,
            run_id=str(run_id),
        )
        kis_frame = await _maybe_await(kis_loader(day, symbol))
        if entry.status == CaptureStatus.COMPLETE:
            verdict = toss_basis_verdict(toss_frame, eod, ratio_min=ratio_min, ratio_tolerance=tolerance)
        else:
            verdict = TossBasisVerdict(accepted=False, reason=str(entry.reason), volume_ratio=None)
        collected.append(compare_toss_to_kis(kis_frame, toss_frame, verdict=verdict))
        if index % _HEARTBEAT_EVERY_SAMPLES == 0 or index == len(ordered):
            logger.info(
                "[DATA] stage=toss_equivalence_audit progress=samples %d/%d",
                index,
                len(ordered),
            )
    n = len(collected)
    accepted = [m for m in collected if m.accepted]
    consolidated = [m for m in collected if m.gate_reason == "toss_consolidated_tape"]
    mismatched = [
        m
        for m in accepted
        if m.ohlc_exact_share is None
        or m.volume_exact_share is None
        or float(m.ohlc_exact_share) < 1.0
        or float(m.volume_exact_share) < 1.0
    ]
    ratios = sorted(m.volume_ratio for m in accepted if m.volume_ratio is not None)
    quantiles: dict[str, float] = {}
    if ratios:
        arr = np.asarray(ratios, dtype=np.float64)
        for name, pct in (("p5", 5.0), ("p25", 25.0), ("p50", 50.0), ("p75", 75.0), ("p95", 95.0)):
            quantiles[name] = float(np.percentile(arr, pct))
    return EquivalenceReport(
        n_symbol_days=int(n),
        accepted_share=float(len(accepted) / n) if n else 0.0,
        consolidated_share=float(len(consolidated) / n) if n else 0.0,
        accepted_mismatch_share=float(len(mismatched) / len(accepted)) if accepted else 0.0,
        accepted_kis_only_bars=int(sum(m.kis_only_bars for m in accepted)),
        volume_ratio_quantiles=quantiles,
        metrics=tuple(collected),
    )


def _exit_code_for_report(report: EquivalenceReport, *, threshold: float) -> int:
    accepted = int(round(report.accepted_share * report.n_symbol_days))
    if report.n_symbol_days == 0 or accepted == 0:
        return 1
    if report.accepted_mismatch_share > float(threshold) or report.accepted_kis_only_bars > 0:
        return 1
    return 0


def _default_out_path() -> Path:
    return Path("scratch") / "toss_equivalence_audit.json"


def _report_payload(report: EquivalenceReport) -> dict[str, Any]:
    return {
        "n_symbol_days": report.n_symbol_days,
        "accepted_share": report.accepted_share,
        "consolidated_share": report.consolidated_share,
        "accepted_mismatch_share": report.accepted_mismatch_share,
        "accepted_kis_only_bars": report.accepted_kis_only_bars,
        "volume_ratio_quantiles": dict(report.volume_ratio_quantiles),
        "metrics": [asdict(m) for m in report.metrics],
    }


async def _run_with_session(  # noqa: PLR0913
    *,
    samples: Sequence[tuple[str, str]],
    client: Any,
    http_session: Any,
    profile: CollectionSettings,
    eod_volumes: Mapping[tuple[str, str], float],
    kis_loader: Callable[..., pd.DataFrame],
    evidence_store: CaptureStore,
    run_id: str,
) -> EquivalenceReport:
    return await run_equivalence_audit(
        samples=samples,
        client=client,
        session=http_session,
        profile=profile,
        eod_volumes=eod_volumes,
        kis_loader=kis_loader,
        evidence_store=evidence_store,
        run_id=run_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Sample stored KIS symbol-days, audit gate-accepted Toss bars, write the JSON report."""
    parser = argparse.ArgumentParser(description="Audit gate-accepted Toss bars against stored KIS bars.")
    parser.add_argument("--start", default=None, help="Inclusive sample start YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Inclusive sample end YYYY-MM-DD.")
    parser.add_argument("--n", type=int, default=200, help="Number of symbol-days to sample.")
    parser.add_argument("--seed", type=int, default=7, help="Seeded sampling seed.")
    parser.add_argument("--out", default=str(_default_out_path()), help="Report JSON path.")
    parser.add_argument(
        "--max-false-accept-share",
        type=float,
        default=DEFAULT_MAX_FALSE_ACCEPT_SHARE,
        help="Non-zero exit when accepted_mismatch_share exceeds this.",
    )
    parser.add_argument("--run-id", default=None, help="Audit run identity (default toss-audit-<seed>).")
    args = parser.parse_args(argv)
    threshold = float(args.max_false_accept_share)

    from src.backfill.intraday.toss_regular_backfill import _load_inputs, _open_toss_client
    from src.data.intraday_store import intraday_partition_path

    profile = CollectionSettings()
    today = datetime.now(SEOUL).date()
    wide_history, prepared, calendar = _load_inputs(today)
    if not calendar:
        raise ValueError("toss equivalence audit found no trading calendar")
    start = str(args.start) if args.start else min(calendar)
    last_past = max((d for d in calendar if d < today.isoformat()), default=None)
    if last_past is None:
        raise ValueError("toss equivalence audit found no strictly past trading day")
    end = min(str(args.end), last_past) if args.end else last_past
    date.fromisoformat(start)
    date.fromisoformat(end)
    volumes = pd.to_numeric(prepared["volume"], errors="coerce")
    days = pd.to_datetime(prepared["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    symbols = prepared["symbol"].astype(str).str.zfill(6)
    eod_volumes = {
        (str(day), str(symbol)): float(volume)
        for day, symbol, volume in zip(days.tolist(), symbols.tolist(), volumes.tolist())
    }

    def _stored_loader(day: str) -> pd.DataFrame:
        target = intraday_partition_path(1, str(day), INTRADAY_SESSION_REGULAR)
        if not target.exists():
            return pd.DataFrame()
        return pd.read_parquet(target)

    def _kis_loader(day: str, symbol: str) -> pd.DataFrame:
        frame = _stored_loader(day)
        if frame.empty:
            return frame
        sub = frame[(frame["symbol"].astype(str) == str(symbol)) & (frame["vendor"].astype(str) == "kis")]
        return sub.reset_index(drop=True)

    samples = sample_kis_symbol_days(
        start=start, end=end, n=int(args.n), seed=int(args.seed),
        stored_loader=_stored_loader, calendar=calendar,
    )
    logger.info(
        "[DATA] stage=toss_equivalence_audit status=START samples=%d estimated_calls=%d",
        len(samples),
        _CALLS_PER_SAMPLE * len(samples),
    )
    tmpdir = tempfile.mkdtemp(prefix="toss_equiv_")
    try:
        evidence_store = CaptureStore(Path(tmpdir))
        client = _open_toss_client()

        async def _run() -> EquivalenceReport:
            from src.backfill.intraday.toss_regular_backfill import _http_session

            async with _http_session(client) as http_session:
                return await _run_with_session(
                    samples=samples,
                    client=client,
                    http_session=http_session,
                    profile=profile,
                    eod_volumes=eod_volumes,
                    kis_loader=_kis_loader,
                    evidence_store=evidence_store,
                    run_id=str(args.run_id) if args.run_id else f"toss-audit-{int(args.seed)}",
                )

        report = asyncio.run(_run())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_report_payload(report), indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "[DATA] stage=toss_equivalence_audit status=DONE samples=%d accepted_share=%.4f mismatch_share=%.4f kis_only=%d out=%s",
        report.n_symbol_days,
        report.accepted_share,
        report.accepted_mismatch_share,
        report.accepted_kis_only_bars,
        out,
    )
    if _exit_code_for_report(report, threshold=threshold):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry; logic covered via unit scenarios
    from src.utils.cli_logging import configure_cli_logging

    configure_cli_logging()
    raise SystemExit(main())
