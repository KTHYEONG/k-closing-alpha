"""Point-in-time security-class verdict for price panels (data-layer ownership).

The live cohort excludes non-screenable instruments (non-주권, non-보통주, 관리종목, SPAC,
투자주의환기) using the KRX issue-base-info classification observed on the previous trading
day. Training and certification panels need the same verdict for every historical row so that
UniverseSpec.exclude_non_screenable_class screens both populations identically. Dates before
the classification panel's coverage fall back to a static short-code proxy; dates inside the
coverage never do.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.contract import UniverseSpec

logger = logging.getLogger(__name__)

SECURITY_CLASSIFICATION_PARQUET_FILENAME: str = "security_classification.parquet"
"""File name of the classification panel under settings.ALTDATA_DIR (single definition)."""

SCREENABLE_SOURCE_COL: str = "screenable_source"
"""Provenance column paired with SCREENABLE_CLASS_COL: ScreenableSource value per row."""

COMMON_SHARE_CODE_SUFFIX: str = "0"
"""Last character of every KRX common-share short code.

KRX derives preferred and class-share codes from the common code by replacing the last
character (005930 -> 005935; alphanumeric 00088K), so a different last character marks a
non-common issue. Evidence: over 2026-09-08..2026-10-01 (44,220 classified panel rows) no row
with another last character was classified screenable.
"""

FOREIGN_ISSUER_CODE_PREFIX: str = "9"
"""First character of KRX short codes assigned to foreign issuers (외국주권) and depositary
receipts (주식예탁증권).

Evidence: every 외국주권/주식예탁증권 row in the covered window carries this prefix and no
screenable row does (0 false exclusions over 44,220 rows).
"""

CLASSIFICATION_VERDICT_COLUMNS: tuple[str, ...] = ("date", "symbol", "is_screenable")
"""Minimal column projection read for verdict resolution (column pruning)."""

CLASSIFICATION_ATTRIBUTE_COLUMNS: tuple[str, ...] = ("security_group", "security_kind", "section_type")
"""Raw KRX attributes read only by the agreement report to break down proxy misses."""

__all__ = [
    "CLASSIFICATION_ATTRIBUTE_COLUMNS",
    "CLASSIFICATION_VERDICT_COLUMNS",
    "COMMON_SHARE_CODE_SUFFIX",
    "FOREIGN_ISSUER_CODE_PREFIX",
    "SCREENABLE_SOURCE_COL",
    "SECURITY_CLASSIFICATION_PARQUET_FILENAME",
    "ScreenableClassProvenance",
    "ScreenableSource",
    "attach_screenable_class",
    "build_proxy_agreement_report",
    "load_classification_panel",
    "main",
    "proxy_screenable",
]


class ScreenableSource(StrEnum):
    """Origin of a row's is_screenable verdict."""

    REAL = "real"
    PROXY = "proxy"


@dataclass(frozen=True)
class ScreenableClassProvenance:
    """Counters emitted with every verdict attachment.

    Attributes:
        n_rows: Panel rows that received a verdict.
        n_real: Rows resolved from the classification panel.
        n_proxy: Rows resolved by the short-code proxy (as-of date before coverage_start).
        n_real_absent: Real-resolved rows whose symbol had no classification row on the
            as-of date (verdict False, mirroring live set membership).
        n_non_screenable_real: Real-resolved rows with verdict False.
        n_non_screenable_proxy: Proxy-resolved rows with verdict False.
        coverage_start: First classification date, YYYY-MM-DD.
        coverage_end: Last classification date, YYYY-MM-DD.
    """

    n_rows: int
    n_real: int
    n_proxy: int
    n_real_absent: int
    n_non_screenable_real: int
    n_non_screenable_proxy: int
    coverage_start: str
    coverage_end: str

    def to_log_kv(self) -> str:
        """Return a flat space-joined key=value string in field-declaration order."""
        return " ".join(f"{f.name}={getattr(self, f.name)}" for f in dataclasses.fields(self))


def proxy_screenable(symbols: pd.Series) -> np.ndarray:
    """Short-code proxy of the KRX screenability verdict for pre-coverage dates.

    Catches preferred/class shares and foreign issuers/DRs only. 관리종목, SPAC,
    투자주의환기 and non-stock vehicles (REIT, infrastructure, investment companies) carry
    ordinary common-share codes and are NOT detectable from the code; the proxy therefore
    never excludes more than the real verdict (measured false exclusions: 0) but may admit
    names the real verdict excludes.

    Args:
        symbols: KRX short codes (6-character, numeric or alphanumeric).

    Returns:
        Bool array aligned with symbols; False for a code whose last character is not
        COMMON_SHARE_CODE_SUFFIX or whose first character is FOREIGN_ISSUER_CODE_PREFIX.

    Raises:
        ValueError: When any symbol is not a 6-character KRX short code
            (src.backfill.altdata.config.is_krx_short_code).
    """
    from src.backfill.altdata.config import is_krx_short_code

    vals = pd.Series(symbols).astype(str).to_numpy()
    if len(vals) == 0:
        return np.zeros(0, dtype=bool)
    bad = [v for v in vals.tolist() if not is_krx_short_code(v)]
    if bad:
        raise ValueError(f"proxy_screenable received non KRX short codes: {bad[:3]}")
    s = pd.Series(vals.astype(str))
    last = s.str[-1].to_numpy()
    first = s.str[0].to_numpy()
    return np.asarray(
        (last == COMMON_SHARE_CODE_SUFFIX) & (first != FOREIGN_ISSUER_CODE_PREFIX),
        dtype=bool,
    )


def load_classification_panel(
    path: str | os.PathLike[str] | None = None,
    *,
    columns: Sequence[str] = CLASSIFICATION_VERDICT_COLUMNS,
) -> pd.DataFrame:
    """Read the classification panel for verdict resolution.

    Args:
        path: Panel parquet; None selects settings.ALTDATA_DIR /
            SECURITY_CLASSIFICATION_PARQUET_FILENAME.
        columns: Projection to read; must include CLASSIFICATION_VERDICT_COLUMNS.

    Returns:
        Frame with normalized datetime64 ``date`` (midnight), str ``symbol`` and bool
        ``is_screenable`` plus any extra requested columns.

    Raises:
        FileNotFoundError: When the parquet does not exist.
        ValueError: When the panel is empty, a required column is missing, is_screenable
            has nulls, or a (date, symbol) key repeats.
    """
    from src import settings

    cols = list(columns)
    missing_req = [c for c in CLASSIFICATION_VERDICT_COLUMNS if c not in cols]
    if missing_req:
        raise ValueError(f"load_classification_panel columns must include {missing_req}")
    src = Path(path) if path is not None else Path(settings.ALTDATA_DIR) / SECURITY_CLASSIFICATION_PARQUET_FILENAME
    if not src.exists():
        raise FileNotFoundError(f"security_classification not found: {src}")
    frame = pd.read_parquet(src, columns=cols)
    if frame.empty:
        raise ValueError(f"security_classification panel is empty: {src}")
    missing = [c for c in CLASSIFICATION_VERDICT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"security_classification panel missing columns: {missing}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if out["date"].isna().any():
        raise ValueError("security_classification panel has unparseable dates")
    out["symbol"] = out["symbol"].astype(str)
    if out["is_screenable"].isna().any():
        raise ValueError("security_classification panel has null is_screenable")
    out["is_screenable"] = out["is_screenable"].astype(bool)
    if int(out.duplicated(subset=["date", "symbol"]).sum()):
        raise ValueError("security_classification panel repeats (date, symbol)")
    return out.reset_index(drop=True)


def attach_screenable_class(
    panel: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    market_dates: np.ndarray,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, ScreenableClassProvenance]:
    """Attach the point-in-time security-class verdict to every panel row.

    A row dated T is decided at T's close, before KRX publishes T's classification, so its
    verdict is the classification observed on the previous trading day — the same as-of rule
    the live cohort applies via load_security_classification. As-of dates before the
    classification coverage start use proxy_screenable; as-of dates inside the coverage must
    be present in the classification panel.

    Args:
        panel: Price panel with date and symbol columns (never mutated).
        classification: Output of load_classification_panel.
        market_dates: Sorted unique trading dates of the full panel; supplies the previous
            trading day of each row.
        date_col: Name of the trading-date column.
        symbol_col: Name of the symbol column.

    Returns:
        (new frame with SCREENABLE_CLASS_COL bool and SCREENABLE_SOURCE_COL str columns,
        provenance counters).

    Raises:
        ValueError: When classification is empty; a panel date is not in market_dates; an
            as-of date at/after the coverage start has no classification rows (message names
            the count and the first missing dates); or a row without a previous trading day
            is dated after the coverage start.
    """
    from src.strategy.contract import SCREENABLE_CLASS_COL

    if classification.empty:
        raise ValueError("attach_screenable_class received an empty classification panel")
    cal = pd.DatetimeIndex(pd.to_datetime(np.asarray(market_dates)).ravel()).normalize()
    if len(cal) == 0:
        raise ValueError("attach_screenable_class received empty market_dates")
    if bool((cal[1:] <= cal[:-1]).any()) or not bool(cal.is_unique):
        raise ValueError("market_dates must be sorted ascending and unique")
    panel_dates = pd.to_datetime(panel[date_col], errors="coerce").dt.normalize()
    if panel_dates.isna().any():
        raise ValueError("attach_screenable_class received unparseable panel dates")
    pos = cal.get_indexer(panel_dates)
    if bool((pos < 0).any()):
        bad = panel_dates[pos < 0].drop_duplicates().head(3).dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"panel dates not in market_dates: {bad}")
    has_asof = pos > 0
    asof_dates = pd.DatetimeIndex([cal[i - 1] if i > 0 else pd.NaT for i in pos])
    class_dates = pd.to_datetime(classification["date"]).dt.normalize()
    coverage_start = class_dates.min()
    coverage_end = class_dates.max()
    start_ts = pd.Timestamp(coverage_start)
    n = len(panel)
    is_real = np.zeros(n, dtype=bool)
    if bool(has_asof.any()):
        asof_valid = asof_dates[has_asof]
        is_real[has_asof] = asof_valid >= start_ts
    no_asof = ~has_asof
    if bool(no_asof.any()):
        first_dates = panel_dates[no_asof]
        late = first_dates > start_ts
        if bool(late.any()):
            bad = first_dates[late].drop_duplicates().head(3).dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"panel row without previous trading day after coverage start: {bad}")
    real_idx = np.flatnonzero(is_real)
    if len(real_idx):
        class_date_set = set(pd.DatetimeIndex(class_dates.unique()))
        need = sorted({asof_dates[i] for i in real_idx.tolist()})
        missing = [d for d in need if d not in class_date_set]
        if missing:
            first = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in missing[:3]]
            raise ValueError(
                f"security classification gap: {len(missing)} missing as-of dates, first={first}"
            )
    verdict = np.zeros(n, dtype=bool)
    source = np.full(n, ScreenableSource.PROXY.value, dtype=object)
    n_real_absent = 0
    syms = panel[symbol_col].astype(str).to_numpy()
    if len(real_idx):
        right = pd.DataFrame({
            "_cdate": pd.to_datetime(classification["date"]).dt.normalize().to_numpy(),
            "_csym": classification["symbol"].astype(str).to_numpy(),
            "_cflag": classification["is_screenable"].astype(bool).to_numpy(),
        })
        index = right.set_index(["_cdate", "_csym"])["_cflag"]
        keys = pd.MultiIndex.from_arrays(
            [asof_dates[real_idx].to_numpy(dtype="datetime64[ns]"), syms[real_idx]],
            names=["_cdate", "_csym"],
        )
        matched = index.reindex(keys)
        matched_vals = matched.to_numpy()
        absent = pd.isna(matched_vals)
        n_real_absent = int(np.asarray(absent).sum())
        filled = np.where(absent, False, matched_vals.astype(bool))
        verdict[real_idx] = np.asarray(filled, dtype=bool)
        source[real_idx] = ScreenableSource.REAL.value
    proxy_idx = np.flatnonzero(~is_real)
    if len(proxy_idx):
        proxy_vals = proxy_screenable(pd.Series(syms[proxy_idx]))
        verdict[proxy_idx] = np.asarray(proxy_vals, dtype=bool)
    out = panel.copy()
    out[SCREENABLE_CLASS_COL] = np.asarray(verdict, dtype=bool)
    out[SCREENABLE_SOURCE_COL] = np.asarray(source, dtype=object)
    n_real = int(is_real.sum())
    n_proxy = int(n - n_real)
    prov = ScreenableClassProvenance(
        n_rows=int(n),
        n_real=n_real,
        n_proxy=n_proxy,
        n_real_absent=int(n_real_absent),
        n_non_screenable_real=int((~verdict[is_real]).sum()) if n_real else 0,
        n_non_screenable_proxy=int((~verdict[~is_real]).sum()) if n_proxy else 0,
        coverage_start=pd.Timestamp(coverage_start).strftime("%Y-%m-%d"),
        coverage_end=pd.Timestamp(coverage_end).strftime("%Y-%m-%d"),
    )
    return out, prov


def build_proxy_agreement_report(
    panel: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    screen: UniverseSpec | None = None,
) -> pd.DataFrame:
    """Measure the short-code proxy against the real verdict on every classified date.

    The proxy is only trusted where it never excludes a name the real verdict keeps; this
    report is the acceptance evidence for that claim and quantifies what the proxy misses.

    Args:
        panel: Prepared price panel (prepare_price_panel output) carrying the screen columns.
        classification: load_classification_panel output read with
            CLASSIFICATION_ATTRIBUTE_COLUMNS in addition to the verdict columns.
        screen: Class-flag-free training screen defining the "train_screen" scope; its
            exclude_non_screenable_class must be False.

    Returns:
        One row per (scope, date) plus one total row per scope (date NaT), with columns
        scope ("panel" | "train_screen"), date, n_rows, n_unclassified,
        n_real_non_screenable, n_proxy_non_screenable, n_caught, n_false_exclusion,
        n_missed, recall (n_caught / n_real_non_screenable; NaN when the denominator is 0).

    Raises:
        ValueError: When screen.exclude_non_screenable_class is True or the panel shares no
            date with the classification panel.
    """
    from src.strategy.contract import DEFAULT_UNIVERSE, select_universe

    scr = DEFAULT_UNIVERSE if screen is None else screen
    if bool(scr.exclude_non_screenable_class):
        raise ValueError("build_proxy_agreement_report requires a class-flag-free screen")
    panel_dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    class_dates = pd.to_datetime(classification["date"], errors="coerce").dt.normalize()
    shared = set(panel_dates.dropna().unique()) & set(class_dates.dropna().unique())
    if not shared:
        raise ValueError("panel shares no date with the classification panel")
    panel_syms = panel["symbol"].astype(str).to_numpy()
    panel_d = panel_dates.to_numpy(dtype="datetime64[ns]")
    right = pd.DataFrame({
        "date": class_dates.to_numpy(dtype="datetime64[ns]"),
        "symbol": classification["symbol"].astype(str).to_numpy(),
        "is_screenable": classification["is_screenable"].astype(bool).to_numpy(),
    })
    attr_cols = [c for c in CLASSIFICATION_ATTRIBUTE_COLUMNS if c in classification.columns]
    for c in attr_cols:
        right[c] = classification[c].to_numpy()
    lookup = right.set_index(["date", "symbol"])
    train_mask = np.asarray(select_universe(panel, scr), dtype=bool)
    scopes: dict[str, np.ndarray] = {
        "panel": np.ones(len(panel), dtype=bool),
        "train_screen": train_mask,
    }
    rows: list[dict[str, object]] = []
    missed_frames: list[pd.DataFrame] = []
    for scope, mask in scopes.items():
        idx = np.flatnonzero(mask)
        # One grouping pass per scope; a per-date linear scan is O(rows x dates) (hours on the 2016+ panel).
        rows_by_date = {
            key: grp.to_numpy(dtype=np.int64)
            for key, grp in pd.Series(idx).groupby(panel_d[idx], sort=False)
        }
        scope_dates = sorted(set(rows_by_date) & set(shared))
        totals = {"n_rows": 0, "n_unclassified": 0, "n_real_non": 0, "n_proxy_non": 0,
                  "n_caught": 0, "n_false": 0, "n_missed": 0}
        for d in scope_dates:
            day_idx = rows_by_date[d]
            n_rows = len(day_idx)
            keys = pd.MultiIndex.from_arrays(
                [np.full(n_rows, d), panel_syms[day_idx]], names=["date", "symbol"]
            )
            joined = lookup.reindex(keys)
            is_classified = ~joined["is_screenable"].isna().to_numpy()
            n_unclassified = int(n_rows - is_classified.sum())
            real = joined.loc[is_classified, "is_screenable"].astype(bool).to_numpy()
            syms_day = panel_syms[day_idx[is_classified]] if is_classified.any() else np.zeros(0, dtype=object)
            proxy = proxy_screenable(pd.Series(syms_day)) if len(syms_day) else np.zeros(0, dtype=bool)
            n_real_non = int((~real).sum()) if len(real) else 0
            n_proxy_non = int((~proxy).sum()) if len(proxy) else 0
            n_caught = int(((~proxy) & (~real)).sum()) if len(real) else 0
            n_false = int(((~proxy) & real).sum()) if len(real) else 0
            n_missed = int((proxy & (~real)).sum()) if len(real) else 0
            recall = (n_caught / n_real_non) if n_real_non else float("nan")
            rows.append({
                "scope": scope, "date": pd.Timestamp(d), "n_rows": n_rows,
                "n_unclassified": n_unclassified, "n_real_non_screenable": n_real_non,
                "n_proxy_non_screenable": n_proxy_non, "n_caught": n_caught,
                "n_false_exclusion": n_false, "n_missed": n_missed, "recall": recall,
            })
            totals["n_rows"] += n_rows
            totals["n_unclassified"] += n_unclassified
            totals["n_real_non"] += n_real_non
            totals["n_proxy_non"] += n_proxy_non
            totals["n_caught"] += n_caught
            totals["n_false"] += n_false
            totals["n_missed"] += n_missed
            if n_missed:
                miss_keys = keys[is_classified][(proxy & (~real))]
                miss_detail = lookup.reindex(miss_keys).reset_index()
                missed_frames.append(miss_detail)
        total_recall = (totals["n_caught"] / totals["n_real_non"]) if totals["n_real_non"] else float("nan")
        rows.append({
            "scope": scope, "date": pd.NaT, "n_rows": totals["n_rows"],
            "n_unclassified": totals["n_unclassified"],
            "n_real_non_screenable": totals["n_real_non"],
            "n_proxy_non_screenable": totals["n_proxy_non"], "n_caught": totals["n_caught"],
            "n_false_exclusion": totals["n_false"], "n_missed": totals["n_missed"],
            "recall": total_recall,
        })
        coverage_start = pd.Timestamp(min(shared)).strftime("%Y-%m-%d")
        coverage_end = pd.Timestamp(max(shared)).strftime("%Y-%m-%d")
        logger.info(
            "[DATA] stage=screenable_proxy_agreement scope=%s n_days=%d n_rows=%d "
            "n_real_non_screenable=%d n_caught=%d n_false_exclusion=%d n_missed=%d recall=%s "
            "coverage_start=%s coverage_end=%s",
            scope, len(scope_dates), totals["n_rows"], totals["n_real_non"],
            totals["n_caught"], totals["n_false"], totals["n_missed"],
            f"{total_recall:.4f}" if totals["n_real_non"] else "nan",
            coverage_start, coverage_end,
        )
    if missed_frames:
        missed = pd.concat(missed_frames, ignore_index=True)
        key_cols = [c for c in ("security_group", "security_kind", "section_type") if c in missed.columns]
        if key_cols:
            top = missed.value_counts(key_cols).head(10)
            breakdown = {tuple(k) if isinstance(k, tuple) else (k,): int(v) for k, v in top.items()}
        else:
            breakdown = {}
    else:
        breakdown = {}
    logger.info("[DATA] stage=screenable_proxy_missed_breakdown top=%s", breakdown)
    report = pd.DataFrame(rows, columns=[
        "scope", "date", "n_rows", "n_unclassified", "n_real_non_screenable",
        "n_proxy_non_screenable", "n_caught", "n_false_exclusion", "n_missed", "recall",
    ])
    return report


def main(argv: list[str] | None = None) -> None:
    """Write the proxy-agreement report and fail when the proxy ever over-excludes.

    Args:
        argv: Optional argument list (--price-history, --classification, --out).

    Raises:
        RuntimeError: After the report is written, when the "panel" total row has
            n_false_exclusion > 0.
    """
    from src import settings
    from src.data.io_utils import atomic_write_parquet
    from src.data.panel_integrity import load_price_panel
    from src.utils.cli_logging import configure_cli_logging

    configure_cli_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-history", default=str(settings.PRICE_HISTORY_PARQUET_PATH))
    parser.add_argument(
        "--classification",
        default=str(Path(settings.ALTDATA_DIR) / SECURITY_CLASSIFICATION_PARQUET_FILENAME),
    )
    parser.add_argument("--out", default="artifacts/research/screenable_proxy_agreement.parquet")
    args = parser.parse_args(argv)
    panel, _prov = load_price_panel(args.price_history)
    classification = load_classification_panel(
        args.classification,
        columns=[*CLASSIFICATION_VERDICT_COLUMNS, *CLASSIFICATION_ATTRIBUTE_COLUMNS],
    )
    report = build_proxy_agreement_report(panel, classification)
    panel_dates = set(pd.to_datetime(panel["date"]).dt.normalize().unique().tolist())
    class_dates = set(pd.to_datetime(classification["date"]).dt.normalize().unique().tolist())
    coverage_start = pd.Timestamp(classification["date"].min())
    gaps = sorted(d for d in panel_dates if d >= coverage_start and d not in class_dates)
    if gaps:
        logger.warning(
            "[DATA] stage=screenable_proxy_agreement n_coverage_gap_dates=%d first=%s",
            len(gaps),
            [pd.Timestamp(d).strftime("%Y-%m-%d") for d in gaps[:3]],
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(report, out_path)
    total = report[(report["scope"] == "panel") & (report["date"].isna())]
    if not total.empty and int(total.iloc[0]["n_false_exclusion"]) > 0:
        raise RuntimeError(
            f"proxy over-excludes: n_false_exclusion={int(total.iloc[0]['n_false_exclusion'])}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
