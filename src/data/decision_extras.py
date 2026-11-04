"""Parse decision-run raw captures into research extras; pure functions over decoded envelopes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.capture_contracts import BrokerPayload, CaptureDataset, CaptureStatus
from src.data.capture_store import CaptureStore

DECISION_EXTRAS_DATASET: str = "decision_extras"
INVESTOR_DECISION_DATASET: str = "investor_estimate_decision"
KIS_PRICE_NUMERIC_FIELDS: tuple[str, ...] = (
    "vol_tnrt",
    "prdy_vrss_vol_rate",
    "whol_loan_rmnd_rate",
    "last_ssts_cntg_qty",
    "hts_frgn_ehrt",
    "frgn_ntby_qty",
    "pgtr_ntby_qty",
    "w52_hgpr",
    "w52_lwpr",
    "d250_hgpr",
    "d250_lwpr",
    "stck_mxpr",
    "stck_llam",
    "wghn_avrg_stck_prc",
    "marg_rate",
)
KIS_PRICE_CODE_FIELDS: tuple[str, ...] = (
    "vi_cls_code",
    "ovtm_vi_cls_code",
    "short_over_yn",
    "ssts_yn",
    "mrkt_warn_cls_code",
    "invt_caful_yn",
    "temp_stop_yn",
    "iscd_stat_cls_code",
    "crdt_able_yn",
)

_DECISION_CAPTURE_REASON: str = "decision-input"
_KIWOOM_VENDOR: str = "kiwoom"
_KIS_VENDOR: str = "kis"
_KIWOOM_SCAN_ENDPOINT: str = "fluctuation-ranking"
_KIWOOM_SCAN_LIST_KEY: str = "pred_pre_flu_rt_upper"
_KIS_PRICE_ENDPOINT: str = "inquire-price"
_KIS_INVESTOR_ENDPOINT: str = "investor-trend-estimate"
_INVESTOR_QUANTITY_FIELDS: tuple[str, ...] = ("frgn_fake_ntby_qty", "orgn_fake_ntby_qty", "sum_fake_ntby_qty")
_KIWOOM_SCAN_NUMERIC_COLUMNS: tuple[str, ...] = (
    "kiwoom_cntr_str",
    "kiwoom_buy_req",
    "kiwoom_sel_req",
    "kiwoom_flu_rt",
)
_KIWOOM_SCAN_COLUMNS: tuple[str, ...] = (
    "symbol",
    "kiwoom_cntr_str",
    "kiwoom_buy_req",
    "kiwoom_sel_req",
    "kiwoom_flu_rt",
    "kiwoom_received_at",
)
_KIS_PRICE_COLUMNS: tuple[str, ...] = (
    "symbol",
    *(f"kis_{field}" for field in KIS_PRICE_NUMERIC_FIELDS),
    *(f"kis_{field}" for field in KIS_PRICE_CODE_FIELDS),
    "kis_received_at",
)
_INVESTOR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "bsop_hour_gb",
    *_INVESTOR_QUANTITY_FIELDS,
    "received_at",
)


def decision_run_id(store: CaptureStore, snapshot_date: str, *, available_by: datetime) -> str:
    """Return the latest decision-input run of a date that completed by the cutoff.

    Raises:
        ValueError: available_by is naive.
        FileNotFoundError: No COMPLETE or PARTIAL decision run with a cohort qualifies.
    """
    if available_by.utcoffset() is None:
        raise ValueError("available_by must be an aware cutoff")
    qualifying = [
        item
        for item in store.read_manifests(snapshot_date)
        if item.context.capture_reason == _DECISION_CAPTURE_REASON
        and item.status in (CaptureStatus.COMPLETE, CaptureStatus.PARTIAL)
        and item.cohort is not None
        and item.completed_at <= available_by
    ]
    if not qualifying:
        raise FileNotFoundError(f"no qualifying decision run: {snapshot_date!r}")
    return max(qualifying, key=lambda item: (item.completed_at, item.context.run_id)).context.run_id


def kiwoom_scan_rows(envelopes: Iterable[tuple[str, BrokerPayload]]) -> pd.DataFrame:
    """Flatten Kiwoom fluctuation-ranking pages to one row per symbol, keeping the earliest page."""
    best: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
    for _path, envelope in envelopes:
        if not _is_route(envelope, _KIWOOM_VENDOR, CaptureDataset.SCAN, _KIWOOM_SCAN_ENDPOINT):
            continue
        order = (int(envelope["page_index"]), int(envelope["attempt_index"]))
        for raw in _payload(envelope).get(_KIWOOM_SCAN_LIST_KEY) or []:
            if not isinstance(raw, Mapping):
                continue
            symbol = _symbol_or_none(str(raw.get("stk_cd") or "").split("_")[0])
            if symbol is None or (symbol in best and best[symbol][0] <= order):
                continue
            best[symbol] = (
                order,
                {
                    "symbol": symbol,
                    "kiwoom_cntr_str": _float_or_none(raw.get("cntr_str")),
                    "kiwoom_buy_req": _float_or_none(raw.get("buy_req")),
                    "kiwoom_sel_req": _float_or_none(raw.get("sel_req")),
                    "kiwoom_flu_rt": _float_or_none(raw.get("flu_rt")),
                    "kiwoom_received_at": _received_at(envelope),
                },
            )
    frame = pd.DataFrame([row for _order, row in best.values()], columns=list(_KIWOOM_SCAN_COLUMNS))
    return frame.astype(dict.fromkeys(_KIWOOM_SCAN_NUMERIC_COLUMNS, "float64"))


def kis_price_rows(envelopes: Iterable[tuple[str, BrokerPayload]]) -> pd.DataFrame:
    """Flatten KIS inquire-price responses to one row per symbol, keeping the last successful attempt."""
    best: dict[str, tuple[int, dict[str, Any]]] = {}
    for _path, envelope in envelopes:
        if not _is_route(envelope, _KIS_VENDOR, CaptureDataset.PRICE, _KIS_PRICE_ENDPOINT):
            continue
        symbol = _symbol_from_context(envelope)
        output = _payload(envelope).get("output")
        if symbol is None or not isinstance(output, Mapping):
            continue
        attempt = int(envelope["attempt_index"])
        if symbol in best and best[symbol][0] > attempt:
            continue
        row: dict[str, Any] = {"symbol": symbol}
        row.update({f"kis_{field}": _float_or_none(output.get(field)) for field in KIS_PRICE_NUMERIC_FIELDS})
        row.update({f"kis_{field}": _text_or_none(output.get(field)) for field in KIS_PRICE_CODE_FIELDS})
        row["kis_received_at"] = _received_at(envelope)
        best[symbol] = (attempt, row)
    frame = pd.DataFrame([row for _attempt, row in best.values()], columns=list(_KIS_PRICE_COLUMNS))
    return frame.astype(dict.fromkeys((f"kis_{field}" for field in KIS_PRICE_NUMERIC_FIELDS), "float64"))


def investor_series_rows(envelopes: Iterable[tuple[str, BrokerPayload]]) -> pd.DataFrame:
    """Flatten KIS investor-trend-estimate series to one row per (symbol, hour bucket), keeping the last attempt."""
    best: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for _path, envelope in envelopes:
        if not _is_route(envelope, _KIS_VENDOR, CaptureDataset.INVESTOR_ESTIMATE, _KIS_INVESTOR_ENDPOINT):
            continue
        symbol = _symbol_from_context(envelope)
        series = _payload(envelope).get("output2")
        if symbol is None or not isinstance(series, list):
            continue
        attempt = int(envelope["attempt_index"])
        received = _received_at(envelope)
        for raw in series:
            if not isinstance(raw, Mapping):
                continue
            hour = _text_or_none(raw.get("bsop_hour_gb"))
            if hour is None or ((symbol, hour) in best and best[(symbol, hour)][0] > attempt):
                continue
            row: dict[str, Any] = {"symbol": symbol, "bsop_hour_gb": hour}
            row.update({field: _int_or_none(raw.get(field)) for field in _INVESTOR_QUANTITY_FIELDS})
            row["received_at"] = received
            best[(symbol, hour)] = (attempt, row)
    frame = pd.DataFrame([row for _attempt, row in best.values()], columns=list(_INVESTOR_COLUMNS))
    return frame.astype(dict.fromkeys(_INVESTOR_QUANTITY_FIELDS, "Int64"))


def build_decision_extras(
    cohort_symbols: Sequence[str], scan: pd.DataFrame, price: pd.DataFrame, run_id: str
) -> pd.DataFrame:
    """Left-join scan and quote extras onto the cohort, one row per cohort member in cohort order."""
    base = pd.DataFrame({"symbol": list(cohort_symbols)}, columns=["symbol"])
    joined = base.merge(scan, on="symbol", how="left", validate="many_to_one").merge(
        price, on="symbol", how="left", validate="many_to_one"
    )
    joined["run_id"] = run_id
    return joined


def _payload(envelope: BrokerPayload) -> Mapping[str, Any]:
    payload = envelope.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def _is_route(envelope: BrokerPayload, vendor: str, dataset: CaptureDataset, endpoint: str) -> bool:
    context: Mapping[str, Any] = envelope["context"]
    return context.get("vendor") == vendor and context.get("dataset") == dataset.value and context.get("endpoint") == endpoint


def _symbol_from_context(envelope: BrokerPayload) -> str | None:
    return _symbol_or_none(envelope["context"].get("symbol"))


def _symbol_or_none(raw: Any) -> str | None:
    text = "" if raw is None else str(raw).strip()
    return text.zfill(6) if text else None


def _float_or_none(raw: Any) -> float | None:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int_or_none(raw: Any) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _text_or_none(raw: Any) -> str | None:
    text = "" if raw is None else str(raw).strip()
    return text or None


def _received_at(envelope: BrokerPayload) -> datetime:
    return datetime.fromisoformat(str(envelope["received_at"]))
