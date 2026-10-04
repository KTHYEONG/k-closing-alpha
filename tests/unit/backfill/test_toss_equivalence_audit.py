"""Invariant guards for the Toss/KIS equivalence audit."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd

from src.backfill.intraday.toss_equivalence_audit import (
    _exit_code_for_report,
    compare_toss_to_kis,
    run_equivalence_audit,
    sample_kis_symbol_days,
)
from src.backfill.intraday.toss_regular import TossBasisVerdict
from src.config.collection import CollectionSettings
from src.data.capture_store import CaptureStore

_DAY = "2024-02-28"


def _kis_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_date": _DAY,
                "symbol": "005930",
                "ts_hms": ts,
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": vol,
                "value_krw": int(c * vol),
                "has_trade": vol > 0,
                "vendor": "kis",
            }
            for ts, o, h, low, c, vol in rows
        ]
    )


def _toss_frame(rows: list[tuple]) -> pd.DataFrame:
    out = _kis_frame(rows)
    out["vendor"] = "toss"
    return out


def _accept(volume_ratio: float = 1.0) -> TossBasisVerdict:
    return TossBasisVerdict(accepted=True, reason="ok", volume_ratio=volume_ratio)


def test_label_alignment_counts_carry_minutes_without_penalty() -> None:
    kis = _kis_frame([
        (90000, 70000, 70100, 69900, 70050, 100),
        (90100, 70050, 70150, 69950, 70100, 120),
    ])
    toss = _toss_frame([
        (90100, 70000, 70100, 69900, 70050, 100),
        (90200, 70050, 70150, 69950, 70100, 120),
        (90300, 70100, 70100, 70100, 70100, 0),
    ])
    metrics = compare_toss_to_kis(kis, toss, verdict=_accept())
    assert metrics.ohlc_exact_share == 1.0
    assert metrics.volume_exact_share == 1.0
    assert metrics.kis_only_bars == 0
    assert metrics.toss_only_bars == 1
    assert metrics.accepted is True


def test_false_accept_is_surfaced_in_exit_code() -> None:
    kis = _kis_frame([(90000, 70000, 70100, 69900, 70050, 100)])
    toss = _toss_frame([(90100, 70000, 70100, 69900, 70099, 100)])
    metrics = compare_toss_to_kis(kis, toss, verdict=_accept())
    assert metrics.ohlc_exact_share == 0.0
    assert metrics.volume_exact_share == 1.0


def test_kis_only_bars_are_a_defect() -> None:
    kis = _kis_frame([
        (90000, 70000, 70100, 69900, 70050, 100),
        (90100, 70050, 70150, 69950, 70100, 120),
    ])
    toss = _toss_frame([(90100, 70000, 70100, 69900, 70050, 100)])
    metrics = compare_toss_to_kis(kis, toss, verdict=_accept())
    assert metrics.kis_only_bars == 1
    assert metrics.ohlc_exact_share == 1.0


def _hhmmss(minute_of_day: int) -> int:
    return int(f"{minute_of_day // 60:02d}{minute_of_day % 60:02d}00")


def _full_kis_day(symbol: str, day: str, vendor: str = "kis") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "snapshot_date": day,
            "symbol": symbol,
            "ts_hms": _hhmmss(minute),
            "open": 70000,
            "high": 70100,
            "low": 69900,
            "close": 70050,
            "volume": 10,
            "value_krw": 700500,
            "has_trade": True,
            "vendor": vendor,
        }
        for minute in range(9 * 60, 15 * 60 + 20)
    ])


def _stored(days: dict[str, pd.DataFrame]):
    def _load(day: str) -> pd.DataFrame:
        frame = days.get(day)
        return frame.copy() if frame is not None else pd.DataFrame()
    return _load


def test_seeded_sampling_only_takes_full_session_kis_days() -> None:
    days = {
        "2024-02-26": pd.concat(
            [_full_kis_day("005930", "2024-02-26"), _full_kis_day("000660", "2024-02-26", vendor="ls")],
            ignore_index=True,
        ),
        "2024-02-27": _full_kis_day("005930", "2024-02-27").iloc[:10],
        "2024-02-28": pd.concat(
            [_full_kis_day("005930", "2024-02-28"), _full_kis_day("000660", "2024-02-28")],
            ignore_index=True,
        ),
    }
    kw = {
        "start": "2024-02-26",
        "end": "2024-02-28",
        "n": 10,
        "stored_loader": _stored(days),
        "calendar": ["2024-02-26", "2024-02-27", "2024-02-28"],
    }
    first = sample_kis_symbol_days(seed=7, **kw)
    second = sample_kis_symbol_days(seed=7, **kw)
    assert first == second
    assert first == [("2024-02-26", "005930"), ("2024-02-28", "000660"), ("2024-02-28", "005930")]
    third = sample_kis_symbol_days(seed=8, n=2, stored_loader=_stored(days), calendar=list(kw["calendar"]),
                                   start=kw["start"], end=kw["end"])
    assert len(third) == 2
    assert set(third) <= set(first)


def test_seeded_sampling_differs_across_seeds() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for d in range(1, 11):
        day = f"2024-02-{d:02d}"
        frames[day] = pd.concat(
            [_full_kis_day(f"{i:06d}", day) for i in range(1, 11)], ignore_index=True
        )
    base = sample_kis_symbol_days(start="2024-02-01", end="2024-02-10", n=5, seed=1,
                                  stored_loader=_stored(frames),
                                  calendar=[f"2024-02-{d:02d}" for d in range(1, 11)])
    assert len(base) == 5
    assert any(
        sample_kis_symbol_days(start="2024-02-01", end="2024-02-10", n=5, seed=seed,
                               stored_loader=_stored(frames),
                               calendar=[f"2024-02-{d:02d}" for d in range(1, 11)]) != base
        for seed in range(2, 20)
    )


class _GridToss:
    """Newest-first full-session grid with call tracking."""

    def __init__(self, grids: dict[str, list[dict]]) -> None:
        self._grids = grids
        self.calls: list[dict] = []

    async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
        self.calls.append({"symbol": symbol, "count": count, "before": before, "adjusted": adjusted})
        grid = self._grids[str(symbol)]
        bound = datetime.fromisoformat(str(before)) if before else None
        out = []
        for candle in grid:
            current = datetime.fromisoformat(str(candle["timestamp"]))
            if bound is None or current <= bound:
                out.append(candle)
                if len(out) >= int(count):
                    break
        return {"result": {"candles": out}}


def _candle(day: str, hhmmss: str, volume: int = 100, price: str = "70000") -> dict:
    return {
        "timestamp": f"{day}T{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}.000+09:00",
        "openPrice": price,
        "highPrice": price,
        "lowPrice": price,
        "closePrice": price,
        "volume": str(volume),
        "currency": "KRW",
    }


def _toss_grid(day: str, volume: int = 100, close_price: str = "70000") -> list[dict]:
    out = [
        _candle(day, f"{m // 60:02d}{m % 60:02d}00", volume=volume, price=close_price)
        for m in range(9 * 60 + 1, 15 * 60 + 31)
    ]
    assert len(out) == 390
    return list(reversed(out))


def _kis_grid_frame(day: str, symbol: str, close_price: str = "70000") -> pd.DataFrame:
    price = int(close_price)
    return pd.DataFrame([
        {
            "snapshot_date": day,
            "symbol": symbol,
            "ts_hms": int(f"{m // 60:02d}{m % 60:02d}00"),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100,
            "value_krw": price * 100,
            "has_trade": True,
            "vendor": "kis",
        }
        for m in range(9 * 60, 15 * 60 + 30)
    ])


def _profile(tmp_path) -> CollectionSettings:
    return CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture",
        COLLECTION_VERIFIED_CHART_ROUTES={"toss:toss-candles": "KRX"},
    )


def _audit(samples, client, tmp_path, kis_by_symbol, eod_volume: float = 39000.0):
    profile = _profile(tmp_path)
    evidence = CaptureStore(tmp_path / "evidence")
    eod_volumes = {(day, symbol): float(eod_volume) for day, symbol in samples}

    def _kis_loader(day: str, symbol: str) -> pd.DataFrame:
        return kis_by_symbol[(day, symbol)]

    report = asyncio.run(
        run_equivalence_audit(
            samples=samples,
            client=client,
            session=object(),
            profile=profile,
            eod_volumes=eod_volumes,
            kis_loader=_kis_loader,
            evidence_store=evidence,
            run_id="audit-test",
        )
    )
    return report, evidence, client


def test_audit_accepts_identical_days_with_exact_budget(tmp_path) -> None:
    day, symbol = _DAY, "005930"
    client = _GridToss({symbol: _toss_grid(day)})
    report, _evidence, client = _audit(
        [(day, symbol)], client, tmp_path, {(day, symbol): _kis_grid_frame(day, symbol)}
    )
    assert report.n_symbol_days == 1
    assert report.accepted_share == 1.0
    assert report.accepted_mismatch_share == 0.0
    assert report.accepted_kis_only_bars == 0
    assert report.volume_ratio_quantiles["p50"] == 1.0
    assert len(client.calls) == 2
    assert _exit_code_for_report(report, threshold=0.01) == 0


def test_audit_request_budget_scales_with_samples(tmp_path) -> None:
    days = ["2024-02-26", "2024-02-27", "2024-02-28"]
    symbols = ["005930", "000660", "005380"]
    samples = [(day, symbols[i]) for i, day in enumerate(days)]
    grids = {symbol: _toss_grid(day) for (day, symbol) in samples for symbol in [symbol]}
    kis = {(day, symbol): _kis_grid_frame(day, symbol) for day, symbol in samples}
    client = _GridToss(grids)
    report, _evidence, client = _audit(samples, client, tmp_path, kis)
    assert report.n_symbol_days == 3
    assert len(client.calls) == 6


def test_audit_false_accept_fails_exit_code(tmp_path) -> None:
    day, symbol = _DAY, "005930"
    client = _GridToss({symbol: _toss_grid(day, close_price="70100")})
    report, _evidence, _client = _audit(
        [(day, symbol)], client, tmp_path, {(day, symbol): _kis_grid_frame(day, symbol)}
    )
    assert report.accepted_share == 1.0
    assert report.accepted_mismatch_share == 1.0
    assert _exit_code_for_report(report, threshold=0.01) == 1


def test_audit_kis_only_bars_fail_exit_code(tmp_path) -> None:
    day, symbol = _DAY, "005930"
    grid = _toss_grid(day)
    grid_missing = [c for c in grid if not c["timestamp"].startswith(f"{day}T10:00")]
    assert len(grid_missing) < len(grid)
    client = _GridToss({symbol: grid_missing})
    kis = _kis_grid_frame(day, symbol)
    report, _evidence, _client = _audit(
        [(day, symbol)], client, tmp_path, {(day, symbol): kis}
    )
    assert report.accepted_share == 1.0
    assert report.accepted_kis_only_bars > 0
    assert _exit_code_for_report(report, threshold=1.0) == 1


def test_audit_writes_only_to_explicit_evidence_store(tmp_path, monkeypatch) -> None:
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path / "prod-history", raising=False)
    day, symbol = _DAY, "005930"
    client = _GridToss({symbol: _toss_grid(day)})
    evidence_root = tmp_path / "explicit-evidence"
    profile = _profile(tmp_path)
    evidence = CaptureStore(evidence_root)
    report = asyncio.run(
        run_equivalence_audit(
            samples=[(day, symbol)],
            client=client,
            session=object(),
            profile=profile,
            eod_volumes={(day, symbol): 39000.0},
            kis_loader=lambda d, s: _kis_grid_frame(d, s),
            evidence_store=evidence,
            run_id="audit-isolation",
        )
    )
    assert report.n_symbol_days == 1
    assert not (tmp_path / "prod-history").exists()
    assert not (tmp_path / "capture").exists()
    assert any(evidence_root.rglob("*"))


def test_audit_without_any_accepted_day_fails_closed(tmp_path) -> None:
    """A dead vendor (every sample fails) or an empty sample set is missing evidence, never a pass."""
    day, symbol = _DAY, "005930"
    dead = _GridToss({})
    report, _evidence, _client = _audit(
        [(day, symbol)], dead, tmp_path, {(day, symbol): _kis_grid_frame(day, symbol)}
    )
    assert report.accepted_share == 0.0 and report.accepted_mismatch_share == 0.0
    assert _exit_code_for_report(report, threshold=1.0) == 1
    empty, _e, _c = _audit([], _GridToss({}), tmp_path, {})
    assert empty.n_symbol_days == 0
    assert _exit_code_for_report(empty, threshold=1.0) == 1


def test_kis_closing_auction_bar_is_not_a_toss_defect() -> None:
    """The KIS 15:30 start-stamped auction print has no Toss counterpart and must not count as kis_only."""
    kis = _kis_frame([
        (152900, 70000, 70100, 69900, 70050, 100),
        (153000, 70050, 70050, 70050, 70050, 5000),
    ])
    toss = _toss_frame([(153000, 70000, 70100, 69900, 70050, 100)])
    metrics = compare_toss_to_kis(kis, toss, verdict=_accept(0.995))
    assert metrics.kis_only_bars == 0
    assert metrics.kis_bars == 1
    assert metrics.ohlc_exact_share == 1.0


def test_reported_volume_ratio_is_the_gate_ratio_against_eod() -> None:
    """The ratio reported per symbol-day is the gate's Toss/EOD ratio, not a Toss/KIS ratio."""
    kis = _kis_frame([(90000, 70000, 70100, 69900, 70050, 100)])
    toss = _toss_frame([(90100, 70000, 70100, 69900, 70050, 100)])
    assert compare_toss_to_kis(kis, toss, verdict=_accept(0.987)).volume_ratio == 0.987
    empty = compare_toss_to_kis(kis, _toss_frame([]), verdict=TossBasisVerdict(False, "toss_basis_unverifiable", None))
    assert empty.volume_ratio is None


def test_sampling_skips_adjusted_or_unknown_basis_symbol_days() -> None:
    """Corporate-action adjusted KIS history is not comparable with the raw Toss tape and is never sampled."""
    days = {"2024-02-26": pd.concat(
        [_full_kis_day("005930", "2024-02-26"), _full_kis_day("000660", "2024-02-26")], ignore_index=True
    )}
    picked = sample_kis_symbol_days(
        start="2024-02-26", end="2024-02-26", n=10, seed=7, stored_loader=_stored(days),
        calendar=["2024-02-26"], is_raw_basis=lambda day, symbol: symbol == "005930",
    )
    assert picked == [("2024-02-26", "005930")]
