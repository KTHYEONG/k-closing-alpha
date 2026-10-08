"""Invariant guards for NXT consolidated-tape calibration pairs."""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime

import pandas as pd

from src.backfill.intraday.nxt_calibration_pairs import (
    CALIBRATION_TABLE_COLUMNS,
    CalibrationSample,
    build_calibration_table,
    collect_calibration_pairs,
)
from src.config.collection import CollectionSettings
from src.data.capture_store import CaptureStore

_DAY = "2024-02-28"


def _hhmmss(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}{minute_of_day % 60:02d}00"


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


def _toss_grid(day: str, volume: int = 150) -> list[dict]:
    out = [_candle(day, _hhmmss(m), volume=volume) for m in range(9 * 60 + 1, 15 * 60 + 31)]
    assert len(out) == 390
    return out


def _kis_frame(day: str, symbol: str, vendor: str = "kis", volume: int = 100) -> pd.DataFrame:
    stamps = [int(_hhmmss(m)) for m in range(9 * 60, 15 * 60 + 20)] + [153000]
    volumes = [volume] * (len(stamps) - 1) + [1000]
    return pd.DataFrame(
        {
            "snapshot_date": [day] * len(stamps),
            "symbol": [symbol] * len(stamps),
            "ts_hms": stamps,
            "open": [70000] * len(stamps),
            "high": [70100] * len(stamps),
            "low": [69900] * len(stamps),
            "close": [70000] * len(stamps),
            "volume": volumes,
            "value_krw": [70000 * v for v in volumes],
            "has_trade": [True] * len(stamps),
            "vendor": [vendor] * len(stamps),
        }
    )


class _GridToss:
    def __init__(self, grids: dict[str, list[dict]]) -> None:
        self._grids = dict(grids)
        self.calls: list[str] = []

    async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
        self.calls.append(str(symbol))
        grid = self._grids.get(str(symbol), [])
        bound = datetime.fromisoformat(str(before)) if before else None
        out = []
        for candle in grid:
            current = datetime.fromisoformat(str(candle.get("timestamp", "")))
            if bound is None or current <= bound:
                out.append(candle)
                if len(out) >= int(count):
                    break
        return {"result": {"candles": out}}


def _profile(tmp_path) -> CollectionSettings:
    return CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture",
        COLLECTION_VERIFIED_CHART_ROUTES={"toss:toss-candles": "KRX"},
        _env_file=None,
    )


def _eod_for(day: str, symbol: str) -> float:
    return float(380 * 100 + 1000)


def test_truth_only_sampling(tmp_path) -> None:
    """Only raw-basis full-session kis days are paired; ls and adjusted days are skipped."""
    samples = [
        CalibrationSample(_DAY, "000001", _kis_frame(_DAY, "000001"), _eod_for(_DAY, "000001"), True),
        CalibrationSample(_DAY, "000002", _kis_frame(_DAY, "000002", vendor="ls"), _eod_for(_DAY, "000002"), True),
        CalibrationSample(_DAY, "000003", _kis_frame(_DAY, "000003"), _eod_for(_DAY, "000003"), False),
    ]
    client = _GridToss({s: list(reversed(_toss_grid(_DAY))) for s in ("000001", "000002", "000003")})
    evidence = CaptureStore(tmp_path / "evidence")
    pairs = asyncio.run(
        collect_calibration_pairs(
            samples=samples, client=client, session=object(), profile=_profile(tmp_path),
            evidence_store=evidence, run_id="calib-test",
        )
    )
    assert pairs["symbol"].tolist() == ["000001"]
    assert set(client.calls) == {"000001"}


def test_gate_accepted_days_are_not_paired(tmp_path) -> None:
    """A KRX-only Toss tape (gate accepted) carries no NXT signal and is skipped."""
    samples = [CalibrationSample(_DAY, "000001", _kis_frame(_DAY, "000001"), 39000.0, True)]
    client = _GridToss({"000001": list(reversed(_toss_grid(_DAY, volume=100)))})
    evidence = CaptureStore(tmp_path / "evidence")
    pairs = asyncio.run(
        collect_calibration_pairs(
            samples=samples, client=client, session=object(), profile=_profile(tmp_path),
            evidence_store=evidence, run_id="calib-test",
        )
    )
    assert pairs.empty


def test_identity_residual_enforced(tmp_path) -> None:
    """A pair whose EOD differs from V_krx + A beyond tolerance is excluded and counted."""
    samples = [CalibrationSample(_DAY, "000001", _kis_frame(_DAY, "000001"), _eod_for(_DAY, "000001"), True)]
    client = _GridToss({"000001": list(reversed(_toss_grid(_DAY)))})
    evidence = CaptureStore(tmp_path / "evidence")
    pairs = asyncio.run(
        collect_calibration_pairs(
            samples=samples, client=client, session=object(), profile=_profile(tmp_path),
            evidence_store=evidence, run_id="calib-test",
        )
    )
    assert len(pairs) == 1
    good = build_calibration_table(pairs)
    assert list(good.columns) == list(CALIBRATION_TABLE_COLUMNS)
    assert len(good) == 1
    assert good.attrs["n_identity_excluded"] == 0
    row = good.iloc[0]
    assert row["v_krx_1520"] == 380 * 100
    assert row["auction_volume"] == 1000
    assert row["v_cons_full"] == 390 * 150
    assert row["identity_residual"] == 0.0

    bad = pairs.copy()
    bad.loc[0, "eod_volume"] = _eod_for(_DAY, "000001") * 1.2
    filtered = build_calibration_table(bad)
    assert filtered.empty
    assert filtered.attrs["n_identity_excluded"] == 1
    assert list(filtered.columns) == list(CALIBRATION_TABLE_COLUMNS)


def test_no_production_side_effects(tmp_path, monkeypatch) -> None:
    """Nothing is written to production roots; the temporary evidence dir is removed after use."""
    from src.data import intraday_store

    production = tmp_path / "production"
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", production, raising=False)
    samples = [CalibrationSample(_DAY, "000001", _kis_frame(_DAY, "000001"), _eod_for(_DAY, "000001"), True)]
    client = _GridToss({"000001": list(reversed(_toss_grid(_DAY)))})
    evidence_dir = tmp_path / "evidence"
    evidence = CaptureStore(evidence_dir)
    pairs = asyncio.run(
        collect_calibration_pairs(
            samples=samples, client=client, session=object(), profile=_profile(tmp_path),
            evidence_store=evidence, run_id="calib-test",
        )
    )
    assert len(pairs) == 1
    assert not production.exists() or not any(production.rglob("*"))
    assert any(evidence_dir.rglob("*"))
    shutil.rmtree(evidence_dir, ignore_errors=True)
    assert not evidence_dir.exists()


def test_cutoffs_exclude_the_post_decision_minute_on_both_sides() -> None:
    """Toss label 15:21 and the KIS 15:20 start-stamped bar start after the 15:20 decision cutoff."""
    from src.backfill.intraday.nxt_calibration_pairs import (
        _CONS_CONTINUOUS_CUTOFF_HHMMSS,
        _KRX_CONTINUOUS_CUTOFF_HHMMSS,
        _volumes_at_or_below,
    )

    toss = pd.DataFrame({"ts_hms": [151900, 152000, 152100], "volume": [1, 2, 4]})
    kis = pd.DataFrame({"ts_hms": [151800, 151900, 152000], "volume": [1, 2, 4]})
    assert _volumes_at_or_below(toss, _CONS_CONTINUOUS_CUTOFF_HHMMSS) == 3
    assert _volumes_at_or_below(kis, _KRX_CONTINUOUS_CUTOFF_HHMMSS) == 3
