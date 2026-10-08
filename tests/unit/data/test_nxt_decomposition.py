"""Invariant guards for the causal consolidated-to-KRX decomposition estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.nxt_decomposition import (
    DecompositionConfig,
    fit_decomposition,
    predict_share,
    reconstruct_krx_bars,
    share_proxy_history,
)

_SHARE = 0.68
_AUCTION = 365.0


def _config(**overrides: object) -> DecompositionConfig:
    base: dict[str, object] = {
        "ewma_alpha": 0.5,
        "min_prior_days": 2,
        "max_gap_days": 30,
        "auction_fraction_mean": 0.0365,
        "bias_correction": 1.0,
        "volume_rel_err_p90": 0.144,
        "close_bp_err_p90": 11.9,
        "calibrated_through": "2026-09-30",
        "fit_start": "2026-09-01",
        "holdout_start": "2026-08-25",
        "holdout_end": "2026-08-31",
    }
    base.update(overrides)
    return DecompositionConfig(**base)  # type: ignore[arg-type]


def _fit_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "alphas": [0.3, 0.5, 0.7],
        "structures": [(2, 30)],
        "identity_tolerance": 0.05,
        "holdout_fraction": 0.25,
        "min_holdout_days": 2,
        "max_selection_rel_err_p90": 0.5,
        "table_sha256": "test-table-sha256",
    }
    base.update(overrides)
    return base


def _cal_row(
    day: str,
    symbol: str,
    share: float,
    v_cons: float = 10000.0,
    eod_mult: float = 1.0,
    auction: float = _AUCTION,
) -> dict[str, object]:
    v_krx = share * v_cons
    return {
        "date": day,
        "symbol": symbol,
        "v_krx_1520": v_krx,
        "auction_volume": auction,
        "v_cons_1520": v_cons,
        "eod_volume": (v_krx + auction) * eod_mult,
        "close_krx_1519": 50000.0,
        "close_cons_1519": 50000.0,
    }


def _days(start: str, n: int) -> list[str]:
    return pd.date_range(start, periods=n, freq="B").strftime("%Y-%m-%d").tolist()


def _constant_calibration(days: list[str], share: float = _SHARE) -> pd.DataFrame:
    rows = [_cal_row(d, symbol, share) for d in days for symbol in ("000001", "000002")]
    return pd.DataFrame(rows)


def _test_ewma(values: list[float], alpha: float) -> float:
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
    return ema


def _proxy_frame(frame: pd.DataFrame, abar: float) -> dict[str, pd.DataFrame]:
    """Per-symbol eligible rows with production proxies, oldest first."""
    out: dict[str, pd.DataFrame] = {}
    for symbol, group in frame.groupby("symbol", sort=True):
        ordered = group.sort_values("ordinal", kind="stable").reset_index(drop=True).copy()
        ordered["proxy"] = ordered["eod_volume"] * (1.0 - abar) / ordered["v_cons_1520"]
        out[str(symbol)] = ordered
    return out


def _history_series(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        group["proxy"].to_numpy(dtype=np.float64),
        index=pd.Index(group["day"].tolist(), dtype=object),
        dtype=np.float64,
    )


def _selection_days(fit_start: str, fit_end: str, frame: pd.DataFrame) -> set[str]:
    window = sorted(
        {str(d) for d in frame["day"].tolist() if str(fit_start) <= str(d) <= str(fit_end)}
    )
    return set(window[len(window) // 2:])


# ---------------------------------------------------------------------------
# Decomposition invariant scenarios
# ---------------------------------------------------------------------------


def test_fit_scoring_equals_production_predictor() -> None:
    from src.data.nxt_decomposition import _eligible_frame

    early = _days("2026-01-05", 6)
    late = _days("2026-02-23", 8)
    shares = [0.60, 0.66, 0.62, 0.70, 0.64, 0.72, 0.61, 0.69, 0.65, 0.71, 0.63, 0.67, 0.60, 0.73]
    rows: list[dict[str, object]] = []
    for i, day in enumerate(early + late):
        rows.append(_cal_row(day, "000001", shares[i]))
        if i % 2 == 0:
            rows.append(_cal_row(day, "000002", shares[(i + 5) % len(shares)]))
    calibration = pd.DataFrame(rows)
    fit = fit_decomposition(calibration, **_fit_kwargs())  # type: ignore[arg-type]
    cfg = fit.config
    diag = fit.diagnostics
    chosen = next(p for p in diag.frontier if (p.ewma_alpha, p.min_prior_days, p.max_gap_days) == (cfg.ewma_alpha, cfg.min_prior_days, cfg.max_gap_days))

    kept = _eligible_frame(calibration, 0.05, caller="test")
    fit_rows = kept[(kept["day"] >= cfg.fit_start) & (kept["day"] <= cfg.calibrated_through)].reset_index(drop=True)
    holdout_rows = kept[(kept["day"] >= cfg.holdout_start) & (kept["day"] <= cfg.holdout_end)].reset_index(drop=True)
    by_symbol = _proxy_frame(fit_rows, cfg.auction_fraction_mean)
    holdout_by_symbol = _proxy_frame(holdout_rows, cfg.auction_fraction_mean)
    selection = _selection_days(cfg.fit_start, cfg.calibrated_through, fit_rows)

    scope: list[tuple[str, str, float]] = []
    for symbol, group in by_symbol.items():
        scope.extend(
            (symbol, str(row["day"]), float(row["true_share"]))
            for row in group.to_dict(orient="records")
            if str(row["day"]) in selection
        )
    raw_pairs: list[tuple[float, float]] = []
    for symbol, day, true in scope:
        group = by_symbol[symbol]
        decision = pd.Timestamp(day).normalize()
        priors = [
            float(proxy)
            for row_day, proxy in zip(group["day"].tolist(), group["proxy"].tolist(), strict=True)
            if (decision - pd.Timestamp(row_day).normalize()).days >= 1
            and (decision - pd.Timestamp(row_day).normalize()).days <= cfg.max_gap_days
            and np.isfinite(float(proxy))
        ]
        if len(priors) < cfg.min_prior_days:
            history = _history_series(group)
            assert predict_share(history, day, config=cfg) is None
            continue
        raw = _test_ewma(priors, cfg.ewma_alpha)
        assert np.isfinite(raw) and raw > 0.0
        raw_pairs.append((raw, true))
        history = _history_series(group)
        predicted = predict_share(history, day, config=cfg)
        assert predicted is not None
        assert predicted == pytest.approx(raw * cfg.bias_correction, rel=1e-12)
    assert raw_pairs, "selection produced no prediction to conform"
    test_bias = float(np.median([true / raw for raw, true in raw_pairs]))
    assert test_bias == pytest.approx(cfg.bias_correction, rel=1e-12)
    assert chosen.coverage == pytest.approx(len(raw_pairs) / len(scope), rel=1e-12)

    holdout_errors: list[float] = []
    for group in holdout_by_symbol.values():
        ordered = group.sort_values("ordinal", kind="stable").reset_index(drop=True)
        for row in ordered.to_dict(orient="records"):
            day = str(row["day"])
            decision = pd.Timestamp(day).normalize()
            priors = [
                float(proxy)
                for row_day, proxy in zip(ordered["day"].tolist(), ordered["proxy"].tolist(), strict=True)
                if (decision - pd.Timestamp(row_day).normalize()).days >= 1
                and (decision - pd.Timestamp(row_day).normalize()).days <= cfg.max_gap_days
                and np.isfinite(float(proxy))
            ]
            history = _history_series(ordered)
            predicted = predict_share(history, day, config=cfg)
            if len(priors) < cfg.min_prior_days:
                assert predicted is None
                continue
            raw = _test_ewma(priors, cfg.ewma_alpha)
            corrected = raw * cfg.bias_correction
            if not np.isfinite(corrected) or corrected <= 0.0 or corrected > 1.0:
                assert predicted is None
                continue
            assert predicted is not None
            assert predicted == pytest.approx(corrected, rel=1e-12)
            holdout_errors.append(abs(corrected - float(row["true_share"])) / float(row["true_share"]))
    assert holdout_errors, "holdout produced no prediction to conform"
    assert diag.holdout_rel_err_p90 == pytest.approx(float(np.quantile(holdout_errors, 0.9)), rel=1e-12)
    assert diag.holdout_coverage == pytest.approx(len(holdout_errors) / len(holdout_rows), rel=1e-12)


def test_holdout_isolation_fit_side() -> None:
    days = _days("2026-01-05", 10)
    calibration = _constant_calibration(days)
    before = fit_decomposition(calibration, **_fit_kwargs())  # type: ignore[arg-type]

    rescaled = calibration.copy()
    holdout_days = set(_days("2026-01-05", 3))
    mask = rescaled["date"].isin(holdout_days)
    for column in ("v_krx_1520", "auction_volume", "v_cons_1520", "eod_volume"):
        rescaled.loc[mask, column] = rescaled.loc[mask, column] * 1.37
    for column in ("close_krx_1519", "close_cons_1519"):
        rescaled.loc[mask, column] = rescaled.loc[mask, column] + 1000.0
    after = fit_decomposition(rescaled, **_fit_kwargs())  # type: ignore[arg-type]

    assert after.config.auction_fraction_mean == before.config.auction_fraction_mean
    assert after.config.bias_correction == before.config.bias_correction
    assert after.config.ewma_alpha == before.config.ewma_alpha
    assert (after.config.min_prior_days, after.config.max_gap_days) == (
        before.config.min_prior_days,
        before.config.max_gap_days,
    )
    assert after.config.calibrated_through == before.config.calibrated_through
    assert after.config.fit_start == before.config.fit_start


def test_holdout_isolation_holdout_side() -> None:
    days = _days("2026-01-05", 10)
    calibration = _constant_calibration(days)
    before = fit_decomposition(calibration, **_fit_kwargs())  # type: ignore[arg-type]
    cfg = before.config

    def holdout_predictions(frame: pd.DataFrame) -> dict[tuple[str, str], float | None]:
        from src.data.nxt_decomposition import _eligible_frame

        kept = _eligible_frame(frame, 0.05, caller="test")
        holdout = kept[(kept["day"] >= cfg.holdout_start) & (kept["day"] <= cfg.holdout_end)]
        by_symbol = _proxy_frame(holdout, cfg.auction_fraction_mean)
        out: dict[tuple[str, str], float | None] = {}
        for symbol, group in by_symbol.items():
            history = _history_series(group.sort_values("ordinal", kind="stable").reset_index(drop=True))
            for day in group["day"].tolist():
                out[(symbol, str(day))] = predict_share(history, str(day), config=cfg)
        return out

    reference = holdout_predictions(calibration)
    assert any(v is not None for v in reference.values())

    rescaled = calibration.copy()
    fit_days = set(pd.date_range(cfg.fit_start, cfg.calibrated_through, freq="B").strftime("%Y-%m-%d").tolist())
    mask = rescaled["date"].isin(fit_days)
    rescaled.loc[mask, "auction_volume"] = rescaled.loc[mask, "auction_volume"] * 2.0
    rescaled.loc[mask, "eod_volume"] = rescaled.loc[mask, "v_krx_1520"] + rescaled.loc[mask, "auction_volume"]
    refit = fit_decomposition(rescaled, **_fit_kwargs())  # type: ignore[arg-type]
    assert refit.config.auction_fraction_mean != pytest.approx(cfg.auction_fraction_mean)
    assert holdout_predictions(rescaled) == reference


def test_own_row_causality() -> None:
    from src.data.nxt_decomposition import _eligible_frame

    days = _days("2026-01-05", 10)
    calibration = _constant_calibration(days)
    fit = fit_decomposition(
        calibration, **_fit_kwargs(holdout_fraction=0.3, min_holdout_days=1)  # type: ignore[arg-type]
    )
    cfg = fit.config

    def symbol_history(frame: pd.DataFrame, symbol: str) -> pd.Series:
        kept = _eligible_frame(frame, 0.05, caller="test")
        window = kept[(kept["day"] >= cfg.fit_start) & (kept["day"] <= cfg.calibrated_through)]
        group = window[window["symbol"] == symbol].sort_values("ordinal", kind="stable").reset_index(drop=True)
        group = group.copy()
        group["proxy"] = group["eod_volume"] * (1.0 - cfg.auction_fraction_mean) / group["v_cons_1520"]
        return _history_series(group)

    target_day = sorted(pd.date_range(cfg.fit_start, cfg.calibrated_through, freq="B").strftime("%Y-%m-%d").tolist())[2]
    later_day = sorted(pd.date_range(cfg.fit_start, cfg.calibrated_through, freq="B").strftime("%Y-%m-%d").tolist())[3]
    own_before = predict_share(symbol_history(calibration, "000001"), target_day, config=cfg)
    later_before = predict_share(symbol_history(calibration, "000001"), later_day, config=cfg)
    assert own_before is not None and later_before is not None

    changed = calibration.copy()
    mask = (changed["date"] == target_day) & (changed["symbol"] == "000001")
    changed.loc[mask, "eod_volume"] = changed.loc[mask, "eod_volume"] * 1.5
    changed.loc[mask, "v_krx_1520"] = changed.loc[mask, "eod_volume"] - changed.loc[mask, "auction_volume"]
    assert predict_share(symbol_history(changed, "000001"), target_day, config=cfg) == own_before
    assert predict_share(symbol_history(changed, "000001"), later_day, config=cfg) != later_before


def test_proxy_scoring_not_truth_scoring() -> None:
    days = _days("2026-01-05", 10)
    rows: list[dict[str, object]] = []
    for i, day in enumerate(days):
        rows.append(_cal_row(day, "000001", _SHARE, eod_mult=1.00 if i % 2 == 0 else 1.04))
        rows.append(_cal_row(day, "000002", _SHARE, eod_mult=1.00 if i % 2 == 0 else 1.04))
    fit = fit_decomposition(pd.DataFrame(rows), **_fit_kwargs())  # type: ignore[arg-type]
    assert 0.96 < fit.config.bias_correction < 0.99
    assert fit.config.bias_correction != pytest.approx(1.0)
    assert fit.diagnostics.holdout_rel_err_p90 > 0.0


def test_structure_frontier_choice() -> None:
    days = _days("2026-01-05", 12)
    fit = fit_decomposition(
        _constant_calibration(days),
        **_fit_kwargs(structures=[(2, 30), (2, 60), (3, 30), (5, 30)]),  # type: ignore[arg-type]
    )
    cfg = fit.config
    diag = fit.diagnostics
    assert (cfg.ewma_alpha, cfg.min_prior_days, cfg.max_gap_days) == (0.3, 2, 60)
    assert len(diag.frontier) == 12
    first = diag.frontier[0]
    assert (first.ewma_alpha, first.min_prior_days, first.max_gap_days) == (0.3, 2, 60)
    keys = [(-p.coverage, p.rel_err_p90, p.min_prior_days, -p.max_gap_days, p.ewma_alpha) for p in diag.frontier]
    assert keys == sorted(keys)
    narrow = [p for p in diag.frontier if (p.min_prior_days, p.max_gap_days) == (5, 30)]
    assert narrow and all(p.coverage < 1.0 for p in narrow)
    assert all(p.coverage == 1.0 and p.rel_err_p90 == 0.0 for p in diag.frontier if p.min_prior_days == 2)
    assert diag.alpha_at_boundary is True


def test_no_qualifying_candidate_fails_closed() -> None:
    days = _days("2026-01-05", 10)
    rows: list[dict[str, object]] = []
    for i, day in enumerate(days):
        rows.append(_cal_row(day, "000001", _SHARE, eod_mult=1.00 if i % 2 == 0 else 1.04))
        rows.append(_cal_row(day, "000002", _SHARE, eod_mult=1.00 if i % 2 == 0 else 1.04))
    with pytest.raises(ValueError, match="no candidate"):
        fit_decomposition(pd.DataFrame(rows), **_fit_kwargs(max_selection_rel_err_p90=0.0))  # type: ignore[arg-type]


def test_alpha_boundary_flag() -> None:
    days = _days("2026-01-05", 12)
    trend = [0.50 + 0.03 * i for i in range(len(days))]
    rows = [_cal_row(day, symbol, share) for day, share in zip(days, trend, strict=True) for symbol in ("000001", "000002")]
    trended = fit_decomposition(pd.DataFrame(rows), **_fit_kwargs())  # type: ignore[arg-type]
    assert trended.config.ewma_alpha == pytest.approx(0.7)
    assert trended.diagnostics.alpha_at_boundary is True

    long_days = _days("2026-01-05", 20)
    waves = [0.68 + 0.04 * float(np.sin(2.0 * np.pi * i / 7.0 + 1.25)) for i in range(len(long_days))]
    wavy = [_cal_row(day, symbol, share) for day, share in zip(long_days, waves, strict=True) for symbol in ("000001", "000002")]
    interior = fit_decomposition(pd.DataFrame(wavy), **_fit_kwargs())  # type: ignore[arg-type]
    assert interior.config.ewma_alpha == pytest.approx(0.5)
    assert interior.diagnostics.alpha_at_boundary is False


def test_holdout_too_small() -> None:
    calibration = _constant_calibration(_days("2026-01-05", 4))
    with pytest.raises(ValueError, match="min_holdout_days"):
        fit_decomposition(
            calibration, **_fit_kwargs(holdout_fraction=0.25, min_holdout_days=2)  # type: ignore[arg-type]
        )


def test_ineligible_rows_excluded_never_zero_filled() -> None:
    days = _days("2026-01-05", 10)
    base = _constant_calibration(days)
    reference = fit_decomposition(base, **_fit_kwargs())  # type: ignore[arg-type]

    junk = pd.DataFrame([
        {**_cal_row(days[0], "000001", _SHARE), "eod_volume": float("nan")},
        {**_cal_row(days[1], "000002", _SHARE), "v_cons_1520": 0.0},
        {**_cal_row(days[2], "000001", _SHARE), "eod_volume": 1.5 * (0.68 * 10000.0 + _AUCTION)},
        {**_cal_row(days[3], "000002", _SHARE), "auction_volume": -10.0},
        {**_cal_row(days[4], "000001", _SHARE), "v_krx_1520": float("nan")},
        {**_cal_row("2026-03-02", "000001", _SHARE), "eod_volume": float("nan")},
    ])
    polluted = pd.concat([base, junk], ignore_index=True)
    assert len(polluted) == len(base) + 6
    fitted = fit_decomposition(polluted, **_fit_kwargs())  # type: ignore[arg-type]
    assert fitted.config == reference.config
    assert fitted.diagnostics.n_fit_rows + fitted.diagnostics.n_holdout_rows == len(base)
    assert fitted.diagnostics.n_fit_rows == reference.diagnostics.n_fit_rows
    assert fitted.diagnostics.n_holdout_rows == reference.diagnostics.n_holdout_rows


def test_decomposition_config_schema_is_fail_closed() -> None:
    from src.data.nxt_decomposition import decomposition_config_from_payload, decomposition_config_to_payload

    cfg = _config()
    payload = decomposition_config_to_payload(cfg)
    assert decomposition_config_from_payload(payload) == cfg
    assert decomposition_config_from_payload(dict(payload)) == cfg

    legacy = {k: v for k, v in payload.items() if k not in {"fit_start", "holdout_start", "holdout_end"}}
    with pytest.raises(ValueError, match="missing keys"):
        decomposition_config_from_payload(legacy)
    without_holdout = {k: v for k, v in payload.items() if k != "holdout_start"}
    with pytest.raises(ValueError, match="missing keys"):
        decomposition_config_from_payload(without_holdout)


def test_fit_diagnostics_round_trip(tmp_path) -> None:
    from src.data.nxt_decomposition import (
        fit_diagnostics_from_payload,
        fit_diagnostics_to_payload,
        load_fit_diagnostics,
        save_fit_diagnostics,
    )

    fitted = fit_decomposition(_constant_calibration(_days("2026-01-05", 12)), **_fit_kwargs())  # type: ignore[arg-type]
    diagnostics = fitted.diagnostics
    assert diagnostics.frontier, "fit must score candidates into the frontier"
    path = save_fit_diagnostics(diagnostics, tmp_path / "nxt_decomposition_fit_report.json")
    assert load_fit_diagnostics(path) == diagnostics
    assert fit_diagnostics_from_payload(fit_diagnostics_to_payload(diagnostics)) == diagnostics

    with pytest.raises(FileNotFoundError, match="not found"):
        load_fit_diagnostics(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_fit_diagnostics(bad)
    unknown = fit_diagnostics_to_payload(diagnostics) | {"renamed": 1.0}
    with pytest.raises(ValueError, match="unknown keys"):
        fit_diagnostics_from_payload(unknown)
    nan_field = fit_diagnostics_to_payload(diagnostics) | {"holdout_rel_err_p90": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        fit_diagnostics_from_payload(nan_field)
    nan_path = tmp_path / "nan.json"
    import json as _json

    nan_path.write_text(_json.dumps(nan_field), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_fit_diagnostics(nan_path)


def test_fit_is_deterministic() -> None:
    calibration = _constant_calibration(_days("2026-01-05", 12))
    first = fit_decomposition(calibration, **_fit_kwargs(structures=[(2, 30), (3, 20)]))  # type: ignore[arg-type]
    second = fit_decomposition(calibration, **_fit_kwargs(structures=[(2, 30), (3, 20)]))  # type: ignore[arg-type]
    assert first == second


def test_all_scored_rows_conform_and_histories_are_isolated(monkeypatch) -> None:
    import dataclasses

    import src.data.nxt_decomposition as module

    days = _days("2026-01-05", 20)
    calibration = pd.DataFrame([
        _cal_row(day, symbol, 0.65 + 0.03 * np.sin(i))
        for i, day in enumerate(days)
        for symbol in ("A", "B")
        if symbol == "A" or i % 3 != 1
    ])
    original = module._predict_raw
    calls = []

    def traced(priors, decision_ordinal, **kwargs):
        raw = original(priors, decision_ordinal, **kwargs)
        calls.append((list(priors), decision_ordinal, kwargs, raw))
        return raw

    monkeypatch.setattr(module, "_predict_raw", traced)
    fitted = fit_decomposition(calibration, **_fit_kwargs(structures=[(2, 30), (3, 5)]))
    split = (pd.Timestamp(fitted.config.fit_start) - pd.Timestamp("1970-01-01")).days
    assert calls
    assert {c[2]["alpha"] for c in calls} == {0.3, 0.5, 0.7}
    for priors, decision, kwargs, raw in calls:
        assert all((p.ordinal >= split) == (decision >= split) for p in priors)
        config = dataclasses.replace(
            fitted.config, ewma_alpha=kwargs["alpha"], min_prior_days=kwargs["min_prior_days"],
            max_gap_days=kwargs["max_gap_days"], bias_correction=1.0,
        )
        history = pd.Series(
            [p.proxy for p in priors],
            index=[(pd.Timestamp("1970-01-01") + pd.Timedelta(days=p.ordinal)).strftime("%Y-%m-%d") for p in priors],
        )
        day = (pd.Timestamp("1970-01-01") + pd.Timedelta(days=decision)).strftime("%Y-%m-%d")
        predicted = predict_share(history, day, config=config)
        if raw is None:
            assert predicted is None
        else:
            assert predicted == pytest.approx(raw, abs=1e-12, rel=0)


def test_nonfinite_diagnostics_cannot_be_returned_or_saved(tmp_path) -> None:
    import dataclasses

    from src.data.nxt_decomposition import save_decomposition_config, save_fit_diagnostics

    days = _days("2026-01-05", 12)
    calibration = _constant_calibration(days)
    fitted = fit_decomposition(calibration, **_fit_kwargs())
    mask = calibration["date"].isin(days[:3])
    calibration.loc[mask, "v_krx_1520"] = 1e-316
    calibration.loc[mask, "eod_volume"] = 365.0
    with pytest.raises(ValueError, match="finite holdout diagnostics"), np.errstate(all="ignore"):
        fit_decomposition(calibration, **_fit_kwargs())
    calibration = _constant_calibration(days)
    calibration.loc[mask, "close_krx_1519"] = 1e-300
    calibration.loc[mask, "close_cons_1519"] = 1e300
    with pytest.raises(ValueError, match="finite holdout close diagnostics"), np.errstate(all="ignore"):
        fit_decomposition(calibration, **_fit_kwargs())
    for saver, invalid in (
        (save_decomposition_config, dataclasses.replace(fitted.config, volume_rel_err_p90=float("nan"))),
        (save_fit_diagnostics, dataclasses.replace(fitted.diagnostics, holdout_rel_err_p90=float("nan"))),
    ):
        path = tmp_path / saver.__name__
        with pytest.raises(ValueError, match="non-finite"):
            saver(invalid, path)
        assert not path.exists()


def test_zero_coverage_frontier_round_trips(tmp_path) -> None:
    from src.data.nxt_decomposition import load_fit_diagnostics, save_fit_diagnostics

    fitted = fit_decomposition(
        _constant_calibration(_days("2026-01-05", 12)),
        **_fit_kwargs(structures=[(2, 30), (50, 30)]),
    )
    path = save_fit_diagnostics(fitted.diagnostics, tmp_path / "diagnostics.json")
    assert load_fit_diagnostics(path) == fitted.diagnostics
    assert "Infinity" not in path.read_text()


def test_degenerate_candidates_stay_listed() -> None:
    fitted = fit_decomposition(
        _constant_calibration(_days("2026-01-05", 12)),
        **_fit_kwargs(structures=[(2, 30), (50, 30)]),  # type: ignore[arg-type]
    )
    assert (fitted.config.min_prior_days, fitted.config.max_gap_days) == (2, 30)
    starved = [p for p in fitted.diagnostics.frontier if p.min_prior_days == 50]
    assert len(starved) == 3
    assert all(p.coverage == 0.0 for p in starved)
    assert all(p.rel_err_p90 is None for p in starved)


def test_holdout_range_gate_skips_overconfident_rows() -> None:
    days = _days("2026-01-05", 16)
    rows = _constant_calibration(days)
    frame = pd.DataFrame(rows)
    holdout_days = days[:4]
    for day in holdout_days[:2]:
        mask = (frame["date"] == day) & (frame["symbol"] == "000001")
        frame.loc[mask, "v_krx_1520"] = 9000.0
        frame.loc[mask, "v_cons_1520"] = 8000.0
        frame.loc[mask, "auction_volume"] = 365.0
        frame.loc[mask, "eod_volume"] = 9365.0
    fitted = fit_decomposition(frame, **_fit_kwargs())  # type: ignore[arg-type]
    assert fitted.diagnostics.n_holdout_rows == 8
    assert fitted.diagnostics.holdout_coverage == pytest.approx(3 / 8)


def test_holdout_without_predictions_fails_closed() -> None:
    calibration = _constant_calibration(_days("2026-01-05", 6))
    with pytest.raises(ValueError, match="cannot score any holdout"):
        fit_decomposition(calibration, **_fit_kwargs(min_holdout_days=1))  # type: ignore[arg-type]


def test_holdout_without_closes_fails_closed() -> None:
    calibration = _constant_calibration(_days("2026-01-05", 12))
    calibration[["close_krx_1519", "close_cons_1519"]] = float("nan")
    with pytest.raises(ValueError, match="close pair"):
        fit_decomposition(calibration, **_fit_kwargs())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"ewma_alpha": "bogus"}, "mistyped"),
        ({"ewma_alpha": None}, "mistyped"),
        ({"ewma_alpha": 1.5}, "out-of-range"),
        ({"ewma_alpha": 0.0}, "out-of-range"),
        ({"min_prior_days": True}, "mistyped"),
        ({"min_prior_days": 2.5}, "mistyped"),
        ({"min_prior_days": 0}, "out-of-range"),
        ({"max_gap_days": -1}, "out-of-range"),
        ({"bias_correction": 0.0}, "out-of-range"),
        ({"auction_fraction_mean": float("nan")}, "non-finite"),
        ({"volume_rel_err_p90": float("inf")}, "non-finite"),
        ({"volume_rel_err_p90": -0.1}, "negative"),
        ({"calibrated_through": "bogus"}, "mistyped"),
        ({"fit_start": ""}, "mistyped"),
    ],
)
def test_config_payload_rejects_bad_values(mutation: dict[str, object], match: str) -> None:
    from src.data.nxt_decomposition import decomposition_config_from_payload, decomposition_config_to_payload

    payload = decomposition_config_to_payload(_config())
    payload.update(mutation)
    with pytest.raises(ValueError, match=match):
        decomposition_config_from_payload(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"alpha_at_boundary": 1}, "mistyped"),
        ({"table_sha256": 5}, "mistyped"),
        ({"frontier": []}, "empty frontier"),
        ({"frontier": ["x"]}, "JSON objects"),
        ({"holdout_coverage": 1.5}, "out-of-range"),
        ({"holdout_rel_err_p90": float("inf")}, "non-finite"),
        ({"abar_fit": float("nan")}, "non-finite"),
        ({"bias_fit": 0.0}, "out-of-range"),
        ({"bias_holdout": -1.0}, "out-of-range"),
        ({"n_fit_rows": -1}, "out-of-range"),
        ({"n_fit_rows": "18"}, "mistyped"),
        ({"abar_fit": True}, "mistyped"),
    ],
)
def test_diagnostics_payload_rejects_bad_values(mutation: dict[str, object], match: str) -> None:
    from src.data.nxt_decomposition import fit_diagnostics_from_payload, fit_diagnostics_to_payload

    fitted = fit_decomposition(_constant_calibration(_days("2026-01-05", 12)), **_fit_kwargs())  # type: ignore[arg-type]
    payload = fit_diagnostics_to_payload(fitted.diagnostics)
    payload.update(mutation)
    with pytest.raises(ValueError, match=match):
        fit_diagnostics_from_payload(payload)


@pytest.mark.parametrize(
    ("target", "action", "match"),
    [
        ("outer", "drop", "missing keys"),
        ("outer", "unknown", "unknown keys"),
        ("outer", "not-mapping", "JSON object"),
        ("frontier-item", "unknown", "unknown keys"),
        ("frontier-item", "drop", "missing keys"),
        ("frontier-item", "mistyped", "mistyped"),
        ("frontier-item", "alpha", "out-of-range"),
        ("frontier-item", "structure", "out-of-range"),
        ("frontier-item", "coverage", "out-of-range"),
        ("frontier-item", "negative-error", "invalid"),
        ("frontier-item", "null-error", "requires"),
    ],
)
def test_diagnostics_payload_shape_is_fail_closed(target: str, action: str, match: str) -> None:
    from src.data.nxt_decomposition import fit_diagnostics_from_payload, fit_diagnostics_to_payload

    fitted = fit_decomposition(_constant_calibration(_days("2026-01-05", 12)), **_fit_kwargs())  # type: ignore[arg-type]
    payload = fit_diagnostics_to_payload(fitted.diagnostics)
    if (target, action) == ("outer", "drop"):
        payload.pop("bias_fit")
    elif (target, action) == ("outer", "unknown"):
        payload["renamed"] = 1.0
    elif (target, action) == ("outer", "not-mapping"):
        with pytest.raises(ValueError, match=match):
            fit_diagnostics_from_payload(["not", "a", "mapping"])  # type: ignore[arg-type]
        return
    elif (target, action) == ("frontier-item", "unknown"):
        payload["frontier"][0]["renamed"] = 1.0
    elif (target, action) == ("frontier-item", "drop"):
        payload["frontier"][0].pop("coverage")
    elif (target, action) == ("frontier-item", "mistyped"):
        payload["frontier"][0]["rel_err_p90"] = "bogus"
    elif (target, action) == ("frontier-item", "alpha"):
        payload["frontier"][0]["ewma_alpha"] = 2.0
    elif (target, action) == ("frontier-item", "structure"):
        payload["frontier"][0]["min_prior_days"] = 0
    elif (target, action) == ("frontier-item", "coverage"):
        payload["frontier"][0]["coverage"] = 2.0
    elif (target, action) == ("frontier-item", "negative-error"):
        payload["frontier"][0]["rel_err_p90"] = -0.5
    elif (target, action) == ("frontier-item", "null-error"):
        payload["frontier"][0]["rel_err_p90"] = None
    with pytest.raises(ValueError, match=match):
        fit_diagnostics_from_payload(payload)


def test_diagnostics_payload_rejects_unbounded_frontier_error() -> None:
    from src.data.nxt_decomposition import fit_diagnostics_from_payload, fit_diagnostics_to_payload

    fitted = fit_decomposition(_constant_calibration(_days("2026-01-05", 12)), **_fit_kwargs())  # type: ignore[arg-type]
    payload = fit_diagnostics_to_payload(fitted.diagnostics)
    payload["frontier"][0]["rel_err_p90"] = float("inf")
    with pytest.raises(ValueError, match="invalid"):
        fit_diagnostics_from_payload(payload)


def test_fit_measures_holdout_not_selection() -> None:
    fitted = fit_decomposition(_constant_calibration(_days("2026-01-05", 12)), **_fit_kwargs())  # type: ignore[arg-type]
    diag = fitted.diagnostics
    assert diag.holdout_start < diag.fit_start <= diag.fit_end
    assert fitted.config.calibrated_through == diag.fit_end
    assert fitted.config.fit_start == diag.fit_start
    assert (fitted.config.holdout_start, fitted.config.holdout_end) == (diag.holdout_start, diag.holdout_end)
    assert diag.n_fit_rows > 0 and diag.n_holdout_rows > 0
    assert diag.holdout_coverage < 1.0


# ---------------------------------------------------------------------------
# Production predictor, history and reconstruction guards (unchanged behaviour)
# ---------------------------------------------------------------------------


def _cons_bars(symbol: str, volumes: list[float], start_ts: int = 151900) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "symbol": symbol,
            "ts_hms": start_ts + i,
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": volume,
            "value_krw": 50050.0 * volume,
            "has_trade": True,
            "vendor": "toss",
        }
        for i, volume in enumerate(volumes)
    ])


def test_perturbation_invariance() -> None:
    days = _days("2026-03-02", 6)
    history = pd.Series(
        [0.68, 0.69, 0.67, 0.70, 0.68, 0.69],
        index=pd.Index(days, dtype=object),
        dtype=np.float64,
        name="share_proxy",
    )
    cfg = _config()
    decision = "2026-03-13"
    bars = _cons_bars("000001", [1000.0, 1200.0, 900.0])
    share = predict_share(history, decision, config=cfg)
    assert share is not None
    first = reconstruct_krx_bars(bars, share=share, config=cfg)
    extended = pd.concat([
        history,
        pd.Series(
            [0.10, 0.95],
            index=pd.Index(["2026-03-16", "2026-03-17"], dtype=object),
            dtype=np.float64,
        ),
    ])
    assert predict_share(extended, decision, config=cfg) == pytest.approx(share)
    second = reconstruct_krx_bars(bars, share=share, config=cfg)
    pd.testing.assert_frame_equal(first, second)


def test_insufficient_history_fails_closed() -> None:
    cfg = _config(min_prior_days=3, max_gap_days=5)
    thin = pd.Series(
        [0.68, 0.69],
        index=pd.Index(["2026-03-10", "2026-03-11"], dtype=object),
        dtype=np.float64,
    )
    assert predict_share(thin, "2026-03-12", config=cfg) is None
    gapped = pd.Series(
        [0.68, 0.69, 0.70],
        index=pd.Index(["2026-01-05", "2026-01-06", "2026-01-07"], dtype=object),
        dtype=np.float64,
    )
    assert predict_share(gapped, "2026-03-12", config=cfg) is None
    assert predict_share(thin, "2026-03-12", config=_config(min_prior_days=2, max_gap_days=30)) is not None


def test_bad_inputs_never_default() -> None:
    cfg = _config()
    bars_by_day = {
        "2026-03-02": _cons_bars("000001", [100.0, 100.0]),
        "2026-03-03": _cons_bars("000001", [100.0, 100.0]),
        "2026-03-04": _cons_bars("000001", [100.0, 100.0]),
        "2026-03-05": pd.DataFrame(),
    }
    eod_by_day = {
        "2026-03-02": float("nan"),
        "2026-03-03": 0.0,
        "2026-03-04": -500.0,
        "2026-03-05": 10000.0,
    }
    history = share_proxy_history(bars_by_day, eod_by_day, config=cfg)
    assert bool(history.isna().all())
    assert predict_share(history, "2026-03-06", config=cfg) is None


def test_exact_beats_reconstructed() -> None:
    cfg = _config()
    out = reconstruct_krx_bars(_cons_bars("000001", [1000.0]), share=0.68, config=cfg)
    assert str(out["basis"].iloc[0]) == "krx_reconstructed"
    assert set(out["basis"].unique().tolist()) == {"krx_reconstructed"}


def test_cutoff_respected() -> None:
    cfg = _config()
    late = _cons_bars("000001", [1000.0, 500.0], start_ts=152000)
    with pytest.raises(ValueError, match="cutoff"):
        reconstruct_krx_bars(late, share=0.68, config=cfg)
    ok = _cons_bars("000001", [1000.0, 500.0], start_ts=151900)
    out = reconstruct_krx_bars(ok, share=0.5, config=cfg)
    assert float(out["volume"].sum()) == pytest.approx(750.0)


def test_uncertainty_stamped() -> None:
    cfg = _config(volume_rel_err_p90=0.144, close_bp_err_p90=11.9)
    out = reconstruct_krx_bars(_cons_bars("000001", [1000.0]), share=0.68, config=cfg)
    assert (out["basis"] == "krx_reconstructed").all()
    assert np.allclose(out["recon_volume_rel_err_p90"].to_numpy(dtype=np.float64), 0.144)
    assert np.allclose(out["recon_close_bp_err_p90"].to_numpy(dtype=np.float64), 11.9)


def test_fit_refuses_thin_data() -> None:
    thin = pd.DataFrame([_cal_row("2026-03-02", "000001", _SHARE)])
    with pytest.raises(ValueError, match="symbol-days"):
        fit_decomposition(thin, **_fit_kwargs(min_holdout_days=1))  # type: ignore[arg-type]
    broken = _constant_calibration(_days("2026-03-02", 4)).copy()
    broken["eod_volume"] = broken["v_krx_1520"] * 2.0
    with pytest.raises(ValueError, match="symbol-days"):
        fit_decomposition(broken, **_fit_kwargs(min_holdout_days=1))  # type: ignore[arg-type]
    single_symbol = pd.DataFrame([_cal_row(day, "000001", _SHARE) for day in _days("2026-03-02", 6)])
    with pytest.raises(ValueError, match="symbols"):
        fit_decomposition(single_symbol, **_fit_kwargs())  # type: ignore[arg-type]


def test_reconstruct_rejects_invalid_share() -> None:
    cfg = _config()
    bars = _cons_bars("000001", [1000.0])
    for bad in (0.0, -0.2, 1.5, float("nan")):
        with pytest.raises(ValueError, match="share"):
            reconstruct_krx_bars(bars, share=bad, config=cfg)


def test_contract_validation() -> None:
    cal = _constant_calibration(_days("2026-03-02", 12))
    fit_kwargs = _fit_kwargs()
    for bad_alphas in ([], [0.5], [0.0, 0.5, 0.7], [1.5, 0.5, 0.7], [float("nan"), 0.5, 0.7], [0.5, 0.5, 0.7]):
        with pytest.raises(ValueError, match="alphas"):
            fit_decomposition(cal, **{**fit_kwargs, "alphas": bad_alphas})  # type: ignore[arg-type]
    for bad_structures in ([], ["xx"], [(0, 30)], [(2, -1)]):
        with pytest.raises(ValueError, match=r"structures|min_prior_days|max_gap_days"):
            fit_decomposition(cal, **{**fit_kwargs, "structures": bad_structures})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identity_tolerance"):
        fit_decomposition(cal, **{**fit_kwargs, "identity_tolerance": 0.0})  # type: ignore[arg-type]
    for bad_fraction in (0.0, 0.6, float("nan")):
        with pytest.raises(ValueError, match="holdout_fraction"):
            fit_decomposition(cal, **{**fit_kwargs, "holdout_fraction": bad_fraction})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_holdout_days"):
        fit_decomposition(cal, **{**fit_kwargs, "min_holdout_days": 0})  # type: ignore[arg-type]
    for bad_ceiling in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="max_selection_rel_err_p90"):
            fit_decomposition(cal, **{**fit_kwargs, "max_selection_rel_err_p90": bad_ceiling})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="table_sha256"):
        fit_decomposition(cal, **{**fit_kwargs, "table_sha256": 123})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="required columns"):
        fit_decomposition(cal.drop(columns=["eod_volume"]), **fit_kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no calibration rows"):
        fit_decomposition(cal.iloc[0:0], **fit_kwargs)  # type: ignore[arg-type]
    broken_dates = cal.copy()
    broken_dates.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="unparseable dates"):
        fit_decomposition(broken_dates, **fit_kwargs)  # type: ignore[arg-type]

    cfg = _config()
    history = pd.Series(
        [0.68, 0.69],
        index=pd.Index(["2026-03-10", "2026-03-11"], dtype=object),
        dtype=np.float64,
    )
    assert predict_share(history, "not-a-date", config=cfg) is None
    assert predict_share(history, "2026-03-12", config=_config(ewma_alpha=0.0)) is None
    junk_history = pd.Series(
        [0.68, 0.69],
        index=pd.Index(["2026-03-10", "bogus"], dtype=object),
        dtype=np.float64,
    )
    assert predict_share(junk_history, "2026-03-12", config=_config(min_prior_days=2)) is None
    out_of_range = pd.Series(
        [1.5, 1.6],
        index=pd.Index(["2026-03-10", "2026-03-11"], dtype=object),
        dtype=np.float64,
    )
    assert predict_share(out_of_range, "2026-03-12", config=cfg) is None

    with pytest.raises(ValueError, match="required columns"):
        reconstruct_krx_bars(pd.DataFrame([{"ts_hms": 151900}]), share=0.5, config=cfg)
    with pytest.raises(ValueError, match="no bars"):
        reconstruct_krx_bars(
            pd.DataFrame(columns=["ts_hms", "volume", "high", "low", "close"]), share=0.5, config=cfg
        )
    bad_price = _cons_bars("000001", [1000.0])
    bad_price.loc[0, "close"] = float("nan")
    with pytest.raises(ValueError, match="prices"):
        reconstruct_krx_bars(bad_price, share=0.5, config=cfg)
    bad_volume = _cons_bars("000001", [-5.0])
    with pytest.raises(ValueError, match="volumes"):
        reconstruct_krx_bars(bad_volume, share=0.5, config=cfg)
    bad_ts = _cons_bars("000001", [1000.0]).astype({"ts_hms": object})
    bad_ts.loc[0, "ts_hms"] = "bogus"
    with pytest.raises(ValueError, match="ts_hms"):
        reconstruct_krx_bars(bad_ts, share=0.5, config=cfg)


def test_degenerate_calibration_and_history() -> None:
    single_day = pd.DataFrame([_cal_row("2026-03-02", s, _SHARE) for s in ("000001", "000002")])
    with pytest.raises(ValueError, match="symbol-days"):
        fit_decomposition(single_day, **_fit_kwargs(min_holdout_days=1))  # type: ignore[arg-type]
    overflow = pd.DataFrame([
        {
            "date": day,
            "symbol": symbol,
            "v_krx_1520": 1e308,
            "auction_volume": 365.0,
            "v_cons_1520": 1e-308,
            "eod_volume": 1e308,
            "close_krx_1519": 50000.0,
            "close_cons_1519": 50000.0,
        }
        for day in _days("2026-03-02", 4)
        for symbol in ("000001", "000002")
    ])
    with pytest.raises(ValueError, match="symbol-days"), np.errstate(over="ignore"):
        fit_decomposition(overflow, **_fit_kwargs())  # type: ignore[arg-type]

    cfg = _config()
    assert share_proxy_history({}, {}, config=cfg).empty
    assert share_proxy_history(
        {"2026-03-02": _cons_bars("000001", [100.0])},
        {"2026-03-02": 10000.0},
        config=_config(auction_fraction_mean=1.5),
    ).isna().all()
    non_numeric_ts = _cons_bars("000001", [100.0]).astype({"ts_hms": object})
    non_numeric_ts.loc[:, "ts_hms"] = "bogus"
    mixed: dict[str, object] = {
        "2026-03-02": non_numeric_ts,
        "2026-03-03": _cons_bars("000001", [100.0]),
        "NaT": _cons_bars("000001", [100.0]),
    }
    history = share_proxy_history(
        mixed,  # type: ignore[arg-type]
        {"2026-03-02": 10000.0, "2026-03-03": None, "NaT": None},  # type: ignore[dict-item]
        config=cfg,
    )
    assert bool(history.isna().all())
    stamped = pd.Series(
        [0.68, 0.69],
        index=pd.Index(["2026-03-10", "NaT"], dtype=object),
        dtype=np.float64,
    )
    assert predict_share(stamped, "2026-03-12", config=_config(min_prior_days=1)) == pytest.approx(0.68)
    with_none = pd.Series(
        [0.68, None, 0.70],
        index=pd.Index(["2026-03-10", "2026-03-11", "2026-03-12"], dtype=object),
        dtype=object,
    )
    assert predict_share(with_none, "2026-03-13", config=_config(min_prior_days=2)) is not None


def test_proxy_valid_value_and_value_column_branches() -> None:
    cfg = _config()
    history = share_proxy_history(
        {"2026-03-02": _cons_bars("000001", [100.0, 100.0])},
        {"2026-03-02": 10000.0},
        config=cfg,
    )
    assert history["2026-03-02"] == pytest.approx(10000.0 * (1.0 - 0.0365) / 200.0)

    disjoint = pd.DataFrame(
        [_cal_row("2026-03-02", s, 0.6) for s in ("000001", "000002")]
        + [_cal_row("2026-03-03", s, 0.7) for s in ("000003", "000004")]
    )
    with pytest.raises(ValueError, match=r"symbol-days|holdout|no candidate"):
        fit_decomposition(disjoint, **_fit_kwargs())  # type: ignore[arg-type]

    legacy_value = _cons_bars("000001", [1000.0]).drop(columns=["value_krw"])
    legacy_value["value"] = 50050000.0
    out = reconstruct_krx_bars(legacy_value, share=0.5, config=cfg)
    assert float(out["value"].iloc[0]) == pytest.approx(500.0 * (50100.0 + 49900.0 + 50050.0) / 3.0)

    no_value = _cons_bars("000001", [1000.0]).drop(columns=["value_krw"])
    out = reconstruct_krx_bars(no_value, share=0.5, config=cfg)
    assert "value_krw" in out.columns

    zero_price = _cons_bars("000001", [1000.0])
    zero_price.loc[0, "low"] = 0.0
    with pytest.raises(ValueError, match="prices"):
        reconstruct_krx_bars(zero_price, share=0.5, config=cfg)


def test_decomposition_config_json_round_trip(tmp_path) -> None:
    from src.data.nxt_decomposition import (
        decomposition_config_from_payload,
        load_decomposition_config,
        save_decomposition_config,
    )

    cfg = _config()
    path = save_decomposition_config(cfg, tmp_path / "nxt_decomposition_config.json")
    assert load_decomposition_config(path) == cfg
    with pytest.raises(FileNotFoundError, match="not found"):
        load_decomposition_config(tmp_path / "absent.json")
    bad = path.with_name("bad.json")
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_decomposition_config(bad)
    mistyped_file = path.with_name("mistyped.json")
    mistyped_file.write_text('{"ewma_alpha": 0.5}', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_decomposition_config(mistyped_file)
    with pytest.raises(ValueError, match="unknown keys"):
        decomposition_config_from_payload({**cfg.__dict__, "renamed": 1.0})
    dropped = {k: v for k, v in cfg.__dict__.items() if k != "ewma_alpha"}
    with pytest.raises(ValueError, match="missing keys"):
        decomposition_config_from_payload(dropped)
    mistyped = {**cfg.__dict__, "min_prior_days": "two"}
    with pytest.raises(ValueError, match="mistyped"):
        decomposition_config_from_payload(mistyped)
    uncalibrated = {**cfg.__dict__, "calibrated_through": "  "}
    with pytest.raises(ValueError, match="calibrated_through"):
        decomposition_config_from_payload(uncalibrated)
    with pytest.raises(ValueError, match="JSON object"):
        decomposition_config_from_payload(["not", "a", "mapping"])  # type: ignore[arg-type]
    non_finite = {**cfg.__dict__, "volume_rel_err_p90": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        decomposition_config_from_payload(non_finite)


def test_non_finite_bar_volume_fails_closed() -> None:
    """A NaN minute volume never becomes zero: the proxy is NaN and the reconstruction raises."""
    import numpy as np
    import pytest

    from src.data.nxt_decomposition import DecompositionConfig, reconstruct_krx_bars, share_proxy_history

    cfg = _config(min_prior_days=1, max_gap_days=7, auction_fraction_mean=0.04)
    assert isinstance(cfg, DecompositionConfig)
    bars = pd.DataFrame({
        "ts_hms": [90100, 90200], "volume": [1000.0, np.nan], "open": [100, 100], "high": [101, 101],
        "low": [99, 99], "close": [100, 100], "value_krw": [1.0, 1.0],
    })
    proxy = share_proxy_history({"2026-03-02": bars}, {"2026-03-02": 1000.0}, config=cfg)
    assert proxy.isna().all()
    with pytest.raises(ValueError, match="invalid volumes"):
        reconstruct_krx_bars(bars.assign(vendor="toss_cons"), share=0.7, config=cfg)
