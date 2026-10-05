"""Invariant guards for the causal consolidated-to-KRX decomposition estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.nxt_decomposition import (
    DecompositionConfig,
    fit_decomposition_config,
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
    }
    base.update(overrides)
    return DecompositionConfig(**base)  # type: ignore[arg-type]


def _cal_row(day: str, symbol: str, share: float, v_cons: float = 10000.0) -> dict[str, object]:
    v_krx = share * v_cons
    return {
        "date": day,
        "symbol": symbol,
        "v_krx_1520": v_krx,
        "auction_volume": _AUCTION,
        "v_cons_1520": v_cons,
        "eod_volume": v_krx + _AUCTION,
        "close_krx_1519": 50000.0,
        "close_cons_1519": 50000.0,
    }


def _days(start: str, n: int) -> list[str]:
    return pd.date_range(start, periods=n, freq="B").strftime("%Y-%m-%d").tolist()


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


def _constant_calibration(days: list[str], share: float = _SHARE) -> pd.DataFrame:
    rows = [_cal_row(d, symbol, share) for d in days for symbol in ("000001", "000002")]
    return pd.DataFrame(rows)


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


def test_identity_on_synthetic_truth() -> None:
    days = _days("2026-03-02", 10)
    cfg = fit_decomposition_config(
        _constant_calibration(days),
        alphas=[0.2, 0.5, 0.9],
        min_prior_days=2,
        max_gap_days=30,
        identity_tolerance=0.05,
    )
    assert cfg.ewma_alpha == pytest.approx(0.2)
    assert cfg.bias_correction == pytest.approx(1.0)
    cons = _cons_bars("000001", [1000.0, 1000.0, 1000.0])
    out = reconstruct_krx_bars(cons, share=_SHARE, config=cfg)
    assert np.allclose(out["volume"].to_numpy(dtype=np.float64), 680.0, rtol=1e-9)
    expected_value = 680.0 * (50100.0 + 49900.0 + 50050.0) / 3.0
    assert np.allclose(out["value_krw"].to_numpy(dtype=np.float64), expected_value, rtol=1e-9)
    assert (out["close"].to_numpy() == 50050.0).all()


def test_walk_forward_selection_scores_later_rows_only() -> None:
    early = _days("2026-01-05", 12)
    late = _days("2026-02-23", 6)
    rows: list[dict[str, object]] = [
        _cal_row(day, symbol, 0.5 if i % 2 == 0 else 0.9)
        for i, day in enumerate(early)
        for symbol in ("000001", "000002")
    ]
    trend = [0.50, 0.58, 0.66, 0.74, 0.82, 0.90]
    rows.extend(
        _cal_row(day, symbol, trend[i])
        for i, day in enumerate(late)
        for symbol in ("000001", "000002")
    )
    cfg = fit_decomposition_config(
        pd.DataFrame(rows),
        alphas=[0.2, 0.5, 0.9],
        min_prior_days=2,
        max_gap_days=90,
        identity_tolerance=0.05,
    )
    assert cfg.ewma_alpha == pytest.approx(0.9)

    work = pd.DataFrame(rows).copy()
    work["day"] = pd.to_datetime(work["date"], format="mixed").dt.strftime("%Y-%m-%d")
    work["true_share"] = work["v_krx_1520"] / work["v_cons_1520"]

    def median_on(dates: list[str], alpha: float) -> float:
        errors: list[float] = []
        seen: dict[str, list[float]] = {}
        for record in work.sort_values(["symbol", "day"], kind="stable").to_dict(orient="records"):
            prior = list(seen.get(str(record["symbol"]), []))
            seen.setdefault(str(record["symbol"]), []).append(float(record["true_share"]))
            if str(record["day"]) not in set(dates) or not prior:
                continue
            ema = prior[0]
            for value in prior[1:]:
                ema = alpha * value + (1.0 - alpha) * ema
            errors.append(abs(ema - float(record["true_share"])) / float(record["true_share"]))
        return float(np.median(errors))

    all_dates = sorted(work["day"].unique().tolist())
    in_sample = {a: median_on(all_dates, a) for a in (0.2, 0.5, 0.9)}
    assert min(in_sample, key=lambda a: in_sample[a]) == pytest.approx(0.2)
    assert cfg.ewma_alpha != min(in_sample, key=lambda a: in_sample[a])


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
        fit_decomposition_config(
            thin, alphas=[0.5], min_prior_days=2, max_gap_days=30, identity_tolerance=0.05
        )
    broken = _constant_calibration(_days("2026-03-02", 4)).copy()
    broken["eod_volume"] = broken["v_krx_1520"] * 2.0
    with pytest.raises(ValueError, match="identity"):
        fit_decomposition_config(
            broken, alphas=[0.5], min_prior_days=2, max_gap_days=30, identity_tolerance=0.05
        )


def test_reconstruct_rejects_invalid_share() -> None:
    cfg = _config()
    bars = _cons_bars("000001", [1000.0])
    for bad in (0.0, -0.2, 1.5, float("nan")):
        with pytest.raises(ValueError, match="share"):
            reconstruct_krx_bars(bars, share=bad, config=cfg)


def test_contract_validation() -> None:
    cal = _constant_calibration(_days("2026-03-02", 4))
    fit_kwargs: dict[str, object] = {
        "alphas": [0.5],
        "min_prior_days": 2,
        "max_gap_days": 30,
        "identity_tolerance": 0.05,
    }
    for bad_alphas in ([], [0.0], [1.5], [float("nan")]):
        with pytest.raises(ValueError, match="alphas"):
            fit_decomposition_config(cal, **{**fit_kwargs, "alphas": bad_alphas})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_prior_days"):
        fit_decomposition_config(cal, **{**fit_kwargs, "min_prior_days": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_gap_days"):
        fit_decomposition_config(cal, **{**fit_kwargs, "max_gap_days": -1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identity_tolerance"):
        fit_decomposition_config(cal, **{**fit_kwargs, "identity_tolerance": 0.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="required columns"):
        fit_decomposition_config(cal.drop(columns=["eod_volume"]), **fit_kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no calibration rows"):
        fit_decomposition_config(cal.iloc[0:0], **fit_kwargs)  # type: ignore[arg-type]
    broken_dates = cal.copy()
    broken_dates.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="unparseable dates"):
        fit_decomposition_config(broken_dates, **fit_kwargs)  # type: ignore[arg-type]

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
    from src.data.nxt_decomposition import _scored_errors

    single_day = pd.DataFrame([_cal_row("2026-03-02", s, _SHARE) for s in ("000001", "000002")])
    with pytest.raises(ValueError, match="walk-forward"):
        fit_decomposition_config(
            single_day, alphas=[0.5], min_prior_days=2, max_gap_days=30, identity_tolerance=0.05
        )
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
        for day in _days("2026-03-02", 2)
        for symbol in ("000001", "000002")
    ])
    with pytest.raises(ValueError, match="positive-share"), np.errstate(over="ignore"):
        fit_decomposition_config(
            overflow, alphas=[0.5], min_prior_days=2, max_gap_days=30, identity_tolerance=0.05
        )

    frame = pd.DataFrame([
        {"symbol": "000001", "date": "2026-03-02", "day": "2026-03-02", "true_share": 0.0},
        {"symbol": "000001", "date": "2026-03-03", "day": "2026-03-03", "true_share": 0.7},
    ])
    errors, _ = _scored_errors(frame, {"2026-03-03"}, 0.5)
    assert errors == []

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
    with pytest.raises(ValueError, match="walk-forward"):
        fit_decomposition_config(
            disjoint, alphas=[0.5], min_prior_days=2, max_gap_days=30, identity_tolerance=0.05
        )

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
        decomposition_config_from_payload(["not", "a", "mapping"])


def test_non_finite_bar_volume_fails_closed() -> None:
    """A NaN minute volume never becomes zero: the proxy is NaN and the reconstruction raises."""
    import numpy as np
    import pytest

    from src.data.nxt_decomposition import DecompositionConfig, reconstruct_krx_bars, share_proxy_history

    cfg = DecompositionConfig(
        ewma_alpha=0.5, min_prior_days=1, max_gap_days=7, auction_fraction_mean=0.04, bias_correction=1.0,
        volume_rel_err_p90=0.1, close_bp_err_p90=10.0, calibrated_through="2026-01-01",
    )
    bars = pd.DataFrame({
        "ts_hms": [90100, 90200], "volume": [1000.0, np.nan], "open": [100, 100], "high": [101, 101],
        "low": [99, 99], "close": [100, 100], "value_krw": [1.0, 1.0],
    })
    proxy = share_proxy_history({"2026-03-02": bars}, {"2026-03-02": 1000.0}, config=cfg)
    assert proxy.isna().all()
    with pytest.raises(ValueError, match="invalid volumes"):
        reconstruct_krx_bars(bars.assign(vendor="toss_cons"), share=0.7, config=cfg)
