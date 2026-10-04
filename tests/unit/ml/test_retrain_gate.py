def test_evaluate_retrain_promotion_promotes_consistent_candidate() -> None:
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.ml import retrain_gate
    from src.strategy.contract import PRODUCTION_STRATEGY

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe)}

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    eval_frame = pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})

    # When: 후보와 현행이 같은 신호를 본다
    verdict = retrain_gate.evaluate_retrain_promotion(_bundle(_ColumnModel("f1")), _bundle(_ColumnModel("f1")), eval_frame)

    # Then
    assert verdict.promote is True
    assert verdict.reasons == ()
    assert verdict.agreement is not None and verdict.agreement > 0.99



def test_evaluate_retrain_promotion_rejects_divergent_ranking() -> None:
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.ml import retrain_gate
    from src.strategy.contract import PRODUCTION_STRATEGY

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe)}

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    eval_frame = pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})

    # When: 후보가 무관한 신호로 순위를 매긴다(데이터 오염 시그니처)
    verdict = retrain_gate.evaluate_retrain_promotion(_bundle(_ColumnModel("f2")), _bundle(_ColumnModel("f1")), eval_frame)

    # Then
    assert verdict.promote is False
    assert verdict.agreement is not None and verdict.agreement < retrain_gate.RETRAIN_MIN_PREDICTION_AGREEMENT
    assert any("agreement" in reason for reason in verdict.reasons)



def test_evaluate_retrain_promotion_bootstraps_without_live_bundle_but_checks_structure() -> None:
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.ml import retrain_gate
    from src.strategy.contract import PRODUCTION_STRATEGY

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe)}

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    eval_frame = pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})

    class _ConstantModel:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(features))

    class _NanModel:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.full(len(features), np.nan)

    # When/Then: 최초 발행은 구조 검증만으로 승격
    first = retrain_gate.evaluate_retrain_promotion(_bundle(_ColumnModel("f1")), None, eval_frame)
    assert first.promote is True and first.agreement is None

    # And: 일중 상수 예측은 차단
    constant = retrain_gate.evaluate_retrain_promotion(_bundle(_ConstantModel()), None, eval_frame)
    assert constant.promote is False and any("constant" in r for r in constant.reasons)

    # And: 비유한 예측은 차단
    nan = retrain_gate.evaluate_retrain_promotion(_bundle(_NanModel()), None, eval_frame)
    assert nan.promote is False and any("non-finite" in r for r in nan.reasons)

    # And: 라이브 스크린과 다른 번들은 예측 전에 차단
    drifted = _bundle(_ColumnModel("f1"))
    drifted["select_universe"] = dict(drifted["select_universe"], max_tick_cost_bp=99.0)
    parity = retrain_gate.evaluate_retrain_promotion(drifted, None, eval_frame)
    assert parity.promote is False and any("screen parity" in r for r in parity.reasons)



def test_evaluate_retrain_promotion_blocks_feature_contract_change_and_bad_eval_frames() -> None:
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.ml import retrain_gate
    from src.strategy.contract import PRODUCTION_STRATEGY

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe)}

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    eval_frame = pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})

    live = _bundle(_ColumnModel("f1"), feature_cols=("f1",))

    # When/Then: 피처 계약 변경은 수동 인증 대상
    changed = retrain_gate.evaluate_retrain_promotion(_bundle(_ColumnModel("f1")), live, eval_frame)
    assert changed.promote is False and any("feature contract changed" in r for r in changed.reasons)

    # And: 빈 feature_cols
    empty_cols = retrain_gate.evaluate_retrain_promotion(_bundle(_ColumnModel("f1"), feature_cols=()), None, eval_frame)
    assert empty_cols.promote is False and any("feature_cols is empty" in r for r in empty_cols.reasons)

    # And: 평가 프레임에 피처 누락 + 빈 프레임
    bad_frame = retrain_gate.evaluate_retrain_promotion(_bundle(_ColumnModel("f1")), None, eval_frame.iloc[0:0][["date"]])
    assert bad_frame.promote is False
    assert any("missing features" in r for r in bad_frame.reasons)
    assert any("is empty" in r for r in bad_frame.reasons)

    # And: 모든 날짜가 표본 3개 이하라 합의도를 잴 수 없음
    thin = eval_frame.groupby("date").head(3).reset_index(drop=True)
    thin_verdict = retrain_gate.evaluate_retrain_promotion(_bundle(_ColumnModel("f1")), _bundle(_ColumnModel("f1")), thin)
    assert thin_verdict.promote is False and thin_verdict.agreement is None



def test_build_gate_eval_frame_keeps_selected_rows_of_recent_dates(monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    from src.ml import retrain_gate

    dates = pd.bdate_range("2026-09-01", periods=5)
    pool = pd.DataFrame({"date": np.repeat(dates.to_numpy(), 2), "f1": range(10)})
    mask = np.array([True, False] * 5)
    monkeypatch.setattr(retrain_gate, "build_dual_pool", lambda *args, **kwargs: (pool, mask))

    # When
    frame = retrain_gate.build_gate_eval_frame(pd.DataFrame(), np.array([]), {}, eval_days=2)

    # Then
    assert list(pd.to_datetime(frame["date"])) == list(dates[-2:])
    assert list(frame["f1"]) == [6, 8]



def test_load_current_bundle_returns_none_until_published(tmp_path) -> None:
    from src.ml import retrain_gate
    from src.ml.topk_contract import save_production_bundle

    export_dir = str(tmp_path / "topk_ranker")

    # Given/When/Then: 발행 전
    assert retrain_gate.load_current_bundle(export_dir) is None

    # When: 원자적 저장 후
    path = save_production_bundle({"feature_cols": ["f1"], "top_k": 3}, export_dir=export_dir)

    # Then
    assert retrain_gate.load_current_bundle(export_dir) == {"feature_cols": ["f1"], "top_k": 3}
    assert [p.name for p in (tmp_path / "topk_ranker").iterdir()] == ["sizing_pipeline_bundle.joblib"]
    assert path.endswith("sizing_pipeline_bundle.joblib")


def _gate_bundles(version_current, version_candidate):
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.strategy.contract import PRODUCTION_STRATEGY

    class _ColumnModel:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return features["f1"].to_numpy(dtype=np.float64)

    def _bundle(model, version) -> dict:
        bundle = {"feature_cols": ["f1", "f2"], "return_model": model,
                  "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe)}
        if version is not None:
            bundle["feature_contract_version"] = version
        return bundle

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    eval_frame = pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    current = None if version_current == "NONE" else _bundle(_ColumnModel(), version_current)
    candidate = _bundle(_ColumnModel(), version_candidate)
    return current, candidate, eval_frame


def test_evaluate_retrain_promotion_blocks_feature_contract_version_change() -> None:
    from src.ml import retrain_gate

    current, candidate, eval_frame = _gate_bundles("0", "1")

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, current, eval_frame)

    assert verdict.promote is False
    assert any("feature contract changed" in r for r in verdict.reasons)
    assert verdict.agreement is None


def test_evaluate_retrain_promotion_treats_keyless_live_bundle_as_baseline() -> None:
    from src.ml import retrain_gate

    current, candidate, eval_frame = _gate_bundles(None, "1")

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, current, eval_frame)

    assert verdict.promote is True


def test_evaluate_retrain_promotion_rejects_mis_stamped_candidate() -> None:
    from src.ml import retrain_gate

    _current, candidate, eval_frame = _gate_bundles("NONE", "0")

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, None, eval_frame)

    assert verdict.promote is False
    assert any(r.startswith("candidate feature contract") for r in verdict.reasons)


def test_evaluate_retrain_promotion_blocks_malformed_live_stamp_without_raising() -> None:
    from src.ml import retrain_gate

    current, candidate, eval_frame = _gate_bundles(1, "1")

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, current, eval_frame)

    assert verdict.promote is False
    assert any("feature contract" in r for r in verdict.reasons)
    assert verdict.agreement is None


def _cutover_bundles():
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.strategy.contract import COST_AWARE_UNIVERSE, PRODUCTION_STRATEGY

    class _ExplodingModel:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            raise AssertionError("no model may be scored across a strategy change")

    def _bundle(model, strategy_id, screen) -> dict:
        return {"strategy_id": strategy_id, "feature_cols": ["f1", "f2"], "return_model": model,
                "select_universe": dataclasses.asdict(screen),
                "feature_contract_version": "1"}

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    eval_frame = pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    current = _bundle(_ExplodingModel(), "KCA-TOPK-COSTAWARE-001", COST_AWARE_UNIVERSE)
    candidate = _bundle(_ExplodingModel(), "KCA-TOPK-COSTAWARE-002", PRODUCTION_STRATEGY.universe)
    return current, candidate, eval_frame


def test_evaluate_retrain_promotion_refuses_strategy_change_without_scoring() -> None:
    from src.ml import retrain_gate

    current, candidate, eval_frame = _cutover_bundles()

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, current, eval_frame)

    assert verdict.promote is False
    assert verdict.agreement is None
    assert any("--skip-promotion-gate" in r for r in verdict.reasons)


def test_evaluate_retrain_promotion_refuses_screen_change_under_the_same_id() -> None:
    from src.ml import retrain_gate

    current, candidate, eval_frame = _cutover_bundles()
    current = dict(current, strategy_id="KCA-TOPK-COSTAWARE-002")
    drifted = dict(current["select_universe"])
    drifted["max_tick_cost_bp"] = 7.5
    current = dict(current, select_universe=drifted)

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, current, eval_frame)

    assert verdict.promote is False
    assert verdict.agreement is None
    assert any("--skip-promotion-gate" in r for r in verdict.reasons)

    # And: an unreadable live screen blocks instead of raising
    unreadable = dict(current, select_universe="nope")
    blocked = retrain_gate.evaluate_retrain_promotion(candidate, unreadable, eval_frame)

    assert blocked.promote is False
    assert blocked.agreement is None
    assert any("--skip-promotion-gate" in r for r in blocked.reasons)


def test_evaluate_retrain_promotion_uses_agreement_for_the_same_production_screen() -> None:
    import numpy as np
    import pandas as pd

    from src.ml import retrain_gate
    from src.ml.retrain_gate import RETRAIN_MIN_PREDICTION_AGREEMENT

    current, candidate, eval_frame = _cutover_bundles()

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    current = dict(current, strategy_id="KCA-TOPK-COSTAWARE-002",
                     select_universe=dict(candidate["select_universe"]),
                     return_model=_ColumnModel("f1"))
    candidate = dict(candidate, return_model=_ColumnModel("f1"))

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, current, eval_frame)

    assert verdict.promote is True
    assert verdict.agreement is not None and verdict.agreement >= RETRAIN_MIN_PREDICTION_AGREEMENT


def test_evaluate_retrain_promotion_bootstraps_a_002_candidate() -> None:
    from src.ml import retrain_gate

    _current, candidate, eval_frame = _cutover_bundles()

    class _ColumnModel:
        def predict(self, features) -> object:
            import numpy as np

            return features["f1"].to_numpy(dtype=np.float64)

    candidate = dict(candidate, return_model=_ColumnModel())

    verdict = retrain_gate.evaluate_retrain_promotion(candidate, None, eval_frame)

    assert verdict.promote is True
    assert verdict.agreement is None


def test_build_gate_eval_frame_excludes_non_screenable_symbols() -> None:
    import numpy as np
    import pandas as pd

    from src.data.panel_integrity import prepare_price_panel
    from src.ml import retrain_gate

    dates = pd.bdate_range("2023-02-01", periods=30)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(4):
        base = 18000.0
        for d in dates:
            prev = base / 1.05
            rows.append({
                "date": d, "symbol": f"{i:06d}", "open": prev, "high": base * 1.01,
                "low": prev * 0.99, "close": base, "prev_close": prev, "volume": 1e6,
                "market_cap_100m": 3000.0, "trade_value_100m": 500.0, "market": "KOSPI",
                "daily_change_pct": 0.05, "inst_netbuy": float(rng.integers(-10**8, 10**8)),
                "foreign_netbuy": 0.0, "program_netbuy": 0.0, "kospi_pct": 0.001,
                "kosdaq_pct": 0.002, "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
    ph, _prov = prepare_price_panel(pd.DataFrame(rows))
    ph["is_screenable"] = ph["symbol"] != "000000"
    ph["screenable_source"] = "real"
    market_dates = np.array(sorted(ph["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}

    frame = retrain_gate.build_gate_eval_frame(ph, market_dates, d_to_idx, eval_days=30)

    assert "000000" not in set(frame["symbol"].unique())
    assert "000001" in set(frame["symbol"].unique())


def _pit_candidate(**overrides):
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.strategy.contract import PRODUCTION_STRATEGY

    class _ColumnModel:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return features["f1"].to_numpy(dtype=np.float64)

    bundle = {
        "strategy_id": "KCA-TOPK-COSTAWARE-002",
        "top_k": 3,
        "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe),
        "feature_contract_version": "1",
        "model_params": {"n_estimators": 10},
        "seeds": [1],
        "feature_cols": ["f1", "f2"],
        "return_model": _ColumnModel(),
    }
    bundle.update(overrides)
    return bundle


def _pit_eval_frame():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    return pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})


def _pit_report(native_mean=5.0, native_ic=0.02, status="OK", generated_at=None, **overrides):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import dataclasses

    from src.ml.pit_report import PairedDelta, PitHaircutReport, PitReportStatus
    from src.strategy.contract import PRODUCTION_STRATEGY

    stamp = generated_at or datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    kwargs = {
        "generated_at": stamp,
        "strategy_id": "KCA-TOPK-COSTAWARE-002",
        "strategy_fingerprint": "fp",
        "top_k": 3,
        "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe),
        "feature_contract_version": "1",
        "model_params": {"n_estimators": 10},
        "seeds": (1,),
        "status": PitReportStatus(status),
        "panel_date_min": "2026-06-01",
        "panel_date_max": "2026-09-01",
        "n_usable_days": 60,
        "n_paired_days": 60,
        "n_live_days": 60,
        "n_eod_index_days": 0,
        "mean_net_bp": {"eod_full": 8.0, "eod_matched": 8.0, "pit_feature": 6.0, "pit_native": native_mean},
        "haircut": PairedDelta(delta=8.0 - native_mean, ci_low=0.0, ci_high=5.0, p_value=0.1, n_days=60),
        "coverage_component": PairedDelta(delta=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0, n_days=60),
        "feature_component": PairedDelta(delta=1.0, ci_low=0.0, ci_high=2.0, p_value=0.2, n_days=60),
        "selection_component": PairedDelta(delta=1.0, ci_low=0.0, ci_high=2.0, p_value=0.2, n_days=60),
        "pit_native_vs_zero": PairedDelta(delta=native_mean, ci_low=0.0, ci_high=9.0, p_value=0.01, n_days=60),
        "rank_ic_mean": {"eod_full": 0.03, "eod_matched": 0.03, "pit_feature": 0.02, "pit_native": native_ic},
        "rank_ic_haircut": PairedDelta(delta=0.01, ci_low=0.0, ci_high=0.02, p_value=0.2, n_days=60),
        "pick_overlap_mean": {"pit_feature": 0.9, "pit_native": 0.8},
        "haircut_by_index_basis": {"live_1520": 2.0, "eod_fallback": float("nan")},
        "augmentation": None,
    }
    kwargs.update(overrides)
    return PitHaircutReport(**kwargs)


def _pit_gate(mode, **overrides):
    from src.ml.retrain_gate import PitGateConfig, PitGateMode

    kwargs = {"mode": PitGateMode(mode)}
    kwargs.update(overrides)
    return PitGateConfig(**kwargs)


def test_evaluate_retrain_promotion_pit_default_off_preserves_verdict() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    current = _pit_candidate()
    verdict = retrain_gate.evaluate_retrain_promotion(candidate, current, _pit_eval_frame())

    assert verdict.promote is True
    assert verdict.reasons == ()
    assert verdict.agreement is not None and verdict.agreement > 0.99
    assert verdict.pit_status == "OFF"
    assert verdict.pit_reasons == ()


def test_pit_gate_advisory_records_without_blocking() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    current = _pit_candidate()
    report = _pit_report(native_mean=-5.0, native_ic=-0.01)
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, current, _pit_eval_frame(), pit_report=report, pit_gate=_pit_gate("advisory")
    )

    assert verdict.promote is True
    assert verdict.reasons == ()
    assert verdict.pit_status == "FAIL"
    assert verdict.pit_reasons != ()


def test_pit_gate_enforce_blocks_on_fail() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    current = _pit_candidate()
    report = _pit_report(native_mean=-5.0, native_ic=-0.01)
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, current, _pit_eval_frame(), pit_report=report, pit_gate=_pit_gate("enforce")
    )

    assert verdict.promote is False
    assert verdict.pit_status == "FAIL"
    assert any(r.startswith("pit gate FAIL") for r in verdict.reasons)


def test_pit_gate_missing_and_stale_reports() -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from src.ml import retrain_gate

    candidate = _pit_candidate()
    stale_at = (datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=15)).isoformat()

    missing = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=None, pit_gate=_pit_gate("enforce")
    )
    assert missing.pit_status == "MISSING"
    assert missing.promote is False

    stale = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(),
        pit_report=_pit_report(generated_at=stale_at), pit_gate=_pit_gate("enforce"),
    )
    assert stale.pit_status == "STALE"
    assert stale.promote is False


def test_pit_gate_mismatch_names_fields() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate(model_params={"n_estimators": 11}, seeds=[2])
    report = _pit_report()
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=report, pit_gate=_pit_gate("enforce")
    )

    assert verdict.pit_status == "MISMATCH"
    assert any("model_params" in r for r in verdict.pit_reasons)
    assert any("seeds" in r for r in verdict.pit_reasons)
    assert verdict.promote is False


def test_pit_gate_insufficient_report_blocks_in_enforce() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    report = _pit_report(status="INSUFFICIENT_DAYS")
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=report, pit_gate=_pit_gate("enforce")
    )

    assert verdict.pit_status == "INSUFFICIENT"
    assert verdict.promote is False


def test_pit_gate_pass() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    current = _pit_candidate()
    report = _pit_report(native_mean=5.0, native_ic=0.02)
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, current, _pit_eval_frame(), pit_report=report, pit_gate=_pit_gate("enforce")
    )

    assert verdict.pit_status == "PASS"
    assert verdict.promote is True


def test_pit_gate_status_survives_early_return() -> None:
    import pandas as pd

    from src.ml import retrain_gate

    candidate = _pit_candidate()
    report = _pit_report()
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, None, pd.DataFrame({"date": [], "f1": [], "f2": []}),
        pit_report=report, pit_gate=_pit_gate("advisory"),
    )

    assert verdict.promote is False
    assert verdict.pit_status == "PASS"


def test_pit_gate_rejects_naive_now() -> None:
    from datetime import datetime

    import pytest

    from src.ml import retrain_gate

    candidate = _pit_candidate()
    with pytest.raises(ValueError, match="aware"):
        retrain_gate.evaluate_retrain_promotion(
            candidate, None, _pit_eval_frame(), pit_report=_pit_report(),
            pit_gate=_pit_gate("advisory"), now=datetime(2026, 10, 4),
        )


def test_pit_gate_mismatch_strategy_and_top_k() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate(strategy_id="OTHER", top_k=5)
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=_pit_report(), pit_gate=_pit_gate("enforce")
    )

    assert verdict.pit_status == "MISMATCH"
    assert any("strategy_id" in r for r in verdict.pit_reasons)
    assert any("top_k" in r for r in verdict.pit_reasons)
    assert verdict.promote is False


def test_pit_gate_mismatch_missing_and_differing_fields() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    for key in ("select_universe", "model_params", "seeds"):
        candidate = {k: v for k, v in candidate.items() if k != key}
    candidate["feature_contract_version"] = 123
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=_pit_report(), pit_gate=_pit_gate("enforce")
    )

    assert verdict.pit_status == "MISMATCH"
    for name in ("select_universe", "model_params", "seeds", "feature contract version"):
        assert any(name in r for r in verdict.pit_reasons), name
    assert verdict.promote is False


def test_pit_gate_mismatch_screen_and_contract_values() -> None:
    import dataclasses

    from src.ml import retrain_gate
    from src.strategy.contract import COST_AWARE_UNIVERSE

    candidate = _pit_candidate(
        select_universe=dataclasses.asdict(COST_AWARE_UNIVERSE),
        feature_contract_version="0",
    )
    verdict = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=_pit_report(), pit_gate=_pit_gate("enforce")
    )

    assert verdict.pit_status == "MISMATCH"
    assert any("select_universe" in r for r in verdict.pit_reasons)
    assert any("feature contract version" in r for r in verdict.pit_reasons)


def test_pit_gate_malformed_generated_at_is_advisory_status_and_blocks_only_in_enforce() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.ml import retrain_gate

    candidate = _pit_candidate()
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    for generated_at, needle in (("not-a-date", "ISO-8601"), ("2026-10-04T00:00:00", "timezone-aware")):
        advisory = retrain_gate.evaluate_retrain_promotion(
            candidate, _pit_candidate(), _pit_eval_frame(), pit_report=_pit_report(generated_at=generated_at),
            pit_gate=_pit_gate("advisory"), now=now,
        )
        assert advisory.promote is True
        assert advisory.pit_status == "MALFORMED"
        assert any(needle in r for r in advisory.pit_reasons)
        enforced = retrain_gate.evaluate_retrain_promotion(
            candidate, _pit_candidate(), _pit_eval_frame(), pit_report=_pit_report(generated_at=generated_at),
            pit_gate=_pit_gate("enforce"), now=now,
        )
        assert enforced.promote is False
        assert enforced.pit_status == "MALFORMED"


def test_pit_gate_max_haircut_ceiling() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    current = _pit_candidate()
    report = _pit_report(native_mean=5.0, native_ic=0.02)

    over = retrain_gate.evaluate_retrain_promotion(
        candidate, current, _pit_eval_frame(), pit_report=report,
        pit_gate=_pit_gate("enforce", max_haircut_bp=1.0),
    )
    assert over.pit_status == "FAIL"
    assert over.promote is False

    under = retrain_gate.evaluate_retrain_promotion(
        candidate, current, _pit_eval_frame(), pit_report=report,
        pit_gate=_pit_gate("enforce", max_haircut_bp=100.0),
    )
    assert under.pit_status == "PASS"
    assert under.promote is True

