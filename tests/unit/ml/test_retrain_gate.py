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


def _recon_cert_payload(*, verdict="ADOPT", reasons=(), bindings=None, generated_at="2026-10-05T00:00:00+09:00", **overrides):
    from src.ml.pit_report import CalibrationStability, PairedDelta, ReconstructionCertification

    base: dict = {
        "generated_at": generated_at,
        "exact_dir": "exact",
        "recon_dir": "recon",
        "paired_days": ("2026-03-02",),
        "dropped_days": (),
        "coverage_improvement": PairedDelta(delta=4.0, ci_low=2.0, ci_high=6.0, p_value=0.001, n_days=60),
        "reconstruction_feature": PairedDelta(delta=0.2, ci_low=-1.0, ci_high=1.4, p_value=0.6, n_days=60),
        "stability": CalibrationStability(
            passed=True, rel_err_p90=0.10, coverage=0.80, bias_drift=0.01,
            alpha_at_boundary=False, n_holdout_rows=40, detail="",
        ),
        "coverage_by_year_and_basis": {},
        "bindings": bindings,
        "fidelity": {
            "chg_rank_correlation": 0.99,
            "tv_rank_correlation": 0.98,
            "top_k_overlap": 0.97,
            "n_paired_days": 60.0,
            "n_symbol_days": 600.0,
            "n_no_share": 2.0,
        },
        "holdout_start": "2026-08-25",
        "holdout_end": "2026-08-31",
        "gate_config": {"feature_margin_bp": 5.0},
        "gate_verdict": verdict,
        "gate_reasons": tuple(reasons),
    }
    base.update(overrides)
    return ReconstructionCertification(**base)


def _write_config_file(path, content=b'{"ewma_alpha": 0.5}'):
    path.write_bytes(content)
    return path


def _stage_recon_selection(
    live_dir, *, config_path, arm1_report=None, arm2_report=None, verdict="ADOPT", reasons=(),
    arm2=True, corrupt=None, generated_at=None,
):
    import hashlib

    import pandas as pd

    from src.ml.pit_report import (
        RECON_ARM_DIRNAME,
        CertificationBindings,
        save_pit_haircut_report,
        save_reconstruction_certification,
    )

    live_dir.mkdir(parents=True, exist_ok=True)
    if generated_at is None:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        generated_at = (datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(seconds=1)).isoformat()
    eff_arm2 = arm2_report if arm2_report is not None else _pit_report(native_mean=9.0)
    arm_dir = live_dir.parent / RECON_ARM_DIRNAME
    if arm2:
        arm_dir.mkdir(exist_ok=True)
        if corrupt == "arm2":
            (arm_dir / "pit_haircut_report.json").write_text("{bad", encoding="utf-8")
            arm2_sha = "e" * 64
        else:
            save_pit_haircut_report(eff_arm2, pd.DataFrame({"date": []}), out_dir=arm_dir)
            arm2_sha = hashlib.sha256((arm_dir / "pit_haircut_report.json").read_bytes()).hexdigest()
    else:
        arm2_sha = "e" * 64
    config_sha = hashlib.sha256(bytes(config_path.read_bytes())).hexdigest() if config_path.exists() else "a" * 64
    bindings = CertificationBindings(
        decomposition_config_sha256=config_sha,
        calibration_table_sha256="b" * 64,
        exact_report_sha256="c" * 64,
        recon_report_sha256=arm2_sha,
    )
    save_reconstruction_certification(
        _recon_cert_payload(verdict=verdict, reasons=reasons, bindings=bindings, generated_at=generated_at),
        out_path=live_dir / "reconstruction_certification.json",
    )
    if corrupt == "cert":
        (live_dir / "reconstruction_certification.json").write_text("{bad", encoding="utf-8")
    return bindings


def _loader_kwargs(live_dir, config_path, arm1_report, max_age_days=14):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return {
        "arm1_report": arm1_report,
        "decomposition_config_path": config_path,
        "now": datetime.now(ZoneInfo("Asia/Seoul")),
        "max_age_days": max_age_days,
    }


def test_select_pit_report_consumes_arm2_only_on_adopt() -> None:
    from src.ml import retrain_gate

    arm1 = _pit_report(native_mean=-5.0, native_ic=-0.01)
    arm2 = _pit_report(native_mean=9.0, native_ic=0.05)

    selected, note, pit_arm = retrain_gate.select_pit_report(arm1)
    assert selected is arm1 and note == "" and pit_arm == "arm1"

    selected, note, pit_arm = retrain_gate.select_pit_report(
        arm1, recon_report=arm2, recon_adopted=True, recon_detail="verdict ADOPT")
    assert selected is arm2 and "arm-2" in note and pit_arm == "arm2"

    selected, note, pit_arm = retrain_gate.select_pit_report(arm1, recon_report=None, recon_adopted=True)
    assert selected is arm1 and "fallback" in note and pit_arm == "arm1"

    selected, note, pit_arm = retrain_gate.select_pit_report(
        arm1, recon_report={"not": "a report"}, recon_adopted=True)  # type: ignore[arg-type]
    assert selected is arm1 and "fallback" in note and pit_arm == "arm1"

    weak = _pit_report(native_mean=9.0, status="INSUFFICIENT_DAYS")
    selected, note, pit_arm = retrain_gate.select_pit_report(arm1, recon_report=weak, recon_adopted=True)
    assert selected is arm1 and "fallback" in note and pit_arm == "arm1"

    selected, note, pit_arm = retrain_gate.select_pit_report(
        arm1, recon_report=arm2, recon_adopted=False, recon_detail="REJECT: feature")
    assert selected is arm1 and "fallback" in note and "REJECT" in note and pit_arm == "arm1"

    selected, note, pit_arm = retrain_gate.select_pit_report(arm1, recon_report=arm2, recon_adopted=False)
    assert selected is arm1 and "fallback" in note and pit_arm == "arm1"

    selected, note, pit_arm = retrain_gate.select_pit_report(None)
    assert selected is None and pit_arm == "none"


def test_recon_selection_never_improves_on_reject_or_damage(tmp_path) -> None:
    from src.ml import retrain_gate

    arm1 = _pit_report(native_mean=9.0, native_ic=0.05)
    cfg = tmp_path / "decomp.json"
    _write_config_file(cfg)
    assert retrain_gate.load_recon_pit_selection(
        tmp_path / "absent", **_loader_kwargs(tmp_path / "absent", cfg, arm1)
    ) == (None, False, "")

    live = tmp_path / "live"
    _stage_recon_selection(live, config_path=cfg, verdict="REJECT", reasons=("feature significant",))
    report, adopted, detail = retrain_gate.load_recon_pit_selection(
        live, **_loader_kwargs(live, cfg, arm1))
    assert (report, adopted) == (None, False)
    assert "feature significant" in detail

    damaged = tmp_path / "damaged"
    _stage_recon_selection(damaged, config_path=cfg, corrupt="cert")
    report, adopted, detail = retrain_gate.load_recon_pit_selection(
        damaged, **_loader_kwargs(damaged, cfg, arm1))
    assert (report, adopted) == (None, False)
    assert "unreadable" in detail

    no_arm2 = tmp_path / "no_arm2" / "topk_ranker"
    _stage_recon_selection(no_arm2, config_path=cfg, arm2=False)
    report, adopted, detail = retrain_gate.load_recon_pit_selection(
        no_arm2, **_loader_kwargs(no_arm2, cfg, arm1))
    assert (report, adopted) == (None, False)
    assert "missing" in detail

    bad_arm2 = tmp_path / "bad_arm2"
    _stage_recon_selection(bad_arm2, config_path=cfg, corrupt="arm2")
    report, adopted, detail = retrain_gate.load_recon_pit_selection(
        bad_arm2, **_loader_kwargs(bad_arm2, cfg, arm1))
    assert (report, adopted) == (None, False)
    assert "binding mismatch" in detail or "unreadable" in detail or "missing" in detail

    good = tmp_path / "good"
    _stage_recon_selection(good, config_path=cfg, arm2_report=_pit_report(native_mean=9.0, native_ic=0.05))
    report, adopted, detail = retrain_gate.load_recon_pit_selection(
        good, **_loader_kwargs(good, cfg, _pit_report(native_mean=9.0, native_ic=0.05)))
    assert adopted is True and report is not None
    assert float(report.mean_net_bp["pit_native"]) == 9.0


def test_pit_gate_recon_fallback_matches_arm1_status() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    arm1 = _pit_report(native_mean=-5.0, native_ic=-0.01)
    arm2 = _pit_report(native_mean=9.0, native_ic=0.05)
    gate = _pit_gate("enforce")

    alone = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=arm1, pit_gate=gate)
    silent = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=arm1, pit_gate=gate,
        pit_recon_report=None, pit_recon_adopted=False, pit_recon_detail="")
    assert (silent.pit_status, silent.pit_reasons) == (alone.pit_status, alone.pit_reasons)
    for recon_kwargs in (
        {"pit_recon_report": arm2, "pit_recon_adopted": False, "pit_recon_detail": "REJECT: feature"},
        {"pit_recon_report": None, "pit_recon_adopted": True, "pit_recon_detail": "missing"},
    ):
        fallback = retrain_gate.evaluate_retrain_promotion(
            candidate, None, _pit_eval_frame(), pit_report=arm1, pit_gate=gate, **recon_kwargs)
        assert fallback.pit_status == alone.pit_status == "FAIL"
        assert fallback.promote is False
        assert any("arm-1 fallback" in r for r in fallback.pit_reasons)

    adopted = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=arm1, pit_gate=gate,
        pit_recon_report=arm2, pit_recon_adopted=True, pit_recon_detail="verdict ADOPT")
    assert adopted.pit_status == "PASS"
    assert adopted.promote is True
    assert any("arm-2" in r for r in adopted.pit_reasons)



def test_adopt_verdict_contradicting_its_own_evidence_is_not_trusted(tmp_path) -> None:
    """The ADOPT string is re-derived: reasons present, failed stability or a CI touching zero fall back to arm 1."""
    import dataclasses
    import json

    from src.ml import retrain_gate
    from src.ml.pit_report import CalibrationStability, PairedDelta, save_reconstruction_certification

    from src.ml.pit_report import CertificationBindings

    cfg = tmp_path / "decomp.json"
    _write_config_file(cfg)
    arm1 = _pit_report(native_mean=9.0, native_ic=0.05)
    dummy = CertificationBindings(
        decomposition_config_sha256="a" * 64, calibration_table_sha256="b" * 64,
        exact_report_sha256="c" * 64, recon_report_sha256="d" * 64,
    )
    base = _recon_cert_payload(bindings=dummy)
    variants = {
        "reasons": dataclasses.replace(base, gate_reasons=("coverage not improved",)),
        "stability": dataclasses.replace(
            base,
            stability=CalibrationStability(
                passed=False, rel_err_p90=0.3, coverage=0.8, bias_drift=0.01,
                alpha_at_boundary=False, n_holdout_rows=40, detail="",
            ),
        ),
        "ci_zero": dataclasses.replace(
            base, coverage_improvement=PairedDelta(delta=4.0, ci_low=0.0, ci_high=6.0, p_value=0.2, n_days=60)
        ),
        "ci_nan": dataclasses.replace(
            base, coverage_improvement=PairedDelta(delta=4.0, ci_low=float("nan"), ci_high=6.0, p_value=0.2, n_days=60)
        ),
    }
    for name, cert in variants.items():
        live = tmp_path / name
        _stage_recon_selection(live, config_path=cfg)
        save_reconstruction_certification(cert, out_path=live / "reconstruction_certification.json")
        report, adopted, detail = retrain_gate.load_recon_pit_selection(
            live, **_loader_kwargs(live, cfg, arm1))
        assert (report, adopted) == (None, False), name
        assert detail, name
    overflow = tmp_path / "overflow"
    _stage_recon_selection(overflow, config_path=cfg)
    path = overflow / "reconstruction_certification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stability"]["n_holdout_rows"] = 1e999
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert retrain_gate.load_recon_pit_selection(
        overflow, **_loader_kwargs(overflow, cfg, arm1))[:2] == (None, False)
    stringy = tmp_path / "stringy"
    _stage_recon_selection(stringy, config_path=cfg)
    path = stringy / "reconstruction_certification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stability"]["passed"] = "false"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert retrain_gate.load_recon_pit_selection(
        stringy, **_loader_kwargs(stringy, cfg, arm1))[:2] == (None, False)


def test_recon_selection_binding_mismatches_fail_closed(tmp_path) -> None:
    from src.ml import retrain_gate

    cfg = tmp_path / "decomp.json"
    _write_config_file(cfg, b'{"v": 1}')
    arm1 = _pit_report(native_mean=9.0, native_ic=0.05)
    live = tmp_path / "bound"
    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    good_kwargs = _loader_kwargs(live, cfg, arm1)
    assert retrain_gate.load_recon_pit_selection(live, **good_kwargs)[1] is True

    _write_config_file(cfg, b'{"v": 2}')
    report, adopted, detail = retrain_gate.load_recon_pit_selection(live, **_loader_kwargs(live, cfg, arm1))
    assert adopted is False and "binding mismatch" in detail

    _write_config_file(cfg, b'{"v": 1}')
    other_arm1 = _pit_report(native_mean=9.0, native_ic=0.05, model_params={"n_estimators": 11})
    report, adopted, detail = retrain_gate.load_recon_pit_selection(live, **_loader_kwargs(live, cfg, other_arm1))
    assert adopted is False and "identity" in detail

    report, adopted, detail = retrain_gate.load_recon_pit_selection(live, **_loader_kwargs(live, cfg, None))
    assert adopted is False and "arm-1" in detail

    from datetime import timedelta

    stale_kwargs = dict(_loader_kwargs(live, cfg, arm1))
    stale_kwargs["now"] = stale_kwargs["now"] + timedelta(days=30)
    report, adopted, detail = retrain_gate.load_recon_pit_selection(live, **stale_kwargs)
    assert adopted is False and "stale" in detail


def test_recon_adopt_is_recomputed_not_trusted(tmp_path) -> None:
    import dataclasses

    from src.ml import retrain_gate

    cfg = tmp_path / "decomp.json"
    _write_config_file(cfg)
    arm1 = _pit_report(native_mean=9.0, native_ic=0.05)
    live = tmp_path / "recomputed"
    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    from src.ml.pit_report import load_reconstruction_certification

    stored = load_reconstruction_certification(live / "reconstruction_certification.json")
    weak_fidelity = dict(stored.fidelity or {})
    weak_fidelity["top_k_overlap"] = 0.50
    weak = dataclasses.replace(stored, fidelity=weak_fidelity)
    from src.ml.pit_report import save_reconstruction_certification

    save_reconstruction_certification(weak, out_path=live / "reconstruction_certification.json")
    report, adopted, detail = retrain_gate.load_recon_pit_selection(
        live, **_loader_kwargs(live, cfg, arm1))
    assert adopted is False and "top_k_overlap" in detail


def test_recon_selection_loader_fail_closed_branches(tmp_path) -> None:
    from datetime import datetime

    from src.ml import retrain_gate

    cfg = tmp_path / "decomp.json"
    _write_config_file(cfg)
    arm1 = _pit_report(native_mean=9.0, native_ic=0.05)
    live = tmp_path / "branches"
    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    kwargs = _loader_kwargs(live, cfg, arm1)
    import json as json_lib

    path = live / "reconstruction_certification.json"
    payload = json_lib.loads(path.read_text(encoding="utf-8"))
    payload["stability"] = None
    path.write_text(json_lib.dumps(payload), encoding="utf-8")
    assert "incomplete" in retrain_gate.load_recon_pit_selection(live, **kwargs)[2]

    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    path = live / "reconstruction_certification.json"
    payload = json_lib.loads(path.read_text(encoding="utf-8"))
    payload["fidelity"]["tv_rank_correlation"] = "bad"
    path.write_text(json_lib.dumps(payload), encoding="utf-8")
    assert retrain_gate.load_recon_pit_selection(live, **kwargs)[1] is False

    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    payload = json_lib.loads(path.read_text(encoding="utf-8"))
    payload["generated_at"] = "not-a-date"
    path.write_text(json_lib.dumps(payload), encoding="utf-8")
    assert "ISO-8601" in retrain_gate.load_recon_pit_selection(live, **kwargs)[2]

    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    payload = json_lib.loads(path.read_text(encoding="utf-8"))
    payload["generated_at"] = "2026-10-05T00:00:00"
    path.write_text(json_lib.dumps(payload), encoding="utf-8")
    assert "timezone-aware" in retrain_gate.load_recon_pit_selection(live, **kwargs)[2]

    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    naive_kwargs = dict(kwargs, now=datetime(2026, 10, 5))
    assert "timezone-aware" in retrain_gate.load_recon_pit_selection(live, **naive_kwargs)[2]

    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    payload = json_lib.loads(path.read_text(encoding="utf-8"))
    del payload["bindings"]
    path.write_text(json_lib.dumps(payload), encoding="utf-8")
    assert "unreadable" in retrain_gate.load_recon_pit_selection(live, **kwargs)[2]

    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
    missing_cfg = tmp_path / "absent.json"
    missing_kwargs = dict(kwargs, decomposition_config_path=missing_cfg)
    assert "unreadable" in retrain_gate.load_recon_pit_selection(live, **missing_kwargs)[2]

    weak_arm2 = _pit_report(native_mean=9.0, status="INSUFFICIENT_DAYS")
    _stage_recon_selection(live, config_path=cfg, arm2_report=weak_arm2)
    assert "no usable paired evidence" in retrain_gate.load_recon_pit_selection(live, **kwargs)[2]

    _stage_recon_selection(live, config_path=cfg, arm2_report=arm1, generated_at="2099-01-01T00:00:00+00:00")
    assert "future" in retrain_gate.load_recon_pit_selection(live, **kwargs)[2]

    for metric, value in (("rel_err_p90", None), ("coverage", 0.1), ("bias_drift", 0.9),
                          ("alpha_at_boundary", True), ("n_holdout_rows", 0)):
        _stage_recon_selection(live, config_path=cfg, arm2_report=arm1)
        payload = json_lib.loads(path.read_text(encoding="utf-8"))
        payload["stability"][metric] = value
        path.write_text(json_lib.dumps(payload), encoding="utf-8")
        assert retrain_gate.load_recon_pit_selection(live, **kwargs)[1] is False


def test_promotion_verdict_names_scored_arm() -> None:
    from src.ml import retrain_gate

    candidate = _pit_candidate()
    arm1 = _pit_report(native_mean=-5.0, native_ic=-0.01)
    arm2 = _pit_report(native_mean=9.0, native_ic=0.05)
    gate = _pit_gate("enforce")
    fallback = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=arm1, pit_gate=gate,
        pit_recon_report=arm2, pit_recon_adopted=False, pit_recon_detail="REJECT",
    )
    assert fallback.pit_arm == "arm1"
    adopted = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=arm1, pit_gate=gate,
        pit_recon_report=arm2, pit_recon_adopted=True, pit_recon_detail="verdict ADOPT",
    )
    assert adopted.pit_arm == "arm2"
    assert adopted.pit_status == "PASS"
    missing = retrain_gate.evaluate_retrain_promotion(
        candidate, None, _pit_eval_frame(), pit_report=None, pit_gate=gate)
    assert missing.pit_arm == "none"
