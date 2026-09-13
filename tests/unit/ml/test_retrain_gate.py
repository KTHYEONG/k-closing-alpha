def test_evaluate_retrain_promotion_promotes_consistent_candidate() -> None:
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.ml import retrain_gate
    from src.strategy.contract import COST_AWARE_UNIVERSE

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(COST_AWARE_UNIVERSE)}

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
    from src.strategy.contract import COST_AWARE_UNIVERSE

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(COST_AWARE_UNIVERSE)}

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
    from src.strategy.contract import COST_AWARE_UNIVERSE

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(COST_AWARE_UNIVERSE)}

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
    from src.strategy.contract import COST_AWARE_UNIVERSE

    class _ColumnModel:
        def __init__(self, column: str, noise: np.ndarray | None = None) -> None:
            self.column = column
            self.noise = noise

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            values = features[self.column].to_numpy(dtype=np.float64)
            return values if self.noise is None else values + self.noise

    def _bundle(model, feature_cols=("f1", "f2")) -> dict:
        return {"feature_cols": list(feature_cols), "return_model": model,
                "select_universe": dataclasses.asdict(COST_AWARE_UNIVERSE)}

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
    from src.ml.topk_ranker_research import save_production_bundle

    export_dir = str(tmp_path / "topk_ranker")

    # Given/When/Then: 발행 전
    assert retrain_gate.load_current_bundle(export_dir) is None

    # When: 원자적 저장 후
    path = save_production_bundle({"feature_cols": ["f1"], "top_k": 3}, export_dir=export_dir)

    # Then
    assert retrain_gate.load_current_bundle(export_dir) == {"feature_cols": ["f1"], "top_k": 3}
    assert not (tmp_path / "topk_ranker" / "sizing_pipeline_bundle.joblib.tmp").exists()
    assert path.endswith("sizing_pipeline_bundle.joblib")

