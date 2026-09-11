"""일일 예측 진입점 wiring 및 Fast Inference 단위 테스트 (live serving 경로).

SCENARIO_MODEL_PIPELINE_TRAIN_EVAL 의 wiring 단계가 일일 예측 진입점에
연결되었는지 확인하고, 저장된 모델 아티팩트 기반 Fast Inference 동작과
단일 BUY/ABSTAIN 결정을 검증합니다. 학습/재학습 관련 케이스는
``legacy/tests/unit/daily/test_predict.py`` 로 이동되었습니다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.daily.predict as predict

from tests.unit.serving.realtime.fixtures import (
    build_fixed_serving_bundle,
    daily_snapshot_df,
    snapshot_feature_cols,
)

FEATURE_COLS = snapshot_feature_cols()


def test_legacy_gmm_logic_removed() -> None:
    """레거시 GMM/Static 의사결정 및 하드코딩 Safety Floor 가 제거되었는지 확인한다."""
    assert not hasattr(predict, "get_decision_batch")
    assert not hasattr(predict, "GaussianMixture")
    assert not hasattr(predict, "HAS_SKLEARN")
    assert not hasattr(predict, "SAFETY_MAX_FLOOR")
    assert not hasattr(predict, "SAFETY_EXPAND_FLOOR")
    assert not hasattr(predict, "ABSOLUTE_MIN_SCORE")
    assert not hasattr(predict, "MIN_SAMPLES_FOR_GMM")


def test_predict_exposes_only_live_control_apis() -> None:
    """예측 진입점은 학습/재학습 진입점을 노출하지 않습니다."""
    assert callable(predict.load_model_bundle)
    assert not hasattr(predict, "run_model_pipeline")
    assert not hasattr(predict, "train_and_save_real_model_bundle")
    assert not hasattr(predict, "ensure_valid_model_bundle")


def test_main_automated_mode_prints_only_topk_decision_table(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: a populated top-3 sleeve
    sleeve_df = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "name": ["AAA", "BBB", "CCC"],
        "pred": [0.021, 0.017, 0.011],
        "allocation": [1.0 / 3.0] * 3,
    })
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", Mock(return_value=sleeve_df))
    persist_mock = Mock(return_value=3)
    monkeypatch.setattr(predict_mod, "persist_topk_decision", persist_mock)
    print_table_mock = Mock()
    monkeypatch.setattr(predict_mod, "print_table", print_table_mock)

    # When
    predict_mod.main()

    # Then: the reranker table is the single decision surface, and the decision was persisted once
    assert print_table_mock.call_count == 1
    rows = print_table_mock.call_args_list[0].args[0]
    assert [r["Code"] for r in rows] == ["000001", "000002", "000003"]
    assert rows[0]["Alloc%"] == pytest.approx(33.3, abs=0.1)
    persist_mock.assert_called_once()


def test_main_automated_mode_warns_and_prints_nothing_when_no_decision(monkeypatch, caplog) -> None:
    import logging
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: the sleeve yields no actionable decision
    monkeypatch.setattr(
        predict_mod, "run_topk_ranker_sleeve", Mock(return_value=pd.DataFrame())
    )
    persist_mock = Mock()
    monkeypatch.setattr(predict_mod, "persist_topk_decision", persist_mock)
    print_table_mock = Mock()
    monkeypatch.setattr(predict_mod, "print_table", print_table_mock)

    # When
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        predict_mod.main()

    # Then: silence is made explicit as a no-participation banner, and nothing is persisted
    print_table_mock.assert_not_called()
    persist_mock.assert_not_called()
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_load_daily_snapshot_reads_archive_store_and_zero_fills_code(monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: the archive store returns the day's wide snapshot in 100M-KRW units
    captured = {}

    def _fake_fetch(snapshot_date=None, **kwargs):
        captured["snapshot_date"] = snapshot_date
        return pd.DataFrame({"종목코드": [5930, "000660"], "거래대금": [500.0, 300.0], "admitted": [True, False]})

    monkeypatch.setattr(predict_mod, "fetch_archive_snapshot", _fake_fetch)

    # When
    out = predict_mod.load_daily_snapshot(pd.Timestamp("2026-09-09"))

    # Then
    assert captured["snapshot_date"] == "2026-09-09"
    assert out["종목코드"].tolist() == ["005930", "000660"]
    assert out["거래대금"].tolist() == [500.0, 300.0]
    assert out["admitted"].tolist() == [True, False]


def test_run_topk_ranker_sleeve_uses_stored_admitted_without_recompute(monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.daily.predict as predict_mod
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    # Given: a 4-name wide snapshot where the store already marked one rejected
    wide = pd.DataFrame({
        "종목코드": ["000001", "000002", "000003", "000004"],
        "종목명": ["AAA", "BBB", "CCC", "DDD"],
        "종가": [18000.0, 18100.0, 17900.0, 30000.0],
        "전일종가": [17142.86, 17238.10, 17047.62, 28571.43],
        "고가": [18100.0, 18200.0, 18000.0, 30100.0],
        "저가": [17800.0, 17900.0, 17700.0, 29800.0],
        "시가": [17900.0, 18000.0, 17800.0, 29900.0],
        "거래량": [1_000_000, 900_000, 1_100_000, 800_000],
        "거래대금": [500.0, 450.0, 550.0, 400.0],
        "시가총액": [3000.0, 2800.0, 3200.0, 5000.0],
        "기관_순매수": [10.0, -5.0, 20.0, 8.0],
        "외국인_순매수": [5.0, 12.0, -3.0, 6.0],
        "시장구분": ["KOSPI", "KOSPI", "KOSPI", "KOSPI"],
        "kospi": [0.52] * 4,
        "kosdaq": [-0.31] * 4,
        "v_kospi": [15.2] * 4,
        "admitted": [True, True, True, False],
    })
    bundle = build_fixed_serving_bundle(list(FEATURE_COLS))
    bundle["top_k"] = 3

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)

    # When
    out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))

    # Then: the stored verdict alone decides eligibility
    assert len(out) == 3
    assert sorted(out["symbol"].tolist()) == ["000001", "000002", "000003"]
    assert np.allclose(out["allocation"].to_numpy(dtype=np.float64), 1.0 / 3.0)
    assert sorted(out["name"].tolist()) == ["AAA", "BBB", "CCC"]


def test_run_topk_ranker_sleeve_warns_when_bundle_missing(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    import src.daily.predict as predict_mod

    wide = pd.DataFrame({
        "종목코드": ["000001"], "종목명": ["AAA"], "종가": [18000.0], "전일종가": [17142.86],
        "고가": [18100.0], "저가": [17800.0], "시가": [17900.0], "거래량": [1_000_000],
        "거래대금": [500.0], "시가총액": [3000.0], "기관_순매수": [10.0], "외국인_순매수": [5.0],
        "시장구분": ["KOSPI"], "kospi": [0.52], "kosdaq": [-0.31], "v_kospi": [15.2],
        "admitted": [True],
    })
    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d: wide)

    def _missing(import_dir=None):
        raise FileNotFoundError("model artifact bundle not found")

    monkeypatch.setattr(predict_mod, "load_model_bundle", _missing)

    # When
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))

    # Then: fails soft, never silently
    assert out.empty
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_predict_module_drops_champion_surface() -> None:
    import src.daily.predict as predict_mod

    # Then: no champion grading/display helpers and no CSV loader remain
    for gone in (
        "load_condition_snapshot",
        "convert_amount_units_to_krw",
        "load_and_preprocess_data",
        "build_result_rows",
        "select_top_actionable",
        "explain_predictions_with_shap",
        "load_label_encoder_map",
        "LABEL_ENCODER_MAP",
        "predict_daily_sizing",
    ):
        assert not hasattr(predict_mod, gone), gone


def test_persist_topk_decision_writes_new_parquet_and_returns_row_count(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path)

    sleeve_df = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "name": ["AAA", "BBB"],
        "pred": [0.02, 0.01],
        "allocation": [0.5, 0.5],
    })

    written = predict_mod.persist_topk_decision(pd.Timestamp("2026-09-10"), sleeve_df)

    assert written == 2
    saved = pd.read_parquet(tmp_path / "topk_decisions.parquet")
    assert len(saved) == 2
    assert set(saved["symbol"]) == {"000001", "000002"}
    assert (saved["decision_date"] == "2026-09-10").all()
    assert "decided_at" in saved.columns
    assert "bundle_dir" in saved.columns


def test_persist_topk_decision_returns_zero_for_empty_sleeve(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path)

    written = predict_mod.persist_topk_decision(pd.Timestamp("2026-09-10"), pd.DataFrame())

    assert written == 0
    assert not (tmp_path / "topk_decisions.parquet").exists()


def test_persist_topk_decision_dedups_same_date_symbol_on_rerun(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path)

    first = pd.DataFrame({"symbol": ["000001"], "name": ["AAA"], "pred": [0.02], "allocation": [1.0]})
    predict_mod.persist_topk_decision(pd.Timestamp("2026-09-10"), first)

    second = pd.DataFrame({"symbol": ["000001"], "name": ["AAA-updated"], "pred": [0.03], "allocation": [1.0]})
    predict_mod.persist_topk_decision(pd.Timestamp("2026-09-10"), second)

    saved = pd.read_parquet(tmp_path / "topk_decisions.parquet")
    assert len(saved) == 1
    assert saved["name"].iloc[0] == "AAA-updated"


def test_persist_topk_decision_recovers_from_corrupt_existing_parquet(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path)
    target = tmp_path / "topk_decisions.parquet"
    target.write_text("not a valid parquet file")

    sleeve_df = pd.DataFrame({"symbol": ["000001"], "name": ["AAA"], "pred": [0.02], "allocation": [1.0]})

    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        written = predict_mod.persist_topk_decision(pd.Timestamp("2026-09-10"), sleeve_df)

    assert written == 1
    saved = pd.read_parquet(target)
    assert len(saved) == 1
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)


def _sleeve_wide_and_history():
    import numpy as np
    import pandas as pd

    wide = pd.DataFrame({
        "종목코드": ["000001", "000002", "000003", "000004"],
        "종목명": ["AAA", "BBB", "CCC", "DDD"],
        "종가": [18000.0, 18100.0, 17900.0, 30000.0],
        "전일종가": [17142.86, 17238.10, 17047.62, 28571.43],
        "고가": [18100.0, 18200.0, 18000.0, 30100.0],
        "저가": [17800.0, 17900.0, 17700.0, 29800.0],
        "시가": [17900.0, 18000.0, 17800.0, 29900.0],
        "거래량": [1_000_000, 900_000, 1_100_000, 800_000],
        "거래대금": [500.0, 450.0, 550.0, 400.0],
        "시가총액": [3000.0, 2800.0, 3200.0, 5000.0],
        "기관_순매수": [10.0, -5.0, 20.0, 8.0],
        "외국인_순매수": [5.0, 12.0, -3.0, 6.0],
        "시장구분": ["KOSPI", "KOSPI", "KOSPI", "KOSPI"],
        "kospi": [0.52] * 4,
        "kosdaq": [-0.31] * 4,
        "v_kospi": [15.2] * 4,
        "admitted": [True, True, True, False],
    })
    decision = pd.Timestamp("2026-09-09")
    rng = np.random.default_rng(4)
    hist = pd.DataFrame([
        {"date": d, "symbol": s, "open": 17000.0, "close": 17000.0 * (1.0 + 0.02 * float(rng.normal())),
         "prev_close": 17000.0, "volume": 1e6, "inst_netbuy": 1e7, "foreign_netbuy": -1e7}
        for s in wide["종목코드"]
        for d in pd.bdate_range(end=decision - pd.Timedelta(days=1), periods=70)
    ])
    return wide, hist, decision


def test_run_topk_ranker_sleeve_loads_history_when_bundle_needs_it(monkeypatch) -> None:
    import src.daily.predict as predict_mod
    import src.ml.topk_history_features as thf
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    # Given: a v2 bundle whose feature_cols include history features
    wide, hist, decision = _sleeve_wide_and_history()
    bundle = build_fixed_serving_bundle(list(thf.TOPK_FEATURE_COLS_V2))
    bundle["top_k"] = 3
    calls = []

    def _fake_loader(d):
        calls.append(d)
        return hist

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)
    monkeypatch.setattr(thf, "load_serving_price_history", _fake_loader)

    # When
    out = predict_mod.run_topk_ranker_sleeve(decision)

    # Then: history is read once for the decision day and the admitted top-3 is selected
    assert calls == [decision]
    assert len(out) == 3
    assert sorted(out["symbol"].tolist()) == ["000001", "000002", "000003"]
    assert sorted(out["name"].tolist()) == ["AAA", "BBB", "CCC"]


def test_run_topk_ranker_sleeve_stale_history_yields_no_decision(monkeypatch, caplog) -> None:
    import logging

    import src.daily.predict as predict_mod
    import src.ml.topk_history_features as thf
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    wide, _hist, decision = _sleeve_wide_and_history()
    bundle = build_fixed_serving_bundle(list(thf.TOPK_FEATURE_COLS_V2))
    bundle["top_k"] = 3

    def _stale(_d):
        raise ValueError("stale price_history: latest=2026-09-07 < prev_trading_day=2026-09-08")

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)
    monkeypatch.setattr(thf, "load_serving_price_history", _stale)

    # When
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        out = predict_mod.run_topk_ranker_sleeve(decision)

    # Then: fail-closed no participation, never a decision on stale features
    assert out.empty
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_run_topk_ranker_sleeve_skips_history_for_v1_bundle(monkeypatch) -> None:
    import src.daily.predict as predict_mod
    import src.ml.topk_history_features as thf
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    wide, _hist, decision = _sleeve_wide_and_history()
    bundle = build_fixed_serving_bundle(list(FEATURE_COLS))
    bundle["top_k"] = 3

    def _must_not_load(_d):
        raise AssertionError("v1 bundle must not read price_history")

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)
    monkeypatch.setattr(thf, "load_serving_price_history", _must_not_load)

    # When
    out = predict_mod.run_topk_ranker_sleeve(decision)

    # Then
    assert len(out) == 3

