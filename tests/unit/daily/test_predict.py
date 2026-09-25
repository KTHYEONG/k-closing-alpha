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


@pytest.fixture(autouse=True)
def _standard_session(monkeypatch) -> None:
    """Pre-gate scenarios run under a STANDARD session; gate scenarios inject their own resolver."""
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay, SessionKind

    def _resolve(trading_day, **_kwargs):
        return SessionDay(
            trading_date=trading_day,
            kind=SessionKind.STANDARD,
            clock=SessionClock.standard(trading_day),
            provenance="standard",
        )

    monkeypatch.setattr(predict, "resolve_session_day", _resolve)


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
    monkeypatch.setattr(predict_mod, "record_run_outcome", Mock())

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
    monkeypatch.setattr(predict_mod, "record_run_outcome", Mock())

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
    monkeypatch.setattr(predict_mod.settings, "COLLECTION_RAW_ENABLED", False)

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

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
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
    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)

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

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
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

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
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

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)
    monkeypatch.setattr(thf, "load_serving_price_history", _must_not_load)

    # When
    out = predict_mod.run_topk_ranker_sleeve(decision)

    # Then
    assert len(out) == 3


def test_restrict_to_rank_pool_drops_rows_outside_training_screen() -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    # Given: 학습 풀 4행 + 거래대금이 큰 음수 등락 행(Toss union형, 랭크만 왜곡)
    wide, _hist, decision = _sleeve_wide_and_history()
    extra = wide.iloc[[0]].copy()
    extra["종목코드"] = "000005"
    extra["종가"] = 9500.0
    extra["전일종가"] = 10000.0
    extra["고가"] = 9800.0
    extra["저가"] = 9400.0
    extra["시가"] = 9900.0
    extra["거래대금"] = 90000.0
    extra["admitted"] = False
    wide = pd.concat([wide, extra], ignore_index=True)

    # When
    out = predict_mod.restrict_to_rank_pool(wide, decision)

    # Then
    assert out["종목코드"].tolist() == ["000001", "000002", "000003", "000004"]
    assert out.index.tolist() == [0, 1, 2, 3]


def test_restrict_to_rank_pool_fails_closed_on_inconsistent_or_empty_snapshot() -> None:
    import pandas as pd
    import pytest

    import src.daily.predict as predict_mod

    # Given: 풀 밖(음수 등락)인데 admitted로 기록된 모순 행
    wide, _hist, decision = _sleeve_wide_and_history()
    bad = wide.iloc[[0]].copy()
    bad["종목코드"] = "000005"
    bad["종가"] = 9500.0
    bad["전일종가"] = 10000.0
    bad["고가"] = 9800.0
    bad["admitted"] = True
    wide = pd.concat([wide, bad], ignore_index=True)

    # When / Then
    with pytest.raises(ValueError, match="outside the training rank pool"):
        predict_mod.restrict_to_rank_pool(wide, decision)
    with pytest.raises(ValueError, match="empty"):
        predict_mod.restrict_to_rank_pool(wide.iloc[0:0], decision)


def test_run_topk_ranker_sleeve_ranks_within_training_pool(monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod
    import src.serving.realtime.features as features_mod
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    # Given: 학습 풀 4행 + 풀 밖 음수 등락 대형 거래대금 행
    wide, _hist, decision = _sleeve_wide_and_history()
    extra = wide.iloc[[0]].copy()
    extra["종목코드"] = "000005"
    extra["종목명"] = "EEE"
    extra["종가"] = 9500.0
    extra["전일종가"] = 10000.0
    extra["고가"] = 9800.0
    extra["거래대금"] = 90000.0
    extra["admitted"] = False
    wide = pd.concat([wide, extra], ignore_index=True)
    bundle = build_fixed_serving_bundle(list(FEATURE_COLS))
    bundle["top_k"] = 3
    seen = []
    real_build = features_mod.build_topk_ranker_features

    def _spy(df, decision_date, price_history=None):
        seen.append(df["종목코드"].tolist())
        return real_build(df, decision_date, price_history=price_history)

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)
    monkeypatch.setattr(features_mod, "build_topk_ranker_features", _spy)

    # When
    out = predict_mod.run_topk_ranker_sleeve(decision)

    # Then
    assert seen == [["000001", "000002", "000003", "000004"]]
    assert sorted(out["symbol"].tolist()) == ["000001", "000002", "000003"]


def test_load_topk_decision_returns_rows_for_decision_date_only(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path)

    # Given: 저장소 부재
    assert predict_mod.load_topk_decision(pd.Timestamp("2026-09-14")).empty

    # Given: 다른 날짜 1행 + 당일 재실행으로 중복된 000660
    pd.DataFrame({
        "decision_date": ["2026-09-11", "2026-09-14", "2026-09-14", "2026-09-14"],
        "symbol": ["000001", "000660", "005930", "000660"],
        "allocation": [1.0 / 3.0] * 4,
        "pred": [0.1, 0.2, 0.3, 0.25],
    }).to_parquet(tmp_path / "topk_decisions.parquet", index=False)

    # When
    out = predict_mod.load_topk_decision(pd.Timestamp("2026-09-14"))

    # Then
    assert out["symbol"].tolist() == ["005930", "000660"]
    assert out["pred"].tolist() == [0.3, 0.25]
    assert out.index.tolist() == [0, 1]


def test_restrict_to_rank_pool_raises_when_no_row_passes_training_screen() -> None:
    import pytest

    import src.daily.predict as predict_mod

    # Given: 전 종목이 음수 등락이라 학습 스크린 밖, admitted 없음
    wide, _hist, decision = _sleeve_wide_and_history()
    wide["종가"] = 9500.0
    wide["전일종가"] = 10000.0
    wide["고가"] = 9800.0
    wide["admitted"] = False

    # When / Then
    with pytest.raises(ValueError, match="rank pool is empty"):
        predict_mod.restrict_to_rank_pool(wide, decision)


def test_run_topk_ranker_sleeve_reports_failure_to_callback(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    import src.daily.predict as predict_mod

    wide = pd.DataFrame({"종목코드": ["000001"], "종목명": ["AAA"], "admitted": [True]})
    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
    monkeypatch.setattr(predict_mod, "restrict_to_rank_pool", lambda w, _d: w)

    def _missing(import_dir=None):
        raise FileNotFoundError("model artifact bundle not found")

    monkeypatch.setattr(predict_mod, "load_model_bundle", _missing)
    failures: list[Exception] = []

    # When
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"), on_failure=failures.append)

    # Then
    assert out.empty
    assert len(failures) == 1 and isinstance(failures[0], FileNotFoundError)
    messages = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("stage=topk_sleeve" in m and "status=NO_DECISION" in m for m in messages)
    assert not any("\x1b[" in m for m in messages)


def test_run_automated_topk_decision_records_no_decision_on_systemic_failure(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    def _sleeve(_d, *, on_failure=None, on_rank_pool=None):
        on_failure(ValueError("stale price_history: latest=2026-09-10"))
        return pd.DataFrame()

    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", _sleeve)
    persist_mock = Mock()
    monkeypatch.setattr(predict_mod, "persist_topk_decision", persist_mock)
    recorder = Mock()

    # When
    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2026-09-14"), record_fn=recorder, trading_day_fn=lambda _d: True
    )

    # Then
    persist_mock.assert_not_called()
    recorder.assert_called_once()
    args, kwargs = recorder.call_args
    assert args == ("NO_DECISION",)
    assert kwargs["run_date"] == "2026-09-14"
    assert kwargs["reason"] == "ValueError: stale price_history: latest=2026-09-10"
    assert kwargs["metrics"] == {"n_picks": 0, "day": "trading"}


def test_run_automated_topk_decision_treats_holiday_failure_as_ok(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    def _sleeve(_d, *, on_failure=None, on_rank_pool=None):
        on_failure(ValueError("live_rows is empty; nothing to decide on"))
        return pd.DataFrame()

    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", _sleeve)
    monkeypatch.setattr(predict_mod, "persist_topk_decision", Mock())
    recorder = Mock()

    # When: 2026-09-25(금) 추석 연휴
    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2026-09-25"), record_fn=recorder, trading_day_fn=lambda _d: False
    )

    # Then
    args, kwargs = recorder.call_args
    assert args == ("OK",)
    assert kwargs["reason"] == "non_trading_day"
    assert kwargs["metrics"] == {"n_picks": 0, "day": "holiday"}


def test_run_automated_topk_decision_records_ok_for_empty_pool_and_picks(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    def _no_calendar(_d):
        raise AssertionError("calendar lookup only on failure")

    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", lambda _d, *, on_failure=None, on_rank_pool=None: pd.DataFrame())
    recorder = Mock()

    # When: 정상 무결정(admitted < top_k)
    predict_mod.run_automated_topk_decision(pd.Timestamp("2026-09-14"), record_fn=recorder, trading_day_fn=_no_calendar)

    # Then
    args, kwargs = recorder.call_args
    assert args == ("OK",)
    assert kwargs["reason"] == "admitted_below_top_k"
    assert kwargs["metrics"] == {"n_picks": 0}

    sleeve_df = pd.DataFrame(
        {"symbol": ["000001", "000002", "000003"], "name": ["A", "B", "C"], "pred": [0.02, 0.01, 0.005], "allocation": [1 / 3] * 3}
    )
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", lambda _d, *, on_failure=None, on_rank_pool=None: sleeve_df)
    persist_mock = Mock(return_value=3)
    monkeypatch.setattr(predict_mod, "persist_topk_decision", persist_mock)
    monkeypatch.setattr(predict_mod, "print_table", Mock())
    recorder_picks = Mock()

    # When: 정상 결정
    predict_mod.run_automated_topk_decision(pd.Timestamp("2026-09-14"), record_fn=recorder_picks, trading_day_fn=_no_calendar)

    # Then
    persist_mock.assert_called_once()
    args, kwargs = recorder_picks.call_args
    assert args == ("OK",)
    assert kwargs["reason"] == ""
    assert kwargs["metrics"] == {"n_picks": 3}


def test_predict_main_wires_run_outcome_recorder(monkeypatch) -> None:
    from unittest.mock import Mock

    import src.daily.predict as predict_mod

    captured: dict = {}

    def _fake_decision(decision_date, **kwargs):
        captured["date"] = decision_date
        captured.update(kwargs)

    recorder = Mock(return_value={})
    monkeypatch.setattr(predict_mod, "run_automated_topk_decision", _fake_decision)
    monkeypatch.setattr(predict_mod, "record_run_outcome", recorder)

    # When
    predict_mod.main()
    captured["record_fn"]("NO_DECISION", run_date="2026-09-14", reason="x", metrics={"n_picks": 0})

    # Then
    recorder.assert_called_once_with("predict", "NO_DECISION", run_date="2026-09-14", reason="x", metrics={"n_picks": 0})


def test_build_rank_pool_frame_ranks_by_pred_and_flags_selected() -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    # Given
    scored = pd.DataFrame({
        "symbol": ["000001", "000002", "000003", "000004"],
        "admitted": [True, True, False, True],
        "pred": [0.01, 0.03, 0.05, -0.02],
    })
    picks = pd.DataFrame({"symbol": ["000002", "000001", "000004"]})
    names = {"000001": "AAA", "000002": "BBB", "000003": "CCC", "000004": "DDD"}

    # When
    out = predict_mod.build_rank_pool_frame(scored, picks, names, "S@C")

    # Then: pred 내림차순 1-based 순위, 선정 여부, 이름, 모델 버전
    assert out["symbol"].tolist() == ["000003", "000002", "000001", "000004"]
    assert out["rank"].tolist() == [1, 2, 3, 4]
    assert out["selected"].tolist() == [False, True, True, True]
    assert out["name"].tolist() == ["CCC", "BBB", "AAA", "DDD"]
    assert set(out["model_version"]) == {"S@C"}
    assert "rank" not in scored.columns

    # And: 빈 픽(admitted < top_k)이면 전부 미선정
    none_selected = predict_mod.build_rank_pool_frame(scored, pd.DataFrame(), names, "S@C")
    assert not none_selected["selected"].any()
    assert len(none_selected) == 4


def test_persist_rank_pool_predictions_writes_and_dedups_per_date_symbol(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path)
    pool = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "pred": [0.02, 0.01],
        "rank": [1, 2],
        "selected": [True, False],
        "model_version": ["S@C", "S@C"],
    })

    # When: 같은 날 재실행 + 다른 날 기록
    assert predict_mod.persist_rank_pool_predictions(pd.Timestamp("2026-09-15"), pool, code_commit="abc123") == 2
    rerun = pool.assign(pred=[0.03, 0.01])
    assert predict_mod.persist_rank_pool_predictions(pd.Timestamp("2026-09-15"), rerun, code_commit="def456") == 2
    assert predict_mod.persist_rank_pool_predictions(pd.Timestamp("2026-09-16"), pool, code_commit="def456") == 2
    assert predict_mod.persist_rank_pool_predictions(pd.Timestamp("2026-09-16"), pool.iloc[0:0], code_commit="x") == 0

    # Then
    stored = pd.read_parquet(tmp_path / "rank_pool_predictions.parquet")
    assert len(stored) == 4
    day = stored[stored["decision_date"] == "2026-09-15"].sort_values("rank")
    assert day["pred"].tolist() == [0.03, 0.01]
    assert set(day["code_commit"]) == {"def456"}
    assert stored["decided_at"].notna().all()


def test_resolve_code_commit_prefers_kca_code_commit_env_over_git(monkeypatch) -> None:
    import subprocess

    import src.daily.predict as predict_mod

    monkeypatch.setenv("KCA_CODE_COMMIT", "  f7b15ea0c1d2 ")

    def _must_not_be_called(cmd, **kwargs):
        raise AssertionError("git must not be invoked when KCA_CODE_COMMIT is set")

    assert predict_mod.resolve_code_commit(run_fn=_must_not_be_called) == "f7b15ea0c1d2"


def test_resolve_code_commit_falls_back_to_git_when_env_unset(monkeypatch) -> None:
    import subprocess
    from pathlib import Path

    import src.daily.predict as predict_mod

    monkeypatch.delenv("KCA_CODE_COMMIT", raising=False)
    repo_root = str(Path(predict_mod.__file__).resolve().parents[2])

    def _ok(cmd, **kwargs):
        assert cmd == ["git", "rev-parse", "--short=12", "HEAD"]
        assert kwargs == {"cwd": repo_root, "capture_output": True, "text": True, "check": True, "timeout": 10}
        return subprocess.CompletedProcess(cmd, 0, stdout="b846de0abc12\n", stderr="")

    def _fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    def _missing(cmd, **kwargs):
        raise FileNotFoundError("git")

    def _slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 10)

    assert predict_mod.resolve_code_commit(run_fn=_ok) == "b846de0abc12"
    assert predict_mod.resolve_code_commit(run_fn=_fail) == "UNKNOWN"
    assert predict_mod.resolve_code_commit(run_fn=_missing) == "UNKNOWN"
    assert predict_mod.resolve_code_commit(run_fn=_slow) == "UNKNOWN"


def test_run_topk_ranker_sleeve_emits_scored_rank_pool_to_callback(monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.daily.predict as predict_mod
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    # Given: 4행 랭크풀 중 000004만 비적격
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
    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)
    pools: list[pd.DataFrame] = []

    # When
    out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"), on_rank_pool=pools.append)

    # Then: 픽에 모델 버전이 찍히고, 전체 풀이 1회 콜백된다
    assert len(out) == 3
    assert set(out["model_version"]) == {"UNKNOWN@UNKNOWN@UNKNOWN"}
    assert len(pools) == 1
    pool = pools[0]
    assert sorted(pool["symbol"].tolist()) == ["000001", "000002", "000003", "000004"]
    assert pool["rank"].tolist() == [1, 2, 3, 4]
    preds = pool["pred"].to_numpy(dtype=np.float64)
    assert (preds[:-1] >= preds[1:]).all()
    assert int(pool["selected"].sum()) == 3
    assert pool.loc[pool["symbol"] == "000004", "selected"].tolist() == [False]
    assert set(pool.loc[pool["selected"], "symbol"]) == set(out["symbol"])
    assert pool.set_index("symbol").loc["000001", "name"] == "AAA"
    assert set(pool["model_version"]) == {"UNKNOWN@UNKNOWN@UNKNOWN"}

    # And: 콜백이 없으면 동일한 픽만 반환
    again = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"))
    assert again["symbol"].tolist() == out["symbol"].tolist()


def test_run_automated_topk_decision_persists_rank_pool_with_code_commit(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod

    pool_df = pd.DataFrame({"symbol": ["000001"], "pred": [0.02], "rank": [1], "selected": [True], "model_version": ["S@C"]})
    sleeve_df = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"], "name": ["A", "B", "C"],
        "pred": [0.02, 0.01, 0.005], "allocation": [1 / 3] * 3,
    })

    def _no_calendar(_d):
        raise AssertionError("calendar lookup only on failure")

    def _sleeve(_d, *, on_failure=None, on_rank_pool=None):
        on_rank_pool(pool_df)
        return sleeve_df

    pool_persist = Mock(return_value=1)
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", _sleeve)
    monkeypatch.setattr(predict_mod, "persist_topk_decision", Mock(return_value=3))
    monkeypatch.setattr(predict_mod, "persist_rank_pool_predictions", pool_persist)
    monkeypatch.setattr(predict_mod, "resolve_code_commit", lambda: "abc123")
    monkeypatch.setattr(predict_mod, "print_table", Mock())

    # When: 정상 결정
    predict_mod.run_automated_topk_decision(pd.Timestamp("2026-09-15"), record_fn=Mock(), trading_day_fn=_no_calendar)

    # Then
    pool_persist.assert_called_once()
    args, kwargs = pool_persist.call_args
    assert args[0] == pd.Timestamp("2026-09-15")
    assert args[1] is pool_df
    assert kwargs == {"code_commit": "abc123"}

    # And: 빈 픽(admitted < top_k)이어도 풀 예측은 기록된다
    def _empty(_d, *, on_failure=None, on_rank_pool=None):
        on_rank_pool(pool_df)
        return pd.DataFrame()

    pool_persist.reset_mock()
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", _empty)
    predict_mod.run_automated_topk_decision(pd.Timestamp("2026-09-15"), record_fn=Mock(), trading_day_fn=_no_calendar)
    pool_persist.assert_called_once()

    # And: 시스템 실패 경로는 풀 기록 없음
    def _fail(_d, *, on_failure=None, on_rank_pool=None):
        on_failure(ValueError("stale price_history"))
        return pd.DataFrame()

    pool_persist.reset_mock()
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", _fail)
    predict_mod.run_automated_topk_decision(pd.Timestamp("2026-09-15"), record_fn=Mock(), trading_day_fn=lambda _d: True)
    pool_persist.assert_not_called()


def test_run_topk_ranker_sleeve_loads_history_for_flow_only_bundle(monkeypatch) -> None:
    import src.daily.predict as predict_mod
    import src.ml.topk_history_features as thf
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

    # Given: a bundle needs inst_density/inst_rank but no f_* history feature
    wide, hist, decision = _sleeve_wide_and_history()
    bundle = build_fixed_serving_bundle([*FEATURE_COLS, *thf.TOPK_FLOW_FEATURE_COLS])
    bundle["top_k"] = 3
    calls = []

    def _fake_loader(d):
        calls.append(d)
        return hist

    monkeypatch.setattr(predict_mod, "load_daily_snapshot", lambda _d, **_kw: wide)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)
    monkeypatch.setattr(thf, "load_serving_price_history", _fake_loader)

    # When
    out = predict_mod.run_topk_ranker_sleeve(decision)

    # Then: the flow-only feature need still triggers the history load (previously skipped)
    assert calls == [decision]
    assert len(out) == 3


def test_bundle_model_version_formats_strategy_cutoff_and_trained_at() -> None:
    import src.daily.predict as predict_mod

    # Then
    assert (
        predict_mod.bundle_model_version(
            {"strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": "2026-09-11 00:00:00", "trained_at": "2026-09-19T22:05:00+09:00"}
        )
        == "KCA-TOPK-COSTAWARE-001@2026-09-11 00:00:00@2026-09-19T22:05:00+09:00"
    )
    assert (
        predict_mod.bundle_model_version({"strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": "2026-09-11 00:00:00"})
        == "KCA-TOPK-COSTAWARE-001@2026-09-11 00:00:00@UNKNOWN"
    )
    assert predict_mod.bundle_model_version({}) == "UNKNOWN@UNKNOWN@UNKNOWN"


# ---------------------------------------------------------------------------
# closing_capture_06 decision-input invariant guards
# ---------------------------------------------------------------------------

def _certified_wide_frame(completed_at, run_label="run-early"):
    import pandas as pd

    return pd.DataFrame([
        {"종목코드": "000001", "symbol": "000001", "종가": 18000.0, "admitted": True,
         "snapshot_timestamp": pd.Timestamp("2026-09-14 15:20:01", tz="Asia/Seoul"),
         "feature_available_timestamp": completed_at},
        {"종목코드": "000002", "symbol": "000002", "종가": 30000.0, "admitted": False,
         "snapshot_timestamp": pd.Timestamp("2026-09-14 15:20:02", tz="Asia/Seoul"),
         "feature_available_timestamp": completed_at},
    ])


def _publish_certified_run(store, cohort, frame, run_id, completed_at):
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    entries = (CoverageEntry(
        symbol=None, dataset=CaptureDataset.PRICE, venue="KRX", session="regular",
        scheduled_at=None, status=CaptureStatus.COMPLETE, rows=len(frame),
        first_event_time=None, last_event_time=None, reason="decision-input", raw_refs=(),
    ),)
    return store.publish_decision(frame, cohort=cohort, run_id=run_id, completed_at=completed_at, entries=entries)


def test_load_daily_snapshot_replays_certified_input_before_cutoff(tmp_path, monkeypatch) -> None:
    """Immutable 15:20 input survives later legacy changes and late captures."""
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.data.capture_contracts import build_cohort
    from src.data.capture_store import CaptureStore

    kst = ZoneInfo("Asia/Seoul")
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(predict_mod, "_capture_root", lambda: tmp_path / "capture")
    monkeypatch.setattr(predict_mod.settings, "COLLECTION_RAW_ENABLED", True)
    cohort = build_cohort(
        date(2026, 9, 14), ["000001", "000002"], ["000001", "000002"], {},
        eligibility_rule_version="price_history_panel@v1",
    )
    early_at = datetime(2026, 9, 14, 15, 20, 30, tzinfo=kst)
    late_at = datetime(2026, 9, 14, 15, 25, 30, tzinfo=kst)
    _publish_certified_run(store, cohort, _certified_wide_frame(early_at), "run-early", early_at)
    late_frame = _certified_wide_frame(late_at)
    late_frame.loc[late_frame["종목코드"] == "000001", "종가"] = 99999.0
    _publish_certified_run(store, cohort, late_frame, "run-late", late_at)

    out = predict_mod.load_daily_snapshot(
        pd.Timestamp("2026-09-14"), available_by=datetime(2026, 9, 14, 15, 22, 0, tzinfo=kst)
    )
    assert out[out["종목코드"] == "000001"]["종가"].iloc[0] == 18000.0
    assert out["capture_run_id"].iloc[0] == "run-early"
    assert out["cohort_id"].iloc[0] == cohort.cohort_id
    assert out["종목코드"].tolist() == ["000001", "000002"]


def test_load_daily_snapshot_fails_closed_without_legacy_substitution(tmp_path, monkeypatch) -> None:
    """New-mode input absence never falls back to finalized legacy archive."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd
    import pytest

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod, "_capture_root", lambda: tmp_path / "capture")
    monkeypatch.setattr(predict_mod.settings, "COLLECTION_RAW_ENABLED", True)

    def _must_not_fallback(snapshot_date=None, **kwargs):
        raise AssertionError("legacy archive must not substitute certified input")

    monkeypatch.setattr(predict_mod, "fetch_archive_snapshot", _must_not_fallback)
    with pytest.raises(FileNotFoundError):
        predict_mod.load_daily_snapshot(
            pd.Timestamp("2026-09-14"),
            available_by=datetime(2026, 9, 14, 15, 22, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        )


def test_load_daily_snapshot_rejects_naive_cutoff(monkeypatch) -> None:
    """Inference cutoff must be an aware timestamp."""
    from datetime import datetime

    import pandas as pd
    import pytest

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "COLLECTION_RAW_ENABLED", True)
    with pytest.raises(ValueError, match="aware cutoff"):
        predict_mod.load_daily_snapshot(pd.Timestamp("2026-09-14"), available_by=datetime(2026, 9, 14, 15, 22, 0))


def test_sleeve_binds_inference_cutoff_and_provenance(monkeypatch) -> None:
    """Input reference is bound before scoring with certified provenance."""
    from datetime import datetime

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.ml.research.v3_engine import FEATURE_COLS
    from tests.unit.serving.realtime.fixtures import build_fixed_serving_bundle

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
        "시장구분": ["KOSPI"] * 4,
        "kospi": [0.52] * 4, "kosdaq": [-0.31] * 4, "v_kospi": [15.2] * 4,
        "admitted": [True, True, True, False],
        "capture_run_id": ["run-early"] * 4,
        "cohort_id": ["cohort-x"] * 4,
        "feature_available_timestamp": [pd.Timestamp("2026-09-09 15:20:01", tz="Asia/Seoul")] * 4,
    })
    bundle = build_fixed_serving_bundle(list(FEATURE_COLS))
    bundle["top_k"] = 3
    seen = {}

    def _fake_load(decision_date, **kwargs):
        seen.update(kwargs)
        return wide

    pools = []
    monkeypatch.setattr(predict_mod, "load_daily_snapshot", _fake_load)
    monkeypatch.setattr(predict_mod, "load_model_bundle", lambda import_dir=None: bundle)

    out = predict_mod.run_topk_ranker_sleeve(pd.Timestamp("2026-09-09"), on_rank_pool=pools.append)

    assert isinstance(seen.get("available_by"), datetime)
    assert seen["available_by"].tzinfo is not None
    assert set(out["capture_run_id"]) == {"run-early"}
    assert set(out["cohort_id"]) == {"cohort-x"}
    assert out["inference_started_at"].notna().all()
    assert out["input_available_at"].notna().all()
    assert len(pools) == 1
    assert set(pools[0]["capture_run_id"]) == {"run-early"}


def test_capture_root_resolution_follows_settings_override(tmp_path, monkeypatch) -> None:
    """Capture root honors configured overrides and the history default."""
    from pathlib import Path

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod.settings, "COLLECTION_ROOT", tmp_path / "custom")
    assert predict_mod._capture_root() == Path(tmp_path / "custom")
    monkeypatch.setattr(predict_mod.settings, "COLLECTION_ROOT", None)
    assert predict_mod._capture_root() == Path(predict_mod.settings.HISTORY_DIR) / "capture"


def test_load_daily_snapshot_defaults_cutoff_to_call_time(tmp_path, monkeypatch) -> None:
    """Omitted cutoff uses the actual call time without legacy substitution."""
    import pandas as pd
    import pytest

    import src.daily.predict as predict_mod

    monkeypatch.setattr(predict_mod, "_capture_root", lambda: tmp_path / "capture")
    monkeypatch.setattr(predict_mod.settings, "COLLECTION_RAW_ENABLED", True)
    with pytest.raises(FileNotFoundError):
        predict_mod.load_daily_snapshot(pd.Timestamp("2026-09-14"))


def test_predict_fails_closed_on_degraded_snapshot(tmp_path, monkeypatch) -> None:
    """Degraded PARTIAL input never becomes a persisted top-k decision."""
    from datetime import date, datetime
    from unittest.mock import Mock
    from zoneinfo import ZoneInfo

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry, build_cohort
    from src.data.capture_store import CaptureStore

    kst = ZoneInfo("Asia/Seoul")
    monkeypatch.setattr(predict_mod, "_capture_root", lambda: tmp_path / "capture")
    monkeypatch.setattr(predict_mod.settings, "COLLECTION_RAW_ENABLED", True)
    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path / "parquet")

    cohort = build_cohort(
        date(2026, 9, 14), ["000001", "000002"], ["000001", "000002"], {},
        eligibility_rule_version="price_history_panel@v1",
    )
    completed_at = datetime(2026, 9, 14, 15, 20, 30, tzinfo=kst)
    frame = pd.DataFrame([
        {"종목코드": "000001", "symbol": "000001", "종가": 18000.0, "admitted": True,
         "snapshot_timestamp": pd.Timestamp("2026-09-14 15:20:01", tz="Asia/Seoul"),
         "feature_available_timestamp": completed_at},
        {"종목코드": "000002", "symbol": "000002", "종가": 30000.0, "admitted": False,
         "snapshot_timestamp": pd.Timestamp("2026-09-14 15:20:02", tz="Asia/Seoul"),
         "feature_available_timestamp": completed_at},
    ])
    store = CaptureStore(tmp_path / "capture")
    store.publish_decision(
        frame,
        cohort=cohort,
        run_id="run-degraded",
        completed_at=completed_at,
        entries=(CoverageEntry(
            symbol=None, dataset=CaptureDataset.PRICE, venue="KRX", session="regular",
            scheduled_at=None, status=CaptureStatus.PARTIAL, rows=2,
            first_event_time=None, last_event_time=None,
            reason="coverage_below_threshold:0.9800", raw_refs=(),
        ),),
    )

    recorder = Mock()
    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2026-09-14"), record_fn=recorder, trading_day_fn=lambda _d: True
    )

    assert not (tmp_path / "parquet" / "topk_decisions.parquet").exists()
    recorder.assert_called_once()
    args, kwargs = recorder.call_args
    assert args == ("NO_DECISION",)


def _session_day(kind, trading_day):
    from datetime import date

    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay

    target = trading_day if isinstance(trading_day, date) else trading_day.date()
    clock = None if kind.value in ("CLOSED", "UNKNOWN") else SessionClock.standard(target)
    return SessionDay(trading_date=target, kind=kind, clock=clock, provenance="test")


def test_run_automated_topk_decision_skips_inference_on_shifted_session(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.data.session_calendar import SessionKind

    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", Mock(side_effect=AssertionError("must not infer")))
    persist_mock = Mock()
    monkeypatch.setattr(predict_mod, "persist_topk_decision", persist_mock)
    recorder = Mock()

    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2026-10-06"),
        record_fn=recorder,
        trading_day_fn=lambda _d: True,
        session_day_fn=lambda d: _session_day(SessionKind.SHIFTED, d),
    )

    persist_mock.assert_not_called()
    recorder.assert_called_once()
    args, kwargs = recorder.call_args
    assert args == ("NO_DECISION",)
    assert kwargs["run_date"] == "2026-10-06"
    assert kwargs["reason"] == "session_shifted"
    assert kwargs["metrics"] == {"n_picks": 0, "session": "SHIFTED"}


def test_run_automated_topk_decision_skips_inference_on_unverified_calendar(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.data.session_calendar import SessionKind

    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", Mock(side_effect=AssertionError("must not infer")))
    recorder = Mock()

    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2027-01-04"),
        record_fn=recorder,
        trading_day_fn=lambda _d: True,
        session_day_fn=lambda d: _session_day(SessionKind.UNKNOWN, d),
    )

    recorder.assert_called_once()
    args, kwargs = recorder.call_args
    assert args == ("NO_DECISION",)
    assert kwargs["reason"] == "calendar_unverified"
    assert kwargs["metrics"] == {"n_picks": 0, "session": "UNKNOWN"}


def test_run_automated_topk_decision_reports_calendar_disagreement(monkeypatch, tmp_path) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.data.session_calendar import SessionKind

    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", Mock(side_effect=AssertionError("must not infer")))
    monkeypatch.setattr(predict_mod.settings, "PARQUET_DIR", tmp_path)
    recorder = Mock()

    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2026-10-09"),
        record_fn=recorder,
        trading_day_fn=lambda _d: True,
        session_day_fn=lambda d: _session_day(SessionKind.CLOSED, d),
    )

    assert not (tmp_path / "topk_decisions.parquet").exists()
    recorder.assert_called_once()
    args, kwargs = recorder.call_args
    assert args == ("NO_DECISION",)
    assert kwargs["reason"] == "calendar_disagreement"


def test_run_automated_topk_decision_treats_confirmed_closure_as_holiday(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.data.session_calendar import SessionKind

    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", Mock(side_effect=AssertionError("must not infer")))
    recorder = Mock()

    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2026-10-09"),
        record_fn=recorder,
        trading_day_fn=lambda _d: False,
        session_day_fn=lambda d: _session_day(SessionKind.CLOSED, d),
    )

    recorder.assert_called_once()
    args, kwargs = recorder.call_args
    assert args == ("OK",)
    assert kwargs["reason"] == "non_trading_day"
    assert kwargs["metrics"] == {"n_picks": 0, "day": "holiday", "session": "CLOSED"}


def test_run_automated_topk_decision_standard_session_behaves_as_before(monkeypatch) -> None:
    from unittest.mock import Mock

    import pandas as pd

    import src.daily.predict as predict_mod
    from src.data.session_calendar import SessionKind

    sleeve_df = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"], "name": ["A", "B", "C"],
        "pred": [0.02, 0.01, 0.005], "allocation": [1 / 3] * 3,
    })
    monkeypatch.setattr(predict_mod, "run_topk_ranker_sleeve", lambda _d, *, on_failure=None, on_rank_pool=None: sleeve_df)
    persist_mock = Mock(return_value=3)
    monkeypatch.setattr(predict_mod, "persist_topk_decision", persist_mock)
    monkeypatch.setattr(predict_mod, "persist_rank_pool_predictions", Mock())
    monkeypatch.setattr(predict_mod, "print_table", Mock())
    recorder = Mock()

    predict_mod.run_automated_topk_decision(
        pd.Timestamp("2026-10-06"),
        record_fn=recorder,
        trading_day_fn=lambda _d: True,
        session_day_fn=lambda d: _session_day(SessionKind.STANDARD, d),
    )

    persist_mock.assert_called_once()
    args, kwargs = recorder.call_args
    assert args == ("OK",)
    assert kwargs["metrics"] == {"n_picks": 3}
