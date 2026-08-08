"""History-feature research experiment 통합 테스트.

`docs/specs/ml_feature_selection_pipeline.md` 시나리오:
- FS-10 integration: 컨트롤/후보가 동일한 OOF 날짜를 사용하며, 후보는 활성
  아티팩트를 절대 덮어쓰지 않습니다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_selection import FeatureSelectionConfig
from src.ml.history_feature_research import (
    _oof_date_mismatch_diagnostic,
    _promotion_gate,
    run_history_feature_research_experiment,
    validate_research_oof_alignment,
)
from src.ml.history_features import HISTORICAL_CATALOGUE_VERSION


def test_p0b_oof_identity_equal_arrays_pass_gate() -> None:
    """P0-OOF-IDENTITY-02: 동일한 실제 날짜 배열은 표준화·보존되고 identity 게이트를 통과합니다."""
    control = np.array(["2025-01-02", "2025-01-03", "2025-01-04"])
    candidate = np.array(["2025-01-04", "2025-01-03", "2025-01-02"])
    identical, control_norm, candidate_norm = validate_research_oof_alignment(
        control, candidate
    )
    assert identical is True
    assert control_norm == candidate_norm
    assert control_norm == [
        "2025-01-02T00:00:00",
        "2025-01-03T00:00:00",
        "2025-01-04T00:00:00",
    ]
    assert _oof_date_mismatch_diagnostic(control_norm, candidate_norm) == "oof_dates_identical"


def test_p0b_oof_identity_omitted_candidate_date_fails() -> None:
    """P0-OOF-IDENTITY-01: 후보 날짜 누락 시 identical=False 이고 승격 거부 oof_dates_mismatch 입니다."""
    control = np.array(["2025-01-02", "2025-01-03", "2025-01-04"])
    candidate = np.array(["2025-01-02", "2025-01-04"])
    identical, control_norm, candidate_norm = validate_research_oof_alignment(
        control, candidate
    )
    assert identical is False
    assert "missing=['2025-01-03T00:00:00']" in _oof_date_mismatch_diagnostic(
        control_norm, candidate_norm
    )
    control = {
        "aggregate": {
            "candidate": {
                "scheduled_mean_return": 0.015,
                "profit_factor": 1.5,
                "entry_sequence_drawdown": 0.30,
            }
        }
    }
    candidate = {
        "aggregate": {
            "candidate": {
                "scheduled_mean_return": 0.020,
                "profit_factor": 1.5,
                "entry_sequence_drawdown": 0.20,
            }
        }
    }
    gate = _promotion_gate(control, candidate, identical, {"gate_passed": True})
    assert gate["promoted"] is False
    assert "oof_dates_mismatch" in gate["rejected_reasons"]


def test_p0b_oof_identity_duplicate_and_unparsable_fail() -> None:
    """P0-OOF-IDENTITY: 중복 또는 파싱 불가 날짜는 fail-closed 로 False 입니다."""
    assert (
        validate_research_oof_alignment(
            np.array(["2025-01-02", "2025-01-02"]), np.array(["2025-01-02"])
        )[0]
        is False
    )
    assert (
        validate_research_oof_alignment(
            np.array(["2025-01-02"]), np.array(["not-a-date"])
        )[0]
        is False
    )


def _build_trade_log(n_dates: int = 18, n_stocks: int = 8, seed: int = 4) -> pd.DataFrame:
    """스프레드시트 헤더 형태의 원본 매매일지를 생성합니다."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime([f"2024-01-{2 + d:02d}" for d in range(n_dates)])
    rows: list[dict[str, object]] = []
    for date in dates:
        for i in range(n_stocks):
            code = f"{i + 1:06d}"
            base = 10_000 + i * 1_000
            change = float(rng.normal(0.5, 2.0))
            net_return = float(rng.normal(0.2, 1.2))
            close_p = int(base * (1 + change / 100))
            rows.append(
                {
                    "매수날짜": date,
                    "종목코드": code,
                    "(시가)": base,
                    "(고가)": int(base * 1.05),
                    "(저가)": int(base * 0.97),
                    "(종가)": close_p,
                    "(전일종가)": int(base * 0.99),
                    "(시가총액, 억)": 1_000.0 + i * 100,
                    "(거래대금, 억)": 100.0 + i * 20,
                    "(등락률)": change,
                    "(선정 순위)": float(i + 1),
                    "(기관_순매수)": float((i - 2) * 50),
                    "(외국인_순매수)": float(i * 30),
                    "(프로그램_순매수)": float((i - 1) * 10),
                    "(체결강도)": 110.0 + i,
                    "(시장구분)": "KOSPI" if i % 2 == 0 else "KOSDAQ",
                    "(총 종목 수)": float(n_stocks),
                    "(평균 거래대금)": 90.0,
                    "(kospi, %)": 0.3,
                    "(kosdaq, %)": 0.1,
                    "v_kospi": 15.0,
                    "v_kosdaq": 18.0,
                    "(거래량)": 1_000_000 + i * 1_000,
                    "(테마/섹터)": f"theme{i % 3}",
                    "(차트분석)": "신고가 근접",
                    "(매수 가격)": float(close_p * 0.99),
                    "(매도 가격)": float(close_p * 1.02),
                    "(수익률, %)": f"{net_return:.4f}",
                }
            )
    return pd.DataFrame(rows)


def _build_price_history(
    trade_log: pd.DataFrame, seed: int = 4
) -> pd.DataFrame:
    """매매일지 날짜 범위를 덮는 합성 EOD 판넬을 생성합니다."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(sorted(trade_log["매수날짜"].unique()))
    symbols = sorted(trade_log["종목코드"].astype(str).str.zfill(6).unique())
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        price = 10_000.0
        for date in dates:
            o = price
            c = price * (1 + rng.normal(0, 0.008))
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": o,
                    "high": hi,
                    "low": lo,
                    "close": c,
                    "prev_close": price,
                    "market_cap_100m": rng.uniform(100, 5_000),
                    "trade_value_100m": rng.uniform(10, 500),
                    "daily_change_pct": c / price - 1.0,
                    "market": "KOSPI",
                    "volume": rng.uniform(1e4, 1e7),
                    "foreign_netbuy": rng.normal(0, 1e5),
                    "inst_netbuy": rng.normal(0, 1e5),
                    "program_netbuy": rng.normal(0, 5e4),
                    "kospi_pct": rng.normal(0, 0.004),
                    "kosdaq_pct": rng.normal(0, 0.004),
                    "v_kospi": rng.uniform(15, 30),
                    "v_kosdaq": rng.uniform(12, 25),
                }
            )
            price = c
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fs10_control_and_candidate_share_identical_oof_dates(tmp_path: Path) -> None:
    """컨트롤/후보가 동일한 purged OOF 날짜를 사용하고, 후보 진단을 산출합니다."""
    trade_log = _build_trade_log()
    price_history = _build_price_history(trade_log)
    cfg = FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40)
    result = run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history=price_history,
        n_splits=3,
        purge_gap=1,
        feature_selection_config=cfg,
        export_dir=str(tmp_path),
    )
    assert result["comparison"]["identical_oof_dates"] is True
    assert result["comparison"]["control_oof_dates"] == result["comparison"]["candidate_oof_dates"]
    assert result["contract"]["catalogue_version"] == HISTORICAL_CATALOGUE_VERSION
    final_features = result["candidate"]["final_features"]
    assert 5 <= len(final_features) <= 40
    assert result["candidate"]["feature_selection_diagnostics"] is not None
    assert result["candidate_bundle_path"]
    bundle = Path(result["candidate_bundle_path"])
    assert bundle.is_file()


def test_fs10_candidate_never_overwrites_active_artifacts(tmp_path: Path) -> None:
    """후보 번들은 research 디렉터리로만 저장되고 활성 아티팩트를 바꾸지 않습니다."""
    active_path = Path("artifacts/models/sizing_pipeline_bundle.joblib")
    active_before = _sha256(active_path) if active_path.is_file() else None

    trade_log = _build_trade_log(seed=6)
    price_history = _build_price_history(trade_log, seed=6)
    cfg = FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40)
    run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history=price_history,
        n_splits=3,
        purge_gap=1,
        feature_selection_config=cfg,
        export_dir=str(tmp_path),
    )
    active_after = _sha256(active_path) if active_path.is_file() else None
    assert active_before == active_after
    # research 하위에만 저장되었는지 확인합니다.
    research_root = tmp_path / HISTORICAL_CATALOGUE_VERSION
    assert research_root.is_dir()
    assert list(research_root.rglob("sizing_pipeline_bundle.joblib"))


def test_fs10_rejects_missing_history_sources(tmp_path: Path) -> None:
    """가격 이력에 필수 원천 컬럼이 없으면 fail-closed 로 거부합니다."""
    trade_log = _build_trade_log(n_dates=10, n_stocks=6)
    price_history = _build_price_history(trade_log)
    price_history = price_history.drop(columns=["inst_netbuy"])
    with pytest.raises(ValueError, match="required columns"):
        run_history_feature_research_experiment(
            trade_log,
            theme_df=None,
            price_history=price_history,
            n_splits=3,
            purge_gap=1,
            feature_selection_config=FeatureSelectionConfig(
                min_retained=5, max_retained=20, hard_max_retained=40
            ),
            export_dir=str(tmp_path),
        )


def test_hfs07_path_experiment_persists_metrics_and_device_metadata(tmp_path: Path) -> None:
    """경로 기반 실험은 번들에 빌드 지표/장치 메타데이터를 영속화하고 활성을 보존합니다."""
    active_path = Path("artifacts/models/sizing_pipeline_bundle.joblib")
    active_before = _sha256(active_path) if active_path.is_file() else None

    trade_log = _build_trade_log(seed=11)
    price_history = _build_price_history(trade_log, seed=11)
    history_path = tmp_path / "price_history.parquet"
    price_history.to_parquet(history_path)

    cfg = FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40)
    result = run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history_path=str(history_path),
        n_splits=3,
        purge_gap=1,
        feature_selection_config=cfg,
        export_dir=str(tmp_path / "research"),
    )
    assert result["comparison"]["identical_oof_dates"] is True
    assert result["build_metrics"]["batch_count"] >= 1
    assert result["build_metrics"]["output_rows"] >= 1

    bundle_path = Path(result["candidate_bundle_path"])
    assert bundle_path.is_file()
    import joblib

    bundle = joblib.load(bundle_path)
    assert bundle["history_feature_build_metrics"]["batch_count"] >= 1
    assert bundle["catalogue_version"] == HISTORICAL_CATALOGUE_VERSION
    assert bundle["cross_sectional_scope"] == "decision_candidate_panel"
    assert bundle["resolved_screening_device"] in ("cpu", "gpu")
    assert bundle["requested_screening_device"] == "cpu"

    active_after = _sha256(active_path) if active_path.is_file() else None
    assert active_before == active_after


def test_hfs07_experiment_requires_history_input(tmp_path: Path) -> None:
    """history 입력(DataFrame/경로)이 없으면 ValueError 로 거부합니다."""
    trade_log = _build_trade_log(n_dates=10, n_stocks=6)
    with pytest.raises(ValueError, match="price_history or price_history_path"):
        run_history_feature_research_experiment(
            trade_log,
            theme_df=None,
            n_splits=3,
            purge_gap=1,
            export_dir=str(tmp_path),
        )


def test_mto01_rows_after_history_cutoff_rejected_before_fitting(tmp_path: Path) -> None:
    """MTO-01: history 커트오프 이후 행은 후보 학습 전에 거부되고, 컨트롤/후보는
    거부된 날짜를 절대 평가하지 않습니다."""
    trade_log = _build_trade_log(n_dates=18, n_stocks=8)
    price_history = _build_price_history(trade_log)
    history_dates = sorted(price_history["date"].unique())
    last_history_date = history_dates[-4]
    price_history = price_history[price_history["date"] <= last_history_date].copy()

    cfg = FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40)
    result = run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history=price_history,
        n_splits=3,
        purge_gap=1,
        feature_selection_config=cfg,
        export_dir=str(tmp_path),
    )
    # evaluation_cutoff = min(패널 최대, history 최대).
    assert result["contract"]["evaluation_cutoff"] == str(last_history_date)
    assert result["contract"]["excluded_rows_after_cutoff"] > 0
    assert result["contract"]["excluded_dates"]
    retained = pd.to_datetime(result["comparison"]["control_oof_dates"])
    assert len(retained) > 0
    assert retained.max() <= pd.Timestamp(last_history_date)
    assert result["comparison"]["identical_oof_dates"] is True
    assert result["comparison"]["control_oof_dates"] == result["comparison"]["candidate_oof_dates"]
    # 커트오프 이후 날짜는 excluded_dates 에 기록되고 절대 채워지지 않습니다.
    assert all(pd.Timestamp(d) > pd.Timestamp(last_history_date) for d in result["contract"]["excluded_dates"])


def test_mto01_frozen_research_cutoff_caps_evaluation(tmp_path: Path) -> None:
    """MTO-01: 명시적 research_cutoff 는 history 최대와 교집합(최솟값)으로 적용됩니다."""
    trade_log = _build_trade_log(n_dates=24, n_stocks=8)
    price_history = _build_price_history(trade_log)
    history_dates = sorted(price_history["date"].unique())
    frozen = str(history_dates[-6])

    cfg = FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40)
    result = run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history=price_history,
        n_splits=3,
        purge_gap=1,
        feature_selection_config=cfg,
        research_cutoff=frozen,
        export_dir=str(tmp_path),
    )
    # min(frozen, history max) 이 커트오프입니다.
    assert result["contract"]["evaluation_cutoff"] == frozen
    assert result["contract"]["excluded_rows_after_cutoff"] > 0
    assert result["comparison"]["identical_oof_dates"] is True
    assert result["comparison"]["control_oof_dates"] == result["comparison"]["candidate_oof_dates"]


def test_mto02_feq02_quality_report_persisted_and_lower_return_not_promoted(
    tmp_path: Path,
) -> None:
    """MTO-02-FEQ-02: report-only 품질 프로필이 리서치 번들에 영속화되고, 평균
    수익이 대조군보다 낮은 후보는 drawdown 이 낮아도 승급되지 않습니다."""
    import joblib

    from src.ml.history_feature_research import _promotion_gate

    trade_log = _build_trade_log()
    price_history = _build_price_history(trade_log)
    cfg = FeatureSelectionConfig(min_retained=5, max_retained=20, hard_max_retained=40)
    result = run_history_feature_research_experiment(
        trade_log,
        theme_df=None,
        price_history=price_history,
        n_splits=3,
        purge_gap=1,
        feature_selection_config=cfg,
        export_dir=str(tmp_path),
    )
    # report-only: 반환 후보 진단과 저장 번들 모두 품질 리포트를 포함합니다.
    assert result["candidate"]["quality_report"]["version"] == "feature_quality_v2"
    assert result["comparison"]["control_oof_dates"] == result["comparison"]["candidate_oof_dates"]
    assert "candidate_beats_control_mean" in result["promotion"]

    bundle_path = Path(result["candidate_bundle_path"])
    assert bundle_path.is_file()
    saved = joblib.load(bundle_path)
    assert saved["quality_report"]["version"] == "feature_quality_v2"
    assert saved["quality_report"]["n_folds"] == 3
    # 65 전부 비유한 마켓 리지먼트 후보는 source_incomplete 로 보고되고 보간되지 않습니다.
    actions = saved["quality_report"]["actions"]
    market_incomplete = {
        feature
        for feature, feature_actions in actions.items()
        if "source_incomplete" in feature_actions
    }
    assert market_incomplete

    def _agg(mean: float, mdd: float, pf: float = 1.5) -> dict[str, float]:
        return {
            "scheduled_mean_return": mean,
            "profit_factor": pf,
            "entry_sequence_drawdown": mdd,
        }

    control = {"aggregate": {"candidate": _agg(0.015, 0.30)}}
    lower_return = {"aggregate": {"candidate": _agg(0.010, 0.20)}}
    gate = _promotion_gate(
        control,
        lower_return,
        identical_oof_dates=True,
        stability={"gate_passed": True},
    )
    assert gate["promoted"] is False
    assert gate["candidate_beats_control_mean"] is False
    assert "candidate_mean_not_strictly_higher" in gate["rejected_reasons"]
    assert "compounded_mdd_not_strictly_lower" not in gate["rejected_reasons"]

    higher_return = {"aggregate": {"candidate": _agg(0.020, 0.20)}}
    gate_ok = _promotion_gate(
        control,
        higher_return,
        identical_oof_dates=True,
        stability={"gate_passed": True},
    )
    assert gate_ok["promoted"] is True
    assert gate_ok["candidate_beats_control_mean"] is True


def test_p0a_availability_manifest_non_promotable_blocks_promotion() -> None:
    """P0-A: 가용성 증명이 승격 불가이면 모든 지표가 통과해도 승격이 거부됩니다."""
    control = {
        "aggregate": {
            "candidate": {
                "scheduled_mean_return": 0.015,
                "profit_factor": 1.5,
                "entry_sequence_drawdown": 0.30,
            }
        }
    }
    candidate = {
        "aggregate": {
            "candidate": {
                "scheduled_mean_return": 0.020,
                "profit_factor": 1.5,
                "entry_sequence_drawdown": 0.20,
            }
        }
    }
    gate = _promotion_gate(
        control,
        candidate,
        identical_oof_dates=True,
        stability={"gate_passed": True},
        availability_promotable=False,
    )
    assert gate["promoted"] is False
    assert gate["availability_manifest_promotable"] is False
    assert "availability_manifest_non_promotable" in gate["rejected_reasons"]
