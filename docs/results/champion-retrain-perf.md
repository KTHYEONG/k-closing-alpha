# train_tuned_champion_bundle 런타임 분석 (2026-09-06)

기준: `uv run python -m src.ml.retrain --tuned` (기본 40-trial HPO, walkforward, 패널 복원 on).
ADR_20260906 참조 실행 소요 **809.9초**.

## 측정 (실 dev 패널: 33,547행 / 2,614일 / 61피처)

| 구간 | 1회 비용 | 실행 횟수 | 누적(추정) |
|---|---:|---:|---:|
| parquet 로드 + `build_restored_trade_log` + `build_ml_dataset` | 7.2s | 1 | ~7s |
| **HPO inner OOF** (3-fold, `purged_oof_predict` proba=False) | 4.6s | 40 | **~185s** |
| **HPO `rank_ic` 목적함수 루프** (`scipy.stats.spearmanr` × ~2,600그룹) | 1.3s(무부하)~19s(경합) | 40 | **~50–760s** |
| 5-fold 보정 OOF (`purged_oof_predict` proba=True: reg 5 + LGBMClassifier 10 + LogReg 10) | 19.3s | 3 | ~58s |
| `calibrate_blend_weight` (정책평가 5 + 부트스트랩 5,000×4) | — | 1 | ~15s |
| `evaluate_exit_grid` (`attach_next_day_path` 조인 + 부트스트랩 ×3) | — | 1 | ~5s |
| 최종 `build_inline_bundle` (seed 앙상블 5 + ranker + 분위 3 + 보정 2) | — | 1 | ~15s |

무부하 합계 ≈ 330s. 참조 실행 810s는 **공유 머신 경합**(동시 실행 중인 opencode·codex·타 claude·pytest, load avg 6–8/8코어)으로 2–2.5배 팽창.
`build_ml_dataset` 자체는 2.1s로 빠름 — 과거 관측된 90–142s는 전부 환경 요인(별도 메모리 확인).

## 병목 순위

1. **HPO 전체 (trial당 OOF 4.6s + rank_ic 1.3~19s) × 40 = 전체의 60~90%.** 나머지는 모두 합쳐 ~110s.
2. **`rank_ic` 목적함수**: `spearmanr`를 그룹마다 파이썬 루프로 호출(호출당 ~0.5ms 오버헤드 + 경합 시 급증). 벡터화(그룹 내 rank 후 벡터 상관) 시 **동일 결과·~71배** (검증: `mean=0.437774` 완전 일치, 1.29s→0.018s). `src/ml/metrics.py:rank_ic`도 동일 패턴.
3. **중복 5-fold 보정 OOF**: `champion.py:224` `candidate_oof` 와 `champion.py:275` `evaluate_config_oof(dev, …)` 가 **동일 파라미터로 같은 OOF를 두 번 계산**. 후자에 precomputed 주입 시 ~19s 절감.
4. **LGBM 트리 수 상한 800**: 33k행에서 과대. HPO 탐색을 100–400으로 좁히면 fit 시간 ~1.5배 단축.
5. **보정기(p_good/p_bad) 10× LGBMClassifier fit/OOF**: 최종 블렌드 가중치가 계속 0.0(무효)로 수렴 → 그리드가 0만 남을 때 보정기 학습 생략 가능(계약 변경 필요, 후속).

## 권장 조치 (영향 순, 위험도)

| 조치 | 예상 절감 | 위험 | 성격 |
|---|---:|---|---|
| `rank_ic` 벡터화 (tuning.py + metrics.py) | 40×(1~19s) | 낮음 (출력 동일 검증됨) | 드롭인 |
| HPO `hpo_trials` 기본 40→24, `n_estimators` 상한 800→400 | ~2배 | 낮음 (TPE는 20–30 이후 수렴, ml-res 근거) | 설정 |
| 중복 `candidate_oof` 재사용 | ~19s | 낮음 (동작 보존) | 소규모 리팩터 |
| HPO inner OOF에 LGBM early stopping (train 내부 검증 분할) | fit ~2배 | 중간 (탐색 표면 변화) | 로직 변경 |
| Optuna `n_jobs`×LGBM `n_jobs` 조합 튜닝 | 무부하 시 2–3배 | 낮음 | 설정 (경합 머신엔 효과 제한) |
| 한산한 시간대 실행 / `nice`·스레드 핀 | 2–2.5배 | 없음 | 운영 |

동작 보존 항목(1·3번)은 `docs/specs/champion_retrain_hotpath_opt_contract.json`으로 스펙화. HPO 예산 축소(2·4번)는 승격 결과 A/B가 필요한 별도 결정으로 유보.

## 최적화 적용 후 측정 (`champion_retrain_hotpath_opt`, 2026-09-06)

적용: (1) `tune_return_model_params._objective`의 rank_ic 분기를 `src/ml/metrics.py::mean_group_rank_ic`로 교체 — 그룹 내 average-rank 후 Pearson의 벡터화 단일 패스(float64, groupby+rank+bincount, scipy/루프/apply 없음). (2) `champion.py` candidate `evaluate_config_oof`에 `precomputed_oof=candidate_oof` 주입 — 5-fold 보정 OOF 3회→2회 (control 호출은 파라미터가 달라 그대로 계산). HPO 예산·탐색범위·분할·게이트 불변.

### R8 — 전/후 `retrain --tuned` 전체 A/B (40-trial, walkforward, 패널 복원 on)

| | 변경 전 (scipy 루프) | 변경 후 (벡터화+OOF 재사용) | 차이 |
|---|---:|---:|---|
| wall-time | ~867s (14:32 실행, 고경합) | **219s** (15:08 실행) | 경합 차이 포함 (아래 격리 측정 참조) |
| best_params (trial 31) | `num_leaves=54, lr=0.0292, n_estimators=300, …` | 동일 | identical |
| best_value | 0.20207019588605574 | 동일 | 0 |
| Δ / p / promoted | +0.326pp / 0.0 / true | 동일 | 0 |
| cand/ctrl mean, CI, 양 metrics | — | — | 0 (bitwise) |

40 trial 전부의 TPE 파라미터 경로 동일. 8/40 trial의 출력 표시값이 17째 자리에서 1ulp 차이(~1e-17, 벡터합 순서에 따른 부동소수 재결합 노이즈, R2 허용 1e-9 이내) — 탐색 경로·최종 판정에 영향 없음.

### R9 — 서브스텝 실측 (실 dev OOF 33,547행 / 2,614그룹, 동일 박스 연속 측정)

- rank_ic: scipy 루프 13.96s → `mean_group_rank_ic` **0.0152s (918x)**, 값 차 0.0. trial당 40회분 ≈ 560s 절감이 wall-time 차이의 대부분(나머지는 OOF 1회분 ~19s + 경합 완화).
- 5-fold 보정 OOF 실행 횟수: 3→**2** (`test_champion_candidate_oof_computed_once`로 고정).
