# ML 검증체계 개편 결과 (2026-09-06)

관련 ADR: `ADR_20260906_ML_VALIDATION_OVERHAUL` (archive: `docs/decisions/archive/ML_VALIDATION_OVERHAUL/`)
이전 아카이브: `ADR_20260905_BUYABILITY_GATED_RERANKER`, `ADR_20260905_EXECUTION_COST_MODEL`, `ADR_20260906_CEILING_EXCLUDED_PROMOTION_POOL`, `ADR_20260906_ROUND_TRIP_COST_46BP`, `ADR_20260906_CHAMPION_RETRAIN_HOTPATH_OPT`

## 1. 아키텍처 변경

- 신설: `src/ml/decision_labels.py` (행별비용 `attach_per_row_cost_ratio`, 기계적 라벨 `attach_mechanical_return`, 합성 `build_decision_labels`), `src/ml/validation.py` (`ValidationConfig`, `GateOutcome`, `PromotionDecision`, `minimum_detectable_effect`, `paired_t_p_value`, `cpcv_path_evidence`, `evaluate_locked_oos`, `run_promotion_gate`, `publish_bundle`).
- 배선: `champion.py::train_tuned_champion_bundle` — `retarget_with_clip` → `build_decision_labels`, `assert_no_label_leakage`(feature_cols/select_stable_features 후 2회), `estimate_round_trip_cost_bp`에 `measure_auction_impact_bp` 연결, `config.validation` 설정 시 CPCV path evidence + locked OOS + fillable sleeve → `run_promotion_gate` → `publish_bundle`.
- `ChampionTuningConfig` 필드 추가: `label_mode`("journaled"|"mechanical", 기본 journaled), `cost_mode`("flat"|"per_row", 기본 flat), `validation: ValidationConfig | None`(기본 None). 기본값은 하위호환(기존 `retarget_with_clip` bit-identical).
- CLI(`retrain.py`) 추가 플래그: `--label-mode`(기본 mechanical), `--cost-mode`(기본 per_row), `--publish`, `--production-dir`, `--target-notional-100m`(기본 0.5), `--min-ic-path-win-rate`(0.75), `--min-top1-path-win-rate`(0.60), `--min-oos-days`(60). `--publish` 없이 `--oos-reserve-start` 미지정 시 parser.error.
- 게이트 8종: `cpcv_ic_path_win_rate`(≥0.75), `cpcv_top1_path_win_rate`(≥0.60), `cpcv_delta_significance`(max(p_bootstrap,p_paired_t)<α), `selection_dsr`(≥0.95, n_trials=hpo_trials×|p_good_grid|×|policy_candidates|), `oos_min_days`(≥60), `oos_rank_ic_above_mde`(관측>MDE), `oos_top1_sign`(>0, 부호전용), `fillable_top1_sign`(>0, 부호전용). 전부 통과해야 `deployable`, 하나라도 실패 시 `research_only`(candidate 아티팩트는 항상 저장, production 미교체).

## 2. 실행 중 발견·수정한 버그 (구현 후 실측 실행에서 확인)

| 버그 | 위치 | 증상 | 수정 |
|---|---|---|---|
| NaN top-1 픽 크래시 | `cpcv_path_evidence`, `evaluate_locked_oos` | 상위-1 픽이 `eval_net_mechanical=NaN`(다음날 가격 없음, 실측 커버리지 99.68~99.71%)인 날짜에서 `moving_block_bootstrap_delta`가 `ValueError: paired series must hold only finite values`로 크래시 | finite 마스크로 해당 날짜/폴드 제외 후 집계. 회귀테스트 `test_cpcv_path_evidence_skips_days_with_unlabelled_top1_pick`, `test_evaluate_locked_oos_skips_days_with_unlabelled_top1_pick` 추가 |
| 잠재 KeyError (구현 시점 /check에서 발견) | `champion.py` validation 분기 | `label_mode='journaled'`+`validation` 설정+`price_history_df=None` 조합 시 `eval_net_mechanical` 컬럼 부재로 하류 KeyError | `dev`/`oos`에 `config.validation.eval_col` 부재 시 명시적 `ValueError` fail-closed 가드 추가 |
| 중복 임포트 | `champion.py` | `from src.ml.validation import publish_bundle` 두 번(`# noqa: F811`) | 제거 |

## 3. 실측 결과 — Arm A: `label_mode=journaled, cost_mode=flat`

명령: `uv run python -m src.ml.retrain --tuned --oos-reserve-start 2025-09-01 --label-mode journaled --cost-mode flat --hpo-trials 40`
번들: `artifacts/models/research/exp_a_journaled_flat/close_morning61_2025-08-29/sizing_pipeline_bundle.joblib`
데이터: 전체 36,744행(패널복원+상한가배제 후 dev 33,547행 상당), dev 기간 2016-01-04~2025-08-29, 잠긴 OOS 2025-09-01~ (245일, 4,514행). `label_provenance`: `n_dropped_no_mechanical=0`(journaled 모드는 미보유 라벨 드롭 안 함), `mechanical_coverage=0.99712`.

### 3.1 판정 비교 (동일 실행, 두 개의 승격 기준)

| 판정 경로 | 결과 | 델타 | p값 | 표본 |
|---|---|---:|---:|---|
| 기존 워크포워드 부트스트랩(레거시 게이트) | `promoted: true` | +0.249%p/일 (cand 0.809% vs ctrl 0.559%) | 0.0012 | dev 공유일 1,975 |
| **신규 CPCV(8,2) 경로일치 + 잠긴OOS 8게이트** | **`research_only`** | pooled path Δ | max(p_bootstrap,p_paired_t)=0.508 | CPCV 28-path |

**결론: 동일 후보가 레거시 기준으로는 승격, 신규 기준으로는 미승격.** 레거시 판정은 선택과 동일한 OOF 위에서만 유의했던 것으로 재확인됨.

### 3.2 게이트별 상세

| gate | passed | observed | threshold | mde | 비고 |
|---|:--:|---:|---:|---:|---|
| cpcv_ic_path_win_rate | ✅ | 0.750 | 0.750 | 0.00823 | 정확히 경계 |
| cpcv_top1_path_win_rate | ❌ | 0.536 (15/28) | 0.600 | 0.00123 | 미달 |
| cpcv_delta_significance | ❌ | p=0.508 | 0.05 | 0.00123 | p_bootstrap=0.508, p_paired_t=0.131 |
| selection_dsr | ✅ | 0.99999999 | 0.95 | — | n_selection_trials=600(40×5×3) |
| oos_min_days | ✅ | 245 | 60 | — | |
| oos_rank_ic_above_mde | ✅ | 0.05847 | 0.05662(=mde) | 0.05662 | 경계 통과 |
| oos_top1_sign | ✅ | +0.00920 (%/일) | 0 | 0.01878 | 부호만, MDE 대비 16배 작아 크기 판별 불가 |
| fillable_top1_sign | ✅ | +0.00466 (%/일) | 0 | — | measured_share=0.9909 |

### 3.3 HPO/코스트/기타 provenance

- HPO: 40 trials, `best_value(rank_ic)=0.22666`. `p_good_weight`=0.0(계속 무효).
- `control_vs_candidate`(레거시): shared_dates=1975, cand_mean=0.008086, ctrl_mean=0.005594, ci=[0.000949, 0.003924].
- `execution_cost`: statutory=20.0bp, spread=26.182bp, auction_impact=NaN, total=46.182bp, n_rows=27479, **n_impact_measured=0**(dev 구간 대부분 분봉 파티션 미보유 — 2025년만 부분 존재, `measure_auction_impact_bp` 배선은 됐으나 실측 0건), breakeven=19.511bp.
- `buyability_sleeves`: status=evaluated (이전 실행은 skipped였음 — `target_notional_100m` CLI 기본값 0.5 배선 확인).

## 4. 미실행 Arm (다음 액션)

R14 프로토콜의 나머지 2개 arm은 동일 조건(`--oos-reserve-start 2025-09-01`, hpo-trials=40, 동일 seed)으로 미실행:

| arm | 명령 플래그 | 목적 |
|---|---|---|
| B | `--label-mode journaled --cost-mode per_row` | 라벨 고정, 비용만 행별화(실측 분산 p10 32.6bp~p90 60.0bp)했을 때 게이트 영향 분리 |
| C | `--label-mode mechanical --cost-mode per_row` | 운영자 재량 청산 제거, 기계적 실행가능 라벨(사전근거: rankIC +0.0375 p<1e-4, CPCV 27/28) — **헤드라인 top-1 하락 예상**, rankIC 개선이 실제 레버 |

각 arm 실측 소요시간 ~20분(arm A: 18:57~19:17). B/C는 미실행 상태이며 §3의 표 형식 그대로 채워 넣을 것.

## 5. 이전 세션 결과 (참고, 레거시 게이트 기준 — 위 §3.1로 대체됨)

| 실행일 | 비용가정 | Candidate 일평균 | Control 일평균 | Δ | p | 레거시 판정 |
|---|---|---:|---:|---:|---:|:--:|
| 2026-09-06 (상한가배제 첫 승격) | 20bp(구형) | +0.921% | +0.735% | +0.185%p | 0.006 | promoted |
| 2026-09-06 (46bp 재검증) | 46bp(실측 flat) | +0.729% | +0.404% | +0.326%p | 0.0000 | promoted |
| 2026-09-06 (Arm A, 신규 게이트) | 46bp(flat) | 0.809%(dev만) | 0.559% | +0.249%p | 0.0012(레거시)/0.508(CPCV) | **research_only** |

상한가 배제(`classify_ceiling_entry`)와 46bp 비용 상수 자체는 유효하며 변경 없음 — 무효화된 것은 "레거시 워크포워드 부트스트랩 단독으로 승격 판정"이라는 **방법론**.

## 6. 검정력 배경 (변경 없음)

일별 페어드 노이즈(2021+, n=1,352일): top1 sd=4.667%/일, rankIC sd=0.2745. top1 +25bp/일 검출에 2,732일(10.8년) 필요 → OOS top-1 게이트는 부호전용일 수밖에 없는 이유. rankIC는 244일 창에서 MDE≈0.049~0.057로 판별 가능 → 승격 판정의 주 통계량.

## 7. 다음 단계

1. Arm B, C 실행 후 본 문서 §3 형식으로 추가.
2. `n_impact_measured=0` 해소: 청산일 분봉 백필(`src/backfill/intraday/backfill_minute_history.py`) 이후 dev 구간 경매임팩트 실측 재실행.
3. `cpcv_top1_path_win_rate`(0.536) 미달 원인 분해 — 어떤 CPCV 폴드(연도/구간)에서 후보가 대조군에 지는지 폴드별 breakdown 추가.
4. `oos_rank_ic_above_mde`가 경계(0.0585 vs MDE 0.0566)라 표본 추가 시 뒤집힐 수 있음 — 다음 재학습에서 재확인.
