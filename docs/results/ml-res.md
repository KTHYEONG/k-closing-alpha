# ML 재학습·개선 결과 (2026-09-03)

관련 ADR: `ADR_20260903_ML_SPARSE_DATA_ROBUSTNESS`, `ADR_20260903_ML_PIPELINE_GAIN_RECOVERY`

## 1. 실행 요약

- 대상: 종가매매 close-morning 리랭커 (`close_morning61`, 61 스냅샷 피처, `scenario_action` 패널)
- 데이터: `data/parquet/trade_log.parquet` 33,827행 / 2,599 거래일 / 2016-01 ~ 2026-08
- 목적: "튜닝만으로 라이브러리 기본값 대비 유의한 개선이 안 되는" 정확한 원인을 데이터로 규명하고 해결책 검증
- 평가 인프라: `src/ml/robust_eval.py` — CombinatorialPurgedCV(8,2)=28fold/7path, moving-block bootstrap 유의성 검정, Deflated Sharpe
- 운영 반영: **없음**. 활성 번들 불변. 아래 수정은 미커밋.

## 2. 핵심 결론 (한 줄)

리랭커는 **현재 후보군 레짐에서 포화** 상태다. 모델·블렌드·정책·가중·윈도우·앙상블·관망 게이트 어떤 튜닝도 라이브러리 기본값(LightGBM defaults, huber δ=0.9)을 **통계적으로 이기지 못한다**. 진짜 레버는 새 예측 정보(피처)뿐이다.

## 3. 데이터 규명 (moving-block bootstrap, n≈2,165 paired 일별, vs 라이브러리 기본값)

### 3.1 알파는 감소하지 않는다 — "rankIC 0.24→0.17 하락"은 아티팩트

| 연도(테스트) | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|
| walk-forward OOF rankIC | 0.30 | 0.19 | 0.21 | 0.18 | 0.18 | 0.18 |
| 타깃 일간분산(std) | 0.034 | 0.030 | 0.031 | 0.034 | 0.032 | 0.031 |

2020(코로나)만 이상치. 2021년 이후 rankIC ~0.18 **평탄**, 기회분산도 평탄. 감소 통념은 초기 윈도우에 2020이 섞인 결과.

### 3.2 후보군 자체가 붕괴했다 (개선 상한을 규정)

| 구간 | 평균 후보수익 | 승률 | oracle top-1 | (best − mean) 스프레드 |
|---|---:|---:|---:|---:|
| 2016–2020 | **+0.31%** | 53% | +5.5% | +5.2% |
| 2021–2025 | **−0.17%** | 44% | +6.6% | **+6.9%** |

평균 후보가 이제 손실 트레이드. 그러나 oracle는 여전히 +6~7%, 스프레드는 **더 넓어짐** → 리랭커가 할 일은 많아졌으나 성과는 top-1 실현 1.3%→0.6%로 반감. 모델은 여전히 순위 능력 보유(rankIC 0.18).

### 3.3 파이프라인 이득 상쇄 분해 (BASE = 라이브러리 기본값 control)

| knob (BASE에서 1개만 변경) | POST-POLICY Δ | p | 판정 |
|---|---:|---:|---|
| `p_good_weight` 0.5 → 0.0 | **+0.120%/일** | **0.015** | p_good 블렌드가 알파를 희석 |
| `p_good_weight` → 1.0 | −0.007% | 0.87 | 더 섞을수록 무익 |
| 정규화 파라미터 (num_leaves 15·min_child 40·subsample) | **+0.127%/일** (모델단 +0.186% p=0.005) | **0.021** | 실제 이득 |
| `weighting_mode` → `date_balanced` | −0.000% | **0.99** | 무효 |
| `recency_half_life` 504 | +0.063% | 0.31 | 유의하지 않음 |
| `date_balanced` + `recency` | +0.069% | 0.28 | 유의하지 않음 |
| **정규화 + p_good_weight 0 (REG_PG0)** | **+0.239%/일** | **0.0004** | 격리실험 최적 (Sharpe 4.23→5.13) |

### 3.4 `calibrate_blend_weight` 버그 (핵심 원인)

실데이터 grid 평균: `{0.0: 1.29%, 0.25: 1.25%, 0.5: 1.18%, 0.75: 1.14%, 1.0: 1.12%}` — 단조감소, w=0 최선.
**함수는 0.75를 선택.** "보수적 tiebreaker"가 `max−min < 0.005`(50bp)일 때 발동 → 일별수익 스케일에선 **항상 참** → `entry_sequence_drawdown`(MDD 노이즈)로 정렬. 모든 tuned 챔피언이 MDD 노이즈로 뽑은 가중을 배포해 옴. w>0 비용: −11~17bp/일 (w=0.5 p=0.011, w=1.0 p=0.004).

### 3.5 기각된 가설 (전부 무효 또는 음(−))

| 가설 | 결과 |
|---|---|
| 타깃 횡단면 디민 (excess return) | dIC −0.0155, p=0.17 |
| rolling 750일 윈도우 학습 | −16bp (since 2023) |
| top-2 / top-3 분산 진입 | −26 ~ −63bp, p<0.01 |
| ridge / 선형 모델 | −27 ~ −38bp, p<0.05 |
| causal margin / conviction 관망 게이트 | −8 ~ −43bp — 최저확신 분위도 +0.8% EV라 관망은 손해 |
| date-constant 피처 제거 (`v_kospi` 등 4개) | p=0.75 |
| 나이브 공시 + KOSPI200 베이시스 대체데이터 | dIC −0.006, p=0.60 |
| HPO 목적함수 `rank_ic` | leaves 56 선택(정규화 안됨), 결정지표 못 움직임 |
| HPO 목적함수 `cpcv_top1` | leaves 8 선택(정규화 됨), 그러나 게이트 p=0.67 |

## 4. 수정 사항 (자체 정당성으로 배포 가치 있음)

| 파일 | 변경 | 근거 |
|---|---|---|
| `src/ml/tuning.py` | `calibrate_blend_weight` 유의성 게이트 재작성 — 최저 grid 가중 기본, 상위는 `moving_block_bootstrap Δ>0 & p<promotion_alpha`일 때만 채택, MDD tiebreaker 삭제. `per_weight`에 `delta_vs_base`/`p_value_vs_base` 추가. `alpha` kwarg. | 명백한 선택 버그 제거 |
| `src/ml/bundle.py` | `CHAMPION_DEFAULT_MODEL_PARAMS` (num_leaves 15, min_child 40, n_estimators 350, lr 0.03, subsample/colsample 0.8, reg_lambda 1.0) → `build_inline_bundle` 기본층 병합 | bare LightGBM 기본값(31/100) 대신 정규화 prior |
| `src/ml/tuning.py`, `champion.py`, `retrain.py` | `ChampionTuningConfig.model_params_override` + `retrain --no-hpo` — Optuna 스킵 | HPO(11 파라미터 × 2000 노이즈 관측)는 검증노이즈 과적합 |
| `src/serving/realtime/inference.py` | `_CLOSE_MORNING_RERANKER_CONFIG["p_good_weight"] 0.5 → 0.0` (v2 research config 불변) | 배포/서빙 fallback 기본값 |

## 5. 실 파이프라인 검증 — 수정은 작동, 그러나 홀드아웃 게이트 미통과

`train_tuned_champion_bundle`, OOS 예약 `2025-07-01`, n=1,935 shared dates:

| 후보 | 블렌드 선택 | GATE Δ | p | 승격 | top1 / Sharpe / MDD / PF |
|---|---|---:|---:|:---:|---|
| FIXED (`--no-hpo` 정규화 + 유의성 블렌드) | **0.0** (기존 버그 0.75) | +0.037%/일 | **0.572** | ❌ | +1.432% / 5.77 / **0.245** / 2.77 |
| HPO `rank_ic` (12 trials) + 유의성 블렌드 | **0.0** | +0.060%/일 | **0.411** | ❌ | +1.455% / 5.94 / **0.287** / 2.89 |
| 대조군 (라이브러리 기본값) | 0.5 | — | — | — | +1.395% / 5.72 / 0.182 / 2.81 |

- 두 후보 모두 라이브러리 기본값 대비 **+4~6bp/일 (노이즈 수준, p>0.4)**, **MDD·PF는 오히려 악화**.
- 격리 walk-forward ablation의 **+24bp/p=0.0004는 OOS 예약된 `dev` 구간에서 재현 안 됨**. `dev`(2025-07+ 제외)에서 p_good 블렌드는 중립(0.0: +1.432% vs 0.5: +1.431%) — ablation의 −12bp p_good 비용은 **전체데이터 + 라이브러리기본모델 조합 아티팩트**였음.
- `calibrate_blend_weight` 버그 수정은 **확정 검증됨**: 두 실행 모두 0.0 선택, 유의성 테이블상 어떤 상위 가중도 0.0을 못 이김.

## 6. 이전 세션 (참고) — CPCV 강건성 인프라

`ADR_20260903_ML_SPARSE_DATA_ROBUSTNESS`: RUN A(`rank_ic`) p=0.358, RUN B(`cpcv_top1`) p=0.665 — 둘 다 승격 거부. 기존 코인플립 게이트(`cand≥ctrl`)였다면 RUN A 승격됐을 것(노이즈 승격). 새 bootstrap 게이트가 차단. `cpcv_oof_predict` attrs concat 버그, `eval_mode` ghost-switch 수정.

## 7. 한계 및 다음 단계

- `entry_sequence_drawdown`은 포지션 중첩·자본배분 미반영 간이지표. 운영 승격 전 포트폴리오 백테스트(비용·슬리피지·체결실패 포함) 필수.
- **P2 (최근 레짐 레버 = 새 정보만)**: 결정시점 미시구조, `chart_analysis` × 피처 상호작용, 트레일링/상대 변환된 대체데이터. 그룹별로 CPCV + bootstrap 게이트(`Δ>0 & p<0.10` vs 61피처 대조군) 통과 시에만 배포. 정직한 사전확률: 탐색적, 아무것도 안 나올 수 있음 (다일·섹터·공시 3회 시도 이미 실패).
- **P3 (죽은 p_good 연산 제거)**: p_good_weight 0 기본 + calibrate가 계속 0 선택 시, `fit_chrono_calibrator`가 fold당 2회 무의미하게 실행. `any(w>0 for w in grid)` 게이팅 또는 v1 경로에서 p_good/p_bad 제거 → 챔피언 학습시간 ~30% 절감.
