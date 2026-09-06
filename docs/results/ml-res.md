# ML 결과 (2026-09-06, v2 아키텍처)

관련 ADR: `ADR_20260906_CLOSING_ALPHA_ARCHITECTURE_V2`, `ADR_20260906_ML_VALIDATION_OVERHAUL`

## 1. 실행 조건

```
uv run python -m src.ml.retrain --tuned --oos-reserve-start 2025-09-01 --no-gate \
  --export-dir artifacts/models/research/v2_default_run \
  --production-dir artifacts/models/research/v2_default_run/prod_shadow \
  --hpo-trials 40
```

| 파라미터 | 값 |
|---|---|
| label_mode | mechanical (익일시가/전일종가-1, 행별비용 차감) |
| cost_mode | per_row (statutory 20bp + KRX 2틱 스프레드, 종목별) |
| screen | operator_legacy (등락률≥10%, 거래대금≥100억, 시총≥500억, 상한가제외) |
| oos_reserve_start | 2025-09-01 (잠긴 OOS 245일) |
| hpo_trials | 40 |
| `--no-gate` | 레거시 `require_beats_control` 우회 (사유 §4) |

번들: `artifacts/models/research/v2_default_run/close_morning61_2025-08-29/sizing_pipeline_bundle.joblib`

## 2. 최종 판정

**`research_only` (미승격)** — 10개 게이트 중 4개 실패.

| gate | passed | observed | threshold | detail |
|---|:--:|---:|---:|---|
| cpcv_ic_path_win_rate | ✅ | 1.0000 | 0.7500 | mde=0.00976 |
| cpcv_top1_path_win_rate | ✅ | 0.8571 | 0.6000 | mde=0.00107 |
| cpcv_delta_significance | ❌ | 0.2392 | 0.0500 | p_bootstrap=0.2392, p_paired_t=0.00783 |
| selection_dsr | ❌ | 0.0356 | 0.9500 | n_selection_trials=1800(40×5×3×3) |
| oos_min_days | ✅ | 245 | 60 | |
| oos_rank_ic_above_mde | ❌ | **-0.1833** | 0.0649(=mde) | 부호 반전 |
| oos_top1_sign | ✅ | +0.000655 | 0 | mde=0.00860(크기판별불가) |
| fillable_top1_sign | ❌ | -0.001017 | 0 | measured_share=0.9909 |
| temporal_sign_consistency | ✅ | 전반+0.001278/후반+0.000696 | 동일부호 | n=1079/896 |
| execution_profile_measured | ✅ | fill_rate=0.8776 | 측정됨 | §3 |

## 3. 체결 실측 (R1, 이번 세션 신규)

| 항목 | 값 |
|---|---:|
| n_rows (OOS top-1 픽) | 245 |
| n_measured (1틱 패시브 지정가 터치) | 215 |
| fill_rate | 0.8776 (상한값 — 1분봉 low≤limit 터치 기준, 큐포지션 미반영) |
| mean_saving_bp | +12.95 |
| filled_gross_bp | 21.42 |
| pool_gross_bp | 6.55 |
| adverse_selection_bp | +14.87 (체결분이 풀 평균보다 gross 높음 — 역선택이 우호적 방향) |
| saving_survives_adverse_selection | **true** |

## 4. 레거시 게이트 설계 결함 (수정 전 1차 실행에서 발견)

`--no-gate` 없이 최초 실행 시 `require_beats_control`(기본 True)이 `cand=0.00101 < ctrl=0.00559`로 판정해 **`ValueError`를 던지고 프로세스를 죽였다** — 이 시점에 이미 신규 v2 `run_promotion_gate`가 전체 게이트(§2)를 다 계산한 뒤였는데, 레거시 raise가 그 결과를 통째로 버리고 아무 아티팩트도 저장하지 않음. R13("research_only도 candidate 아티팩트는 항상 저장")과 정면 충돌. `--no-gate`로 우회해 재실행했고, **레거시 게이트를 신규 게이트 뒤로 재배치하거나 제거하는 것이 다음 액션 1순위**(§6).

레거시 워크포워드 비교(참고, 판정에 미반영): `control_vs_candidate` shared_dates=1975, cand_mean=0.001014, ctrl_mean=0.005594, Δ=-0.00458, p=0.0000.

## 5. 실행 중 발견·수정한 버그 (2026-09-06, 이번 세션)

| 버그 | 위치 | 증상 | 수정 |
|---|---|---|---|
| NaN top-1 픽 크래시 | `cpcv_path_evidence`, `evaluate_locked_oos` | 기계적 라벨 결측(커버리지 99.68~99.71%) 날짜에서 부트스트랩 `ValueError` | finite 마스크로 해당 날짜 제외 |
| `.attrs` 비교불가 크래시 | `select_by_expected_value` | `feature_manifest` DataFrame이 `.attrs`에 남아 `pd.concat`의 `__finalize__`가 "ambiguous truth value" | 슬라이스 전 `work.attrs = {}` (기존 `cpcv_oof_predict`와 동일 패턴) |
| 미정규화 분봉으로 `execution_profile` 항상 None | `measure_oos_execution_profile` | `read_intraday_range`가 원본 KIS 벤더 포맷(`종목코드`,`stck_cntg_hour`)을 그대로 반환 → `simulate_passive_entry`가 `symbol`/`ts_hms` 못 찾고 fail-open | `_load_normalized_bars_for_entries` 신설: 파티션별 `normalize_bar_frame` 정규화 (2026-09-06 `buyability.py` 버그와 동일 클래스) |
| `--screen`/`--production-dir` 죽은 CLI 인자 | `retrain.py` | 파싱만 되고 `args.screen`/전달 안 됨 | `ChampionTuningConfig(screen=SCREEN_REGISTRY[args.screen])`, `production_dir=args.production_dir` 배선 |
| 레거시 raise가 신규 게이트 결과 폐기 | `champion.py::train_tuned_champion_bundle` | §4 | **미수정** — 다음 액션 |

## 6. 스크린 그리드 (R8, dev 33,547행 기준)

| 스크린 | net_bp/일 | t_stat | 종목/일 | n_days |
|---|---:|---:|---:|---:|
| operator_legacy (≥10%) | -37.996 | 2.075 | 12.55 | 2370 |
| band_2_15 (2~15%) | -19.197 | 6.808 | 7.29 | 2341 |
| band_5_15_highvalue (5~15%,≥3000억) | -26.364 | 0.926 | 1.32 | 663 |

3종 전부 실측비용 기준 net 손실. `selection_trials_multiplier=3`이 DSR 분모에 반영됨(§2 selection_dsr n=1800).

## 7. EV 정책 (R4, 참고용 — 배포 미결정)

| | |
|---|---:|
| n_days_total | 1975 |
| n_days_with_position (EV>0) | 536 |
| buy_rate | 0.2714 |
| mean_ev | 0.002225 |
| mean_realized_return (EV>0 채택일) | 0.004785 |
| mde | 0.000233 |

## 8. 해석

- `oos_rank_ic_above_mde=-0.1833`은 부호 반전 — journaled 라벨로 학습된 모델이 mechanical 라벨(다음날 시가, 운영자 재량 청산 제외)로 채점하면 OOS에서 방향을 잃음. `journaled-label-carries-operator-exit-skill` 메모의 예측과 일치하는 추가 근거.
- `execution_profile`은 처음으로 실측됨: 패시브 절감(+13bp)이 역선택(+14.9bp, 우호적 방향)을 상쇄하고도 순양(+27.8bp) — 체결 방식 전환 자체의 경제성은 이 표본에서 유지.
- `selection_dsr=0.0356`은 이전 세션 arm A(0.99999996, 재확인편향으로 포화)와 달리 정직한 예산(1800회) 반영 시 유의성이 사실상 소멸함을 보여줌 — 게이트가 설계대로 작동.

## 9. 다음 액션

1. **최우선**: §4의 레거시 게이트 순서 결함 수정 — `require_beats_control` raise를 `config.validation is not None`일 때 무력화하거나 `run_promotion_gate` 뒤로 재배치.
2. mechanical 라벨의 OOS 부호반전 원인 분해 — journaled 대비 rankIC 델타를 CPCV 폴드별로 breakdown.
3. `execution_profile`의 `fill_rate` 상한 편향 정량화 — 큐포지션 시뮬레이션(터치 후 체결확률<1) 도입 검토.
4. 스크린 그리드에 완전 유동성 유니버스(시총 하한만, 등락률 무제한) 추가해 -30bp대 기저율의 하한 확인.
