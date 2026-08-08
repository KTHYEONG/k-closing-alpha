# ML Feature Selection & Optimization Detailed Report (Latest v2)

**Date**: 2026-08-08  
**Scope**: Causal History Feature Quality Repair & 2025 Re-training
**Evaluation Mode**: Discovery Mode (`frozen cutoff: 2025-12-30`, `purge_gap: 1`, `n_splits: 5`, `causal_history_v2`)

> 이 문서의 최신 권위 결과는 아래 v2 실행입니다. 기존 v1/2-fold 수치는 하단에
> 비교용으로 남겨 둔 과거 스냅샷입니다.

## 0. Latest v2 Execution (2025 cutoff)

### 적용한 문제 해결

| 항목 | 원인 | 조치 | 결과 |
| :--- | :--- | :--- | :--- |
| 시장 컨텍스트 전부 결측 | `RangeIndex`와 날짜 index의 pandas 정렬 불일치 | 배열 기반 index 할당 + 날짜별 중앙값 집계 | `index_ret_*_0` 유한율 약 99.4% |
| 시장 streaming 경로 불일치 | parquet 배치별 첫 행만 보존 | 배치/최종 컬럼별 순서 불변 중앙값 집계 | DataFrame/Parquet 동일성 테스트 통과 |
| 수급 변화량 결측 | 순매수 0의 0/0 비율 | 부호 보존 signed-log 변화량 | flow ratio 유한율 95.4~99.0% |
| 0 분산 z-score 결측 | 상수 구간을 데이터 결측으로 처리 | 유효한 0 scale을 중립값 0으로 처리 | 정상 상수 구간 보존 |
| 안정성 오판 | 2-fold에서 선택 1회도 rate 0.5로 stable 판정 | 기본 `min_fold_selection_rate=1.0` | 모든 fold 선택만 stable |
| `volume_power` 원천 부족 | trade log의 체결강도 유효 행이 약 0.46% | 보간하지 않고 baseline 결측률 gate에서 제외 | baseline 61→60개 |
| 장기 룩백 메타데이터 오류 | `ma_slope_w`는 실제로 약 2w 관측 필요 | catalogue lookback을 `2*w`로 수정 | `ma_slope_240` lookback 480 |

### 성능 결과

| 지표 | Control | Candidate (v2) |
| :--- | ---: | ---: |
| scheduled dates | 2,040 | 2,040 |
| buy count | 2,035 | 2,035 |
| mean return | **1.5933%** | **1.5001%** |
| win rate | 63.53% | **63.68%** |
| profit factor | **2.8687** | 2.6713 |
| Sharpe | **6.3150** | 5.8920 |
| entry-sequence MDD | **25.87%** | 30.60% |

판정은 **promotion 거부**입니다. Candidate의 평균 수익률과 MDD가 control보다 엄격히
좋지 않아 현재 모델을 대체할 근거가 없습니다. 다만 candidate 자체는 양의 평균수익과
PF>1을 유지하므로 피처 연구용 후보로는 유효합니다.

### v2 품질 진단

- 전체 후보: **780개** (baseline 60 + causal history 720), 최종 선택 400개.
- 5-fold 기준 `capacity_limited` 517개, `unstable` 637개, `source_incomplete` 91개입니다.
  액션은 중복 집계이므로 합계가 후보 수와 일치하지 않습니다.
- 시장 패밀리는 72개 중 70개가 적어도 한 fold에서 선택됐고, 평균 선택률은 83.1%입니다.
- 수급 변화량 ratio 피처는 0 분모 때문에 전부 결측이던 상태에서 벗어났지만, 장기
  rolling 피처는 종목 이력 길이 부족으로 여전히 자연 결측이 발생합니다.
- 가장 큰 잔여 이슈는 품질 결함이 아닌 400개 cap입니다. 양의 gain이 있어도 fold별
  cap 밖으로 밀린 피처가 517개 액션에 포함되므로, cap 변경은 별도 ablation 없이는
  적용하지 않습니다.

### 실행·아티팩트

- 2026 데이터 제외: **139 rows**, evaluation cutoff **2025-12-30**.
- history input **5,046,547 rows**, output **33,520 keys**, 8 batches.
- history build: **457.46s**, peak RSS **1.36GB**, nonfinite sanitization **0**.
- artifact: `artifacts/models/research-2025-quality-v2/causal_history_v2/2025-12-30/sizing_pipeline_bundle.joblib`

이하의 기존 장은 v1/2-fold 실행의 원인·상세를 보존한 역사적 비교 자료입니다.

---

## 1. Feature Selection Overview & Summary

본 보고서는 720개 Causal History 피처 카탈로그와 61개 기존 베이스라인 피처(총 781개 후보)에 대한 **Fold-Local Feature Selection 결과 분석 보고서**입니다.

### Key Metrics Summary
- **Total Candidate Features**: **781 개** (베이스라인 61개 + Causal History 720개)
- **Positive Gain Features**: **589 개**
- **Retained Features (`max_retained`)**: **400 개**
- **Fold Stability (Jaccard Index)**: **0.5810** (Null Baseline: 0.3442, `null_gate_quantile`: 0.95, **`stability_gate_passed = True`**)

---

## 2. Feature Rejection Analysis (피처 거부 사유 분석)

품질 필터링, LightGBM Huber screening gain 평가, Spearman 상관관계 가지치기(correlation threshold 0.98) 단계에서의 거부 상세 내역입니다.

| 거부 사유 (`Rejection Reason`) | 거부 피처 수 | 세부 내용 및 설명 |
| :--- | :---: | :--- |
| **`beyond_max_retained`** | **179 개** | 양의 Gain을 가졌으나 `max_retained` 상한(400개) 초과로 거부됨 |
| **`zero_gain`** | **122 개** | LightGBM Huber 스크리닝 모델에서 Gain 평가 수치가 0 이하인 피처 |
| **`all_nonfinite`** | **65 개** | 데이터 전구간에서 모든 유한값이 존재하지 않아 거부 |
| **`correlated_pair_pruned`**| **10 개** | Absolute Spearman 상관계수 $> 0.98$ 인 피처 쌍에서 Gain 순위로 가지치기 |
| **`train_missing_rate`** | **5 개** | Train 분할 결측률 $> 0.35$ 초과로 인한 거부 |

---

## 3. Top Feature Importance & Gain Analysis

### 3.1. Overall Top 20 Features (Base + History Combined)

| 순위 | 피처 이름 (`Feature Symbol`) | Gain Importance | 피처 범주 / 설명 |
| :---: | :--- | :---: | :--- |
| **1** | `close_position` | **5.5799** | 당일 종가 위치 (Base) |
| **2** | `kosdaq_change` | **2.2014** | KOSDAQ 지수 등락률 (Base) |
| **3** | `relative_change_kosdaq` | **1.5442** | KOSDAQ 대비 상대 등락률 (Base) |
| **4** | `change_rate` | **1.4640** | 종목 등락률 (Base) |
| **5** | `relative_change_rate` | **1.4554** | 시장 대비 상대 등락률 (Base) |
| **6** | `buy_price_change_rate` | **1.1970** | 매수가 대비 등락률 (Base) |
| **7** | `v_kosdaq` | **1.0659** | 코스닥 변동성 지수 (Base) |
| **8** | `kospi_change` | **0.9890** | KOSPI 지수 등락률 (Base) |
| **9** | `total_candidate_count` | **0.9832** | 당일 후보 모수 수 (Base) |
| **10** | `intraday_range` | **0.9595** | 일중 변동 폭 (Base) |
| **11** | `gap_ratio` | **0.7987** | 시가 갭 비율 (Base) |
| **12** | `v_kospi` | **0.7868** | 코스피 변동성 지수 (Base) |
| **13** | `body_ratio` | **0.7457** | 캔들 몸통 비율 (Base) |
| **14** | `inst_net_buy` | **0.6798** | 기관 순매수 (Base) |
| **15** | `foreign_net_buy` | **0.6775** | 외국인 순매수 (Base) |
| **16** | `log_avg_trade_value` | **0.6727** | 로그 평균 거래대금 (Base) |
| **17** | `major_density` | **0.6442** | 주체별 수급 밀도 (Base) |
| **18** | `inst_density` | **0.5753** | 기관 수급 밀도 (Base) |
| **19** | `v_kospi_change` | **0.5285** | 코스피 변동성 변화 (Base) |
| **20** | `prog_dominance` | **0.5226** | 프로그램 매수 지배도 (Base) |

---

### 3.2. Top 20 Causal History Candidate Features

720개 Causal History 피처 중 LightGBM screening에서 가장 높은 상위 기여도를 보인 피처 목록입니다.

| 순위 | 피처 이름 (`Feature Symbol`) | Gain Importance | 피처 패밀리 및 신호 특성 |
| :---: | :--- | :---: | :--- |
| **1** | `relative_flow_strength` | **0.4665** | 수급 강도 (Investor Flow Dynamics) |
| **2** | `change_rate_z` | **0.4545** | Robust Z-Score 등락률 (Mean Reversion) |
| **3** | `gap_ratio_1` | **0.3998** | 1일 시차 시가 갭 비율 (OHLC Range/Gap) |
| **4** | `range_volatility_20` | **0.3927** | 20일 변동폭 변동성 (Volatility Grid) |
| **5** | `major_density_z` | **0.3704** | 주체별 수급 밀도 Robust Z-Score |
| **6** | `parkinson_vol_5` | **0.3251** | 5일 Parkinson High-Low 변동성 |
| **7** | `inst_flow_roll_sum_240`| **0.3226** | 240일 기관 누적 수급 흐름 |
| **8** | `candle_body_0` | **0.3204** | 당일 캔들 실체 비율 |
| **9** | `trix_10` | **0.3152** | 10일 TRIX (Triple EWMA Trend) |
| **10** | `inst_net_buy_pct_rank` | **0.3070** | 기관 순매수 백분위 순위 (Cross-Sectional Rank) |
| **11** | `turnover` | **0.2795** | 거래 회전율 (Liquidity/Turnover) |
| **12** | `rsi_20` | **0.2593** | 20일 상대강도지수 (RSI Momentum) |
| **13** | `trix_5` | **0.2574** | 5일 TRIX 추세 지표 |
| **14** | `relative_change_kospi` | **0.2458** | KOSPI 대비 상대 등락률 |
| **15** | `rsi_14` | **0.2431** | 14일 RSI 상대강도지수 |
| **16** | `range_volatility_60` | **0.2357** | 60일 변동폭 변동성 |
| **17** | `v_kosdaq_change` | **0.2349** | KOSDAQ 변동성 지수 변화율 (Market Context) |
| **18** | `ma_slope_240` | **0.2325** | 240일 장기 이동평균선 기울기 |
| **19** | `scenario_other` | **0.2258** | 시나리오 분류 카테고리 |
| **20** | `bollinger_upper_dist_10`| **0.2227** | 10일 볼린저 밴드 상단 이격도 |

---

## 4. Outer Fold-Local Selection Breakdown

각 Outer Fold (Walk-Forward Train 분할) 단위에서의 피처 생존 및 거부 수치 비교입니다.

| Fold Index | 총 후보 수 (`n_candidates`) | 품질 생존 (`n_survived_quality`) | Positive Gain (`n_positive_gain`) | 최종 거부 수 (`n_rejected`) | 최종 유지 수 (`n_retained`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 781 | 693 | 572 | 381 | **400** |
| **Fold 1** | 781 | 709 | 581 | 381 | **400** |

---

## 5. 결론 및 피처 인사이트

1. **지수 및 수급 밀도 피처의 강한 기여도**: `close_position`, `kosdaq_change`, `relative_change_kosdaq` 등 단기 위치 및 시장 대비 상대 수급 지표가 상위 기여도를 지배하고 있습니다.
2. **Causal History 피처의 보조적 기여**: 720개 피처 중 `relative_flow_strength`, `change_rate_z`, `parkinson_vol_5`, `inst_flow_roll_sum_240` 등이 중장기 추세/변동성/수급 누적 신호로서 상위 400개 피처 목록에 다수 채택되었습니다.
3. **안정적인 Fold 교집합**: Outer Fold 간 선택 피처의 Jaccard 유사도가 **0.5810**으로 Null 모형(0.3442)을 크게 상회하여, 날짜 분할 변화에 흔들리지 않는 높은 결정적 피처 선택 안정성을 입증했습니다.
