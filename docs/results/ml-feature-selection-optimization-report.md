# ML Feature Selection & Optimization Detailed Report

**Date**: 2026-08-08  
**Scope**: 720 Causal History Candidate Features & Fold-Local Feature Selection Audit  
**Evaluation Mode**: Discovery Mode (`frozen cutoff: 2025-12-30`, `purge_gap: 1`, `n_splits: 2`)  

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
