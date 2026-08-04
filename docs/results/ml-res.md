# [2026-08-04] ML Model Training & Purged Walk-Forward Backtest Evaluation Report

## 1. Executive Summary & Objective

본 보고서는 `k-closing-alpha` 시스템의 ML 학습 파이프라인(`Purged Group TimeSeries Split Walk-Forward CV`)을 실제 매매일지 DB 데이터에 실행하여 얻은 실증적 성능 수치와 지표를 기반으로, 모델의 우수성과 구조적 한계점을 디테일하고 냉정하게 평가 및 분석한 종합 보고서입니다.

- **실행 대상 데이터**: 33,934 건 (횡단면 일단위 스냅샷 결합 데이터, 2017년 ~ 2026년)
- **검증 기법**: `PurgedGroupTimeSeriesSplit` (n_splits=5, purge_gap=1일) - Look-ahead Bias 및 Serial Correlation 완전 차단
- **평가 모델**:
  1. **Primary Ranker (`LGBMRanker`)**: 횡단면 상위 종목 선별 (LambdaRank)
  2. **Primary Regressor (`LGBMRegressor`)**: Huber Loss 기반 순수익률 직접 추정
  3. **Baseline Model (`Ridge Linear Regression`)**: 선형 벤치마크
  4. **Dynamic Position Sizing Engine**: Quantile Regressor(q10, q50, q90) + Calibrated Classifier(p_good, p_bad) 기반 유틸리티 스코어링 및 비중 배분 엔진

---

## 2. Dataset Specification & Target Analysis

### 2.1 데이터셋 사양 (Dataset Specification)
- **총 샘플 수 (Total Sample Count)**: 33,934 건
- **학습 피처 차원 (Feature Dimension)**: **49개** 수치형 피처 (횡단면 Robust Z-Score, Log 변환, 상대 강도, 시장 레짐 등)
- **비용 공제 반영 (Cost Deducted)**: 왕복 거래 비용 **`0.20%` (0.0020)** (증권거래세 0.15% + 수수료 0.028% + 슬리피지 0.022%)를 타깃 변수(`target_return = net_return - 0.20%`)에 미리 차감하여 Net-of-Cost 알파 평가.

### 2.2 타깃 수익률 (`target_return`) 통계적 분포

| 통계 항목 (Metric) | 수치 (Value) | 해석 및 의미 |
| :--- | :---: | :--- |
| **샘플 수 (Count)** | `33,934` | 전체 일별 조건검색 포착 종목 총합 |
| **평균 순수익률 (Mean)** | **`+0.0511%`** | 무작위 매수 시 기대 수수료 차감 후 수익률 |
| **표준편차 (Std Dev)** | **`3.4280%`** | 일일 수익률의 높은 변동성 수준 |
| **최소값 (Min)** | `-10.0000%` | 하방 꼬리 부분 클리핑 제한 |
| **하방 10% (q10)** | **`-3.5600%`** | 하방 손실 위험 마진 |
| **하방 25% (q25)** | **`-2.0500%`** | 1차 손절 바운더리 |
| **중앙값 (q50 / Median)** | **`-0.2000%`** | **비용 차감 후 중앙값이 음수(-0.20%)에 위치** |
| **상방 75% (q75)** | `+1.6900%` | 상방 익절 영역 진입점 |
| **상방 90% (q90)** | **`+4.3300%`** | 상방 손익비 극대화 영역 |
| **최대값 (Max)** | `+10.0000%` | 상방 꼬리 부분 클리핑 제한 |

> **냉정한 냉철적 분석**:
> 데이터셋 전체의 중앙값(`q50`)이 **`-0.20%`**에 형성되어 있어, 정밀한 선별 필터링 없이 무작위로 진입할 경우 수수료 및 슬리피지로 인해 계좌가 우하향하는 구조적 음의 편향(Negative Bias)을 가집니다. 따라서 모델의 핵심 역량은 단순히 수익을 내는 종목을 찾는 것뿐만 아니라, **손실 가능성이 높은 85%+ 이상의 무익한 후보 종목을 'Pass(진입 거부)' 시키는 정밀 방어 능력**에 있습니다.

---

## 3. Comprehensive Model Performance Metrics & Comparison

CV Out-Of-Fold(OOF) 전체 구간 및 3개 알고리즘의 오프라인 랭킹/수익성 지표 비교입니다.

### 3.1 횡단면 랭킹 및 백테스트 평가 지표 비교

| 지표명 (Metric) | Baseline (`Ridge`) | Primary (`LGBMRanker`) | Primary (`LGBMRegressor`) | 비고 / Best Model |
| :--- | :---: | :---: | :---: | :--- |
| **NDCG@1** | 0.5026 | 0.5332 | **0.5424** | `LGBMRegressor` 우세 (+0.0398 vs Ridge) |
| **NDCG@3** | 0.5220 | 0.5351 | **0.5457** | `LGBMRegressor` 우세 (+0.0237 vs Ridge) |
| **Rank IC (Spearman)** | 0.1729 | 0.1870 | **0.2116** | `LGBMRegressor` 우세 (+0.0387 vs Ridge) |
| **Top-1 평균 순수익률** | +1.0776% | +1.3220% | **+1.4366%** | `LGBMRegressor` 최고 (Ridge 대비 +33.3% 향상) |
| **Top-3 평균 순수익률** | +0.7030% | +0.7314% | **+0.8632%** | `LGBMRegressor` 최고 |
| **승률 (Win Rate)** | 59.26% | 61.90% | **62.27%** | `LGBMRegressor` 우세 (+3.01%p vs Ridge) |
| **Profit Factor** | 2.0954 | 2.2843 | **2.5142** | `LGBMRegressor` 우세 (손익비 월등) |
| **Sharpe Ratio (연율화)** | 4.4199 | 5.0486 | **5.5697** | `LGBMRegressor` 우세 (샤프지수 5.57 달성) |
| **Equal-Weight Baseline Sharpe** | 0.1802 | 0.1802 | 0.1802 | 단순 무작위 매수 대비 샤프지수 30.9배 능가 |

---

## 4. Yearly Breakdown & Temporal Stability Analysis

시간 흐름 및 연도별 장세 변화에 따른 모델의 일관성과 레짐 체인지 대응 능력을 검증하기 위해, `LGBMRegressor` 및 `LGBMRanker` 모델의 연도별 OOF 백테스트 결과를 분해 분석했습니다.

### 4.1 연도별 Top-1 & Top-3 성능 분해 표 (`LGBMRegressor` 기준)

| 연도 (Year) | Top-1 순수익률 (%) | Top-3 순수익률 (%) | 승률 (Win Rate %) | Profit Factor | 연율화 Sharpe Ratio | 시장 장세 평가 |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2017** | **+1.9914%** | +1.5258% | 70.91% | 3.8594 | 7.5575 | 대형 강세장 (Ultra Bull) |
| **2018** | **+1.5864%** | +1.0721% | 66.67% | 3.7223 | 7.7751 | 미중 무역분쟁 약세장 (Bear) |
| **2019** | **+1.7046%** | +1.0428% | 66.26% | 2.9515 | 6.5829 | 박스권 장세 (Range) |
| **2020** | **+2.2513%** | +1.5974% | 65.73% | 3.3503 | 7.5435 | 유동성 폭발 유동장 (Post-COVID) |
| **2021** | **+2.0183%** | +0.9856% | 69.76% | 3.6088 | 7.8955 | 테마 순환매장 (Bull/Vol) |
| **2022** | **+1.0047%** | +0.5067% | 56.91% | 1.9081 | 3.8407 | 금리인하 악재 대형 금리 하락장 |
| **2023** | **+1.4920%** | +0.7755% | 62.45% | 2.5881 | 5.7247 | 2차전지/AI 쏠림 장세 |
| **2024** | **+1.1779%** | +0.5814% | 59.84% | 2.1907 | 4.7666 | 밸류업/반도체 차별화 장세 |
| **2025** | **+1.0482%** | +0.8030% | 55.79% | 2.1462 | 4.5042 | 고금리 장기화 및 지정학 리스크 |
| **2026 (최근)** | **`-0.2248%`** | **`-0.2330%`** | **49.28%** | **0.8785** | **`-0.8067`** | **최근 최근 샘플 레짐 붕괴 발생** |

> **Critical Discovery (냉정한 맹점 분석)**:
> 1. **2017~2025년 장기 안정성**: 2017년부터 2025년까지 9년간 모델은 시장 상승/하락과 무관하게 연평균 Top-1 수익률 `+1.0% ~ +2.25%`, 승률 `55% ~ 70%`, Sharpe `3.8 ~ 7.9` 수준의 압도적인 성과를 일관되게 보여주었습니다.
> 2. **2026년 최근 데이터 성능 급락**: 2026년 구간에서는 Top-1 순수익률이 **`-0.2248%`**, 승률 **`49.28%`**, Profit Factor **`0.8785`**로 하강하며 손실 구간으로 반전되었습니다. 이는 최근 시장 패턴 변화(급변동성/저스프레드/알고리즘 교란) 또는 전처리된 피처의 최근 레짐 미반영 현상에 기인한 것으로 해석됩니다.

---

## 5. Feature Importance & Data Attribution

5개 폴드 앙상블 `LGBMRanker` 모델의 피처 분할 횟수(Split Count) 평균 기준 상위 15개 핵심 피처입니다.

| 순위 | 피처명 (Feature Name) | Mean Split Count | 피처 범주 | 도메인 해석 및 영향 |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `intraday_range` | **111.4** | Price Dynamics | 당일 고가와 저가 간의 일중 변동폭 (변동성 크기 반영) |
| **2** | `buy_price_change_rate` | **100.8** | Price Dynamics | 매수 시점 가격의 전일 대비 등락률 (추세 및 모멘텀) |
| **3** | `log_volume` | **100.2** | Liquidity | 당일 거래량의 Log 스케일 (수급 및 거래 활성도) |
| **4** | `body_ratio` | **90.6** | Candle Shape | 캔들 몸통 비율 (당일 시가 대비 종가 추진력) |
| **5** | `gap_ratio` | **85.4** | Price Dynamics | 시가 갭 비율 (장 시작 시 시장 기대감) |
| **6** | `log_avg_trade_value` | **84.8** | Liquidity | 평균 거래대금의 Log 스케일 (종목 유동성 지지선) |
| **7** | `v_kospi_change` | **82.0** | Regime | KOSPI 변동성 지수 변화율 (시장 위협 감지) |
| **8** | `log_market_cap_100m` | **78.0** | Scale | 시가총액 Log 스케일 (소형주 슬리피지 예방 필터) |
| **9** | `close_position` | **77.8** | Candle Shape | 고저 대비 종가 위치 (상방 종가 마감 강도) |
| **10** | `gap_ratio_z` | **76.8** | Relative Z-score | 시가 갭 비율의 횡단면 Robust Z-Score |
| **11** | `change_rate` | **76.6** | Price Dynamics | 당일 등락률 |
| **12** | `change_rate_z` | **74.0** | Relative Z-score | 당일 등락률의 횡단면 Robust Z-Score |
| **13** | `v_kosdaq_change` | **73.8** | Regime | KOSDAQ 변동성 지수 변화율 |
| **14** | `relative_change_rate` | **71.6** | Relative Strength | 지수 대비 종목 상대 수익률 |
| **15** | `v_kospi` | **71.0** | Regime | KOSPI 변동성 지수 절대수치 |

---

## 6. Dynamic Sizing Engine Metrics Analysis

Quantile Regressors(q10, q50, q90)와 Calibrated Classifier(p_good, p_bad)를 결합한 유틸리티 기반 포지션 사이징 엔진의 실제 실증 분포입니다.

### 6.1 Sizing Grade 분포 (`31,156` 샘플 분석)

```
Strong  :   1,199 건  (  3.85% )  -->  1.5x 배수 할당
Good    :     432 건  (  1.39% )  -->  1.0x 배수 할당
Weak    :     160 건  (  0.51% )  -->  0.5x 배수 할당
Pass    :  29,365 건  ( 94.25% )  -->  0.0x 진입 거부 (Pass)
```

### 6.2 Sizing Metrics 통계적 스냅샷

- **Utility Score 분포**:
  - **평균 (Mean)**: **`-1.9892`** (전체 포착 종목의 대다수가 비수익 위험 영역에 존재)
  - **중앙값 (q50)**: `-1.8605`
  - **최대값 (Max)**: `+3.1923` (최상위 알파 종목은 높은 플러스 유틸리티 확보)
- **Allocation Ratio (자산 비중)**:
  - **평균 비중**: `0.022%` (전체 자산 대비)
  - **Pass 비율**: **`94.25%`** 의 종목에 대해 투자 비중 0.0%를 부여함으로써 손실 위험 거래를 사전에 전면 차단.
  - **최대 비중 (Max Allocation)**: `16.95%` (가장 확신도 높은 극소수 종목에 집중 배분)

---

## 7. Cold-Headed Evaluation & Risk Identification (냉정한 종합 평가 및 위험요인)

### 7.1 강점 (Strengths)
1. **월등한 횡단면 알파 발굴**: `LGBMRegressor` 및 `LGBMRanker` 모델 모두 Rank IC `0.18 ~ 0.21`, Sharpe `5.0 ~ 5.5`를 달성하여 단순 무작위 매수(Sharpe 0.18) 대비 확연한 횡단면 선별 능력을 증명함.
2. **비용 후 순수익 극대화**: 왕복 0.20% 수수료 차감 후에도 Top-1 선택 시 평균 **`+1.44%`**의 높은 일일 순수익률을 기록함.
3. **철저한 손실 방어 필터링**: Sizing Engine이 전체 조건검색 포착 종목 중 **`94.25%`**를 `Pass(비중 0.0%)` 처리하여 음의 기댓값을 가진 불필요한 거래 발생을 원천 방지함.

### 7.2 한계점 및 개선 과제 (Limitations & Vulnerabilities)
1. **2026년 최근 구간 레짐 붕괴 (Regime Decay)**:
   - 2017~2025년 동안의 우수한 성과와 달리, 2026년 최근 데이터에서는 Sharpe `-0.8067`, Win Rate `49.28%`로 급락함.
   - **원인 추정**: 시장 변동성 지수(`v_kospi`, `v_kosdaq`)의 구조적 변화나 매수 체결가와 시가/종가 간의 갭 변동 패턴이 달라졌음에도 과거 패턴에 과적합(Overfitting)되었을 가능성.
2. **선형 모델(`Ridge`)의 상대적 열세**:
   - `Ridge` 모델은 Rank IC `0.1729`, Sharpe `4.42`로 Tree 기반 모델 대비 랭킹 성능 및 수익률(Top-1 +1.07%)이 전반적으로 뒤처짐. 비선형 피처 상호작용 반영이 부족함.
3. **범주형 피처(Categorical Features) 미반영 한계**:
   - LightGBM 학습 시 `market_type`, `theme_sector`, `chart_analysis` 등 문자열 범주형 컬럼이 수치형으로 인코딩되지 않아 현재 모델 학습에서 제외되고 수치형 피처만 활용되고 있음.
