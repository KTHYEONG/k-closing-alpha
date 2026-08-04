
## 결론

이 프로젝트는 일반적인 **수익률 회귀 문제**보다 다음과 같이 설계하는 것이 적합합니다.

> **당일 후보 종목을 하나의 그룹으로 묶고 순위를 예측한 뒤, 예상수익·하방위험·확률 신뢰도를 별도로 추정하여 투자 비중을 결정한다.**

권장 구성은 다음입니다.

1. **LightGBM/CatBoost 랭킹 모델**: 당일 후보 중 우선순위 결정
2. **Quantile Regression**: 예상수익과 하방위험 추정
3. **확률 분류 모델 + 확률 보정**: 목표수익 달성 확률 추정
4. **날짜 단위 Walk-forward 검증**: 미래 데이터 누수 차단
5. **OOF 예측값 기반 비중 등급**: Strong / Good / Weak / Pass

LightGBM은 `lambdarank`와 query/group 기반 랭킹을 지원하고, CatBoost도 YetiRank 계열의 랭킹 목적함수를 제공한다. LightGBM은 quantile regression 목적함수도 지원한다. ([LightGBM][1])

---

# 1. 가장 먼저 확인해야 할 데이터 누수

알고리즘보다 먼저 아래 문제를 해결해야 합니다.

## 1.1 매수 시점과 데이터 확정 시점

예를 들어 15시 20분에 주문을 결정하는 전략이라면 다음 데이터의 **당일 최종값**을 그대로 사용하면 안 됩니다.

* 당일 종가
* 당일 최종 고가·저가
* 당일 최종 거래대금·거래량
* 최종 기관·외국인·프로그램 순매수
* 장 종료 후 산출되는 체결강도

주문 결정 시점이 15시 20분이라면 반드시 **15시 20분 스냅샷**을 저장해야 합니다.

반대로 장 종료 후 데이터를 이용한다면 실제 매수 가격은 당일 종가가 아니라 다음 거래일 시가, VWAP 또는 실제 체결 가능 가격으로 정의해야 합니다. 일본 주식의 cross-sectional 예측 연구에서도 종가 시점 정보를 사용한 경우 다음 날 시가에 투자하는 형태로 실행 시점을 분리했다. ([arXiv][2])

각 변수에 다음 컬럼을 추가하는 것이 좋습니다.

```text
signal_timestamp
feature_available_timestamp
order_timestamp
execution_timestamp
```

---

## 1.2 매수한 종목만 저장하면 안 됨

모델의 입력 단위는 반드시 다음이어야 합니다.

```text
날짜 × 당일 포착된 모든 후보 종목
```

즉, 매수하지 않은 후보 종목도 동일한 매도 규칙으로 미래수익률을 계산해야 합니다.

예를 들어 당일 후보가 12개였는데 실제 매수한 2개만 보유하고 있다면, 나머지 10개의 결과가 없으므로 모델은 다음을 학습할 수 없습니다.

* 선택한 종목이 다른 후보보다 실제로 우수했는가
* 기존 선정 순위가 유효했는가
* 어떤 후보를 제외해야 하는가

**매매 종목 데이터가 아니라 후보군 데이터셋**이어야 합니다.

---

## 1.3 종목코드와 선정 순위

* `종목코드`: 숫자형으로 넣으면 안 됩니다. 처음에는 제외하거나 범주형으로 사용하십시오.
* `선정 순위`: 반드시 모델 입력으로 넣은 버전과 제외한 버전을 비교해야 합니다.
* `차트분석`: 사전에 작성된 값만 사용해야 합니다. 결과를 확인한 후 작성된 분석이면 완전한 누수입니다.
* `테마/섹터`: 당시 시점의 분류를 사용해야 합니다. 현재 테마 분류를 과거 데이터에 소급 적용하면 편향이 생길 수 있습니다.

CatBoost는 숫자형·범주형·텍스트 변수를 직접 지원하므로, 테마·섹터·표준화된 차트 패턴을 처리하는 challenger 모델로 적합합니다. ([CatBoost][3])

---

# 2. 타깃 변수 설계

## 2.1 실제 체결 기준 순수익률

모델 타깃은 단순한 종가 변화율이 아니라 실제 투자 가능한 순수익률이어야 합니다.

[
r^{net}*i =
\frac{
P^{sell}*{exec}(1-fee_{sell}-tax)
}{
P^{buy}*{exec}(1+fee*{buy})
}-1
]

여기에 실제 슬리피지를 체결 가격에 반영합니다.

중요한 원칙은 다음과 같습니다.

* 매도 규칙이 다르면 모델도 분리
* 익일 시가 매도와 익일 종가 매도를 하나의 타깃으로 혼합하지 않음
* 손절·익절·기간청산이 혼합되면 해당 규칙을 완전히 재현
* 거래세와 수수료는 해당 연도의 실제 비용 적용
* 거래정지, 갭, 상·하한가, 미체결을 백테스트에서 처리

---

## 2.2 세 종류의 타깃을 동시에 사용

### A. 랭킹 타깃

각 날짜 안에서 실현 순수익률을 순위화합니다.

예:

| 당일 수익률 백분위 | 랭킹 라벨 |
| -----------------: | --------: |
|           하위 20% |         0 |
|             20~40% |         1 |
|             40~60% |         2 |
|             60~80% |         3 |
|           상위 20% |         4 |

후보가 적은 날짜에는 실제 순위를 0~4 사이로 변환합니다.

```text
query_id = 매수날짜
item = 종목
relevance = 당일 후보군 내 미래 순수익률 등급
```

이 방식은 “수익률 2.1%를 정확하게 맞히는 것”보다 “오늘 후보 중 무엇이 더 좋은가”에 직접 대응합니다.

### B. 연속 수익률 타깃

```text
target_return = 실제 순수익률
```

* Huber loss 회귀
* MAE 회귀
* Quantile regression

으로 수익률 크기를 추정합니다.

### C. 확률 타깃

```text
target_good = 1 if 순수익률 > 거래비용 + 목표마진 else 0
target_bad  = 1 if 순수익률 < 허용손실기준 else 0
```

이를 통해 다음을 추정합니다.

```text
P(목표수익 달성)
P(큰 손실 발생)
```

---

# 3. 권장 알고리즘

## 3.1 반드시 구축할 기준 모델

복잡한 모델만 비교하면 성과가 실제 개선인지 판단할 수 없습니다.

| 모델                | 목적                              |
| ------------------- | --------------------------------- |
| 기존 선정 순위      | 현재 로직 기준선                  |
| 후보 전체 동일가중  | 선별 자체의 효과 확인             |
| Ridge/ElasticNet    | 선형 기준 모델                    |
| Logistic Regression | 수익 확률 기준 모델               |
| 단순 규칙           | 거래대금·등락률 등 기존 휴리스틱 |

머신러닝 자산가격 연구에서도 선형모형과 비선형 머신러닝 모델을 함께 비교하는 방식이 사용된다. ([OUP Academic][4])

---

## 3.2 핵심 모델: LightGBM Ranker

첫 번째 주력 모델로 권장합니다.

```text
Model       : LGBMRanker
Objective   : lambdarank
Group       : 날짜별 후보 종목 수
Metric      : NDCG@1, NDCG@3, NDCG@5
Label       : 날짜별 수익률 등급 0~4
```

초기 탐색 범위 예시는 다음과 같습니다.

```text
max_depth        : 3~6
num_leaves       : 7~31
learning_rate    : 0.01~0.05
min_child_samples: 100~500
feature_fraction : 0.6~0.9
bagging_fraction : 0.6~0.9
```

3만 행은 딥러닝을 우선 적용할 정도로 큰 데이터가 아닙니다. 특히 같은 날짜 후보들은 동일한 시장 상태를 공유하기 때문에 실질적인 독립 표본 수는 3만 개보다 거래일 수에 더 가깝습니다. 따라서 얕은 트리와 강한 정규화가 필요합니다.

---

## 3.3 Challenger: CatBoost Ranker

다음 변수가 중요하다면 CatBoost를 함께 비교하십시오.

* 시장구분
* 테마
* 섹터
* 표준화된 차트 패턴
* 종목 특성 범주
* 복수 테마 태그

추천 목적함수:

```text
YetiRank
YetiRankPairwise
QueryRMSE
```

CatBoost의 공식 문서는 여러 랭킹 목적함수와 query 단위 학습을 제공한다. ([CatBoost][5])

LightGBM과 CatBoost를 처음부터 평균내지 말고, 각각의 OOF 성과를 비교한 후 두 모델이 독립적인 개선을 제공할 때만 앙상블하십시오.

---

## 3.4 하방위험 모델: Quantile Regression

평균 예상수익률 하나만 예측하면 베팅 비중을 결정하기 어렵습니다.

다음 세 분위수를 별도로 예측하십시오.

```text
q10 = 비관적 수익률
q50 = 중앙 예상수익률
q90 = 낙관적 수익률
```

예측 구간:

[
[q_{0.1}, q_{0.9}]
]

활용 예:

```text
중앙 예상수익은 높지만 q10이 매우 낮음
→ 고수익·고위험 종목

중앙 예상수익은 중간이지만 q10이 양호함
→ 안정적인 종목
```

Conformalized Quantile Regression은 quantile regression을 calibration 데이터로 보정하여 적응형 예측구간을 구성하는 방법이다. 다만 금융시장의 비정상성과 시계열 분포 변화 때문에 이론적 보장만 믿지 말고 기간별 실제 coverage를 별도로 확인해야 한다. ([arXiv][6])

---

# 4. 피처 엔지니어링

현재 변수는 상당히 유용하지만, 절대값보다 **상대값과 정규화 값**을 추가해야 합니다.

## 4.1 가격·차트

```text
시가갭 = 시가 / 전일종가 - 1
일중수익률 = 종가 / 시가 - 1
일중범위 = (고가 - 저가) / 전일종가
종가위치 = (종가 - 저가) / (고가 - 저가)
1·3·5·10·20일 수익률
5·10·20일 실현변동성
ATR / 종가
20일 최고가 대비 위치
20일 최저가 대비 위치
```

## 4.2 거래량·거래대금

```text
거래량 / 20일 평균 거래량
거래대금 / 20일 평균 거래대금
거래대금 / 시가총액
당일 후보군 내 거래대금 백분위
시장 전체 대비 거래대금 백분위
```

## 4.3 수급

순매수 금액 그대로보다 다음 비율이 안정적입니다.

```text
기관 순매수 / 거래대금
외국인 순매수 / 거래대금
프로그램 순매수 / 거래대금
기관+외국인 합산 / 거래대금
각 수급 변수의 당일 후보군 내 percentile rank
```

## 4.4 상대강도

```text
종목 등락률 - KOSPI 등락률
종목 등락률 - KOSDAQ 등락률
종목 등락률 - 섹터 평균 등락률
섹터 등락률 - 시장 등락률
테마 내 수익률 순위
```

## 4.5 시장 상태

```text
KOSPI/KOSDAQ 추세
V-KOSPI/V-KOSDAQ 수준과 변화율
시장 상승 종목 비율
시장 수익률 분산
후보 종목 수
후보 평균 거래대금
대형주/소형주 상대수익률
```

시장 상태에 따라 산업·시장 변수의 예측 중요도가 변할 수 있다는 연구 결과도 있어, 개별 종목 피처와 시장 regime 피처를 함께 넣는 것이 합리적이다. ([arXiv][7])

---

## 4.6 당일 횡단면 정규화

각 날짜별로 다음 변환을 추가하십시오.

```python
feature_rank = feature.rank(pct=True)
feature_z = clip((feature - daily_median) / daily_MAD, -5, 5)
```

특히 다음 변수는 로그 변환 후 사용하십시오.

```text
log(시가총액)
log(거래대금)
log(거래량)
```

원본값과 횡단면 percentile을 둘 다 보유하는 것이 좋습니다.

---

# 5. 최종 점수와 비중 등급

## 5.1 종합 Utility Score

예시 구조입니다.

[
U_i =
q50_i
-\lambda \max(0,-q10_i)
-\gamma(q90_i-q10_i)
-cost_i
]

여기에 다음을 결합합니다.

```text
rank_score       : 랭킹 모델 점수
expected_return  : q50
downside_risk    : q10
uncertainty      : q90 - q10
p_good           : 목표수익 달성 확률
p_bad            : 큰 손실 확률
```

최종 score 예:

[
Score_i =
w_1 Rank_i+
w_2 q50_i+
w_3 P(good)_i-
w_4 Downside_i-
w_5 Uncertainty_i
]

`w1~w5`, `λ`, `γ`는 백테스트 전체에서 직접 고르는 것이 아니라 **inner walk-forward OOF 데이터에서만** 결정해야 합니다.

---

## 5.2 확률 보정

트리 모델의 `0.7`이라는 확률이 실제 승률 70%를 의미하지 않을 수 있으므로 보정이 필요합니다.

추천 순서:

1. Walk-forward OOF 확률 생성
2. OOF 예측과 실제 결과로 calibration 모델 학습
3. Sigmoid calibration 우선
4. 표본이 충분하면 isotonic도 비교
5. Brier score와 reliability curve 평가

Scikit-learn의 calibration 문서도 모델 학습에 사용한 데이터와 calibration 데이터를 분리해야 편향을 피할 수 있다고 설명한다. ([Scikit-learn][8])

---

## 5.3 등급 방식

고정 확률 기준보다 OOF 분포에 기반한 기준이 안전합니다.

| 등급   | 조건 예시                                    | 비중 척도 |
| ------ | -------------------------------------------- | --------: |
| Strong | 종합점수 상위 10%, 높은 보정확률, 양호한 q10 |       1.5 |
| Good   | 상위 25%, 기대수익 양수, 위험 허용           |       1.0 |
| Weak   | 기대수익 양수지만 불확실성 큼                |       0.5 |
| Pass   | 기대수익 또는 하방조건 미충족                |         0 |

`1.5 / 1.0 / 0.5`는 금액이 아니라 상대 배수입니다.

실제 비중은 다음처럼 계산할 수 있습니다.

[
Position_i =
BaseAmount
\times GradeMultiplier_i
\times \frac{TargetVol}{PredictedVol_i}
]

이후 다음 한도를 적용합니다.

```text
종목별 최대 비중
섹터별 최대 비중
하루 전체 신규 투자 한도
거래대금 대비 주문 한도
동시 보유 종목 수
일일 손실 한도
```

초기에는 Kelly Criterion보다 이 방식이 안전합니다. Kelly는 확률과 기대수익 추정 오차에 매우 민감하므로 충분한 실전 calibration 이후에만 축소형으로 검토하는 편이 적절합니다.

---

# 6. 검증 방식

## 6.1 무작위 K-fold 금지

다음 방식은 사용하면 안 됩니다.

```python
train_test_split(shuffle=True)
KFold(shuffle=True)
StratifiedKFold(shuffle=True)
```

시계열에서는 미래 데이터를 이용해 과거를 평가하게 될 수 있으므로 시간순 분리가 필요하다. Scikit-learn의 `TimeSeriesSplit`도 미래 표본으로 학습해 과거를 평가하는 문제를 피하기 위한 시간순 분할을 제공한다. ([Scikit-learn][9])

---

## 6.2 권장 데이터 분할

2025년을 실제 최종 평가에 사용한 적이 없다고 가정하면 다음 구조가 적절합니다.

```text
개발 구간: 2016~2024
최종 테스트: 2025
```

개발 구간 내부:

```text
Fold 1: Train 2016~2018 → Validation 2019
Fold 2: Train 2016~2019 → Validation 2020
Fold 3: Train 2016~2020 → Validation 2021
Fold 4: Train 2016~2021 → Validation 2022
Fold 5: Train 2016~2022 → Validation 2023
Fold 6: Train 2016~2023 → Validation 2024
```

이 구조에서:

* 같은 날짜 후보들은 전부 같은 fold에 배치
* 날짜 내부에서 일부 종목만 train으로 보내지 않음
* 각 fold의 예측값을 합쳐 OOF 데이터 생성
* 하이퍼파라미터와 등급 기준은 OOF에서 결정
* 최종적으로 2016~2024로 재학습
* 2025는 한 번만 평가

이미 2025년 결과를 보면서 피처·규칙·모델을 바꿨다면 2025년은 진정한 holdout이 아닙니다. 이 경우 2026년 paper trading 또는 live 기록이 최종 검증 구간이 됩니다.

---

## 6.3 Purging과 Gap

타깃 산출 기간이 검증 구간과 겹치면 경계에서 누수가 발생합니다.

예:

```text
D일 종가 매수
D+3일 종가 매도
```

이 경우 validation 시작 직전 최소 3거래일의 train 표본은 제거해야 합니다.

일반적으로:

```text
gap >= 최대 보유기간 H
```

그리고 train 표본의 `[매수시점, 매도시점]`이 validation 표본의 결과 구간과 겹치면 제거합니다.

여러 전략과 하이퍼파라미터를 반복 테스트할수록 우연히 좋은 백테스트를 선택할 가능성이 증가한다. PBO와 Deflated Sharpe Ratio는 각각 백테스트 과최적화와 반복 선택으로 부풀려진 Sharpe를 평가하기 위해 제안된 방법이다. ([SSRN][10])

---

# 7. 평가 지표

## 7.1 모델 지표

### 랭킹

```text
NDCG@1
NDCG@3
Precision@1
Precision@3
날짜별 Spearman Rank IC
기존 선정 순위 대비 top-k uplift
```

### 회귀

```text
MAE
Huber loss
Rank correlation
Quantile coverage
예측구간 평균 폭
```

### 확률

```text
Brier score
Log loss
Calibration curve
Expected Calibration Error
```

단순 accuracy나 ROC-AUC만으로 투자 모델을 선택하면 안 됩니다.

---

## 7.2 투자 성과 지표

반드시 비용 차감 후 계산하십시오.

```text
Top-1 평균·중앙 수익률
Top-3 동일가중 수익률
승률
평균 이익 / 평균 손실
Profit factor
누적수익률
연환산 변동성
Sharpe / Sortino
최대 낙폭
월별 최악 수익률
거래 회전율
연속 손실 횟수
거래 불가능·미체결 비율
```

그리고 다음 집단별로 분리합니다.

```text
KOSPI / KOSDAQ
상승장 / 하락장
고변동성 / 저변동성
시가총액 구간
거래대금 구간
후보 종목 수
섹터·테마
연도
```

전체 기간 수익이 좋아도 1~2개 특정 연도나 특정 섹터에 성과가 집중되어 있다면 신뢰하기 어렵습니다.

---

# 8. 권장 개발 순서

## 1단계: 데이터 감사

* 모든 후보 종목 존재 여부
* 데이터 확정 시각
* 실제 주문 가능 시각
* 수정주가·액면분할·상장폐지 처리
* 거래비용·미체결 처리
* 차트분석 작성 시점 확인

## 2단계: 단순 베이스라인

```text
기존 선정 순위
Ridge
Logistic Regression
LightGBM Regression
```

## 3단계: 랭킹 모델

```text
LGBMRanker
CatBoostRanker
날짜별 relevance label
NDCG@1·3
```

## 4단계: 위험 추정

```text
q10 / q50 / q90
p_good / p_bad
확률 calibration
```

## 5단계: 비중 로직

```text
Strong / Good / Weak / Pass
변동성 역가중
종목·섹터·유동성 한도
```

## 6단계: 최종 검증

```text
2025 holdout
또는 2026 사전등록 paper trading
```

---

# 최종 추천 로직

```text
[당일 후보군 생성]
        ↓
[주문시점 기준 피처 동결]
        ↓
[LightGBM Ranker로 후보 순위 산출]
        ↓
[Quantile 모델로 q10/q50/q90 예측]
        ↓
[Classifier로 목표수익 달성 확률 예측]
        ↓
[OOF 기반 확률 보정]
        ↓
[수익·하방위험·불확실성 종합점수]
        ↓
[Strong / Good / Weak / Pass]
        ↓
[변동성·유동성·섹터 한도 적용]
        ↓
[실제 결과 저장 및 정기 재학습]
```

이 데이터 규모에서는 **복잡한 딥러닝보다 데이터 시점 정합성, 날짜 단위 랭킹, purged walk-forward, OOF 기반 비중 결정**이 성능과 신뢰도를 좌우할 가능성이 높습니다. 가장 현실적인 첫 버전은 `LGBMRanker + LightGBM Quantile + calibrated Logistic/GBDT` 조합입니다.

[1]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html?utm_source=chatgpt.com
[2]: https://arxiv.org/abs/2002.06975?utm_source=chatgpt.com
[3]: https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic?utm_source=chatgpt.com
[4]: https://academic.oup.com/rfs/article/33/5/2223/5758276?utm_source=chatgpt.com
[5]: https://catboost.ai/docs/en/concepts/loss-functions-ranking?utm_source=chatgpt.com
[6]: https://arxiv.org/abs/1905.03222?utm_source=chatgpt.com
[7]: https://arxiv.org/abs/2003.02515?utm_source=chatgpt.com
[8]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com
[9]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?utm_source=chatgpt.com
[10]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253&utm_source=chatgpt.com
