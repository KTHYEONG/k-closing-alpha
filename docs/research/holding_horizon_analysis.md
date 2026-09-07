# 보유 기간별 알파 분석 (Holding Horizon Analysis)

- **작성일자**: 2026-09-07
- **분석 대상**: 전수 유니버스 (5,454,300개 일봉), 전략 스크린 3종, ML 랭커 후보군 (33,547건 non-ceiling)
- **분석 기간**: 2016-01-04 ~ 2026-09-04 (10.6년, 2,618 거래일, 2,518개 종목)
- **비용 기준**: 편도 23bp / 왕복 46bp (KRX 법정 세금 20bp + KRX 2틱 호가 스프레드 26bp)
- **표본 분할**: DEV (< 2025-09-01, 2,370 거래일), 잠긴 OOS (>= 2025-09-01, 245 거래일)

> [!WARNING]
> **Methodology Invalidation Notice (2026-09-07)**
> 1. Level C (ML Top-1/3/5) metrics in this document were generated using final-model historical rescoring and are in-sample contaminated (`INVALID_FOR_STRATEGY_DECISION`).
> 2. The claim that `shift(-h)` "perfectly reflects suspensions and trading halts" was inaccurate: `shift(-h)` matched across calendar gaps without flagging untradable days.
> 3. See `docs/research/corrected_exit_analysis.md` and `docs/research/research_validation_v2.md` for corrected calendar-aware results.

---

## 1. 연구 목적

기존 `k-closing-alpha` 프로젝트는 **"종가 매수 → 익일 시가 매도"** 1일 오버나이트를 기본 구조로 채택해왔으나, 최근 검증에서 익일 시가 청산의 기대수익이 낮고 ML 모델이 OOS에서 부호 반전을 겪는 문제가 보고되었다.

이에 따라 **"알파가 실제로 어느 holding horizon에 존재하는가?"**를 밝히기 위해:
1. 종가 진입(Close T) 이후 D+1 Open/High/Low/Close, D+2 Open/High/Low/Close, D+3 Open/High/Low/Close, D+5 Close까지의 전방 경로(Forward Path)를 전수 추적한다.
2. D+1 Open → D+1 Close, D+1 Close → D+2 Close, D+2 Close → D+3 Close의 **점증 수익률(Incremental Return)**을 분해하여 알파가 오버나이트 갭에 있는지, 장중 연속성에 있는지, 혹은 2~3일 단기 추세에 있는지를 통계적으로 규명한다.
3. 모델 편향을 배제하기 위해 **전체 유니버스(Level A) → 스크린 풀(Level B) → ML 픽(Level C)** 순서로 Model-Free 분석부터 체계적으로 수행한다.

---

## 2. 데이터 및 샘플 정의

### 2.1 데이터 범위 및 정합성
- **가격 데이터**: `data/history/price_history.parquet` (2,518 종목, 2,618 거래일)
- **종목별 전방 시프트**: 거래정지, 휴장일, 상장폐지를 완벽히 반영한 종목별 forward shift (`shift(-1)`, `shift(-2)`, `shift(-3)`, `shift(-5)`).
- **상한가 필터**: 당일 등락률 >= 29% 및 종가 == 고가인 상한가 잠금 종목은 체결 불가로 전면 제외(`classify_ceiling_entry`).

### 2.2 분석 계층 (Hierarchy of Analysis)
- **Level A (전체 유동성 유니버스)**: 거래대금 >= 100억, 시가총액 >= 500억, 상한가 제외 (전체 590,846개 봉, 일평균 225.7개 종목).
- **Level B (기존 스크린 풀)**:
  - `operator_legacy`: 등락률 >= 10%, 거래대금 >= 100억, 시가총액 >= 500억 (일평균 12.8개 종목).
  - `band_2_15`: 등락률 2~15%, 거래대금 >= 100억, 시가총액 >= 500억 (일평균 7.3개 종목).
  - `band_5_15_highvalue`: 등락률 5~15%, 거래대금 >= 3000억, 시가총액 >= 500억 (일평균 1.3개 종목).
- **Level C (ML 랭커 픽)**: OOF/모델 채점 기준 Top-1, Top-3 Equal-Weight, Top-5 Equal-Weight.

---

## 3. 전방 경로(Forward Path) 성과 요약

### 3.1 Level A: 전체 유동성 유니버스 (Market Baseline)

전체 유동성 종목(일평균 226종목)의 무조건부(unconditional) 전방 기대수익률:

| Horizon | Gross Return | Net Return (46bp) | Median Net | Win Rate | Profit Factor | t-stat | Sharpe |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **D+1 Open** | +18.9 bp | **−27.1 bp** | −18.8 bp | 33.6% | 0.35 | −15.88 | −5.07 |
| **D+1 Close** | −7.7 bp | **−53.7 bp** | −43.8 bp | 32.8% | 0.35 | −17.88 | −5.71 |
| **D+2 Close** | −15.4 bp | **−61.4 bp** | −47.9 bp | 38.2% | 0.44 | −14.14 | −4.52 |
| **D+3 Close** | −19.4 bp | **−65.4 bp** | −49.7 bp | 39.8% | 0.50 | −12.07 | −3.86 |
| **D+5 Close** | −23.8 bp | **−69.8 bp** | −51.2 bp | 40.5% | 0.53 | −10.45 | −3.35 |

> **핵심 발견 1**: 시장 전체 유동성 풀에서 종가 매수는 **보유 기간이 늘어날수록 손실이 선형적으로 누적**된다 (Gross 기준 D1 Open +18.9bp → D1 Close -7.7bp → D2 Close -15.4bp → D3 Close -19.4bp).

---

### 3.2 Level B: 전략 스크린 비교 (Screening Edge)

#### [1] Operator Legacy 스크린 (`등락률 >= 10%`, 고변동성 모멘텀 풀)

| Horizon | Gross Return | Net Return (46bp) | Median Net | Win Rate | Profit Factor | t-stat | Sharpe | 95% Bootstrap CI |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **D+1 Open** | +16.9 bp | **−29.1 bp** | −28.7 bp | 40.2% | 0.61 | −8.58 | −2.66 | [−35.9bp, −22.1bp] |
| **D+1 Close** | −45.9 bp | **−91.9 bp** | −93.3 bp | 35.0% | 0.47 | −14.43 | −4.48 | [−103.5bp, −79.1bp] |
| **D+2 Close** | −60.8 bp | **−106.8 bp** | −107.5 bp | 38.3% | 0.53 | −12.18 | −3.78 | [−123.1bp, −90.5bp] |
| **D+3 Close** | −68.4 bp | **−114.4 bp** | −123.0 bp | 39.7% | 0.56 | −11.17 | −3.47 | [−133.6bp, −94.8bp] |

#### [2] Band 2~15% 스크린 (중간 모멘텀 풀)

| Horizon | Gross Return | Net Return (46bp) | Median Net | Win Rate | Profit Factor | t-stat | Sharpe | 95% Bootstrap CI |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **D+1 Open** | +28.6 bp | **−17.4 bp** | −24.0 bp | 42.0% | 0.76 | −4.73 | −1.48 | [−24.3bp, −10.0bp] |
| **D+1 Close** | −22.8 bp | **−68.8 bp** | −87.3 bp | 37.7% | 0.60 | −9.33 | −2.92 | [−83.9bp, −53.9bp] |
| **D+2 Close** | −32.9 bp | **−78.9 bp** | −92.5 bp | 39.9% | 0.65 | −8.06 | −2.53 | [−98.5bp, −59.6bp] |
| **D+3 Close** | −32.6 bp | **−78.6 bp** | −109.4 bp | 42.3% | 0.70 | −6.66 | −2.09 | [−103.6bp, −55.2bp] |

> **핵심 발견 2**: 
> 1. 기존 운영자 스크린(`>=10%`)은 D+1 Open Gross가 +16.9bp에 불과하며, D+1 Close에는 Gross -45.9bp, D+3 Close에는 Gross -68.4bp로 **급격한 차익실현 역풍(Reversal)**에 직면한다.
> 2. `Band 2~15%` 풀이 `Legacy >=10%` 풀보다 D+1 Open Gross가 +11.7bp 더 높다 (+28.6bp vs +16.9bp). 그러나 두 스크린 모두 거래비용(46bp) 차감 후 전 구간 음수이다.

---

### 3.3 Level C: ML 랭커 성과 및 OOS 검증

#### [1] 실측 OOF / Locked OOS Top-1 성과 (정직한 OOS 분할 기준)
*(근거: `docs/results/ml-res.md` 및 `docs/results/exit-grid-bracket-d1.md` 실측 검증)*

| 규칙 / Horizon | 전체 (2,058일) | DEV (< 2025-09) | 잠긴 OOS (2025-09+, 241일) | 판정 |
| :--- | ---: | ---: | ---: | :---: |
| **현행 (익일 시가 청산)** | +1.6 bp (t=0.2) | −1.1 bp | **+21.6 bp** (t=0.8) | 기준선 (비용후 제로 알파) |
| **익절 5% + MOC 청산** | **+15.1 bp** (t=1.1) | **+17.8 bp** (t=1.3) | **−5.6 bp** | OOS 부호 반전 (미승격) |
| **고정 D+1 종가 청산** | −42.0 bp | −38.0 bp | −65.0 bp | 기각 |
| **고정 D+2 종가 청산** | −58.0 bp | −52.0 bp | −91.0 bp | 기각 |
| **고정 D+3 종가 청산** | −71.0 bp | −66.0 bp | −108.0 bp | 기각 |

> **핵심 발견 3**: 
> ML 리랭커를 적용하더라도 **순수 고정 청산(Fixed Exit) 기준으로는 D+1 Open이 최선**이며, D+1 Close, D+2 Close, D+3 Close로 갈수록 성과가 급격히 파괴된다. OOS 구간에서 D+1 Open은 +21.6bp로 버텼으나, D+2/D+3 보유는 손실 폭만 키운다.

---

## 4. 점증 수익률 (Incremental Return) 분해

알파의 발생 구간을 분리하기 위해 각 단계별 incremental return을 계산하였다:

$$\text{D+1 Intraday} = \frac{\text{Close}_{T+1}}{\text{Open}_{T+1}} - 1$$
$$\text{D+1 to D+2} = \frac{\text{Close}_{T+2}}{\text{Close}_{T+1}} - 1$$
$$\text{D+2 to D+3} = \frac{\text{Close}_{T+3}}{\text{Close}_{T+2}} - 1$$

| 분석 그룹 | D+1 장중 변화 (Open→Close) | D+1 종가 → D+2 종가 | D+2 종가 → D+3 종가 |
| :--- | :---: | :---: | :---: |
| **Level A (유동성 유니버스)** | **−26.0 bp** (t=−10.20, 승률 41.6%) | **−7.9 bp** (t=−2.57, 승률 51.0%) | **−4.1 bp** (t=−1.34, 승률 51.1%) |
| **Level B (Legacy >=10%)** | **−61.1 bp** (t=−10.74, 승률 39.5%) | **−15.6 bp** (t=−2.61, 승률 48.2%) | **−8.9 bp** (t=−1.62, 승률 48.0%) |
| **Level B (Band 2~15%)** | **−49.6 bp** (t=−7.43, 승률 40.9%) | **−9.7 bp** (t=−1.41, 승률 47.7%) | **−0.5 bp** (t=−0.08, 승률 47.5%) |
| **Level C (Top-1 Pick)** | **−107.8 bp** (t=−7.09, 승률 37.5%) | **−13.3 bp** (t=−0.89, 승률 44.1%) | **−10.4 bp** (t=−0.74, 승률 42.0%) |

### 통계적 결론
1. **D+1 장중 연속성은 전무하며, 강력한 역전(Reversal)만 존재한다**:
   - D+1 시가 매수 후 종가 매도 시 **일평균 -26bp ~ -108bp의 치명적 손실(t = -7 ~ -10)** 발생.
   - 종가 진입자의 알파는 오직 **Close(T) → Open(T+1) 오버나이트 갭**에만 국한된다.
2. **D+2, D+3으로 갈수록 추가 알파 없이 시장 노이즈와 비용만 누적된다**:
   - D+1 종가에서 D+2 종가로 넘어갈 때 점증 수익률은 −8bp ~ −16bp로 여전히 음수이다.
   - D+2 종가에서 D+3 종가 역시 −0.5bp ~ −10bp로 추가 알파가 전혀 발생하지 않는다.

---

## 5. MFE(최대 유리 변동) 및 MAE(최대 불리 변동) 분석

진입 종가 대비 각 보유 기간 동안의 고점 및 저점 도달 폭:

| Horizon | Mean MFE | Median MFE | Mean MAE | Median MAE | MAE 10% 꼬리 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **D+1 구간** | +582 bp (+5.82%) | +485 bp | **−384 bp (−3.84%)** | **−311 bp** | **−949 bp** |
| **D+2 누적** | +795 bp (+7.95%) | +672 bp | **−598 bp (−5.98%)** | **−515 bp** | **−1,380 bp** |
| **D+3 누적** | +942 bp (+9.42%) | +801 bp | **−742 bp (−7.42%)** | **−650 bp** | **−1,650 bp** |

### MFE / MAE 인사이트
- **MFE(상방)**: D+1 장중 고점은 평균 +5.82%까지 도달하므로, 장중 익절(Take-Profit 5%)이 발동될 여지가 충분하다.
- **MAE(하방)**: 그러나 D+1 장중 저점 중간값이 **-3.11%**, 평균이 **-3.84%**에 달한다. 
- 보유 기간을 D+3으로 늘리면 MFE는 +5.8%에서 +9.4%로 3.6%p 증가하지만, MAE는 -3.8%에서 -7.4%로 3.6%p 악화된다. **상방 잠재력과 하방 리스크가 정확히 1:1로 상쇄**되므로 보유 기간 연장은 리스크 대비 보상이 없다.

---

## 6. 결론 요약

1. **현재 신호의 유일한 알파 구간은 `Close(T) → Open(T+1)` (오버나이트 갭)이다.**
2. **`D+1 Open → D+1 Close` 장중 연속성은 통계적으로 완전히 기각된다 (t = -10.74, 점증 -61bp 손실).**
3. **`D+2 / D+3` 단기 보유 연장 가설 역시 완전히 기각된다.** 점증 수익률이 D+1→D+2 (-16bp), D+2→D+3 (-9bp)로 지속 음수이며, MAE만 -3.8%에서 -7.4%로 2배 폭증한다.
4. 따라서 **"D+2/D+3 멀티데이 보유로 전략을 확장해야 한다"는 가설은 객관적 데이터에 의해 폐기(REJECT)**되어야 한다.
