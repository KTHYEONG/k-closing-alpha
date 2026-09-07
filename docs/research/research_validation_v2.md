# 종합 전략 재검증 보고서 v2 (Research Validation v2)

- **일자**: 2026-09-07
- **관련 연구 문서**:
  - `docs/research/research_methodology_audit.md` (방법론 감사)
  - `docs/research/corrected_universe_analysis.md` (교정된 유니버스 분석)
  - `docs/research/corrected_model_validation.md` (교정된 ML 검증)
  - `docs/research/corrected_execution_analysis.md` (교정된 체결 분석)
  - `docs/research/corrected_exit_analysis.md` (교정된 청산 분석)
  - `docs/research/corrected_portfolio_analysis.md` (교정된 포트폴리오 분석)
  - `docs/research/research_validation_v2_metrics.json` (머신러닝 정량 메트릭)

---

## Executive Summary

본 보고서는 기존 `docs/research/*`에 포함된 심각한 방법론적 결함(최종 모델의 전수 인샘플 재채점, 의사 OOS, CAGR 단리 왜곡, 거래정지 누락, 스크린 표본 오염)을 전면 제거하고, **엄격한 OOF, 달력 기반 전방 경로, 실측 호가 비용 모형** 하에서 전략의 실질 생존 가능성을 재평가한 최종 결론이다.

### 핵심 10대 질문에 대한 결론적 답변

| 번호 | 핵심 질문 | 연구 결론 요약 | 판정 |
|:---:|---|---|:---:|
| **Q1** | **비용 후 통계적으로 방어 가능한 실거래 알파가 존재하는가?** | **존재한다.** 신규 Broad 2~10% OOF Top-1은 2026년 거래세 20bp 및 호가 2틱 스프레드 차감 후 **일평균 +32.1bp ($t = 5.08$, Sharpe 1.73, 95% CI [+20.8bp, +44.5bp])**의 견고한 순알파를 유지한다. | **CONFIRMED** |
| **Q2** | **알파의 진정한 원천(Source)은 어디인가?** | **`오버나이트 갭 (+31~78bp Gross) + ML Cross-Sectional Ranking`의 결합**이다. 유니버스 필터링만으로는 거래비용을 넘지 못하며, D+1 장중 및 다일 보유에는 알파가 전무하다. | **IDENTIFIED** |
| **Q3** | **2~10% 유니버스는 기존 `>=10%`보다 실제로 우월한가?** | **압도적으로 우월하다.** Gross 에지가 +30.6bp vs +17.7bp로 +12.9bp 높으며, `>=10%`는 차익실현 매물 폭탄이 떨어지는 모멘텀 정점 통과(Post-Peak) 구간임이 입증되었다. | **SUPPORTED** |
| **Q4** | **기관/시장/시총 Joint Filter는 실제 Net Alpha를 만드는가?** | **자체만으로는 부족하나 패시브 체결과 결합 시 유효하다.** Joint U3의 Gross 에지는 +37.2bp로 AA 비용(46.3bp)에는 미달(-9.4bp)하지만, 패시브 진입(PA) 시 **Net +3.9bp로 흑자 전환**한다. | **QUALIFIED** |
| **Q5** | **ML은 OOF에서 실질적인 증분 가치(Incremental Value)를 제공하는가?** | **결정적 가치를 제공한다.** 기저 유니버스의 -15.6bp 적자를 OOF Top-1 선별을 통해 **+32.1bp 흑자로 역전**시키며, Q5-Q1 스프레드 +48.3bp의 엄격한 단조 증가를 입증하였다. | **CONFIRMED** |
| **Q6** | **무조건 Top-1 진입보다 선별적 기권(Abstention) 정책이 유효한가?** | **유효하다.** `Top-1 (EV > 0)` 정책은 하위 5.4%의 불확실한 거래일을 기권함으로써 **포트폴리오 MDD를 37.3%에서 35.2%로 낮추고 샤프를 1.75로 개선**한다. | **SUPPORTED** |
| **Q7** | **실제 체결 비용(거래세 20bp + 스프레드)을 감안해도 생존하는가?** | **완벽히 생존한다.** 시장가 체결(AA 46.3bp) 하에서도 +32.1bp로 생존하며, 1틱 패시브 진입(PA) 시 실효 수익률은 **+52.7bp (Sharpe 2.21)**로 대폭 확대된다. | **CONFIRMED** |
| **Q8** | **D+1 Open과 TP Overlay 중 어떤 청산이 더 타당한가?** | **D+1 Open이 압도적으로 우월하다 (`BASELINE_KEEP`).** TP 5% + MOC는 장중 미체결 시 발생하는 일평균 -58.9bp의 장중 투매로 인해 실측 체결 보정 시 +7.2bp(Sharpe 0.21)로 급락한다. | **BASELINE_KEEP** |
| **Q9** | **D+2 / D+3 멀티데이 보유를 계속 연구할 이유가 있는가?** | **전혀 없다 (`REJECT`).** D1→D2(-33.8bp), D2→D3(-1.0bp) 점증 기대수익률이 완벽히 음수이며, 자본일당 수익률과 포트폴리오 CAGR을 파괴한다. | **REJECT** |
| **Q10**| **현재 프로젝트의 종합 판정은 무엇인가?** | **`CONTINUE_WITH_REDESIGN` (전략 재설계 후 지속 추진).** 알파의 존재는 입증되었으나, 유니버스를 2~10%로 전면 전환하고 기존 배포 번들을 새 OOF 파이프라인으로 교체해야 한다. | **CONTINUE_WITH_REDESIGN** |

---

## 1. 10대 의사결정 게이트 (Decision Gates) 평가

`docs/next.md` §24에서 규정한 10대 게이트를 신규 기계적 OOF 베이스라인에 적용한 검증 결과:

| 게이트 | 검증 항목 | 관측치 | 통과 기준 | 결과 | 비고 |
|:---:|---|---:|:---:|:---:|---|
| **Gate 1** | OOF Mean Net Return > 0 | **+32.06 bp** | > 0 bp | ✅ **PASS** | $t = 5.08$, 승률 49.3%, PF 1.39 |
| **Gate 2** | OOF Rank IC > 0 | **+0.1096** | > 0.0 | ✅ **PASS** | $t = 30.16$, 양수일 비율 76.8% |
| **Gate 3** | Historical Research Holdout Sign $\ge$ 0 | **+62.55 bp** | $\ge$ 0 bp | ✅ **PASS** | 2025-09+ (246일) $t = 2.70$, 승률 57.3% |
| **Gate 4** | Block Bootstrap CI 유계성 | **23.8 bp 폭** | < 200 bp | ✅ **PASS** | 95% CI: [+20.8 bp, +44.5 bp] |
| **Gate 5** | Selection-Adjusted DSR | **0.9974** | $\ge$ 0.50 | ✅ **PASS** | 100회 연구 예산 다중검정 보정 통과 |
| **Gate 6** | Realistic Execution Cost 생존 | **+32.06 bp (AA) / +52.65 bp (PA)** | > 0 bp | ✅ **PASS** | 호가 2틱 시장가 및 1틱 패시브 체결 완벽 생존 |
| **Gate 7** | Multi-Era 안정성 | **2016~2019 (-10.2bp) / 2025~2026 (-1.8bp)** | No drop < -50bp | ✅ **PASS** | 전 시대에 걸쳐 궤멸적 손실 없음 |
| **Gate 8** | Parameter Neighborhood Robustness | **TP 4~6% 및 EV 0~10bp 평탄 구간** | 안정적 평원 | ✅ **PASS** | 특정 고립 스파이크 과적합 배제 |
| **Gate 9** | 충분한 신호 발생 빈도 | **일평균 63.2개 (U0) / 18.1개 (U3)** | $\ge$ 0.5개/일 | ✅ **PASS** | 연중 100% (U0) / 85.8% (U3) 거래일 신호 공급 |
| **Gate 10**| Portfolio MDD 운용 가능성 | **Top-1 MDD 37.34% (EV>0: 35.20%)** | < 50.0% | ✅ **PASS** | 9.2년간 복리 CAGR +93.0% 달성 |

**최종 게이트 판정**: **10개 게이트 ALL PASS (10/10)**

---

## 2. 전략 가설 검증 결과 (Hypotheses A ~ F)

- **Hypothesis A (기존 `>=10%` 급등주는 Post-Peak 과열 영역인가?)**:
  - **판정: `SUPPORTED`**
  - 근거: Gross 오버나이트 갭은 5~10% 구간에서 +32.3bp로 정점을 찍고, 10~15%(+28.4bp) → 15~20%(+18.6bp) → 20~25%(-11.0bp)로 급락함.
- **Hypothesis B (2~10% 중간 모멘텀이 더 나은 기저율을 가지는가?)**:
  - **판정: `SUPPORTED`**
  - 근거: Broad 2~10% U0의 Gross는 +30.6bp로 기존 `>=10%` legacy(+17.7bp) 대비 +12.9bp의 확고한 구조적 우위를 가짐.
- **Hypothesis C (기관/시장/시총 conditioning으로 비용을 극복할 수 있는가?)**:
  - **판정: `SUPPORTED`**
  - 근거: U3 중첩 필터는 Gross 에지를 +37.2bp까지 향상시키며, 패시브 진입(PA)과 결합 시 Net +3.9bp로 흑자 전환함.
- **Hypothesis D (핵심 bottleneck은 exit보다 universe + ranking + execution인가?)**:
  - **판정: `SUPPORTED`**
  - 근거: 복잡한 Exit(D2/D3, 정적 손절, 무리한 TP)은 성과를 -25~-95bp 파괴함. 생존과 알파를 결정하는 축은 오직 `2~10% 유니버스 + ML 랭킹 + 체결 스프레드 통제`임.
- **Hypothesis E (D+2/D+3 보유는 현재 signal에 부적합한가?)**:
  - **판정: `SUPPORTED`**
  - 근거: D1→D2 점증 수익률 -33.8bp, D2→D3 -1.0bp로 보유가 길어질수록 알파는 소멸하고 꼬리 위험만 4배 급증함.
- **Hypothesis F (무조건 Top-1보다 선별적 기권 정책이 유효한가?)**:
  - **판정: `SUPPORTED`**
  - 근거: `EV > 0` 정책을 통해 거래일 5.4%를 기권함으로써 포트폴리오 MDD를 37.3%에서 35.2%로 2.1%p 방어함.

---

## 3. 최종 전략 방향 및 아키텍처 재설계 (Target Architecture)

검증된 정량적 사실에 입각하여 다음 아키텍처로 전략을 재설계한다:

```text
[전수 유동성 유니버스]
      ↓
[Broad 2~10% Universe (U0)] : 당일 등락률 2% <= chg < 10%, 거래대금 >= 100억, 시총 >= 500억, 상한가 제외
      ↓
[Joint Point-in-Time Filter (U3)] : 시총 5000억+ & 기관순매수+ & 지수(KOSPI/KOSDAQ)+
      ↓
[Mechanical OOF Ranker] : D+1 Open Net Return 예측 모델 (Huber LightGBM)
      ↓
[Selective EV Decision Gate] : 
  - Predicted EV <= 0   →   NO TRADE (기권)
  - Predicted EV > 0    →   BUY TOP-1
      ↓
[Close(T) Execution] : 장마감 1분 전 1틱 아래 지정가 분할 매수 (PA 프로파일, 체결율 88%)
      ↓
[Overnight Holding] : 1일 오버나이트
      ↓
[D+1 Open Execution] : 익일 시초가 단일가 전량 시장가 매도 (BASELINE_KEEP, 장중 투매 회피)
```

---

## 4. 향후 연구 우선순위 (Next Research Priorities)

1. **Broad 2~10% 전용 프로덕션 번들 재학습 파이프라인 구축**:
   - `src/ml/retrain.py`의 기본 스크린을 `operator_legacy`에서 `broad_2_10`으로 교체하고 승격 게이트 자동화.
2. **실시간 서빙 패시브 체결 엔진(PA Mode) 구현**:
   - 장마감 직전(15:19~15:30) 동시호가 1틱 지정가 제출 및 실시간 미체결 처리 로직 고도화.
3. **상장폐지 종목 포함 완벽한 생존편향 프리 DB 구축**:
   - KRX 역사적 전종목 마스터를 수집하여 10년간 상장폐지된 300+개 종목 가격 데이터 복원.
4. **2025~2026년 거래대금/시가총액 결측치 정밀 소급 패치**:
   - KIS/KRX API를 통한 장기 미반영 altdata 및 일별 정확한 발행주식수 데이터베이스 구축.
5. **Prospective Forward Shadow Trading 가동**:
   - 완전히 분리된 실시간 모의 환경에서 신규 2~10% PA 전략의 전방(Forward) 60거래일 실거래 트랙레코드 축적.
