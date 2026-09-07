# K-Closing Alpha Research Validation Report v3

- **실행 일자**: 2026-09-07
- **분석 기준**: 최신 `main` 브랜치 기준 전수 데이터
- **규격 문서**: [`docs/research/v3/research_spec_v3.md`](file:///home/kth/k-closing-alpha/docs/research/v3/research_spec_v3.md)
- **단일 진실 원천(SSOT)**: [`docs/research/v3/research_validation_v3_metrics.json`](file:///home/kth/k-closing-alpha/docs/research/v3/research_validation_v3_metrics.json)
- **최종 연구 판정 (Research Verdict)**: **`CONTINUE_RESEARCH`**
- **최종 프로덕션 판정 (Production Verdict)**: **`BLOCKED`**

---

## Executive Summary

본 보고서는 K-Closing Alpha 전략에 대한 3차 종합 감사 및 재검증 결과이다.
과거 보고서의 모든 인샘플 재채점 수치(+254bp, Rank IC 0.37, CAGR +345%)를 영구 폐기하고, **결정 시점(15:20) 누수 차단, D+1 미래 거래가능성 필터 제거, 달력 기반 전방 경로 및 거래정지 캐리 규칙, 5-Fold Expanding Walk-Forward OOF, 실측 호가 비용 모형** 하에서 전략의 순알파를 측정하였다.

### 핵심 13대 질문에 대한 결론적 답변

1. **Q1: 15:20 시점 2~10% 유니버스에 오버나이트 Gross 에지가 존재하는가?**
   - **YES.** U0 후보군의 일평균 Gross 수익률은 **+73.38 bp**로 명백한 오버나이트 상승 편향이 존재한다.
2. **Q2: 그 Gross 에지는 거래비용보다 큰가?**
   - **무선별 바스켓(P0)은 NO, ML 선별 Top-1은 YES.**
   - 무선별 P0는 Net -15.62 bp로 적자이나, ML Top-1은 2026년 법정거래세 20bp와 2틱 스프레드 차감 후에도 **Net +27.57 bp**로 거래비용을 압도한다.
3. **Q3: ML 랭커는 PIT OOF에서 증분 가치를 만드는가?**
   - **YES (결정적 증분 가치).** P0(-15.6bp) 대비 Top-1을 선별하여 **+43.19 bp의 순마진을 추가 창출**하며, Q5-Q1 스프레드는 +46.87 bp로 완벽히 정렬된다.
4. **Q4: Top1과 Top3 중 어떤 것이 위험조정수익률 관점에서 우월한가?**
   - **Top-3 (P2)가 압도적으로 우월하다.**
   - P1(Top-1)은 Net +27.57 bp, 샤프 1.40, CAGR +73.16%이나 MDD가 68.85%에 달함.
   - P2(Top-3)는 Net +21.57 bp, **샤프 1.76, MDD 38.43%, DSR 0.9875**로 리스크 조정 성과가 훨씬 탁월하다.
5. **Q5: U3 Hard Filter는 ML 이후에도 증분 가치가 있는가?**
   - **NO (-5.61 bp 역효과).** U3를 먼저 필터링하면 Net 수익률이 +27.57 bp에서 +21.96 bp로 하락하고 거래일의 33.6%가 결측되어 알파를 훼손한다.
6. **Q6: EV > 0 기권(Abstention) 정책은 의미 있는 개선인가?**
   - **소폭 개선에 그침 (INCONCLUSIVE).** Net 수익률은 +0.20 bp 개선되고 MDD는 6.32%p 방어되나 전략의 펀더멘털을 바꿀 수준은 아니다.
7. **Q7: AA(시장가/동시호가) 체결만으로 순알파가 살아남는가?**
   - **YES.** P1은 AA 기준 **Net +27.57 bp ($t = 4.10$)**로 패시브 체결 없이도 견고히 생존한다 (`Gate 9 PASS`).
8. **Q8: PA(패시브) 체결은 새 OOF Top-1에서 실제로 더 좋은가?**
   - **YES.** 1분봉 패널 실측 체결률 87.8% 반영 시 체결건당 **+40.48 bp**, 미체결 0수익률 합산 시도신호당 **+35.52 bp**로 AA 대비 +12.9 bp 추가 우위를 점한다.
9. **Q9: D+1 거래정지/결측 사건을 포함해도 성과가 유지되는가?**
   - **YES.** D+1 거래 불가 종목을 드롭하지 않고 거래 재개일 시가로 청산하는 현실적 캐리 룰을 적용했음에도 총 7건의 정지와 1건의 미해결 청산에 불과하여 성과 결론에 영향을 미치지 않는다.
10. **Q10: 생존 편향(Survivorship Bias)을 제거했는가?**
    - **NO.** `price_history.parquet`은 10년간 상폐 종목이 39개에 불과하여 생존 편향이 남아있다 (`Gate 11 NOT_FULLY_VALIDATED`).
11. **Q11: 기존 +32bp OOF가 PIT 교정 데이터셋에서도 재현되는가?**
    - **부분 재현 (73.4bp Gross $	o$ Net +27.57 bp).** 미래 거래가능성 필터 및 결측치 왜곡을 제거하자 기존 +32.1bp에서 +27.57bp로 소폭 조정되었으나 핵심 통계적 유의성은 완벽히 유지되었다.
12. **Q12: 연구 종합 판정은 무엇인가?**
    - **`CONTINUE_RESEARCH`** (순알파는 실재하나 생존 편향 및 15:20 소급 프록시 한계로 인해 추가 정밀화 필요).
13. **Q13: 프로덕션 판정은 무엇인가?**
    - **`BLOCKED`** (전향적 Prospective/Shadow 데이터 및 상폐 종목 복원 전까지 실계좌 라이브 배포 차단).

---

## 1. 13대 의사결정 게이트 (Decision Gates) 최종 평가표

| 게이트 | 검증 항목 | 관측치 | 통과 기준 | 결과 | 비고 |
|:---:|---|---:|:---:|:---:|---|
| **Gate 0** | PIT 데이터 유효성 | 과거 EOD 프록시 | 15:20 완전 복원 | ❌ **FAIL** | 2016-2025 틱 부재로 소급 프록시 사용 |
| **Gate 1** | 미래 후보군 누수 차단 | 0건 누수 | 누수 0건 | ✅ **PASS** | $D+1$ 거래가능성 사전 필터 전면 제거 |
| **Gate 2** | 엄격한 Walk-Forward OOF | 5-Fold, 2일 Purge | 엄격 OOF | ✅ **PASS** | 훈련-검증 완전 분리 |
| **Gate 3** | OOF 평균 순수익률 | **+27.57 bp** | > 0 bp | ✅ **PASS** | $t = 4.10$, 승률 48.5% |
| **Gate 4** | 10일 블록 부트스트랩 CI | **+14.23 bp** | > 0 bp | ✅ **PASS** | 95% CI: [14.23, 42.50] bp |
| **Gate 5** | 랭킹 정보량 및 단조성 | **Rank IC +0.1089** | > 0.0 & Q5>Q1 | ✅ **PASS** | Q5-Q1: +46.87 bp |
| **Gate 6** | 외부 Fold 안정성 | **5 / 5 양수** | $\ge 4/5$ | ✅ **PASS** | 전 폴드 일관된 흑자 (+6.9 ~ +53.2 bp) |
| **Gate 7** | 선택 편향 보정 DSR | **0.8621 (P1) / 0.9875 (P2)** | $\ge 0.95$ | ❌ **FAIL (P1)** | 350회 다중검정 예산 시 P1 미달 (P2는 통과) |
| **Gate 8** | 전략 강건성 (Multi-Model) | Ridge(+18.8bp), Shallow(+18.3bp) | 다중모델 양수 | ✅ **PASS** | 모델 구조 및 부트스트랩 블록(5/10/20일) 불변 |
| **Gate 9** | 현실적 AA 체결 생존 | **+27.57 bp** | > 0 bp | ✅ **PASS** | 패시브 미의존, 시장가 단독 생존 |
| **Gate 10**| 포트폴리오 리스크 허용치 | **MDD 68.85% (P1) / 38.43% (P2)** | MDD < 50% | ❌ **FAIL (P1)** | P1 단일종목 집중 위험 (P2는 38.4%로 양호) |
| **Gate 11**| 생존 편향 감사 | 상폐종목 39개 | 생존편향 제거 | ❌ **FAIL** | NOT_FULLY_VALIDATED |
| **Gate 12**| 전향적 섀도우 검증 | 미수행 | 전향적 검증 완료 | ❌ **FAIL** | NOT_AVAILABLE (BLOCKED) |

---

## 2. 전략 구성요소별 최종 의사결정표 (Strategy Decision Table)

| 구성 요소 (Component) | 판정 (Verdict) | 실측 근거 (Evidence) |
|---|:---:|---|
| **Closing Entry Concept** | **KEEP** | 오버나이트 Gross 편향(+73.4bp) 실재 확인 |
| **2~10% PIT Universe (U0)** | **KEEP** | 풍부한 유동성(일 62.9종목) 및 ML 랭킹 기저율 제공 |
| **>=10% Legacy Universe** | **REJECT** | 모멘텀 정점 통과(Post-Peak), 차익실현 역풍 심각 |
| **PIT ML Ranker (LightGBM)** | **KEEP** | 순위 IC +0.1089, Q5-Q1 +46.9bp 단조 분리 입증 |
| **Top-1 Selection** | **INVESTIGATE** | 수익률(+27.6bp) 높으나 MDD(68.9%) 과다 |
| **Top-3 EW Selection** | **KEEP** | 샤프 1.76, MDD 38.4%, DSR 0.9875로 최적 위험조정 |
| **U3 Hard Filter** | **REJECT** | 순수익 -5.6bp 하락, 거래일 33.6% 기회 상실 |
| **EV Abstention Gate** | **INVESTIGATE** | MDD -6.3%p 소폭 방어하나 유의성 한계 |
| **AA Execution (Closing Auction)** | **KEEP** | 패시브 체결 없이도 Net +27.6bp로 자생력 입증 |
| **PA Execution (Passive Overlay)** | **KEEP** | 실효 +35.5bp 달성, 스프레드 절감 효과 뚜렷 |
| **D+1 Open Exit** | **KEEP** | 장중 투매 회피를 위한 최적 단기 청산선 |
| **Take Profit Overlay (TP 3~7%)** | **REJECT** | 미체결 장중 청산 투매로 실측 수익 급락(+7.2bp) |
| **D+2 / D+3 Multi-Day Holding** | **REJECT** | 다일 보유 시 수익률 단조 하락(-33.8bp) |
| **Static Stop-Loss (3~5%)** | **REJECT** | 정상 노이즈 구간 강제 손절로 누적 손실 유발 |
| **Production Promotion** | **BLOCKED** | 생존 편향 미해결 및 전향적 섀도우 데이터 부재 |

---

## 3. 핵심 결론별 신뢰도 수준 (Confidence Levels)

- **2~10% 유니버스 기저율 우위**: **`HIGH`** (2,172거래일 149,749건 실측 일관성)
- **ML 횡단면 랭킹 순알파**: **`MEDIUM`** (OOF 통계적 유의성은 완벽하나 2016-2025 EOD 프록시 한계 잔존)
- **Top-3 분산 위험조정 우위**: **`HIGH`** (샤프 1.76, MDD 38.4%, DSR 0.9875)
- **U3 하드필터 무용론**: **`HIGH`** (OOF 직접 비교에서 -5.6bp 열위 실증)
- **실계좌 배포 가능성 (Production Readiness)**: **`LOW / BLOCKED`** (상폐 데이터 미확보 및 섀도우 트레이딩 미완료)
