# Point-in-Time Universe Analysis v3

- **문서 버전**: `v3.0.0`
- **단일 진실 원천(SSOT)**: [`docs/research/v3/research_validation_v3_metrics.json`](file:///home/kth/k-closing-alpha/docs/research/v3/research_validation_v3_metrics.json)

---

## 1. 2~10% 유니버스 기저율 (Base Rate) 및 유동성

1. **U0_PIT 정의**:
   - $0.02 \le \text{daily\_change} < 0.10$
   - 거래대금 $\ge 100\text{억원}$, 시가총액 $\ge 500\text{억원}$
   - 상한가 제외, 거래량 및 주가 정상
2. **후보군 규모**:
   - 2~5% 구간: 일평균 37.8개
   - 5~10% 구간: 일평균 25.0개
   - 합산 U0 후보군: 일평균 62.8개로 풍부한 횡단면 공급을 확보함.
   - 반면 10~15%(8.2개), 25~29%(0.7개)는 극단적 후보 기근 및 변동성 정점 구간임.

---

## 2. 무선별 바스켓(P0) 성과: 유니버스 자체의 순알파 부재

- **P0 (U0 Equal-Weight Basket)**:
  - 총 거래일: 2172일
  - 총 관측 신호: 149,749건
  - Gross 수익률: **+30.62 bp**
  - Net AA 수익률: **-15.62 bp** ($t = -7.52$)
  - 샤프지수: **-2.56**
  - 95% 블록 부트스트랩 CI: [-20.15 bp, -11.36 bp]
  - 포트폴리오 MDD: **97.46%**, CAGR: **-32.54%**

### 핵심 결론
1. 2~10% 유니버스는 Gross 기준 +30.62 bp의 오버나이트 상승 편향을 가지나, **거래비용(거래세 20bp + 스프레드 ~26bp)을 감안하면 Net -15.62 bp로 완전히 적자**이다.
2. 따라서 유니버스 필터링만으로는 거래 전략이 성립하지 않으며, **횡단면 랭킹(Cross-Sectional ML Ranker)에 의한 선별이 알파 창출의 필수 조건**이다.
