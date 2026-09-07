# Execution Model Validation v3 (AA vs PA)

- **문서 버전**: `v3.0.0`
- **단일 진실 원천(SSOT)**: [`docs/research/v3/research_validation_v3_metrics.json`](file:///home/kth/k-closing-alpha/docs/research/v3/research_validation_v3_metrics.json)

---

## 1. 체결 모델별 실효 비용 및 수익률

| 체결 모드 | 진입 방식 | 청산 방식 | 왕복 호가 스프레드 | 거래세 | 총 거래비용 | Top-1 실효 Net | 체결률 |
|---|---|---|---:|---:|---:|---:|:---:|
| **AA (Primary)** | 장마감 15:30 동시호가 | 익일 09:00 시가 시장가 | 2틱 (~26 bp) | 20 bp | **45.8 bp** | **+27.57 bp** | 100% |
| **PA (Passive Overlay)** | 장마감 15:19 1틱 지정가 | 익일 09:00 시가 시장가 | 1틱 (~13 bp) | 20 bp | **32.9 bp** | **+40.48 bp** | 87.8% |
| **Conservative Stress** | 보수적 스트레스 체결 | 익일 09:00 시가 시장가 | 가산 26 bp | 20 bp | **46.0 bp** | **+27.38 bp** | 100% |

---

## 2. 패시브 진입(PA) 실측 체결 분석 및 미체결 처리 규약

1. **체결률 (Fill Rate)**: 1분봉 패널 실측 기준 **87.8%**
2. **역선택 (Adverse Selection)**:
   - 체결된 신호 평균 수익률: **+40.48 bp**
   - 시장가 체결(AA) 대비 스프레드 절감 및 역선택 실측치: **+12.91 bp**
3. **미체결 신호 0수익률 반영 원칙 (Section 22 준수)**:
   - 미체결된 12.2%의 신호를 수익률 계산에서 제외하지 않고, 수익률 0%로 엄격히 합산.
   - **전체 시도 신호당 실효 수익률 (Return per Attempted Signal)**:
     $$\text{Effective Net} = 0.8776 \times 40.48 + (1 - 0.8776) \times 0.0 = \mathbf{+35.52\text{ bp}}$$

---

## 3. AA 체결 자생력 판정 (`Gate 9 PASS`)

- Primary 전략인 **P1은 패시브 체결 보너스 없이도 순수 시장가 AA 체결 하에서 일평균 +27.57 bp의 순알파를 입증함**.
- 따라서 전략은 **체결 모델 의존적(Execution-Dependent)이지 않으며, AA 단독으로도 손익분기점을 명백히 초과**함 (`Gate 9 PASS_AA_STANDALONE`).
