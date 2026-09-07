# K-Closing Alpha Research Specification v3 (Frozen)

- **문서 버전**: `v3.0.0`
- **동결 일자**: 2026-09-07
- **상태**: **`FROZEN_BEFORE_EXECUTION`** (사후 파라미터/피처/게이트 조정 일체 금지)
- **규격 파일 (SSOT)**: [`docs/research/v3/research_spec_v3.json`](file:///home/kth/k-closing-alpha/docs/research/v3/research_spec_v3.json)

---

## 1. 의사결정 타임라인 및 체결 메커니즘

| 구분 | 시각 (KST) | 정의 및 제약 |
|---|---|---|
| **Candidate Snapshot** | `15:18:00` | 조건검색 및 틱/호가 스냅샷 폴링 (`collect.py`) |
| **Model Inference** | `15:18:30` | 횡단면 피처 생성 및 모델 추론 수행 |
| **Decision Freeze** | `15:20:00` | 당일 최종 매수 후보 종목 및 슬롯 확정 (이후 정보 진입 불가) |
| **Order Submission** | `15:20:00` | 동시호가 시장가/지정가 주문 접수 |
| **Primary Execution (AA)** | `15:30:00` | 장 마감 동시호가 단일가 체결 (체결가 = 당일 `Close`) |
| **Overlay Execution (PA)** | `15:19:00` | 15:19 직전 1틱 아래 지정가 제출 (체결률/역선택 실측 분리) |
| **Exit Execution** | 익일 `09:00:00` | D+1 시가(Open) 전량 시장가 청산 |
| **Exit Suspension Rule** | D+1 미체결 시 | 표본에서 제거하지 않고 최초 거래 재개 시가(`first_subsequent_tradable_open`) 청산 및 자본 락 기록 |

---

## 2. 유니버스 정의 (Point-in-Time)

1. **U0_PIT (Broad Momentum Primary)**:
   - $0.02 \le \text{change\_asof} < 0.10$ (2% 이상 10% 미만)
   - $\text{trade\_value\_asof} \ge 100\text{억원}$
   - $\text{market\_cap\_asof} \ge 500\text{억원}$ (결측 시 종목별 직전값 ffill만 허용, 임의 상수 fillna 금지, 미해결 시 제외)
   - 상한가($\text{change} \ge 29\%$ 및 고가=종가) 제외
   - 체결 가능(Close > 0, Volume > 0)
2. **U3_PIT (Conditional Quality Universe)**:
   - `U0_PIT` 만족
   - $\text{market\_cap\_asof} \ge 5000\text{억원}$
   - $\text{institution\_net\_buy\_asof} > 0$
   - $\text{market\_index\_return\_asof} > 0$ (KOSPI는 코스피, KOSDAQ은 코스닥 지수)

---

## 3. 피처 세트 및 모델 사양

- **사전 고정 14대 피처**:
  `chg_ratio`, `log_tv`, `log_mc`, `body_ratio`, `upper_shadow_ratio`, `intraday_range`, `inst_density`, `foreign_density`, `kospi_pct`, `kosdaq_pct`, `v_kospi`, `tv_rank`, `inst_rank`, `chg_rank`
- **주 모델 (Primary Model)**:
  - Algorithm: `LightGBM Regressor`
  - Objective: `huber` ($\alpha = 0.9$)
  - Hyperparameters: `n_estimators=60`, `learning_rate=0.03`, `random_state=42`, `verbosity=-1`
  - Target: $D+1\text{ Open Net AA}$ (수익률 $[-10\%, +10\%]$ 클리핑)
- **민감도 검증 모델 (Robustness Only)**:
  - `M0_Ridge`: `Ridge(alpha=1.0)`
  - `M2_ShallowLGBM`: `LGBMRegressor(objective='huber', max_depth=3, num_leaves=7, n_estimators=40, learning_rate=0.03)`

---

## 4. 교차 검증 및 누수 방지 원칙

- **CV Scheme**: 5-Fold Expanding Walk-Forward
- **Purge Gap**: 훈련 종료일과 검증 시작일 사이 2 KRX 거래일 버퍼
- **Train Set Rule**: 검증 시작 이전 날짜 중 라벨이 유효한 행만 훈련에 포함
- **Validation Set Rule**: 검증일 $T$의 유니버스 기준을 충족하는 모든 후보를 모델 평가 대상으로 삼음 (**미래 $D+1$ 거래 가능 여부나 라벨 결측을 이유로 사후 제외 금지**)
- **Data Scaling / Imputation**: 오직 Train Fold 내부에서만 fit 수행

---

## 5. 비용 모델 및 파이프라인

- **Base Case (2026_Normalized_Cost)**: 법정 거래세 20bp + 호가 2틱 스프레드 (`2 * tick / price * 10000`)
- **Stress Case**: 전 종목 일괄 46bp
- **Passive Overlay (PA)**: 법정 거래세 20bp + 호가 1틱 스프레드
- **비교 대상 파이프라인**:
  - `P0`: U0_PIT $\to$ Equal-Weight 후보 바스켓 $\to$ AA
  - `P1`: U0_PIT $\to$ ML Top1 $\to$ AA (Primary Pipeline)
  - `P2`: U0_PIT $\to$ ML Top3 EW $\to$ AA
  - `P3`: U0_PIT $\to$ ML Top1 (EV > 0) $\to$ AA
  - `P4`: U3_PIT $\to$ ML Top1 $\to$ AA
  - `P5`: U3_PIT $\to$ ML Top3 EW $\to$ AA

---

## 6. 다중검정 예산 및 13대 의사결정 게이트

- **다중검정 예산 (Multiple Testing Budget)**: 이전 78개 과제 및 하이퍼파라미터 그리드를 반영하여 기본 **350회**, 스트레스 **500회**로 사전 고정.
- **게이트 구성**:
  - `Gate 0`: PIT 유효성 (Mandatory)
  - `Gate 1`: 미래 거래가능성 누수 차단 (Mandatory)
  - `Gate 2`: 엄격한 Walk-Forward OOF (Mandatory)
  - `Gate 3`: OOF 평균 순수익 > 0
  - `Gate 4`: 10일 블록 부트스트랩 95% CI 하단 > 0
  - `Gate 5`: 순위 단조성 (Rank IC > 0 및 5분위 정렬)
  - `Gate 6`: 외부 Fold 안정성 (5개 Fold 중 4개 이상 양수)
  - `Gate 7`: 선택 편향 보정 DSR $\ge 0.95$
  - `Gate 8`: 복수 모델/부트스트랩 강건성 (동적 계산)
  - `Gate 9`: AA 체결 자생력 (AA Net > 0)
  - `Gate 10`: 포트폴리오 리스크 허용치 (MDD < 50%)
  - `Gate 11`: 생존 편향 감사 (Mandatory)
  - `Gate 12`: 전향적 검증 완료 여부 (Mandatory, historical 데이터는 항상 미완료)
