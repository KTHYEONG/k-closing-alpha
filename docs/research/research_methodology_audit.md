# 연구 방법론 감사 보고서 (Research Methodology Audit)

- **일자**: 2026-09-07
- **감사 대상**: `scratch/run_full_reassessment.py`, 기존 `docs/research/*.md`, 모델 아티팩트 및 파이프라인 전반
- **목적**: 기존 전략 연구 보고서에 포함된 방법론적 오류를 전수 식별·격리하고, 전략적 판단 근거로 사용할 수 있는 유효 데이터와 무효 데이터를 엄격히 분리

---

## 1. Executive Summary: 핵심 감사 결함 7선

| # | 감사 영역 | 발견된 방법론적 결함 | 영향 및 결과 | 판정 |
|---|---|---|---|:---:|
| **1** | **인샘플 재채점 (In-Sample Rescoring)** | `sizing_pipeline_bundle.joblib`(학습 종료일 2026-09-03)의 최종 랭커로 전체 히스토리컬 패널(2016~2026)을 일괄 predict하여 Top-1/3/5를 산출함. | Top-1 D+1 Open +254.7bp, Full Rank IC 0.37, Sharpe 7~8 등 비정상적 고수익 발생 (인샘플 오염). | **INVALID** |
| **2** | **의사 OOS (Pseudo-OOS)** | 번들이 2026-09-03까지 이미 학습된 상태에서 단순히 `date >= 2025-09-01`로 필터링한 구간을 OOS로 지칭함. | 2025-09 이후 데이터가 이미 훈련 세트에 포함된 상태에서 평가됨 (진짜 OOS 아님). | **INVALID** |
| **3** | **포트폴리오 CAGR 왜곡** | 일별 산술평균 수익률에 252를 곱한 단리 연환산(`mean * 252`)을 `CAGR +345.2%`로 표기함. 실제 현금/MTM/복리 미반영. | 복리 기하평균 왜곡, 자본 배분 및 슬롯 제약 왜곡. | **INVALID** |
| **4** | **유니버스 스크린 누락 오류** | `screen_band_2_15`(2~15%)를 생성하면서 99.8%가 10%+ 급등주로 구성된 `ml_training_panel`을 필터링함. | 2~10% 중간 모멘텀 종목이 사실상 전무(64행 불과), 실제로는 10~15% 구간만 평가됨. | **INVALID** |
| **5** | **단변량 스프레드 단순 합산** | 기관 순매수(+83.7bp), 코스닥 상승일(+76.6bp) 등 단변량 횡단면 스프레드를 단순 합산하여 미래 알파가 클 것으로 추정함. | 요인 간 상관관계 무시, 실제 다변량 결합 시 표본 급감 및 알파 중복 미반영. | **INVALID** |
| **6** | **전방 경로 거래정지(Suspension) 누락** | `grouped.shift(-h)` 방식으로 단순 다음 관측 봉을 D+1로 매칭. 거래정지 기간 및 시장 거래일 달력 미고려. | 거래정지 159건 누락, 8년 정지 종목(036220: 2016→2024)을 D+1로 매칭하는 오류 발생. | **CORRECTED** |
| **7** | **생존 편향 및 결측치** | `price_history.parquet` 내 10년간 상장폐지 종목이 39개에 불과(현재 상장종목 위주 소급), 2025~2026년 거래대금 결측 40% 발생. | 상장폐지 리스크 과소추정, 결측치 처리 미흡으로 유동성 풀 왜곡. | **IDENTIFIED** |

---

## 2. 모델 아티팩트 Provenance 매트릭스

기존 연구 및 배포 파이프라인에서 혼용되던 모델들을 명확히 분리·기록한다.

| Artifact ID | Artifact 경로 | 학습 기간 | 라벨 모드 | 비용 모드 | 피처 세트 | 모델 유형 | 승격 상태 | 평가 기간 | 평가 행 훈련 포함 여부 | 최종 검증 상태 |
|---|---|---|---|---|---|---|---|---|:---:|:---:|
| **A1. Production Live Bundle** | `artifacts/models/sizing_pipeline_bundle.joblib` | 2016-01-04 ~ 2026-09-03 | journaled | flat (46bp) | `close_morning61` | LGBMRanker + SeedEnsemble | promoted_live | 2016-01-04 ~ 2026-09-03 | **YES (100%)** | **IN_SAMPLE_CONTAMINATED** |
| **A2. Candidate Research v2** | `artifacts/models/research/v2_default_run/...` | 2016-01-04 ~ 2025-08-29 | mechanical | per_row | `close_morning61` | LGBMRanker + HuberRegressor | research_only (미승격) | 2025-09-01 ~ 2026-09-03 (245일) | **NO** | **OOS_SIGN_INVERTED** (Rank IC: -0.1833) |
| **A3. Legacy Journaled Model** | `legacy/ml_research/...` | 2016 ~ 2023 | journaled | flat (20bp/46bp) | legacy base | LightGBM | deprecated | 2016 ~ 2023 | **YES** | **INVALID_FOR_DECISION** |
| **A4. Broad Universe OOF Baseline** | `scratch/corrected_research_results.json` | 2016-01-04 ~ 2026-09-04 (5-fold Purged CV) | mechanical (D+1 Open net) | per_row (KRX 20bp+tick) | broad_14feats | LGBMRegressor (Huber) | research_baseline | 2016-01-04 ~ 2026-09-04 (2,171일) | **NO (Strict OOF)** | **VALID_OOF_BASELINE** |

---

## 3. 세부 결함 분석 및 실측 증거

### 3.1 최종 모델의 전수 재채점 (In-Sample Rescoring)
`scratch/run_full_reassessment.py` 라인 63, 86~107:
```python
bundle = joblib.load("artifacts/models/sizing_pipeline_bundle.joblib")
scores = rank_model.predict(x_features[feature_cols])
ml_returns["score"] = scores
top1_idx = pool_non_ceiling.groupby("trade_date")["score"].idxmax()
```
- 번들의 메타데이터 확인 결과: `training_cutoff: 2026-09-03 00:00:00`.
- 즉, 2016년부터 2026년 9월까지의 전체 데이터를 보고 학습된 최종 모델이 과거 데이터를 그대로 채점함.
- 이에 따라 산출된 Top-1 D+1 Open 수익률(+254.7bp), Full Rank IC(0.3737), TP 그리드 수익률(+276~291bp), 샤프 7.5~8.5는 전부 과적합 인샘플 오염 수치임.

### 3.2 2025-09+ 구간의 Pseudo-OOS 성격 및 재정의
- 상기 번들은 2026-09-03까지 학습되었으므로 `trade_date >= 2025-09-01` 구간 역시 인샘플이었음.
- 또한 2025-09+ 구간은 이미 exit 그리드, TP 선택, 피처 선택에 반복적으로 노출됨.
- 따라서 향후 2025-09+ 구간은 **`Historical Research Holdout` (소비된 연구 홀드아웃)**으로 재정의하며, 처녀 데이터(Virgin OOS)로 취급하지 않는다.

### 3.3 전방 경로 거래정지(Suspension) 처리 결함
기존 `src/ml/forward_path.py` 라인 100~104:
```python
grouped = ph.groupby("symbol", sort=False)
for h in clean_horizons:
    ph[f"d{h}_date"] = grouped["date"].shift(-h)
    for col in ("open", "high", "low", "close"):
        ph[f"d{h}_{col}"] = pd.to_numeric(grouped[col].shift(-h), ...)
```
- **문제점**: 거래정지나 휴장으로 인해 특정 종목의 일봉이 결측된 경우, `shift(-1)`은 다음 달력 거래일이 아닌 수일~수년 뒤 재개된 첫 봉을 가져옴.
- **실측 사례**:
  - `036220`: 2016-05-04 진입 후 2024-03-13 거래 재개 (8년간 거래정지). `shift(-1)`은 2024년 시가를 2016년의 D+1 시가로 취급함.
  - `005930` (삼성전자): 2018년 50:1 액면분할 당시 2018-04-30~2018-05-03 거래정지(volume=0, open=0). `forward_path.py`는 이를 정상 가격으로 보거나 건너뜀.
  - `ml_training_panel` 전체에서 D+1 시점에 거래 불가(결측 106건 + volume=0 정지 53건 = 총 159건) 종목이 존재함에도 무조건 체결로 처리됨.
- **수정 완료**: `src/ml/forward_path.py`에 거래소 거래 달력(`market_next_trading_date`)과 종목 관측 봉(`symbol_next_observed_bar`)을 분리하고, volume==0 및 달력 불일치 시 `suspended=True`, `tradable=False`로 처리하도록 전면 개정함 (`tests/unit/ml/test_forward_path.py` 단위 테스트 8건 통과).

### 3.4 유니버스 스크린 표본 오염 (`screen_band_2_15`)
- `scratch/run_full_reassessment.py`에서 `screen_band_2_15`를 `pool_non_ceiling`(ml_panel 기반)에서 슬라이싱함.
- 그러나 `ml_training_panel.parquet`은 과거 운영자가 등락률 10% 이상만 기록한 수기 가상매매 일지로서 36,861행 중 36,797행(99.8%)이 10%+ 종목임 (2~10%는 단 64행).
- 따라서 이전 보고서의 `Band 2~15%` 결과(-19.2bp/일, 7.29종목/일)는 사실상 `Band 10~15%`를 평가한 것이었으며, 2~10% 중간 모멘텀 풀은 전혀 분석되지 못했음.

### 3.5 생존 편향 및 포인트-인-타임 데이터 감사
- `data/history/price_history.parquet` 전수(5,454,300행, 2,518개 종목) 중 2026-09-04 이전 거래가 종료된 종목은 단 39개에 불과함.
- 지난 10년간 한국 증시에서 상장폐지된 수백 개 기업이 누락되어 있어 **명백한 생존 편향(Survivorship Bias)**이 존재함.
- 2025~2026년 구간에서 `trade_value_100m` 및 `market_cap_100m` 컬럼 결측률이 약 35~40%에 달함. `close * volume / 1e8`로 거래대금을 복원하고 시가총액을 발행주식수로 전방 보간(ffill)하지 않으면 유동성 풀이 왜곡됨.

---

## 4. 무효화된 지표 (Invalidated Metrics List)

다음 지표들은 인샘플 오염, 방법론적 결함, 비현실적 체결 가정으로 인해 **전략적 판단 근거로 사용이 영구 금지**된다 (`INVALID_FOR_STRATEGY_DECISION`).

| 무효화 지표명 | 기존 보고 수치 | 무효화 사유 | 대체 유효 지표 |
|---|---:|---|---|
| **Top-1 D+1 Open Net** | +254.7 bp / 일 | 최종 모델 인샘플 재채점 오염 | **새 OOF Baseline Top-1: +32.1 bp** (t=5.08) |
| **Take Profit 3~7% + MOC** | +276.2 ~ +290.7 bp / 일 | 인샘플 픽 + High>=TP 시 100% 체결 비현실 가정 | **OOF 실측 체결 보정 TP 5%: +7.2 bp** (Sharpe 0.21) |
| **Full / DEV Rank IC** | 0.3737 / 0.3628 | 최종 모델 인샘플 평가 오염 | **새 OOF Baseline Rank IC: +0.1096** (t=30.16) |
| **D+1 포트폴리오 CAGR** | +345.2% | 일평균 단리 연환산(`mean*252`) 왜곡 표기 | **True Discrete NAV CAGR: +93.0%** (MDD 37.3%) |
| **Screen Band 2~15% 성과** | 7.29종목, -19.2 bp | 10%+ 풀에서 슬라이싱되어 10~15%만 평가됨 | **전수 2~10% U0 풀: 63.2종목, Gross +30.6 bp** |
| **단변량 수급·지수 합산 에지** | 기관+지수 단순합산 | 다변량 교차상관 무시 | **중첩 다변량 Joint U3 Net(AA): -9.4 bp / Net(PA): +3.9 bp** |

---

## 5. 유효한 기존 연구 결과 (Preserved Valid Findings)

다음 결론들은 모델-프리(Model-Free) 또는 정직한 분할 데이터에 기반하여 검증되었으므로 유효하게 유지된다.

1. **정적 손절(Static Stop-Loss 3~5%)의 파괴성 (REJECT)**:
   - 후보군의 51.0%가 장중 -3%를 터치하는 정상 노이즈 밴드에서 바닥 손절을 강제하여 일평균 -50~-95bp의 손실 누적 (`REJECT_FOR_CURRENT_STRATEGY`).
2. **멀티데이(D+2 / D+3) 보유의 알파 소멸 (REJECT)**:
   - D1 Open → D1 Close 점증 손실(-58.9bp), D1 Close → D2 Close 점증 손실(-33.8bp)로 보유 기간 증가 시 손실 단조 증가 (`REJECT_FOR_CURRENT_RESEARCH_CYCLE`).
3. **D+1 Open 고정 청산의 상대적 우월성 (BASELINE_KEEP)**:
   - 장중 급격한 차익실현 역풍(-59bp)을 회피할 수 있는 유일한 고정 청산 기준선.
4. **패시브 진입 체결 경제성 (PA Scenario)**:
   - 1틱 지정가 진입 시 체결률 87.8%, 스프레드 13bp 절감 효과 확인.
