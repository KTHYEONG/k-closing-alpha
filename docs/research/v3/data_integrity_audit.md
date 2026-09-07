# Data Integrity & Survivorship Audit v3

- **일자**: 2026-09-07
- **감사 대상**: `data/history/price_history.parquet`, `data/history/archive.parquet`, `data/history/intraday/1m/regular`
- **핵심 목표**: 생존 편향(Survivorship Bias), 결측치 보정의 무결성, 거래정지(Suspension) 처리, 기업공시/액면분할 등의 전방 무결성 평가.

---

## 1. 생존 편향 (Survivorship Bias) 감사

| 항목 | 실측 관측치 | 시장 실질치 (KRX 공시 기준) | 편향 및 왜곡 평가 |
|---|---|---|---|
| **전체 고유 종목수** | 2,518개 | 2,500~2,700개 (연도별 변동) | 상장 종목 풀 유지 |
| **10년간 상장폐지 종목수** | **39개** | **400개 이상** (2016~2026 누적 상폐) | **심각한 생존 편향 확인 (Survivorship Bias Present)** |
| **원인 분석** | `price_history.parquet`은 현존 상장종목 위주로 소급 수집됨 | 부도/합병/상폐된 부실 한계기업 데이터가 과거에서 대거 누락됨 | 수익률 상방 왜곡 및 MDD/CVaR 하방 왜곡 가능성 |
| **판정 영향** | `survivorship_free = False` | **`Gate 11 FAIL`** | **`PRODUCTION_VALIDATION_BLOCKED_BY_SURVIVORSHIP_BIAS` 선언 의무** |

### 생존 편향 영향 평가
1. 상장폐지된 종목이 데이터셋에서 빠져있으므로, 과거 2~10% 상승 종목 중 상폐로 이어진 극단적 꼬리 위험(-100% 손실)이 과소평가되었을 가능성이 있다.
2. 다만 K-Closing Alpha 전략은 $D+1$ 시가(09:00)에 즉시 전량 청산하므로, 상장폐지 사유 발생(장 마감 후 공시) 시 익일 거래정지에 걸리는 빈도가 관건이다.
3. 실측 결과 $U0$ 후보 중 $D+1$ 시가 거래 불가 종목은 전체 164,550건 중 **88건(0.054%)**에 불과하여, 초단기 오버나이트 전략 특성상 생존 편향이 전체 승률을 180도 뒤집을 수준은 아니나 실전 승격은 완전히 차단된다.

---

## 2. 결측치 (Missing Values) 전수 조사 및 처리 규약

| 필드명 | 결측 구간 | 결측률 | 기존 잘못된 처리 | v3 교정 처리 (Zero-Leakage Fail-Closed) |
|---|---|---:|---|---|
| **`trade_value_100m`** | 2025-01 ~ 2026-09 | ~38% | 없음 / 단순 결측 방치 | `close * volume / 1e8`로 거래대금 수학적 복원 |
| **`market_cap_100m`** | 2025-01 ~ 2026-09 | ~35% | `fillna(500.0)` 강제 편입 | **종목별 직전값 `ffill()`만 허용**, 그래도 NaN이면 `UNKNOWN` 처리 후 Fail-Closed 유니버스 제외 |
| **`d1_open` 결측/정지** | 전 구간 | 0.054% (88건) | `valid_target`으로 사전 삭제 | **사전 필터링 엄금**, 모델이 Top1 선택 시 `EXIT_SUSPENDED` 기록 후 거래 재개 시가로 캐리 |

---

## 3. 거래정지 (Suspension) 전방 경로 감사

### 3.1 거래정지 발생 시 처리 원칙
1. 과거 연구(`v2` 포함)에서는 `valid_target = target.notna() & d1_tradable`을 사용하여 $D+1$에 거래정지될 종목을 사전에 후보군에서 제외하는 **미래 정보 누수(Future Tradability Leakage)**를 범했다.
2. `v3`에서는 검증일 $T$의 후보군 생성 시 미래 $D+1$ 정보를 일체 보지 않는다.
3. 모델이 고른 Top-1 종목이 $D+1$에 거래정지(`volume == 0` 또는 `open <= 0` 또는 결측)된 경우:
   - 해당 거래를 절대로 표본에서 드롭하지 않는다.
   - 최초 거래 재개일 시가(`first_subsequent_tradable_open`)에 체결된 것으로 간주한다.
   - 자본 잠김 일수(`holding_days`, `capital_lock_days`)를 계산하여 포트폴리오 회전율과 현금 흐름에 반영한다.
   - 영구 미재개(상장폐지) 시 `UNRESOLVED_EXIT`로 플래그하고 보수적 민감도(-100%)를 적용한다.

---

## 4. 데이터 커버리지 요약 (Data Coverage Summary)

| 데이터 레이어 | 대상 기간 | 거래일수 | 행 수 | PIT 신뢰 수준 | 용도 |
|---|---|---:|---:|:---:|---|
| **Full Historical Price** | 2016-01-04 ~ 2026-09-04 | 2,618일 | 5,454,300 | Medium (EOD Proxy) | 10년 장기 Walk-Forward OOF 백테스트 |
| **Intraday 1-Minute Panel** | 2025-09-04 ~ 2026-09-04 | 243일 | 486개 파일 | **High (Exact 15:19)** | 15:20 드리프트 및 PA 체결 실측 |
| **Realtime Serving Archive** | 2026-08-04 ~ 2026-09-07 | 24일 | 538건 | **High (Exact 15:18)** | 호가창 스냅샷 및 실시간 정합성 검증 |
| **Condition History Cleaned** | 2025-12-29 ~ 2026-08-04 | 144일 | 5,270건 | High | HTS 조건검색 재현성 검증 |
