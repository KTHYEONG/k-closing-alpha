# Component Architecture Specification

> **Target Audience:** 소프트웨어 아키텍트, 기술 면접관  
> **Overview:** 시스템을 구성하는 핵심 계층별 단일 책임(Single Responsibility) 및 입출력 인터페이스 정의.

---

## 1. 계층별 컴포넌트 매트릭스

| 컴포넌트 계층 | 담당 역할 | 입력 (Input) | 출력 (Output) | 주요 모듈 위치 |
| :--- | :--- | :--- | :--- | :--- |
| **Realtime Collector** | 15:20 후보 스캔, 단면 수집, 12bp 틱비용 필터링 | 전시장 실시간 시세, 10호가, 잠정수급 | `archive.parquet`, `orderbook/` | `src/daily/collect.py`<br>`src/daily/universe_scan.py` |
| **Inference Engine** | 28차원 PIT 피처 산출, 5-Seed LGBM 앙상블 추론 | 당일 15:20 스냅샷, 과거 일별 패널, 모델 번들 | `topk_decisions.parquet` (Top-3) | `src/daily/predict.py`<br>`src/serving/realtime/` |
| **Close Finalizer** | 15:30:30 단일가 체결 3중 검증 및 종가 갱신 | KIS 체결가 TR, 10호가 TR | `archive.parquet` (종가_확정=True) | `src/daily/finalize_close.py` |
| **Paper Broker** | 실주문 없이 WebSocket 체결틱 기반 가상 체결 | 추론 결과, 확정 종가, KIS 실시간 틱 | 가상 체결/포지션/원장 (`data/paper/`) | `src/daily/paper_trade.py`<br>`src/execution/paper_broker.py` |
| **ML & Research** | CPCV(8,2) 검증, 비용 스트레스 분석, 번들 빌드 | 10년치 전종목 일별 수정주가 패널 | 학습 모델 번들, 성능 보고서 | `src/ml/topk_ranker_research.py`<br>`src/ml/retrain.py` |
| **Cost Engine** | 법정거래세 스케줄, 호가단위 틱비용, 시장충격 산출 | 주가, 거래일, 시장구분(KOSPI/KOSDAQ) | 종목별 틱비용(bp), 거래세율(bp) | `src/execution/cost_model.py` |
| **Data Lake Store** | Parquet 코덱, 패널 정합성 복구, 원자적 I/O | 원천 시세 데이터, 일중 분봉/틱 | 컬럼형 파티션 스토리지 | `src/data/` |
| **Multi-Broker Gateway** | 4개 증권사 Rate Limit 제어 및 API 통신 | 증권사별 요청 파라미터 | 정규화된 시세/호가/차트 데이터 | `src/api/` |

---

## 2. 핵심 컴포넌트 세부 명세

### 1) Realtime Collector (`src/daily/collect.py`, `universe_scan.py`)
* **책임**: 15:20:00에 발화하여 키움 API로 당일 2%~10% 상승 종목을 스캔하고, KIS API로 현재가·10호가·잠정수급을 병렬 수집.
* **불변 규칙**: 15:20~15:30 외 실행 거부, 주가대별 1틱 비용 12.0bp 초과 및 상한가 제외, 단면 정상률 99% 미만 시 저장 차단.

### 2) Inference Engine (`src/daily/predict.py`, `src/serving/realtime/`)
* **책임**: 15:21:00에 스냅샷을 읽어 28차원 피처(`TOPK_FEATURE_COLS_V2`)를 산출하고, 5개 시드 LightGBM 앙상블로 Top-3 종목을 선정.
* **불변 규칙**: 결측치 발생 시 플레이스홀더를 채우지 않고 즉시 Fail-Closed 예외 발생, 적격 종목 부족 시 전액 현금 보유.

### 3) Close Finalizer (`src/daily/finalize_close.py`)
* **책임**: 15:30:30에 장마감 단일가 매매 결과를 수집하여 3중 Fail-Closed 게이트를 통과한 경우에만 공식 종가로 확정.
* **불변 규칙**: 체결 시각(>=15:30), 단일가 마감코드('3'), 현재가-호가 체결가 일치가 모두 참이어야 함.

### 4) Paper Broker (`src/daily/paper_trade.py`, `src/execution/paper_broker.py`)
* **책임**: 실계좌 주문 전송 없이 정수 수량을 계산하여 진입(15:30)하고, WebSocket 실시간 체결틱(`H0STCNT0`)을 바탕으로 익일 09:00 시초가에 기계적 전량 청산.
* **불변 규칙**: 사후 갭 필터 전면 배제 (슬리피지 포함 실체결 반영).

### 5) ML Research & Validation Engine (`src/ml/`)
* **책임**: 8개 시계열 블록을 28개 조합으로 분할하는 Combinatorial Purged CV(8,2)를 통해 시계열 과적합을 차단하고, 모델 번들 및 검증 리포트 산출.
* **불변 규칙**: 당일 단면 평균을 차감한 `date_demeaned` 타깃으로 순수 횡단면 상대 랭킹 학습.
