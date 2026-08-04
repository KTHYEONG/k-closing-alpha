# 📐 [SPEC] ML 데이터 전처리 및 피처 엔지니어링 명세서 (Parquet 기반)

## 1. 개요 및 목적 (Overview & Goals)

본 명세서는 `data/parquet` 디렉터리의 `trade_log.parquet` 및 `theme.parquet` 데이터를 기반으로, `docs/plan.md`에서 정의한 랭킹(Ranking), 연속 수익률 회귀(Regression), 목표/손실 확률 분류(Classification) multi-task ML 학습을 수행하기 위한 표준화된 데이터 전처리 및 피처 엔지니어링 파이프라인을 구축하는 것을 목적으로 한다.

### 1.1 핵심 문제점 (Key Issues Addressed)
1. **열 이름 불확실성 및 특수문자 제거**: 스프레드시트 출처의 원본 컬럼명에 포함된 `( )` 괄호, 단위(`억`, `%`), 공백 등의 특수문자를 영문/한글 표준 식별자(clean column name)로 1:1 매핑 정규화.
2. **다중 타깃 (Multi-Target) 표준화**:
   - `target_rank`: 동일 `trade_date` 그룹 내 미래 순수익률의 5단계 백분위 랭킹 라벨 (0~4).
   - `target_return`: 클리핑 및 조정이 반영된 연속 실현 순수익률.
   - `target_good` / `target_bad`: 목표 수익 달성 여부 및 허용 손실 초과 여부 이진 라벨.
3. **횡단면 정규화 및 상대 피처 구축**:
   - 단일 절대값 수치(거래대금, 시가총액 등)의 한계를 극복하기 위해 당일 후보군 횡단면 백분위(`pct_rank`) 및 상대 비율 지표(수급/거래대금, 시가 갭 등) 산출.
4. **데이터 누수(Data Leakage) 차단 및 매수가격(`buy_price`) 기준 정합성**:
   - API 최종 종가(15:30 동시호가 체결가) 대신 실제 매수 결정/체결 시점(15:19~20)의 **`buy_price` (실제 매수가격)**를 기준으로 수익률 및 당일 상승폭 피처(`buy_price_change_rate`)를 산출하여 Look-ahead bias 및 동시호가 갭 슬리피지 왜곡을 전면 차단.
   - 메타데이터(`buy_price`, `sell_price`, `stock_code` 등)는 학습 피처 집합(`X`)에서 완전 격리.

---

## 2. 컬럼 매핑 스키마 (Column Schema Normalization)

| 원본 컬럼명 (Raw Column) | 정규화 컬럼명 (Clean Column) | 데이터 타입 | 설명 |
| :--- | :--- | :--- | :--- |
| `매수날짜` | `trade_date` | datetime64[ns] | 거래일 (그룹핑/query_id 기준) |
| `종목코드` | `stock_code` | string (zfill 6) | 6자리 종목코드 |
| `(시가)` | `open_price` | float64 | 매수당일 시가 |
| `(고가)` | `high_price` | float64 | 매수당일 고가 |
| `(저가)` | `low_price` | float64 | 매수당일 저가 |
| `(종가)` | `close_price` | float64 | 매수당일 종가 |
| `(전일종가)` | `prev_close_price` | float64 | 전일 종가 |
| `(시가총액, 억)` | `market_cap_100m` | float64 | 시가총액 (억원) |
| `(거래대금, 억)` | `trade_value_100m` | float64 | 거래대금 (억원) |
| `(등락률)` | `change_rate` | float64 | 당일 등락률 (%) |
| `(선정 순위)` | `selection_rank` | float64 | 초기 조건 검색 선정 순위 |
| `(기관_순매수)` | `inst_net_buy` | float64 | 기관 순매수 금액 |
| `(외국인_순매수)` | `foreign_net_buy` | float64 | 외국인 순매수 금액 |
| `(프로그램_순매수)` | `prog_net_buy` | float64 | 프로그램 순매수 금액 |
| `(체결강도)` | `volume_power` | float64 | 체결강도 (%) |
| `(시장구분)` | `market_type` | string/cat | KOSPI / KOSDAQ 구분 |
| `(총 종목 수)` | `total_candidate_count`| float64 | 당일 포착된 전체 후보 종목 수 |
| `(평균 거래대금)` | `avg_trade_value` | float64 | 20일 평균 거래대금 |
| `(kospi, %)` | `kospi_change` | float64 | 당일 코스피 등락률 (%) |
| `(kosdaq, %)` | `kosdaq_change` | float64 | 당일 코스닥 등락률 (%) |
| `v_kospi` | `v_kospi` | float64 | V-KOSPI 변동성 지수 |
| `v_kosdaq` | `v_kosdaq` | float64 | V-KOSDAQ 변동성 지수 |
| `(거래량)` | `volume` | float64 | 당일 거래량 |
| `(테마/섹터)` | `theme_sector` | string/cat | 주도 테마 / 섹터 |
| `(차트분석)` | `chart_analysis` | string/cat | 정형화된 차트 패턴 정산 결과 |
| `(매수 가격)` | `buy_price` | float64 | 실제 매수 체결가 (메타) |
| `(매도 가격)` | `sell_price` | float64 | 실제 매도 체결가 (메타) |
| `(수익률, %)` | `net_return` | float64 | 순수익률 (타겟 원본) |

---

## 3. 다중 타깃 설계 (Multi-Target Engineering Blueprint)

`docs/plan.md` 2장에 준거하여 아래 3종의 타깃 변수를 생성한다.

```python
# 1. Continuous Return Target (클리핑 처리로 오버피팅 방지)
df['target_return'] = df['net_return'].clip(-10.0, 10.0)

# 2. Ranking Target (당일 query_id 그룹 내 백분위 기준 0~4 등급)
def assign_daily_rank(group_df):
    if len(group_df) < 5:
        # 후보가 적은 경우 uniform rank 변환
        ranks = group_df['net_return'].rank(method='min', ascending=True)
        return ((ranks - 1) / len(group_df) * 5).astype(int).clip(0, 4)
    else:
        return pd.qcut(group_df['net_return'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4]).astype(int)

df['target_rank'] = df.groupby('trade_date', group_keys=False).apply(assign_daily_rank)

# 3. Probability Targets (목표수익 / 큰 손실 이진 분류)
df['target_good'] = (df['net_return'] >= 1.0).astype(int)   # 목표 1.0% 이상 달성
df['target_bad']  = (df['net_return'] <= -2.0).astype(int)  # 허용 손실 -2.0% 이하
```

---

## 4. 피처 엔지니어링 명세 (Feature Engineering Specification)

### 4.1 기본 변환 (Log & Signed Log Scaling)
- **로그 스케일링**: `market_cap_100m`, `trade_value_100m`, `volume`, `avg_trade_value` -> `np.log1p(clip(lower=0))`
- **부호 포함 로그 스케일링**: `inst_net_buy`, `foreign_net_buy`, `prog_net_buy` -> `np.sign(x) * np.log1p(np.abs(x))`

### 4.2 가격 / 수급 상대 지표 (Relative Ratios)
- `buy_price_change_rate` = `(buy_price - prev_close_price) / prev_close_price` (15:19~20 실제 매수가격 기준 당일 등락률)
- `gap_ratio` = `(open_price - prev_close_price) / prev_close_price` (시가 갭 비율)
- `intraday_return` = `(close_price - open_price) / open_price`
- `major_density` = `(inst_net_buy + foreign_net_buy) / trade_value_100m` (메이저 수급 밀도)
- `prog_dominance` = `prog_net_buy / trade_value_100m` (프로그램 주도성)
- `rank_ratio` = `selection_rank / total_candidate_count.clip(lower=1)` (선정순위 상대위치)
- `relative_change_rate` = `buy_price_change_rate - np.where(market_type=='KOSDAQ', kosdaq_change, kospi_change)` (15:20 매수가격 기준 시장 대비 상대 등락률)

### 4.3 횡단면 백분위 순위 (Cross-Sectional Daily Percentile Ranks)
동일 `trade_date` 그룹 내부에서 percentile rank (`rank(pct=True)`)를 계산하여 시장 장세 변동에 강건한 피처 생성:
- `trade_value_pct_rank`
- `inst_net_buy_pct_rank`
- `foreign_net_buy_pct_rank`
- `change_rate_pct_rank`

---

## 5. 모듈 구조 및 파이프라인 구현 계획 (Pipeline Architecture)

- **생성 모듈**: `src/processing/preprocessor_v2.py`
- **핵심 함수**:
  - `clean_column_names(df: pd.DataFrame) -> pd.DataFrame`
  - `engineer_features(df: pd.DataFrame) -> pd.DataFrame`
  - `create_multi_targets(df: pd.DataFrame) -> pd.DataFrame`
  - `build_ml_dataset(trade_log_df: pd.DataFrame, theme_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series], list[str]]`

---

## 6. 검증 계획 (Validation & Testing)

1. **단위 테스트**: `tests/test_preprocessor_v2.py` 구축
   - 특수문자 열 이름 정규화 테스트
   - `target_rank` 5단계 정수 (0~4) 할당 검증 및 NaN 결측 없음 확인
   - `target_good` / `target_bad` 분류 라벨 0/1 분포 검증
   - `X` 피처 행렬 내 미래 정보 / 메타데이터(타깃, 매수가격 등) 유출 여부 단동 검사
2. **`uv run pytest` 수행 및 타입 체킹 (`uv run mypy src/processing/preprocessor_v2.py`)**
