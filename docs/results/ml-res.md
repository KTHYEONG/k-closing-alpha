# ML 재학습 결과

## 1. 실행 요약

- 실행일: 2026-08-09 (Asia/Seoul)
- 학습 cutoff: 2026-08-03
- 목적: 실전 투입 전용 모델을 교체하지 않고, 최신 이력으로 재학습한 후보 번들의 품질과 정책 지표를 기록
- 결과: 학습 및 후보 번들 저장 성공
- 운영 반영: 미반영. 기존 `artifacts/models/sizing_pipeline_bundle.joblib`에는 쓰기 작업을 하지 않음

이번 실행은 `legacy.ml_research.training.retrain_bundle.train_and_save_real_model_bundle`의 기본 검증 경로를 사용했다. 이 함수는 purged OOF 정책을 먼저 보정한 뒤 전체 이력 최종 모델을 학습하고, 버전이 붙은 후보 디렉터리에만 저장한다.

## 2. 재현 명령

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u -c '
import json
from legacy.ml_research.training.retrain_bundle import train_and_save_real_model_bundle

bundle = train_and_save_real_model_bundle(
    export_dir="artifacts/models/research/ml-res-2026-08-09"
)
print(json.dumps({
    "feature_set": bundle["feature_set"],
    "panel_mode": bundle["panel_mode"],
    "training_cutoff": bundle["training_cutoff"],
    "feature_count": len(bundle["feature_cols"]),
    "policy_metadata": bundle["policy_metadata"],
}, ensure_ascii=False, default=str))
'
```

학습 함수의 고정 설정은 `n_splits=5`, `purge_gap=1`이다. `close_morning61` + `scenario_action` 조합에서는 `close-morning-reranker-v1` 점수 설정을 사용한다.

## 3. 입력 데이터

| 데이터 | 경로 | 관측치/범위 |
|---|---|---|
| 매매 로그 | `data/parquet/trade_log.parquet` | 33,934행, 2,488종목, 2016-01-04~2026-08-03 |
| 테마 정보 | `data/parquet/theme.parquet` | 학습 시 존재하면 조인 |
| 가격 이력 | `data/history/price_history.parquet` | 피처 생성에 사용되는 이력 저장소 |

원본 매매 로그의 `(수익률, %)` 범위는 -31.03%~33.64%, 중앙값은 0.00%였다. 학습 라벨은 `decimal_net` 단위이며 왕복 거래비용 0.20%를 차감한 순수익 기준으로 `target_good=+1%`, `target_bad=-2%` 임계값을 사용한다.

## 4. 모델 및 피처 구성

- feature set: `close_morning61`
- panel mode: `scenario_action`
- 수치 피처: 61개 (범주형 문자열 피처는 LightGBM 입력에서 제외)
- OOF/일별 점수 컬럼: `decision_score`
- 점수 설정: `rank_weight=1.0`, `p_good_weight=0.5`, version `close-morning-reranker-v1`
- 정책: `always_buy_top1`, policy version `ml-single-stock-v1`
- 정책 보정 cutoff: `2026-08-03 00:00:00`

피처 목록:

```text
change_rate, selection_rank, inst_net_buy, foreign_net_buy, prog_net_buy,
volume_power, total_candidate_count, kospi_change, kosdaq_change, v_kospi,
v_kosdaq, scenario_is_sangtta, scenario_is_120_breakout,
scenario_is_volume_surge, scenario_is_new_high, scenario_is_near_new_high,
scenario_is_limitup_next_day, scenario_is_rising_bearish, scenario_other,
scenario_count_for_stock_date, is_multi_scenario_stock_date,
has_sangtta_for_stock_date, turnover, inst_density, foreign_density,
major_density, prog_dominance, rank_ratio, relative_change_kospi,
relative_change_kosdaq, sector_relative_change, v_kospi_change,
v_kosdaq_change, log_market_cap_100m, log_trade_value_100m, log_volume,
log_avg_trade_value, trade_value_pct_rank, inst_net_buy_pct_rank,
foreign_net_buy_pct_rank, change_rate_pct_rank, major_density_pct_rank,
prog_dominance_pct_rank, gap_ratio_pct_rank, turnover_pct_rank, change_rate_z,
major_density_z, prog_dominance_z, turnover_z, inst_density_z, close_position,
body_ratio, upper_shadow_ratio, intraday_range, buy_price_change_rate,
gap_ratio, relative_change_rate, buy_price_change_rate_z, gap_ratio_z,
relative_flow_strength
```

## 5. OOF 정책 평가

| 지표 | 결과 |
|---|---:|
| 스케줄된 날짜 | 2,155 |
| 매수 결정 | 1,903 |
| 관망(ABSTAIN) | 252 |
| 매수율 | 88.31% |
| 스케줄 기준 평균 수익률 | 1.1934% |
| 스케줄 기준 승률 | 54.15% |
| 활성 거래 평균 수익률 | 1.3514% |
| 활성 거래 승률 | 61.32% |
| Profit factor | 2.2981 |
| 스케줄 기준 Sharpe | 4.7608 |
| 진입 순서 기준 drawdown | 44.84% |

`entry_sequence_drawdown`은 진입 순서 수익률을 누적한 간이 지표이며, 청산·포지션 중첩·자본 배분을 반영한 포트폴리오 MDD가 아니다. 따라서 이 표만으로 실전 수익성이나 안전성을 확정할 수 없다.

## 6. 산출물 및 무결성

- 후보 번들: `artifacts/models/research/ml-res-2026-08-09/close_morning61_2026-08-03/sizing_pipeline_bundle.joblib`
- 후보 파일 크기: 3,701,662 bytes
- 후보 SHA-256: `8827d9f731d96644a28808bd0db41aee034a67d00df79f9b49541ac7af38`
- 현재 운영 번들 SHA-256: `69cbc2df08437bd70c506d70a13fd9b95677a5186aa7bed26352ef733f7b9d2c`

번들에는 `feature_manifest`, `calibrators`, `rank_model`, `return_model`, `single_stock_policy`, `policy_metadata`, `training_cutoff` 등이 포함되어 있다. 후보 경로는 운영 경로와 분리되어 있어 재학습 과정에서 실시간 추론 모델이 바뀌지 않는다.

## 7. 검증 상태와 다음 단계

- 학습 프로세스: exit code 0, 후보 저장 완료
- 학습 후 저장된 번들 로드/스키마 검증: 완료
- 본 저장소 리팩터링 회귀 테스트: `209 passed, 1 warning`
- 이번 결과에는 별도 미사용 기간의 OOS/페이퍼 트레이딩 결과가 없다.

운영 승격 전에는 (1) cutoff 이후 완전 미사용 기간의 OOS 평가, (2) 비용·슬리피지·체결 실패를 포함한 포트폴리오 백테스트, (3) 기존 운영 번들과의 동일 입력 shadow 비교, (4) 승인된 후보만 운영 경로로 원자적 교체하는 릴리스 절차를 추가로 통과해야 한다.

---

# ML 챔피언 튜닝 파이프라인 (src/ml/ 이관 + 근거화)

## 8. 실행 요약 (2026-09-01)

`legacy/ml_research/` 재학습 경로를 `src/ml/` 로 재작성 이관(git 추적·ruff·mypy·pytest 대상).
동시에 학습 과정을 근거 기반으로 전환: Optuna 중첩 purged walk-forward 튜닝, 최종 모델 seed 앙상블,
OOF 기반 `p_good` blend 가중치 보정, 라벨 클립/Huber δ config화, chronological 확률 보정기,
date-balanced·recency 샘플 가중 A/B, 임베고 OOS 예약 + 동일 날짜 대조군 승격 게이트.

- 입력: `data/parquet/trade_log.parquet` 33,944행 / 2,599일
- 산출물: `scratch/champion_out/close_morning61_2026-02-27/` 후보 번들 (활성 번들 불변)
- 진입점: `python -m src.ml.retrain [--tuned ...]`

## 9. 성과 (동일 2,070 OOF 날짜, paired 대조)

| 지표 | 대조군(현행 기본값) | 튜닝 후보 | 변화 |
|---|---:|---:|---:|
| 스케줄 평균수익 | 1.2028% | 1.2795% | +6.4% |
| Sharpe | 4.93 | 5.19 | +0.26 |
| Profit Factor | 2.38 | 2.50 | +0.12 |
| 스케줄 승률 | 53.96% | 54.64% | +0.68%p |
| 활성거래 평균수익 | 1.369% | 1.457% | +6.4% |
| entry-seq drawdown | 26.20% | 25.58% | -0.6%p |

승격 게이트 PASS (`cand_mean 0.012795 >= ctrl_mean 0.012028`, shared_dates 2070).

## 10. 근거화 결과

- **HPO 최적**: `num_leaves=44, learning_rate=0.011, n_estimators=750, colsample_bytree=0.81, subsample=0.72` — 라이브러리 기본값(31/0.1/100) 대비 저학습률·정규화로 과적합 억제가 이득의 주원인. inner OOF top-1 수익 0.0104→0.0157.
- **blend 가중치 `p_good_weight` → 0.0**: 그리드 {0, 0.25, 0.5, 0.75, 1.0} 에서 스케줄 수익 평탄(~1.28%), DD는 0.0에서 최저(25.6%) → 1.0에서 40.0%. 기존 하드코딩 0.5는 무효했고 DD를 악화시켰음이 데이터로 확인됨.
- **date_balanced 샘플 가중**이 `current` 대비 게이트 통과에 기여.
- **OOS 110행(2026-03-01~)** 은 HPO·보정·정책 선택 어디에도 미사용 (`assert_oos_excluded` fail-closed).

## 11. 한계 및 다음 단계

- 여전히 OOF이며 예약 OOS(약 11일, 110행)는 표본 부족.
- `entry_sequence_drawdown` 은 포지션 중첩·자본배분 미반영 간이지표.
- 개선폭(+6.4%)은 통계적으로 작음 — 현 61피처셋 신호 상한 근접. 추가 상승은 다일 히스토리 피처(별도 스펙) 필요.
- 운영 승격 전: (1) 비용·슬리피지·체결실패 포함 포트폴리오 백테스트, (2) 완전 미사용 OOS 구간 확대 평가, (3) 활성 번들 shadow 비교.
