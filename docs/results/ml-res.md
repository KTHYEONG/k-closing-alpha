# ML 재학습·개선 결과 (2026-09-06)

관련 ADR: `ADR_20260905_BUYABILITY_GATED_RERANKER`, `ADR_20260905_EXECUTION_COST_MODEL`, `ADR_20260906_CEILING_EXCLUDED_PROMOTION_POOL`

## 1. 핵심 결론

**상한가(+30%) 종가 픽을 학습/승격 풀에서 실제로 제거**(`classify_ceiling_entry` → `champion.py` dev/control_dev 필터링)한 후 처음으로 튜닝된 챔피언이 대조군을 통계적으로 유의하게 이겼다: **Δ+0.185%p/일, p=0.006** (2026-09-06 실행, 40-trial HPO, 실데이터 36,861행). 이전 "리랭커 포화"/"rankIC≈0" 결론은 상한가 오염 상태에서 측정된 것이었다 — 오염 제거 후에는 진짜 개선 여지가 있었다.

거래비용 모델은 이번 세션에 **두 번 정정**됐다: 최종적으로 매수(~15:19)·매도(~09:00+) 둘 다 연속거래로 체결됨을 운영자가 확인 → 왕복 스프레드 2틱 + 세율 20bp(2026-01-01 시행) = **실측 46.1bp**. 단, 오늘 승격된 모델의 `target_return`은 아직 **구형 20bp 상수**로 계산되어 있어 실비용보다 낙관적이다 — 다음 단계 참조.

## 2. 상한가 배제 (`ADR_20260905_BUYABILITY_GATED_RERANKER` → `ADR_20260906_CEILING_EXCLUDED_PROMOTION_POOL`)

- 정의: `close/prev_close >= 1.29 AND close >= high`. 코퍼스 8.7%(2026년 24.1%), top-1 픽의 41.7%(2026년 68.3%) 차지.
- 4번 독립 재확인: 상한가 포함 풀 top1은 비현실적 성과(Sharpe 3.14, 연 +802%) → 제외 시 현실적 성과(Sharpe 0.61, 연 +20.7%). CPCV(8,2) 28/28 만장일치로 "제외가 수치를 낮춘다" = 애초 그 알파가 체결 불가능한 픽에서 나온 허수였다는 뜻.
- **`ADR_20260906`에서 실제 배선**: `champion.py`의 `dev`/`control_dev`에서 상한가 행 제거(학습·승격 게이트 양쪽 동일 적용) + `predict.py` 실서빙에서 상한가 종목 강제 Pass/배분0.
- 등락률 순위 필터·비용 순위 필터·top-K 랭킹 결합 등 추가로 탐색한 "종목 선정 로직"은 **전부 무효화됨**(재검증 스크립트의 룩어헤드 버그로 인한 허상 성과였음, §4 참조) — 상한가 배제만 유일하게 살아남음.

## 3. 실행 결과 — 첫 승격 (`ADR_20260906`, 2026-09-06)

`train_tuned_champion_bundle` 전체 실행(HPO 40 trials, 소요 809.9초, `price_history` 신선도 정상):

| | Candidate (튜닝) | Control (기본값) |
|---|---:|---:|
| 일평균 target_return | **+0.921%** | +0.735% |
| Sharpe | **4.18** | 3.48 |
| 승률 | 52.0% | 51.1% |
| Profit Factor | **2.12** | 1.86 |

**Δ+0.185%p/일, p=0.006, CI[+0.053%,+0.312%] → `promoted: true`**. HPO 최적: `num_leaves=63, lr=0.024, n_estimators=650`(rank_ic=0.207). p_good 블렌드 가중치 0.0 재확인(계속 무효). 상한가 배제로 dev/control_dev 각 33,547행(36,861행 중 ~9% 제거).

**⚠️ 이 수치는 여전히 구형 20bp 비용 가정 위에서 계산됨** — `create_multi_targets`가 `ROUND_TRIP_COST_RATIO`(0.0020)로 비용을 차감하는데, 같은 실행의 `execution_cost` provenance는 실측 **46.1bp**(statutory 20bp+spread 26.1bp)를 보고한다. candidate가 control을 이긴다는 판정(Δ, p값)은 양쪽에 동일하게 적용된 편향이라 유효하지만, 절대 수익률(+0.92%/일)은 실비용 반영 시 더 낮아진다.

## 4. 거래비용 모델 정정 경위 (`ADR_20260905_EXECUTION_COST_MODEL`)

세 번의 시행착오:
1. **최초**: 왕복 2틱 스프레드 + 세금 가정 → 46.9bp
2. **오판**: "종가매매는 동시호가(단일가매매) 체결이라 스프레드 없음" → `round_trip_ticks=0`으로 "정정" → 20bp — **틀림**
3. **운영자 확인 후 재정정**: 매수는 15:19 **연속거래**(동시호가 시작 직전 시장가/지정가, 상따는 15:00~15:19 관찰 후 매수), 매도도 09:00 직후 **연속거래**. 둘 다 스프레드 크로싱 발생 → 원래(1번) 가정이 맞았음. 세율만 2026-01-01 시행분(코스피/코스닥 공통 0.20%, 매도시에만) 반영해 21→20bp로 소폭 갱신. **최종 실측 46.1bp.**

같은 재검증 중 별개로, 분석 스크립트 자체의 룩어헤드 버그(`attach_next_day_path`가 붙인 익일 OHLC 컬럼이 모델 피처에 실수로 포함됨) 발견 — 이 버그 상태에서 "등락률 필터+top3 랭킹" 조합이 Sharpe 1.69로 보였으나 버그 수정 후 Sharpe -0.15로 완전 무효.

## 5. 청산 타이밍 레버 (여전히 미승격)

같은 2026-09-06 실행의 `exit_policy_grid`:

| TP | 후보 평균 | 현행 평균 | Δ | p | 승격 |
|---|---:|---:|---:|---:|:--:|
| 3% | 0.518% | 0.427% | +0.091%p | 0.323 | ❌ |
| 4% | 0.585% | 0.427% | +0.158%p | 0.133 | ❌ |
| 5% | 0.583% | 0.427% | +0.156%p | 0.164 | ❌ |

이전 세션(§구버전, `ADR_20260903_ML_EXIT_POLICY_RESEARCH`)에서 통과했던 TP5%(p=0.0024)가 이번 CPCV 재평가에서는 미통과 — 표본/기간 차이로 추정, 재확인 필요. 실체결률 측정이 여전히 승격 전제조건.

## 6. 검정력 분석 — 대기로 해결 불가 (변경 없음)

일별 페어드 노이즈(2021+, n=1,352일): top1 sd=4.667%/일, rankIC sd=0.2745. top1 +25bp/일 효과 검출에 2,732일(10.8년) 필요. 진입일 분봉 미시구조 피처는 확증 불가.

## 7. 다음 단계

1. **최우선**: `create_multi_targets`의 `ROUND_TRIP_COST_RATIO`를 실측 46bp로 갱신 후 오늘 승격된 모델을 재검증 — 지금 판정은 구형 20bp 가정 위에 있음.
2. 청산일 분봉 백필 실행(`src/backfill/intraday/backfill_minute_history.py`) — KIS 보관한도 소실 진행 중.
3. `buyability_sleeves`를 실제 포지션 사이즈로 활성화(`buyability_target_notional_100m` 설정) — 이번 실행은 skipped.
4. 청산 타이밍 TP grid, 이전 세션 대비 재현성 확인(§5의 p값 차이 원인 규명).

## 8. 46bp 실측 비용 재검증 (`round_trip_cost_46bp`, 2026-09-06)

`ROUND_TRIP_COST_RATIO`를 `_STATUTORY_COST_RATIO(0.0020) + _SPREAD_COST_RATIO(0.0026) = 0.0046`으로 갱신 후 `uv run python -m src.ml.retrain --tuned` 전체 실행(HPO 40 trials, walkforward, panel restoration on, 소요 ~870초):

| | Candidate (튜닝) | Control (기본값) |
|---|---:|---:|
| 일평균 target_return | **+0.729%** | +0.404% |
| Sharpe | **3.35** | 1.88 |
| 승률 | 50.8% | 47.2% |
| Profit Factor | **1.83** | 1.40 |

**Δ+0.326%p/일, p=0.0000, CI[+0.184%,+0.473%] → `promoted: true`** (shared_dates 2,175, moving_block_bootstrap, α=0.10). 구형 20bp에서의 Δ+0.185%p(p=0.006) 대비 델타가 커졌고 승격 판정은 유지된다 — 동일 상수 이동은 양쪽에 동일 적용되므로 판정은 모델 품질이 구동한다(R3). HPO 최적: `num_leaves=54, lr=0.0292, n_estimators=300`(rank_ic=0.20207). p_good 블렌드 가중치 0.0 재확인. dev/control_dev 각 33,547행(동일 패널). 산출 번들: `artifacts/models/close_morning61_2026-09-03/sizing_pipeline_bundle.joblib` (`round_trip_cost` 0.0046 양쪽 경로 일치).

`execution_cost` provenance: statutory 20.0bp + spread 26.12bp = **total 46.12bp** (n_rows 30,999, `n_impact_measured` 0 — 인트라데이 미연결 시 fail-open, breakeven 18.32bp). 결정→동시호가 드리프트는 라벨 상수에 합산하지 않고 `auction_impact_bp` 행별 항으로 분리 유지(R6).

클래스 균형 이동(dev 패널 33,547행, `LABEL_THRESHOLDS` 0.01/-0.02 고정): 20bp 가정 시 target_good 31.48% / target_bad 26.06% → 46bp 실측 시 **target_good 28.18% / target_bad 29.93%**. 의도된 효과이며 임계값은 이동하지 않음(R5).

`exit_policy_grid`는 여전히 미승격(TP3% p=0.405 등) — 청산 레버 결론 변경 없음.
