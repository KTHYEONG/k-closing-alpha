# ML v3 Calendar/Flow Feature Evaluation

Date: 2026-08-05  
Evaluation cutoff: 2025-12-30  
Candidate: `production_calendar_flow + scenario_action + lgb_regressor`  
Control: `base40 + scenario_action + lgb_regressor`

## 1. Executive result

The implemented calendar/flow feature set was executed on the complete
available history through 2025. It did not improve the primary Top-1 or Top-3
return objective against `base40`. It slightly improved Rank IC and
capital-weighted return, but reduced Sharpe and increased maximum drawdown.

The candidate is therefore **not promoted**. `base40` remains the stronger
default research candidate; `production_calendar_flow` should remain an
ablation candidate until a fresh, untouched evaluation period supports it.

## 2. Reproduction conditions

| Item | Value |
| --- | --- |
| Source | `data/parquet/trade_log.parquet` |
| Theme mapping | `data/parquet/theme.parquet` |
| Raw rows | 33,934 |
| Scenario-action rows before evaluation | 33,792 |
| Conflict rejects | 114 |
| Rows through cutoff | 33,678 |
| Date range used | 2016-01-04 to 2025-12-30 |
| Unique dates | 2,451 |
| OOF rows | 31,045 |
| Model | LightGBM Huber Regressor |
| CV | Purged Group Time-Series Walk-Forward |
| Folds | 5 |
| Purge gap | 1 trade-date group |
| Group key | `trade_date` |
| Target | `target_return` (`decimal_net`) |
| Cost | 0.20% round-trip cost deducted once in target construction |
| Timestamp policy | Fixed common 15:20 KST source snapshot; no row-level timestamp validation |
| Artifact side effect | None; evaluation did not overwrite a model bundle |

Both models used the same rows, scenario-action resolver, target, folds, model
type, and LightGBM random state. The only changed input was the feature set.

## 3. Feature set

`base40` produced 51 numeric model features after excluding categorical fields.
`production_calendar_flow` produced 60 numeric features by adding exactly nine:

- `weekday_is_monday`, `weekday_is_tuesday`, `weekday_is_wednesday`,
  `weekday_is_thursday`, `weekday_is_friday`
- `flow_consensus`
- `flow_alignment_direction`
- `flow_turnover`
- `friday_selection_rank_pct`

The weekday columns are one-hot indicators. `flow_consensus` is the sum of the
signs of institutional, foreign, and program flow densities. The directional
alignment feature is the signed net flow divided by the sum of absolute flow
components, with zero output for a zero denominator. `flow_turnover` preserves
the existing clipped interaction formula. `friday_selection_rank_pct` is the
selection-rank percentile interaction that is zero outside Friday.

## 4. OOF metrics

### 4.1 Primary model comparison

Returns below are decimal net returns shown as percentages.

| Metric | `production_calendar_flow` | `base40` | Difference (candidate - base40) |
| --- | ---: | ---: | ---: |
| NDCG@1 | 0.4593 | 0.4663 | -0.0070 |
| NDCG@3 | 0.4709 | 0.4763 | -0.0053 |
| Rank IC | 0.1331 | 0.1320 | +0.0012 |
| Top-1 mean net return | **1.1097%** | **1.1393%** | -0.0295%p |
| Top-3 mean net return | 0.5855% | 0.6228% | -0.0372%p |
| Win rate | 59.12% | 58.53% | +0.59%p |
| Profit factor | 2.079 | 2.112 | -0.032 |
| Sharpe | 4.369 | 4.473 | -0.104 |
| Capital-weighted return | 0.2594% | 0.2556% | +0.0038%p |
| Top-1 turnover | 100.0% | 100.0% | 0.0%p |
| Maximum drawdown | 40.46% | 40.23% | +0.23%p |

The selection-rank baseline on the same OOF dates was:

| Metric | Selection-rank baseline |
| --- | ---: |
| Top-1 mean net return | 0.4939% |
| Top-3 mean net return | 0.2229% |
| Win rate | 54.46% |
| Profit factor | 1.433 |
| Sharpe | 2.168 |
| Capital-weighted return | 0.0925% |
| Maximum drawdown | 73.07% |

The candidate retains substantial uplift over the hand-ranked baseline, but
its incremental uplift over `base40` is negative on the primary return and
risk-adjusted objectives.

### 4.2 Yearly stability

| Year | Candidate Top-1 | Base40 Top-1 | Candidate Top-3 | Base40 Top-3 | Candidate Sharpe | Base40 Sharpe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 2.388% | 2.636% | 1.351% | 1.381% | 9.71 | 10.65 |
| 2018 | 1.082% | 1.252% | 0.752% | 0.815% | 5.17 | 5.69 |
| 2019 | 1.368% | 1.034% | 0.686% | 0.528% | 5.69 | 4.33 |
| 2020 | 1.738% | 1.946% | 0.926% | 1.052% | 6.24 | 7.37 |
| 2021 | 1.068% | 0.920% | 0.485% | 0.418% | 3.96 | 3.48 |
| 2022 | 0.657% | 0.928% | 0.376% | 0.417% | 2.59 | 3.73 |
| 2023 | 1.089% | 0.660% | 0.437% | 0.450% | 3.91 | 2.31 |
| 2024 | 0.690% | 1.198% | 0.316% | 0.471% | 2.76 | 4.71 |
| 2025 | 0.765% | 0.689% | 0.458% | 0.588% | 3.23 | 2.80 |

The candidate wins in 2019, 2021, 2023, and 2025, but loses in 2017, 2018,
2020, 2022, and 2024. The pattern is not a stable broad-period improvement.

## 5. Interpretation

1. The new features add a small amount of independent ordering information:
   Rank IC increases from 0.1320 to 0.1331.
2. That information does not translate into better Top-1/Top-3 economic
   selection. Top-1 return falls by 0.0295 percentage points and Top-3 falls
   by 0.0372 percentage points.
3. Capital-weighted return improves by only 0.0038 percentage points, while
   maximum drawdown is marginally worse. The change is not sufficient to claim
   a risk-adjusted improvement.
4. Top-1 turnover remains 100%. The OOF Top-1 statistic is therefore an ideal
   daily selector, not a directly executable portfolio return.
5. The 2025 candidate result is positive, but this is still an OOF result and
   not an untouched holdout. It does not override the mixed yearly behavior.

## 6. Decision

| Decision | Result |
| --- | --- |
| Promote `production_calendar_flow` over `base40` | **No** |
| Keep candidate implementation | **Yes, as explicit research feature set** |
| Current default research feature set | `base40` |
| Next experiment | Feature ablation: flow-only, weekday-only, and interaction-only with fixed folds |
| Live deployment status | **Not approved** |

Before any live promotion, run a completely untouched evaluation period with
the same 15:20 snapshot contract, observed transaction costs, explicit
portfolio sizing, and a drawdown budget. The present result supports continued
research, not production activation.

