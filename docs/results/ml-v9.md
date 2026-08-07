# ML v9 Algorithm-Family Prequential Ensemble Report

Date: 2026-08-06  
Execution: real-data `prequential_outer_oof`, five-fold purged walk-forward  
Cutoff: 2025-12-30  
Feature/panel: `close_morning61 + scenario_action`  
Policy: `always_buy_top1`, one-date purge, 252-date warm-up, 0.20% round-trip cost

## Reproduction

| Item | Value |
| --- | ---: |
| Raw rows | 33,935 |
| Panel rows | 33,678 |
| Trade dates | 2,451 |
| Numeric features | 61 |
| Outer folds | 5 |
| Resolved expert workers | 3 |
| Return-expert fold fits | 20 |
| Inner evaluator invocations | 0 |
| Risk OOF invocations | 1 |
| Wall time | 221.182 s |
| Expert OOF time | 197.434 s |
| Risk OOF time | 12.756 s |
| Selection time | 1.026 s |

The prequential selector uses only prior outer OOF folds. Its source folds were
`[]`, `[0]`, `[0,1]`, `[0,1,2]`, and `[0,1,2,3]`; no fold used its own or a
future validation fold.

## Aggregate OOF result

| Metric | LightGBM baseline | Prequential candidate | Delta |
| --- | ---: | ---: | ---: |
| Scheduled mean return | 1.5316% | 1.5378% | +0.0062 pp |
| Scheduled win rate | 63.73% | 63.63% | -0.10 pp |
| Profit factor | 2.740 | 2.751 | +0.011 |
| Scheduled Sharpe | 6.044 | 6.090 | +0.047 |
| Compounded MDD | 33.94% | 29.66% | -4.29 pp |
| Buy / scheduled dates | 2,035 / 2,040 | 2,035 / 2,040 | unchanged |

The strict research promotion gate passed: the candidate had a higher mean,
lower MDD, positive net mean, PF above one, and at least one non-baseline
recipe. `build_research_bundle=False` was used, so no artifact was created and
the production bundle was not changed.

## Fold selection

| Fold | Source folds | Selected recipe | Candidate mean | Candidate MDD |
| ---: | --- | --- | ---: | ---: |
| 0 | — | `lgb_only` | 1.5962% | 15.08% |
| 1 | 0 | `lgb_catboost_equal` | 2.1038% | 17.98% |
| 2 | 0,1 | `all_four_equal` | 1.3924% | 19.00% |
| 3 | 0,1,2 | `lgb_catboost_equal` | 1.2309% | 19.58% |
| 4 | 0,1,2,3 | `lgb_catboost_equal` | 1.3659% | 29.66% |

## Standalone and fixed-recipe diagnostics

CatBoost was the strongest standalone challenger on drawdown (29.01%) and the
`lgb_catboost_equal` recipe was selected in three of five folds. The all-four
recipe was selected once. XGBoost improved standalone drawdown but lowered
mean return in the fixed blend; Random Forest was useful for diversification
but was not selected as a standalone recipe.

The fixed all-OOF recipe diagnostics were:

| Recipe | Mean | PF | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: |
| `lgb_only` | 1.3197% | 2.627 | 5.423 | 33.94% |
| `lgb_xgb_equal` | 1.2851% | 2.630 | 5.414 | 27.34% |
| `lgb_catboost_equal` | 1.3614% | 2.679 | 5.554 | 29.66% |
| `lgb_random_forest_equal` | 1.3388% | 2.643 | 5.480 | 27.11% |
| `lgb_xgb_catboost_equal` | 1.3117% | 2.653 | 5.465 | 24.38% |
| `all_four_equal` | 1.3164% | 2.688 | 5.547 | 24.47% |

These standalone/recipe diagnostics use the full outer OOF series and are
diagnostic only; promotion is based on the fold-selected candidate series in
the aggregate table.

## Verdict

The optimized implementation completed the intended five-fold experiment in
221 seconds with 20 return fits and no nested refits. It improved the causal
candidate's aggregate mean, Sharpe, PF, and MDD versus the LightGBM baseline.
The result is research-positive, not production-approved: capture-time
provenance, execution slippage, and untouched paper/live OOS validation remain
required before deployment.
