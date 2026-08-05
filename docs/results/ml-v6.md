# ML v6 Close-to-Morning Reranker Execution Report

Date: 2026-08-05  
Execution: full real-data retraining and 5-fold purged group walk-forward OOF  
Candidate: `close_morning61 + scenario_action + close-morning-reranker-v1`

Returns follow the project contract: buy one stock at the close and sell the
following morning. The source is the common 15:20 KST snapshot; no timestamp
columns or row-level timestamp validation are used.

## 1. Reproduction conditions

| Item | Value |
| --- | --- |
| Raw rows | 33,934 |
| OOF dates | 2,155 |
| Model | LightGBM Huber return model + chronological `p_good` model |
| CV | 5-fold purged group walk-forward, one-date purge |
| Panel | `scenario_action` |
| Target | `target_return` decimal net, 0.20% round-trip cost deducted once |
| Training cutoff | 2026-08-03 |

## 2. OOF feature and policy comparison

| Feature/policy | Scheduled mean | Active-trade mean | Win rate | PF | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base40` rank-only | 0.858% | 0.971% | 57.38% | 1.88 | 3.56 | 44.84% |
| `snapshot49` rank-only | 1.113% | 1.261% | 60.43% | 2.16 | 4.41 | 44.84% |
| `close_morning61` rank-only | 1.147% | 1.299% | 61.74% | 2.23 | 4.58 | 44.84% |
| **`close_morning61` reranker** | **1.193%** | **1.351%** | **61.32%** | **2.30** | **4.76** | **44.84%** |

The reranker score is:

```text
decision_score = rank_pct + 0.5 * p_good_pct
```

Both percentiles are calculated within each date's candidate panel. Compared
with rank-only `close_morning61`, scheduled mean return improved by 0.0463
percentage points and Sharpe by 0.1834. MDD did not improve; a separate causal
`p_bad` gate lowered MDD in exploration but also lowered scheduled return, so it
is not activated by this release.

## 3. Single-stock execution policy

- Policy: `always_buy_top1`
- BUY dates: 1,903 / 2,155
- ABSTAIN dates: 252, all warm-up dates with insufficient policy history
- Post-warm-up behavior: exactly one top `decision_score` stock per eligible day
- Scheduled win rate: 54.15%
- Active-trade win rate: 61.32%
- Close-to-morning compounded MDD: 44.84%
- Quality score: **68/100**

## 4. Bundle and serving contract

The candidate bundle was saved under
`/tmp/ml-close-morning-reranker-artifacts/close_morning61_2026-08-03` and
contains:

- 61 numeric feature columns;
- `single_stock_policy.score_col = "decision_score"`;
- `oof_score_col = daily_score_col = "decision_score"`;
- `decision_score_config.version = "close-morning-reranker-v1"`;
- `decision_score_config.p_good_weight = 0.5`.

The result remains research-stage evidence. Promotion requires an untouched
post-freeze paper/live period with positive scheduled net mean and PF above 1
under the same cost contract.
