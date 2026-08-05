# ML v5 Close-to-Morning Pipeline Execution Report

Date: 2026-08-05  
Execution: full real-data retraining and 5-fold purged walk-forward OOF  
Candidate: `close_morning61 + scenario_action + lgb_regressor`

This report records the latest pipeline execution. Returns use the project
contract: buy one stock at the close and sell the following morning. The source
rows are the common 15:20 KST snapshot; timestamp columns and timestamp-based
validation are not used.

## 1. Reproduction conditions

| Item | Value |
| --- | --- |
| Raw rows | 33,934 |
| Panel | `scenario_action` |
| Model | LightGBM Huber regressor |
| CV | 5-fold purged group walk-forward, one-date purge |
| OOF dates | 2,155 |
| Target | `target_return` (decimal net, existing 0.20% round-trip cost deducted) |
| Bundle training cutoff | 2026-08-03 |

## 2. Feature-set comparison

| Feature set | Features | NDCG@1 | Scheduled mean | Active-trade mean | Active win rate | PF | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base40` | 51 | 0.5002 | 0.858% | 0.971% | 57.38% | 1.88 | 3.56 |
| `snapshot49` | 60 | 0.5286 | 1.113% | 1.261% | 60.43% | 2.16 | 4.41 |
| **`close_morning61`** | **61** | **0.5282** | **1.147%** | **1.299%** | **61.74%** | **2.23** | **4.58** |

For `close_morning61`, the scheduled mean includes warm-up abstention dates;
active-trade mean is the mean over dates where a stock was actually bought.

## 3. Single-stock policy and risk

- Policy selected: `always_buy_top1`
- BUY dates: 1,903 / 2,155
- ABSTAIN dates: 252, all due to insufficient policy history during warm-up
- Buy rate after warm-up: 100% (one top-ranked stock per eligible date)
- Scheduled win rate: 54.52%
- Active-trade win rate: 61.74%
- Close-to-morning compounded MDD: 44.84%
- Transparent quality score: **67/100**

No margin-based abstention candidate improved scheduled return or Sharpe in this
run, so no fixed abstention threshold was activated.

## 4. Bundle integrity

The real candidate bundle was written to `/tmp/ml-close-morning-artifacts` and
contained:

- `feature_set = close_morning61`
- 61 feature columns
- persisted `single_stock_policy` (`always_buy_top1`)
- `oof_score_col = pred`
- `daily_score_col = rank_score`

The result remains research-stage evidence. Promotion still requires an
untouched post-freeze close-to-morning paper/live period with positive net mean
return and PF above 1 under the same cost contract.
