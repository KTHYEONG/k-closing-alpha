# ML v4 Close-to-Morning Quality Frozen Benchmark

Date: 2026-08-05  
Frozen status: post-freeze paper/live promotion pending  
Candidate: `close_morning61 + scenario_action + lgb_regressor`  
Controls: `base40`, `snapshot49` (identical OOF dates)

This file is the frozen benchmark for `ml_close_to_morning_quality`. Numbers
below are the frozen reference results. Feature or policy parameters must **not**
be retuned against any post-freeze period; that period is reserved as an
untouched promotion test.

## 1. Strategy contract

Each trade buys one selected stock at the close and exits the next morning.
`net_return` is the realised close-to-morning return and `target_return` is its
decimal-net value after the existing 0.20% round-trip cost deduction. Positions
do not overlap, so daily compounded return and MDD are valid strategy metrics.
Every raw feature is part of the common 15:20 KST source snapshot (static source
contract): no timestamp columns, per-row timestamp validation, or
timestamp-derived features.

## 2. Reproduction conditions

| Item | Value |
| --- | --- |
| Raw rows | 33,934 |
| Panel | scenario-action |
| CV | 5-fold purged group walk-forward, one-date purge |
| Model | LightGBM Huber regressor |
| Evaluation cutoff | 2025-12-30 |
| OOF rows | 31,045 |
| Unique dates | 2,040 |
| Target | `target_return` (decimal net, cost deducted once) |
| MDD | Compounded close-to-morning strategy metric |

## 3. Frozen champion comparison

| Feature set | Features | Top-1 net mean | Sharpe | MDD | Single-stock active mean | Active win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base40` | 51 | 1.1393% | 4.473 | 40.23% | 1.0472% | 57.66% |
| `snapshot49` | 60 | 1.4569% | 5.665 | 40.72% | 1.4040% | 61.35% |
| `interaction53` | 64 | 1.4898% | 5.689 | 37.77% | 1.4308% | 61.63% |
| `production_calendar_flow` | 60 | 1.1097% | 4.369 | 40.46% | 1.0407% | 58.00% |
| **`close_morning61`** | **61** | **1.5405%** | **5.947** | **29.34%** | **1.4909%** | **62.08%** |

`close_morning61` is exactly `snapshot49` plus `relative_flow_strength` — the
product of same-date percentile ranks of major investor-flow density and price
change. Its one-feature ablation over `snapshot49` improves Top-1 return by
0.0836 percentage points, Sharpe by 0.281, and MDD by 11.38 percentage points.
`interaction53` is not promoted: `range_efficiency` and `flow_turnover` dilute
the stronger isolated signal.

## 4. Production-readiness score: 71/100 (research alpha, not approved)

| Dimension | Weight | Score | Evidence |
| --- | ---: | ---: | --- |
| Selection edge | 30 | 27 | Best OOF NDCG@1 0.5032 and Rank IC 0.2037. |
| Net economics | 20 | 17 | Top-1 net mean 1.5405%, win rate 62.70%, PF 2.663, Sharpe 5.947. |
| Risk and stability | 20 | 15 | MDD 29.34%; all OOF years 2017–2025 are positive. |
| Validation independence | 20 | 8 | Purged walk-forward OOF exists, but feature selection reused the complete history. |
| Daily deployment integrity | 10 | 4 | The policy exists, but the default feature set was weaker and saved bundles omitted the calibrated policy. |
| **Total** | **100** | **71** | Strong research alpha, not production-approved. |

## 5. Decision and promotion gate

- Margin-based abstention lowers scheduled-date return and Sharpe; keep the
  causally selected `always_buy_top1` policy and do not activate a fixed
  abstention gate in this iteration.
- The implementation sets `close_morning61` as the default candidate training
  feature set and persists the calibrated `SingleStockPolicy` in candidate
  bundles (`oof_score_col="pred"` → `daily_score_col="rank_score"`).
- **Promotion requires a post-freeze close-to-morning paper/live period with
  positive net mean return and PF above 1 under the same 0.20% round-trip cost
  contract.** Do not retune features or policy parameters using that period.
- Until then the strategy remains research-only.
