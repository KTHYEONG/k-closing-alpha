# ML v7 Risk-Aware Reranker v2 Execution Report

Date: 2026-08-05  
Execution: real-data nested purged walk-forward selection through 2025-12-30  
Candidate: `close_morning61 + scenario_action + close-morning-reranker-v2-research`  
Baseline: `close_morning61 + scenario_action + close-morning-reranker-v1` (unchanged)

The v2 experiment reuses the calibrated `p_bad` OOF estimate to break ties away
from elevated loss-risk candidates:

```text
decision_score = rank_pct + w_good * p_good_pct - w_bad * p_bad_pct
w_good = 0.5,  w_bad in {0, 0.5, 1.0}
```

The `p_bad` penalty is bounded by the existing +1% good / -2% bad label severity
contract. Selection is nested and strictly causal: each outer fold chooses
`w_bad` only from an inner purged walk-forward OOF built exclusively from that
fold's outer-train dates, then applies the chosen configuration to the outer
validation dates exactly once. No 2026 data is used (2026 is incomplete).

## 1. Reproduction conditions

| Item | Value |
| --- | --- |
| Raw rows through 2025-12-30 | 33,795 |
| Scenario-action panel rows | 33,678 |
| Trade dates | 2,451 |
| Outer OOF dates (all folds) | 2,040 |
| Model | LightGBM Huber return model + chronological `p_good`/`p_bad` models |
| CV | Outer 5-fold purged group walk-forward, one-date purge |
| Inner selection | Per-outer-fold 5-fold purged walk-forward on outer-train dates only |
| Policy | `always_buy_top1` with existing warm-up semantics (252 dates) |
| Target | `target_return` decimal net, 0.20% round-trip cost deducted once |
| Score | `decision_score = rank_pct + 0.5 * p_good_pct - w_bad * p_bad_pct` |

The warm-up window (252 dates) is anchored to the full timeline; because every
outer validation window starts after that window, each fold keeps exactly one
warm-up abstention (5 abstentions / 2,040 dates), matching the baseline's
"warm-up dates are kept" semantics.

## 2. Fold-wise nested selection

| Fold | Outer-train groups | Chosen `w_bad` | Inner baseline mean / MDD | Inner best-eligible mean / MDD |
| --- | ---: | ---: | ---: | ---: |
| 0 | 410 | **1.0** | 0.1460% / 11.57% | 0.1532% / 11.33% (`w_bad=1.0`) |
| 1 | 818 | 0.0 | 1.0947% / 15.08% | none (penalties cut mean) |
| 2 | 1,226 | 0.0 | 1.3042% / 19.16% | none (penalties cut mean) |
| 3 | 1,634 | 0.0 | 1.3949% / 23.76% | none (penalties cut mean) |
| 4 | 2,042 | 0.0 | 1.1884% / 23.76% | none (penalties cut mean) |

Only fold 0 found a penalty eligible under the conservative rule (inner scheduled
mean no lower than v1 and compounded MDD strictly lower). Folds 1-4 fail-closed
to v1 because every non-zero penalty reduced inner scheduled mean below the v1
baseline; fold 2 additionally shows that `w_bad=0.5` lowering inner MDD (16.21%
vs 19.16%) is correctly rejected because the mean dropped below baseline.

## 3. Outer OOF v1 vs v2 (identical dates, always_buy_top1)

| Metric | v1 (`w_bad=0`) | v2 (nested-selected) | Delta |
| --- | ---: | ---: | ---: |
| Scheduled mean | 1.5316% | 1.4726% | -0.0590 pp |
| Scheduled win rate | 63.73% | 63.19% | -0.54 pp |
| Profit factor | 2.74 | 2.67 | -0.07 |
| Scheduled Sharpe | 6.04 | 5.86 | -0.18 |
| Compounded MDD | 33.94% | 33.94% | 0.00 pp |
| BUY / scheduled dates | 2,035 / 2,040 | 2,035 / 2,040 | 0 |

The MDD did not improve at all (33.94% → 33.94%) and scheduled mean fell by
0.059 percentage points. The single fold that activated `w_bad=1.0` (fold 0)
degraded out-of-sample: outer scheduled mean 1.5962% → 1.3010% with unchanged
MDD, so the inner-selected penalty did not generalize. These v1 baseline figures
are recomputed on the identical nested outer OOF dates and are **not** directly
comparable to `ml-v6.md` (different data cutoff and warm-up accounting).

## 4. Yearly results (outer OOF, scheduled mean / PF / Sharpe)

| Year | v1 mean | v2 mean | v1 PF | v2 PF | v1 Sharpe | v2 Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 2.188% | 1.888% | 5.22 | 5.04 | 9.33 | 9.22 |
| 2018 | 1.541% | 1.225% | 3.50 | 2.89 | 7.17 | 5.92 |
| 2019 | 1.571% | 1.489% | 3.00 | 2.92 | 6.59 | 6.36 |
| 2020 | 2.313% | 2.313% | 3.81 | 3.81 | 8.19 | 8.19 |
| 2021 | 1.630% | 1.630% | 2.76 | 2.76 | 6.24 | 6.24 |
| 2022 | 1.105% | 1.105% | 2.09 | 2.09 | 4.29 | 4.29 |
| 2023 | 1.330% | 1.330% | 2.34 | 2.34 | 5.18 | 5.18 |
| 2024 | 1.332% | 1.332% | 2.32 | 2.32 | 5.12 | 5.12 |
| 2025 | 1.209% | 1.209% | 2.23 | 2.23 | 4.82 | 4.82 |

The v2 penalty only affected 2017-2019 (fold 0's outer window) and reduced
scheduled mean in every affected year without lowering MDD.

## 5. Verdict

**REJECTED ablation.** The nested-selected v2 does **not** improve compounded MDD
over v1 (33.94% → 33.94%) and **reduces** scheduled mean (1.5316% → 1.4726%) on
the nested outer OOF, so it fails the promotion gate in
`docs/specs/ml_improvement_proposal.md`:

> "eligible for the next stage only when it improves compounded MDD over v1
> without reducing scheduled mean on the nested outer OOF"

The production candidate is unchanged: v1 remains active, and no bundle or
reranker default was overwritten. This is a selection/risk-control experiment,
not evidence that `p_bad` has proven alpha; `docs/results/ml-v1.md` records the
calibration uncertainty that the gate was designed to respect.
