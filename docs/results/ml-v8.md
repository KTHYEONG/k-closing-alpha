# ML v8 Recency-Adaptive Dual-Horizon Ensemble Report

Date: 2026-08-05  
Execution: real-data nested purged walk-forward selection through 2025-12-30  
Candidate: `close_morning61 + scenario_action + close-morning-recency-ensemble-research`  
Baseline: `close_morning61 + scenario_action + close-morning-reranker-v1` (unchanged)

The experiment replaces the single expanding-history Huber return model with a
causal dual-horizon ensemble: the stable expanding expert and a recently
weighted expert (half-life `h` on trade-date groups, `exp(-ln(2) * age / h)`
normalized to mean one). Decision-time scores are groupwise percentile ranks
only, preserving the existing `p_good` rank contribution and `always_buy_top1`
policy:

```text
ensemble_rank = (1 - alpha) * pct_rank(pred_expanding) + alpha * pct_rank(pred_recent)
decision_score = pct_rank(ensemble_rank) + 0.5 * pct_rank(p_good)
```

`alpha = 0` is the v1 baseline, retained exactly once. The fixed candidate set is
`h in {252, 504}` and `alpha in {0, 0.25, 0.50, 0.75, 1.00}` — no continuous
hyperparameter search. Selection is nested and strictly causal: each outer fold
chooses `(h, alpha)` only from an inner purged walk-forward OOF built
exclusively from that fold's outer-train dates, then applies the chosen
configuration to the outer validation dates exactly once. No 2026 data is used
(2026 is incomplete).

## 1. Reproduction conditions

| Item | Value |
| --- | --- |
| Raw rows through 2025-12-30 | 33,934 |
| Scenario-action panel rows | 33,678 |
| Trade dates | 2,451 |
| Outer OOF dates (all folds) | 2,040 |
| Model | Expanding Huber + half-life Huber return experts + chronological `p_good` model |
| CV | Outer 5-fold purged group walk-forward, one-date purge |
| Inner selection | Per-outer-fold 5-fold purged walk-forward on outer-train dates only |
| Policy | `always_buy_top1` with existing warm-up semantics (252 dates) |
| Target | `target_return` decimal net, 0.20% round-trip cost deducted once |
| Score | `decision_score = pct_rank(ensemble_rank) + 0.5 * pct_rank(p_good)` |

The warm-up window (252 dates) is anchored to the full timeline; because every
outer validation window starts after that window, each fold keeps exactly one
warm-up abstention (5 abstentions / 2,040 dates), matching the baseline's
"warm-up dates are kept" semantics. The recomputed v1 baseline here is
1.5316% / PF 2.74 / Sharpe 6.04 / MDD 33.94% — identical to the nested outer OOF
baseline in `docs/results/ml-v7.md`, confirming cutoff and warm-up consistency.

## 2. Fold-wise nested selection

| Fold | Outer-train groups | Chosen `(h, alpha)` | Inner baseline mean / MDD | Inner best-eligible mean / MDD |
| --- | ---: | ---: | ---: | ---: |
| 0 | 410 | v1 (`alpha=0`) | 0.1460% / 11.57% | none (504\|0.5: 0.2313% / 11.76%, MDD not strictly lower) |
| 1 | 818 | (504, 0.75) | 1.0947% / 15.08% | 1.1121% / 12.12% |
| 2 | 1,226 | (252, 0.50) | 1.3042% / 19.16% | 1.3625% / 17.10% |
| 3 | 1,634 | (252, 0.50) | 1.3949% / 23.75% | 1.4195% / 19.50% |
| 4 | 2,042 | (252, 0.50) | 1.1884% / 23.75% | 1.2611% / 20.65% |

Fold 0 (2016–2017 only) fail-closed to v1: every non-zero candidate lowered inner
compounded MDD below baseline only at the cost of mean, or failed the strict MDD
test — the conservative rule correctly rejects `504|0.5` (higher mean 0.2313% vs
0.1460% but MDD 11.76% vs 11.57%). Folds 1–4 each found a `(h, alpha)` candidate
with inner scheduled mean no lower than v1 and strictly lower compounded MDD; the
recently weighted expert consistently reduces deep-drawdown regimes as history
lengthens.

## 3. Outer OOF v1 vs candidate (identical dates, always_buy_top1)

| Metric | v1 (`alpha=0`) | Candidate (nested-selected) | Delta |
| --- | ---: | ---: | ---: |
| Scheduled mean | 1.5316% | 1.5933% | +0.0617 pp |
| Scheduled win rate | 63.73% | 63.53% | -0.20 pp |
| Profit factor | 2.74 | 2.87 | +0.13 |
| Scheduled Sharpe | 6.04 | 6.31 | +0.27 |
| Compounded MDD | 33.94% | 25.87% | -8.07 pp |
| BUY / scheduled dates | 2,035 / 2,040 | 2,035 / 2,040 | 0 |

The recency ensemble improves both economics and tail risk on the concatenated
nested outer OOF: scheduled mean +0.062 pp, PF +0.13, Sharpe +0.27, and
compounded MDD down 8.07 pp. Note the improvement is not uniform per fold —
fold 2's outer candidate MDD is slightly higher than its own baseline (25.87% vs
25.13%) — but the fold-level selections were made from inner OOF alone, and the
concatenated series passes both promotion legs.

Fold-wise outer mean / MDD (baseline → candidate): 1.5962% / 15.08% → 1.5962% /
15.08% (fold 0, unchanged); 2.0941% / 19.29% → 2.1801% / 17.77%; 1.4364% /
25.13% → 1.4943% / 25.87%; 1.2322% / 25.18% → 1.3248% / 19.58%; 1.2994% /
33.94% → 1.3710% / 24.60%.

## 4. Yearly results (outer OOF, scheduled mean / PF / Sharpe)

| Year | v1 mean | candidate mean | v1 PF | candidate PF | v1 Sharpe | candidate Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 2.188% | 2.188% | 5.22 | 5.22 | 9.33 | 9.33 |
| 2018 | 1.541% | 1.541% | 3.50 | 3.50 | 7.17 | 7.17 |
| 2019 | 1.571% | 1.687% | 3.00 | 3.24 | 6.59 | 7.11 |
| 2020 | 2.313% | 2.340% | 3.81 | 3.91 | 8.19 | 8.45 |
| 2021 | 1.630% | 1.761% | 2.76 | 3.18 | 6.24 | 7.08 |
| 2022 | 1.105% | 0.926% | 2.09 | 1.82 | 4.29 | 3.51 |
| 2023 | 1.330% | 1.549% | 2.34 | 2.69 | 5.18 | 5.95 |
| 2024 | 1.332% | 1.344% | 2.32 | 2.36 | 5.12 | 5.17 |
| 2025 | 1.209% | 1.397% | 2.23 | 2.52 | 4.82 | 5.56 |

2017–2018 are unchanged (fold 0 chose v1). 2019–2021 improve strongly; 2022
degrades (1.105% → 0.926%), then 2023–2025 recover with 2025 at its best PF
(2.52) and Sharpe (5.56). The 2022 dip is the only year with a non-trivial mean
loss, offset by the 8.07 pp MDD reduction and the post-2022 gains.

## 5. Verdict

**NESTED OOF GATE PASSED.** The nested-selected candidate beats the v1 baseline
on scheduled mean (1.5316% → 1.5933%) and has lower compounded MDD (33.94% →
25.87%) on the concatenated outer OOF, and it satisfies the untouched paper/live
gate conditions (positive scheduled net mean, PF 2.87 > 1 under the unchanged
0.20% round-trip cost).

Per `docs/specs/ml_recency_adaptive_ensemble.md`, the v1 production candidate
remains unchanged unless the nested gate passes; the gate passing means the
versioned research bundle (expanding + recent return models and
`recency_ensemble_config`) may now be produced on request. Production promotion
is **still not granted** here: this run does not solve missing capture-time
provenance, execution slippage, or the untouched out-of-sample/paper-live
validation requirement — those remain hard promotion prerequisites. The fold
selections and this report must not be used to overfit further candidates.
