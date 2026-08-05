# ML Algorithm-Family Ensemble Improvement

## Goal

Measure and, only when independently validated, serve a model-family ensemble
for the current `close_morning61 + scenario_action` return-ranking path. The
experiment compares LightGBM with XGBoost, CatBoost, and Random Forest under
the same causal data, features, labels, policy, costs, and validation dates.

## Evidence and diagnosis

The project already has a recency ensemble, not an algorithm-family ensemble.
`run_close_morning_recency_ensemble_experiment` combines an expanding and a
252/504-date recency-weighted LightGBM Huber model. Its 2025-12-30 nested OOF
report improved scheduled mean from 1.5316% to 1.5933%, Sharpe from 6.04 to
6.31, PF from 2.74 to 2.87, and compounded MDD from 33.94% to 25.87%.

The generic fold factory currently permits only `ridge`, `lgb_ranker`, and
`lgb_regressor`. The current runtime dependencies contain LightGBM and
scikit-learn, but not XGBoost or CatBoost. Consequently no historical result
can claim that an XGBoost/CatBoost/Random Forest blend improves this project.

The current real-data panel is sufficiently large for a controlled experiment:
33,817 `scenario_action` rows, 2,589 trade dates (2016-01-04 through
2026-08-03), and 63 `close_morning61` columns. To isolate algorithm effect,
version 1 uses the existing numeric input columns; it does not add the two
available categorical columns (`market_type`, `theme_sector`).

## Design decision

Use a fixed, nested, rank-level blend rather than a learned stacker.

| Expert | Role | Expected marginal value | Initial setting |
| --- | --- | --- | --- |
| LightGBM Huber | Baseline/champion | Existing validated score | Existing constructor |
| XGBoost pseudo-Huber | Challenger | Different histogram-tree implementation; likely correlated, so improvement must be proven | Seed 42, one worker |
| CatBoost Huber | Challenger | Different ordered-tree implementation; numeric-only first isolates algorithm effect | Seed 42, one worker, no file output |
| RandomForestRegressor | Challenger | Bagged-tree diversification; expected to be slower and possibly lower signal | Seed 42, one worker |

Raw predictions cannot be averaged because their scales and loss behavior are
not comparable. For every trade date, each prediction becomes a percentile
rank, then the recipe takes a convex weighted mean. The current probability
contribution remains unchanged:

```text
ensemble_rank = sum(weight[m] * pct_rank(pred[m], by=trade_date))
decision_score = pct_rank(ensemble_rank, by=trade_date) + 0.5 * pct_rank(p_good, by=trade_date)
```

The fixed recipes are `lgb_only`, the three two-expert LightGBM blends,
`lgb_xgb_catboost_equal`, and `all_four_equal`. This compact list evaluates
each requested family while avoiding a continuous weight search over the same
OOF history.

## Causal evaluation and selection

1. Build the existing `close_morning61`, `scenario_action` panel and use
   `target_return` (decimal net, including the unchanged 0.20% round-trip
   cost), `trade_date`, `stock_code`, and `chart_analysis`.
2. Run the existing outer five-fold `PurgedGroupTimeSeriesSplit` with the
   current one-date purge. All four return experts produce aligned outer OOF
   predictions.
3. For each outer fold, create a separate inner purged walk-forward OOF using
   only the outer-train partition. Evaluate every fixed recipe under the
   existing `always_buy_top1` policy and unchanged warm-up semantics.
4. `lgb_only` is always valid. A challenger is eligible only if its finite
   inner scheduled mean is no lower and its finite entry-sequence MDD is
   strictly lower than `lgb_only`. Choose the lowest MDD, then highest mean,
   then fewest non-LightGBM experts, then lexical recipe id. Insufficient
   history or any invalid/misaligned OOF falls back to `lgb_only`.
5. Apply each selected recipe once to its untouched outer validation fold,
   concatenate the scheduled return sequences chronologically, and report
   aggregate, fold, yearly, standalone-expert, and recipe metrics.

The research bundle may be built only when a non-baseline aggregate recipe has
strictly higher scheduled mean, strictly lower compounded MDD, positive net
mean, and PF > 1. This mirrors the existing recency gate but additionally
requires an actual multi-model winner. Paper/live validation, capture-time
provenance, and execution slippage remain separate promotion prerequisites.

## Integration boundary

`src/ml/training/fitting.py` owns deterministic estimator creation, while
`experiments.py` owns nested selection. The public facade re-exports the new
experiment. `sizing_engine.py` gains a versioned, opt-in
`algorithm_ensemble_config` and model mapping; at inference it applies the
same groupwise percentile-rank blend. It rejects a bundle containing both this
configuration and `recency_ensemble_config`, so the two ensemble dimensions
are not stacked before a separate experiment proves their interaction.

The production bundle service is deliberately unchanged. A passing result can
create a versioned research artifact only; it cannot silently replace the
active model.

## Acceptance evidence

Implementation must add the five contract scenarios, run their targeted unit
tests, run the existing recency-ensemble tests unchanged, and produce a dated
real-data report with the fixed data cutoff, recipe table, fold choices,
aggregate metrics, and explicit `promoted` verdict. A non-passing experiment
is a valid result and must leave the production path untouched.
