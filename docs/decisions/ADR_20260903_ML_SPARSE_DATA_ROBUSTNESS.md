# ADR_20260903_ML_SPARSE_DATA_ROBUSTNESS — Sparse-data ML robustness

## Context
Panel: 33,827 rows / 2,599 days, median 12 candidates/day, 12% days <5
candidates, 158 single-candidate days. Bottleneck is evaluation power, not
the model. Prior promotion gate compared one paired daily mean on a ~123-day
window — a verified +16-18% decision-metric gain is noise under it.

## Correction of prior claim (R10)
Alpha is stable, not decaying. Walk-forward OOF rankIC by test-year
2021-2025: 0.19/0.21/0.18/0.18/0.18; 2020's 0.30 is a COVID-volatility
artifact. Daily target dispersion flat ~0.031 every year. The 0.24->0.17
"decline" was a 2020-window artifact. Effort targets a stable ceiling.

## Verified lever vs rejected hypotheses (R9)
- ISOLATED (raw top-1 argmax, no policy/weighting): regularization-biased HPO
  space (num_leaves 8..63, min_child_samples floor 20, min_split_gain,
  path_smooth) lifts library-default top-1 +1.057%/day Sharpe 3.97 ->
  +1.23-1.25%/day Sharpe 4.6-4.7, bootstrap p=0.018-0.025, monotone in
  regularization strength.
- REJECT: cross-sectional target demeaning (dIC -0.0155, p=0.17).
- REJECT: naive disclosure + KOSPI200-basis bolt-on (dIC -0.006, p=0.60;
  disclosure columns rank 43-62 of 70).

## Real-data pipeline runs (retrain, OOS reserve 2025-07-01, date_balanced+recency504)
Both HPO configs FAIL the new significance gate (shared_dates=1935 dev OOF,
moving-block bootstrap vs library-default control):
- A walkforward / rank_ic: best rankIC 0.220, HPO picked num_leaves=56.
  Gate delta +0.065%/day, p=0.358 -> NOT promoted. cand top1 +1.460% Sharpe
  5.87 vs control +1.395% Sharpe 5.72.
- B cpcv / cpcv_top1: best_value 0.0154, HPO picked num_leaves=8 (objective
  works as designed). Gate delta -0.033%/day, p=0.665 -> NOT promoted. cand
  top1 +1.362% Sharpe 5.58 vs control +1.395% Sharpe 5.72.
Conclusion: the isolated +18bp regularization gain does NOT survive the full
policy + date_balanced/recency + p_good-blend stack. With the current
61-feature snapshot representation, tuning alone yields no
statistically-defensible improvement over LightGBM defaults. The gate
correctly refuses both (the old cand>=ctrl gate would have promoted A on
+1.460 > +1.395 -- a noise promotion). rank_ic HPO picks leaves=56 while
cpcv_top1 picks leaves=8: confirmed rank_ic is flat w.r.t. the decision
metric.

## Decision
CPCV(8,2)=28 folds / 7 paths evaluation, moving-block bootstrap promotion
gate (delta>0 and p<0.10), DSR trial discount, reg-biased HPO default.
eval_mode is the single OOF-routing switch consumed by
tune_return_model_params (not a config-only field); hpo_objective='cpcv_top1'
requires eval_mode='cpcv'; retrain.py exposes --hpo-objective so the CPCV
path is reachable from the CLI. Backward compatible: eval_mode='walkforward',
hpo_objective='rank_ic' defaults unchanged.

## Fixes applied during check/run
- eval_mode was a ghost field (defined + CLI-wired, never read); made it the
  live switch in _objective, added --hpo-objective, cross-check
  cpcv_top1<->cpcv, purge-aware CPCV n_groups floor
  (k_test*(1+purge+embargo)+1).
- cpcv_oof_predict pd.concat raised on dev_df.attrs['feature_manifest']
  (a DataFrame) -> every CPCV trial returned -inf. Fixed by clearing .attrs
  on the working frame and per-fold slices; regression test added.

## Roadmap
P2: retrain --tuned --eval-mode cpcv --hpo-trials 60, monotone_constraints
validation per-feature via CPCV path top-1, ship iff p<0.10 vs control.
P3: alt-data gated per-group (delta>0, p<0.10 vs 61-feature control) after
adapter repair + backfill. P4: evaluate or drop legacy lambdarank fallback.
