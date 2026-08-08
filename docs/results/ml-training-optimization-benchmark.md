# ML Training Optimization Benchmark

Date: 2026-08-08

This is an execution benchmark, not a promotion result. It uses the frozen
history cutoff `2025-12-30`, `scenario_action`, purge gap 1, CPU screening,
discovery mode, and two outer folds. The full five-fold confirmation remains a
separate gated run.

## Runs

| Run | Result |
| --- | --- |
| Cold cache | Causal panel completed: 5,046,547 source rows → 33,520 keys, 722 columns, 8 batches, 267.89 s, peak RSS 2.71 GiB. The cache artifact is 92.7 MB. Control completed with scheduled mean 1.3353%, PF 2.34, MDD 29.63%. Candidate selection did not emit a completion event within the 900 s discovery budget and was stopped; no promotion decision was made. |
| Warm cache 1 | 33,520 rows × 722 columns read in 0.657 s; process RSS 427 MiB. |
| Warm cache 2 | 33,520 rows × 722 columns read in 0.106 s; process RSS 435 MiB. |
| Warm cache 3 | 33,520 rows × 722 columns read in 0.099 s; process RSS 437 MiB. |

The first attempted cold run with a 100,000-row Arrow batch was externally
terminated before `history_panel_built`. Reducing the batch to 25,000 rows
completed the causal build and kept the measured peak RSS within the 8 GiB
budget. This batch setting is therefore the benchmark baseline for the next
implementation pass.

## Diagnosis

The warm cache removes roughly four and a half minutes of causal panel build,
but the 720-feature fold-local selection remains the dominant bottleneck after
the control arm. The next optimization must instrument and reduce the
selection screen/correlation work before attempting a full five-fold
confirmation. These results do not justify production promotion.

Artifacts:

- Status/events: `/tmp/k_closing_ml_benchmark2/cold/run_status.json`,
  `/tmp/k_closing_ml_benchmark2/cold/run_events.jsonl`
- Warm cache: `/tmp/k_closing_ml_benchmark2/cache/`
