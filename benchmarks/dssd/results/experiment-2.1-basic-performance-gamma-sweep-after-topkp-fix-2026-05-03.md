# Experiment 2.1 Basic Performance After Top-k/Top-p Fix

## Context

This rerun checks whether the DSSD verifier sampling correctness fix changes
the high-acceptance greedy performance in experiment 2.1.

## Settings

| Field | Value |
|---|---|
| Model pair | OPT-125M / OPT-6.7B |
| Edge model | `/root/autodl-tmp/opt-125m` |
| Verifier model | `/root/autodl-tmp/opt-6.7b` |
| Prompt file | `benchmarks/dssd/prompts/high_acceptance_opt.jsonl` |
| prompt_len | 128 |
| output_tokens | 256 |
| temperature | 0.0 |
| ignore_eos | true |
| repeats | 20 |
| warmup_repeats | 2 |
| Model runner | v1 / v1 |
| Execution | `--no-enforce-eager` |
| Network simulation | disabled |

Each DSSD gamma was run three times. Target-only was also run three times to
check whether the machine-level baseline shifted.

## Raw Results

| Mode | Gamma | Raw JSON |
|---|---:|---|
| Target-only | - | `experiment-2.1-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-local-repeat20-after-topkp-fix-20260503-062341-sweep1.json` |
| Target-only | - | `experiment-2.1-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-local-repeat20-after-topkp-fix-20260503-062341-sweep2.json` |
| Target-only | - | `experiment-2.1-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-local-repeat20-after-topkp-fix-20260503-062341-sweep3.json` |
| DSSD | 1 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g1-local-repeat20-after-topkp-fix-20260503-060849-sweep1.json` |
| DSSD | 1 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g1-local-repeat20-after-topkp-fix-20260503-060849-sweep2.json` |
| DSSD | 1 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g1-local-repeat20-after-topkp-fix-20260503-060849-sweep3.json` |
| DSSD | 2 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g2-local-repeat20-after-topkp-fix-20260503-060849-sweep1.json` |
| DSSD | 2 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g2-local-repeat20-after-topkp-fix-20260503-060849-sweep2.json` |
| DSSD | 2 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g2-local-repeat20-after-topkp-fix-20260503-060849-sweep3.json` |
| DSSD | 4 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g4-local-repeat20-after-topkp-fix-20260503-060849-sweep1.json` |
| DSSD | 4 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g4-local-repeat20-after-topkp-fix-20260503-060849-sweep2.json` |
| DSSD | 4 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g4-local-repeat20-after-topkp-fix-20260503-060849-sweep3.json` |
| DSSD | 6 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g6-local-repeat20-after-topkp-fix-20260503-060849-sweep1.json` |
| DSSD | 6 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g6-local-repeat20-after-topkp-fix-20260503-060849-sweep2.json` |
| DSSD | 6 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g6-local-repeat20-after-topkp-fix-20260503-060849-sweep3.json` |
| DSSD | 8 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-local-repeat20-after-topkp-fix-20260503-060849-sweep1.json` |
| DSSD | 8 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-local-repeat20-after-topkp-fix-20260503-060849-sweep2.json` |
| DSSD | 8 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-local-repeat20-after-topkp-fix-20260503-060849-sweep3.json` |

## Summary

| Mode | Gamma | Current runs server token/s | Current mean +/- stdev | Previous token/s | Change vs previous | Current speedup |
|---|---:|---|---:|---:|---:|---:|
| Target-only | - | 104.68, 104.81, 104.71 | 104.73 +/- 0.07 | 103.29 | +1.4% | 1.00x |
| DSSD | 1 | 127.34, 120.10, 121.77 | 123.07 +/- 3.79 | 133.72 | -8.0% | 1.18x |
| DSSD | 2 | 169.76, 163.09, 168.32 | 167.06 +/- 3.51 | 191.86 | -12.9% | 1.60x |
| DSSD | 4 | 208.71, 224.95, 202.99 | 212.21 +/- 11.39 | 261.24 | -18.8% | 2.03x |
| DSSD | 6 | 259.15, 243.01, 264.14 | 255.43 +/- 11.05 | 313.26 | -18.5% | 2.44x |
| DSSD | 8 | 381.26, 283.61, 355.45 | 340.11 +/- 50.60 | 360.81 | -5.7% | 3.25x |

Draft acceptance rates and average accepted lengths matched the previous
experiment values:

| Gamma | Draft Acceptance | Avg Accepted Len / Round |
|---:|---:|---:|
| 1 | 93.94% | 0.94 |
| 2 | 96.02% | 1.92 |
| 4 | 89.29% | 3.57 |
| 6 | 84.11% | 5.05 |
| 8 | 81.99% | 6.56 |

## Interpretation

Target-only throughput did not regress; it is slightly above the previous
103.29 token/s baseline. DSSD throughput is lower than the previous experiment
for gamma 1, 2, 4, and 6, while gamma 8 remains noisy and averages slightly
below the previous result.

The verifier top-k/top-p fix should not affect this greedy experiment directly,
because experiment 2.1 uses `temperature=0.0`. Therefore the measured drop should
be treated as a current-run DSSD performance regression or benchmark variability
that needs additional profiling, not as direct evidence that the non-greedy
top-k/top-p correction itself slowed the greedy path.
