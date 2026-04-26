# Experiment 2.2 Acceptance Rate and Performance

## Basic Information

| Field | Value |
|---|---|
| Date | 2026-04-26 |
| Git commit | `06dda04ca` plus local 2.2 prompt/result files |
| Experiment | 2.2 接受率对性能的影响 |
| Purpose | 对比高接受率连续文本和普通混合 prompt 下的 DSSD 加速效果，观察 draft acceptance rate 与吞吐收益的关系。 |

## Settings

| Field | Value |
|---|---|
| High-acceptance prompt | `benchmarks/dssd/prompts/high_acceptance_opt.jsonl` |
| General prompt | `benchmarks/dssd/prompts/general_mixed_prompt.jsonl` |
| Prompt selection | first prompt, truncated to 128 tokens |
| prompt_len | 128 |
| output_tokens | 256 |
| temperature | 0.0 |
| ignore_eos | true |
| repeats | 20 |
| warmup_repeats | 2 |
| warmup_tokens | 0 |
| Edge GPU / verifier GPU | 0 / 1 |
| Model runner | v1 / v1 |
| Execution | `--no-enforce-eager` |
| Network simulation | disabled |
| Server token/s metric | mean request output token/s, excluding model startup and warmup repeats |

## Summary Results

| Model Pair | Prompt Type | Target-only token/s | Best Gamma | Draft Acceptance | Avg Accepted Len / Round | DSSD token/s | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| OPT-125M / OPT-6.7B | High acceptance | 103.29 | 8 | 81.99% | 6.56 | 360.81 | 3.49x |
| OPT-125M / OPT-6.7B | General mixed | 103.30 | 8 | 45.39% | 3.63 | 218.02 | 2.11x |
| Qwen3-0.6B / Qwen3-8B | High acceptance | 93.05 | 4 | 69.49% | 2.78 | 153.94 | 1.65x |
| Qwen3-0.6B / Qwen3-8B | General mixed | 94.40 | 4 | 55.00% | 2.20 | 130.51 | 1.38x |

## OPT-125M / OPT-6.7B General Prompt Sweep

| Mode | Gamma | Draft Acceptance | Avg Accepted Len / Round | Server token/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Target-only | - | - | - | 103.30 | 1.00x |
| DSSD | 1 | 83.57% | 0.84 | 125.63 | 1.22x |
| DSSD | 2 | 77.72% | 1.55 | 164.78 | 1.60x |
| DSSD | 4 | 61.49% | 2.46 | 190.05 | 1.84x |
| DSSD | 6 | 52.42% | 3.15 | 217.90 | 2.11x |
| DSSD | 8 | 45.39% | 3.63 | 218.02 | 2.11x |

## Qwen3-0.6B / Qwen3-8B High-Acceptance Prompt Sweep

| Mode | Gamma | Draft Acceptance | Avg Accepted Len / Round | Server token/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Target-only | - | - | - | 93.05 | 1.00x |
| DSSD | 1 | 88.24% | 0.88 | 96.67 | 1.04x |
| DSSD | 2 | 78.50% | 1.57 | 127.40 | 1.37x |
| DSSD | 4 | 69.49% | 2.78 | 153.94 | 1.65x |
| DSSD | 6 | 59.06% | 3.54 | 149.86 | 1.61x |
| DSSD | 8 | 51.47% | 4.12 | 152.27 | 1.64x |

This Qwen3 high-acceptance sweep uses `benchmarks/dssd/prompts/general_mixed_prompt.jsonl`.

## Qwen3-0.6B / Qwen3-8B General Prompt Sweep

| Mode | Gamma | Draft Acceptance | Avg Accepted Len / Round | Server token/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Target-only | - | - | - | 94.40 | 1.00x |
| DSSD | 1 | 64.74% | 0.65 | 89.92 | 0.95x |
| DSSD | 2 | 55.33% | 1.11 | 103.65 | 1.10x |
| DSSD | 4 | 55.00% | 2.20 | 130.51 | 1.38x |
| DSSD | 6 | 43.66% | 2.62 | 127.55 | 1.35x |
| DSSD | 8 | 38.08% | 3.05 | 119.29 | 1.26x |

This Qwen3 general prompt sweep uses `benchmarks/dssd/prompts/high_acceptance_opt.jsonl`.

## Raw Results

| Case | Raw JSON |
|---|---|
| OPT high acceptance target-only | `experiment-2.1-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-local-repeat20-20260426.json` |
| OPT high acceptance DSSD gamma=8 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-local-repeat20-20260426.json` |
| OPT general target-only | `experiment-2.2-target-only-opt-125m-opt-6.7b-general-p128-o256-local-repeat20-20260426.json` |
| OPT general DSSD gamma=1 | `experiment-2.2-dssd-opt-125m-opt-6.7b-general-p128-o256-g1-local-repeat20-20260426.json` |
| OPT general DSSD gamma=2 | `experiment-2.2-dssd-opt-125m-opt-6.7b-general-p128-o256-g2-local-repeat20-20260426.json` |
| OPT general DSSD gamma=4 | `experiment-2.2-dssd-opt-125m-opt-6.7b-general-p128-o256-g4-local-repeat20-20260426.json` |
| OPT general DSSD gamma=6 | `experiment-2.2-dssd-opt-125m-opt-6.7b-general-p128-o256-g6-local-repeat20-20260426.json` |
| OPT general DSSD gamma=8 | `experiment-2.2-dssd-opt-125m-opt-6.7b-general-p128-o256-g8-local-repeat20-20260426.json` |
| Qwen3 high acceptance target-only (`general_mixed_prompt.jsonl`) | `experiment-2.2-target-only-qwen3-0.6b-qwen3-8b-general-p128-o256-local-repeat20-20260426.json` |
| Qwen3 high acceptance DSSD gamma=1 (`general_mixed_prompt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-general-p128-o256-g1-local-repeat20-20260426.json` |
| Qwen3 high acceptance DSSD gamma=2 (`general_mixed_prompt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-general-p128-o256-g2-local-repeat20-20260426.json` |
| Qwen3 high acceptance DSSD gamma=4 (`general_mixed_prompt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-general-p128-o256-g4-local-repeat20-20260426.json` |
| Qwen3 high acceptance DSSD gamma=6 (`general_mixed_prompt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-general-p128-o256-g6-local-repeat20-20260426.json` |
| Qwen3 high acceptance DSSD gamma=8 (`general_mixed_prompt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-general-p128-o256-g8-local-repeat20-20260426.json` |
| Qwen3 general target-only (`high_acceptance_opt.jsonl`) | `experiment-2.2-target-only-qwen3-0.6b-qwen3-8b-high-accept-p128-o256-local-repeat20-20260426.json` |
| Qwen3 general DSSD gamma=1 (`high_acceptance_opt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-high-accept-p128-o256-g1-local-repeat20-20260426.json` |
| Qwen3 general DSSD gamma=2 (`high_acceptance_opt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-high-accept-p128-o256-g2-local-repeat20-20260426.json` |
| Qwen3 general DSSD gamma=4 (`high_acceptance_opt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-high-accept-p128-o256-g4-local-repeat20-20260426.json` |
| Qwen3 general DSSD gamma=6 (`high_acceptance_opt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-high-accept-p128-o256-g6-local-repeat20-20260426.json` |
| Qwen3 general DSSD gamma=8 (`high_acceptance_opt.jsonl`) | `experiment-2.2-dssd-qwen3-0.6b-qwen3-8b-high-accept-p128-o256-g8-local-repeat20-20260426.json` |

## Command Template

General prompt DSSD runs used:

```bash
PYTHONPATH=$PWD .venv/bin/python benchmarks/dssd/benchmark_edge_verifier_decode.py \
  --model <verifier-model> \
  --edge-model <edge-model> \
  --verifier-model <verifier-model> \
  --edge-cuda-visible-devices 0 \
  --verifier-cuda-visible-devices 1 \
  --edge-model-runner v1 \
  --verifier-model-runner v1 \
  --prompt-jsonl benchmarks/dssd/prompts/general_mixed_prompt.jsonl \
  --prompt-len 128 \
  --decode-tokens 256 \
  --warmup-tokens 0 \
  --repeats 20 \
  --warmup-repeats 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 512 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager \
  --gamma <1|2|4|6|8> \
  --json
```

Target-only runs used the same verifier server construction and HTTP `/generate` path with 2 warmup repeats and 20 measured repeats.

## Notes

The OPT high-acceptance row reuses experiment 2.1 repeat=20 results. For OPT-125M / OPT-6.7B, switching from the high-acceptance continuous prompt to the general mixed prompt lowers the best-run draft acceptance from 81.99% to 45.39%, and the best speedup drops from 3.49x to 2.11x.

For Qwen3-0.6B / Qwen3-8B, `general_mixed_prompt.jsonl` is treated as the high-acceptance condition because it has the higher measured draft acceptance. It reaches its best throughput at `gamma=4`. Larger `gamma` values still accept more tokens per round, but the lower draft acceptance offsets the benefit enough that `gamma=6` and `gamma=8` do not improve over `gamma=4`.
