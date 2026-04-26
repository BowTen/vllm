# Experiment 2.1 Basic Performance and Gamma Sweep

## Basic Information

| Field | Value |
|---|---|
| Date | 2026-04-26 |
| Git commit | `0601b5e9b` plus local benchmark fix for forwarding `--no-enforce-eager` to the verifier subprocess |
| Experiment | 2.1 基础性能对比与 gamma sweep |
| Purpose | 在本地双端 edge-verifier 设置下，对比 DSSD 与 target-only 的吞吐表现，并分析 `gamma` 对性能的影响。 |

## Settings

| Field | Value |
|---|---|
| Model pair | OPT-125M / OPT-6.7B |
| Edge model | `/root/autodl-tmp/opt-125m` |
| Verifier model | `/root/autodl-tmp/opt-6.7b` |
| Tokenizer | `/root/autodl-tmp/opt-125m` |
| Prompt file | `benchmarks/dssd/prompts/high_acceptance_opt.jsonl` |
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

## Raw Results

| Mode | Gamma | Raw JSON |
|---|---:|---|
| Target-only | - | `experiment-2.1-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-local-repeat20-20260426.json` |
| DSSD | 1 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g1-local-repeat20-20260426.json` |
| DSSD | 2 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g2-local-repeat20-20260426.json` |
| DSSD | 4 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g4-local-repeat20-20260426.json` |
| DSSD | 6 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g6-local-repeat20-20260426.json` |
| DSSD | 8 | `experiment-2.1-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-local-repeat20-20260426.json` |

## Results

| Model Pair | Mode | Gamma | Draft Acceptance | Avg Accepted Len / Round | Server token/s | Speedup |
|---|---|---:|---:|---:|---:|---:|
| OPT-125M / OPT-6.7B | Target-only | - | - | - | 103.29 | 1.00x |
| OPT-125M / OPT-6.7B | DSSD | 1 | 93.94% | 0.94 | 133.72 | 1.29x |
| OPT-125M / OPT-6.7B | DSSD | 2 | 96.02% | 1.92 | 191.86 | 1.86x |
| OPT-125M / OPT-6.7B | DSSD | 4 | 89.29% | 3.57 | 261.24 | 2.53x |
| OPT-125M / OPT-6.7B | DSSD | 6 | 84.11% | 5.05 | 313.26 | 3.03x |
| OPT-125M / OPT-6.7B | DSSD | 8 | 81.99% | 6.56 | 360.81 | 3.49x |

## Commands

DSSD runs used the `benchmarks.dssd.benchmark_edge_verifier_decode` helper with:

```bash
PYTHONPATH=$PWD .venv/bin/python benchmarks/dssd/benchmark_edge_verifier_decode.py \
  --model /root/autodl-tmp/opt-6.7b \
  --edge-model /root/autodl-tmp/opt-125m \
  --verifier-model /root/autodl-tmp/opt-6.7b \
  --edge-cuda-visible-devices 0 \
  --verifier-cuda-visible-devices 1 \
  --edge-model-runner v1 \
  --verifier-model-runner v1 \
  --prompt-jsonl benchmarks/dssd/prompts/high_acceptance_opt.jsonl \
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

Target-only used the same verifier server construction and HTTP transport, calling verifier `/generate` directly for 2 warmup repeats and 20 measured repeats.

## Notes

Before the full run, the benchmark launcher was fixed so `BenchmarkConfig.enforce_eager=False` is propagated to the verifier subprocess as `--no-enforce-eager`. Without that fix, the edge process used compiled execution while the verifier process stayed in eager mode.

In this high-acceptance prompt, DSSD is faster than target-only for all tested `gamma` values. The best measured setting in the repeat=20 run is `gamma=8`, reaching 360.81 token/s and 3.49x speedup. Although draft acceptance decreases as `gamma` grows, the larger accepted length per verification round more than offsets the extra draft work in this local no-network setting.
