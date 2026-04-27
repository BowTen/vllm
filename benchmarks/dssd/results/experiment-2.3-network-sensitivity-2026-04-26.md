# Experiment 2.3 Network Sensitivity

## Basic Information

| Field | Value |
|---|---|
| Date | 2026-04-26 |
| Git commit | `e074dc4ad` plus local 2.3 result files |
| Experiment | 2.3 网络条件敏感性实验 |
| Purpose | 评估固定链路延迟和 100 Mbps 带宽限制对 DSSD 与 target-only 吞吐的影响。 |

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
| DSSD gamma | 8 |
| repeats | 20 |
| warmup_repeats | 2 |
| warmup_tokens | 0 |
| Edge GPU / verifier GPU | 0 / 1 |
| Model runner | v1 / v1 |
| Execution | `--no-enforce-eager` |
| Network latency | symmetric one-way request/response latency: 0, 10, 20, 50 ms |
| Bandwidth conditions | unlimited, 100 Mbps symmetric request/response bandwidth |
| Server token/s metric | mean request output token/s, excluding model startup and warmup repeats |

## Results: Unlimited Bandwidth

| One-way Latency | Target-only token/s | DSSD token/s | DSSD Speedup | Draft Acceptance | Avg Accepted Len / Round |
|---:|---:|---:|---:|---:|---:|
| 0ms | 105.66 | 306.18 | 2.90x | 81.99% | 6.56 |
| 10ms | 104.77 | 149.52 | 1.43x | 81.99% | 6.56 |
| 20ms | 103.98 | 105.53 | 1.01x | 81.99% | 6.56 |
| 50ms | 101.59 | 56.73 | 0.56x | 81.99% | 6.56 |

## Results: 100 Mbps Bandwidth

| One-way Latency | Target-only token/s | DSSD token/s | DSSD Speedup | Draft Acceptance | Avg Accepted Len / Round |
|---:|---:|---:|---:|---:|---:|
| 0ms | 105.64 | 285.21 | 2.70x | 81.99% | 6.56 |
| 10ms | 104.90 | 150.30 | 1.43x | 81.99% | 6.56 |
| 20ms | 103.88 | 106.59 | 1.03x | 81.99% | 6.56 |
| 50ms | 101.58 | 56.53 | 0.56x | 81.99% | 6.56 |

## Raw Results

| Condition | Target-only JSON | DSSD JSON |
|---|---|---|
| unlimited, 0ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat0ms-unlimited-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat0ms-unlimited-repeat20-20260426.json` |
| unlimited, 10ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat10ms-unlimited-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat10ms-unlimited-repeat20-20260426.json` |
| unlimited, 20ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat20ms-unlimited-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat20ms-unlimited-repeat20-20260426.json` |
| unlimited, 50ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat50ms-unlimited-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat50ms-unlimited-repeat20-20260426.json` |
| 100 Mbps, 0ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat0ms-100mbps-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat0ms-100mbps-repeat20-20260426.json` |
| 100 Mbps, 10ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat10ms-100mbps-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat10ms-100mbps-repeat20-20260426.json` |
| 100 Mbps, 20ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat20ms-100mbps-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat20ms-100mbps-repeat20-20260426.json` |
| 100 Mbps, 50ms | `experiment-2.3-target-only-opt-125m-opt-6.7b-high-accept-p128-o256-lat50ms-100mbps-repeat20-20260426.json` | `experiment-2.3-dssd-opt-125m-opt-6.7b-high-accept-p128-o256-g8-lat50ms-100mbps-repeat20-20260426.json` |

## Command Template

DSSD runs used `benchmarks/dssd/benchmark_edge_verifier_decode.py` helpers with:

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
  --gamma 8 \
  --request-latency-ms <0|10|20|50> \
  --response-latency-ms <0|10|20|50> \
  --json
```

For 100 Mbps runs, `--request-bandwidth-bytes-per-s 12500000` and
`--response-bandwidth-bytes-per-s 12500000` were also applied. Target-only runs
used the same verifier server construction, prompt tokens, sampling parameters,
HTTP transport, network calibration, and `/generate` path.

## Notes

DSSD throughput is highly sensitive to fixed per-round latency. With unlimited
bandwidth, DSSD speedup falls from 2.90x at 0ms one-way latency to 1.01x at
20ms, and becomes slower than target-only at 50ms. The measured draft acceptance
rate remains unchanged across network conditions because the prompt, model pair,
temperature, and `gamma` are fixed.

The 100 Mbps bandwidth limit has little impact in this greedy OPT experiment
because the request and response payloads are small: DSSD sends draft token ids
and scalar q values upstream, and greedy rejection returns a token id rather
than full logits. At 0ms, calibration warned that the target 100 Mbps transfer
time for small payloads was faster than the measured local HTTP overhead, so
the simulator skipped negative extra delay as designed.
