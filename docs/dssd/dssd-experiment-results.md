# DSSD Experiment Results

此文件用于持续记录 DSSD benchmark results。追加新的
experiments 时请按日期分节，并将原始 JSON result paths 保留在
summary table 旁边，以便将 results 追溯到 benchmark output。

## 2026-04-19: Single-Engine Eager Baseline

### Goal

测量当前 DSSD
single-engine paths 与 native vLLM baselines 在单独单请求场景下的 decode throughput：

- DSSD edge engine with Qwen3-0.6B.
- DSSD verifier engine with Qwen3-8B.
- Native vLLM engine with Qwen3-0.6B.
- Native vLLM engine with Qwen3-8B.

这里不衡量完整 edge+verifier DSSD 的 end-to-end performance。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- GPU used: `CUDA_VISIBLE_DEVICES=0`.
- Model paths:
  - `/root/autodl-tmp/Qwen3-0.6B`
  - `/root/autodl-tmp/Qwen3-8B`
- Benchmark script: `benchmarks/dssd/benchmark_local_engines.py`.
- Model runner: 所有 engines 均为 v1。
- Execution mode: eager，CUDA graph disabled。

### Fixed Parameters

| Parameter | Value |
|---|---:|
| `prompt_len` | 128 |
| `max_new_tokens` | 256 |
| `warmup_iters` | 2 |
| `benchmark_iters` | 5 |
| `max_model_len` | 416 |
| `gpu_memory_utilization` | 0.9 |
| `temperature` | 0.0 |
| `top_p` | 1.0 |
| `top_k` | 0 |
| EOS behavior | ignore EOS, generate full output length |

### Commands

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine edge \
  --model /root/autodl-tmp/Qwen3-0.6B \
  --edge-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/edge-qwen3-0.6b-p128-o256-v1-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine verifier \
  --model /root/autodl-tmp/Qwen3-8B \
  --verifier-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/verifier-qwen3-8b-p128-o256-v1-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/Qwen3-0.6B \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/native-qwen3-0.6b-p128-o256-v1-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/Qwen3-8B \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/native-qwen3-8b-p128-o256-v1-eager.json
```

### Raw Result Files

- `benchmarks/dssd/results/edge-qwen3-0.6b-p128-o256-v1-eager.json`
- `benchmarks/dssd/results/verifier-qwen3-8b-p128-o256-v1-eager.json`
- `benchmarks/dssd/results/native-qwen3-0.6b-p128-o256-v1-eager.json`
- `benchmarks/dssd/results/native-qwen3-8b-p128-o256-v1-eager.json`

### Results

| Engine | Model | Generated Tokens | Avg Seconds / Request | Total Seconds | Tokens / Second |
|---|---|---:|---:|---:|---:|
| DSSD edge | Qwen3-0.6B | 1280 | 4.1671 | 20.8357 | 61.43 |
| DSSD verifier | Qwen3-8B | 1280 | 5.4393 | 27.1967 | 47.06 |
| Native vLLM | Qwen3-0.6B | 1280 | 1.7309 | 8.6546 | 147.90 |
| Native vLLM | Qwen3-8B | 1280 | 2.8462 | 14.2308 | 89.95 |

### Derived Comparisons

| Comparison | Ratio |
|---|---:|
| DSSD edge Qwen3-0.6B / native Qwen3-0.6B | 0.415x |
| DSSD verifier Qwen3-8B / native Qwen3-8B | 0.523x |

### Notes

- DSSD single-engine paths 在 eager mode 下比 native vLLM baselines 更慢。
- edge Qwen3-0.6B path 的回退更明显：`61.43 tok/s`，而 native Qwen3-0.6B
  为 `147.90 tok/s`。
- verifier Qwen3-8B path 达到 `47.06 tok/s`，而 native Qwen3-8B 为
  `89.95 tok/s`。
- 下一步有价值的检查：
  - 对 edge Qwen3-0.6B 和 verifier Qwen3-8B 运行 `--phase-timing`。
  - 使用 `--no-enforce-eager` 重复这四组 baselines，以测量
    CUDA graph path。

## 2026-04-19: Single-Engine No-Eager Baseline

### Goal

在禁用 eager mode 的情况下重复 single-engine baseline，从而启用
torch.compile 和 CUDA graph capture。这衡量的是更接近常规
vLLM optimized decode 的 performance path。

这里使用与上方 eager baseline 相同的 models、GPU、prompt length、
output length、warmup count 和 benchmark iteration count。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- GPU used: `CUDA_VISIBLE_DEVICES=0`.
- Model paths:
  - `/root/autodl-tmp/Qwen3-0.6B`
  - `/root/autodl-tmp/Qwen3-8B`
- Benchmark script: `benchmarks/dssd/benchmark_local_engines.py`.
- Model runner: 所有 engines 均为 v1。
- Execution mode: `--no-enforce-eager`, with torch.compile and CUDA graph
  enabled by vLLM.

### Fixed Parameters

| Parameter | Value |
|---|---:|
| `prompt_len` | 128 |
| `max_new_tokens` | 256 |
| `warmup_iters` | 2 |
| `benchmark_iters` | 5 |
| `max_model_len` | 416 |
| `gpu_memory_utilization` | 0.9 |
| `temperature` | 0.0 |
| `top_p` | 1.0 |
| `top_k` | 0 |
| EOS behavior | ignore EOS, generate full output length |

### Commands

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine edge \
  --model /root/autodl-tmp/Qwen3-0.6B \
  --edge-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/edge-qwen3-0.6b-p128-o256-v1-no-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine verifier \
  --model /root/autodl-tmp/Qwen3-8B \
  --verifier-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/verifier-qwen3-8b-p128-o256-v1-no-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/Qwen3-0.6B \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/native-qwen3-0.6b-p128-o256-v1-no-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/Qwen3-8B \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/native-qwen3-8b-p128-o256-v1-no-eager.json
```

### Raw Result Files

- `benchmarks/dssd/results/edge-qwen3-0.6b-p128-o256-v1-no-eager.json`
- `benchmarks/dssd/results/verifier-qwen3-8b-p128-o256-v1-no-eager.json`
- `benchmarks/dssd/results/native-qwen3-0.6b-p128-o256-v1-no-eager.json`
- `benchmarks/dssd/results/native-qwen3-8b-p128-o256-v1-no-eager.json`

### Results

| Engine | Model | Generated Tokens | Avg Seconds / Request | Total Seconds | Tokens / Second |
|---|---|---:|---:|---:|---:|
| DSSD edge | Qwen3-0.6B | 1280 | 0.4779 | 2.3893 | 535.71 |
| DSSD verifier | Qwen3-8B | 1280 | 2.2680 | 11.3401 | 112.87 |
| Native vLLM | Qwen3-0.6B | 1280 | 0.4022 | 2.0108 | 636.58 |
| Native vLLM | Qwen3-8B | 1280 | 2.1884 | 10.9422 | 116.98 |

### Derived Comparisons

| Comparison | Ratio |
|---|---:|
| DSSD edge Qwen3-0.6B / native Qwen3-0.6B | 0.842x |
| DSSD verifier Qwen3-8B / native Qwen3-8B | 0.965x |

### No-Eager Speedup Over Eager

| Engine | Model | Speedup |
|---|---|---:|
| DSSD edge | Qwen3-0.6B | 8.72x |
| DSSD verifier | Qwen3-8B | 2.40x |
| Native vLLM | Qwen3-0.6B | 4.30x |
| Native vLLM | Qwen3-8B | 1.30x |

### Notes

- 禁用 eager mode 后，single-engine performance 基本恢复。
- DSSD edge Qwen3-0.6B 达到 `535.71 tok/s`，约为 native
  Qwen3-0.6B 的 `84.2%`。
- DSSD verifier Qwen3-8B 达到 `112.87 tok/s`，约为 native
  Qwen3-8B 的 `96.5%`。
- 当前主要剩余的 single-engine 差距在 edge Qwen3-0.6B path 上。
- 下一步有价值的检查：
  - 对 edge Qwen3-0.6B 运行 `--phase-timing --no-enforce-eager`，定位
    剩余差距。
  - 使用 `--no-enforce-eager` 运行完整的 edge+verifier DSSD benchmark。

## 2026-04-19: Native vLLM Speculative Decoding Smoke

### Goal

对 vLLM 的 native draft-model speculative decoding 运行一次 smoke benchmark，
使用 Qwen3-8B 作为 target model，Qwen3-0.6B 作为 draft model。

此实验使用 `vllm bench latency`，而不是
`benchmarks/dssd/benchmark_local_engines.py`。下方的 output tokens per second
按 `output_len / avg_latency` 计算。sampling configuration 使用
`vllm bench latency` 的默认值（`temperature=1.0`、`top_p=1.0`、
`ignore_eos=True`），因此它与前面 greedy single-engine baselines
并不是严格 apples-to-apples 的对比。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- GPU used: `CUDA_VISIBLE_DEVICES=0`.
- Target model: `/root/autodl-tmp/Qwen3-8B`.
- Draft model: `/root/autodl-tmp/Qwen3-0.6B`.
- Benchmark command: `vllm bench latency`.
- Execution mode: `--no-enforce-eager`, with torch.compile and CUDA graph
  enabled by vLLM.
- vLLM native speculative method: `draft_model`.

### Fixed Parameters

| Parameter | Value |
|---|---:|
| `input_len` | 128 |
| `output_len` | 256 |
| `batch_size` | 1 |
| `num_iters_warmup` | 2 |
| `num_iters` | 5 |
| `max_model_len` | 416 |
| `gpu_memory_utilization` | 0.9 |
| `num_speculative_tokens` | 4 |
| `draft_tensor_parallel_size` | 1 |
| Detokenization | disabled |

### Command

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
.venv/bin/python -m vllm.entrypoints.cli.main bench latency \
  --model /root/autodl-tmp/Qwen3-8B \
  --speculative-config '{"model":"/root/autodl-tmp/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":4,"draft_tensor_parallel_size":1}' \
  --input-len 128 \
  --output-len 256 \
  --batch-size 1 \
  --num-iters-warmup 2 \
  --num-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --disable-detokenize \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k4-v1-no-eager.json
```

### Raw Result File

- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k4-v1-no-eager.json`

Note: 这个 JSON path 后来被下方的 gamma sweep 覆盖了。本节中的 smoke
numbers 保留自原始 console output。

### Results

| Metric | Value |
|---|---:|
| Average latency | 3.2788 s |
| Median latency | 2.9328 s |
| P90 latency | 4.0941 s |
| Output tokens / second | 78.08 |

### Latencies

| Iteration | Latency |
|---:|---:|
| 1 | 2.7705 s |
| 2 | 2.9328 s |
| 3 | 3.2117 s |
| 4 | 2.7964 s |
| 5 | 4.6824 s |

### Runtime Speculative Metrics From Logs

benchmark log 在运行期间输出了 speculative decoding metrics：

- Mean acceptance length: about `2.25` to `2.26`.
- Average draft acceptance rate: about `31.4%` to `31.5%`.
- Per-position acceptance rate snapshots:
  - `0.567, 0.375, 0.190, 0.129`
  - `0.499, 0.343, 0.241, 0.172`

这些从 log 推导出的 metrics 对 smoke validation 很有用，但并未写入
latency JSON file。

### Notes

- 此次运行成功完成，并加载了 target 和 draft models。
- Native vLLM 为 draft-model speculative decoding 禁用了 async scheduling。
- 仅 output 的 throughput 为 `78.08 tok/s`，低于先前 native
  Qwen3-8B no-eager single-engine baseline 的 `116.98 tok/s`。
- 对 `k=4` 而言 acceptance rate 相对偏低，因此这个 model pair 在下结论前
  可能需要先做 `num_speculative_tokens` sweep。

## 2026-04-19: Native vLLM Speculative Decoding Gamma Sweep

### Goal

对 Qwen3-8B target 与 Qwen3-0.6B draft model pair 的
vLLM native draft-model speculative decoding length 做 sweep。

此实验使用 `vllm bench latency`，配置与 smoke run 相同。
output tokens per second 按 `output_len / avg_latency` 计算。Acceptance
statistics 来自 benchmark phase 期间周期性输出的 `SpecDecoding metrics`
log lines，因为 latency JSON files 不包含 speculative acceptance metrics。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- GPU used: `CUDA_VISIBLE_DEVICES=0`.
- Target model: `/root/autodl-tmp/Qwen3-8B`.
- Draft model: `/root/autodl-tmp/Qwen3-0.6B`.
- Benchmark command: `vllm bench latency`.
- Execution mode: `--no-enforce-eager`, with torch.compile and CUDA graph
  enabled by vLLM.
- vLLM native speculative method: `draft_model`.
- 设置 `VLLM_LOG_STATS_INTERVAL=1` 以更频繁输出 acceptance metrics。

### Fixed Parameters

| Parameter | Value |
|---|---:|
| `input_len` | 128 |
| `output_len` | 256 |
| `batch_size` | 1 |
| `num_iters_warmup` | 2 |
| `num_iters` | 5 |
| `max_model_len` | 416 |
| `gpu_memory_utilization` | 0.9 |
| `draft_tensor_parallel_size` | 1 |
| Detokenization | disabled |

### Command

```bash
for k in 1 2 3 4 6 8; do
  CUDA_VISIBLE_DEVICES=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_USE_V2_MODEL_RUNNER=0 \
  VLLM_LOG_STATS_INTERVAL=1 \
  .venv/bin/python -m vllm.entrypoints.cli.main bench latency \
    --model /root/autodl-tmp/Qwen3-8B \
    --speculative-config "{\"model\":\"/root/autodl-tmp/Qwen3-0.6B\",\"method\":\"draft_model\",\"num_speculative_tokens\":${k},\"draft_tensor_parallel_size\":1}" \
    --input-len 128 \
    --output-len 256 \
    --batch-size 1 \
    --num-iters-warmup 2 \
    --num-iters 5 \
    --max-model-len 416 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --disable-detokenize \
    --no-enforce-eager \
    --output-json "benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k${k}-v1-no-eager.json" \
    2>&1 | tee "benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k${k}-v1-no-eager.log"
done
```

### Raw Result Files

- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k1-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k1-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k2-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k2-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k3-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k3-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k4-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k4-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k6-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k6-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k8-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-qwen3-8b-draft-qwen3-0.6b-p128-o256-k8-v1-no-eager.log`

### Results

| k | Avg Latency | Median Latency | P90 Latency | Output Tokens / Second | Accepted / Drafted | Aggregate Draft Acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.4076 s | 2.3633 s | 2.6261 s | 106.33 | 416 / 869 | 47.9% |
| 2 | 1.9557 s | 1.7590 s | 2.4170 s | 130.90 | 693 / 1280 | 54.1% |
| 3 | 3.7947 s | 3.0025 s | 5.2527 s | 67.46 | 534 / 2517 | 21.2% |
| 4 | 3.3997 s | 3.0420 s | 4.1716 s | 75.30 | 692 / 2392 | 28.9% |
| 6 | 5.6149 s | 4.7627 s | 8.0541 s | 45.59 | 575 / 4170 | 13.8% |
| 8 | 7.0825 s | 6.2736 s | 10.3869 s | 36.15 | 583 / 5344 | 10.9% |

### Log-Derived Acceptance Length

| k | Drafted-Token-Weighted Mean Acceptance Length |
|---:|---:|
| 1 | 1.48 |
| 2 | 2.08 |
| 3 | 1.64 |
| 4 | 2.16 |
| 6 | 1.83 |
| 8 | 1.87 |

### Notes

- `k=2` 是本次 sweep 中最快的点：`130.90 tok/s`。
- `k=2` 快于先前的 native Qwen3-8B no-eager baseline
  （`116.98 tok/s`），但 benchmark sampling path 不同，因此这不是
  严格 apples-to-apples 的对比。
- `k=3` 和 `k=4` 更慢，因为新增的 draft work 没有被足够多的 accepted
  tokens 抵消。
- `k=6` 和 `k=8` 进一步退化。draft workload 增长很多，但 aggregate
  acceptance rate 分别下降到约 `13.8%` 和 `10.9%`。
- 这里的 Acceptance metrics 是从周期性 log windows 中解析并在
  benchmark phase 上聚合得到的。若要获得更干净的 acceptance-rate 记录，
  可用 `vllm serve` 加 `vllm bench serve` 重新运行最佳 candidate，
  这样会把 speculative metrics 写入 JSON。
- 后续测试的最佳 candidate 是 `k=2`。

## 2026-04-19: DSSD End-to-End Gamma Sweep

### Goal

对与上方 native vLLM speculative sweep 相同的 Qwen3-8B verifier
和 Qwen3-0.6B edge model pair 做 DSSD end-to-end speculative decode length
sweep。

此次运行使用更新后的 DSSD benchmark，记录以下内容：

- 包含 prefill 的完整 request timing：`request_*`
- 排除 prefill/bootstrap 的 decode-only timing：`decode_*`
- 用于诊断的 verifier-path-only timing：`verify_path_*`

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- Edge GPU: `CUDA_VISIBLE_DEVICES=0`.
- Verifier GPU: `CUDA_VISIBLE_DEVICES=1`.
- Edge model: `/root/autodl-tmp/Qwen3-0.6B`.
- Verifier model: `/root/autodl-tmp/Qwen3-8B`.
- Benchmark script: `benchmarks/dssd/benchmark_edge_verifier_decode.py`.
- Model runner: edge 和 verifier 均为 v1。
- Execution mode: `--no-enforce-eager`, with torch.compile and CUDA graph
  enabled by vLLM.

### Fixed Parameters

| Parameter | Value |
|---|---:|
| `prompt_len` | 128 |
| `decode_tokens` | 256 |
| `warmup_tokens` | 0 |
| `warmup_repeats` | 2 |
| `repeats` | 5 |
| `max_model_len` | 416 |
| `gpu_memory_utilization` | 0.9 |
| `max_num_batched_tokens` | 128 |
| `max_num_seqs` | 2 |
| Network simulation | disabled |

### Command

```bash
for k in 1 2 3 4 6 8; do
  PYTHONPATH=$PWD \
  .venv/bin/python benchmarks/dssd/benchmark_edge_verifier_decode.py \
    --model /root/autodl-tmp/Qwen3-8B \
    --edge-model /root/autodl-tmp/Qwen3-0.6B \
    --verifier-model /root/autodl-tmp/Qwen3-8B \
    --edge-model-runner v1 \
    --verifier-model-runner v1 \
    --edge-cuda-visible-devices 0 \
    --verifier-cuda-visible-devices 1 \
    --prompt-len 128 \
    --decode-tokens 256 \
    --warmup-tokens 0 \
    --warmup-repeats 2 \
    --repeats 5 \
    --gamma "${k}" \
    --max-model-len 416 \
    --gpu-memory-utilization 0.9 \
    --no-enforce-eager \
    --json \
    2>&1 | tee "benchmarks/dssd/results/dssd-qwen3-0.6b-qwen3-8b-p128-o256-k${k}-v1-no-eager.log"
done
```

### Raw Result Files

- `benchmarks/dssd/results/dssd-qwen3-0.6b-qwen3-8b-p128-o256-k1-v1-no-eager.log`
- `benchmarks/dssd/results/dssd-qwen3-0.6b-qwen3-8b-p128-o256-k2-v1-no-eager.log`
- `benchmarks/dssd/results/dssd-qwen3-0.6b-qwen3-8b-p128-o256-k3-v1-no-eager.log`
- `benchmarks/dssd/results/dssd-qwen3-0.6b-qwen3-8b-p128-o256-k4-v1-no-eager.log`
- `benchmarks/dssd/results/dssd-qwen3-0.6b-qwen3-8b-p128-o256-k6-v1-no-eager.log`
- `benchmarks/dssd/results/dssd-qwen3-0.6b-qwen3-8b-p128-o256-k8-v1-no-eager.log`

### Results

| gamma | Request Total | Request Tokens / Second | Decode Total | Decode Tokens / Second | Verify-Path Tokens / Second | Draft Acceptance | Avg Accepted Len / Round |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.7803 s | 67.75 | 3.7279 s | 68.70 | 81.07 | 33.3% | 0.333 |
| 2 | 3.6087 s | 71.16 | 3.5542 s | 72.25 | 96.28 | 33.8% | 0.675 |
| 3 | 3.6313 s | 71.21 | 3.5819 s | 72.21 | 108.22 | 30.5% | 0.916 |
| 4 | 3.0877 s | 83.02 | 3.0377 s | 84.39 | 135.64 | 33.9% | 1.358 |
| 6 | 3.5682 s | 83.28 | 3.5272 s | 84.45 | 160.11 | 27.7% | 1.664 |
| 8 | 2.7795 s | 93.09 | 2.7378 s | 94.54 | 197.86 | 32.2% | 2.576 |

### Notes

- `gamma=8` 是这次扩展 DSSD sweep 中最快的点，无论是完整
  request timing（`93.09 tok/s`）还是 decode-only timing（`94.54 tok/s`）。
- 在这个本地双 GPU DSSD setup 中，更高的 `gamma` 一直到 `8` 都在持续提升
  throughput，这与上方 native 单进程 speculative sweep 不同，后者最佳点是
  `k=2`。
- 相对于上方的 native vLLM speculative sweep：
  - DSSD `gamma=8` 慢于 native 最佳 `k=2`（`130.90 tok/s`）。
  - 在 full-request throughput 上，DSSD `gamma=8` 快于 native `k=3`
    （`67.46 tok/s`）、`k=4`（`75.30 tok/s`）、`k=6`（`45.59 tok/s`）
    和 `k=8`（`36.15 tok/s`）。
- 这些并不是严格 apples-to-apples 的硬件对比：
  - 上方 native sweep 使用了一张 GPU
  - 本次 DSSD sweep 为 edge 和 verifier 各使用一张 GPU
- `verify_path_tokens/s` 随 `gamma` 明显上升，这表明在更长的 draft lengths
  下 verifier round overhead 被更好地摊薄了。
- `gamma=6` 和 `gamma=8` 仍然偶尔会出现较差的 repeats，但均值趋势仍在上升，
  因为成功的 rounds 相比低 gamma 设置更能摊薄 verifier overhead。

## Future Experiments

请在此标题下方追加新的 experiment sections。使用相同结构：
goal、environment、fixed parameters、commands、raw result files、results
和 notes。
