# DSSD OPT Experiment Results

该文件用于持续记录基于 OPT 的 DSSD benchmark 结果。新增实验时，请按日期追加新节，并将原始 JSON 或 log 路径保留在各汇总表旁边，以便将结果追溯到 benchmark 输出。

## 2026-04-19: Single-Engine Eager Baseline

### Goal

测量当前 DSSD 单引擎路径与原生 vLLM baseline 的独立单请求 decode 吞吐：

- DSSD edge engine with OPT-125m.
- DSSD verifier engine with OPT-6.7b.
- Native vLLM engine with OPT-125m.
- Native vLLM engine with OPT-6.7b.

这不衡量完整 edge+verifier 的 DSSD 端到端性能。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- 使用的 GPU: `CUDA_VISIBLE_DEVICES=0`。
- Model paths:
  - `/root/autodl-tmp/opt-125m`
  - `/root/autodl-tmp/opt-6.7b`
- Benchmark script: `benchmarks/dssd/benchmark_local_engines.py`。
- 所有 engine 均使用 v1 Model runner。
- Execution mode: eager，禁用 CUDA graph。

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
  --model /root/autodl-tmp/opt-125m \
  --edge-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/edge-opt-125m-p128-o256-v1-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine verifier \
  --model /root/autodl-tmp/opt-6.7b \
  --verifier-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/verifier-opt-6.7b-p128-o256-v1-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/opt-125m \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/native-opt-125m-p128-o256-v1-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/opt-6.7b \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --output-json benchmarks/dssd/results/native-opt-6.7b-p128-o256-v1-eager.json
```

### Raw Result Files

- `benchmarks/dssd/results/edge-opt-125m-p128-o256-v1-eager.json`
- `benchmarks/dssd/results/verifier-opt-6.7b-p128-o256-v1-eager.json`
- `benchmarks/dssd/results/native-opt-125m-p128-o256-v1-eager.json`
- `benchmarks/dssd/results/native-opt-6.7b-p128-o256-v1-eager.json`

### Results

| Engine | Model | Generated Tokens | Avg Seconds / Request | Total Seconds | Tokens / Second |
|---|---|---:|---:|---:|---:|
| DSSD edge | OPT-125m | 1280 | 0.7354 | 3.6771 | 348.10 |
| DSSD verifier | OPT-6.7b | 1280 | 2.7225 | 13.6125 | 94.03 |
| Native vLLM | OPT-125m | 1280 | 0.8303 | 4.1513 | 308.34 |
| Native vLLM | OPT-6.7b | 1280 | 2.5861 | 12.9306 | 98.99 |

### Derived Comparisons

| Comparison | Ratio |
|---|---:|
| DSSD edge OPT-125m / native OPT-125m | 1.129x |
| DSSD verifier OPT-6.7b / native OPT-6.7b | 0.950x |

### Notes

- 在 eager mode 下，DSSD edge OPT-125m 路径已经快于 native OPT-125m。
- verifier OPT-6.7b 路径接近 native OPT-6.7b，但仍然略慢。
- 这与更早的 Qwen3 配对呈现出不同模式；在那组实验中，eager mode 对 DSSD 路径的惩罚明显更强。

## 2026-04-19: Single-Engine No-Eager Baseline

### Goal

在禁用 eager mode 的情况下重复单引擎 baseline，允许 torch.compile 和 CUDA graph capture。这样测得的性能路径更接近正常的 vLLM 优化 decode。

这里使用与上方 eager baseline 相同的模型、GPU、prompt length、output length、warmup 次数和 benchmark 迭代次数。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- 使用的 GPU: `CUDA_VISIBLE_DEVICES=0`。
- Model paths:
  - `/root/autodl-tmp/opt-125m`
  - `/root/autodl-tmp/opt-6.7b`
- Benchmark script: `benchmarks/dssd/benchmark_local_engines.py`。
- 所有 engine 均使用 v1 Model runner。
- Execution mode: `--no-enforce-eager`，由 vLLM 启用 torch.compile 和 CUDA graph。

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
  --model /root/autodl-tmp/opt-125m \
  --edge-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/edge-opt-125m-p128-o256-v1-no-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine verifier \
  --model /root/autodl-tmp/opt-6.7b \
  --verifier-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/verifier-opt-6.7b-p128-o256-v1-no-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/opt-125m \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/native-opt-125m-p128-o256-v1-no-eager.json
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/dssd/benchmark_local_engines.py \
  --engine native \
  --model /root/autodl-tmp/opt-6.7b \
  --native-model-runner v1 \
  --prompt-len 128 \
  --max-new-tokens 256 \
  --warmup-iters 2 \
  --benchmark-iters 5 \
  --max-model-len 416 \
  --gpu-memory-utilization 0.9 \
  --no-enforce-eager \
  --output-json benchmarks/dssd/results/native-opt-6.7b-p128-o256-v1-no-eager.json
```

### Raw Result Files

- `benchmarks/dssd/results/edge-opt-125m-p128-o256-v1-no-eager.json`
- `benchmarks/dssd/results/verifier-opt-6.7b-p128-o256-v1-no-eager.json`
- `benchmarks/dssd/results/native-opt-125m-p128-o256-v1-no-eager.json`
- `benchmarks/dssd/results/native-opt-6.7b-p128-o256-v1-no-eager.json`

### Results

| Engine | Model | Generated Tokens | Avg Seconds / Request | Total Seconds | Tokens / Second |
|---|---|---:|---:|---:|---:|
| DSSD edge | OPT-125m | 1280 | 0.1481 | 0.7404 | 1728.68 |
| DSSD verifier | OPT-6.7b | 1280 | 1.7646 | 8.8231 | 145.07 |
| Native vLLM | OPT-125m | 1280 | 0.1716 | 0.8580 | 1491.90 |
| Native vLLM | OPT-6.7b | 1280 | 1.6865 | 8.4324 | 151.80 |

### Derived Comparisons

| Comparison | Ratio |
|---|---:|
| DSSD edge OPT-125m / native OPT-125m | 1.159x |
| DSSD verifier OPT-6.7b / native OPT-6.7b | 0.956x |

### No-Eager Speedup Over Eager

| Engine | Model | Speedup |
|---|---|---:|
| DSSD edge | OPT-125m | 4.97x |
| DSSD verifier | OPT-6.7b | 1.54x |
| Native vLLM | OPT-125m | 4.84x |
| Native vLLM | OPT-6.7b | 1.53x |

### Notes

- 禁用 eager mode 后，各条路径都有提升，但相对排序与 eager mode 基本保持一致。
- DSSD edge OPT-125m 路径仍然快于 native OPT-125m。
- DSSD verifier OPT-6.7b 路径仍然略慢于 native OPT-6.7b，但差距很小。

## 2026-04-19: Native vLLM Speculative Decoding Smoke

### Goal

对 vLLM 原生 draft-model speculative decoding 做一次 smoke benchmark，其中 OPT-6.7b 作为 target model，OPT-125m 作为 draft model。

该实验使用 `vllm bench latency`，而不是 `benchmarks/dssd/benchmark_local_engines.py`。下方的 output tokens per second 通过 `output_len / avg_latency` 推导得到。采样配置采用 `vllm bench latency` 的默认值，因此它与前面的 greedy 单引擎 baseline 并不是严格的一一对应对比。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- 使用的 GPU: `CUDA_VISIBLE_DEVICES=0`。
- Target model: `/root/autodl-tmp/opt-6.7b`。
- Draft model: `/root/autodl-tmp/opt-125m`。
- Benchmark command: `vllm bench latency`。
- Execution mode: `--no-enforce-eager`，由 vLLM 启用 torch.compile 和 CUDA graph。
- vLLM native speculative method: `draft_model`。

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
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_LOG_STATS_INTERVAL=1 \
.venv/bin/python -m vllm.entrypoints.cli.main bench latency \
  --model /root/autodl-tmp/opt-6.7b \
  --speculative-config '{"model":"/root/autodl-tmp/opt-125m","method":"draft_model","num_speculative_tokens":4,"draft_tensor_parallel_size":1}' \
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
  --output-json benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k4-v1-no-eager-smoke.json \
  2>&1 | tee benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k4-v1-no-eager-smoke.log
```

### Raw Result Files

- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k4-v1-no-eager-smoke.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k4-v1-no-eager-smoke.log`

### Results

| Metric | Value |
|---|---:|
| Average latency | 2.9723 s |
| Median latency | 3.1052 s |
| P90 latency | 3.1390 s |
| Output tokens / second | 86.13 |

### Log-Derived Speculative Metrics

smoke log 在运行期间周期性输出 `SpecDecoding metrics` 窗口。对这些输出窗口聚合后得到：

- Accepted / drafted tokens: `115 / 6364`
- Aggregate draft acceptance: `1.8%`
- Drafted-token-weighted mean acceptance length: `1.07`

### Notes

- 本次运行成功完成，并加载了 target 和 draft 两个模型。
- Native vLLM 为 draft-model speculative decoding 禁用了 async scheduling。
- 在 `k=4` 时 Acceptance 已经很低，因此这组模型需要先做一次 gamma sweep 才能下结论。

## 2026-04-19: Native vLLM Speculative Decoding Gamma Sweep

### Goal

对 OPT-6.7b target 与 OPT-125m draft 这一模型配对，扫描 vLLM 原生 draft-model speculative decoding length。

该实验使用 `vllm bench latency`，配置与 smoke run 相同。output tokens per second 通过 `output_len / avg_latency` 推导得到。Acceptance 统计是从周期性输出的 `SpecDecoding metrics` log 窗口中解析出来的，因为 latency JSON 文件不包含 speculative metrics。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- 使用的 GPU: `CUDA_VISIBLE_DEVICES=0`。
- Target model: `/root/autodl-tmp/opt-6.7b`。
- Draft model: `/root/autodl-tmp/opt-125m`。
- Benchmark command: `vllm bench latency`。
- Execution mode: `--no-enforce-eager`，由 vLLM 启用 torch.compile 和 CUDA graph。
- vLLM native speculative method: `draft_model`。
- `VLLM_LOG_STATS_INTERVAL=1`，用于更频繁地输出 acceptance metrics。

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
    --model /root/autodl-tmp/opt-6.7b \
    --speculative-config "{\"model\":\"/root/autodl-tmp/opt-125m\",\"method\":\"draft_model\",\"num_speculative_tokens\":${k},\"draft_tensor_parallel_size\":1}" \
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
    --output-json "benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k${k}-v1-no-eager.json" \
    2>&1 | tee "benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k${k}-v1-no-eager.log"
done
```

### Raw Result Files

- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k1-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k1-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k2-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k2-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k3-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k3-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k4-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k4-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k6-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k6-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k8-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-p128-o256-k8-v1-no-eager.log`

### Results

| k | Avg Latency | Median Latency | P90 Latency | Output Tokens / Second | Accepted / Drafted | Aggregate Draft Acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.4965 s | 2.6269 s | 2.6695 s | 102.54 | 130 / 1586 | 8.2% |
| 2 | 2.6794 s | 2.8272 s | 2.8366 s | 95.54 | 167 / 3008 | 5.6% |
| 3 | 2.8330 s | 2.9153 s | 2.9716 s | 90.36 | 111 / 4788 | 2.3% |
| 4 | 2.9941 s | 3.1282 s | 3.1712 s | 85.50 | 104 / 6328 | 1.6% |
| 6 | 3.5923 s | 3.7294 s | 3.7880 s | 71.26 | 101 / 9492 | 1.1% |
| 8 | 4.6664 s | 4.8704 s | 4.9251 s | 54.86 | 129 / 13000 | 1.0% |

### Log-Derived Acceptance Length

| k | Drafted-Token-Weighted Mean Acceptance Length |
|---:|---:|
| 1 | 1.08 |
| 2 | 1.11 |
| 3 | 1.07 |
| 4 | 1.07 |
| 6 | 1.06 |
| 8 | 1.08 |

### Notes

- `k=1` 是这次 sweep 中最快的点：`102.54 tok/s`。
- 与更早的 Qwen3 配对不同，这组 OPT 模型不会从更大的 speculative length 中受益。随着 `k` 增加，吞吐单调下降。
- 即使在 `k=1` 时 Acceptance 也偏低，到 `k=4` 时已经降到 `2%` 以下。
- mean acceptance length 一直维持在接近 `1.0`，这意味着大多数成功的 speculative round 只拿回 verifier token，几乎无法摊薄 draft 侧工作。
- 对这组模型而言，native speculative decoding 仍然慢于 native target-only OPT-6.7b no-eager decode（`151.80 tok/s`）。

## 2026-04-19: High-Acceptance Prompt Sweep

### Goal

验证前面 OPT 配对的低 acceptance 是否主要来自 prompt choice。

这次实验改用手工挑选的高接受率连续文本，只使用
`benchmarks/dssd/prompts/high_acceptance_opt.jsonl` 中第一条 prompt，并统一：

- 用 `/root/autodl-tmp/opt-125m` tokenizer 编码。
- 截断到前 `128` 个 prompt token。
- 对 native speculative 和 DSSD 都重复同一条 prompt 做 warmup 和 timed repeats。

这样可以排除“native 与 DSSD 实际吃到的 prompt 不同”这一因素。

### Environment

- GPUs: NVIDIA GeForce RTX 5090 x2，每张 32 GiB。
- Native speculative GPU: `CUDA_VISIBLE_DEVICES=0`。
- DSSD edge GPU: `CUDA_VISIBLE_DEVICES=0`。
- DSSD verifier GPU: `CUDA_VISIBLE_DEVICES=1`。
- Target / verifier model: `/root/autodl-tmp/opt-6.7b`。
- Draft / edge model: `/root/autodl-tmp/opt-125m`。
- Prompt file: `benchmarks/dssd/prompts/high_acceptance_opt.jsonl`。
- Source prompt token count before truncation: `201`。
- Execution mode: `--no-enforce-eager`。
- Model runner: v1。

### Fixed Parameters

| Parameter | Value |
|---|---:|
| `prompt_len` | 128 |
| `output_len` / `decode_tokens` | 256 |
| `warmup_iters` / `warmup_repeats` | 2 |
| `benchmark_iters` / `repeats` | 5 |
| `max_model_len` | 416 |
| `gpu_memory_utilization` | 0.9 |
| `temperature` | 0.0 |
| `top_p` | 1.0 |
| `top_k` | 0 |
| EOS behavior | ignore EOS, generate full output length |

### Commands

Native speculative 采用离线 `vllm.LLM.generate()` 路径，而不是 `vllm bench latency` 或
`vllm bench serve`，原因有两点：

- 需要精确复用与 DSSD 相同的 `prompt_token_ids[:128]`。
- 需要直接从 `llm.get_metrics()` 读取 acceptance counters。

执行方式是对 `k in {1,2,3,4,6,8}` 循环运行一段 inline Python benchmark：

```bash
for k in 1 2 3 4 6 8; do
  CUDA_VISIBLE_DEVICES=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_USE_V2_MODEL_RUNNER=0 \
  .venv/bin/python - "$k" "benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k${k}-v1-no-eager.json" <<'PY'
# 读取 high_acceptance_opt.jsonl 的第一条 prompt
# 用 /root/autodl-tmp/opt-125m tokenizer 编码并截断到 128 tokens
# 构造 LLM(model=/root/autodl-tmp/opt-6.7b, speculative_config={draft_model=/root/autodl-tmp/opt-125m, num_speculative_tokens=k})
# warmup 2 次，benchmark 5 次
# 记录单请求 latency、output tokens / second，以及 llm.get_metrics() 的 acceptance counter 增量
PY
done
```

DSSD 端继续使用 `benchmark_edge_verifier_decode.py`，但通过 `--prompt-jsonl`
喂入同一条高接受率 prompt。下列命令和结果对应的是 correctness fix 之后的 rerun；
此前错误的 DSSD 结果已删除，不再保留：

```bash
for k in 1 2 3 4 6 8; do
  PYTHONPATH=$PWD \
  .venv/bin/python benchmarks/dssd/benchmark_edge_verifier_decode.py \
    --model /root/autodl-tmp/opt-6.7b \
    --edge-model /root/autodl-tmp/opt-125m \
    --verifier-model /root/autodl-tmp/opt-6.7b \
    --edge-model-runner v1 \
    --verifier-model-runner v1 \
    --edge-cuda-visible-devices 0 \
    --verifier-cuda-visible-devices 1 \
    --prompt-len 128 \
    --prompt-jsonl benchmarks/dssd/prompts/high_acceptance_opt.jsonl \
    --decode-tokens 256 \
    --warmup-tokens 0 \
    --warmup-repeats 2 \
    --repeats 5 \
    --gamma "${k}" \
    --max-model-len 416 \
    --gpu-memory-utilization 0.9 \
    --no-enforce-eager \
    --json \
    2>&1 | tee "benchmarks/dssd/results/dssd-opt-125m-opt-6.7b-high-accept-p128-o256-k${k}-v1-no-eager-after-correctness-fix.log"
done
```

### Raw Result Files

- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k1-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k1-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k2-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k2-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k3-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k3-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k4-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k4-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k6-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k6-v1-no-eager.log`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k8-v1-no-eager.json`
- `benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k8-v1-no-eager.log`
- `benchmarks/dssd/results/dssd-opt-125m-opt-6.7b-high-accept-p128-o256-k1-v1-no-eager-after-correctness-fix.log`
- `benchmarks/dssd/results/dssd-opt-125m-opt-6.7b-high-accept-p128-o256-k2-v1-no-eager-after-correctness-fix.log`
- `benchmarks/dssd/results/dssd-opt-125m-opt-6.7b-high-accept-p128-o256-k3-v1-no-eager-after-correctness-fix.log`
- `benchmarks/dssd/results/dssd-opt-125m-opt-6.7b-high-accept-p128-o256-k4-v1-no-eager-after-correctness-fix.log`
- `benchmarks/dssd/results/dssd-opt-125m-opt-6.7b-high-accept-p128-o256-k6-v1-no-eager-after-correctness-fix.log`
- `benchmarks/dssd/results/dssd-opt-125m-opt-6.7b-high-accept-p128-o256-k8-v1-no-eager-after-correctness-fix.log`

### Native Speculative Results

| k | Avg Latency | Output Tokens / Second | Accepted / Drafted | Aggregate Draft Acceptance | Mean Acceptance Length |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4046 s | 182.26 | 620 / 660 | 93.9% | 1.94 |
| 2 | 0.9810 s | 260.95 | 845 / 880 | 96.0% | 2.92 |
| 3 | 0.8270 s | 309.56 | 935 / 1035 | 90.3% | 3.71 |
| 4 | 0.7093 s | 360.93 | 1000 / 1120 | 89.3% | 4.57 |
| 6 | 0.7107 s | 360.24 | 1085 / 1290 | 84.1% | 6.05 |
| 8 | 0.7118 s | 359.67 | 1115 / 1360 | 82.0% | 7.56 |

### DSSD Results

| gamma | Request Tokens / Second | Decode Tokens / Second | Verify-Path Tokens / Second | Draft Acceptance | All-Accept Round Rate | Avg Accepted Len / Round |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104.98 | 106.53 | 141.23 | 93.9% | 93.9% | 0.939 |
| 2 | 137.53 | 140.16 | 208.76 | 96.0% | 94.3% | 1.920 |
| 3 | 160.47 | 163.98 | 270.26 | 90.3% | 85.5% | 2.710 |
| 4 | 199.58 | 203.87 | 341.35 | 89.3% | 83.9% | 3.571 |
| 6 | 255.86 | 262.20 | 443.51 | 84.1% | 76.7% | 5.047 |
| 8 | 300.91 | 309.12 | 560.46 | 82.0% | 70.6% | 6.559 |

### Derived Comparisons

| Comparison | Value |
|---|---:|
| Native speculative best point | `k=4`, `360.93 tok/s` |
| DSSD best full-request point | `gamma=8`, `300.91 tok/s` |
| DSSD best decode-only point | `gamma=8`, `309.12 tok/s` |
| Native target-only OPT-6.7b no-eager baseline | `151.80 tok/s` |
| Native speculative best / native target-only | `2.38x` |
| DSSD best request throughput / native target-only | `1.98x` |

### Notes

- 高接受率 prompt 对 native draft-model speculative decoding 的作用非常明显。OPT-125m + OPT-6.7b 在这条 prompt 上的 aggregate draft acceptance 达到 `82%` 到 `96%`，最佳点 `k=4` 为 `360.93 tok/s`。
- 相比前面随机 token / 低接受率 prompt 的 native sweep，这说明模型配对本身并不是问题；至少在这条文本上，OPT draft 与 target 的一致性足够高。
- correctness fix 之后，DSSD 的 acceptance 已经与 native 对齐到同一量级：`gamma=1..8` 分别对应 `93.9% / 96.0% / 90.3% / 89.3% / 84.1% / 82.0%`，和上方 native speculative sweep 基本一致。
- 这说明先前的异常低 acceptance 不是模型配对问题，也不是 prompt mismatch，而是 DSSD 实现中的 correctness bug。
- 在这组高接受率 prompt 上，当前 DSSD 吞吐会随着 `gamma` 增大持续上升；这次 sweep 的最快点是 `gamma=8`，full-request `300.91 tok/s`，decode-only `309.12 tok/s`。
- 与 native speculative 最佳点 `k=4` 的 `360.93 tok/s` 相比，当前 DSSD 仍然更慢，但差距已经缩小到同一量级。
- 与 native target-only OPT-6.7b no-eager baseline（`151.80 tok/s`）相比，当前 DSSD 在这条高接受率 prompt 上已经明显更快；不过这个对比不是严格同成本比较，因为 DSSD 使用了 edge + verifier 双 GPU，而 native target-only 只使用单 GPU。

## Future Experiments

请在该标题下方追加新的实验节。使用相同结构：goal、environment、fixed parameters、commands、raw result files、results 和 notes。
