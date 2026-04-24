# DSSD 功能介绍与使用说明

本文档说明当前 DSSD 系统已经支持的主要功能，以及如何启动服务、使用前端页面进行测试。本文只介绍功能和用法，不展开内部实现细节。

## 功能概览

当前系统支持以下功能：

- **DSSD 分布式投机采样**：edge 侧运行 draft 模型，verifier 侧运行 target 模型。edge 生成草稿 token，verifier 验证后返回接受长度、bonus token 或拒绝位置的 target 信息。
- **Target-only 基线模式**：edge 直接请求 verifier 上的 target 模型完成生成，用于和 DSSD 模式对比延迟、吞吐和输出效果。
- **Web benchmark 页面**：edge server 提供浏览器页面，可以输入 prompt、选择推理模式、设置采样参数并查看结果。
- **服务端推理指标**：除浏览器端到端耗时外，页面还展示 edge server 内部统计的服务端推理耗时和 token/s。
- **DSSD 接受率指标**：DSSD 模式会展示 draft acceptance rate、all-accept round rate、平均每轮接受长度等指标。
- **网络模拟**：前端可以设置 edge 和 verifier 之间的模拟延迟与带宽，用于观察通信条件变化对 DSSD 和 target-only 的影响。
- **EOS 自动推断**：edge server 默认从 edge 模型 tokenizer 推断 `eos_token_id`，必要时也可以通过 `--eos-token-id` 手动覆盖。
- **HTTP 服务接口**：edge 和 verifier 均提供 HTTP 接口，支持独立部署和远程通信。

## 推理模式

### DSSD 模式

DSSD 模式用于测试分布式投机采样流程：

1. edge 使用小模型生成 draft tokens。
2. edge 将 draft tokens 和对应的概率值发送给 verifier。
3. verifier 使用 target 模型验证 draft tokens。
4. edge 根据 verifier 返回结果提交 accepted tokens，或处理 rejected token。
5. 重复上述流程直到达到 `max_tokens` 或遇到 EOS。

前端中的 `DSSD` 模式会走这条路径，并展示 acceptance 相关指标。

### Target-only 模式

Target-only 模式用于基线对比：

1. edge 将 prompt 和采样参数发送给 verifier。
2. verifier 直接使用 target 模型完成生成。
3. edge 返回 verifier 的生成结果。

该模式不使用 edge draft 模型，因此 acceptance 指标显示为 `-`。它适合用来和 DSSD 模式比较同一 prompt 下的延迟、token/s 和输出质量。

## 启动服务

建议在项目根目录运行命令，并设置：

```bash
PYTHONPATH=$PWD
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

当前 DSSD 主要按 model runner v1 路径使用，因此启动时建议显式传入：

```bash
--model-runner-version v1
```

### 启动 verifier server

verifier server 加载 target 模型，并提供 `/open_session`、`/verify_round`、`/generate` 等接口。

OPT-6.7B 示例：

```bash
PYTHONPATH=$PWD PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
.venv/bin/python -m vllm.dssd.entrypoints.verifier_server \
  --host 127.0.0.1 \
  --port 18021 \
  --ready-file /tmp/dssd-verifier-ready.json \
  --model /root/autodl-tmp/opt-6.7b \
  --model-runner-version v1 \
  --gamma 4 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager
```

### 启动 edge server

edge server 加载 draft 模型，连接 verifier server，并提供前端页面。

OPT-125M 示例：

```bash
PYTHONPATH=$PWD PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
.venv/bin/python -m vllm.dssd.entrypoints.edge_server \
  --host 127.0.0.1 \
  --port 6006 \
  --ready-file /tmp/dssd-edge-ready.json \
  --verifier-url http://127.0.0.1:18021 \
  --model /root/autodl-tmp/opt-125m \
  --model-runner-version v1 \
  --gamma 4 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager
```

### 模型组合

已经验证过的常用组合包括：

- `opt-125m` 作为 edge draft 模型，`opt-6.7b` 作为 verifier target 模型。
- `Qwen3-0.6B` 作为 edge draft 模型，`Qwen3-8B` 作为 verifier target 模型。

注意事项：

- edge 模型和 verifier 模型的 tokenizer 应保持兼容。
- edge server 默认从 edge tokenizer 自动推断 EOS token id；如果需要覆盖，可以显式传入 `--eos-token-id`。
- 对比 DSSD 和 target-only 时，应使用相同的 prompt、`max_tokens` 和 `temperature`。

## 前端页面使用

启动 edge server 后，浏览器访问：

```text
http://127.0.0.1:6006/
```

页面支持以下输入：

- **输入文本**：待生成的 prompt。
- **最大生成长度**：对应 `max_tokens`。
- **采样温度**：对应 `temperature`。设置为 `0.0` 时为贪心采样。
- **网络延迟 ms**：模拟 edge 和 verifier 通信链路的固定延迟。
- **带宽 Mbps**：模拟 edge 和 verifier 通信链路的带宽限制。
- **token/s 计入 prompt token**：控制 token/s 统计时是否将 prompt token 加入分子。
- **模式**：选择 `DSSD` 或 `Target-only`。

点击 `开始生成` 后，页面会请求 edge server 的 `/benchmark_complete` 接口并展示生成结果和指标。

## Benchmark 指标说明

页面右侧展示以下指标：

| 指标 | 含义 |
|---|---|
| 总耗时 | 浏览器测量的端到端耗时，包含浏览器到 edge 的 HTTP 请求、服务端处理和响应返回。 |
| token / s | 按浏览器端到端耗时计算的吞吐。可选择只统计 output token，或统计 prompt + output token。 |
| 服务端推理耗时 | edge server 内部统计的推理耗时，不包含浏览器到 edge 的 HTTP 开销。 |
| 服务端 token / s | 按服务端推理耗时计算的吞吐。 |
| Prompt Tokens | prompt 编码后的 token 数。 |
| Output Tokens | 实际生成的 token 数。 |
| 计费口径 Tokens | 当前 token/s 分子使用的 token 数。 |
| Draft Acceptance | DSSD 模式下，accepted draft tokens / total draft tokens。 |
| All-Accept Round Rate | DSSD 模式下，全部 draft tokens 都被接受的轮次占比。 |
| Avg Accepted Len / Round | DSSD 模式下，平均每轮接受的 draft token 数。 |

Target-only 模式不产生 draft 验证过程，因此 acceptance 相关指标显示为 `-`。

## 网络模拟

前端的 `网络延迟 ms` 和 `带宽 Mbps` 用于模拟 edge 和 verifier 之间的通信链路。

示例：

- `网络延迟 ms = 20` 表示每次 edge-verifier 请求或响应额外加入 20ms 固定延迟。
- `带宽 Mbps = 100` 表示按 100 Mbps 估算 payload 传输时间。
- 两者都为 `0` 或留空时，不启用模拟网络。

网络模拟只作用于 edge server 和 verifier server 之间的通信，不包含浏览器到 edge server 的网络。

对不同模式的影响：

- **DSSD 模式**：会影响 open session、每轮 verify round、close session 等 edge-verifier 通信。
- **Target-only 模式**：会影响 edge 请求 verifier `/generate` 的通信。

## HTTP 接口

### Edge server

| 接口 | 作用 |
|---|---|
| `GET /` | 返回 Web benchmark 页面。 |
| `POST /generate` | 使用 DSSD 生成 token ids。 |
| `POST /complete` | 使用 DSSD 生成文本。 |
| `POST /benchmark_complete` | 前端 benchmark 使用的接口，返回文本、token 数、耗时和 DSSD 统计指标。 |

`/benchmark_complete` 请求示例：

```json
{
  "req_id": "ui-1",
  "mode": "dssd",
  "prompt": "请用中文简要介绍一下分布式投机采样。",
  "sampling_params": {
    "max_tokens": 128,
    "temperature": 0.0
  },
  "network_simulation": {
    "latency_ms": 20,
    "bandwidth_mbps": 100
  },
  "lora_request": null
}
```

其中 `mode` 可选：

- `dssd`
- `target_only`

响应示例：

```json
{
  "req_id": "ui-1",
  "mode": "dssd",
  "text": "生成文本",
  "server_inference_seconds": 1.23,
  "prompt_token_count": 20,
  "output_token_count": 128,
  "total_rounds": 32,
  "total_draft_tokens": 128,
  "total_accepted_tokens": 96,
  "draft_acceptance_rate": 0.75,
  "all_accept_round_rate": 0.5,
  "avg_accepted_len_per_round": 3.0
}
```

### Verifier server

| 接口 | 作用 |
|---|---|
| `POST /open_session` | 为一次 DSSD 请求打开 verifier session，并返回 bootstrap token。 |
| `POST /verify_round` | 验证 edge 发送的一轮 draft tokens。 |
| `POST /generate` | target-only 模式使用，直接由 verifier 的 target 模型完成生成。 |
| `POST /close_session` | 关闭 verifier session 并释放资源。 |

## 使用建议

- 做 DSSD 和 target-only 对比时，固定 prompt、`max_tokens`、`temperature`、模型组合和网络模拟参数。
- 贪心采样实验使用 `temperature=0.0`。
- 测网络影响时，可以先设置固定 `max_tokens`，然后分别改变 `网络延迟 ms` 和 `带宽 Mbps`。
- 如果只关心模型推理本身，优先看服务端推理耗时和服务端 token/s。
- 如果关心用户真实感知延迟，优先看浏览器端到端总耗时和浏览器端 token/s。

## 当前限制

- 当前主要验证和使用的是 model runner v1。
- 网络模拟是 edge-verifier 链路模拟，不等价于真实公网网络环境。
- Target-only 模式用于 baseline 对比，不产生 DSSD acceptance 指标。
- 不同模型组合需要保证 tokenizer 兼容，否则 DSSD 和 target-only 的对比可能失真。
