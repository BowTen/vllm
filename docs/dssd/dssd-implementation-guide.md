# DSSD 系统实现说明

本文档从系统实现角度介绍当前 DSSD 在 vLLM 代码库中的落地方式，可作为毕业论文中“系统设计与实现”相关章节的参考材料。文档重点说明模块职责、推理流程、状态维护、通信协议和测试验证，不展开使用教程。

## 1. 系统实现概述

本系统基于 vLLM 实现 DSSD（Distributed Split Speculative Decoding）推理流程。系统被划分为 edge 和 verifier 两个运行角色：

- **edge**：运行较小的 draft 模型，负责生成候选 token、保存 draft token 的概率信息，并根据 verifier 返回结果提交或回滚 token。
- **verifier**：运行较大的 target 模型，负责验证 edge 生成的 draft tokens，并返回接受长度、bonus token 或拒绝位置所需的信息。

两端通过 HTTP 通信。edge 对外提供 benchmark 页面和生成接口，verifier 对外提供 session 管理、验证和 target-only 生成接口。整体实现保留了 vLLM 的模型加载、worker 执行、scheduler、KV cache 和 sampler 等基础能力，同时在其上加入 DSSD 所需的 draft、verify、commit 和 rollback 控制流程。

## 2. 代码模块划分

DSSD 相关代码主要位于 `vllm/dssd/` 目录下。

| 模块 | 主要职责 |
|---|---|
| `vllm/dssd/edge/` | edge 侧解码引擎、draft 采样、状态同步和调度适配。 |
| `vllm/dssd/verifier/` | verifier 侧解码引擎、target 模型验证、采样判断和状态同步。 |
| `vllm/dssd/service/` | edge/verifier 对外服务封装，将 engine 能力组织成生成和验证 API。 |
| `vllm/dssd/transport/` | edge 到 verifier 的通信层，包括 HTTP transport、in-process transport 和网络模拟。 |
| `vllm/dssd/protocol/` | edge 和 verifier 通信使用的数据结构。 |
| `vllm/dssd/entrypoints/` | verifier server、edge server、命令行 runner、运行时构造逻辑和前端页面。 |

其中 `service` 层负责连接上层 HTTP 接口和下层解码引擎。`edge_service.py` 实现完整 DSSD 主循环，`verifier_service.py` 封装 verifier 的 open、verify、generate 和 close 操作。

## 3. DSSD 推理流程实现

一次 DSSD 生成请求由 `DSSDEdgeService.generate_with_stats()` 驱动，其主要流程如下。

### 3.1 打开 session

edge 收到请求后，先将 prompt 编码为 token ids，然后调用 verifier 的 `open_session`：

1. verifier 使用 target 模型执行 prompt prefill。
2. verifier 从 target 模型 logits 中采样或选择第一个 bootstrap token。
3. verifier 返回 `bootstrap_token_id`。
4. edge 使用 draft 模型执行 prompt prefill，并把 verifier 返回的 bootstrap token 注入 edge session。

这样，edge 和 verifier 在生成开始前拥有相同的已确认前缀。

### 3.2 生成 draft tokens

每轮 DSSD 中，edge 调用本地 draft 模型生成最多 `gamma` 个 draft tokens。每生成一个 draft token，edge 会保存：

- `draft_token_ids`：本轮 draft token id 列表。
- `draft_q_values`：每个 draft token 在 draft 模型分布中的概率值。
- `draft logits`：edge 本地保存的 draft 模型分布，用于 reject 后的 residual resample。

上行发送给 verifier 的数据只包含 draft token ids 和对应 q values，不发送完整 draft logits。

### 3.3 verifier 验证 draft

edge 将一轮 draft 信息封装为 `VerifyRoundRequest`，发送给 verifier：

```text
req_id
committed_token_id
draft_token_ids
draft_q_values
```

verifier 基于当前已确认前缀和本轮 draft tokens 执行 target 模型前向计算，得到各 draft 位置对应的 target logits，然后执行接受/拒绝判断。

返回结果使用 `VerifyRoundResponse` 表示：

- 如果所有 draft tokens 都被接受，返回 `accepted_len = len(draft_token_ids)` 和 `bonus_token_id`。
- 如果某个 draft token 被拒绝，返回 `accepted_len` 和拒绝位置所需的 payload。
- 在非贪心采样下，拒绝时返回 `rejected_target_logits`。
- 在贪心采样下，拒绝时可以只返回 `rejected_token_id`。

### 3.4 edge 提交、回滚和重采样

edge 收到 verifier 返回结果后，根据 `accepted_len` 更新本地状态：

- 对已接受的 draft tokens，保留并提交。
- 如果本轮全部接受，提交 verifier 返回的 `bonus_token_id`。
- 如果发生拒绝，edge 回滚未接受的 draft tokens，并生成一个替代 token。

非贪心采样发生拒绝时，edge 使用 verifier 返回的 target logits 和本地保存的 draft logits 构造 residual distribution：

```text
residual = max(0, P - Q)
```

然后从归一化后的 residual distribution 中重新采样替代 token。

贪心采样发生拒绝时，edge 直接使用 verifier 返回的 `rejected_token_id`。这样可以避免在拒绝场景下传输完整词表 logits。

### 3.5 停止条件

每轮提交后，edge 检查：

- 已生成 token 数是否达到 `max_tokens`。
- 最近提交的 token 中是否包含 EOS。

如果超过 `max_tokens`，edge 会回滚多余 token；如果遇到 EOS，edge 会回滚 EOS 后面的 token 并结束请求。

## 4. Edge 端实现

edge 端的主要实现位于：

- `vllm/dssd/service/edge_service.py`
- `vllm/dssd/edge/engine_v1.py`
- `vllm/dssd/edge/sampler_v1.py`
- `vllm/dssd/edge/state_bridge_v1.py`
- `vllm/dssd/edge/scheduler.py`

`DSSDEdgeService` 是 edge 侧主控类，负责组织完整 DSSD 推理循环。`EdgeDecodeEngineV1` 负责和 vLLM worker 交互，执行 prompt prefill、单步 decode、draft 生成、外部 token 注入、rollback 和 session 关闭。

edge 侧状态主要包括：

- `prompt_token_ids`：请求 prompt。
- `token_ids`：当前 session 中的 token 序列。
- `prompt_len`：prompt 长度。
- `total_len`：当前逻辑序列长度。
- `num_computed_tokens`：vLLM 已计算 token 数。
- `round_state`：当前 DSSD 轮次中的 draft tokens、q values 和 logits。

`EdgeStateBridgeV1` 负责把 DSSD session 状态同步到 vLLM model runner 的 request state 和 input batch 中。它维护 token 写入、pending token、computed token 标记和 rollback 后状态恢复等逻辑。

## 5. Verifier 端实现

verifier 端的主要实现位于：

- `vllm/dssd/service/verifier_service.py`
- `vllm/dssd/verifier/engine_v1.py`
- `vllm/dssd/verifier/sampler_v1.py`
- `vllm/dssd/verifier/state_bridge_v1.py`
- `vllm/dssd/verifier/scheduler.py`

`DSSDVerifierService` 对外提供四类能力：

- `open_session()`：执行 prompt prefill，并返回 bootstrap token。
- `verify_round()`：验证一轮 draft tokens。
- `generate()`：target-only 模式下直接使用 target 模型生成。
- `close_session()`：释放 session 和 KV cache 资源。

`VerifierDecodeEngineV1` 负责 verifier 侧状态维护和 worker 执行。`open_session()` 会创建 verifier session，执行 prompt prefill，然后从 prefill logits 中得到 bootstrap token。注意 bootstrap token 在返回给 edge 后，verifier 会保持 session 状态在 prompt 之后，后续由 verify round 将 committed token 写入状态。

`verify_round()` 会在执行前保存快照。如果验证过程中发生异常，系统会回滚 verifier session、request state、input batch 和 round state，避免一次失败请求污染后续请求。

target-only 模式复用 verifier 的 session 和 verify round 状态推进逻辑。它通过空 draft 的 verify round 连续生成 token，从而保持 target-only 路径和 DSSD verifier 路径使用一致的状态维护方式。

## 6. vLLM 推理状态维护

DSSD 需要把自定义的 token 提交和回滚流程映射到 vLLM 的请求状态中。当前实现基于 vLLM model runner v1 路径，主要维护以下状态：

- session 中的 `token_ids`、`total_len` 和 `num_computed_tokens`。
- model runner 中的 request state，例如 `output_token_ids` 和 `num_computed_tokens`。
- input batch 中的 `token_ids_cpu`、`is_token_ids`、`num_tokens_no_spec` 和 `num_computed_tokens_cpu`。
- KV cache block 的分配和释放。

edge 和 verifier 都通过 scheduler adapter 构造 vLLM 可执行的 `SchedulerOutput`。不同阶段对应不同的 scheduler output：

- open session / prefill：调度 prompt tokens。
- decode：调度一个新 token。
- verify round：调度 committed token 和 draft tokens。
- close session：释放请求。

state bridge 的职责是把 DSSD 的逻辑状态变更同步到 vLLM 内部状态。例如：

- edge 注入 verifier 返回的 bootstrap token。
- edge 提交 verifier 返回的 bonus token 或 rejected token。
- edge 回滚未被接受的 draft tokens。
- verifier 将 committed token 和 accepted draft prefix 写入 request state。
- verifier 在异常时恢复快照。

通过这种方式，系统在保留 vLLM 执行路径的同时，实现了 DSSD 所需的细粒度控制。

## 7. 通信协议与数据传输

DSSD 的通信数据结构定义在 `vllm/dssd/protocol/types.py`。

### 7.1 Open session

`OpenSessionRequest` 包含：

- `req_id`
- `prompt_token_ids`
- `sampling_params`
- `lora_request`

`OpenSessionResponse` 返回：

- `req_id`
- `bootstrap_token_id`

### 7.2 Verify round

`VerifyRoundRequest` 包含：

- `req_id`
- `committed_token_id`
- `draft_token_ids`
- `draft_q_values`

`VerifyRoundResponse` 包含：

- `req_id`
- `accepted_len`
- `bonus_token_id`
- `rejected_token_id`
- `rejected_target_logits`

其中 `bonus_token_id`、`rejected_token_id` 和 `rejected_target_logits` 三者必须且只能存在一个。

### 7.3 Close session

`CloseSessionRequest` 和 `CloseSessionAck` 只包含 `req_id`，用于释放 verifier 侧 session 和 KV cache 资源。

### 7.4 Target-only generate

target-only 模式复用 open session 请求格式。edge 将 prompt token ids 和 sampling params 发送给 verifier 的 `/generate` 接口，verifier 完成 target 模型生成后返回 output token ids。

## 8. 采样与验证逻辑

verifier 的采样验证逻辑主要位于 `vllm/dssd/verifier/sampler_v1.py`。

### 8.1 非贪心采样

非贪心采样下，verifier 对每个 draft token 计算 target 模型概率 `p_i(x_i)`，并结合 edge 上传的 draft 概率 `q_i(x_i)` 执行投机采样接受判断：

```text
accept probability = min(1, p_i(x_i) / q_i(x_i))
```

如果发生拒绝，verifier 返回拒绝位置的 target logits。edge 使用 target logits 和本地 draft logits 完成 residual resample。

### 8.2 贪心采样

当 `temperature=0.0` 时，系统按贪心采样处理。verifier 直接比较 target 模型在每个位置的 argmax token 和 edge draft token：

- 相同则接受。
- 不同则拒绝，并返回 target argmax token。

这种情况下无需返回完整 logits，因此通信量更小。

## 9. HTTP 服务与前端 Benchmark

系统提供两个 HTTP server：

- `verifier_server.py`
- `edge_server.py`

verifier server 提供：

- `POST /open_session`
- `POST /verify_round`
- `POST /generate`
- `POST /close_session`

edge server 提供：

- `GET /`
- `POST /generate`
- `POST /complete`
- `POST /benchmark_complete`

前端页面位于 `vllm/dssd/entrypoints/static/edge_benchmark.html`。页面支持：

- 输入 prompt。
- 设置 `max_tokens` 和 `temperature`。
- 选择 `DSSD` 或 `Target-only`。
- 设置模拟网络延迟和带宽。
- 显示浏览器端到端耗时。
- 显示服务端推理耗时。
- 显示 token/s。
- 显示 DSSD acceptance 指标。

`/benchmark_complete` 是前端主要使用的接口。该接口会返回生成文本、token 数、服务端耗时、DSSD 轮次统计和 acceptance 指标。

## 10. 网络模拟实现

网络模拟功能位于：

- `vllm/dssd/transport/fake_network.py`
- `vllm/dssd/transport/http_verifier_transport.py`
- `vllm/dssd/entrypoints/edge_server.py`

`FakeNetwork` 使用固定延迟和带宽估算传输时间：

```text
transfer_time = fixed_latency_ms / 1000 + payload_bytes / bandwidth_bytes_per_s
```

HTTP transport 在发送请求和接收响应时，会根据序列化后的 payload bytes 调用网络模拟器。前端传入的 `network_simulation` 只在单次 `/benchmark_complete` 请求期间生效；请求结束后，edge server 会恢复 verifier transport 原有的网络配置，避免影响后续请求。

## 11. Benchmark 指标统计

edge service 在 DSSD 生成过程中维护 `EdgeGenerationStats`，统计：

- `total_rounds`
- `total_draft_tokens`
- `total_accepted_tokens`
- `all_accept_rounds`

由这些原始计数派生出：

- draft acceptance rate
- all-accept round rate
- avg accepted len per round

edge server 还在 `/benchmark_complete` 中记录服务端推理耗时。浏览器页面会额外测量端到端耗时，并分别计算浏览器端 token/s 和服务端 token/s。

## 12. 启动入口与运行时构造

运行时构造逻辑位于 `vllm/dssd/entrypoints/runtime_factory.py`。

verifier server 启动时会：

1. 根据命令行参数创建 vLLM runtime。
2. 构造 verifier scheduler、state bridge 和 sampler。
3. 创建 `VerifierDecodeEngineV1`。
4. 封装为 `DSSDVerifierService`。

edge server 启动时会：

1. 根据命令行参数创建 vLLM runtime。
2. 构造 edge scheduler、state bridge 和 draft sampler。
3. 创建 `EdgeDecodeEngineV1`。
4. 创建 HTTP verifier transport。
5. 读取 tokenizer，并自动推断或使用手动传入的 EOS token id。
6. 封装为 `DSSDEdgeService`。

## 13. 测试验证

DSSD 实现包含多层测试：

| 测试类型 | 覆盖内容 |
|---|---|
| service 测试 | edge/verifier 服务逻辑、DSSD 主循环、target-only 路径。 |
| engine 测试 | edge/verifier engine 的状态推进、提交、回滚和关闭。 |
| state bridge 测试 | session 状态与 vLLM request/input batch 状态同步。 |
| sampler 测试 | draft 采样、verify round、贪心和非贪心验证逻辑。 |
| transport 测试 | HTTP payload 编解码、网络模拟、错误处理。 |
| entrypoint 测试 | edge/verifier server 的接口、parser、runtime factory 和前端页面。 |
| process integration 测试 | server 进程启动、请求交互和关闭流程。 |
| real smoke 测试 | 使用真实 vLLM runtime 的端到端小规模验证。 |

这些测试保证了 DSSD 的核心路径、HTTP 通信路径、网络模拟和 benchmark 页面在代码层面可回归验证。

## 14. 当前实现限制

当前实现仍有一些限制：

- 主要验证和使用的是 vLLM model runner v1 路径。
- benchmark 以单请求交互式测试为主，尚未覆盖高并发服务场景。
- 网络模拟基于软件 sleep 和 payload bytes 估算，不等价于真实网络链路控制。
- DSSD 要求 edge 模型和 verifier 模型 tokenizer 兼容。
- target-only 模式用于 baseline 对比，不产生 DSSD acceptance 指标。

## 15. 论文写作建议

论文中可以将本文内容拆分到以下章节：

- **系统总体架构**：引用系统实现概述和模块划分。
- **核心推理流程实现**：引用 DSSD 推理流程、edge 端实现和 verifier 端实现。
- **状态维护与 vLLM 集成**：引用 vLLM 推理状态维护。
- **通信协议与优化**：引用通信协议、采样验证逻辑和贪心采样优化。
- **实验平台与测试工具**：引用 HTTP 服务、前端 benchmark、网络模拟和测试验证。
