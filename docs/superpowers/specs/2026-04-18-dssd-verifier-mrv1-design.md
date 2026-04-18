# DSSD Verifier MRV1 第一版设计

**目标**

在不兼容现有 verifier-v2 内核实现的前提下，单独为 old model runner v1 补一套 DSSD verifier backend。第一版目标是基于真实 old runner 跑通单请求 `open_session -> verify_round -> close_session`，优先保证 DSSD 分布语义正确，不把 cudagraph/非 eager 路径作为必须同时解决的问题。

**范围**

- 只看 verifier 侧
- 单请求
- decoder-only 文本模型
- 固定 `gamma`
- `async_scheduling=False`
- `enforce_eager=True`
- 每轮 verify 只跑一次真实 `query_len = 1 + gamma` forward
- `open_session` 返回的 bootstrap token 严格来自 target 模型分布，但不在该阶段 commit
- `verify_round` 只提交 `committed_token + accepted draft prefix`

**不做的事**

- 不兼容现有 `VerifierDecodeEngine` 的 v2 内部 hook
- 不强制抽象成统一 v1/v2 backend 框架
- 第一版不要求 cudagraph 路径打通
- 不支持 penalty、bad words、structured outputs
- 不支持多请求
- 不把 DSSD verifier 并入 old runner 原生 speculative decode 主路径

## 总体方案

采用“共享系统层 + 独立 MRV1 backend”方案：

- 系统层和协议层继续复用现有 DSSD verifier 结构
- 对 old runner 单独新增 `engine_v1 / state_bridge_v1 / sampler_v1`
- 第一版尽量不改 `vllm/v1`；若实现中发现直接读取 `execute_model_state` 过于脆弱，唯一允许的上游补丁是补一个和 v2 同语义的 `take_execute_model_state()` helper

这样做的原因有两个：

1. old runner 与当前 verifier-v2 依赖的状态模型差异很大，强行复用会让抽象失真
2. old runner 本身已经支持 `scheduled_spec_decode_tokens` 和 `execute_model_state`，足以支撑 DSSD verifier 的最小闭环

## 代码落点

### 复用

- `vllm/dssd/verifier/types.py`
- `vllm/dssd/verifier/scheduler.py`
- `vllm/dssd/service/verifier_service.py`
- 协议层 `OpenSessionRequest / VerifyRoundRequest / CloseSessionAck`

### 新增

- `vllm/dssd/verifier/engine_v1.py`
  - old MRV1 verifier 主状态机
- `vllm/dssd/verifier/state_bridge_v1.py`
  - 维护 `session / runner.requests / input_batch` 三份状态一致
- `vllm/dssd/verifier/sampler_v1.py`
  - 基于 old runner `execute_model_state.logits + spec_decode_metadata` 的 DSSD accept/reject

### 入口调整

- `vllm/dssd/verifier/__init__.py`
  - 导出 MRV1 backend 类
- `vllm/dssd/entrypoints/runtime_factory.py`
  - 增加 verifier 端 model runner 版本选择
- `vllm/dssd/entrypoints/verifier_server.py`
  - 如需真实服务切换 MRV1，则增加 `--model-runner-version`

## old MRV1 可复用能力

old runner 已具备以下 verifier 需要的关键能力：

- 通过 `scheduled_spec_decode_tokens` 表达 speculative suffix
- `execute_model()` 后将 `logits / spec_decode_metadata / sample_hidden_states` 暂存到 `execute_model_state`
- `InputBatch` 显式维护：
  - `token_ids_cpu`
  - `num_tokens_no_spec`
  - `num_computed_tokens_cpu`
  - `spec_token_ids`
- `CachedRequestState` 显式维护：
  - `output_token_ids`
  - `num_computed_tokens`
  - `block_ids`

因此 MRV1 verifier 不需要像 v2 一样直接改 GPU `req_states`，而是可以通过显式 request/batch 状态完成前缀推进。

## 核心状态模型

MRV1 backend 只维护并同步以下三份状态：

### 1. `VerifierSession`

作为 DSSD 语义镜像，负责保存：

- `prompt_token_ids`
- `token_ids`
- `num_computed_tokens`
- `total_len`

### 2. `runner.requests[req_id]`

作为 old runner 的真实 request 状态，负责保存：

- `output_token_ids`
- `num_computed_tokens`
- `block_ids`

### 3. `runner.input_batch`

作为下一轮 forward 输入来源，负责保存：

- `token_ids_cpu`
- `num_tokens_no_spec`
- `num_computed_tokens_cpu`
- `spec_token_ids`

`state_bridge_v1` 的唯一职责就是在这三份状态之间做最小同步。

## 核心状态机

### 1. `open_session`

`open_session()` 的职责是建立 prompt 前缀，并返回 bootstrap token，但不提交该 token。

执行步骤：

1. 创建 `VerifierSession`
2. 复用共享 `VerifierSchedulerAdapter.allocate_blocks()` 为 prompt 分配 KV blocks
3. 构造 prompt prefill 的 `SchedulerOutput`
4. 调 old runner `worker.execute_model(...)`
5. 直接从 `execute_model_state` 取出 prefill 末尾 logits
6. 用 `sampler_v1` 基于 old runner 的 `SamplingMetadata` 采样 bootstrap token
7. `state_bridge_v1.finish_prefill_without_commit(...)` 只把 prompt 标记为已计算，不写入 bootstrap token

结束后应满足：

- `session.num_computed_tokens == session.prompt_len`
- `session.total_len == session.prompt_len`
- `session.token_ids` 不包含 bootstrap token
- `runner.requests[req_id].output_token_ids` 不包含 bootstrap token
- `input_batch.token_ids_cpu` 的输出段不包含 bootstrap token

### 2. `verify_round`

`verify_round()` 的输入固定为：

- `committed_token_id`
- `draft_token_ids`
- `draft_q_values`

这一轮 forward 的语义固定为 `[committed_token] + draft`。

执行步骤：

1. `state_bridge_v1.begin_round(...)`
   - 将 `committed_token_id` 先写入：
     - `session.token_ids`
     - `runner.requests[req_id].output_token_ids`
     - `input_batch.token_ids_cpu`
     - `input_batch.num_tokens_no_spec`
   - 但此时不推进 `num_computed_tokens`
2. 复用共享 `scheduler.build_verify_step(...)`
   - `scheduled_spec_decode_tokens` 承载 `draft_token_ids`
3. 调 old runner `worker.execute_model(...)`
4. 从 `execute_model_state` 中直接取：
   - `logits`
   - `spec_decode_metadata`
   - `input_batch.sampling_metadata`
5. `sampler_v1` 完成 DSSD accept/reject：
   - 算 `accepted_len`
   - 全接受时返回 `bonus_token_id`
   - 拒绝时返回 reject step 的 processed target logits
6. `state_bridge_v1.finish_round(...)`
   - 只把 accepted draft prefix 追加进主状态
   - 一次性推进 `num_computed_tokens += 1 + accepted_len`

一轮结束后，真正进入 verifier 主状态的 token 只有：

- `committed_token`
- accepted draft prefix

绝不能进入 verifier 主状态的结果有：

- `bonus_token`
- reject 后 edge 将采用的新 token
- `rejected_target_logits`

### 3. `close_session`

`close_session()` 的职责是释放 request state、round state 和 KV blocks。

执行步骤：

1. 复用 `scheduler.free_blocks(session)`
2. 复用 `scheduler.build_close_step(req_id)`
3. 让 worker 清理 old runner 的 request / batch 状态
4. 删除 `session`
5. 删除 `round_state`
6. 保证没有悬空的 `execute_model_state`

## `state_bridge_v1` 设计

`state_bridge_v1.py` 不直接复制 v2 bridge 的做法，而是围绕 old runner 的显式状态表工作。

建议职责如下：

- `finish_prefill_without_commit(session, runner)`
  - 只对齐 prompt 已计算状态
- `begin_round(session, request, runner)`
  - 先写入 committed token，但不推进 computed tokens
- `finish_round(session, request, result, runner)`
  - 只追加 accepted draft prefix，并推进 computed tokens
- `remove_round_state(session)`
  - 清理 round bookkeeping

明确约束：

- `begin_round()` 之后、`finish_round()` 之前，`committed_token` 已经存在于前缀镜像中
- 但 `num_computed_tokens` 仍表示“本轮 forward 前已经计算完成的真实前缀长度”
- `finish_round()` 才真正把 `1 + accepted_len` 记入已计算长度

## `sampler_v1` 设计

`sampler_v1.py` 第一版不复用 old runner 默认 `sample_tokens()` 或 `RejectionSampler.forward()`，但复用 old runner 的采样参数与 logits processor 语义。

### `open_session` bootstrap

- 输入：
  - `execute_model_state.logits`
  - `input_batch.sampling_metadata`
- 输出：
  - `bootstrap_token_id`

要求：

- 只采样，不做任何 bookkeeping
- 采样分布必须与 target-only 一致

### `verify_round` accept/reject

- 输入：
  - `logits`
  - `spec_decode_metadata`
  - `sampling_metadata`
  - `draft_q_values`
- 输出：
  - `accepted_len`
  - `bonus_token_id | rejected_target_logits`

执行语义：

1. 取 target logits rows 和 bonus logits row
2. 复用 old sampler 的 logits processor 逻辑得到 processed target logits
3. 在 GPU 上根据 `p_j / q_j` 算 accepted prefix length
4. 全接受时从 bonus row 采样 bonus token
5. 拒绝时返回 reject step 对应的 processed target logits

第一版随机数语义不复用现有 v2 `ops.py` 的 seed 方案，而是单独围绕 old runner 的 `SamplingMetadata.generators` 或默认 RNG 实现。

## 是否修改 `vllm/v1`

第一版实现目标是：

- **默认不修改** old `vllm/v1`

允许的唯一上游补丁是：

- 给 old `GPUModelRunner` 补一个 `take_execute_model_state()` helper

该 helper 只允许做一件事：

- 取出并清空 `execute_model_state`

不允许第一版去修改：

- old runner 默认 `sample_tokens()` 主流程
- old runner 默认 rejection sampler 语义
- speculative decode 主执行链路

## 测试设计

测试放在 `tests/dssd/verifier/`，建议新增 v1 专用测试文件。

### 1. `test_state_bridge_v1.py`

覆盖：

- `open_session` 不 commit bootstrap token
- `begin_round` 只写 committed token，不提前推进 `num_computed_tokens`
- `finish_round` 只追加 accepted draft prefix
- reject 时主状态不混入 bonus 或 reject 后 token

### 2. `test_sampler_v1.py`

覆盖：

- 全拒绝
- 部分接受
- 全接受

断言：

- `accepted_len`
- `bonus_token_id`
- `rejected_target_logits`

三者语义正确。

### 3. `test_engine_v1.py`

使用 fake old runner，覆盖：

- `open_session -> verify_round -> close_session` 调用顺序
- verify 每轮只执行一次 `1 + draft_len` forward
- `engine_v1` 不调用 old runner 默认 `sample_tokens()` 做 verifier 逻辑

### 4. `test_engine_v1_smoke.py`

真实 old MRV1 worker，小模型，单请求，`enforce_eager=True`，覆盖：

- `open_session` 后 session 前缀长度仍等于 prompt_len
- `verify_round` 后主状态只推进 `committed_token + accepted prefix`
- `close_session` 后 request/KV 释放完成

## 验收标准

- 新增 MRV1 verifier backend，不破坏现有 verifier-v2
- 单请求真实 old runner 链路可跑通：
  - `open_session`
  - `verify_round`
  - `close_session`
- `open_session` 返回的 bootstrap token 不 commit
- `verify_round` 每轮只执行一次真实 `1 + gamma` forward
- verifier 主状态只推进 `committed_token + accepted draft prefix`
- `bonus_token` 和 `rejected_target_logits` 都通过旁路返回

## 第一阶段明确限制

- `model_runner_version == "v1"`
- `async_scheduling=False`
- `enforce_eager=True`
- 单请求
- decoder-only
- 固定 gamma
- 无 penalty
- 无 bad words
- 无 structured outputs

## 第二阶段收敛方向

第一阶段完成后，再评估以下增强：

1. 去掉 `enforce_eager=True` 限制，验证 old runner 的固定 `1 + gamma` query 能否稳定落到编译/graph 路径
2. 对齐 edge-v1 与 verifier-v1 的真实联调 smoke
3. 评估是否需要将 old runner `take_execute_model_state()` 正式下沉成共享 helper

## 风险

- old runner 的默认 speculative decode bookkeeping 很重，MRV1 verifier 需要确保“只借 forward，不借默认采样提交”
- 随机数语义如果直接套用 v2 `ops.py`，会与 old runner 的 sampler state 不一致
- `session / runner.requests / input_batch` 三份状态若不同步，下一轮 verify 很容易错位

## 最终结论

MRV1 verifier 第一版适合走“独立 backend”路线，而不是改现有 verifier-v2。共享系统层、单独补 MRV1 内核，可以把实现风险控制在 DSSD 自己的代码内，同时保留后续向 old runner 性能路径继续收敛的空间。
