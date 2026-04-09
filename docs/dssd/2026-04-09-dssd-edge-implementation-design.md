# DSSD Edge 第一版实现设计

**目标**

在 `vllm/dssd/edge/` 下实现一套 edge 侧薄适配层，直接复用真实 `vllm/v1` 的 `GPUWorker`、`GPUModelRunner`、`KVCacheManager` 和 `Sampler`。第一版只打通 edge 本地执行流，不实现 transport、service 和真实 verifier 联调。

**范围**

- 单请求
- decoder-only 文本模型
- 固定 `gamma`
- edge 本地 decode 必须始终走 `query_len=1` 的 vLLM 原生 decode 快路径
- reject 后只做逻辑回退，不做尾部 KV block 物理释放
- verifier 相关输入先由测试或 stub 注入，不在这一版实现

**不做的事**

- 不改 `vllm/v1` 主热路径
- 不做多请求调度
- 不做 transport/mock network
- 不做复杂采样约束的完整兼容
- 不做异常恢复和会话容错

## 代码落点

第一版代码放在 `vllm/dssd/edge/`，拆成 5 个文件：

- `types.py`
  - `EdgeSession`
  - `EdgeRoundState`
- `scheduler.py`
  - `EdgeSchedulerAdapter`
- `state_bridge.py`
  - `EdgeStateBridge`
- `sampler.py`
  - `DSSDEdgeDraftSampler`
- `engine.py`
  - `EdgeDecodeEngine`

这套结构直接对应 `docs/dssd/edge_demo/`，但会去掉 demo 语义，改成真实可实现的薄适配层。

## 核心状态机

### 1. `open_session`

- 创建 `EdgeSession`
- 为 prompt 构造真实 `Request`
- 通过 `KVCacheManager.allocate_slots()` 申请 prefill blocks
- 仅建立会话，不执行模型

### 2. `prefill(session, bootstrap_token_id)`

- 构造一次 prompt prefill 的 `SchedulerOutput`
- 执行 `worker.execute_model(...)`
- 不在 edge 本地采样首 token
- 将 verifier 返回的 `bootstrap_token_id` 直接注入 `req_states`

执行完成后应满足：

- `session.num_computed_tokens == session.prompt_len`
- `session.total_len == session.prompt_len + 1`
- `session.token_ids == prompt_token_ids + [bootstrap_token_id]`

### 3. `decode_one(session, input_token_id, processed_logits_dst)`

- 先将 `input_token_id` 写入 `req_states.last_sampled_tokens`
- 构造一次 `query_len=1` 的 decode `SchedulerOutput`
- 执行 `execute_model()`
- 从 `execute_model_state` 取出当前 step 的 hidden states
- 计算 logits
- 应用 sampling params，得到 processed logits
- 采样一个 draft token，并提取该 token 的 `q_value`
- 调 `postprocess()` 提交该 draft token
- 同步 `EdgeSession` 镜像状态

### 4. `draft(session, first_token_id, gamma)`

- 清空上一轮 `EdgeRoundState`
- 初始化 `(gamma, vocab_size)` 的 logits buffer
- 从 `first_token_id` 开始循环执行 `decode_one()`
- 每一步记录：
  - `draft_token_id`
  - `draft_q_value`
  - 对应整行 processed logits

返回的 `EdgeRoundState` 用于后续 verifier 请求和 reject 后 residual resample。

### 5. `commit_external_token(session, token_id)`

这个接口用于提交两类 token：

- verifier all-accept 后返回的 `bonus_token`
- reject 后 edge 基于 residual distribution 本地 resample 得到的新 token

因为这些 token 不是本地 decode 采样出来的，所以不能走 `postprocess()`，而是必须直接写入：

- `req_states.last_sampled_tokens`
- `req_states.all_token_ids`
- `req_states.total_len`
- `req_states.num_computed_tokens`

同时同步 `session.token_ids`、`session.total_len`、`session.num_computed_tokens`。

### 6. `rollback(session, rejected_count)`

第一版只做逻辑回退：

- 回退 `session.token_ids`
- 回退 `session.total_len`
- 回退 `session.num_computed_tokens`
- 重写 `req_states.total_len`
- 重写 `req_states.num_computed_tokens`
- 重写 `req_states.last_sampled_tokens`

不做：

- 物理删除尾部 KV block
- penalty 状态回滚

### 7. `close_session(session)`

- 调 `KVCacheManager.free(request)` 释放 request 持有的 blocks
- 通过 `finished_req_ids` 走一次 worker 侧清理
- 删除 `EdgeSession`
- 清空 round-state

## 关键语义

### `bootstrap_token`

- 由 verifier 产生
- 由 edge 在 prefill 后立即 commit
- 在 edge 侧它是“已确认前缀”的一部分，不是 draft token

### `draft token`

- 只能通过本地 `decode_one()` 产生
- 必须经过 `postprocess()` 提交
- 会写入 `EdgeRoundState`

### `external token`

- 来自 verifier 或 edge residual resample
- 不能伪装成本地 sample 结果
- 必须通过 `state_bridge` 手动注入

## 性能边界

这一版要守住的核心是 edge 单端性能，而不是端到端性能。

要点如下：

- 本地 decode 始终维持 `query_len=1`
- 继续使用 vLLM 原生 page attention / cudagraph / KV cache 路径
- `processed logits` 保留在 GPU，并复用本轮 scratch buffer
- 不引入高层 `generate()` 或逐 token 重新建请求

第一版允许的性能妥协：

- `q_value` 提取时允许读取单个标量
- reject 后不回收尾部空 block
- verifier/transport 还未接入，因此不讨论端到端时延

## 测试边界

第一版只写最小单测，不做端到端联调。

### `scheduler`

- prefill 是否走 `allocate_slots`
- decode 是否按增量 token 申请 block
- close 是否走 `free`

### `state_bridge`

- bootstrap 注入是否正确推进会话镜像和 `req_states`
- external token commit 是否不走 `postprocess`
- rollback 是否只做逻辑回退

### `sampler`

- `processed_logits` 是否来自应用采样参数后的 logits
- `q_value` 是否等于 sampled token 的 softmax 概率
- logits buffer 是否按 `(gamma, vocab_size)` 复用

### `engine`

- `prefill -> draft -> commit_external_token -> rollback -> close_session` 的调用顺序
- `decode_one()` 是否只执行一次 `query_len=1` forward 和一次 `postprocess()`

## 为什么先不改 `vllm/v1`

第一版 edge 的主要目标是验证：

- `req_states` 写法是否正确
- block 生命周期是否闭合
- `draft/q/logits` 采集语义是否正确

这些问题都可以在 `vllm/dssd/edge/` 这一层解决，不必先把改动打进 `vllm/v1` 主执行流。等 edge 本地链路稳定后，再考虑把更通用的能力抽回 `vllm/v1`。
