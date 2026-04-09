# DSSD Verifier 第一版设计

**目标**

在 `vllm/dssd/verifier/` 下实现一套 DSSD verifier 正式内核，直接复用真实 `vllm/v1` 的 `GPUWorker`、`GPUModelRunner`、`KVCacheManager` 和采样链路。第一版不做 `service`、`transport` 或 `mock network`，但必须能基于真实小模型跑通单请求 `open_session -> verify_round -> close_session` 链路。

**范围**

- 单请求
- decoder-only 文本模型
- 固定 `gamma`
- verifier 每轮只跑一次 `query_len = 1 + gamma` 的真实 forward
- accept/reject 热路径必须是 GPU-first
- 允许对 `vllm/v1` 做少量内部接口补充

**不做的事**

- 不实现 `service`、`transport`、`mock network`
- 不做多请求调度
- 不做 penalty、bad words、结构化约束的完整兼容
- 不做 edge-verifier 正式联调
- 不把 DSSD verifier 并入现有 `vllm/v1/spec_decode` 主流程
- 不为了第一版抽象上游友好的通用 spec decode 框架

## 总体方案

第一版采用“小接口下沉”方案：

- 主状态机和 DSSD 语义放在 `vllm/dssd/verifier/`
- `vllm/v1` 只补 verifier 需要但当前缺失的最小内部执行能力
- 不把 DSSD 业务语义灌进通用执行主路径

这样做的目的有两个：

1. 保住真实模型、真实 KV、真实 attention/cudagraph 路径
2. 把 DSSD 特有状态机集中在 `vllm/dssd/verifier/`，避免第一版改动面失控

## 代码落点

第一版 verifier 正式代码放在 `vllm/dssd/verifier/`，建议拆成 6 个文件：

- `types.py`
  - `VerifierSession`
  - `VerifierRoundRequest`
  - `VerifierRoundResult`
  - `VerifierRoundState`
- `scheduler.py`
  - `VerifierSchedulerAdapter`
- `state_bridge.py`
  - `VerifierStateBridge`
- `sampler.py`
  - `DSSDVerifierSampler`
- `engine.py`
  - `VerifierDecodeEngine`
- `ops.py`
  - accept/reject 用到的 Triton/CUDA helper

这套结构直接对应 `docs/dssd/verifier_demo/` 的职责划分，但去掉 demo 语义，改成真实可执行的薄适配层。

## 核心状态机

### 1. `open_session`

`open_session()` 的职责是建立 prompt 前缀状态，并返回 bootstrap token，但不提交该 token。

执行步骤：

1. 创建 `VerifierSession`
2. 用真实 `KVCacheManager.allocate_slots()` 为 prompt prefill 分配 blocks
3. 构造一次 prompt prefill 的 `SchedulerOutput`
4. 执行真实 prefill forward
5. 从 target 模型分布采样 bootstrap token
6. 只返回 bootstrap token，不推进 verifier 主状态

执行完成后应满足：

- prompt 已经被标记为“已计算”
- `session.num_computed_tokens == session.prompt_len`
- `session.total_len == session.prompt_len`
- `session.token_ids` 里不包含 bootstrap token
- `req_states.all_token_ids` 的输出前缀里不包含 bootstrap token

### 2. `verify_round`

`verify_round()` 的输入固定为：

- `committed_token_id`
- `draft_token_ids`
- `draft_q_values`

这一轮的执行语义固定为 `[committed_token] + draft`。

执行步骤：

1. 把 `committed_token_id` 写入 `req_states.last_sampled_tokens`
2. 把 `draft_token_ids` 写入 `req_states.draft_tokens`
3. 构造一次 `query_len = 1 + len(draft_token_ids)` 的 `SchedulerOutput`
4. 执行真实 forward
5. 在 verifier 自己的 GPU sampler 里完成 accept/reject
6. 若全接受，旁路返回 `bonus_token_id`
7. 若拒绝，旁路返回拒绝位置对应的 `rejected_target_logits`
8. 在 `postprocess()` 前手动提交 `committed_token`
9. 调 `postprocess()` 只提交 accepted draft prefix

一轮结束后，真正进入 verifier 主状态的 token 只有：

- `committed_token`
- accepted draft prefix

绝不能进入 verifier 主状态的 token 或结果有：

- `bonus_token`
- reject 后 edge 后续会采用的新 token
- 整行 reject logits

状态推进约束如下：

- `num_computed_tokens += 1 + accepted_len`
- `total_len += 1 + accepted_len`
- `token_ids` 只追加 `committed_token + accepted_draft_prefix`
- `bonus_token` 只通过返回值旁路给 edge，不写进 `token_ids`

### 3. `close_session`

`close_session()` 的职责是释放 request state、round state 和 KV blocks。

执行步骤：

1. 构造当前 request 的真实 close 视图
2. 调 `KVCacheManager.free(...)` 释放该 request 持有的 blocks
3. 通过 `finished_req_ids` 走一次 worker 侧清理
4. 删除 `session`
5. 删除 `round_state`

## `req_states` 关键语义

### `bootstrap_token`

- 由 verifier 在 `open_session()` 末尾从 target 模型分布采样
- 只能返回给 edge
- 不能在 `open_session()` 阶段写入 `all_token_ids`
- 不能在 `open_session()` 阶段推进 `total_len`

### `committed_token`

- 是本轮输入 token
- 也是本轮需要确认进入前缀的 token
- 必须在 `postprocess()` 前手动写进：
  - `last_sampled_tokens`
  - `all_token_ids`
  - `total_len`

但它不属于 `postprocess()` 视角里的“本轮 sampled output token”。

### `accepted draft prefix`

- 是本轮真正通过 `postprocess()` 提交的 sampled tokens
- 必须进入 `sampled_token_ids`
- 必须推进 `num_computed_tokens` 和 `total_len`

### `bonus_token` 与 `rejected_target_logits`

- 都只允许通过 `VerifierRoundResult` 旁路返回
- 都不能污染 verifier 主状态
- 都不能提前写进 `all_token_ids` 或 `last_sampled_tokens`

## `vllm/v1` 最小内部接口补充

第一版只建议补 3 组内部能力。

### 1. verifier query 输入准备能力

需要一个内部入口，让 verifier 能复用 `[committed_token] + draft` 的 expanded input / logits row 组织方式，但不被当前 spec decode 的 rejection sampler 语义绑住。

目标是：

- 复用真实 `scheduled_spec_decode_tokens` 输入形态
- 复用真实 `InputBatch` / `execute_model_state`
- 允许 verifier 自己接管 accept/reject 尾段

### 2. 只采样、不提交的 bootstrap 能力

需要一个内部 helper，在拿到 `hidden_states + input_batch` 后，可以直接完成 token 采样，但不推进：

- `all_token_ids`
- `total_len`
- `num_computed_tokens`

该能力只服务 `open_session()` 的 bootstrap 采样。

### 3. 手动提交 `committed_token` 的能力

需要一个内部 helper，安全地把当前轮输入 token 写进：

- `last_sampled_tokens`
- `all_token_ids`
- `total_len`

随后允许 `postprocess()` 只处理 accepted draft prefix，而不是把 `committed_token` 也混进 sampled output 语义。

## GPU-first accept/reject 设计

第一版 accept/reject 热路径必须保持 GPU-first。

保留在 GPU 上的数据包括：

- `processed_logits`
- `draft_q_values` 对应 tensor
- reject 位置对应的 target logits

第一版的 `ops.py` 负责提供最小 kernel/helper，以完成：

- 对每个 draft step 计算 `p_j / q_j`
- 按 seed + position 语义生成 accept/reject 所需随机数
- 得到 `accepted_len`
- 在全接受时保留 bonus token 采样路径
- 在拒绝时提取 reject step 对应 logits row

这里不接受逐 step 的 CPU host-device 同步回退。

## 测试与验收

测试放在 `tests/dssd/verifier/`，分 3 层。

### 1. 轻量语义测试

锁定最容易被后续改坏的约束：

- `open_session()` 不得 commit bootstrap token
- `verify_round()` 必须先提交 `committed_token`，再只提交 accepted draft prefix
- reject 时返回 reject step 对应的 processed target logits
- `close_session()` 必须释放 KV blocks 和 round/session state

### 2. 组件级真实执行测试

直接实例化真实 `GPUWorker/GPUModelRunner` 跑小模型，验证：

- `open_session -> verify_round -> close_session` 单请求链路可跑通
- `verify_round()` 每轮只执行一次 `query_len = 1 + gamma` 的真实 forward
- `accepted_len`、`bonus_token_id`、`rejected_target_logits` 三种结果语义正确
- `session` 镜像状态和 `req_states` 推进一致

### 3. 最小一致性测试

构造固定 prompt、固定 committed token、固定 draft 和固定 q-values 的可重复用例，验证：

- 全接受时，bonus token 来自 target 模型采样路径
- 拒绝时，返回的 logits 行确实对应 reject step
- verifier 主状态不会提前混入 `bonus_token` 或 reject 后的新 token

### 验收标准

- 正式代码落在 `vllm/dssd/verifier/`
- `vllm/v1` 只新增 verifier 必需的最小内部接口
- 单请求真实模型链路可跑通 `open_session -> verify_round -> close_session`
- accept/reject 热路径保持 GPU-first

## 风险与约束

- 当前 `GPUModelRunner` 把 speculative query 组织和内部 spec decode 语义绑得较紧，新增接口时必须控制在“通用执行能力”层面
- 第一版只覆盖单请求、固定 `gamma`、decoder-only 文本模型；如果后续扩到多请求或复杂采样约束，状态推进逻辑需要重新审视
- 第一版允许少量 `vllm/v1` 内部改动，但不以“可上游合并”为目标优化抽象
