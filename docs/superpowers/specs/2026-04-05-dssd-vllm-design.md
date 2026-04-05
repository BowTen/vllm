# 基于 vLLM 的 DSSD 边云分布式投机采样系统设计

**目标**

在 vLLM 上实现一个贴近 DSSD 原始算法的边云分布式投机采样原型系统。系统的对外入口位于边端，边端部署草稿模型并直接服务用户请求；云端部署目标模型并作为 verifier 服务，为边端的每一轮 draft 提供 Accept/Reject 验证。开发与实验阶段在一台双 4090 机器上双端部署，但系统边界必须从一开始就按真实网络通信来设计。

**设计原则**

- 用户请求归边端所有，云端不直接对外生成文本。
- 上行只发送 `draft_token_ids + q_values + prefix_delta`，不发送完整 `Q_j(x)`。
- 下行仅在拒绝时返回单个 `P_j(x)` 完整分布；全部接受时只返回 `bonus_token`。
- DSSD 的控制逻辑放在 vLLM 前端/控制层，模型计算能力通过 EngineCore utility path 暴露。
- 尽量复用 vLLM 现有的模型执行、KV cache、worker 进程模型，以及 speculative decoding 的多位置 logits 输入展开能力。
- 第一版优先保证系统形态与算法正确性，不追求一开始就支持全部 OpenAI/vLLM 特性。

---

## 1. 系统定位

### 1.1 角色划分

系统由两个 vLLM 实例组成：

- **边端 Edge vLLM**
  - 加载草稿模型。
  - 暴露用户访问入口。
  - 负责 tokenization、draft 生成、reject 后 residual resample、最终流式输出。
- **云端 Verifier vLLM**
  - 加载目标模型。
  - 不直接面向终端用户。
  - 作为内部 verifier 服务，接收边端的 draft 轮次请求并返回验证结果。

### 1.2 目标部署形态

- **开发/实验阶段**
  - 在同一台机器上部署边端和云端，使用真实 RPC 接口通信。
  - 允许在 transport 层注入时延、带宽等网络模拟参数。
- **最终系统形态**
  - 边端和云端可独立部署。
  - 边端启动时绑定一个 verifier 服务，后续持续使用该 verifier 完成 DSSD 轮次验证。

### 1.3 第一版范围

第一版仅支持：

- text-only
- 单用户请求优先
- 有状态 verifier session
- 固定或配置化 `gamma`
- 基本采样参数：`temperature`、`top_k`、`top_p`、`max_tokens`、`stop_token_ids`
- 边端流式输出

第一版不支持：

- multimodal
- structured output
- tool calling
- beam search
- parallel sampling
- LoRA
- 复杂 prefix cache 恢复

---

## 2. 总体架构

### 2.1 架构决策

本系统不直接改造 vLLM 现有 speculative decode 主流程成为远程 verifier 版本，而是在 vLLM 中新增一套 `DSSD edge mode + DSSD verifier mode`。原因如下：

- 现有 speculative decode 默认请求生命周期归目标模型所有，不适合“请求归边端”的系统语义。
- DSSD 要求上行仅传 `q(x_i)` 标量，而现有 rejection sampler 默认依赖 draft 全分布。
- DSSD 的系统难点首先在边云协议、session 同步与状态机，而不是本地 rejection sampling 数学本身。

因此，本设计采用：

- **边端控制层**：负责 DSSD 状态机与用户输出。
- **云端 verifier 服务层**：负责 session、一致性、accept/reject 决策。
- **EngineCore utility 层**：负责向上暴露 DSSD 专用引擎能力。
- **Worker / ModelRunner hook 层**：负责 draft 轮次执行与 verifier 多位置前向。

### 2.2 模块分层

| 层级 | 模块 | 责任 |
| --- | --- | --- |
| API / Serving | `DSSDEdgeServingChat`、`DSSDVerifierRouter` | 分别提供边端用户入口与云端内部 verifier 接口 |
| Control | `VerifierBindingManager`、`DSSDRoundCoordinator`、`DSSDEdgeSessionManager`、`DSSDVerifierSessionManager` | 负责会话、状态机、协议一致性 |
| Engine Utility | `DSSDSessionRunner`、`DSSDBatchPlanner`、`DSSDExecutionAdapter` | 将 DSSD 高层请求转换为底层引擎执行 |
| Worker Hook | `DSSDDraftRunner`、`DSSDVerifierRunner`、`DSSDResampleExecutor` | 负责具体模型前向、draft 缓存、resample |
| Shared | `DSSDConfig`、`DSSDProtocol`、`DSSDTransport`、`DSSDMetrics` | 配置、协议、网络、实验统计 |

---

## 3. 模块设计与嵌入点

## 3.1 配置模块

新增 `DSSDConfig`，建议放入 `vllm/config` 并接入 [arg_utils.py](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/vllm/engine/arg_utils.py)。

建议字段：

- `enabled: bool`
- `role: Literal["edge", "verifier"]`
- `gamma: int`
- `verifier_url: str | None`
- `bind_timeout_s: float`
- `request_timeout_s: float`
- `protocol_version: str`
- `auth_mode / auth_payload`
- `network_simulation`：
  - `latency_ms`
  - `bandwidth_mbps`
  - `jitter_ms`
- `metrics_enabled: bool`

职责：

- 控制边端/云端模式
- 暴露 verifier 绑定参数
- 支撑实验期网络模拟

## 3.2 边端入口模块

新增 `DSSDEdgeServingChat`，挂在 OpenAI chat serving 路径上，替代当前直接 `engine_client.generate()` 的路径。

嵌入位置：

- [api_server.py](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/vllm/entrypoints/openai/api_server.py)
- chat completion 对应的 router / serving 模块

职责：

- 接收用户请求
- 创建边端 DSSD session
- 驱动 `DSSDRoundCoordinator`
- 将已确认 token 流式返回给用户

设计要求：

- 对用户仍保持边端是唯一入口
- 不把临时 draft token 作为最终输出返回

## 3.3 云端 verifier 入口模块

新增 `DSSDVerifierRouter`，作为内部服务接口，不直接暴露普通 chat completion。

职责：

- 暴露 `bind / create_session / verify_round / close_session`
- 处理认证、协议版本校验、参数校验
- 将请求路由到 `DSSDVerifierSessionManager`

## 3.4 边端会话管理模块

新增 `DSSDEdgeSessionManager`，维护 `DSSDEdgeSessionState`。

长期状态建议包含：

- `request_id`
- `local_session_id`
- `verifier_binding_id`
- `verifier_session_id`
- `seq_no`
- `prompt_token_ids`
- `committed_token_ids`
- `pending_prefix_delta_token_ids`
- `sampling_params_fingerprint`
- `stop_checker_state`
- `status`
- `created_at`
- `last_activity_at`
- `metrics`

职责：

- 创建/关闭边端 session
- 保存已确认 prefix
- 保存待同步到云端的 `prefix_delta`
- 为 round coordinator 提供状态读写接口

## 3.5 边端轮次协调模块

新增 `DSSDRoundCoordinator`，这是系统核心状态机。

建议状态：

- `INIT`
- `LOCAL_READY`
- `REMOTE_READY`
- `DRAFTING`
- `VERIFYING`
- `COMMITTING`
- `STREAMING`
- `FINISHED`
- `ERROR`

职责：

- 驱动单轮流程：`draft -> uplink -> verify -> reject/resample -> commit -> stream`
- 确保每轮只提交已确认 token
- 在 reject 时触发本地 residual resample
- 维护 `seq_no`
- 记录 DSSD 轮次统计

## 3.6 云端会话管理模块

新增 `DSSDVerifierSessionManager`，维护 `DSSDVerifierSessionState`。

建议字段：

- `verifier_session_id`
- `target_engine_session_id`
- `binding_id`
- `seq_no`
- `committed_token_ids`
- `committed_len`
- `sampling_params_fingerprint`
- `rng_state`
- `status`
- `expire_at`
- `last_activity_at`
- `last_response_cache`

职责：

- 创建/关闭 verifier session
- 处理 `seq_no` 幂等与乱序
- 在验证前应用 `prefix_delta`
- 维护 verifier 侧 committed prefix
- 缓存上一次 round response 用于重试

## 3.7 verifier 批处理模块

新增 `VerifierRoundBatcher`。

职责：

- 聚合多个 `verify_round`
- 按模型、tokenizer、`gamma`、采样签名分桶
- 调用底层 `dssd_verify_round_batch`
- 将批量结果拆分回填到各 session

第一版约束：

- 可以先只支持小批量甚至单批路径
- 但模块边界应从一开始保留，避免后期重构

## 3.8 Engine utility 模块

新增 `DSSDSessionRunner`，挂在 EngineCore utility path 上。

复用路径：

- [core.py](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/vllm/v1/engine/core.py)
- [core_client.py](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/vllm/v1/engine/core_client.py)

原因：

- 现有 `EngineCoreRequestType.UTILITY` 已经提供了适合扩展的调用链
- 可以避免再为 DSSD 内部能力新增一套引擎 RPC

建议新增 utility 方法：

- 边端
  - `dssd_create_edge_session`
  - `dssd_draft_round`
  - `dssd_resample_reject`
  - `dssd_commit_tokens`
  - `dssd_close_edge_session`
- 云端
  - `dssd_create_verifier_session`
  - `dssd_verify_round_batch`
  - `dssd_commit_prefix_delta`
  - `dssd_close_verifier_session`

## 3.9 Worker / ModelRunner hook 模块

新增三个底层执行模块：

- `DSSDDraftRunner`
  - 边端从 committed prefix 出发做 `gamma` 步 draft rollout
  - 保存完整 `Q_j(x)` 到本地 round cache
  - 返回 `draft_token_ids + q_values + q_dists_handle`
- `DSSDVerifierRunner`
  - 云端根据 committed prefix + 外部 draft token，一次前向产出 `P_1...P_{γ+1}`
  - 尽量复用 vLLM 现有 speculative decode 多位置 logits 输入展开
- `DSSDResampleExecutor`
  - 在边端根据 `Q_j(x)` 与 verifier 下发的 `P_j(x)` 做 residual resample

嵌入位置：

- [gpu_model_runner.py](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/vllm/v1/worker/gpu_model_runner.py)

设计要求：

- 不改普通 `sample_tokens()` 的外部语义
- 不把 DSSD 的协议状态写进普通 `Request` 主流程

---

## 4. 状态对象与协议

## 4.1 边端长期状态

定义 `DSSDEdgeSessionState`：

- 只保存已确认输出与待同步前缀增量
- 不保存未验证的 draft 作为长期状态

定义 `DSSDEdgeRoundCache`：

- `seq_no`
- `gamma`
- `draft_token_ids`
- `q_values`
- `q_dists_handle`
- `draft_len`
- `draft_finished`
- `created_at`

设计要求：

- 完整 `Q_j(x)` 保留在 engine 内部
- API / serving 层只拿轻量 round 结果

## 4.2 云端长期状态

定义 `DSSDVerifierSessionState`：

- 维护 verifier 侧 committed prefix
- 维护 `seq_no` 与 `rng_state`
- 维护 `last_response_cache`

设计要求：

- verifier 必须有状态
- 同一 `seq_no` 的重试直接回放缓存结果
- 不得在重试时重新采样 accept/reject 随机数或 bonus token

## 4.3 协议对象

建议用 `msgspec.Struct` 定义在 `vllm/v1/dssd/protocol.py`。

核心 RPC：

### `BindVerifier`

请求字段：

- `protocol_version`
- `edge_instance_id`
- `tokenizer_hash`
- `vocab_hash`
- `supported_gamma_max`
- `auth_payload`

响应字段：

- `binding_id`
- `protocol_version`
- `verifier_model_id`
- `tokenizer_hash`
- `vocab_hash`
- `supported_gamma_max`
- `capabilities`

### `CreateSession`

请求字段：

- `binding_id`
- `request_id`
- `prompt_token_ids`
- `sampling_params_digest`
- `max_new_tokens`
- `stop_token_ids`

响应字段：

- `verifier_session_id`
- `accepted_prompt_len`
- `expires_at`

### `VerifyRound`

请求字段：

- `binding_id`
- `verifier_session_id`
- `seq_no`
- `prefix_delta_token_ids`
- `draft_token_ids`
- `q_values`

响应字段：

- `verifier_session_id`
- `seq_no`
- `accepted_count`
- `all_accepted`
- `bonus_token_id`
- `reject_index`
- `reject_target_probs`
- `finished`
- `finish_reason`

### `CloseSession`

请求字段：

- `verifier_session_id`
- `reason`

响应字段：

- `closed`

## 4.4 协议硬约束

- 上行不传 `Q_j(x)` 完整分布
- 下行拒绝时最多只传一个 `P_j(x)`
- `VerifyRound` 必须带 `seq_no`
- `sampling_params` 在 `CreateSession` 固定，后续按 digest 校验
- verifier 对同一 `seq_no` 必须幂等

---

## 5. 单轮执行流程

## 5.1 初始化

1. 边端启动时通过 `BindVerifier` 绑定 verifier。
2. 双方校验：
   - 协议版本
   - tokenizer hash
   - vocab hash
   - `gamma` 上限
   - 认证信息
3. 校验失败则边端服务不进入 ready。

## 5.2 用户请求开始

1. 用户请求到达边端 `DSSDEdgeServingChat`。
2. 边端创建本地 edge session。
3. 边端通过 `CreateSession` 在云端创建 verifier session。
4. 双端 session 建立后，交由 `DSSDRoundCoordinator` 驱动。

## 5.3 单轮 DSSD

每一轮按如下顺序执行：

1. 边端调用 `dssd_draft_round`：
   - 基于 committed prefix 执行 `gamma` 步草稿生成
   - 返回 `draft_token_ids`
   - 返回 `q_values`
   - 在本地 engine 内缓存 `Q_j(x)`，以 `q_dists_handle` 标识
2. 边端发送 `VerifyRound`：
   - `prefix_delta_token_ids`
   - `draft_token_ids`
   - `q_values`
   - `seq_no`
3. 云端 session manager 先应用 `prefix_delta`
4. 云端 batcher 聚合多个 round
5. 云端调用 `dssd_verify_round_batch`
6. 云端 verifier service 根据 `P_j(x_j)` 与 `q_j(x_j)` 做 accept/reject 判定
7. 返回结果：
   - 全接受：`accepted_count=gamma`，返回 `bonus_token_id`
   - 拒绝：返回 `reject_index=j` 与 `reject_target_probs=P_j(x)`
8. 边端 coordinator 处理结果：
   - 全接受：提交 `x_1...x_gamma + bonus_token`
   - 拒绝：调用 `dssd_resample_reject`，本地基于 `Q_j(x)` 与 `P_j(x)` 做 residual resample，提交 `x_1...x_{j-1} + x'_j`
9. 若发生 reject：
   - 将 `x'_j` 写入 `pending_prefix_delta_token_ids`
   - 下一轮随请求同步到云端 verifier
10. 边端将本轮新确认 token 流式输出给用户。

## 5.4 结束条件

以下任一条件触发结束：

- 达到 `max_tokens`
- 命中 stop token / stop string
- verifier 或 edge 判定 finished
- 用户中断请求
- 边云同步错误或 RPC 失败

结束时必须：

- 关闭边端 session
- 关闭云端 verifier session
- 释放 round cache 与 KV 资源

---

## 6. 一致性、失败处理与简单性约束

## 6.1 一致性规则

- 边端长期状态只包含 committed prefix
- 云端长期状态只包含 committed prefix
- 未验证 draft 永远只属于本轮临时状态
- reject 后生成的 `x'_j` 通过下一轮 `prefix_delta` 同步到云端

## 6.2 幂等规则

- `BindVerifier` 可重复调用但应返回同一能力信息
- `CreateSession` 失败则不创建半残 session
- `VerifyRound(seq_no=n)` 若收到重试：
  - 如果 `n == last_seq_no`，返回缓存结果
  - 如果 `n < last_seq_no`，返回 out-of-date 错误
  - 如果 `n > last_seq_no + 1`，返回 desync 错误

## 6.3 失败处理

第一版采用 fail-closed：

- verifier 不可达时，边端请求直接失败
- 不 silently fallback 到“只用草稿模型”
- verifier 返回协议不一致或 tokenizer 不一致时，边端拒绝启动或拒绝会话创建

## 6.4 简单性约束

为保证系统真正可用且可控，第一版不做：

- 在普通 `generate()` 主路径里混入 DSSD 会话逻辑
- 在普通 `Request` 中塞入大量 DSSD 专用字段
- 在 `rejection_sampler.py` 主流程上直接打补丁实现边云协议

---

## 7. 与 vLLM 现有能力的复用关系

## 7.1 直接复用

- 模型加载与 executor/worker 架构
- EngineCore utility 调用链
- KV cache 与 worker 进程模型
- speculative decode 对“外部 draft token 触发多位置 logits 前向”的输入展开思路

## 7.2 不直接复用的部分

- 现有 speculative decode 的请求所有权语义
- 现有 rejection sampler 对 draft 全分布的直接依赖
- 普通 `generate()` 的 sample-then-output 主流程

## 7.3 核心判断

应复用 speculative decode 的“多位置前向基础设施”，但不复用它的“本地 spec decode 控制语义”。

---

## 8. 指标与实验支持

新增 `DSSDMetrics`，建议记录：

- 每轮 `accepted_count`
- 平均 acceptance length
- reject position 分布
- 边端 draft latency
- 云端 verify latency
- round trip latency
- 上行字节数
- 下行字节数
- resample 次数
- TTFT
- TPOT
- 请求总时延

开发/实验阶段应支持：

- transport 层时延注入
- transport 层带宽限制
- 开关普通 vLLM baseline
- 开关 DSSD 模式

---

## 9. 分阶段实施顺序

## 9.1 阶段 1：协议与骨架

目标：

- 写出 `DSSDConfig`
- 写出 `DSSDProtocol`
- 打通 `bind / create_session / close_session`
- 搭起边端与云端 session manager 骨架
- 搭起 EngineCore utility 骨架

验收：

- 同机双进程下边端能绑定 verifier
- 能创建和关闭 verifier session
- `verify_round` 可以先返回 dummy response

## 9.2 阶段 2：先实现云端 verifier 能力

目标：

- 实现 `dssd_create_verifier_session`
- 实现 `dssd_commit_prefix_delta`
- 实现 `dssd_verify_round_batch`
- 实现 `DSSDVerifierRunner`

验收：

- 给定 committed prefix 和 draft token，verifier 能产出 `P_1...P_{γ+1}`
- 能正确构造 accept/reject 结果
- 全接受时能返回 `bonus_token`
- 拒绝时能返回单个 `P_j(x)`

## 9.3 阶段 3：实现边端 draft 与 residual resample

目标：

- 实现 `dssd_create_edge_session`
- 实现 `dssd_draft_round`
- 实现 `dssd_resample_reject`
- 实现 `dssd_commit_tokens`
- 实现 `DSSDDraftRunner`

验收：

- 边端能独立完成单轮 draft
- 收到 verifier reject 后能在本地完成 residual resample
- committed prefix 更新正确

## 9.4 阶段 4：拼接端到端链路

目标：

- 实现 `DSSDEdgeServingChat`
- 实现 `DSSDRoundCoordinator`
- 将边端用户请求与云端 verifier 完整接通
- 接入流式输出与 finish 条件处理

验收：

- 用户请求进入边端后能完整执行 DSSD 多轮生成
- 输出始终来自边端
- 边云 session 在结束时正确清理

## 9.5 阶段 5：实验与性能能力

目标：

- 实现 `VerifierRoundBatcher`
- 接入网络模拟
- 完善 metrics
- 补充 baseline 切换

验收：

- 能稳定支撑实验
- 能输出论文需要的 acceptance / latency / communication 指标

---

## 10. 测试策略

### 10.1 单元测试

- 协议对象的序列化/反序列化
- `seq_no` 幂等与乱序检测
- `prefix_delta` 同步逻辑
- residual resample 数学正确性
- verifier accept/reject 判定逻辑

### 10.2 集成测试

- 同机双端 bind / create / verify / close
- 单轮全接受
- 单轮中途拒绝
- 多轮 reject 后继续生成
- verifier 重试与幂等回放

### 10.3 端到端测试

- 边端 OpenAI chat 请求完整走 DSSD
- 用户能看到边端流式输出
- verifier 崩溃或超时时请求 fail-closed

---

## 11. 设计结论

本设计选择在 vLLM 中新增一套 DSSD 专用边云模式，而不是把现有 speculative decoding 直接远程化。边端负责用户入口、draft、resample 与最终输出；云端负责有状态 verifier session 与多位置验证；两者通过显式协议和 session 同步实现 DSSD 原始算法要求。底层实现上复用 vLLM 的 EngineCore utility path 与 speculative decoding 的多位置前向基础设施，但不复用其本地控制语义。该方案兼顾了算法贴合度、系统可用性、实验可控性以及后续论文写作所需的清晰系统边界。
