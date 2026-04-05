# DSSD-on-vLLM Handoff

## 1. 项目目标

本分支的目标是在 vLLM 中实现一个可运行的 DSSD 原型系统，形态是：

- 边端对外服务用户请求，加载草稿模型
- 云端作为有状态 verifier 服务，加载目标模型
- 开发阶段同机双端部署，但接口和控制流按真实边云通信设计
- 当前优先级是“先把系统形态和 DSSD 控制面跑通”，再逐步把 draft/verifier 的 placeholder hook 替换成真实模型执行

设计基线文档：

- `docs/superpowers/specs/2026-04-05-dssd-vllm-design.md`
- `docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md`

## 2. 工作区与当前状态

当前实现工作都在这个 worktree 下进行：

- worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`
- branch: `dssd-vllm-codex`

当前最新相关提交：

- `7d14d62d6` `fix: preserve DSSD execution request package attribute`
- `57856c2a9` `refactor: tighten verifier execution request boundary`
- `edcd9d38d` `refactor: require verifier execution prefix delta`
- `a9e9f364c` `dssd: carry committed prefix into verifier execution`
- `5bf05864a` `fix: commit accepted verifier tokens per round`
- `f3b0efbdb` `feat: track verifier sessions in engine utilities`
- `3d2474f9f` `feat: add verifier forward probability helpers`
- `7e0cc6195` `refactor: return verifier forward results from engine hooks`
- `48c6e171d` `fix: expose DSSD worker hooks and verifier decisions`
- `0ea1f697e` `feat: route DSSD utility through collective RPC`
- `c43864a55` `feat: add DSSD single-round edge loop`
- `fc72aa8c4` `feat: wire DSSD edge control plane`

当前 git 状态基本干净，只剩一个未跟踪目录：

- `docs/superpowers/plans/`

这不是当前实现残留的代码改动，但新对话里继续工作时要注意别误删。

## 3. 当前已经实现到什么程度

### 3.1 配置与入口

已经完成：

- `DSSDConfig`、CLI 参数接入
- edge/verifier 角色切换
- edge `/v1/chat/completions` 进入 DSSD 控制路径
- verifier 私有 HTTP 路由：
  - `bind`
  - `create_session`
  - `verify_round`
  - `close_session`

关键文件：

- `vllm/config/dssd.py`
- `vllm/engine/arg_utils.py`
- `vllm/entrypoints/openai/chat_completion/dssd_serving.py`
- `vllm/entrypoints/serve/dssd/api_router.py`
- `vllm/entrypoints/openai/api_server.py`

### 3.2 控制面与协议

已经完成：

- DSSD 边云协议对象
- HTTP transport
- edge/verifier session manager
- verifier bind/create/verify/close 控制面
- fail-closed 路由行为
- edge 单轮 DSSD 控制流

关键文件：

- `vllm/v1/dssd/protocol.py`
- `vllm/v1/dssd/transport.py`
- `vllm/v1/dssd/edge/session.py`
- `vllm/v1/dssd/edge/coordinator.py`
- `vllm/v1/dssd/verifier/session.py`
- `vllm/v1/dssd/verifier/service.py`

### 3.3 Engine utility 与 worker hook

已经完成：

- `EngineCore` / `core_client` 的 DSSD utility 方法
- `session_runner -> collective_rpc -> WorkerBase -> model_runner` 真实调用链
- verifier forward 结果已经从“最终响应”收成了“forward probabilities”
- verifier accept/reject 决策放在 service 层完成，而不是 GPU hook 层
- verifier session 的 engine-side store 已经建立
- verifier 每轮 accepted token 会正确 commit 回 session

关键文件：

- `vllm/v1/engine/core.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/dssd/engine/session_runner.py`
- `vllm/v1/worker/worker_base.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/dssd/worker/verifier_runner.py`

### 3.4 最近这几轮刚收掉的点

这一段是后续最容易断线的地方：

- 新增了内部请求 `DSSDVerifierExecutionRequest`
  - 作用：把 verifier worker 执行层和外部 `VerifyRoundRequest` 解耦
  - 当前字段只保留 worker 真执行需要的内容：
    - `binding_id`
    - `verifier_session_id`
    - `seq_no`
    - `committed_token_ids`
    - `draft_token_ids`
    - `q_values`
- `DSSDSessionRunner.dssd_verify_round()` 不再把外部 `VerifyRoundRequest` 原样传给 worker
  - 现在会先用 engine 内 verifier session 的 `committed_token_ids` 组装内部 execution request
  - `prefix_delta_token_ids` 只在成功后写回 session，不会因为 transient retry 被重复污染
- `DSSDVerifierExecutionRequest` 仍保留包级属性：
  - `from vllm.v1.dssd import DSSDVerifierExecutionRequest` 可用
  - 但它不在 `vllm.v1.dssd.__all__` 中，避免被当成正式公共 API 宣传

相关文件：

- `vllm/v1/dssd/protocol.py`
- `vllm/v1/dssd/__init__.py`
- `vllm/v1/dssd/engine/session_runner.py`
- `tests/v1/dssd/test_protocol.py`
- `tests/v1/worker/test_dssd_verifier_runner.py`
- `tests/v1/worker/test_dssd_worker_base.py`

## 4. 当前系统“能做什么”

当前原型已经具备以下能力：

- edge/verifier 双端服务形态已经存在
- edge 端用户请求会进入 DSSD 控制路径，而不是静默回落到普通本地生成
- verifier 有状态 session 可以创建、校验、缓存和关闭
- 单轮 DSSD 控制流已经打通：
  - `draft_round`
  - `verify_round`
  - `optional residual resample`
  - `response`
- accept/reject 判定、bonus token sample、session RNG、幂等缓存都在 verifier service 层
- verifier engine-side session 会跟踪 committed prefix

可以参考的对外说明文档：

- `docs/serving/dssd_edge_verifier.md`

## 5. 当前明确还没完成的部分

这部分最重要，后续新对话应该直接从这里继续。

### 5.1 verifier 真实模型执行还没接上

`GPUModelRunner.dssd_verify_round()` 现在还是 placeholder。

当前状态：

- 它已经接受 `DSSDVerifierExecutionRequest`
- 但还没有真正走：
  - `scheduled_spec_decode_tokens`
  - `_prepare_inputs()`
  - `_calc_spec_decode_metadata()`
  - 从真实 logits 提取 `P_1...P_{gamma+1}`

也就是说：

- control plane 是真的
- session/utility path 是真的
- verifier forward contract 是真的
- 但 GPU verifier 的前向计算目前还是假的

这是当前第一优先级主线。

### 5.2 draft 真实模型执行也还是 placeholder

`GPUModelRunner.dssd_draft_round()` 目前也还是 placeholder。

所以现在的 edge draft / verifier forward 都还没有真正使用草稿模型和目标模型做概率计算。

### 5.3 DSSD edge 还是单轮同步路径

当前 edge 路径是 bring-up 版本：

- 单轮同步 DSSD 回合
- 非完整 streaming
- 更偏“控制流打通”而非“最终生产形态”

### 5.4 还没有完成多轮 verifier replay / request-KV 融合

虽然 verifier session 已经在 engine 层存在，但还没有真正把它和 `GPUModelRunner` 的 request/KV state 挂成一体。

也就是说，后续必须解决：

- 如何在 worker 侧构造一个“单请求 replay-spec-decode batch”
- 如何让 `committed_token_ids + draft_token_ids` 映射到现有 batch/state 结构
- 是先做保守的 prefix replay，再逐步优化成更真实的 session/KV 复用

## 6. 推荐的下一步实现顺序

下一位继续推进时，建议按这个顺序做，不要一上来同时改太深：

### Step A: 先做纯 helper，不直接碰完整 forward

建议先新增一个纯 helper，把 `DSSDVerifierExecutionRequest` 变成“单请求 spec-decode batch 视图”。

建议目标不是一步到位执行，而是先把这些输入准备逻辑可测试化：

- request id
- `num_scheduled_tokens`
- `scheduled_spec_decode_tokens`
- `CachedRequestState` / `NewRequestData` 所需的最小字段
- `committed_token_ids` 和 `draft_token_ids` 在 `token_ids_cpu` 里的布局关系

理由：

- 先把输入语义钉住
- 再接 `GPUModelRunner._update_states()` / `_prepare_inputs()`
- 可以避免在 KV、scheduler、worker hook 三个层面同时 debug

### Step B: verifier 先走保守 replay 方案

推荐第一版真实 verifier forward 用“整段 committed prefix replay”。

即使性能差一些，也更稳，更适合先跑通毕设原型。

推荐原因：

- 当前重点不是做上游可合并优化
- 而是尽快把 DSSD 真 forward 链路跑通
- replay 方案更容易复用现有 `GPUModelRunner` 批处理/输入准备路径

### Step C: 接 `_prepare_inputs()` 和 `_calc_spec_decode_metadata()`

一旦 Step A 的 helper 稳定，就让 `GPUModelRunner.dssd_verify_round()`：

1. 生成单请求 `SchedulerOutput`
2. 用 replay 方式更新 request state / input batch
3. 调 `_prepare_inputs()`
4. 取真实 `spec_decode_metadata`
5. 用 `build_verifier_result_from_logits()` 输出 `VerifierForwardResult`

### Step D: 再考虑 engine-side verifier session 与真实 KV 复用

这一步可以后做。先通，再快。

## 7. 已验证的测试命令

当前最可靠的一组 DSSD 回归命令是：

```bash
source .venv/bin/activate && pytest --noconftest \
  tests/v1/dssd/test_protocol.py \
  tests/v1/engine/test_dssd_core_utility.py \
  tests/v1/dssd/test_verifier_service.py \
  tests/v1/worker/test_dssd_draft_runner.py \
  tests/v1/worker/test_dssd_verifier_runner.py \
  tests/v1/worker/test_dssd_worker_base.py \
  tests/v1/dssd/test_round_coordinator.py \
  tests/v1/e2e/dssd/test_dssd_smoke.py \
  tests/entrypoints/test_dssd_verifier_router.py \
  tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py \
  tests/v1/dssd/test_transport.py -q
```

最新结果：

- `52 passed in 2.44s`

## 8. 环境与已知坑

### 8.1 当前 `.venv` 适合做 DSSD focused tests

这套环境已经足够支撑当前 focused DSSD 回归，但不代表完整 upstream test matrix 已经收齐。

### 8.2 `test_gpu_model_runner.py` 不适合拿来当当前稳定回归

当前环境下，这个文件可能在 collection 阶段因为缺少 `vllm.vllm_flash_attn` CUDA 扩展而失败。

这和当前 DSSD 控制面改动无关，所以最近几轮没有把它放进稳定回归集合。

### 8.3 不要再把 `test_dssd_verifier_runner.py` 做成双 protocol module 加载

这个坑刚修过。后续如果再改这类 stub-style test，注意：

- 让 `session_runner` 和测试使用同一份 `vllm.v1.dssd.protocol` module 对象
- 否则很容易出现“类名一样但 `isinstance` 不成立”的假阴性

## 9. 新对话里建议直接给出的上下文

如果你开新对话继续推进，建议直接提供下面这些路径，能节省很多上下文恢复时间：

- `docs/superpowers/specs/2026-04-05-dssd-vllm-design.md`
- `docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md`
- `docs/superpowers/handoffs/2026-04-05-dssd-vllm-handoff.md`
- `docs/serving/dssd_edge_verifier.md`
- `vllm/v1/dssd/protocol.py`
- `vllm/v1/dssd/engine/session_runner.py`
- `vllm/v1/dssd/verifier/service.py`
- `vllm/v1/dssd/worker/verifier_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`

新对话最推荐的起手任务是：

- “继续把 `DSSDVerifierExecutionRequest` 接到 `GPUModelRunner` 的真实 spec-decode 输入准备路径，先做单请求 replay 方案，不追求优化”

## 10. 一句话总结

这个分支现在已经把 DSSD 的配置面、边云控制面、session/utility path、单轮 edge 流程、verifier decision 逻辑都搭好了；真正还缺的核心，是把 verifier 和 draft 的 placeholder GPU hook 换成真实模型前向，尤其是 verifier 的 spec-decode 真实路径接入。
