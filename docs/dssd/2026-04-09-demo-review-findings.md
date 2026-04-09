# DSSD Demo Review Findings

日期：2026-04-09

范围：
- `docs/dssd/edge_demo/`
- `docs/dssd/verifier_demo/`
- `docs/dssd/dssd-engine-fake.md`
- `docs/dssd/DSSD-algorithm.md`

目标：
- 重点检查算法逻辑正确性
- 重点检查是否满足“输出分布等于 target model”
- 重点检查是否保留 vLLM fast path 的性能边界
- 不重点关注异常处理和边角防御代码

## Findings

### 高优先级

1. verifier 端 prefill 的 KV block 生命周期还没闭合，`open_session` 仍然不是一条可照搬的 vLLM 实现路径。

- 位置：
  - [verifier_demo/scheduler.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/verifier_demo/scheduler.py#L26)
  - [verifier_demo/engine.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/verifier_demo/engine.py#L79)
- 现状：
  - `VerifierSchedulerAdapter.allocate_blocks()` 仍然直接返回空 block。
  - `open_session()` 随后用这组空 `block_ids` 进入 prefill。
- 问题：
  - demo 仍然漏掉了 verifier prefill 的 `KVCacheManager.allocate_slots(...)`。
  - 这不是单纯“未优化”，而是 target 侧首轮 block 分配语义仍未补全。
- 风险：
  - 如果后面按这版 demo 直接映射真实实现，verifier 首轮 prefill 会在 block 生命周期上走偏。

2. edge 端在“最近提交段里遇到 EOS 后回滚尾部 token”时，会把外部 token 也按“本地已计算 token”回退，导致 `num_computed_tokens` 镜像少 1。

- 位置：
  - [edge_demo/service.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/edge_demo/service.py#L121)
  - [edge_demo/service.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/edge_demo/service.py#L189)
  - [edge_demo/types.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/edge_demo/types.py#L97)
  - [edge_demo/state_bridge.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/edge_demo/state_bridge.py#L44)
- 现状：
  - `_commit_verify_result()` 把本轮 `committed_count` 固定为 `accepted_len + 1`。
  - `_rollback_tokens_after_recent_eos()` 如果在这段里遇到 EOS，会把 EOS 后尾部都交给 `decode_engine.rollback()`。
  - 但 `commit_external_token()` 注入的 bonus/resample token 是 `computed_delta=0`。
  - `EdgeSession.rollback()` 却是无差别 `num_computed_tokens -= count`。
- 问题：
  - 一旦出现“accepted draft prefix 里先到 EOS，而最后一个新 token 是 external bonus/resample”的轮次，edge 的逻辑长度镜像就会错一格。
- 风险：
  - 后续 decode 的位置、长度或 block 申请语义会被污染。

### 中优先级

3. verifier 的 accept/reject 判定已经离开了你要保的 vLLM 高性能尾路径，当前实现没法支撑“单端 verify 不弱于原始 vLLM”。

- 位置：
  - [verifier_demo/sampler.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/verifier_demo/sampler.py#L97)
  - [verifier_demo/sampler.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/verifier_demo/sampler.py#L112)
- 现状：
  - 每个 speculative step 都会做 `.item()`。
  - 每步都会创建 CPU `torch.Generator`。
  - accept/reject 的 uniform random 也在 CPU 上逐步生成。
- 判断：
  - 这仍然是合法的 accept/reject 分布实现，没有直接破坏算法正确性。
  - 但性能上已经把 verifier round 的后处理从 GPU 连续路径拉回 CPU 逐步循环。
- 风险：
  - 不满足“verifier 单独 verify/decode 不弱于原始 vLLM”的目标。

4. edge 端当前保存 `Q_j` 的方式不满足文档里的性能目标。

- 位置：
  - [edge_demo/sampler.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/edge_demo/sampler.py#L16)
  - [edge_demo/types.py](/home/zz/workspace/vllm/.worktree/dssd-v1/docs/dssd/edge_demo/types.py#L59)
- 现状：
  - draft 每一步都会 `detach().clone()` 一整行 full-vocab `processed_logits`。
  - 这些大张量被挂在 Python `list` 上。
- 判断：
  - DSSD 算法确实要求 edge 保留 `Q_j` 以便 reject 后本地 resample。
  - 但按这版 demo 的写法，每步都会产生 full-vocab clone 和 Python 对象分配。
- 风险：
  - 和“edge 单独 decode 不弱于原始 vLLM”冲突。

## 总体结论

- 主干 accept/reject 语义、`[committed] + draft` 的 verifier 输入形态、以及 reject 时由 edge 本地 resample 的职责划分，整体上已经和伪代码基本对齐。
- 当前最值得优先修的，不是异常处理，而是：
  - verifier prefill block 分配
  - edge 的 EOS 后回滚计数语义
  - verifier accept/reject 的 CPU 尾路径
  - edge 的 `Q_j` 缓存方式

## 当前验证说明

本次 review 参考了现有 demo 测试，但这些测试主要是 AST/语法级检查，不能证明运行时正确性：

- `docs/dssd/edge_demo/test_edge_demo_ast.py`
- `docs/dssd/verifier_demo/test_close_session_ast.py`
- `docs/dssd/verifier_demo/test_gamma_source_demo.py`
- `docs/dssd/verifier_demo/test_postprocess_semantics_demo.py`
- `docs/dssd/verifier_demo/test_rng_semantics_demo.py`

因此，上面的 findings 以代码审查和与伪代码/算法文档对照为准，而不是以“测试是否通过”为准。
