## 背景
我在做我的本科毕设，要基于vllm来实现分布式投机采样原型系统，即DSSD，算法参考在 /home/zz/workspace/vllm/docs/DSSD-algorithm.md

## 实现要求
 - 尽量简单，快速跑起来，nano风格，独立于vllm库开发，不用考虑合并，不用规范，不用错误处理。但要有测试以提高开发效率和保证正确性
 - 要完整利用vllm执行流的性能，比如pageattention,cudagraph，最终edge端和verifier端单独decode速度必须不弱于原始vllm引擎
 - 只考虑单请求
 - 该系统理想中的运行方式是：先在云端启动verifier，边端启动edge，用户请求发送到edge，edge再连接verifier执行DSSD逻辑，然后返回结果给用户。
 - 对于edge和verifier的通信要实现一个模拟网络模块，我要在单机双卡的机器上同时部署两端来做实验，要能模拟网络延迟、带宽信息
 - 必须保证DSSD输出分布等于target模型的分布


## 伪代码

DSSD流程
1.  edge收到prompt请求，分词，开启会话{发送ids到verifier}，prefill
    verifier接受收ids，返回确认，prefill
2.  edge起草，发送ids+q
    veri接收ids+q，进行一次推理，校验，返回接受数+新采样token/拒绝位置P，回退kvache
    edge接收结果，回退kvcache，用新token或自己resample作为下个token开始起草
    ...
    edge接收结果，检查是否有end，有则返回


```python
# edge
class DSSDEdgeEngineCore:
    def is_end(self, ids):
        return any(t == self.tokenizer.eos_token_id for t in ids)

    def resample(self, q_dist, rsp):
        # Phase 5: residual distribution resampling
        residual = max(0, rsp.p_dist - q_dist)  # elementwise, shape=(|V|,)
        return sample(residual / residual.sum())

    def generate(self, prompt):
        ids = self.tokenizer.encode(prompt)
        prompt_len = len(ids)
        if not self.verifier.prefill(ids):
            return None
        self.prefill(ids)

        draft, q_xi, q_dists = [], [], []
        while True:
            rsp = self.verifier.response()
            reject_count = len(draft) - rsp.accepted_len
            if reject_count > 0:
                self.kvcache.evict(reject_count)
                ids = ids[:-reject_count]

            new_token = rsp.next_token if reject_count == 0 else self.resample(q_dists[rsp.accepted_len], rsp)
            ids.append(new_token)

            if self.is_end(ids[-rsp.accepted_len-1:]):
                while ids[-1] != self.tokenizer.eos_token_id:
                    ids.pop()
                break

            draft, q_xi, q_dists = [], [], []
            next_token = new_token
            for _ in range(gamma):
                next_token, q_dist = self.model_executor.inference(next_token)
                draft.append(next_token)
                q_xi.append(q_dist[next_token])
                q_dists.append(q_dist)
                ids.append(next_token)

            self.verifier.verify(new_token, draft, q_xi)


        self.verifier.clear()
        self.clear()

        return self.decode(ids[prompt_len:])

# verifier
class DSSDVerifierEngineCore:
    
    def verify_loop():
        
        while true:
            ids = self.edge.request()
            next_token = self.prefill(ids)
            self.edge.response(next_token=next_token, accepted_len=0)

            while true:
                request = self.edge.request()
                if request.clear():
                    break
                draft = request.draft()
                p_dists = self.model_executor.inference([request.new_token()]+ draft)
                verify_output = self.verify(draft, request.q_values(), p_dists)
                self.kvcache.evict(len(draft)-verify_output.accepted_len)

                self.edge.response(verify_output)

            self.kvcache.clear()

    def verify(self, draft, q_values, p_dists):
        # Phase 3: accept/reject loop
        accepted_len = 0
        for j in range(len(draft)):
            x_j = draft[j]
            p_j = p_dists[j][x_j]
            q_j = q_values[j]
            if random() < min(1, p_j / q_j):  # accept
                accepted_len += 1
            else:                               # reject
                return VerifyOutput(accepted_len=accepted_len, p_dist=p_dists[j])
        # all accepted: sample bonus token from P_{γ+1}
        bonus_token = sample(p_dists[len(draft)])
        return VerifyOutput(accepted_len=accepted_len, next_token=bonus_token)
```

## Edge端架构设计要点

- edge 本质上是：`本地常驻 decode engine + verifier transport + 协调层`。
- 本地 decode 必须始终走 vLLM 原生 `query_len=1` 快路径，不能退回高层 `generate()`。
- `KV` block 的分配、追加、释放必须走真实 `KVCacheManager` 语义，不能手搓 `block_ids`。
- reject 后第一版只做逻辑回退，不做物理删 KV；同时先关闭 penalty，避免回退额外状态。
- `Q_j` 必须保留在 GPU 上，并用本轮 scratch buffer 复用，避免每步创建新 tensor。
- 能尽量守住 edge 单端性能，但端到端性能仍然会受 verifier RTT 和 reject 下行 `P_j` 影响。

## Verifier端架构实现

- verifier 的最小参考实现已经单独落在 `docs/dssd/verifier_demo/`。
- 建议直接对照下面这些文件阅读，而不是再维护一份文档内接口骨架：
  - `service.py`
  - `engine.py`
  - `state_bridge.py`
  - `scheduler.py`
  - `sampler.py`
  - `types.py`

## Verifier端架构设计要点

### 1. 总体结论

- verifier 必须是一个常驻的 verify 服务，不能退化成普通 `generate()` 接口。
- verifier 唯一值得保的高性能路径，是每轮只跑一次 `query_len=1+gamma` 的 forward。
- 后续统一轮次的输入语义固定为 `[committed_token] + draft`。

### 2. 核心状态机

- `open_session`
  - 先做 prompt prefill。
  - 然后由 verifier 从 target 模型分布采样 prompt 后的第一个 token。
  - 这个 bootstrap token 只返回给 edge，不在 `open_session` 阶段 commit 进 verifier prefix。

- `verify_round`
  - edge 把本轮 `committed_token`、`draft_token_ids`、`draft_q_values` 发给 verifier。
  - verifier 只把 `committed_token` 写进 `last_sampled_tokens`，把 `draft_token_ids` 写进 speculative suffix，对应一次 `[committed_token] + draft` forward。
  - 本轮真正 commit 的只有：
    - `committed_token`
    - accepted draft prefix
  - 如果全接受，`bonus_token` 旁路返回给 edge，不写进当前轮 prefix。
  - 如果拒绝，返回被拒位置的 target logits，由 edge 本地 resample。

- `close_session`
  - 释放 request state、round state 和 KV blocks。

### 3. committed_token 与 postprocess 语义

- `committed_token` 是本轮输入，但也是本轮要确认进入前缀的 token。
- 所以它不能直接混进 `sampled_token_ids`，否则会和 vLLM `postprocess()` 的“本轮新输出”语义打架。
- 正确做法是：
  - 先手动把 `committed_token` 写入 request state
  - 再让 `postprocess()` 只处理 accepted draft prefix
- 这样：
  - `last_sampled_tokens`、`all_token_ids`、`total_len` 的推进语义仍然和 vLLM 兼容
  - `bonus_token` 和 reject 后的新 token 不会提前污染 verifier 主状态

### 4. 会受到 vLLM 限制的地方

- verifier 依赖 vLLM 的内部 request state：
  - `last_sampled_tokens`
  - `draft_tokens`
  - `num_computed_tokens`
  - `total_len`
- 这意味着第一版基本要直接操作 `GPUModelRunner.req_states`，没有稳定公共 API 可用。
- 另外还有两个明确限制：
  - `GPUModelRunner` 当前把 speculative query 形态和内部 speculator 绑在一起，DSSD 需要把这两者解耦
  - 现有 `SamplerOutput` 不足以表达 DSSD verifier 的旁路结果，还需要额外返回 `accepted_len`、`bonus_token_id` 或 `rejected_target_logits`

### 5. 性能边界

- verifier 侧能争取保住性能的前提是：
  - `gamma` 固定
  - 每轮只有一次 `1+gamma` forward
  - 新增逻辑只放在 bootstrap 和 sampler 尾段
- 当前 demo 的问题不在于“用了 CPU”，而在于在 accept/reject 热路径里引入了细粒度 host-device 同步。
- 真实实现要保性能，这段 accept/reject 尾处理应重新放回 GPU：
  - `processed_logits` 保持在 GPU
  - `draft_q_values` 在 verifier 侧转成 GPU tensor
  - 用 Triton/CUDA kernel 完成 `p/q` 比较和 `tl.rand(seed, pos)` 风格的 accept/reject 判定
  - 全接受时 bonus token 继续复用现有 `gumbel_sample()`
  - 拒绝时再从 GPU 取被拒位置对应的 target logits 返回给 edge
- 不能先验保证性能的情况主要有：
  - attention backend 只支持 `UNIFORM_SINGLE_TOKEN_DECODE`
  - `gamma` 动态变化
  - penalty / bad words / 更复杂采样约束打开
  - 多模态、encoder-decoder、pipeline parallel

### 6. 第一版实现边界

- 第一版先限制为：
  - 单请求
  - decoder-only 文本模型
  - 固定 `gamma`
  - 无 penalty / 无 bad words / 无复杂结构化约束
  - reject 时先返回 verifier target logits，不在 verifier 本地做 recover/resample

- 第一版 verifier 最该优先保证的是：
  - `open_session` 返回的 bootstrap token 严格来自 target 模型分布，但不在该阶段 commit
  - 每轮 verify 真正落到一次 `1+gamma` forward
  - 只把已确认前缀推进到 verifier 主状态
  - `bonus_token` 和 `rejected_target_logits` 都通过旁路返回，不污染 runner 主状态
