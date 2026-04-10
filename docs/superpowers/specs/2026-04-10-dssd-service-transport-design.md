# DSSD Service/Transport 第一版设计

**目标**

在不改动 `vllm/dssd/edge/` 和 `vllm/dssd/verifier/` 现有内核职责的前提下，补齐 DSSD 第一版正式系统层：

- `edge service`
- `verifier service`
- `transport`
- `fake network`

第一版必须能在单机双卡环境下，以本地进程内消息通道模拟 edge 与 verifier 的交互，并支持固定延迟、固定带宽的网络模型。

**范围**

- 单请求
- 单会话
- decoder-only 文本模型
- 固定 `gamma`
- 本地进程内 transport
- 固定单向延迟
- 固定单向带宽
- edge 端本地 resample
- 端到端 correctness 优先

**不做的事**

- 不做真实 socket / HTTP / gRPC
- 不做多会话并发调度
- 不做网络抖动、丢包、乱序、重传
- 不做 batch > 1
- 不在这一步追 verifier 高性能路径收敛
- 不修改现有 edge/verifier 内核状态机语义

## 总体方案

第一版采用“四层分离”方案：

1. `edge/verifier` 内核继续只负责各自单端执行流
2. `protocol` 负责定义双方交换的消息
3. `service` 负责把协议消息映射到内核调用
4. `transport/fake network` 负责消息传输与时延模拟

这样做的目的有两个：

1. 把 DSSD 协调逻辑与单端内核解耦，避免把网络/协议语义灌进 `edge` 或 `verifier` 内核
2. 为后续替换成本地进程外 RPC 保留清晰边界，届时只需要替换 transport，而不需要重写 service 或内核

## 代码落点

建议正式代码新增以下目录：

- `vllm/dssd/protocol/`
  - `types.py`
- `vllm/dssd/service/`
  - `edge_service.py`
  - `verifier_service.py`
- `vllm/dssd/transport/`
  - `fake_network.py`
  - `local_channel.py`
  - `verifier_transport.py`

建议新增测试目录：

- `tests/dssd/protocol/`
- `tests/dssd/service/`
- `tests/dssd/transport/`

现有目录保持职责不变：

- `vllm/dssd/edge/`
  - 只负责本地 decode/prefill/draft/rollback/commit 内核
- `vllm/dssd/verifier/`
  - 只负责 target 侧 `open_session / verify_round / close_session`

## 协议层设计

`protocol/types.py` 负责定义 edge 与 verifier 之间交换的正式消息类型。

第一版只需要 3 组请求/响应：

### 1. `open_session`

请求字段：

- `req_id`
- `prompt_token_ids`
- `sampling_params`
- `lora_request`

响应字段：

- `req_id`
- `bootstrap_token_id`

### 2. `verify_round`

请求字段：

- `req_id`
- `committed_token_id`
- `draft_token_ids`
- `draft_q_values`

响应字段：

- `req_id`
- `accepted_len`
- `bonus_token_id | rejected_target_logits`

这里保持和 verifier 正式内核一致：

- 全接受时返回 `bonus_token_id`
- 拒绝时返回 `rejected_target_logits`

### 3. `close_session`

请求字段：

- `req_id`

响应字段：

- `req_id`
- `ack=True`

协议层不引入传输细节字段，不包含延迟、带宽、时间戳等网络语义。

## Verifier Service 设计

`service/verifier_service.py` 只负责“协议消息 -> verifier 内核调用 -> 协议响应”。

职责如下：

- `open_session(...)`
  - 调 `VerifierDecodeEngine.open_session(...)`
  - 返回 `bootstrap_token_id`
- `verify_round(...)`
  - 根据 `req_id` 找到 session
  - 调 `VerifierDecodeEngine.verify_round(...)`
  - 返回正式 round result
- `close_session(...)`
  - 调 `VerifierDecodeEngine.close_session(...)`

第一版 verifier service 不关心网络、不关心 edge 侧状态、不做调度。

如果后续需要服务循环，可以在这一层增加 `serve_forever()`，但第一版不强制要求独立常驻线程模型。只要 transport 能同步地把请求送进 verifier service 并拿到返回值即可。

## Edge Service 设计

`service/edge_service.py` 负责完整的 DSSD 协调流程，是系统层的主状态机。

第一版建议提供两个公开入口：

### 1. `open_session(...)`

执行顺序：

1. 调 verifier transport 的 `open_session`
2. 调 edge decode engine 的 `open_session`
3. 调 edge decode engine 的 `prefill`
4. 用 verifier 返回的 `bootstrap_token_id` 完成 edge 侧 bootstrap

### 2. `generate(...)`

第一版 `generate()` 负责串起完整闭环：

1. `open_session`
2. edge 本地 `draft(gamma)`
3. 通过 transport 发 `verify_round`
4. 根据 verifier 响应：
   - 全接受：commit `bonus_token`
   - 拒绝：rollback rejected draft，并基于 `rejected_target_logits` 本地 resample，再 commit external token
5. 命中 EOS 后停止
6. `close_session`

这层的核心约束是：

- edge service 只通过 transport 接触 verifier
- edge service 不直接调用 verifier 内核
- edge service 保留 edge 侧 resample 职责

## Transport 设计

第一版 transport 使用本地进程内同步 transport，而不是网络协议栈。

### `local_channel.py`

提供最小双端消息通道抽象，职责是：

- 发送一条消息
- 接收一条消息
- 不改变消息内容
- 不关心业务字段

第一版可以用标准库队列或等价结构实现。

### `verifier_transport.py`

对 edge service 暴露同步调用接口：

- `open_session(...)`
- `verify_round(...)`
- `close_session(...)`

内部流程是：

1. 将协议请求编码为本地消息对象
2. 交给 fake network 计算传输开销
3. 送入 verifier service
4. 再通过 fake network 模拟响应返回
5. 返回协议响应给 edge service

`verifier_transport.py` 是 edge 侧唯一可见的 verifier 访问入口。

## Fake Network 设计

`transport/fake_network.py` 负责网络开销模拟。

第一版只支持两个参数：

- `fixed_latency_ms`
- `bandwidth_bytes_per_s`

不支持：

- 抖动
- 丢包
- 乱序
- 重传
- 队列竞争

### 延迟模型

单向传输耗时：

`transfer_time = fixed_latency + payload_bytes / bandwidth`

往返通信由请求和响应各自独立计算，因此一次 `verify_round` 的 transport 开销为：

`request_transfer_time + response_transfer_time`

### payload 大小语义

第一版不追求二进制序列化精确开销，但必须满足两个约束：

1. 不同消息大小应体现不同传输时间
2. 拒绝时返回整行 `rejected_target_logits` 的响应应显著大于全接受时只返回 `bonus_token_id` 的响应

因此建议使用统一的“可估算字节数”规则：

- token id 按整数大小估算
- float/logits tensor 按 dtype 对应字节数估算
- Python 对象额外开销忽略

这样可以保证网络模型简单但方向正确。

## 端到端数据流

第一版完整数据流如下：

1. 用户请求进入 `edge_service.generate`
2. `edge_service` 调 `verifier_transport.open_session`
3. `verifier_transport` 经 `fake_network` 将请求同步送入 `verifier_service`
4. `verifier_service` 调 `VerifierDecodeEngine.open_session`
5. verifier 返回 `bootstrap_token_id`
6. `edge_service` 调 `EdgeDecodeEngine.open_session + prefill`
7. 进入循环：
   - edge 本地 `draft(gamma)`
   - 通过 transport 发 `verify_round`
   - verifier service 调 `verify_round`
   - edge 根据响应 commit / rollback / resample
8. 命中 EOS 后：
   - `edge_service.close_session`
   - `verifier_transport.close_session`
   - `verifier_service.close_session`

这一层要明确保持同步语义。第一版不引入后台线程、事件循环或异步 future。

## 错误处理边界

第一版错误处理保持最小且明确：

- `req_id` 不存在时抛错
- `verify_round` 参数不合法时抛错
- `close_session` 重复关闭时抛错或显式拒绝
- transport 不吞异常，底层 service/engine 报错应直接向上传播

第一版不做容错恢复，也不做网络级错误模拟。

## 测试与验收

测试建议分 3 层。

### 1. transport / fake network 单元测试

验证：

- 固定延迟与带宽换算正确
- 小消息与大消息的传输时间不同
- `bonus_token_id` 响应比 `rejected_target_logits` 响应更小
- local channel 能正确传递三类请求与响应

### 2. service 单元测试

验证：

- verifier service 能把协议请求正确映射到 verifier 内核
- edge service 能在全接受分支正确 commit `bonus_token`
- edge service 能在拒绝分支正确 rollback 并基于 `rejected_target_logits` 本地 resample
- close_session 会同时清理 edge 与 verifier 会话

### 3. 最小端到端集成测试

验证：

- in-process `edge service + verifier service + transport + fake network` 能跑通完整会话
- `temperature=0`、固定 `gamma` 下，DSSD 输出与 target-only 基线一致
- 使用固定延迟和固定带宽时，请求与响应确实会产生可观测等待

### 验收标准

- 正式代码中存在独立的 `protocol / service / transport` 边界
- edge service 不直接依赖 verifier 内核
- fake network 至少支持固定延迟和固定带宽
- 单请求端到端链路可跑通
- `temperature=0` 下 DSSD 输出与 target-only 基线一致

## 风险与约束

- 第一版 transport 是同步本地实现，因此只能验证系统语义与固定网络开销，不能代表真实分布式调度行为
- `rejected_target_logits` 体积较大，若 payload 大小估算过粗，网络模拟的对比度可能不够明显
- edge service 的 EOS 与 rollback 语义必须直接复用当前正式内核约束，不能重新发明一套状态推进逻辑
- 当前阶段先保证 correctness 和接口清晰度，不把性能问题混入 service/transport 测试
