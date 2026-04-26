# DSSD Thesis Experiment List

本文档记录毕业论文《基于边云协同的大小模型联合推理技术研究》的规范化实验列表。
后续实验结果、原始日志和 JSON 文件应保存到 `benchmarks/dssd/results/`，并在对应实验项下补充结果路径。

## 论文实验主线

本文的主要贡献不是提出新的解码算法，而是围绕 DSSD 构建一个基于 vLLM 的边云协同大小模型联合推理系统，并通过系统实现和实验评估验证该类方法在真实推理框架中的可行性、性能收益和适用边界。

实验分为两大部分：

1. 系统正确性与可行性验证。
2. 边云协同场景下的性能研究。

## 1. 系统正确性与可行性验证

### 1.1 模块化测试验证

**实验目的**

验证 DSSD 系统各模块在功能层面的实现正确性，覆盖协议结构、采样逻辑、edge/verifier 引擎、服务层、通信层、网络模拟和启动入口等核心路径。

**建议命令**

```bash
pytest tests/dssd -v --import-mode=importlib
pytest tests/benchmarks/test_dssd_system_benchmark.py -v
pytest tests/benchmarks/test_dssd_local_engine_benchmark.py -v
```

**本次结果**

结果记录：`benchmarks/dssd/results/experiment-1.1-module-tests-2026-04-25.md`

| 命令 | 结果 |
|---|---|
| `pytest tests/dssd --ignore=tests/dssd/service/test_real_smoke.py -q --import-mode=importlib` | `221 passed` |
| `pytest tests/dssd/service/test_real_smoke.py -q --import-mode=importlib` | `9 passed` |
| `pytest tests/benchmarks/test_dssd_system_benchmark.py -v` | `19 passed` |
| `pytest tests/benchmarks/test_dssd_local_engine_benchmark.py -v` | `13 passed` |

**论文指标**

| 测试类别 | 覆盖内容 | 测试结果 |
|---|---|---|
| 协议结构测试 | open session、verify round、close session 请求与响应结构 | 通过 |
| 采样逻辑测试 | draft 采样、贪心验证、拒绝分支、residual resample | 通过 |
| Edge 引擎测试 | prefill、draft、commit、rollback、session close | 通过 |
| Verifier 引擎测试 | open session、verify round、target-only generate、状态恢复 | 通过 |
| 服务层测试 | DSSD 主循环、target-only 路径、acceptance 统计 | 通过 |
| 通信层测试 | HTTP transport、in-process transport、网络模拟 | 通过 |
| 启动入口测试 | edge/verifier server 参数解析、接口返回、异常处理 | 通过 |

**预期结论**

系统各核心模块能够按预期完成独立功能和模块间协作，为后续端到端推理实验提供实现层面的正确性基础。

### 1.2 贪心解码输出一致性验证

**实验目的**

验证 DSSD 在贪心解码条件下不改变 target 模型输出。若 DSSD 的状态同步、接受拒绝判断、提交回滚逻辑正确，则 DSSD 输出 token 序列应与 target-only 输出 token 序列完全一致。

**唯一指标**

```text
输出一致率 = DSSD 输出 token 序列与 target-only 输出 token 序列完全一致的请求数 / 总请求数
```

**建议设置**

| 项目 | 设置 |
|---|---|
| 对比模式 | DSSD vs Target-only |
| 解码方式 | greedy decoding |
| temperature | 0.0 |
| 输出长度 | 64 或 128 tokens |
| gamma | 1、2、4、8 |
| Prompt 数量 | 5 到 10 条固定 prompt |
| 模型组合 | OPT-125M / OPT-6.7B |

**本次结果**

结果记录：`benchmarks/dssd/results/experiment-1.2-greedy-output-consistency-2026-04-25.md`

正式结果采用 OPT-125M / OPT-6.7B 模型组合，使用 5 条固定 prompt、`prompt_len=128`、`output_tokens=64`、`temperature=0.0`。恢复原始 packed verifier 路径后，OPT 组合在 `gamma=1,2,4,8` 上仍达到 100% 输出一致率。

**论文结果表**

| 模型组合 | gamma | Prompt 数量 | 输出一致率 |
|---|---:|---:|---:|
| OPT-125M / OPT-6.7B | 1 | 5 | 100% |
| OPT-125M / OPT-6.7B | 2 | 5 | 100% |
| OPT-125M / OPT-6.7B | 4 | 5 | 100% |
| OPT-125M / OPT-6.7B | 8 | 5 | 100% |

**预期结论**

在正式验证采用的 OPT-125M / OPT-6.7B 组合上，DSSD 模式与 target-only 模式的输出一致率达到 100%，说明本文实现的 DSSD 系统在该受控配置下没有改变 target 模型的贪心解码结果，验证了端到端推理流程的语义正确性。

## 2. 边云协同场景下的性能研究

### 2.1 基础性能对比与 gamma sweep

**实验目的**

在本地双端 edge-verifier 设置下，对比 DSSD 与 target-only 的吞吐表现，并分析草稿长度 `gamma` 对性能的影响。

**对比模式**

| 模式 | 含义 |
|---|---|
| Target-only | edge 只转发请求，由 verifier 侧 target 模型直接生成 |
| DSSD | edge 小模型生成 draft tokens，verifier 大模型验证并返回接受结果 |

**建议固定设置**

| 项目 | 建议值 |
|---|---|
| prompt_len | 128 |
| output_tokens | 256 |
| temperature | 0.0 |
| repeats | 20 |
| warmup_repeats | 2 |
| model runner | v1 |
| execution | `--no-enforce-eager` |
| 网络模拟 | 关闭 |
| prompt 文件 | `benchmarks/dssd/prompts/high_acceptance_opt.jsonl` |
| prompt 选择 | 使用文件中的第一条 prompt，并截断为 `128` tokens |

**Prompt 说明**

实验 2.1 使用 `benchmarks/dssd/prompts/high_acceptance_opt.jsonl` 中第一条连续说明文 prompt。该 prompt 内容围绕 scientific measurement，语言结构稳定、重复性较强、可预测性较高，适合在较高 draft acceptance 条件下观察 `gamma` 对 DSSD 吞吐的影响。

本实验不混用多个 prompt。原因是 2.1 的目标是分析 `gamma` 对基础性能的影响，应尽量固定 prompt 分布，避免 prompt 接受率变化干扰实验结论。多 prompt、不同接受率和不同模型组合的影响放到实验 2.2 单独分析。

**变量**

```text
gamma = 1, 2, 4, 6, 8
```

**论文结果表**

**本次结果**

结果记录：`benchmarks/dssd/results/experiment-2.1-basic-performance-gamma-sweep-2026-04-25.md`

| 模型组合 | 模式 | gamma | Draft Acceptance | Avg Accepted Len / Round | Server token/s | Speedup |
|---|---|---:|---:|---:|---:|---:|
| OPT-125M / OPT-6.7B | Target-only | - | - | - | 103.29 | 1.00x |
| OPT-125M / OPT-6.7B | DSSD | 1 | 93.94% | 0.94 | 133.72 | 1.29x |
| OPT-125M / OPT-6.7B | DSSD | 2 | 96.02% | 1.92 | 191.86 | 1.86x |
| OPT-125M / OPT-6.7B | DSSD | 4 | 89.29% | 3.57 | 261.24 | 2.53x |
| OPT-125M / OPT-6.7B | DSSD | 6 | 84.11% | 5.05 | 313.26 | 3.03x |
| OPT-125M / OPT-6.7B | DSSD | 8 | 81.99% | 6.56 | 360.81 | 3.49x |

**预期结论**

在高接受率场景下，DSSD 应能明显快于 target-only。随着 `gamma` 增大，验证轮数减少，verifier round overhead 被摊薄，吞吐通常会上升；但当接受率下降时，过大的 `gamma` 可能导致 draft 计算浪费，收益下降。

### 2.2 接受率对性能的影响

**实验目的**

说明 DSSD 的性能收益并非无条件成立，而是依赖 draft 模型与 target 模型的一致性、prompt 类型和接受率。

**建议设置**

| Prompt 类型 | 预期接受率 | 作用 |
|---|---:|---|
| 高接受率连续文本 | 高 | 展示 DSSD 加速上限 |
| 普通或不稳定 prompt | 中低 | 展示 DSSD 收益边界 |

本实验使用两个固定 prompt 文件：

- 高接受率连续文本：`benchmarks/dssd/prompts/high_acceptance_opt.jsonl`
- 普通或不稳定 prompt：`benchmarks/dssd/prompts/general_mixed_prompt.jsonl`

对 Qwen3-0.6B / Qwen3-8B，结果表按实测 draft acceptance 归类：`general_mixed_prompt.jsonl` 作为高接受率条件，`high_acceptance_opt.jsonl` 作为普通 prompt 对照。

**建议模型组合**

| 模型组合 | 预期 |
|---|---|
| OPT-125M / OPT-6.7B | 高接受率 prompt 下效果较好 |
| Qwen3-0.6B / Qwen3-8B | 普通 prompt 下收益可能较弱 |

**论文结果表**

| 模型组合 | Prompt 类型 | 最优 gamma | Draft Acceptance | Server token/s | Speedup vs Target-only |
|---|---|---:|---:|---:|---:|
| OPT-125M / OPT-6.7B | 高接受率 | 8 | 81.99% | 360.81 | 3.49x |
| OPT-125M / OPT-6.7B | 普通 prompt | 8 | 45.39% | 218.02 | 2.11x |
| Qwen3-0.6B / Qwen3-8B | 高接受率 | 4 | 69.49% | 153.94 | 1.65x |
| Qwen3-0.6B / Qwen3-8B | 普通 prompt | 4 | 55.00% | 130.51 | 1.38x |

**结果记录**

- `benchmarks/dssd/results/experiment-2.2-acceptance-performance-2026-04-26.md`

**预期结论**

DSSD 的性能收益与 draft acceptance rate 强相关。当小模型生成内容更容易被大模型接受时，单轮验证可以提交更多 token，系统吞吐提升明显；当接受率较低时，频繁拒绝会导致 draft 计算浪费和更多通信轮次，DSSD 收益下降甚至可能低于 target-only。

### 2.3 网络条件敏感性实验

**实验目的**

研究边云协同链路中的固定延迟和带宽限制对 DSSD 性能的影响，分析 DSSD 在不同网络条件下的适用边界。

**建议固定设置**

| 项目 | 建议值 |
|---|---|
| 模型组合 | OPT-125M / OPT-6.7B |
| Prompt 类型 | 高接受率 prompt |
| gamma | 选择 2.1 中的最优值，例如 8 |
| output_tokens | 256 |
| temperature | 0.0 |

**变量**

```text
单向延迟：0ms, 10ms, 20ms, 50ms
带宽：100 Mbps 或不限制
```

**论文结果表**

| 单向延迟 | Target-only token/s | DSSD token/s | DSSD Speedup | Draft Acceptance |
|---:|---:|---:|---:|---:|
| 0ms | 待填 | 待填 | 待填 | 待填 |
| 10ms | 待填 | 待填 | 待填 | 待填 |
| 20ms | 待填 | 待填 | 待填 | 待填 |
| 50ms | 待填 | 待填 | 待填 | 待填 |

**预期结论**

随着链路延迟增加，DSSD 会受到更明显影响，因为它需要多轮 edge-verifier 交互。较大的 `gamma` 可以减少验证轮数，从而缓解固定往返延迟带来的损失。当延迟过高或接受率不足时，DSSD 收益可能下降，说明边云协同推理需要根据网络状态选择模式或调整 `gamma`。

## 结果整理规范

每次实验完成后，应记录以下内容：

| 字段 | 说明 |
|---|---|
| 日期 | 实验执行日期 |
| Git commit | 当前代码版本 |
| 命令 | 完整命令，包含环境变量 |
| 模型路径 | edge model 和 verifier model |
| 参数 | prompt_len、output_tokens、gamma、temperature、网络参数等 |
| 原始结果路径 | `benchmarks/dssd/results/` 下的 JSON 或 log |
| 摘要结果 | token/s、speedup、acceptance、输出一致率等 |
| 备注 | 异常、失败、复跑原因、与预期不一致之处 |

建议所有原始结果文件命名包含：

```text
实验类型-模型组合-prompt类型-prompt长度-output长度-gamma-网络条件-日期.log/json
```
