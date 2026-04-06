# DSSD Global Roadmap And Agent Context Document Design

## Goal

在 `dssd-vllm-codex` 分支内新增一个长期维护的全局路线图文档，用来同时承担两件事：

- 记录 DSSD 项目的当前开发进度、当前确认主线、候选方向、阶段路线，以及每次任务完成后的路线重排结果
- 作为持久 agent 上下文，让一个新会话的 agent 在只阅读这一份文档的前提下，也能尽快进入开发状态

这个文档不是一次性的 implementation plan，而是后续开发默认参考的执行导航板和启动上下文。

## Why This Document Exists

当前仓库里已经有：

- 设计基线文档
- 一次性的 implementation plan
- 单次 handoff 文档

这些文档各自有用，但都不承担“持续更新的总控板”角色。随着 DSSD 开发推进，缺少一个统一位置来回答下面这些问题：

- 当前项目已经完成到什么程度
- 哪些工作已经验证过，应该直接继续做
- 哪些方向只是备选，还不能作为当前承诺
- 最近一轮开发为什么调整了路线
- 当前下一步到底应该做什么

因此需要新增一个全局路线图文档，把“状态记录”“路线决策”和“新会话启动上下文”收拢到一个长期维护的文件里。

## Document Role

这个路线图文档承担五个固定职责：

1. 记录当前 DSSD 项目的高层开发状态。
2. 区分“已验证可继续开发”和“只是候选方向”。
3. 维护当前阶段路线和手边任务队列。
4. 记录每次任务完成后路线如何被重排。
5. 为新会话 agent 提供最小但足够的启动上下文。

它不替代已有的设计文档，也不替代细化 implementation plan。

- 设计文档负责说明系统形态和模块设计。
- implementation plan 负责拆分某一阶段的详细执行步骤。
- 全局路线图负责跨阶段地维护“现在在哪里、接下来做什么、为什么这么排”，并在文档开头提供直接可用的 agent 启动信息。

## Proposed File

新增文件：

- `docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`

文件位置放在 `plans/` 下，而不是 `handoffs/` 或 `specs/` 下，原因是它本质上是一个长期更新的执行文档，而不是一次性交接或纯设计说明。

## Required Structure

全局路线图文档应包含以下固定章节。

### 1. `Quick Start for New Agents`

这一节放在文档最开头，目标是让一个新会话 agent 在两到三分钟内进入开发状态。内容要保持简洁，但必须自包含到足以启动工作。至少包括：

- 当前 worktree 路径
- 当前 branch
- 当前远程仓库信息
- Python / venv / `uv` 使用约定
- 代理开启约定
- 大文件和模型权重存放约定
- 当前推荐的验证命令入口
- 当前最需要避免的误判或误操作

这一节不记录瞬时 shell 输出，也不复制大段历史过程，只保留会直接影响“如何开始开发”的事实。

### 2. `Usage`

放在文档开头，说明使用方法。至少包括：

- 这个文档的用途
- 什么时候必须更新
- 哪些章节是动态的
- 如何根据实际开发结果调整路线
- 这个文档与 handoff / spec / implementation plan 的关系

这一节的作用是降低后续交接成本，让任何接手的人都能先理解文档的维护规则，再使用它。

### 3. `Environment Facts`

这一节只记录稳定且会影响开发的环境事实。至少覆盖：

- 仓库根目录、工作区目录、当前 worktree
- 当前主要分支和远程关系
- Python 开发约定
- 常用依赖和测试环境约定
- 代理与网络约定
- 大文件与数据目录约定

边界要求：

- 只保留稳定事实
- 不记录一次性的临时输出
- 不把零散踩坑内容塞进这里，踩坑归 `Known Pitfalls`

### 4. `Current Status`

记录项目当前总体状态，使用高层摘要而不是细碎变更列表。至少覆盖：

- 当前已经打通的能力
- 当前实现边界
- 当前最新基线提交或阶段描述
- 目前仍然存在的核心限制

### 5. `Validated Next Work`

这一节只列“已验证可继续开发”的主线任务。进入这一节的事项必须满足：

- 已经从代码、测试、交接文档或设计约束中确认是实际缺口
- 不依赖尚未确认的前提
- 当前可以直接排入开发顺序

这一节是后续默认的主任务池。

### 6. `Candidate Directions`

这一节列“只是候选方向”。放在这里的工作通常具有以下特征之一：

- 可能要做，但当前时机未定
- 依赖前序任务结果，不能提前承诺
- 属于优化项、扩展项或实验项
- 目前还缺少足够证据判断是否值得做

这一节的目标是保留思路，但不污染当前主线。

### 7. `Phase Roadmap`

从阶段视角组织工作，把主线压缩成少量连续阶段。例如：

- Phase 1: verifier true execution path
- Phase 2: draft true execution path
- Phase 3: multi-round edge loop
- Phase 4: worker/session/KV integration and batching stabilization
- Phase 5: streaming and end-to-end validation

每个阶段应描述：

- 阶段目标
- 完成定义
- 与前后阶段的依赖关系

### 8. `Working Queue`

这一节只保留当前真正排到手边的少量任务，用于日常执行。要求：

- 数量少
- 顺序明确
- 只包含近期准备实际动手的项

这部分会最频繁更新，是执行时的直接入口。

### 9. `Known Pitfalls`

这一节记录已经踩过的坑、容易误判的实现状态，以及新 agent 最容易犯错的地方。至少包括：

- 哪些路径已经是真的，哪些仍然是 placeholder
- 哪些测试只是 stub smoke，不代表真实模型执行已完成
- 哪些未跟踪文件或目录不要误删
- 哪些状态清理、session 语义、prefix 语义曾经出过问题

这一节的目标是降低重复踩坑，而不是记录所有历史细节。

### 10. `Verification Commands`

这一节汇总当前最可靠的验证命令。至少包括：

- 当前推荐的最小回归命令
- 如果有，阶段性补充测试命令
- 命令适用范围的简短说明

这一节应该写成“新会话 agent 可以直接复制运行”的形式。

### 11. `Decision Rules`

明确路线图如何调整，至少要覆盖：

- 什么情况下把事项从 `Candidate Directions` 提升到 `Validated Next Work`
- 什么情况下从 `Validated Next Work` 降级或移除
- 什么情况下要整体重排阶段顺序
- 任务完成后最少要更新哪些章节

### 12. `Change Log`

用时间顺序记录路线调整。每条记录至少包含：

- 做完了什么任务
- 新发现了什么事实或限制
- 路线是否调整
- 如果调整了，调整原因是什么

这一节的作用是保留“为什么路线变了”的因果链，而不是只保留当前结果。

## Update Policy

这个文档必须显式声明自己是动态文档，而不是静态计划。维护规则如下：

1. 每完成一个实际开发任务，至少更新一次：
   - `Current Status`
   - `Working Queue`
   - `Known Pitfalls`
   - `Verification Commands`（如果验证方式变化）
   - `Change Log`
2. 每完成一个阶段，或者发现影响优先级的新事实时，可以重排：
   - `Validated Next Work`
   - `Phase Roadmap`
   - `Candidate Directions`
3. `Quick Start for New Agents` 和 `Environment Facts` 只在启动方式、环境约束、常用命令或关键路径变化时更新，不做无意义改写。
4. 如果实际实现结果推翻了旧路线，应先更新文档，再按新路线继续开发。
5. 不允许把候选方向直接写成当前承诺；需要先升级到 `Validated Next Work`。
6. 新会话 agent 开始开发前，默认先阅读这份文档，再根据 `Working Queue` 或 `Validated Next Work` 决定下一步，而不是先从旧 handoff 猜当前主线。

## Initial Content Requirements

全局路线图的首版内容应基于当前分支事实，而不是空模板。首版至少要写清：

- 当前 worktree / branch / remote / 环境约定
- 目前已经完成的控制面、session、utility path、single-round edge bring-up
- verifier replay / true logits 路径的当前状态
- draft 仍为 placeholder 的事实
- multi-round edge / streaming / session-KV 复用尚未完成
- 当前建议主线顺序
- 当前最可靠的验证命令
- 当前已知的关键踩坑和注意事项

也就是说，首版文档应该是“可直接指导下一步开发”的状态，而不是只有章节标题。

## Relationship To Ongoing Development

后续开发默认遵循以下顺序：

1. 新会话先阅读 `Quick Start for New Agents`、`Current Status` 和 `Working Queue`。
2. 从 `Working Queue` 选择当前任务。
3. 任务完成后更新路线图。
4. 根据新信息判断下一任务是否仍来自 `Validated Next Work`。
5. 如果不是，先修改路线图，再继续开发。

这保证路线图真正参与开发决策，而不是事后补记。

## Scope Boundaries

这个文档聚焦 DSSD 项目主线，不承担以下职责：

- 不详细替代某个子任务的 implementation plan
- 不记录逐文件改动清单
- 不作为 PR 描述
- 不把所有远期想法都升级成当前路线
- 不变成冗长的历史日志汇总

## Acceptance Criteria

该设计视为完成，当满足以下条件：

- 仓库内存在一个新的全局 DSSD 路线图文档
- 文档开头包含可直接用于新会话启动的 `Quick Start for New Agents`
- 文档包含明确的使用方法
- 文档包含稳定的环境事实、关键踩坑和验证命令
- 文档区分“已验证可继续开发”和“只是候选方向”
- 文档记录当前开发进度和后续阶段路线
- 文档声明自己会在每个任务完成后动态更新
- 文档首版内容足以指导下一步开发
