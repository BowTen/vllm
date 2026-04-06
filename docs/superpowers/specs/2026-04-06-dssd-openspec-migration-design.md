# DSSD OpenSpec Workflow Migration Design

## Goal

把当前 DSSD 分支里自定义的 `roadmap + spec + implementation plan + handoff` 工作流迁移为以 OpenSpec 为默认入口的工作流，同时保留我们已经验证有效的两类能力：

- 面向单个开发切片的 proposal / design / tasks 闭环
- 面向整个长期项目的全局进度板、agent 启动上下文、主线判断和动态重排

迁移完成后，OpenSpec 应成为这个分支的默认工作流入口；后续不再新增独立的 `docs/superpowers/plans/*roadmap*` 作为主执行板。

## Why This Migration Exists

当前 DSSD 分支已经形成了一套能工作的自定义文档流：

- [2026-04-05-dssd-vllm-global-roadmap.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md)
- [2026-04-05-dssd-vllm-design.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/specs/2026-04-05-dssd-vllm-design.md)
- [2026-04-05-dssd-vllm-implementation.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md)

这套流程已经证明有效，但有两个问题：

1. 它不是工具原生工作流，跨会话 agent 只能靠约定读取自定义文档。
2. 它把“单 change 闭环”和“长期项目总控板”都塞在 `docs/superpowers` 里，迁移和复用成本偏高。

OpenSpec 刚初始化完成，提供了标准的 `change -> proposal/specs/design/tasks -> apply -> archive` 闭环。这非常适合承接单个 change 的执行，但默认 schema 只覆盖 per-change 文档，不覆盖我们当前 roadmap 承担的全局项目状态。

因此迁移目标不是“原样复制现有 roadmap”，而是把当前工作流拆成：

- OpenSpec 原生 change 闭环
- 一个最小扩展的全局 program artifact

## Non-Goals

本次迁移设计不做以下事情：

- 不重写 OpenSpec 的 `apply` / `archive` 主流程
- 不尝试把所有历史 DSSD 文档逐字迁移成多个 OpenSpec change
- 不在第一步就做重型自定义 schema
- 不改变当前 DSSD 代码主线或测试策略

## Current Constraints

### 1. OpenSpec 的默认模型是 per-change

`spec-driven` schema 的默认结构是：

- `proposal.md`
- `specs/**/*.md`
- `design.md`
- `tasks.md`

每个 change 独立存在于 `openspec/changes/<name>/`，完成后归档到 `openspec/changes/archive/...`。它天然适合管理一次功能开发或一次重构切片。

### 2. 当前 DSSD 路线图承担了跨 change 的长期状态职责

现有全局 roadmap 不只是任务清单，还承担：

- 当前项目进度
- 当前主线与候选方向
- 当前工作队列
- 已知踩坑
- agent 启动上下文
- 路线重排历史

这些内容都不是某一个具体 change 独有的，也不应该随着某一个 change 归档而消失。

### 3. 最终目标是不再维护单独 roadmap

迁移后的默认 source of truth 应该在 OpenSpec 体系内，而不是继续维护一份 OpenSpec 外部的全局执行板。

## Design Principles

- 以 OpenSpec 为唯一默认工作流入口。
- 尽量复用 OpenSpec 原生 `spec-driven` 结构，不重造 proposal/design/tasks 语义。
- 只补当前默认 schema 无法承接的“全局项目状态”层。
- 新会话 agent 应能通过 OpenSpec 内的 artifact，直接获得启动上下文和当前主线。
- 单个 change 继续保持可归档；全局项目状态必须跨 change 持续存在。

## Proposed Workflow Shape

迁移后的工作流由两层组成。

### Layer 1: Program Layer

在项目级 schema 中新增一个常驻 artifact：

- `program`

它承接当前 roadmap 的长期职责，作为所有新 agent 的默认第一入口。

### Layer 2: Change Layer

继续保留 OpenSpec 原生 change 闭环：

- `proposal`
- `specs`
- `design`
- `tasks`
- `apply`
- `archive`

也就是说：

- `program` 负责“整个 DSSD 项目当前处于什么状态”
- `proposal/specs/design/tasks` 负责“当前这个 change 要做什么、怎么做、做到哪里”

## Schema Strategy

基于 OpenSpec 默认 `spec-driven` schema fork 一个项目内 schema，例如：

- `dssd-spec-driven`

这个 schema 只做轻量扩展：

- 保留 `proposal/specs/design/tasks` 的原生职责
- 新增 `program` artifact
- 通过 `openspec/config.yaml` 的 `context` 和 `rules` 补充项目级写作与执行约束

不做深度定制，不修改 OpenSpec CLI 主命令模型。

## Proposed Artifact Set

### 1. `program`

建议生成路径：

- `program.md`

职责：

- 替代当前独立 roadmap
- 提供新 agent 启动上下文
- 维护当前项目状态、主线和动态队列
- 记录跨 change 的路线调整

该 artifact 是项目常驻文档，不随着单个 change 归档。

建议固定章节：

- `Quick Start for New Agents`
- `Current Status`
- `Mainline Priorities`
- `Conditional Work`
- `Working Queue`
- `Known Pitfalls`
- `Verification Commands`
- `Decision Rules`
- `Change Log`

### 2. `proposal`

继续表示单个 change 的：

- 动机
- 范围
- 能力边界
- 影响面

### 3. `specs`

继续表示单个 change 的 requirement delta。

### 4. `design`

继续表示单个 change 的技术方案。

### 5. `tasks`

继续表示单个 change 的实施清单，并作为 `apply` 的勾选来源。

## Artifact Responsibilities

迁移后职责边界必须明确如下。

### `program` 负责

- 当前项目整体状态
- 当前主线与非主线判断
- 下一个应该创建什么 change
- 当前全局验证入口
- 已知长期坑点
- 最近为什么调整路线

### `proposal/specs/design/tasks` 负责

- 当前这个 change 自己的动机、要求、方案和执行步骤

### `program` 不负责

- 单个 change 的详细任务拆分
- 详细设计推演
- 精确 requirement delta 文本

### 单个 change 不负责

- 重写整个项目全局状态
- 重新定义长期主线顺序，除非执行结果触发 `program` 更新

## Mapping From Existing Documents

### 1. 全局 roadmap

现有：

- [2026-04-05-dssd-vllm-global-roadmap.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md)

迁移目标：

- 作为首版 `program.md` 的直接输入

迁移方式：

- 保留其有效章节结构
- 重新命名并放入 OpenSpec schema 管理范围
- 后续不再把它当作 OpenSpec 外部主执行板继续扩写

### 2. DSSD 总体设计

现有：

- [2026-04-05-dssd-vllm-design.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/specs/2026-04-05-dssd-vllm-design.md)

迁移目标：

- 作为初始项目基线设计输入
- 为后续主 specs 与 future changes 提供参考

迁移方式：

- 不强制拆成一次性多个 specs
- 先作为 program/context 的补充背景
- 后续在真正的 change 中逐步把仍然有效的 requirement 沉淀进 `openspec/specs/`

### 3. 一次性 implementation plan

现有：

- [2026-04-05-dssd-vllm-implementation.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md)

迁移目标：

- 不再作为长期主执行板继续使用

迁移方式：

- 将其中尚未完成或后续仍需要的主线工作，拆成一个个新的 OpenSpec change
- 每个 change 自己生成 `proposal/specs/design/tasks`

### 4. handoff 文档

现有 handoff 继续作为历史交接记录保留，但不再承担当前 source of truth 角色。

## Operational Workflow After Migration

迁移后的默认操作顺序应固定为：

1. 新会话 agent 先读取 `program`
2. 根据 `Working Queue` / `Mainline Priorities` 判断当前要推进的切片
3. 为该切片创建一个 OpenSpec change
4. 在该 change 下完成 `proposal -> specs -> design -> tasks`
5. 执行 `apply`
6. 实际开发与验证
7. 完成后同时更新两处：
   - 当前 change 的 `tasks`
   - 全局 `program`
8. 归档当前 change
9. 回到 `program` 判断下一步

关键约束：

- 每完成一个 change，都必须更新 `program`
- 如果执行结果改变了优先级，先更新 `program`，再开始下一个 change
- `program` 是所有 change 的前置上下文，不允许被跳过

## Program Update Rules

`program` 应替代当前 roadmap 的动态维护规则。

至少要求：

1. 每完成一个实际 change，更新：
   - `Current Status`
   - `Working Queue`
   - `Known Pitfalls`
   - `Verification Commands`（如果有变化）
   - `Change Log`
2. 当 change 实施结果改变主线判断时，更新：
   - `Mainline Priorities`
   - `Conditional Work`
   - `Decision Rules`（如果需要）
3. 新会话 agent 默认先读 `program`，再读当前 active changes

## Proposed Migration Phases

### Phase 1: Workflow Definition

目标：

- 确认迁移边界和 artifact 职责
- fork 项目内 schema

完成定义：

- `dssd-spec-driven` schema 存在
- `program` artifact 已定义
- `openspec/config.yaml` 已写入项目级 context/rules

### Phase 2: Program Bootstrap

目标：

- 把现有 roadmap 迁成首版 `program.md`

完成定义：

- 现有 roadmap 的有效内容已经进入 `program`
- 新 agent 可以只读 OpenSpec artifact 获得启动上下文

### Phase 3: Existing Docs Rebinding

目标：

- 明确旧 `docs/superpowers` 文档的新角色

完成定义：

- roadmap 不再继续作为主执行板扩写
- 总体设计与旧 implementation plan 只作为迁移输入和历史参考

### Phase 4: First Real Change Under OpenSpec

目标：

- 用一个真实 DSSD 后续任务验证新工作流

完成定义：

- 至少一个 change 走完 proposal/specs/design/tasks/apply/archive
- `program` 在 change 完成后被同步更新

## Risks And Trade-offs

### Risk 1: 仍然存在“全局文档”

即使迁入 OpenSpec，`program` 依然是一个长期存在的全局文档。区别只是它成为 OpenSpec artifact，而不是外部 roadmap 文件。

Trade-off：

- 这是有意保留的最小扩展
- 因为 OpenSpec 默认只覆盖 per-change，不覆盖长期项目状态

### Risk 2: 旧文档与新文档可能并存一段时间

迁移初期会同时存在：

- 旧 `docs/superpowers` 文档
- 新 OpenSpec artifact

Trade-off：

- 需要明确 source of truth 的切换点
- 切换完成后，旧文档只保留参考角色，不再继续扩写

### Risk 3: schema 轻量扩展仍然需要维护约定

即使新增 `program`，OpenSpec CLI 默认仍然更偏向 per-change 视角。

Trade-off：

- 需要在 schema 说明和项目 context/rules 中显式告诉 agent：
  - 先读 `program`
  - 完成 change 后回写 `program`

## Recommended Next Step

在这个设计获批后，下一步应执行一个单独的 implementation plan，完成以下动作：

1. fork `spec-driven` 为项目内 `dssd-spec-driven`
2. 为新 schema 增加 `program` artifact 与模板
3. 把现有 roadmap 内容迁入首版 `program.md`
4. 在 `openspec/config.yaml` 写入项目级 context 和 artifact rules
5. 用一个真实 DSSD follow-up task 作为第一个 OpenSpec change 验证迁移结果

## Success Criteria

当以下条件同时满足时，可以认为迁移成功：

- OpenSpec 成为该分支默认工作流入口
- 新 agent 只读 OpenSpec artifact 就能进入开发
- 单个 change 能走完原生 OpenSpec 闭环
- 全局项目状态不再依赖独立 roadmap 文件
- 旧 `docs/superpowers` 文档不再承担 source of truth 角色
