# DSSD OpenSpec Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 DSSD 分支的自定义 roadmap 工作流迁移到 OpenSpec，建立项目级 `program` source of truth，并用一个真实 DSSD follow-up change 验证新流程。

**Architecture:** 基于 OpenSpec 默认 `spec-driven` schema fork 一个项目内 `dssd-spec-driven` schema，通过 `../../program.md` 这条生成路径把新的 `program` artifact 固定到 [openspec/program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md)。单个开发切片继续走 `proposal/specs/design/tasks/apply/archive` 原生闭环；跨 change 的长期上下文、主线队列和变更历史统一收口到 `program.md`。

**Tech Stack:** OpenSpec CLI, YAML, Markdown, bash, `uv`, pytest

---

## Preflight

在 worktree `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex` 内执行：

```bash
source .venv/bin/activate
openspec --version
openspec schemas
```

预期：

- `.venv` 可正常激活
- `openspec` 命令可用
- `spec-driven` 出现在 schema 列表里

---

## Planned Files

### OpenSpec Workflow

- Create: `openspec/schemas/dssd-spec-driven/schema.yaml`
  定义项目内自定义 schema；在默认 `spec-driven` 基础上新增全局 `program` artifact，并让 `proposal` 依赖它。
- Create: `openspec/schemas/dssd-spec-driven/templates/program.md`
  定义新的 `program` 文档模板。
- Modify: `openspec/schemas/dssd-spec-driven/templates/proposal.md`
  如有必要，加入对 `program` / 当前主线的引用约束。
- Modify: `openspec/schemas/dssd-spec-driven/templates/design.md`
  如有必要，强调 change 设计与 `program` 的主线/风险/验证保持一致。
- Modify: `openspec/schemas/dssd-spec-driven/templates/tasks.md`
  如有必要，要求任务完成后同步更新 `openspec/program.md`。
- Modify: `openspec/config.yaml`
  切换 schema 到 `dssd-spec-driven`，并写入项目级 `context` 与 artifact `rules`。
- Create: `openspec/program.md`
  迁移后的全局 source of truth。

### Legacy Docs Rebinding

- Modify: `docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`
  增加 legacy notice，明确 `openspec/program.md` 为新 source of truth。
- Modify: `docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md`
  增加 legacy notice，明确后续执行以 OpenSpec changes 为主。
- Modify: `docs/superpowers/specs/2026-04-05-dssd-vllm-design.md`
  增加 reference note，明确其角色已降为背景设计输入。

### First Real OpenSpec Change

- Create: `openspec/changes/stabilize-real-model-mixed-baseline/proposal.md`
- Create: `openspec/changes/stabilize-real-model-mixed-baseline/specs/dssd-real-model-regression/spec.md`
- Create: `openspec/changes/stabilize-real-model-mixed-baseline/design.md`
- Create: `openspec/changes/stabilize-real-model-mixed-baseline/tasks.md`
  作为迁移后的第一个真实 change，承接当前已确认的 mixed baseline blocker。

---

### Task 1: Create The Project-Local OpenSpec Schema

**Files:**
- Create: `openspec/schemas/dssd-spec-driven/schema.yaml`
- Create: `openspec/schemas/dssd-spec-driven/templates/program.md`
- Modify: `openspec/schemas/dssd-spec-driven/templates/proposal.md`
- Modify: `openspec/schemas/dssd-spec-driven/templates/design.md`
- Modify: `openspec/schemas/dssd-spec-driven/templates/tasks.md`
- Modify: `openspec/config.yaml`

- [ ] **Step 1: Prove the custom schema does not exist yet**

Run:

```bash
source .venv/bin/activate
openspec schema which dssd-spec-driven
```

Expected:

```text
schema not found
```

- [ ] **Step 2: Fork the default schema into the project**

Run:

```bash
source .venv/bin/activate
openspec schema fork spec-driven dssd-spec-driven
find openspec/schemas/dssd-spec-driven -maxdepth 2 -type f | sort
```

Expected:

```text
openspec/schemas/dssd-spec-driven/schema.yaml
openspec/schemas/dssd-spec-driven/templates/design.md
openspec/schemas/dssd-spec-driven/templates/proposal.md
openspec/schemas/dssd-spec-driven/templates/spec.md
openspec/schemas/dssd-spec-driven/templates/tasks.md
```

- [ ] **Step 3: Replace the forked schema with the DSSD-specific artifact graph**

Edit [schema.yaml](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/schemas/dssd-spec-driven/schema.yaml) to this content:

```yaml
name: dssd-spec-driven
version: 1
description: DSSD OpenSpec workflow with a global program document
artifacts:
  - id: program
    generates: ../../program.md
    description: Global program board and persistent agent context
    template: program.md
    instruction: >
      Maintain the project-wide source of truth for DSSD.

      Required sections:
      - Quick Start for New Agents
      - Current Status
      - Mainline Priorities
      - Conditional Work
      - Working Queue
      - Known Pitfalls
      - Verification Commands
      - Decision Rules
      - Change Log

      This artifact is global, not change-local. Update it whenever a completed
      change alters project status, priorities, pitfalls, or verification.
    requires: []
  - id: proposal
    generates: proposal.md
    description: Initial proposal document outlining the change
    template: proposal.md
    instruction: >
      Create the proposal document that establishes WHY this change is needed.

      Read ../../program.md before writing. The proposal must reference the
      current mainline or explain why the change is a conditional follow-up.
    requires:
      - program
  - id: specs
    generates: specs/**/*.md
    description: Detailed specifications for the change
    template: spec.md
    requires:
      - proposal
  - id: design
    generates: design.md
    description: Technical design document with implementation details
    template: design.md
    requires:
      - proposal
  - id: tasks
    generates: tasks.md
    description: Implementation checklist with trackable tasks
    template: tasks.md
    requires:
      - specs
      - design
apply:
  requires:
    - tasks
  tracks: tasks.md
  instruction: |
    Read ../../program.md first, then read the change-local artifacts.
    Work through pending tasks, mark them complete as you go, and update
    ../../program.md before concluding the change.
```

- [ ] **Step 4: Create the new `program` template**

Create [program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/schemas/dssd-spec-driven/templates/program.md) with:

```markdown
# DSSD Program

## Quick Start for New Agents

- Active worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`
- Active branch: `dssd-vllm-codex`
- Read this file first, then use `Working Queue` as the default entry point.
- Activate the existing venv before verification: `source .venv/bin/activate`
- Use `uv` for Python dependency management.
- Prefer mirror sources first; use `proxy on` only if direct access still fails.
- Keep large model and data artifacts under `/data/zz/`.

## Current Status

<!-- Project-wide DSSD status -->

## Mainline Priorities

<!-- Non-optional current mainline work -->

## Conditional Work

<!-- Candidate or conditional follow-up work -->

## Working Queue

<!-- Immediate next items -->

## Known Pitfalls

<!-- High-signal implementation traps -->

## Verification Commands

<!-- Copy-pasteable verification commands -->

## Decision Rules

<!-- Rules for promoting/demoting work and updating this file -->

## Change Log

<!-- Reverse-chronological roadmap updates -->
```

- [ ] **Step 5: Add DSSD-specific guidance to proposal / design / tasks templates**

Update [proposal.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/schemas/dssd-spec-driven/templates/proposal.md) so it starts with:

```markdown
> Read `../../program.md` before filling this file.
> If this proposal is not directly on the current mainline, say why it is being promoted now.

## Why
```

Update [design.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/schemas/dssd-spec-driven/templates/design.md) so it starts with:

```markdown
> Read `../../program.md` and `proposal.md` before filling this file.
> The design must name the current blocker or mainline gap it resolves, and list the verification command that will prove the change.

## Context
```

Update [tasks.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/schemas/dssd-spec-driven/templates/tasks.md) so it starts with:

```markdown
> Read `../../program.md`, `proposal.md`, and `design.md` before filling this file.
> The final task group must include updating `../../program.md` after verification.

## 1. <!-- Task Group Name -->
```

- [ ] **Step 6: Replace `openspec/config.yaml` with the DSSD project context**

Edit [config.yaml](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/config.yaml) to:

```yaml
schema: dssd-spec-driven

context: |
  Project: DSSD on vLLM
  Repo root: /home/zz/workspace/vllm
  Active worktree: /home/zz/workspace/vllm/.worktrees/dssd-vllm-codex
  Active branch: dssd-vllm-codex
  Python work must use the existing virtualenv; prefer uv.
  Use mirror sources first; use proxy on only if mirrors fail.
  Large files and model weights belong under /data/zz/.
  The OpenSpec source of truth is openspec/program.md.
  Each change must read openspec/program.md before writing proposal/design/tasks.
  Each completed change must update openspec/program.md before it is considered done.

rules:
  program:
    - Keep the file concise and self-contained.
    - Treat it as the persistent agent context for new sessions.
    - Do not let candidate directions overwrite current mainline priorities.
  proposal:
    - State whether the change is current mainline or conditional follow-up.
    - Reference the exact blocker, regression, or capability gap being addressed.
  design:
    - Name the concrete verification command that proves the change.
    - Record any migration or rollback concern if the change alters workflow files.
  tasks:
    - End with a task to update openspec/program.md after verification.
    - Keep tasks reviewable and scoped to one coherent change.
```

- [ ] **Step 7: Validate the custom schema**

Run:

```bash
source .venv/bin/activate
openspec schema validate dssd-spec-driven
```

Expected:

```text
Schema 'dssd-spec-driven' is valid
```

- [ ] **Step 8: Smoke-test the global `program` artifact path**

Run:

```bash
source .venv/bin/activate
openspec new change schema-smoke
openspec status --change schema-smoke --json
rm -rf openspec/changes/schema-smoke
```

Expected JSON fields:

```json
{
  "schemaName": "dssd-spec-driven",
  "artifacts": [
    {
      "id": "program",
      "outputPath": "../../program.md",
      "status": "ready"
    }
  ]
}
```

- [ ] **Step 9: Commit the schema cut**

Run:

```bash
git add openspec/config.yaml openspec/schemas/dssd-spec-driven
git commit -m "docs: add dssd openspec schema"
```

---

### Task 2: Bootstrap `openspec/program.md` From The Current Roadmap

**Files:**
- Create: `openspec/program.md`
- Modify: `openspec/config.yaml`

- [ ] **Step 1: Confirm the program file does not exist yet**

Run:

```bash
test -f openspec/program.md && echo exists || echo missing
```

Expected:

```text
missing
```

- [ ] **Step 2: Create the first real `openspec/program.md`**

Create [program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md) with this initial content:

```markdown
# DSSD Program

## Quick Start for New Agents

- Active worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`
- Active branch: `dssd-vllm-codex`
- Read this file first, then inspect the active change in `openspec/changes/`.
- Activate the existing venv before verification: `source .venv/bin/activate`
- Use `uv` for Python dependency management.
- Prefer mirror sources first; use `proxy on` only if direct access still fails.
- Keep large model and data artifacts under `/data/zz/`.
- Current workflow source of truth lives in OpenSpec, not in `docs/superpowers/plans/`.

## Current Status

- DSSD config surface, edge/verifier split, HTTP control plane, verifier replay path, draft replay path, multi-round edge loop, and streaming path all exist on this branch.
- The current blocking issue is that the mixed real-model baseline is not stable enough to call the mainline complete.
- The concrete failing gate is the mixed regression that combines protocol/coordinator/smoke tests with the real-model DSSD regression.

## Mainline Priorities

1. Stabilize the mixed real-model DSSD baseline so the fresh combined regression passes.
2. Only after the mixed baseline is stable, reassess whether additional verifier aggregation or productization work should be promoted.

## Conditional Work

- Engine/service-side verifier batch aggregation, but only if a concrete multi-request need appears.
- Additional OpenAI-surface metadata or usage parity, but only if streaming validation exposes a concrete gap.
- Deeper worker-side reuse or optimization beyond the current validated path.

## Working Queue

1. Create and use the OpenSpec change `stabilize-real-model-mixed-baseline`.
2. Isolate why `tests/v1/e2e/dssd/test_dssd_real_model.py` passes alone but fails in the mixed suite.
3. After a verified fix, update this file and archive the change.

## Known Pitfalls

- The current fresh blocker is not a generic unit-test failure; it only appears in the mixed suite.
- Stub-based DSSD tests must not leak module-level `vllm.*` shims into the real-model path.
- `GPUModelRunner.dssd_draft_round()` still replays the full prefix on every draft step; preserve sampler-based semantics if that path changes.
- The verifier direct-request baseline currently uses worker-local scratch execution, not persistent worker-side KV reuse.

## Verification Commands

```bash
source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_round_coordinator.py tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py -q
source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_verifier_service.py -q
source .venv/bin/activate && pytest --noconftest tests/v1/e2e/dssd/test_dssd_real_model.py -q
source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_real_model.py tests/v1/e2e/dssd/test_dssd_smoke.py -q
```

## Decision Rules

- Promote work into `Mainline Priorities` only when current code, tests, or regressions show a concrete gap.
- A change is not complete until both its own verification passes and this file is updated.
- If implementation changes what is currently blocking the mainline, update `Current Status`, `Mainline Priorities`, `Working Queue`, and `Change Log` together.

## Change Log

- 2026-04-06: Bootstrapped OpenSpec `program.md` from the legacy DSSD roadmap and the fresh mixed-baseline audit. The active next step is `stabilize-real-model-mixed-baseline`.
```

- [ ] **Step 3: Prove the global artifact now resolves as done**

Run:

```bash
source .venv/bin/activate
openspec new change program-smoke
openspec status --change program-smoke --json
rm -rf openspec/changes/program-smoke
```

Expected JSON fields:

```json
{
  "schemaName": "dssd-spec-driven",
  "artifacts": [
    {
      "id": "program",
      "outputPath": "../../program.md",
      "status": "done"
    },
    {
      "id": "proposal",
      "outputPath": "proposal.md",
      "status": "ready"
    }
  ]
}
```

- [ ] **Step 4: Verify the document structure**

Run:

```bash
rg -n '^## ' openspec/program.md
```

Expected:

```text
## Quick Start for New Agents
## Current Status
## Mainline Priorities
## Conditional Work
## Working Queue
## Known Pitfalls
## Verification Commands
## Decision Rules
## Change Log
```

- [ ] **Step 5: Commit the program bootstrap**

Run:

```bash
git add openspec/program.md
git commit -m "docs: bootstrap dssd openspec program"
```

---

### Task 3: Rebind Legacy Docs To The New Source Of Truth

**Files:**
- Modify: `docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`
- Modify: `docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md`
- Modify: `docs/superpowers/specs/2026-04-05-dssd-vllm-design.md`

- [ ] **Step 1: Add a legacy notice to the old roadmap**

Insert this block at the top of [2026-04-05-dssd-vllm-global-roadmap.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md), immediately under the title:

```markdown
> **Legacy Notice**
>
> This document is retained as historical branch context.
> The current DSSD workflow source of truth has moved to
> [openspec/program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md).
> New work should start from OpenSpec, not by extending this file.
```

- [ ] **Step 2: Add a legacy notice to the old implementation plan**

Insert this block at the top of [2026-04-05-dssd-vllm-implementation.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md), immediately under the title:

```markdown
> **Legacy Notice**
>
> This plan is retained as historical planning context for the initial DSSD bring-up.
> Ongoing work should be planned as OpenSpec changes under `openspec/changes/`,
> with [openspec/program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md)
> as the current global source of truth.
```

- [ ] **Step 3: Add a reference note to the old DSSD design**

Insert this block at the top of [2026-04-05-dssd-vllm-design.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/specs/2026-04-05-dssd-vllm-design.md), immediately under the title:

```markdown
> **Reference Note**
>
> This document remains the baseline system design reference for DSSD on vLLM.
> Workflow ownership has moved to OpenSpec:
> use [openspec/program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md)
> for current program status and `openspec/changes/` for active work.
```

- [ ] **Step 4: Verify that each legacy doc now points to OpenSpec**

Run:

```bash
rg -n "openspec/program.md|openspec/changes/" docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md docs/superpowers/specs/2026-04-05-dssd-vllm-design.md
```

Expected:

```text
all three files contain the new OpenSpec reference note
```

- [ ] **Step 5: Commit the legacy-doc rebind**

Run:

```bash
git add docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md docs/superpowers/specs/2026-04-05-dssd-vllm-design.md
git commit -m "docs: rebind legacy dssd workflow docs to openspec"
```

---

### Task 4: Seed The First Real OpenSpec Change

**Files:**
- Create: `openspec/changes/stabilize-real-model-mixed-baseline/proposal.md`
- Create: `openspec/changes/stabilize-real-model-mixed-baseline/specs/dssd-real-model-regression/spec.md`
- Create: `openspec/changes/stabilize-real-model-mixed-baseline/design.md`
- Create: `openspec/changes/stabilize-real-model-mixed-baseline/tasks.md`

- [ ] **Step 1: Create the active change directory**

Run:

```bash
source .venv/bin/activate
openspec new change stabilize-real-model-mixed-baseline
openspec status --change stabilize-real-model-mixed-baseline --json
```

Expected JSON fields:

```json
{
  "schemaName": "dssd-spec-driven",
  "artifacts": [
    {
      "id": "program",
      "status": "done"
    },
    {
      "id": "proposal",
      "status": "ready"
    }
  ]
}
```

- [ ] **Step 2: Create the change proposal**

Create [proposal.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/changes/stabilize-real-model-mixed-baseline/proposal.md) with:

```markdown
## Why

The current DSSD branch cannot honestly claim mainline completion because the
fresh mixed regression still fails even though the focused and standalone
real-model tests pass. This makes the workflow and validation story unreliable.

## What Changes

- Stabilize the mixed real-model DSSD baseline so the combined regression passes.
- Isolate any shared state, fixture leakage, or startup drift that only appears
  when the real-model test runs in the mixed suite.
- Keep the focused regressions and the standalone real-model regression passing.

## Capabilities

### New Capabilities
- `dssd-real-model-regression`: stable mixed regression behavior for the real-model DSSD path

### Modified Capabilities
- `dssd-streaming`: tighten regression isolation so mixed-suite behavior matches standalone behavior

## Impact

- `tests/v1/e2e/dssd/test_dssd_real_model.py`
- `tests/v1/e2e/dssd/test_dssd_smoke.py`
- `tests/v1/dssd/test_protocol.py`
- `tests/v1/dssd/test_round_coordinator.py`
- Any DSSD helper or fixture code required to eliminate cross-test pollution
```

- [ ] **Step 3: Create the delta spec**

Create [spec.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/changes/stabilize-real-model-mixed-baseline/specs/dssd-real-model-regression/spec.md) with:

```markdown
## ADDED Requirements

### Requirement: Mixed DSSD regression matches standalone real-model behavior

The DSSD test suite SHALL preserve non-empty real-model non-streaming responses
when the real-model regression runs in the mixed baseline alongside the stub-based
protocol, coordinator, and smoke regressions.

#### Scenario: Mixed baseline preserves non-streaming response content
- **WHEN** the mixed DSSD regression suite is executed with the real-model regression included
- **THEN** the non-streaming real-model response content is not empty

### Requirement: Regression isolation does not break standalone validation

The DSSD test suite SHALL keep the standalone real-model regression and the
existing focused DSSD regressions passing while fixing the mixed-suite failure.

#### Scenario: Standalone real-model regression still passes
- **WHEN** the standalone real-model DSSD regression is executed
- **THEN** it passes after the mixed-suite fix is applied
```

- [ ] **Step 4: Create the change design**

Create [design.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/changes/stabilize-real-model-mixed-baseline/design.md) with:

```markdown
## Context

The current blocker only appears in the mixed regression:
`tests/v1/dssd/test_protocol.py`,
`tests/v1/dssd/test_round_coordinator.py`,
`tests/v1/e2e/dssd/test_dssd_real_model.py`,
and `tests/v1/e2e/dssd/test_dssd_smoke.py`.
The standalone real-model test passes by itself, which strongly suggests shared
state leakage, fixture pollution, or startup-path drift.

## Goals / Non-Goals

**Goals:**
- Reproduce and fix the mixed-suite failure.
- Keep standalone real-model validation passing.
- Keep stub-based regressions isolated from the real-model path.

**Non-Goals:**
- Broader DSSD refactors unrelated to the mixed-suite failure.
- New productization work outside the regression fix.

## Decisions

- Debug the failure as a test-isolation problem first, not as a general model-execution rewrite.
- Use the mixed regression command as the primary gate.
- Treat `openspec/program.md` as the workflow source of truth and update it after the fix is verified.

## Risks / Trade-offs

- The root cause may involve more than one shared-state leak, so the first narrow fix may not be sufficient.
- Real-model startup behavior can be sensitive to environment and GPU budget, so verification must include the standalone regression too.
```

- [ ] **Step 5: Create the change tasks**

Create [tasks.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/changes/stabilize-real-model-mixed-baseline/tasks.md) with:

```markdown
## 1. Reproduce And Localize

- [ ] 1.1 Re-run the mixed DSSD baseline and capture the failing assertion plus the exact empty-response symptom.
- [ ] 1.2 Compare the mixed run against the standalone real-model regression to identify shared-state differences.
- [ ] 1.3 Narrow the failure to fixture pollution, module leakage, or startup drift before editing code.

## 2. Fix And Verify

- [ ] 2.1 Implement the minimal fix for the mixed-suite failure.
- [ ] 2.2 Re-run the standalone real-model regression and the focused DSSD regressions to confirm no regression.
- [ ] 2.3 Re-run the mixed DSSD baseline and confirm it passes.

## 3. Close The Change

- [ ] 3.1 Update `../../program.md` with the new status, next priorities, pitfalls, and change-log entry.
- [ ] 3.2 Archive the change after verification succeeds.
```

- [ ] **Step 6: Verify that the first change is apply-ready**

Run:

```bash
source .venv/bin/activate
openspec status --change stabilize-real-model-mixed-baseline --json
```

Expected JSON fields:

```json
{
  "schemaName": "dssd-spec-driven",
  "applyRequires": ["tasks"],
  "artifacts": [
    {
      "id": "program",
      "status": "done"
    },
    {
      "id": "proposal",
      "status": "done"
    },
    {
      "id": "specs",
      "status": "done"
    },
    {
      "id": "design",
      "status": "done"
    },
    {
      "id": "tasks",
      "status": "done"
    }
  ]
}
```

- [ ] **Step 7: Commit the first active OpenSpec change**

Run:

```bash
git add openspec/changes/stabilize-real-model-mixed-baseline
git commit -m "docs: seed first dssd openspec change"
```

---

### Task 5: Verify The Migrated Workflow End-To-End

**Files:**
- Modify: `openspec/program.md`

- [ ] **Step 1: Confirm the program file and the first change are both visible**

Run:

```bash
source .venv/bin/activate
openspec list --json
```

Expected:

```json
[
  {
    "name": "stabilize-real-model-mixed-baseline"
  }
]
```

- [ ] **Step 2: Confirm proposal instructions still work under the custom schema**

Run:

```bash
source .venv/bin/activate
openspec instructions proposal --change stabilize-real-model-mixed-baseline --json
```

Expected JSON fields:

```json
{
  "schemaName": "dssd-spec-driven",
  "artifactId": "proposal",
  "outputPath": "proposal.md"
}
```

- [ ] **Step 3: Confirm apply instructions can be generated**

Run:

```bash
source .venv/bin/activate
openspec instructions apply --change stabilize-real-model-mixed-baseline --json
```

Expected:

```text
JSON output listing the context files and pending tasks for the active change
```

- [ ] **Step 4: Add the migration completion note to `openspec/program.md`**

Append this bullet to the `Change Log` section in [program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md):

```markdown
- 2026-04-06: Migrated the DSSD workflow to OpenSpec with the project-local `dssd-spec-driven` schema and seeded `stabilize-real-model-mixed-baseline` as the first active change.
```

- [ ] **Step 5: Run the final workflow sanity checks**

Run:

```bash
source .venv/bin/activate
openspec schema validate dssd-spec-driven
openspec status --change stabilize-real-model-mixed-baseline
rg -n "OpenSpec|openspec/program.md|stabilize-real-model-mixed-baseline" openspec/program.md docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md docs/superpowers/specs/2026-04-05-dssd-vllm-design.md
```

Expected:

```text
schema validation succeeds
the active change is visible and apply-ready
the legacy docs and program file all point to the new OpenSpec workflow
```

- [ ] **Step 6: Commit the workflow migration close-out**

Run:

```bash
git add openspec/program.md
git commit -m "docs: finalize dssd openspec workflow migration"
```
