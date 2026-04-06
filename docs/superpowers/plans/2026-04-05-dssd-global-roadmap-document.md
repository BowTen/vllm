# DSSD Global Roadmap Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在仓库内新增一个首版 DSSD 全局路线图文档，同时把它写成可直接提供给新会话 agent 使用的持久上下文。

**Architecture:** 实现只涉及文档层，不改运行时代码。先创建一个自包含的 roadmap 文档，写入 Quick Start、环境事实、当前状态、主线与候选路线、验证命令和已知踩坑；然后再做一次面向“新 agent 冷启动”的自检，确认文档足够简洁且能直接指导下一步开发。

**Tech Stack:** Markdown, git worktree, rg, sed

**Post-Implementation Note:** 最终 roadmap 在 review 收尾阶段补入了 `Decision Rules` 章节，并把优先级顺序收敛为“draft execution -> multi-round edge flow -> verifier worker-state/KV integration -> real end-to-end validation”。如果这个计划文档与最终 roadmap 的阶段顺序有出入，以最终 roadmap 为准。

---

### Task 1: Create The Global Roadmap Skeleton And Seed Context

**Files:**
- Create: `docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`
- Reference: `docs/superpowers/specs/2026-04-05-dssd-global-roadmap-design.md`
- Reference: `docs/superpowers/handoffs/2026-04-05-dssd-vllm-handoff.md`
- Reference: `docs/superpowers/specs/2026-04-05-dssd-vllm-design.md`
- Reference: `docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md`

- [ ] **Step 1: Verify the roadmap file does not already exist**

Run:

```bash
test ! -e /home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md
```

Expected:

```text
Exit code 0 with no output
```

- [ ] **Step 2: Create the roadmap document with the approved section structure**

Write `docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md` with these top-level sections:

```md
# DSSD Global Roadmap

## Quick Start for New Agents

## Usage

## Environment Facts

## Current Status

## Validated Next Work

## Candidate Directions

## Phase Roadmap

## Working Queue

## Known Pitfalls

## Verification Commands

## Change Log
```

- [ ] **Step 3: Fill the startup sections with concise, self-contained facts**

Populate `Quick Start for New Agents`, `Usage`, and `Environment Facts` with concrete facts from the current branch:

```md
## Quick Start for New Agents

- Worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`
- Branch: `dssd-vllm-codex`
- Repo root: `/home/zz/workspace/vllm`
- Upstream: `origin/dssd-vllm-codex`
- Use Python virtualenvs; prefer `uv` for Python dependency management.
- Activate the existing venv before verification commands: `source .venv/bin/activate`
- If network access is needed and direct access fails, prefer mirror sources first; only then use `proxy on`.
- Keep large model/data artifacts under `/data/zz/`; do not commit them into the repo.
- Read this document first, then use `Working Queue` as the default entry point for the next task.

## Usage

- This document is the default DSSD execution board and persistent agent context.
- Update `Current Status`, `Working Queue`, `Known Pitfalls`, and `Change Log` after every completed development task.
- Update `Verification Commands` when the recommended verification path changes.
- Re-rank `Validated Next Work`, `Candidate Directions`, and `Phase Roadmap` whenever implementation results change priorities.
- Do not promote a candidate item into the active queue until it is validated by current code, tests, or design constraints.

## Environment Facts

- Main repo root: `/home/zz/workspace/vllm`
- Active DSSD worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`
- Active branch: `dssd-vllm-codex`
- Remote: `origin` -> `git@github.com:BowTen/vllm.git`
- Python workflow: venv required, prefer `uv`
- Proxy workflow: use mirror sources first, `proxy on` only when needed
- Large file workflow: place weights and datasets under `/data/zz/` and link into the workspace if needed
```

- [ ] **Step 4: Seed the current-status sections from the existing DSSD branch state**

Populate `Current Status`, `Validated Next Work`, `Candidate Directions`, and `Phase Roadmap` with concrete branch facts:

```md
## Current Status

- DSSD config surface, edge/verifier role split, verifier HTTP routes, engine utility path, verifier session tracking, and single-round edge control flow are implemented.
- Verifier replay now reaches a real logits path through `GPUModelRunner.dssd_verify_round()`, but the implementation is still a conservative replay path rather than full session/KV reuse.
- Draft execution is still placeholder-based.
- Edge execution is still single-round and non-streaming.

## Validated Next Work

1. Replace placeholder draft execution with real draft-model token/probability generation.
2. Extend edge control flow from single-round bring-up to a full multi-round DSSD loop.
3. Convert verifier replay from conservative prefix replay into stable worker-side request/session/KV integration.
4. Add real end-to-end validation with actual edge/verifier model execution.

## Candidate Directions

- More aggressive verifier batching and scheduling optimizations.
- Richer network simulation beyond the current latency/bandwidth delay model.
- More advanced prefix/KV reuse beyond the first stable worker-side integration.
- Broader productization features outside the current text-only prototype scope.

## Phase Roadmap

- Phase 1: stabilize verifier real execution path and worker/session semantics
- Phase 2: implement true draft execution
- Phase 3: implement full multi-round edge coordination
- Phase 4: stabilize worker-side state reuse, batching, and e2e realism
- Phase 5: add streaming and stronger end-to-end validation
```

- [ ] **Step 5: Confirm the expected section headings exist**

Run:

```bash
rg -n '^## ' /home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md
```

Expected:

```text
Matches for all 11 required `##` section headings
```

### Task 2: Add Actionable Queue, Pitfalls, Verification, And Change Log

**Files:**
- Modify: `docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`
- Reference: `vllm/v1/dssd/worker/verifier_runner.py`
- Reference: `vllm/v1/worker/gpu_model_runner.py`
- Reference: `vllm/v1/dssd/edge/coordinator.py`
- Reference: `tests/v1/e2e/dssd/test_dssd_smoke.py`

- [ ] **Step 1: Add a short working queue with the next concrete tasks**

Populate `Working Queue` with a small ordered list:

```md
## Working Queue

1. Implement real draft execution in `GPUModelRunner.dssd_draft_round()` and its helper path.
2. Replace the single-round edge flow with a real multi-round DSSD loop that commits only accepted tokens.
3. Decide whether verifier worker state should first stabilize on replay-plus-state caching or move directly to request/KV integration.
4. Add a real-model e2e validation path after the draft and multi-round paths are no longer placeholder-driven.
```

- [ ] **Step 2: Record the most important current pitfalls**

Populate `Known Pitfalls` with concrete warnings from the current branch:

```md
## Known Pitfalls

- `GPUModelRunner.dssd_draft_round()` is still placeholder logic; do not mistake the current edge flow for real draft-model execution.
- The verifier path is more real than before, but it currently relies on conservative replay rather than stable worker-side KV reuse.
- `tests/v1/e2e/dssd/test_dssd_smoke.py` is a stub smoke test; it validates control-plane wiring, not true model execution.
- The worktree currently has untracked `docs/superpowers/plans/`; do not delete it as cleanup without checking intent.
- Earlier verifier replay fixes had to handle empty committed prefixes and replay-state cleanup; treat prefix semantics and request cleanup as fragile areas.
```

- [ ] **Step 3: Record the current recommended verification commands**

Populate `Verification Commands` with directly runnable commands:

```md
## Verification Commands

- Focused DSSD verifier regression:

  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py tests/v1/worker/test_dssd_worker_base.py tests/v1/dssd/test_verifier_service.py -q`

- Broader DSSD smoke and control-plane coverage:

  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`

- Document structure sanity check:

  `rg -n '^## ' docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`
```

- [ ] **Step 4: Add the initial change-log entry**

Seed `Change Log` with the first entry that explains why the document exists:

```md
## Change Log

### 2026-04-05

- Created the first global DSSD roadmap document.
- Consolidated current branch status, startup context, validated next work, candidate directions, and verification commands into one place.
- Declared this document the default entry point for new DSSD development sessions on `dssd-vllm-codex`.
```

- [ ] **Step 5: Review the document as a cold-start agent would**

Run:

```bash
sed -n '1,260p' /home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md
```

Expected:

```text
A concise, self-contained document that explains how to start, what is done, what remains, what to avoid, and what to run.
```

- [ ] **Step 6: Confirm git sees the new roadmap file**

Run:

```bash
git -C /home/zz/workspace/vllm/.worktrees/dssd-vllm-codex status --short --branch
```

Expected:

```text
The new roadmap file appears as an added or untracked doc in the DSSD worktree.
```
