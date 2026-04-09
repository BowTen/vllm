# DSSD Edge Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `docs/dssd/edge_demo/` 下写一份基于真实 vLLM 组件命名的 edge 端示意代码，便于阅读 DSSD 协调逻辑，并和现有 `verifier_demo` 对照。

**Architecture:** 采用 `service -> engine -> state_bridge / sampler / scheduler / types` 的分层方式。主流程覆盖 `open_session -> prefill -> draft -> verify response commit / rollback -> close_session`，与真实 vLLM 执行流强耦合的点只保留最小占位和明确注释。

**Tech Stack:** Python, dataclasses, PyTorch, vLLM v1 `GPUWorker/GPUModelRunner/SchedulerOutput/Sampler`

---

### Task 1: 建立 edge demo 包结构

**Files:**
- Create: `docs/dssd/edge_demo/__init__.py`
- Create: `docs/dssd/edge_demo/types.py`
- Test: `docs/dssd/edge_demo/test_edge_demo_ast.py`

- [ ] Step 1: 先写 AST 结构测试，约束 `types.py` 里的协议类型和 `service.py` 的入口类。
- [ ] Step 2: 运行测试，确认在实现前失败。
- [ ] Step 3: 用 dataclass 固定 `open_session / verify / session / round_state` 这些协议对象。

### Task 2: 写 scheduler / state bridge / draft sampler

**Files:**
- Create: `docs/dssd/edge_demo/scheduler.py`
- Create: `docs/dssd/edge_demo/state_bridge.py`
- Create: `docs/dssd/edge_demo/sampler.py`
- Test: `docs/dssd/edge_demo/test_edge_demo_ast.py`

- [ ] Step 1: 在测试里约束 `build_prefill_step / build_decode_step / free_blocks / rollback` 等关键方法。
- [ ] Step 2: 运行测试，确认仍然失败在缺失实现上。
- [ ] Step 3: 用最小代码实现这些类，并用中文注释标出真实 vLLM 接入点。

### Task 3: 写 edge engine 和 service 主流程

**Files:**
- Create: `docs/dssd/edge_demo/engine.py`
- Create: `docs/dssd/edge_demo/service.py`
- Modify: `docs/dssd/edge_demo/test_edge_demo_ast.py`

- [ ] Step 1: 在测试里约束 `draft / rollback / commit_verify_result / close_session` 的调用链。
- [ ] Step 2: 运行测试，确认失败。
- [ ] Step 3: 写最小实现，使结构测试通过，并补上 `py_compile` 校验。
