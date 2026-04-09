# DSSD Verifier Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `docs/dssd/verifier_demo/` 下写一份基于真实 vLLM 组件命名的 verifier 端示意代码，便于阅读算法逻辑和后续映射真实实现。

**Architecture:** 采用 `service -> engine -> state_bridge / sampler / scheduler / types` 的分层方式。主流程尽量完整，和真实 vLLM 调度/KV 分配强耦合的点只保留最小占位和明确注释，不追求当前可运行。

**Tech Stack:** Python, dataclasses, PyTorch, vLLM v1 `GPUWorker/GPUModelRunner/SchedulerOutput/Sampler`

---

### Task 1: 建立 verifier demo 包结构

**Files:**
- Create: `docs/dssd/verifier_demo/__init__.py`
- Create: `docs/dssd/verifier_demo/types.py`

- [ ] Step 1: 新建包入口和共享类型
- [ ] Step 2: 用 dataclass 固定 `open_session / verify_round / result / session` 这些协议对象
- [ ] Step 3: 保持类型命名和 `docs/dssd/dssd-engine-fake.md` 一致

### Task 2: 写调度器和状态桥接

**Files:**
- Create: `docs/dssd/verifier_demo/scheduler.py`
- Create: `docs/dssd/verifier_demo/state_bridge.py`

- [ ] Step 1: 在 `scheduler.py` 里把 `open_session / verify_round / close_session` 对应到 `SchedulerOutput`
- [ ] Step 2: 在 `state_bridge.py` 里直接写 `req_states.last_sampled_tokens / draft_tokens` 的访问
- [ ] Step 3: 用中文注释写清楚哪些地方只是示意，真实实现要接 vLLM 私有状态

### Task 3: 写 DSSD sampler 和 decode/service 主流程

**Files:**
- Create: `docs/dssd/verifier_demo/sampler.py`
- Create: `docs/dssd/verifier_demo/engine.py`
- Create: `docs/dssd/verifier_demo/service.py`

- [ ] Step 1: 在 `sampler.py` 里实现 accept/reject、bonus token、reject logits 返回
- [ ] Step 2: 在 `engine.py` 里实现 `open_session -> prefill/bootstrap -> verify_round -> close_session`
- [ ] Step 3: 在 `service.py` 里实现最外层会话管理，保证入口方法可跳转
