# Program: DSSD on vLLM

## Purpose

This program turns vLLM into a research-oriented edge-cloud DSSD prototype.
The edge instance owns the user-facing request lifecycle and streaming output.
The verifier instance hosts the target model and only serves internal DSSD verification traffic.

The baseline system design remains documented in
`docs/superpowers/specs/2026-04-05-dssd-vllm-design.md`.
OpenSpec is now the workflow source of truth for current baseline capabilities and future changes.

## System Boundaries

- Edge vLLM
  - Exposes the OpenAI-compatible user entrypoint.
  - Runs the draft model.
  - Performs residual resampling after verifier rejection.
  - Streams only committed output tokens to the user.
- Verifier vLLM
  - Runs the target model.
  - Exposes internal `bind`, `create_session`, `verify_round`, and `close_session` APIs.
  - Maintains verifier session state and accept/reject decisions.
- DSSD control path
  - Lives in vLLM serving/control layers rather than replacing the existing speculative decode mainline.
  - Uses EngineCore utility paths and worker/model-runner hooks for draft and verifier execution.
- Transport and experiments
  - Even single-machine development uses real RPC boundaries.
  - Transport may inject latency, bandwidth, and jitter for experiment setup.

## Current Baseline

The repository baseline already includes the core DSSD pipeline:

- Phase 1 complete in capability terms:
  protocol, config, serving skeleton, verifier routing, and engine utility wiring exist.
- Phase 2 complete in capability terms:
  verifier replay, accept/reject logic, and bonus-token handling exist.
- Phase 3 complete in capability terms:
  edge draft execution, reject-side residual resampling, and committed-prefix updates exist.
- Phase 4 complete:
  end-to-end edge serving, round coordination, streaming, finish handling, and session cleanup exist.
- Phase 5 is only partially complete:
  basic batching scaffolding, transport simulation, and minimal metrics exist, but experiment-grade metrics and full experiment-readiness are not yet complete.

## Baseline Capabilities

The baseline OpenSpec specs should describe committed system behavior in capability slices, not historical implementation phases.
Expected capability areas include:

- Edge serving and streaming ownership
- Verifier control-plane protocol and session lifecycle
- Verifier replay and accept/reject semantics
- Edge draft, resampling, and commit flow
- Experiment transport simulation and minimal request metrics

## Out of Scope for the Baseline

Unless a future change explicitly expands scope, the baseline remains focused on:

- text-only requests
- single-request-first correctness and system shape
- stateful verifier sessions
- basic sampling parameters

The baseline does not aim to fully support:

- multimodal
- structured output
- tool calling
- beam search
- parallel sampling
- LoRA
- full parity with all vLLM serving features

## OpenSpec Workflow Rules

- `openspec/specs/` records behaviors the repository already commits to support.
- `openspec/changes/` records active deltas from that baseline.
- Historical implementation phases should be translated into capability specs rather than copied verbatim.
- Unfinished work, starting with Phase 5 experiment support, should be advanced through new OpenSpec changes.

## Reference Documents

- Baseline design:
  `docs/superpowers/specs/2026-04-05-dssd-vllm-design.md`
- vLLM architecture reference:
  `/home/zz/workspace/vllm/docs/vllm-architecture.md`
- vLLM OpenAI server codemap:
  `/home/zz/workspace/vllm/docs/codemap-qwen3.5-openai-server.md`
