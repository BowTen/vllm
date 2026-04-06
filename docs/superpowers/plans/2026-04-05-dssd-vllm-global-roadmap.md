# DSSD Global Roadmap

> **Legacy Notice**
>
> This document is retained as historical branch context.
> The current DSSD workflow source of truth has moved to
> [openspec/program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md).
> New work should start from OpenSpec, not by extending this file.

## Quick Start for New Agents
- Active worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`
- Active branch: `dssd-vllm-codex`
- Upstream: `origin/dssd-vllm-codex`
- Read this document first, then use `Working Queue` as the default entry point.
- If you need a quick sanity check before editing, run:
  `rg -n '^## ' /home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`
- Use Python virtualenvs; prefer `uv` for Python dependency management.
- Activate the existing venv before verification commands: `source .venv/bin/activate`
- If network access is needed and direct access fails, prefer mirror sources first; only then use `proxy on`.
- Keep large model and data artifacts under `/data/zz/`; do not commit them into the repo.
- Most important current pitfalls:
  - `GPUModelRunner.dssd_draft_round()` still replays the full prefix on every draft step; preserve the sampler-based `q_values` / `q_distributions` semantics when changing it.
  - Stub-based DSSD tests now isolate `sys.modules` locally; do not reintroduce module-level `vllm.*` stubs or they will pollute the real-model regression path.
  - The real-model DSSD regression uses a fixed small `kv_cache_memory_bytes` budget to avoid shared-container GPU profiling jitter.

## Usage
- This is the default DSSD execution board and persistent agent context for this branch.
- Update `Current Status`, `Working Queue`, `Known Pitfalls`, and `Change Log` after every completed development task.
- Update `Verification Commands` when the recommended verification path changes.
- Re-rank `Validated Next Work`, `Candidate Directions`, and `Phase Roadmap` whenever implementation results change priorities.
- Do not promote a candidate item into the active queue until it is validated by current code, tests, or design constraints.
- This document complements the design spec, handoff note, and implementation plan:
  - `docs/superpowers/specs/2026-04-05-dssd-vllm-design.md` explains intended system shape and constraints.
  - `docs/superpowers/handoffs/2026-04-05-dssd-vllm-handoff.md` captures the most recent transfer state.
  - `docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md` breaks a stage into executable steps.
  - this roadmap keeps the live branch status, priority order, and next-agent entry point current.

## Environment Facts
- Main repo root: `/home/zz/workspace/vllm`
- Active DSSD worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`
- Active branch: `dssd-vllm-codex`
- Remote `origin` points to `git@github.com:BowTen/vllm.git`
- Python workflow requires a virtualenv; prefer `uv` for environment and dependency management.
- Verification commands should run after `source .venv/bin/activate`.
- Network workflow: prefer mirror sources first; use `proxy on` only if mirror-based access still fails.
- Large files, including weights and datasets, belong under `/data/zz/`; link them into the workspace if needed.
- Avoid committing large artifacts into the repo.

## Current Status
- DSSD config surface is in place.
- Edge and verifier role split is implemented.
- Verifier HTTP routes are implemented.
- EngineCore utility RPC path is implemented.
- Verifier session tracking is implemented.
- Verifier replay reaches a real logits path through `GPUModelRunner.dssd_verify_round()` and now routes through a validated worker-local `execute_verifier_replay_request()` scratch helper instead of building the replay `SchedulerOutput` entirely in `verifier_runner.py`.
- The validated direct-request baseline now keeps only session-summary worker state across rounds (`seq_no`, committed-prefix extension, and `seq_no == 0` reset fail-closed semantics); verifier scratch requests and batch rows are cleaned after each round instead of persisting as worker-local cached request/KV state.
- Verifier close now has an explicit worker-side cleanup path: `DSSDSessionRunner.close_verifier_session()` dispatches `dssd_close_verifier_session` to workers before deleting the engine-side session, so failed worker cleanup remains retryable instead of leaving a half-closed engine/service/worker mismatch.
- Worker-side verifier replay scratch execution now supports batch-of-N at the helper boundary: `execute_verifier_replay_request()` is a batch-of-1 wrapper over a batched scratch helper, and `run_verifier_replay_forward()` is likewise a batch-of-1 wrapper over a batched verifier replay path.
- `VerifierRoundBatcher` is now a typed grouping boundary (`VerifierRoundBatchKey` / `VerifierRoundBatchItem` / `VerifierRoundBatch`) with focused regression coverage, but engine/service-side verify aggregation is still intentionally deferred and the public path still executes batch-of-1.
- Draft execution now replays request sampling params through a real worker-side sampler path, including greedy, `min_p`, and output-history-dependent penalty semantics, but still uses conservative per-step replay rather than session or KV reuse.
- Edge execution now runs a real multi-round DSSD loop that only surfaces verifier-confirmed tokens, with fail-closed handling for seq mismatch and zero-progress rounds.
- Edge chat execution now fails closed when request sampling params cannot be resolved, instead of silently falling back to greedy draft replay.
- A real-model DSSD end-to-end regression now launches actual edge and verifier `AsyncLLM` engines, routes round trips through `HTTPDSSDTransport` and `DSSDVerifierService`, and confirms worker/session invariants under real model execution.
- Stub-based protocol, coordinator, and smoke regressions now isolate their `vllm.*` import shims per test module so they can coexist with the real-model validation path in one regression run.
- The real-model regression currently stabilizes shared-container GPU startup with a fixed small `kv_cache_memory_bytes` budget and a temporary `current_platform.is_cuda` override around engine bring-up.
- Edge chat execution now supports verified-delta round streaming: `request.stream=True` returns minimal OpenAI SSE chunks (role, verified delta content, finish, optional usage, `[DONE]`) on top of the same fail-closed multi-round loop used by the non-streaming path.
- Streaming error chunks now preserve the validated fail-closed status codes from the shared DSSD path instead of silently downgrading transport/preflight/runtime failures to generic `400` payloads.
- Real-model validation now covers both non-streaming and streaming DSSD execution, so actual edge/verifier engines plus `HTTPDSSDTransport` are gated under both response modes.
- Verifier service close now treats `None` from the async engine close hook as success, which matches the current real-model utility RPC behavior and keeps verifier session cleanup validated under the streaming/non-streaming real-model gate.

## Validated Next Work
1. No remaining unconditional mainline items are active in the current roadmap; promote new work only when current code or tests expose a concrete need beyond the validated streaming baseline.
2. Promote engine/service-side verifier batch aggregation only when the current tests or experiment harness expose a concrete multi-request need beyond the typed batch boundary already in place.
3. Revisit stricter worker-side round/broadcast robustness or OpenAI-surface metadata/usage parity only when current streaming validation exposes a concrete productization gap rather than as speculative cleanup.

## Candidate Directions
- More aggressive verifier batching and scheduling optimizations.
- Richer network simulation beyond the current latency and bandwidth delay model.
- More advanced prefix and KV reuse beyond the first stable worker-side integration.
- Broader productization features outside the current text-only prototype scope.

## Phase Roadmap
- Phase 1: completed. Full multi-round edge coordination now commits only verifier-confirmed tokens and fails closed on zero-progress rounds.
- Phase 2: completed. Draft replay now respects request sampling semantics instead of the old greedy fallback.
- Phase 3: completed. Verifier replay now reuses worker-side cached request state across rounds while preserving conservative accept/reject semantics.
- Phase 4: completed. Real-model end-to-end validation now exercises actual edge and verifier execution and keeps the lightweight stub regressions isolated from that path.
- Phase 5: completed. Direct-request ownership/lifecycle, worker-side scratch batching, and verified-delta edge/OpenAI streaming are now all validated on the real-model baseline; deeper engine/service-side aggregation or additional robustness work stays conditional on a concrete need.

## Working Queue
1. No active non-optional mainline queue items remain after the validated streaming cut; only conditional follow-up work remains.
2. Promote engine/service-side verifier batch aggregation only if a concrete multi-request source appears; do not treat deeper aggregation as the default next step now that the typed batch boundary already exists.
3. Revisit additional close/round robustness hardening or OpenAI-surface metadata/usage parity only if the current streaming baseline exposes a concrete correctness or productization gap.

## Known Pitfalls
- `GPUModelRunner.dssd_draft_round()` still replays the full prefix on every draft step; when changing that path, preserve the new sampler-based `q_values` / `q_distributions` semantics for greedy, `min_p`, and penalty-aware output history.
- The validated verifier direct-request baseline is a scratch helper, not persistent worker-side verifier KV reuse: `execute_verifier_replay_request()` synthesizes local block IDs for one round only and must clean request/batch residue in the helper `finally`.
- The validated verifier batching cut is still a scratch batch boundary, not a public multi-request service path: batch-of-N exists inside the worker/verifier helper layer, but the public real-model path still depends on batch-of-1 wrappers.
- Worker-side verifier replay now keeps only session-summary state across rounds (`seq_no`, committed-prefix extension, and `seq_no == 0` reset fail-closed semantics); do not silently rebuild broader cached-request assumptions on top of that baseline.
- Verifier close cleanup is now explicit and ordered: worker `dssd_close_verifier_session` cleanup must happen before deleting the engine-side verifier session, or close becomes half-failed and non-retryable.
- `GPUModelRunner.execute_verifier_replay_request()` must preserve the same no-grad / inference-mode execution contract as `execute_model()`; otherwise verifier replay can regress on memory and execution behavior even if unit tests stay green.
- Batched verifier result splitting currently relies on metadata ordering matching request ordering inside the scratch prefill batch; preserve that assumption or replace it with a stronger validated mapping before attempting deeper cached/decode-style reuse.
- `tests/v1/e2e/dssd/test_dssd_smoke.py` still validates lightweight HTTP/control-plane wiring rather than true model execution; use `tests/v1/e2e/dssd/test_dssd_real_model.py` when you need real edge/verifier execution evidence.
- The real-model regression currently depends on `engine_core.call_utility_async("dssd_*", ...)`, a fixed `kv_cache_memory_bytes=128 * 1024 * 1024`, and a temporary `current_platform.is_cuda` override to stay stable in this shared container. Preserve that setup or replace it with a stronger validated startup path before changing the test.
- Stub-based DSSD regressions now save and restore `sys.modules` locally; do not move those `vllm.*` stubs back to module import time or they will break mixed real-model and lightweight regression runs.
- The edge loop now fails closed on zero-progress rounds and mismatched `seq_no`; keep those guards intact when changing coordinator state semantics.
- The edge chat path now also fails closed on unresolved sampling params; do not reintroduce silent greedy fallback unless a stronger validated recovery path replaces it.
- The validated streaming cut intentionally emits only a minimal OpenAI SSE surface (role chunk, verified-delta content chunks, finish chunk, optional usage, `[DONE]`); do not quietly expand that surface without fresh validation.
- The DSSD streaming path still bypasses some of the base chat-serving metadata/usage plumbing; if middleware or observability starts depending on `request_metadata.final_usage_info`, validate and wire that explicitly instead of assuming parity.
- Real-model verifier close currently relies on treating `None` from the async engine close hook as success; preserve that behavior unless the underlying utility RPC path starts returning a stronger validated boolean result.
- The active worktree may contain untracked roadmap or spec docs under `docs/superpowers/`; check intent before cleaning them up.
- Earlier verifier replay fixes had to handle empty committed prefixes, request cleanup, and cached-request synchronization; treat prefix semantics and transient-versus-persistent cleanup boundaries as fragile areas.

## Verification Commands
- Run these from the active worktree: `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex`.
- Focused verifier replay caching regression:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py -q`
- Focused DSSD draft and verifier regression:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_draft_runner.py tests/v1/worker/test_dssd_verifier_runner.py tests/v1/worker/test_dssd_worker_base.py tests/v1/dssd/test_verifier_service.py -q`
- Focused DSSD streaming regression:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_round_coordinator.py tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py -q`
- Focused verifier service close regression:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_verifier_service.py -q`
- Real-model edge/verifier validation:
  `source .venv/bin/activate && pytest --noconftest tests/v1/e2e/dssd/test_dssd_real_model.py -q`
- Mixed DSSD regression baseline with real-model plus lightweight protocol/control-plane coverage:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_real_model.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
- Document structure sanity check:
  `rg -n '^## ' /home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/docs/superpowers/plans/2026-04-05-dssd-vllm-global-roadmap.md`

## Decision Rules
- Promote an item from `Candidate Directions` into `Validated Next Work` only when current code, tests, or design constraints show it is an actual near-term requirement.
- Update `Current Status`, `Working Queue`, `Known Pitfalls`, and `Change Log` after every completed development task.
- Re-rank `Validated Next Work` and `Phase Roadmap` when implementation results change priorities, then record the reason in `Change Log`.
- Preserve fail-closed behavior for edge-loop `seq_no` mismatches and zero-progress rounds unless a stronger validated recovery path replaces it.
- Preserve fail-closed behavior for unresolved edge sampling params and preserve sampler-derived draft `q` semantics unless a stronger validated alternative replaces them.
- Preserve the validated verifier scratch-helper lifecycle (`execute_verifier_replay_request()` scratch cleanup, session-summary-only cross-round state, `seq_no == 0` reset, and prefix-divergence fail-closed cleanup) unless a stronger validated ownership model replaces it.
- Preserve the batch-of-1 public verifier path on top of the new batched scratch helper unless a stronger validated engine/service-side aggregation path replaces it.
- Do not keep worker-local verifier request or KV state alive across successful rounds unless block ownership, cleanup on close, and the real-model gate all validate that lifecycle explicitly.
- Keep verifier close cleanup ordered as worker cleanup first, engine-session deletion second, unless a stronger validated retry model replaces it.
- Keep `GPUModelRunner.execute_verifier_replay_request()` on the same inference/no-grad contract as `execute_model()` unless fresh validation proves a different contract is safe.
- Preserve local `sys.modules` isolation in the stub-based DSSD regressions so the real-model validation path can remain in the same regression baseline.
- Keep the real-model validation path on a fixed, explicitly validated startup budget unless engine-side memory profiling becomes deterministic enough to remove that workaround with fresh evidence.
- Use the real-model validation path as the gate for deeper direct request/KV integration; do not land worker/session changes that break `tests/v1/e2e/dssd/test_dssd_real_model.py` or the mixed 4-file regression baseline.
- Preserve verified-delta streaming semantics: only stream tokens after verifier confirmation and after `_apply_response_constraints()` has established the visible prefix for that round.
- Preserve the current minimal DSSD SSE shape unless a stronger validated OpenAI-surface parity requirement replaces it.
- Treat `None` from the async verifier close hook as success until the real-model utility RPC path proves it can return a stronger boolean contract consistently.
- If actual implementation results invalidate the current route, update this roadmap before starting the next coding task.

## Change Log
### 2026-04-06
- Completed the Working Queue item to add verified-delta round streaming on top of the validated multi-round edge loop and real-model DSSD baseline.
- Reworked `DSSDRoundCoordinator` so non-streaming and streaming now share the same verified round iterator, which emits only verifier-confirmed deltas after response constraints are applied and keeps zero-progress / `seq_no` fail-closed behavior intact.
- Reworked `DSSDEdgeServingChat` so `request.stream=True` returns a thin DSSD SSE path instead of the old `501`, while `request.stream=False` keeps the final-response path.
- Added focused streaming regressions for coordinator verified-delta iteration, DSSD serving stream delegation, verifier-service close handling when the async close hook returns `None`, and real-model streaming SSE coverage.
- Fixed a follow-up review finding by preserving original `503` / `500` status codes in DSSD streaming error chunks and adding direct regressions for transport-missing and fail-closed runtime streaming errors.
- Verified the streaming cut with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_round_coordinator.py tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py -q`
  which passed with `21 passed`;
  with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_verifier_service.py -q`
  which passed with `5 passed`;
  with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/e2e/dssd/test_dssd_real_model.py -q`
  which passed with `2 passed`;
  and with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_real_model.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which passed with `35 passed`.
- Re-ranked the roadmap so no unconditional mainline items remain: the previous streaming / broader end-to-end realism queue item is now complete, while deeper verifier aggregation and additional robustness work remain conditional on concrete need instead of mandatory next steps.
- Completed the next worker-side batching sub-goal by promoting verifier replay from a single-request scratch helper to a batched scratch helper, while keeping the public path as batch-of-1 wrappers so the real-model validation path stays stable.
- Reworked `GPUModelRunner` to execute multiple verifier scratch requests in one helper batch and reworked `verifier_runner.py` so single-request replay is now a wrapper over a batched replay path with per-request logits/spec-metadata splitting.
- Typed `VerifierRoundBatcher` into explicit batch key/item/group structures and added focused regressions for typed grouping, batch-of-2 scratch replay, and batch-of-1 wrapper semantics.
- Kept the worker lifecycle rules unchanged while adding batching: scratch residue still cleans up per round, session state still keeps only summary fields, close still cleans workers before deleting the engine-side session, and the public real-model path still runs through batch-of-1 wrappers.
- Verified the worker-side batching cut with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py tests/v1/dssd/test_verifier_batch_planner.py tests/v1/worker/test_dssd_draft_runner.py -q`
  which passed with `51 passed`;
  and with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_real_model.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which passed with `30 passed`.
- Updated `tests/v1/worker/test_gpu_model_runner.py` so its verifier scratch test now targets the current batch-of-1 wrapper contract instead of the removed pre-batch helper seam; this file still cannot be collected in the current environment because importing `vllm._C.abi3.so` hits a local symbol mismatch, so only `py_compile` validation was possible here.
- Re-ranked the roadmap so streaming and broader end-to-end realism are now the next active mainline items, because the typed batching boundary exists and the public batch-of-1 path is already validated under the mixed real-model gate.
- Completed the direct-request ownership/lifecycle sub-goal for Phase 5 by turning verifier replay into a validated worker-local scratch helper with explicit close cleanup instead of prematurely promoting persistent worker-side cached request/KV reuse.
- Reworked `GPUModelRunner.execute_verifier_replay_request()` so it runs under `@torch.inference_mode()`, builds a per-round scratch request/batch row, and always cleans verifier scratch residue after execution while leaving logits/spec metadata available until `verifier_runner.py` clears transient state.
- Simplified verifier replay worker semantics so cross-round state now keeps only the session summary (`seq_no`, committed-prefix extension, and `seq_no == 0` reset fail-closed behavior), rather than assuming a worker cached request survives across rounds.
- Added an explicit worker close path by routing `dssd_close_verifier_session` through `DSSDSessionRunner`, `WorkerBase`, `GPUModelRunner`, and the verifier worker helper, and fixed the close ordering so worker cleanup happens before the engine-side verifier session is deleted.
- Added focused regressions for success/no-residue cleanup, round 2 replay without cached worker requests, non-output PP scratch cleanup, worker close cleanup, worker-base close routing, GPU-model-runner inference/scratch cleanup, and close retryability when worker cleanup fails.
- Verified the ownership/lifecycle cut with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py tests/v1/worker/test_dssd_worker_base.py tests/v1/worker/test_dssd_draft_runner.py -q`
  which passed with `50 passed`;
  and with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_real_model.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which passed with `30 passed`.
- Re-ranked Phase 5 so broader worker-side reuse/batching is now the next active non-optional item, because the scratch-helper baseline plus explicit close lifecycle is now validated under both focused regressions and the real-model gate.
- Attempted the next direct request/KV integration cut by routing verifier replay through a new worker-local `GPUModelRunner.execute_verifier_replay_request()` helper instead of constructing the replay `SchedulerOutput` entirely in `verifier_runner.py`.
- Spec review passed for that boundary change, but quality review and the real-model gate found blocking lifecycle issues before the task could be marked complete.
- Confirmed a fresh mixed regression failure with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_real_model.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which failed because the new direct-request helper constructs `SamplingParams(...)` without importing it, and because the current helper path still lacks explicit worker-side ownership/lifecycle guarantees strong enough to replace the conservative replay-plus-state baseline.
- Re-ranked Phase 5 so explicit direct-request ownership/lifecycle is now the immediate sub-goal before any broader worker-side reuse or batching, because the prototype showed that persistent worker-local state without a validated close path and ownership model is too fragile to promote.
- Completed the Working Queue item to add a real-model DSSD end-to-end validation path with actual edge and verifier execution.
- Added `tests/v1/e2e/dssd/test_dssd_real_model.py`, which launches real edge and verifier `AsyncLLM` engines, routes requests through `HTTPDSSDTransport` plus `DSSDVerifierService`, and validates that draft/verify/create_session/close_session all execute under actual model inference.
- Stabilized the real-model regression for this shared GPU container by fixing a small `kv_cache_memory_bytes` budget and avoiding startup-time memory-profiling jitter that otherwise made the second engine flaky.
- Reworked `tests/v1/e2e/dssd/test_dssd_smoke.py`, `tests/v1/dssd/test_protocol.py`, and `tests/v1/dssd/test_round_coordinator.py` so their lightweight `vllm.*` stubs are installed and restored inside local fixtures instead of polluting `sys.modules` at module import time.
- Verified the real-model validation cut with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/e2e/dssd/test_dssd_real_model.py -q`
  which passed with `1 passed`;
  and with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_real_model.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which passed with `30 passed`.
- Re-ranked the roadmap so deeper direct request/KV integration is now the next active item, because replay-plus-state caching and the real-model validation baseline now cover the current worker/session boundary well enough to move deeper.
- Completed the Working Queue item to convert verifier replay from one-shot per-round new requests into replay-plus-state caching across verifier rounds while preserving conservative accept/reject semantics.
- Reworked verifier replay scheduling so the first round still uses `scheduled_new_reqs`, while later rounds reuse worker-side cached requests through `scheduled_cached_reqs`, committed-prefix synchronization, and a `last committed token + draft` replay suffix.
- Added fail-closed verifier replay reset/branch protection so `seq_no == 0` drops stale worker replay state and committed-prefix divergence tears down the cached request instead of reusing ambiguous worker state.
- Added a verifier replay recovery regression covering the bootstrap path where worker cached requests survive but `_dssd_verifier_replay_sessions` is missing, so cached-request reuse remains tested even after partial worker-state loss.
- Verified the verifier replay caching cut with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py -q`
  which passed with `30 passed`;
  with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_draft_runner.py tests/v1/worker/test_dssd_verifier_runner.py tests/v1/worker/test_dssd_worker_base.py tests/v1/dssd/test_verifier_service.py -q`
  which passed with `50 passed`;
  and with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which passed with `29 passed`.
- Re-ranked the roadmap so real-model end-to-end validation is now the next active item, because replay-plus-state caching has stabilized the worker/session boundary enough to validate the full path before deeper direct request/KV integration.
- Started Phase 3 verifier stabilization by adding an explicit `DSSDVerifierReplayState` helper around the single-request replay/spec-decode worker view, so verifier request identity, prompt/spec layout, and scheduler input semantics now have a reusable state boundary.
- Verified the new verifier helper/state layer with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py -q`
  which passed with `24 passed`;
  and with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py tests/v1/worker/test_dssd_worker_base.py tests/v1/dssd/test_verifier_service.py -q`
  which passed with `30 passed`.
- Re-ranked the queue so the next active verifier task is cross-round replay-plus-state caching, because the helper/state boundary is now in place and direct request/KV integration remains deferred.
- Chose `replay-plus-state caching` as the verifier mainline for Phase 3 and deferred direct request/KV integration to the later state-reuse phase, because the roadmap/spec/handoff all favor stabilizing conservative replay semantics before deeper worker/KV coupling.
- Re-ranked the verifier queue around a reusable single-request replay/spec-decode helper plus session-backed worker-state caching, so the next implementation cut can lock verifier input/state semantics before broader integration.
- Completed the Working Queue item to plumb request sampling params through edge draft replay and remove the old greedy fallback semantics from draft `q_values` / `q_distributions`.
- Reworked `GPUModelRunner.dssd_draft_round()` replay sampling to use a processed-logprobs `Sampler` path with prompt/output-history-aware metadata, so greedy, `min_p`, and penalty-aware sampling now align with request semantics.
- Made the edge chat path fail closed when request sampling params cannot be resolved, and updated smoke/control-plane tests to provide explicit sampling params under the stricter contract.
- Verified the updated draft and edge paths with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/worker/test_dssd_draft_runner.py tests/v1/worker/test_dssd_verifier_runner.py tests/v1/worker/test_dssd_worker_base.py tests/v1/dssd/test_verifier_service.py -q`
  which passed with `42 passed`;
  and with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which passed with `29 passed`.
- Re-ranked the roadmap to make verifier route selection the next active item, because draft sampling semantics are no longer the main blocker and verifier worker-state shape now determines the next non-optional implementation path.
- Completed the Working Queue item to replace the single-round edge path with a real multi-round DSSD loop in `DSSDRoundCoordinator`.
- Added multi-round coordinator coverage for full-accept and reject-resample flows, plus a zero-progress fail-closed regression and HTTP smoke coverage for residual prefix-delta propagation.
- Verified the updated coordinator path with:
  `source .venv/bin/activate && pytest --noconftest tests/v1/dssd/test_protocol.py tests/v1/dssd/test_round_coordinator.py tests/v1/e2e/dssd/test_dssd_smoke.py -q`
  which passed with `26 passed`.
- Re-ranked the roadmap so sampling-parameter plumbing is now the first active item, because the edge multi-round control loop is no longer the main blocker.

### 2026-04-05
- Created the first global DSSD roadmap document.
- Consolidated current branch status, startup context, validated next work, candidate directions, and verification commands into one place.
- Declared this document the default entry point for new DSSD development sessions on `dssd-vllm-codex`.
- Re-ranked the roadmap to draft execution first, then multi-round edge flow, then verifier worker-state integration, based on the current branch state after verifier replay bring-up.
- Replaced `GPUModelRunner.dssd_draft_round()` placeholder behavior with a real worker-side replay path that builds per-step scheduler inputs, samples draft tokens from cached logits, and returns real `q_values` / `q_distributions`.
- Added focused draft-runner tests for helper construction, replay rollout, cleanup, and pipeline-parallel reply selection.
- Re-ranked the roadmap to multi-round edge flow first, then sampling-parameter plumbing for draft replay, then verifier worker-state integration, because the draft placeholder is now gone but the edge path is still single-round and the draft utility still uses greedy fallback semantics.
