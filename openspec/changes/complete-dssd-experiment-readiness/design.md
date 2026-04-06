## Context

The DSSD-on-vLLM branch already satisfies the core edge/verifier pipeline:
protocol, verifier replay, edge draft and resample, round coordination, and
streaming all exist and are covered by focused regressions. Phase 5 remains the
unfinished part. Current code provides only a narrow experiment foundation:

- transport-level latency, jitter, and bandwidth simulation in
  `vllm/v1/dssd/transport.py`
- request-local counters in `vllm/v1/dssd/metrics.py`
- a verifier batch planner that groups typed items but is not on the main
  verifier execution path

That is enough for functional smoke tests, but not enough for thesis-grade
evaluation. The missing gap is not the DSSD algorithm itself; it is experiment
observability, reproducible baseline comparison, and integrated verifier-side
batch execution.

## Goals / Non-Goals

**Goals:**
- Emit machine-readable DSSD experiment metrics for acceptance, rejection,
  communication, and latency.
- Integrate verifier round batching for compatible requests without changing
  DSSD round semantics.
- Provide a reproducible way to compare DSSD runs against a non-DSSD baseline
  under aligned model and sampling settings.

**Non-Goals:**
- Redesign the already working DSSD edge/verifier round semantics.
- Expand DSSD support to multimodal, tool-calling, or full serving-surface
  parity.
- Pursue deep performance optimization beyond what is needed for experiment
  readiness and stable comparison.

## Decisions

### 1. Keep the minimal experiment foundation as a separate baseline capability

The current transport simulation and minimal counters are already true today, so
they stay in baseline specs instead of being reintroduced inside this change.
This keeps the change focused on missing experiment-facing behavior.

Alternative considered:
- Modify the foundation spec directly. Rejected because it would blur the line
  between already-shipped behavior and new experiment requirements.

### 2. Add observability as an explicit DSSD capability, not as ad-hoc logging

Experiment outputs should be defined as a capability with clear requirements for
which metrics exist and how they are emitted. The implementation can use
request-scoped collectors and per-round events, but the contract should be about
the resulting metrics rather than a specific storage mechanism.

Alternative considered:
- Rely on manual logs or test-only assertions. Rejected because that is too weak
  for repeatable experiment analysis.

### 3. Batch verifier rounds at the verifier-side aggregation boundary

The existing `VerifierRoundBatcher` already groups requests by typed keys. The
new work should integrate batching near the verifier service or session-runner
boundary, where compatible requests are visible together and can still fall back
to the existing per-request path when batching is impossible or unsafe.

Alternative considered:
- Batch only inside worker hooks. Rejected because the higher-level verifier
  service would still have no contract for compatibility, ordering, or fallback.

### 4. Treat baseline comparison as a first-class experiment workflow

Experiment runs need an explicit way to choose DSSD mode versus a non-DSSD
baseline while keeping prompt, model, and sampling settings aligned. This
should be represented as an experiment-facing capability rather than an
unwritten convention of manual flags.

Alternative considered:
- Leave baseline comparison to ad-hoc shell commands. Rejected because drift in
  prompts, model choice, or sampling settings would invalidate results.

## Risks / Trade-offs

- [Risk] Batching may accidentally perturb request ordering or session semantics
  -> Mitigation: batch only compatible requests, preserve per-session ordering,
  and retain the current unbatched path as fallback.
- [Risk] Richer metrics can add overhead to the hot path
  -> Mitigation: keep collectors request-scoped and lightweight, and make
  experiment export opt-in where possible.
- [Risk] DSSD and baseline runs may diverge in configuration
  -> Mitigation: define experiment mode selection around shared prompt/model/
  sampling inputs and validate the exported run metadata.

## Migration Plan

1. Add the new experiment capability specs first so the target behavior is
   explicit.
2. Implement metrics collection and export without changing default DSSD round
   semantics.
3. Integrate verifier batching behind a safe compatibility check and fallback.
4. Add experiment baseline selection and regression coverage.

Rollback is straightforward because the work is additive: batching can fall back
to the current per-request verifier path, and experiment outputs can be disabled
without removing the existing DSSD serving pipeline.

## Open Questions

- What export surface is most useful for thesis analysis: direct JSON artifacts,
  benchmark summaries, or both?
- Should verifier batching be wired at the service layer, the session runner,
  or both with a narrow interface between them?
