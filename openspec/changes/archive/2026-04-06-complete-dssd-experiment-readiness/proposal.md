## Why

Phase 5 is only partially complete today. The repository has basic transport
simulation, minimal request metrics, and verifier batching scaffolding, but it
still cannot produce the experiment-grade outputs and execution behavior needed
for repeatable thesis evaluation of DSSD versus baselines.

## What Changes

- Add experiment observability requirements for per-round acceptance, reject
  position, communication, and latency reporting.
- Add verifier round batching requirements that move the existing batch planner
  scaffold onto a real verifier execution path for compatible rounds.
- Add experiment baseline requirements for reproducible DSSD and non-DSSD
  comparison runs under aligned prompt, model, and sampling settings.
- Keep the existing experiment foundation as baseline context instead of
  redefining already completed behavior.

## Capabilities

### New Capabilities
- `dssd-experiment-observability`: Thesis-grade DSSD metrics and machine-readable
  experiment outputs.
- `dssd-verifier-round-batching`: Integrated batching of compatible verifier
  rounds while preserving per-session semantics.
- `dssd-experiment-baselines`: Reproducible experiment mode selection and result
  comparison between DSSD and baseline serving.

### Modified Capabilities
- None.

## Impact

- Affected code will likely include `vllm/v1/dssd/metrics.py`,
  `vllm/v1/dssd/engine/batch_planner.py`,
  `vllm/v1/dssd/engine/session_runner.py`,
  `vllm/v1/dssd/verifier/service.py`, edge coordination paths, and experiment
  or benchmark-facing configuration surfaces.
- New tests will be needed for metric collection, batching semantics, and
  experiment-mode selection.
