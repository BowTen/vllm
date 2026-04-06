## Why

The branch can now run live DSSD requests, but experiment work is still mostly
manual: users have to hand-start verifier and edge, vary flags by hand, and
collect results from ad-hoc logs or one-off curls. That is enough for bring-up,
but not for repeatable thesis-style sweeps over model pairs, gamma, network
profiles, and DSSD versus target-only baseline comparisons.

## What Changes

- Add a config-driven DSSD experiment harness that can launch DSSD runs with
  selected draft and target models, plus target-only baseline runs that use the
  same target model, prompt set, sampling, and network-simulation settings.
- Add sweep support so one experiment definition can expand into multiple runs
  across model pairs, latency/bandwidth/jitter profiles, and DSSD versus
  target-only baseline modes.
- Add machine-readable result capture for experiment runs, including run-level
  metadata, summary metrics, and detailed per-request DSSD traces that an
  external harness can archive offline.
- Add documentation and regression coverage for the harness configuration,
  command generation, and result-export path.

## Capabilities

### New Capabilities
- `dssd-experiment-harness`: Config-driven orchestration of reproducible DSSD
  and target-only baseline experiment runs, including sweep expansion, result
  pairing, and artifact output.

### Modified Capabilities
- `dssd-experiment-observability`: Add an opt-in export surface that writes
  per-request DSSD experiment results to harness-readable artifacts instead of
  keeping them only inside in-process request state.

## Impact

- Likely affected code includes `vllm/benchmarks/`, DSSD config and serving
  layers, edge experiment-result publishing, and benchmark or docs entrypoints.
- New tests will be needed for experiment-config parsing, target-only baseline
  pairing, subprocess command building, speedup calculation, and experiment
  trace export.
- Users will gain a reproducible experiment workflow without relying on ad-hoc
  shell loops copied from external repositories.
