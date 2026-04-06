## 1. Experiment Harness Foundation

- [x] 1.1 Add a benchmark-facing experiment config model and validation path for
  draft/target models, experiment mode, network profiles, prompts, and output
  settings
- [x] 1.2 Implement sweep expansion from one experiment definition into concrete
  run cases with aligned DSSD and target-only baseline inputs
- [x] 1.3 Implement DSSD verifier/edge command builders plus target-only
  baseline subprocess lifecycle management with readiness checks

## 2. Result Export and Aggregation

- [x] 2.1 Add an opt-in edge-side sink that writes `dssd_experiment_result`
  payloads to harness-readable JSONL artifacts
- [x] 2.2 Implement harness-side request execution and collection of summary
  metrics, commands, and detailed DSSD traces into an output directory
- [x] 2.3 Emit normalized summary artifacts such as run-manifest JSON and CSV or
  JSON summaries for offline analysis, including speedup versus matching
  target-only baseline cases

## 3. Docs and Verification

- [x] 3.1 Add regression tests for config parsing, target-only baseline
  selection, sweep expansion, command construction, speedup calculation, and
  experiment trace export
- [x] 3.2 Add a smoke-style benchmark or harness test that exercises one local
  DSSD case and one target-only baseline case end to end with stubs or
  lightweight engines
- [x] 3.3 Document the experiment harness workflow, including example configs
  for Qwen and OPT model pairs and common latency or gamma sweeps
