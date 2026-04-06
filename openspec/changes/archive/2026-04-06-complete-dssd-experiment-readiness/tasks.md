## 1. Experiment Observability

- [x] 1.1 Extend DSSD request and round metrics collection to cover the
  observability fields required by `dssd-experiment-observability`
- [x] 1.2 Add a machine-readable export path or summary surface for DSSD
  experiment metrics
- [x] 1.3 Add regression tests for request-level and round-level experiment
  metric output

## 2. Verifier Round Batching

- [x] 2.1 Integrate the existing verifier batch planner into a verifier-side
  execution path with compatibility checks
- [x] 2.2 Preserve unbatched fallback semantics for incompatible or unsafe batch
  candidates
- [x] 2.3 Add regression tests for compatible batching, incompatible separation,
  and per-request result mapping

## 3. Experiment Baselines

- [x] 3.1 Add an experiment-facing mode selection path for DSSD and non-DSSD
  baseline runs with aligned inputs
- [x] 3.2 Export run metadata needed to compare DSSD and baseline results
- [x] 3.3 Add regression or smoke coverage for the experiment baseline workflow
