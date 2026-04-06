## Context

The current DSSD baseline is ready to run real edge/verifier requests and to
record experiment metrics, but the workflow around those capabilities is still
primitive. The main gaps are:

- experiment runs are launched manually, usually by starting verifier and edge
  processes in separate terminals
- sweep dimensions such as drafter model, verifier model, gamma, and network
  latency are not described declaratively in one place
- DSSD request metrics are currently published only into `raw_request.state`,
  which is useful for tests but not for an external experiment harness
- the original paper repository uses ad-hoc shell loops and CSV append scripts,
  which are good references for the desired knobs but are not a good long-term
  shape for this vLLM branch

The new work should turn the existing DSSD serving path into something that can
be used for repeatable local experiments without requiring users to manually
compose commands and scrape logs.

## Goals / Non-Goals

**Goals:**
- Provide one harness entrypoint that can run DSSD experiments from a single
  config file or CLI invocation.
- Support selecting draft and target models, ports, GPU assignments, gamma,
  sampling settings, prompt inputs, and transport network-simulation knobs.
- Support aligned DSSD and target-only baseline comparisons from the same
  experiment definition.
- Persist experiment outputs in machine-readable artifacts that are easy to
  analyze offline, including detailed DSSD traces when available.
- Keep the workflow close enough to the original paper scripts that users can
  reproduce common sweeps without rewriting them by hand.

**Non-Goals:**
- Replace the general-purpose `vllm bench serve` tool for all serving
  benchmarks.
- Add cluster scheduling, remote SSH orchestration, or multi-node deployment.
- Redesign DSSD round semantics or the verifier protocol.
- Expose DSSD experiment internals by default on the public OpenAI response
  schema.

## Decisions

### 1. Implement the experiment workflow as a Python harness under `vllm/benchmarks`

The experiment runner should live next to the repository's other benchmark
tools, not as a standalone shell script. That keeps it testable, allows reuse
of existing request and readiness utilities, and avoids another pile of
machine-specific bash loops.

Alternative considered:
- Copy the paper repo's `config/search_*.sh` pattern directly. Rejected because
  it is hard to test, hard to validate, and drifts quickly once model paths,
  ports, or output shapes change.

### 2. Use a declarative JSON experiment config with sweep expansion

The harness should accept either direct CLI flags for a single run or a JSON
  config file for repeatable sweeps. JSON avoids adding a new parser dependency
  while still making matrices explicit. The config should describe:

- model selections and optional model pairs
- edge/verifier GPU or `CUDA_VISIBLE_DEVICES` assignments
- gamma and experiment mode selections
- network profiles with latency, bandwidth, and jitter
- prompt source and generation settings
- output directory and artifact naming

The harness can expand list-valued fields into a Cartesian run matrix so one
config produces a reproducible set of concrete runs.

Alternative considered:
- YAML configs. Rejected for the initial change because JSON is sufficient and
  avoids introducing another dependency surface.

### 3. Launch DSSD services and target-only baseline services as managed subprocesses

The harness should own process lifecycle for local experiments. For DSSD runs
it builds verifier and edge commands, starts them, waits for `/health`, runs
requests, then tears them down. For target-only baseline runs it starts a
single target-model server and runs the same request set and sampling settings
without any DSSD control path. DSSD runs must still apply the already-known
runtime constraints such as `--no-async-scheduling`.

Alternative considered:
- Require users to pre-start verifier and edge manually. Rejected because that
  defeats the goal of reproducibility and makes sweep automation brittle.

### 4. Export DSSD request traces through an opt-in JSONL sink

Detailed DSSD metrics already exist in memory, but an external harness cannot
see `raw_request.state`. The edge experiment path should therefore support an
opt-in file sink, configured through DSSD experiment settings, that appends one
JSON record per completed request. Each record should include:

- request summary metrics
- round metrics
- run metadata already generated today
- identifiers that let the harness join requests back to a run manifest

This keeps the public response schema stable while making experiment traces
available offline.

Alternative considered:
- Return DSSD experiment results directly in `/v1/chat/completions` responses.
  Rejected because it would couple a benchmark-only detail to the public API.

### 5. Emit both summary and detailed experiment artifacts, with explicit target-baseline pairing

The harness should produce two artifact layers:

- a run manifest plus summary rows, suitable for CSV/JSON aggregation across
  runs
- detailed JSONL traces for individual requests, either copied from the edge
  sink or normalized into a harness-owned output directory

That matches the paper repo's practical need for quick summary tables while
keeping the richer DSSD metrics available for later analysis. Summary rows
should also compute `speedup_vs_target_baseline` by pairing each DSSD case with
the matching target-only baseline case under the same target model, request
workload, sampling settings, and network profile.

Alternative considered:
- Summary-only CSV output. Rejected because it would discard the per-request and
  per-round observability that this branch already computes.

## Risks / Trade-offs

- [Risk] Local experiment orchestration may be fragile across GPU memory
  profiles or model families
  -> Mitigation: keep runtime options explicit in config, surface the exact
  commands used, and fail fast when readiness checks or startup logs indicate
  invalid settings.
- [Risk] Writing detailed JSONL traces can add I/O overhead
  -> Mitigation: make trace export opt-in and keep the format append-only and
  line-delimited.
- [Risk] Sweep configs can become too large or ambiguous
  -> Mitigation: validate configs early, expand them into explicit run manifests,
  and persist the resolved run matrix before execution.
- [Risk] Baseline and DSSD runs may still drift in hidden server flags
  -> Mitigation: derive both modes from the same resolved experiment inputs,
  keep the target model identical between them, and record the final commands
  plus run metadata.

## Migration Plan

1. Add spec coverage for the new experiment harness and the observability export
   sink.
2. Implement config parsing, sweep expansion, and command building in a harness
   module under `vllm/benchmarks`.
3. Add the opt-in experiment trace sink on the edge side and connect it to
   existing `dssd_experiment_result` payloads.
4. Implement process lifecycle, request execution, target-baseline pairing, and
   artifact writing.
5. Add docs and regression coverage for config validation, export behavior, and
   summary generation.

Rollback remains straightforward because the work is additive: users can keep
running verifier and edge manually, and the trace sink can be disabled without
changing the DSSD serving contract.

## Open Questions

- The first version should start with a smaller prompt-file format plus a fixed
  chat request shape.
- The first version should focus on single-request and low-rate experiment
  fidelity first.
