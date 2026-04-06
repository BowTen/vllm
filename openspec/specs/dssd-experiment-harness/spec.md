## Purpose

Define the config-driven local experiment harness for comparing DSSD against a
target-only baseline under aligned prompt, sampling, and network conditions.

## Requirements

### Requirement: Experiment harness accepts declarative DSSD run configuration

The repository SHALL provide an experiment-facing harness that accepts a
machine-readable run definition for DSSD experiments.

#### Scenario: Run definition selects edge, target, and network parameters
- **WHEN** a user provides an experiment definition for a DSSD run
- **THEN** the harness accepts configuration for the draft model, target model,
  experiment mode, gamma, sampling settings, and network-simulation parameters
  needed for that run

#### Scenario: Sweep definition expands into concrete run cases
- **WHEN** a user provides list-valued sweep fields such as multiple model pairs
  or latency profiles
- **THEN** the harness expands them into explicit concrete run cases before
  execution

### Requirement: Experiment harness orchestrates aligned DSSD and target-only baseline runs

The experiment harness SHALL be able to execute DSSD and target-only baseline
runs under aligned prompt and sampling inputs.

#### Scenario: Harness launches a DSSD run
- **WHEN** an experiment case is resolved in DSSD mode
- **THEN** the harness launches the verifier and edge processes required for
  that case, waits for readiness, and executes the configured prompt workload

#### Scenario: Harness launches a target-only baseline comparison run
- **WHEN** an experiment case is resolved in target-baseline mode
- **THEN** the harness launches a single target-model serving process, executes
  the same prompt and sampling inputs, and does not require any draft-model or
  DSSD verifier workflow for that case

#### Scenario: Target-only baseline stays aligned with its matching DSSD case
- **WHEN** the harness resolves a DSSD case and a matching target-baseline case
- **THEN** both cases use the same target model, prompt workload, sampling
  settings, and network profile so the resulting metrics can be compared fairly

### Requirement: Experiment harness persists offline analysis artifacts

The experiment harness SHALL persist machine-readable artifacts for each
resolved run.

#### Scenario: Completed run emits summary and detail artifacts
- **WHEN** an experiment case completes
- **THEN** the harness writes a run manifest plus summary output and preserves
  any detailed per-request experiment traces associated with that run

#### Scenario: DSSD summary rows report speedup against matching target baseline
- **WHEN** a DSSD case has a matching completed target-baseline case
- **THEN** the summary output includes a speedup metric derived from the DSSD
  result and that matching target-only baseline result

#### Scenario: Failed run still records execution context
- **WHEN** an experiment case fails during startup or request execution
- **THEN** the harness records the resolved run parameters, commands, and failure
  status in the output artifacts
