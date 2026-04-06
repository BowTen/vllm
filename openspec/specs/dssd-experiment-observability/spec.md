## Purpose

Define the experiment-grade observability contract for DSSD runs, including
machine-readable request summaries and per-round acceptance metrics.

## Requirements

### Requirement: DSSD experiment runs expose request-level summary metrics

The system SHALL emit machine-readable request-level summaries for DSSD
experiment runs, including communication totals, accepted-token totals,
rejected-round totals, and end-to-end latency summary.

#### Scenario: Completed DSSD request exports summary metrics
- **WHEN** a DSSD experiment request completes
- **THEN** the exported result includes the request's uplink bytes, downlink
  bytes, total accepted tokens, total rejected rounds, and request-level latency
  summary

### Requirement: DSSD experiment runs expose round-level acceptance metrics

The system SHALL emit per-round experiment metrics that describe verifier
acceptance behavior and round timing.

#### Scenario: Round metrics include acceptance and timing fields
- **WHEN** a DSSD round is executed during an experiment run
- **THEN** the exported round record includes accepted-count information and the
  round's draft, verify, or round-trip latency fields

#### Scenario: Rejecting rounds expose the first reject position
- **WHEN** a verifier round rejects part of a draft proposal
- **THEN** the exported round record includes the first reject position for that
  round

### Requirement: DSSD experiment results can be exported for external harnesses

The system SHALL provide an opt-in export surface that persists completed DSSD
experiment results outside in-process request state.

#### Scenario: Completed request appends a machine-readable trace record
- **WHEN** an edge experiment run is configured with an external result sink and
  a DSSD request completes
- **THEN** the system appends a machine-readable record containing the request
  summary, round metrics, and run metadata for that request

#### Scenario: Disabled sink preserves existing serving behavior
- **WHEN** no external result sink is configured for the experiment run
- **THEN** the system preserves the current serving behavior and does not require
  any public API response changes to complete the request
