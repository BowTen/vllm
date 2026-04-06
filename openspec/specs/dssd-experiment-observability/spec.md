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
