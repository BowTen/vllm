## ADDED Requirements

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
