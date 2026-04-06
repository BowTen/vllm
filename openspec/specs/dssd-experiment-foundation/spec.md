## Purpose

Define the baseline experiment-oriented DSSD infrastructure that already exists
today: transport-level network simulation and minimal per-request metrics.

## Requirements

### Requirement: DSSD transport supports configurable network simulation

The DSSD baseline SHALL allow transport calls to apply artificial delay derived
from configured latency, jitter, and optional bandwidth parameters.

#### Scenario: Simulated network delay applies to request and response payloads
- **WHEN** an HTTP DSSD transport is configured with network simulation
  parameters and sends a verifier control-plane request
- **THEN** it applies artificial delay before the request and before consuming
  the response payload using the configured delay model

### Requirement: DSSD request metrics track minimal per-request counters

The DSSD baseline SHALL provide a request-scoped metrics object that can
accumulate uplink bytes, downlink bytes, accepted tokens, and rejected rounds.

#### Scenario: Request metrics accumulate communication bytes
- **WHEN** a DSSD request records multiple uplink and downlink payload sizes
- **THEN** the request metrics reflect the cumulative byte counts

#### Scenario: Request metrics accumulate accept and reject counters
- **WHEN** a DSSD request records accepted token counts and rejected rounds
  across multiple verifier interactions
- **THEN** the request metrics reflect the cumulative accepted-token and
  rejected-round totals
