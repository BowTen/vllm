## Purpose

Define the baseline control-plane contract between edge and verifier, including
binding, remote-session lifecycle, and ordered verifier-round progression.

## Requirements

### Requirement: Verifier exposes the DSSD control-plane lifecycle

The DSSD baseline SHALL provide internal verifier APIs for binding, remote
session creation, per-round verification, and remote session closure.

#### Scenario: Edge binds before verifier-backed generation
- **WHEN** an edge instance prepares to send DSSD traffic to a verifier
- **THEN** it can establish a verifier binding through a bind request

#### Scenario: Edge creates and closes remote verifier sessions
- **WHEN** a DSSD request starts and later finishes
- **THEN** the verifier accepts `create_session` and `close_session` requests
  scoped to that binding

### Requirement: Verifier sessions enforce ordered round progression

The DSSD baseline SHALL track verifier session state and reject inconsistent
round sequencing instead of silently advancing ambiguous state.

#### Scenario: Prefix deltas are applied before round verification
- **WHEN** a `verify_round` request includes committed-prefix delta together
  with draft tokens for the next round
- **THEN** the verifier applies that delta to the verifier session state before
  evaluating the round

#### Scenario: Sequence mismatches fail closed
- **WHEN** a `verify_round` request arrives with an unexpected sequence number
  for the verifier session
- **THEN** the verifier returns an error and does not silently advance the
  session state
