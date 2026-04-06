## Purpose

Define the baseline DSSD round semantics for draft proposal generation,
verifier-side accept or reject decisions, and edge-side commit and resample
behavior.

## Requirements

### Requirement: Edge draft rounds produce verifier-checkable proposals

The DSSD baseline SHALL execute local draft rounds that produce draft-token
proposals and q-values suitable for verifier validation.

#### Scenario: Edge starts a new draft round
- **WHEN** edge generation enters a new DSSD round
- **THEN** the edge draft path produces draft token ids and their associated
  q-values for the verifier request

### Requirement: Verifier rounds return DSSD accept and reject outcomes

The DSSD baseline SHALL evaluate committed prefix plus draft proposals against
verifier replay and return accept or reject information following DSSD round
semantics.

#### Scenario: Fully accepted rounds may include a bonus token
- **WHEN** the verifier accepts every draft token in a round
- **THEN** it returns an accepted result and may include a verifier-side bonus
  token for the edge to commit

#### Scenario: Rejected rounds identify the first rejection point
- **WHEN** the verifier rejects a draft token in a round
- **THEN** it returns the first reject position together with the verifier
  distribution needed for edge-side residual resampling

### Requirement: Edge commits accepted tokens and resamples rejects locally

The DSSD baseline SHALL advance committed edge state using verifier-confirmed
tokens and SHALL perform residual resampling locally when a verifier rejection
occurs.

#### Scenario: Reject paths append a local residual sample
- **WHEN** the verifier rejects a round after some accepted draft prefix
- **THEN** the edge commits the accepted prefix, performs local residual
  resampling, and appends the sampled replacement token to committed state

#### Scenario: Accepted rounds advance committed state before later rounds
- **WHEN** the verifier accepts a round, with or without a bonus token
- **THEN** the edge advances committed prefix state before using those tokens in
  subsequent round coordination or final output handling
