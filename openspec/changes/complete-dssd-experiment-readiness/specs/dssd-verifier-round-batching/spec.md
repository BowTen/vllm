## ADDED Requirements

### Requirement: Compatible verifier rounds are grouped before execution

The system SHALL group verifier-round requests into batches only when they share
compatible batching keys, including gamma and sampling signature.

#### Scenario: Matching compatibility keys share one verifier batch
- **WHEN** multiple verifier-round requests are pending with the same gamma and
  sampling signature
- **THEN** the verifier batching path groups them into the same execution batch

#### Scenario: Incompatible rounds stay in separate batches
- **WHEN** verifier-round requests differ in gamma or sampling signature
- **THEN** they are placed into separate verifier batches

### Requirement: Batched verification preserves per-request semantics

The system SHALL return per-request verifier results that preserve the same
accept or reject semantics as the unbatched verifier path.

#### Scenario: Batched execution preserves per-session result mapping
- **WHEN** a verifier batch completes
- **THEN** each request in the batch receives the result corresponding to its
  own verifier session and round inputs

#### Scenario: Verifier falls back to unbatched execution when batching is unsafe
- **WHEN** the verifier cannot batch a pending round without violating
  compatibility or session-ordering constraints
- **THEN** it executes that round through the existing unbatched verifier path
