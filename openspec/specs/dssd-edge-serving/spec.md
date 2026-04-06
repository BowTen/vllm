## Purpose

Define the baseline user-facing responsibilities of the DSSD edge instance,
including request ownership, output ownership, and remote-session cleanup.

## Requirements

### Requirement: Edge owns the user-facing DSSD request lifecycle

The DSSD baseline SHALL route DSSD-enabled user chat completion requests
through the edge instance, which owns request state, round coordination, and
final response construction.

#### Scenario: Edge serves DSSD-enabled chat completions
- **WHEN** DSSD edge mode handles a chat completion request
- **THEN** the request is coordinated by the edge-side DSSD serving path rather
  than by a verifier-side user endpoint

#### Scenario: Edge remains the only user-visible output authority
- **WHEN** verifier validation occurs during a DSSD generation
- **THEN** the final non-streaming response and any streamed chunks are emitted
  by the edge instance

### Requirement: Edge streams only committed tokens and finalizes remote state

The DSSD baseline SHALL expose only verifier-confirmed output tokens to the
user and SHALL attempt remote session cleanup when a DSSD request finishes or
aborts after verifier interaction.

#### Scenario: Rejected draft tokens are not streamed
- **WHEN** a draft round is partially or fully rejected
- **THEN** only the committed token delta is streamed to the user

#### Scenario: Finishing a DSSD request closes the verifier session
- **WHEN** a DSSD request completes or terminates after a remote verifier
  session has been created
- **THEN** the edge coordinator attempts to close the verifier session before
  releasing local request state
