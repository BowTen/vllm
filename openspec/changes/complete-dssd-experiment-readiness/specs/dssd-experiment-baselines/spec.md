## ADDED Requirements

### Requirement: Experiment workflow supports aligned DSSD and baseline runs

The system SHALL provide an experiment-facing workflow that can run DSSD mode
and a non-DSSD baseline mode under aligned prompt, model, and sampling inputs.

#### Scenario: Experiment mode selects DSSD versus non-DSSD execution
- **WHEN** an experiment run is configured for DSSD mode or baseline mode
- **THEN** the workflow executes the selected mode without changing the shared
  prompt, model, and sampling inputs for that run

### Requirement: Experiment outputs identify the execution mode and settings

The system SHALL export enough run metadata to make DSSD and baseline results
comparable offline.

#### Scenario: Exported run metadata identifies comparison context
- **WHEN** an experiment run finishes
- **THEN** the exported result identifies the execution mode, relevant model
  identifiers, sampling parameters, and network-simulation settings used for
  that run
