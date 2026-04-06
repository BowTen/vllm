# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.utils import config


@config
class DSSDNetworkSimulationConfig:
    """Configuration for optional DSSD network simulation."""

    latency_ms: float = 0.0
    """Injected one-way latency in milliseconds."""
    bandwidth_mbps: float | None = None
    """Optional network bandwidth cap in megabits per second."""
    jitter_ms: float = 0.0
    """Injected jitter in milliseconds."""


@config
class DSSDConfig:
    """Configuration for DSSD edge/verifier mode."""

    enabled: bool = False
    """Enable DSSD mode."""
    role: str | None = None
    """DSSD role, either `edge` or `verifier` when enabled."""
    gamma: int = 4
    """Number of draft tokens per verification round."""
    experiment_mode: str = "dssd"
    """Experiment mode for edge runs, either `dssd` or `baseline`."""
    verifier_url: str | None = None
    """Verifier endpoint URL required for edge mode."""
    network_simulation: DSSDNetworkSimulationConfig = field(
        default_factory=DSSDNetworkSimulationConfig
    )
    """Optional local network simulation knobs."""

    def validate(self) -> "DSSDConfig":
        if not self.enabled:
            return self

        if self.role not in {"edge", "verifier"}:
            raise ValueError("DSSDConfig.role must be 'edge' or 'verifier'")

        if self.gamma < 1:
            raise ValueError("DSSDConfig.gamma must be >= 1")

        if self.experiment_mode not in {"dssd", "baseline"}:
            raise ValueError(
                "DSSDConfig.experiment_mode must be 'dssd' or 'baseline'"
            )

        if (
            self.role == "edge"
            and self.experiment_mode == "dssd"
            and not self.verifier_url
        ):
            raise ValueError("DSSD edge mode requires verifier_url")

        return self
