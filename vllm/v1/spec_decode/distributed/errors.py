# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


class VerifierSessionMissingError(RuntimeError):
    """The verifier no longer has runtime state for the requested session."""


class VerifierQueueTimeoutError(TimeoutError):
    """The verifier timed out a proposal before it started verification."""
