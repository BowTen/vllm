# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod

from .protocol import (
    BindVerifierRequest,
    BindVerifierResponse,
    VerifyRoundRequest,
    VerifyRoundResponse,
)


class SimulatedNetworkMixin:
    async def _apply_network_simulation(self, payload_size_bytes: int) -> None:
        del payload_size_bytes
        return None


class DSSDTransport(ABC):
    @abstractmethod
    async def bind_verifier(
        self, request: BindVerifierRequest
    ) -> BindVerifierResponse:
        raise NotImplementedError

    @abstractmethod
    async def verify_round(
        self, request: VerifyRoundRequest
    ) -> VerifyRoundResponse:
        raise NotImplementedError
