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


class DSSDTransport(ABC):
    @abstractmethod
    def bind_verifier(
        self, request: BindVerifierRequest
    ) -> BindVerifierResponse:
        raise NotImplementedError

    @abstractmethod
    def verify_round(
        self, request: VerifyRoundRequest
    ) -> VerifyRoundResponse:
        raise NotImplementedError
