# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx
import msgspec

from .protocol import (
    BindVerifierRequest,
    BindVerifierResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    VerifyRoundRequest,
    VerifyRoundResponse,
)


class SimulatedNetworkMixin:
    def __init__(self, network_simulation: Any | None = None) -> None:
        self.network_simulation = network_simulation

    async def _apply_network_simulation(self, payload_size_bytes: int) -> None:
        if self.network_simulation is None:
            return

        latency_ms = max(getattr(self.network_simulation, "latency_ms", 0.0), 0.0)
        jitter_ms = max(getattr(self.network_simulation, "jitter_ms", 0.0), 0.0)
        bandwidth_mbps = getattr(self.network_simulation, "bandwidth_mbps", None)

        delay_s = latency_ms / 1000.0
        if jitter_ms:
            delay_s += jitter_ms / 1000.0
        if bandwidth_mbps:
            bytes_per_second = bandwidth_mbps * 1_000_000 / 8
            delay_s += payload_size_bytes / bytes_per_second
        if delay_s > 0:
            await asyncio.sleep(delay_s)


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

    @abstractmethod
    async def create_session(
        self, request: CreateSessionRequest
    ) -> CreateSessionResponse:
        raise NotImplementedError

    @abstractmethod
    async def close_session(
        self, request: CloseSessionRequest
    ) -> CloseSessionResponse:
        raise NotImplementedError


class HTTPDSSDTransport(SimulatedNetworkMixin, DSSDTransport):
    def __init__(
        self,
        base_url: str,
        network_simulation: Any | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(network_simulation=network_simulation)
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(base_url=self.base_url)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def bind_verifier(
        self, request: BindVerifierRequest
    ) -> BindVerifierResponse:
        return await self._post(
            "/server/dssd/bind",
            request,
            BindVerifierResponse,
        )

    async def create_session(
        self, request: CreateSessionRequest
    ) -> CreateSessionResponse:
        return await self._post(
            "/server/dssd/sessions",
            request,
            CreateSessionResponse,
        )

    async def verify_round(
        self, request: VerifyRoundRequest
    ) -> VerifyRoundResponse:
        return await self._post(
            "/server/dssd/verify_round",
            request,
            VerifyRoundResponse,
        )

    async def close_session(
        self, request: CloseSessionRequest
    ) -> CloseSessionResponse:
        return await self._post(
            "/server/dssd/close_session",
            request,
            CloseSessionResponse,
        )

    async def _post(self, path: str, request: Any, response_type: Any):
        payload = msgspec.to_builtins(request)
        await self._apply_network_simulation(len(msgspec.json.encode(payload)))
        response = await self._client.post(path, json=payload)
        response.raise_for_status()
        data = response.json()
        await self._apply_network_simulation(len(msgspec.json.encode(data)))
        return msgspec.convert(data, type=response_type)
