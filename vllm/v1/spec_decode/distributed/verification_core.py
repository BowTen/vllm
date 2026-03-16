# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Protocol

from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    OpenSessionRequest,
    OpenSessionResponse,
    ResyncSessionRequest,
    ResyncSessionResponse,
    VerificationResult,
)


class VerificationRunner(Protocol):
    async def open_session(
        self, request: OpenSessionRequest
    ) -> OpenSessionResponse: ...

    async def close_session(self, request: CloseSessionRequest) -> None: ...

    async def resync_session(
        self, request: ResyncSessionRequest
    ) -> ResyncSessionResponse: ...

    async def verify_proposal(
        self, proposal: DraftProposal
    ) -> VerificationResult: ...


@dataclass
class CloudSessionRecord:
    session_id: str
    version: int
    last_activity_ts: float


class CloudSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, CloudSessionRecord] = {}

    def register(self, session_id: str, version: int) -> None:
        self._sessions[session_id] = CloudSessionRecord(
            session_id=session_id,
            version=version,
            last_activity_ts=time.monotonic(),
        )

    def touch(self, session_id: str, version: int | None = None) -> None:
        record = self._sessions[session_id]
        record.last_activity_ts = time.monotonic()
        if version is not None:
            record.version = version

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def has(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get(self, session_id: str) -> CloudSessionRecord:
        return self._sessions[session_id]


class ProposalScheduler:
    def __init__(self, runner: VerificationRunner) -> None:
        self.runner = runner
        self._queue: asyncio.Queue[
            tuple[DraftProposal, asyncio.Future[VerificationResult]]
        ] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

    async def enqueue(self, proposal: DraftProposal) -> VerificationResult:
        if self._closed:
            raise RuntimeError("ProposalScheduler is closed.")
        self._ensure_worker()
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((proposal, future))
        return await future

    async def shutdown(self) -> None:
        self._closed = True
        worker = self._worker_task
        if worker is None:
            return
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        self._worker_task = None

    def _ensure_worker(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._run(), name="DistributedSpecProposalScheduler"
            )

    async def _run(self) -> None:
        while True:
            proposal, future = await self._queue.get()
            try:
                result = await self.runner.verify_proposal(proposal)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)


class VerificationCore:
    def __init__(self, runner: VerificationRunner) -> None:
        self.runner = runner
        self.registry = CloudSessionRegistry()
        self.scheduler = ProposalScheduler(runner)

    async def open_session(
        self, request: OpenSessionRequest
    ) -> OpenSessionResponse:
        response = await self.runner.open_session(request)
        self.registry.register(request.session_id, response.session_version)
        return response

    async def verify_proposal(
        self, proposal: DraftProposal
    ) -> VerificationResult:
        result = await self.scheduler.enqueue(proposal)
        if self.registry.has(proposal.session_id):
            self.registry.touch(proposal.session_id, result.verifier_version)
        return result

    async def resync_session(
        self, request: ResyncSessionRequest
    ) -> ResyncSessionResponse:
        response = await self.runner.resync_session(request)
        if self.registry.has(request.session_id):
            self.registry.touch(request.session_id, response.session_version)
        else:
            self.registry.register(request.session_id, response.session_version)
        return response

    async def close_session(self, request: CloseSessionRequest) -> None:
        await self.runner.close_session(request)
        self.registry.close(request.session_id)

    async def shutdown(self) -> None:
        await self.scheduler.shutdown()
