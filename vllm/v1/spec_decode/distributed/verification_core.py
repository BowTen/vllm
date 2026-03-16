# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from vllm.v1.spec_decode.distributed.errors import VerifierQueueTimeoutError
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

    async def verify_proposals_batch(
        self, proposals: list[DraftProposal]
    ) -> list[VerificationResult]: ...


@dataclass
class CloudSessionRecord:
    session_id: str
    version: int
    prefix_len: int
    opened_at_ts: float
    last_activity_ts: float
    active_proposals: int = 0


class CloudSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, CloudSessionRecord] = {}

    def register(self, session_id: str, version: int, prefix_len: int) -> None:
        now = time.monotonic()
        self._sessions[session_id] = CloudSessionRecord(
            session_id=session_id,
            version=version,
            prefix_len=prefix_len,
            opened_at_ts=now,
            last_activity_ts=now,
        )

    def touch(
        self,
        session_id: str,
        version: int | None = None,
        prefix_len: int | None = None,
    ) -> None:
        record = self._sessions[session_id]
        record.last_activity_ts = time.monotonic()
        if version is not None:
            record.version = version
        if prefix_len is not None:
            record.prefix_len = prefix_len

    def mark_proposal_started(self, session_id: str) -> None:
        record = self._sessions[session_id]
        record.active_proposals += 1
        record.last_activity_ts = time.monotonic()

    def mark_proposal_finished(
        self,
        session_id: str,
        version: int | None = None,
        prefix_len: int | None = None,
    ) -> None:
        record = self._sessions[session_id]
        record.active_proposals = max(0, record.active_proposals - 1)
        self.touch(session_id, version=version, prefix_len=prefix_len)

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def has(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get(self, session_id: str) -> CloudSessionRecord:
        return self._sessions[session_id]

    def get_idle_session_ids(self, idle_timeout_s: float) -> list[str]:
        cutoff = time.monotonic() - idle_timeout_s
        return [
            session_id
            for session_id, record in self._sessions.items()
            if record.active_proposals == 0 and record.last_activity_ts <= cutoff
        ]


@dataclass
class SchedulerQueueItem:
    proposal: DraftProposal
    future: asyncio.Future[VerificationResult]
    enqueued_ts: float


class ProposalScheduler:
    def __init__(
        self,
        runner: VerificationRunner,
        max_batch_size: int = 8,
        batch_wait_ms: float = 1.0,
        max_batch_total_tokens: int | None = None,
        queue_timeout_ms: float | None = None,
    ) -> None:
        self.runner = runner
        self.max_batch_size = max(1, max_batch_size)
        self.batch_wait_s = max(0.0, batch_wait_ms / 1000.0)
        self.max_batch_total_tokens = max_batch_total_tokens
        self.queue_timeout_s = (
            None if queue_timeout_ms is None else max(0.0, queue_timeout_ms / 1000.0)
        )
        self._queue: asyncio.Queue[SchedulerQueueItem] = asyncio.Queue()
        self._pending_items: deque[SchedulerQueueItem] = deque()
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

    async def enqueue(self, proposal: DraftProposal) -> VerificationResult:
        if self._closed:
            raise RuntimeError("ProposalScheduler is closed.")
        self._ensure_worker()
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(
            SchedulerQueueItem(
                proposal=proposal,
                future=future,
                enqueued_ts=time.monotonic(),
            )
        )
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
            first_item = await self._get_batch_start_item()
            batch = await self._collect_batch(first_item)
            if not batch:
                continue
            proposals = [item.proposal for item in batch]
            try:
                results = await self._verify_batch(proposals)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
            else:
                if len(results) != len(batch):
                    exc = ValueError(
                        "verify_proposals_batch returned an unexpected result count: "
                        f"{len(results)} != {len(batch)}."
                    )
                    for item in batch:
                        if not item.future.done():
                            item.future.set_exception(exc)
                    continue
                for item, result in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(result)

    async def _collect_batch(
        self,
        first_item: SchedulerQueueItem,
    ) -> list[SchedulerQueueItem]:
        batch: list[SchedulerQueueItem] = []
        seen_session_ids: set[str] = set()
        total_tokens = 0
        total_tokens = self._try_add_to_batch(
            batch,
            seen_session_ids,
            first_item,
            total_tokens,
        )
        if self.max_batch_size == 1:
            return batch

        if self.batch_wait_s > 0:
            deadline = time.monotonic() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                timeout_s = deadline - time.monotonic()
                if timeout_s <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    break
                total_tokens = self._try_add_to_batch(
                    batch,
                    seen_session_ids,
                    item,
                    total_tokens,
                )

        while len(batch) < self.max_batch_size:
            try:
                total_tokens = self._try_add_to_batch(
                    batch,
                    seen_session_ids,
                    self._queue.get_nowait(),
                    total_tokens,
                )
            except asyncio.QueueEmpty:
                break
        return batch

    async def _verify_batch(
        self, proposals: list[DraftProposal]
    ) -> list[VerificationResult]:
        verify_batch = getattr(self.runner, "verify_proposals_batch", None)
        if callable(verify_batch):
            return await verify_batch(proposals)
        return [await self.runner.verify_proposal(proposal) for proposal in proposals]

    async def _get_batch_start_item(self) -> SchedulerQueueItem:
        if self._pending_items:
            return self._pending_items.popleft()
        return await self._queue.get()

    def _try_add_to_batch(
        self,
        batch: list[SchedulerQueueItem],
        seen_session_ids: set[str],
        item: SchedulerQueueItem,
        current_total_tokens: int,
    ) -> int:
        if self._is_expired(item):
            if not item.future.done():
                item.future.set_exception(
                    VerifierQueueTimeoutError(
                        "Proposal timed out in verifier queue."
                    )
                )
            return current_total_tokens

        proposal = item.proposal
        if proposal.session_id in seen_session_ids and batch:
            self._pending_items.append(item)
            return current_total_tokens

        proposal_tokens = max(1, len(proposal.draft_token_ids))
        if (
            self.max_batch_total_tokens is not None
            and batch
            and current_total_tokens + proposal_tokens > self.max_batch_total_tokens
        ):
            self._pending_items.append(item)
            return current_total_tokens

        batch.append(item)
        seen_session_ids.add(proposal.session_id)
        return current_total_tokens + proposal_tokens

    def _is_expired(self, item: SchedulerQueueItem) -> bool:
        if self.queue_timeout_s is None:
            return False
        return (time.monotonic() - item.enqueued_ts) > self.queue_timeout_s


class VerificationCore:
    def __init__(
        self,
        runner: VerificationRunner,
        scheduler_max_batch_size: int = 8,
        scheduler_batch_wait_ms: float = 1.0,
        scheduler_max_batch_tokens: int | None = None,
        scheduler_queue_timeout_ms: float | None = None,
        session_idle_timeout_s: float | None = None,
    ) -> None:
        self.runner = runner
        self.registry = CloudSessionRegistry()
        self.scheduler = ProposalScheduler(
            runner,
            max_batch_size=scheduler_max_batch_size,
            batch_wait_ms=scheduler_batch_wait_ms,
            max_batch_total_tokens=scheduler_max_batch_tokens,
            queue_timeout_ms=scheduler_queue_timeout_ms,
        )
        self.session_idle_timeout_s = session_idle_timeout_s
        self._reaper_task: asyncio.Task[None] | None = None

    async def open_session(
        self, request: OpenSessionRequest
    ) -> OpenSessionResponse:
        self._ensure_reaper()
        response = await self.runner.open_session(request)
        self.registry.register(
            request.session_id,
            response.session_version,
            len(request.prompt_token_ids),
        )
        return response

    async def verify_proposal(
        self, proposal: DraftProposal
    ) -> VerificationResult:
        self._ensure_reaper()
        if self.registry.has(proposal.session_id):
            self.registry.mark_proposal_started(proposal.session_id)
        try:
            result = await self.scheduler.enqueue(proposal)
        except Exception:
            if self.registry.has(proposal.session_id):
                self.registry.mark_proposal_finished(proposal.session_id)
            raise
        if self.registry.has(proposal.session_id):
            self.registry.mark_proposal_finished(
                proposal.session_id,
                version=result.verifier_version,
                prefix_len=self._compute_prefix_len_after_verification(
                    proposal,
                    result,
                ),
            )
        return result

    async def resync_session(
        self, request: ResyncSessionRequest
    ) -> ResyncSessionResponse:
        self._ensure_reaper()
        response = await self.runner.resync_session(request)
        if self.registry.has(request.session_id):
            self.registry.touch(
                request.session_id,
                response.session_version,
                len(request.accepted_prefix_token_ids),
            )
        else:
            self.registry.register(
                request.session_id,
                response.session_version,
                len(request.accepted_prefix_token_ids),
            )
        return response

    async def close_session(self, request: CloseSessionRequest) -> None:
        self._ensure_reaper()
        await self.runner.close_session(request)
        self.registry.close(request.session_id)

    async def shutdown(self) -> None:
        await self.scheduler.shutdown()
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None
        shutdown = getattr(self.runner, "shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if inspect.isawaitable(result):
                await result

    def _ensure_reaper(self) -> None:
        if self.session_idle_timeout_s is None or self.session_idle_timeout_s <= 0:
            return
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(
                self._run_idle_session_reaper(),
                name="DistributedSpecVerifierIdleSessionReaper",
            )

    async def _run_idle_session_reaper(self) -> None:
        assert self.session_idle_timeout_s is not None
        sleep_s = min(max(self.session_idle_timeout_s / 2.0, 0.05), 1.0)
        while True:
            await asyncio.sleep(sleep_s)
            for session_id in self.registry.get_idle_session_ids(
                self.session_idle_timeout_s
            ):
                await self.runner.close_session(
                    CloseSessionRequest(session_id=session_id)
                )
                self.registry.close(session_id)

    def _compute_prefix_len_after_verification(
        self,
        proposal: DraftProposal,
        result: VerificationResult,
    ) -> int:
        prefix_len = proposal.accepted_prefix_len + result.accepted_len
        if result.reject_pos is None and result.bonus_token_id is not None:
            prefix_len += 1
        return prefix_len
