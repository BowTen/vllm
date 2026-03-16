# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import torch

from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.spec_decode.distributed.errors import VerifierSessionMissingError
from vllm.v1.spec_decode.distributed.logprobs import (
    build_logprobs_lists,
    build_logprobs_tensors,
    pack_sample_logprobs,
)
from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    OpenSessionRequest,
    OpenSessionResponse,
    PackedLogprobs,
    ResyncSessionRequest,
    ResyncSessionResponse,
    SamplingMetadata,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.runtime import (
    DraftProposalOutput,
    EdgeDraftRunner,
    deserialize_probs,
)
from vllm.v1.spec_decode.distributed.structured_output import (
    StructuredOutputFactory,
    StructuredOutputSession,
    accept_structured_output_tokens,
)
from vllm.v1.spec_decode.distributed.sampling import (
    is_terminal_token,
    sample_recovered_token,
)


class VerifierClient(Protocol):
    async def open_session(
        self,
        request: OpenSessionRequest,
    ) -> OpenSessionResponse: ...

    async def verify_proposal(
        self,
        proposal: DraftProposal,
    ) -> VerificationResult: ...

    async def resync_session(
        self,
        request: ResyncSessionRequest,
    ) -> ResyncSessionResponse: ...

    async def close_session(self, request: CloseSessionRequest) -> None: ...


@dataclass
class EdgeSessionState:
    request_id: str
    prompt_len: int
    sampling: SamplingMetadata
    generator: torch.Generator
    accepted_prefix_token_ids: list[int]
    structured_output_session: StructuredOutputSession | None = None
    version: int = 0
    proposal_id: int = 0


@dataclass
class EdgeStepResult:
    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None


class EdgeSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, EdgeSessionState] = {}

    def add(self, session: EdgeSessionState) -> None:
        self._sessions[session.request_id] = session

    def pop(self, request_id: str) -> EdgeSessionState | None:
        return self._sessions.pop(request_id, None)

    def get(self, request_id: str) -> EdgeSessionState:
        return self._sessions[request_id]

    def has(self, request_id: str) -> bool:
        return request_id in self._sessions

    def clear(self) -> None:
        self._sessions.clear()


class EdgeSessionCore:
    def __init__(
        self,
        draft_runner: EdgeDraftRunner,
        verifier: VerifierClient,
        structured_output_factory: StructuredOutputFactory | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        self._draft_runner = draft_runner
        self._verifier = verifier
        self._structured_output_factory = structured_output_factory
        self._num_speculative_tokens = num_speculative_tokens
        self.registry = EdgeSessionRegistry()

    def create_session(self, request: EngineCoreRequest) -> EdgeSessionState:
        if self.registry.has(request.request_id):
            raise ValueError(f"Session {request.request_id} already exists.")
        assert request.prompt_token_ids is not None
        sampling = SamplingMetadata.from_sampling_params(
            request.sampling_params  # type: ignore[arg-type]
        )
        seed = sampling.seed
        if seed is None:
            seed = torch.seed()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        sampling.structured_output_max_rollback = self._num_speculative_tokens
        session = EdgeSessionState(
            request_id=request.request_id,
            prompt_len=len(request.prompt_token_ids),
            sampling=sampling,
            generator=generator,
            accepted_prefix_token_ids=list(request.prompt_token_ids),
            structured_output_session=(
                self._structured_output_factory.create_session(
                    request.request_id,
                    sampling,
                )
                if self._structured_output_factory is not None
                else None
            ),
        )
        self.registry.add(session)
        return session

    async def run_request(
        self,
        request: EngineCoreRequest,
        session: EdgeSessionState,
    ) -> AsyncIterator[EdgeStepResult]:
        response = await self._verifier.open_session(
            OpenSessionRequest(
                session_id=request.request_id,
                prompt_token_ids=list(session.accepted_prefix_token_ids),
                sampling_metadata=session.sampling,
                initial_version=session.version,
            )
        )
        if (
            response.vocab_size is not None
            and response.vocab_size != self._draft_runner.vocab_size
        ):
            raise ValueError(
                "Draft and verifier vocab sizes differ: "
                f"{self._draft_runner.vocab_size} != {response.vocab_size}."
            )
        prompt_logprobs_tensors = build_logprobs_tensors(response.prompt_logprobs)
        if prompt_logprobs_tensors is not None:
            yield EdgeStepResult(
                request_id=request.request_id,
                new_token_ids=[],
                new_prompt_logprobs_tensors=prompt_logprobs_tensors,
            )

        while True:
            finish_reason = self._maybe_finish_without_new_tokens(session)
            if finish_reason is not None:
                yield EdgeStepResult(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=finish_reason,
                )
                return

            proposal_output = await self._draft_runner.propose(
                session_id=request.request_id,
                proposal_id=session.proposal_id,
                base_version=session.version,
                accepted_prefix_token_ids=session.accepted_prefix_token_ids,
                prompt_len=session.prompt_len,
                sampling=session.sampling,
                generator=session.generator,
                structured_output_session=session.structured_output_session,
            )
            proposal = proposal_output.proposal
            if not proposal.draft_token_ids:
                yield EdgeStepResult(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.LENGTH,
                )
                return

            try:
                verification = await self._verifier.verify_proposal(proposal)
            except VerifierSessionMissingError:
                await self._verifier.resync_session(
                    self._build_resync_request(session)
                )
                verification = await self._verifier.verify_proposal(proposal)
            (
                new_token_ids,
                new_logprobs,
                finish_reason,
                stop_reason,
                needs_resync,
            ) = self._apply_verification_result(
                session,
                proposal_output,
                verification,
            )

            if needs_resync and finish_reason is None:
                await self._verifier.resync_session(self._build_resync_request(session))

            yield EdgeStepResult(
                request_id=request.request_id,
                new_token_ids=new_token_ids,
                new_logprobs=new_logprobs,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )
            session.proposal_id += 1
            if finish_reason is not None:
                return

    async def close_session(self, request_id: str) -> None:
        self.registry.pop(request_id)
        self._draft_runner.close_session(request_id)
        with contextlib.suppress(Exception):
            await self._verifier.close_session(CloseSessionRequest(session_id=request_id))

    async def abort_sessions(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            await self.close_session(request_id)

    def clear_local_state(self) -> None:
        self.registry.clear()
        self._draft_runner.clear_sessions()

    def shutdown(self) -> None:
        self.clear_local_state()
        shutdown_runner = getattr(self._draft_runner, "shutdown", None)
        if callable(shutdown_runner):
            shutdown_runner()
        if self._structured_output_factory is not None:
            self._structured_output_factory.close()

    def _build_resync_request(
        self,
        session: EdgeSessionState,
    ) -> ResyncSessionRequest:
        return ResyncSessionRequest(
            session_id=session.request_id,
            accepted_prefix_token_ids=list(session.accepted_prefix_token_ids),
            edge_version=session.version,
            prompt_len=session.prompt_len,
            sampling_metadata=session.sampling,
        )

    def _apply_verification_result(
        self,
        session: EdgeSessionState,
        proposal_output: DraftProposalOutput,
        verification: VerificationResult,
    ) -> tuple[
        list[int],
        LogprobsLists | None,
        FinishReason | None,
        int | str | None,
        bool,
    ]:
        if verification.base_version != session.version:
            raise ValueError(
                f"Session {session.request_id} verification version mismatch: "
                f"got {verification.base_version}, expected {session.version}."
            )
        committed = list(verification.accepted_token_ids)
        committed_logprobs = (
            list(verification.accepted_logprobs)
            if session.sampling.logprobs is not None
            else []
        )
        needs_resync = verification.reject_pos is not None
        if needs_resync:
            target_probs = deserialize_probs(
                verification.target_probs_at_reject_pos or b"",
                self._draft_runner.vocab_size,
            )
            reject_pos = verification.reject_pos
            assert reject_pos is not None
            if reject_pos >= len(proposal_output.draft_token_distributions):
                raise ValueError(
                    f"Session {session.request_id} missing local draft distribution "
                    f"for reject position {reject_pos}."
                )
            sampled_token_id = sample_recovered_token(
                target_probs,
                proposal_output.draft_token_distributions[reject_pos],
                session.generator,
                greedy=session.sampling.temperature <= 0,
            )
            committed.append(sampled_token_id)
            if session.sampling.logprobs is not None:
                packed = pack_sample_logprobs(
                    target_probs,
                    sampled_token_id,
                    session.sampling.logprobs,
                )
                assert packed is not None
                committed_logprobs.append(packed)
        elif verification.bonus_token_id is not None:
            committed.append(verification.bonus_token_id)
            if (
                session.sampling.logprobs is not None
                and verification.bonus_logprobs is not None
            ):
                committed_logprobs.append(verification.bonus_logprobs)

        if (
            session.sampling.logprobs is not None
            and len(committed_logprobs) != len(committed)
        ):
            raise ValueError(
                f"Session {session.request_id} logprobs mismatch: "
                f"{len(committed_logprobs)} entries for {len(committed)} tokens."
            )

        emitted, emitted_logprobs, finish_reason, stop_reason = (
            self._trim_committed_tokens(
                session,
                committed,
                committed_logprobs,
            )
        )
        session.accepted_prefix_token_ids.extend(emitted)
        accept_structured_output_tokens(session.structured_output_session, emitted)
        session.version += len(emitted)
        return emitted, emitted_logprobs, finish_reason, stop_reason, needs_resync

    def _trim_committed_tokens(
        self,
        session: EdgeSessionState,
        committed: list[int],
        committed_logprobs: list[PackedLogprobs],
    ) -> tuple[
        list[int],
        LogprobsLists | None,
        FinishReason | None,
        int | str | None,
    ]:
        emitted: list[int] = []
        emitted_logprobs: list[PackedLogprobs] = []
        for token_id in committed:
            if committed_logprobs:
                emitted_logprobs.append(committed_logprobs[len(emitted)])
            emitted.append(token_id)
            output_len_after = (
                len(session.accepted_prefix_token_ids) - session.prompt_len
            ) + len(emitted)
            if is_terminal_token(token_id, session.sampling, output_len_after):
                stop_reason = (
                    token_id if token_id in session.sampling.stop_token_ids else None
                )
                return (
                    emitted,
                    build_logprobs_lists(emitted_logprobs),
                    FinishReason.STOP,
                    stop_reason,
                )
            if (
                session.sampling.max_tokens is not None
                and output_len_after >= session.sampling.max_tokens
            ):
                return (
                    emitted,
                    build_logprobs_lists(emitted_logprobs),
                    FinishReason.LENGTH,
                    None,
                )
        return emitted, build_logprobs_lists(emitted_logprobs), None, None

    def _maybe_finish_without_new_tokens(
        self, session: EdgeSessionState
    ) -> FinishReason | None:
        if session.sampling.max_tokens is None:
            return None
        output_len = len(session.accepted_prefix_token_ids) - session.prompt_len
        if output_len >= session.sampling.max_tokens:
            return FinishReason.LENGTH
        return None
