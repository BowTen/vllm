# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    OpenSessionRequest,
    OpenSessionResponse,
    ResyncSessionRequest,
    ResyncSessionResponse,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.runtime import (
    BaseCausalLMRuntime,
    IncrementalRuntimeState,
    resolve_runtime_device,
    resolve_torch_dtype,
    serialize_probs,
)

logger = init_logger(__name__)


@dataclass
class CloudSession:
    session_id: str
    prompt_len: int
    accepted_prefix_token_ids: list[int]
    version: int
    sampling: Any
    generator: torch.Generator
    runtime_state: IncrementalRuntimeState | None = None


class TargetVerificationRunner(BaseCausalLMRuntime):
    def __init__(
        self,
        model_name: str,
        device: str | None,
        dtype: str | torch.dtype,
        trust_remote_code: bool,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=resolve_runtime_device(device),
            dtype=resolve_torch_dtype(dtype),
            trust_remote_code=trust_remote_code,
        )
        self._lock = asyncio.Lock()
        self._sessions: dict[str, CloudSession] = {}

    async def open_session(
        self,
        request: OpenSessionRequest,
    ) -> OpenSessionResponse:
        async with self._lock:
            if request.session_id in self._sessions:
                raise ValueError(f"Session {request.session_id} already exists.")
            seed = request.sampling_metadata.seed
            if seed is None:
                seed = torch.seed()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + 1)
            self._sessions[request.session_id] = CloudSession(
                session_id=request.session_id,
                prompt_len=len(request.prompt_token_ids),
                accepted_prefix_token_ids=list(request.prompt_token_ids),
                version=request.initial_version,
                sampling=request.sampling_metadata,
                generator=generator,
            )
            return OpenSessionResponse(
                session_id=request.session_id,
                session_version=request.initial_version,
                vocab_size=self.vocab_size,
            )

    async def close_session(self, request: CloseSessionRequest) -> None:
        async with self._lock:
            self._sessions.pop(request.session_id, None)

    async def resync_session(
        self,
        request: ResyncSessionRequest,
    ) -> ResyncSessionResponse:
        async with self._lock:
            return await asyncio.to_thread(self._resync_session_sync, request)

    async def verify_proposal(
        self,
        proposal: DraftProposal,
    ) -> VerificationResult:
        async with self._lock:
            return await asyncio.to_thread(self._verify_proposal_sync, proposal)

    def _verify_proposal_sync(self, proposal: DraftProposal) -> VerificationResult:
        session = self._get_session(proposal.session_id)
        if proposal.base_version != session.version:
            raise ValueError(
                f"Session {proposal.session_id} version mismatch: "
                f"got {proposal.base_version}, expected {session.version}."
            )
        if proposal.accepted_prefix_len != len(session.accepted_prefix_token_ids):
            raise ValueError(
                f"Session {proposal.session_id} prefix mismatch: "
                f"got {proposal.accepted_prefix_len}, "
                f"expected {len(session.accepted_prefix_token_ids)}."
            )

        accepted_prefix = list(session.accepted_prefix_token_ids)
        runtime_state = self.sync_runtime_state(
            session.runtime_state,
            accepted_prefix,
        )
        session.runtime_state = runtime_state
        temp_state = self.clone_runtime_state(runtime_state)
        accepted_token_ids: list[int] = []
        for reject_pos, draft_token_id in enumerate(proposal.draft_token_ids):
            sample = self.sample_next_token_from_state(
                temp_state,
                prompt_len=session.prompt_len,
                sampling=session.sampling,
                generator=session.generator,
            )
            if sample.token_id != draft_token_id:
                prefix = accepted_prefix + accepted_token_ids
                session.accepted_prefix_token_ids = prefix
                session.runtime_state = temp_state
                session.version = proposal.base_version + len(accepted_token_ids)
                return VerificationResult(
                    session_id=proposal.session_id,
                    proposal_id=proposal.proposal_id,
                    base_version=proposal.base_version,
                    accepted_len=len(accepted_token_ids),
                    accepted_token_ids=accepted_token_ids,
                    reject_pos=reject_pos,
                    target_probs_at_reject_pos=serialize_probs(sample.probs),
                    verifier_version=session.version,
                )
            accepted_token_ids.append(draft_token_id)
            self.advance_runtime_state(temp_state, draft_token_id)

        session.accepted_prefix_token_ids = accepted_prefix + accepted_token_ids
        session.runtime_state = temp_state
        session.version = proposal.base_version + len(accepted_token_ids)

        bonus_token_id = None
        if accepted_token_ids and not proposal.draft_stopped:
            sample = self.sample_next_token_from_state(
                temp_state,
                prompt_len=session.prompt_len,
                sampling=session.sampling,
                generator=session.generator,
            )
            bonus_token_id = sample.token_id
            session.accepted_prefix_token_ids.append(bonus_token_id)
            self.advance_runtime_state(temp_state, bonus_token_id)
            session.runtime_state = temp_state
            session.version += 1

        return VerificationResult(
            session_id=proposal.session_id,
            proposal_id=proposal.proposal_id,
            base_version=proposal.base_version,
            accepted_len=len(accepted_token_ids),
            accepted_token_ids=accepted_token_ids,
            verifier_version=session.version,
            bonus_token_id=bonus_token_id,
        )

    def _resync_session_sync(
        self,
        request: ResyncSessionRequest,
    ) -> ResyncSessionResponse:
        session = self._get_session(request.session_id)
        session.accepted_prefix_token_ids = list(request.accepted_prefix_token_ids)
        session.runtime_state = self.sync_runtime_state(
            session.runtime_state,
            session.accepted_prefix_token_ids,
        )
        session.version = request.edge_version
        return ResyncSessionResponse(
            session_id=request.session_id,
            session_version=session.version,
        )

    def _get_session(self, session_id: str) -> CloudSession:
        if session_id not in self._sessions:
            raise ValueError(f"Unknown verifier session {session_id}.")
        return self._sessions[session_id]
