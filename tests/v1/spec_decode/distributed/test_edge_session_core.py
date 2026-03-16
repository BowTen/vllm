# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.spec_decode.distributed.logprobs import pack_sample_logprobs
from vllm.v1.spec_decode.distributed.edge_session import EdgeSessionCore
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
    DraftProposalOutput,
    serialize_probs,
)


def make_request(
    request_id: str = "request-1",
    max_tokens: int = 5,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-external",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            seed=7,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


class FakeDraftRunner:
    vocab_size = 128

    def __init__(self) -> None:
        self.closed_sessions: list[str] = []

    async def propose(
        self,
        session_id: str,
        proposal_id: int,
        base_version: int,
        accepted_prefix_token_ids: list[int],
        prompt_len: int,
        sampling: Any,
        generator: torch.Generator,
        structured_output_session: Any | None = None,
    ) -> DraftProposalOutput:
        del prompt_len, sampling, generator, structured_output_session
        proposal_tokens = {
            0: ([11, 12], False),
            1: ([13], False),
            2: ([88], True),
        }
        draft_token_ids, draft_stopped = proposal_tokens[proposal_id]
        return DraftProposalOutput(
            proposal=DraftProposal(
                session_id=session_id,
                proposal_id=proposal_id,
                base_version=base_version,
                accepted_prefix_len=len(accepted_prefix_token_ids),
                draft_token_ids=draft_token_ids,
                draft_token_probs=[1.0] * len(draft_token_ids),
                draft_stopped=draft_stopped,
            ),
            stopped=draft_stopped,
        )

    def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

    def clear_sessions(self) -> None:
        self.closed_sessions.clear()


class FakeVerifierClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def open_session(
        self,
        request: OpenSessionRequest,
    ) -> OpenSessionResponse:
        self.calls.append(("open_session", request.session_id))
        prompt_logprobs = []
        if request.sampling_metadata.prompt_logprobs is not None:
            probs_2 = torch.zeros(FakeDraftRunner.vocab_size, dtype=torch.float32)
            probs_2[2] = 1.0
            probs_3 = torch.zeros(FakeDraftRunner.vocab_size, dtype=torch.float32)
            probs_3[3] = 1.0
            packed_2 = pack_sample_logprobs(
                probs_2,
                2,
                request.sampling_metadata.prompt_logprobs,
            )
            packed_3 = pack_sample_logprobs(
                probs_3,
                3,
                request.sampling_metadata.prompt_logprobs,
            )
            assert packed_2 is not None
            assert packed_3 is not None
            prompt_logprobs = [packed_2, packed_3]
        return OpenSessionResponse(
            session_id=request.session_id,
            session_version=request.initial_version,
            vocab_size=FakeDraftRunner.vocab_size,
            prompt_logprobs=prompt_logprobs,
        )

    async def verify_proposal(
        self,
        proposal: DraftProposal,
    ) -> VerificationResult:
        self.calls.append(
            ("verify_proposal", proposal.proposal_id, list(proposal.draft_token_ids))
        )
        if proposal.proposal_id == 0:
            probs_11 = torch.zeros(FakeDraftRunner.vocab_size, dtype=torch.float32)
            probs_11[11] = 1.0
            probs_12 = torch.zeros(FakeDraftRunner.vocab_size, dtype=torch.float32)
            probs_12[12] = 1.0
            probs_99 = torch.zeros(FakeDraftRunner.vocab_size, dtype=torch.float32)
            probs_99[99] = 1.0
            packed_11 = pack_sample_logprobs(probs_11, 11, 1)
            packed_12 = pack_sample_logprobs(probs_12, 12, 1)
            packed_99 = pack_sample_logprobs(probs_99, 99, 1)
            assert packed_11 is not None
            assert packed_12 is not None
            assert packed_99 is not None
            return VerificationResult(
                session_id=proposal.session_id,
                proposal_id=proposal.proposal_id,
                base_version=proposal.base_version,
                accepted_len=len(proposal.draft_token_ids),
                accepted_token_ids=list(proposal.draft_token_ids),
                verifier_version=proposal.base_version + len(proposal.draft_token_ids) + 1,
                bonus_token_id=99,
                accepted_logprobs=[packed_11, packed_12],
                bonus_logprobs=packed_99,
            )
        if proposal.proposal_id == 1:
            probs = torch.zeros(FakeDraftRunner.vocab_size, dtype=torch.float32)
            probs[77] = 1.0
            return VerificationResult(
                session_id=proposal.session_id,
                proposal_id=proposal.proposal_id,
                base_version=proposal.base_version,
                accepted_len=0,
                accepted_token_ids=[],
                verifier_version=proposal.base_version,
                reject_pos=0,
                target_probs_at_reject_pos=serialize_probs(probs),
            )
        probs_88 = torch.zeros(FakeDraftRunner.vocab_size, dtype=torch.float32)
        probs_88[88] = 1.0
        packed_88 = pack_sample_logprobs(probs_88, 88, 1)
        assert packed_88 is not None
        return VerificationResult(
            session_id=proposal.session_id,
            proposal_id=proposal.proposal_id,
            base_version=proposal.base_version,
            accepted_len=len(proposal.draft_token_ids),
            accepted_token_ids=list(proposal.draft_token_ids),
            verifier_version=proposal.base_version + len(proposal.draft_token_ids),
            accepted_logprobs=[packed_88],
        )

    async def resync_session(
        self,
        request: ResyncSessionRequest,
    ) -> ResyncSessionResponse:
        self.calls.append(
            ("resync_session", request.session_id, list(request.accepted_prefix_token_ids))
        )
        return ResyncSessionResponse(
            session_id=request.session_id,
            session_version=request.edge_version,
        )

    async def close_session(self, request: CloseSessionRequest) -> None:
        self.calls.append(("close_session", request.session_id))


@pytest.mark.asyncio
async def test_edge_session_core_streams_verified_steps_and_tracks_registry():
    request = make_request(max_tokens=5)
    draft_runner = FakeDraftRunner()
    verifier = FakeVerifierClient()
    core = EdgeSessionCore(draft_runner, verifier)

    session = core.create_session(request)
    assert core.registry.has(request.request_id)

    steps = [step async for step in core.run_request(request, session)]

    assert [step.new_token_ids for step in steps] == [[11, 12, 99], [77], [88]]
    assert steps[-1].finish_reason == FinishReason.LENGTH
    assert [call[0] for call in verifier.calls] == [
        "open_session",
        "verify_proposal",
        "verify_proposal",
        "resync_session",
        "verify_proposal",
    ]
    assert verifier.calls[3][2][-1] == 77
    assert session.accepted_prefix_token_ids[-5:] == [11, 12, 99, 77, 88]

    await core.close_session(request.request_id)
    assert not core.registry.has(request.request_id)
    assert draft_runner.closed_sessions == [request.request_id]
    assert verifier.calls[-1] == ("close_session", request.request_id)


@pytest.mark.asyncio
async def test_edge_session_core_abort_sessions_releases_state():
    request = make_request(request_id="request-2")
    draft_runner = FakeDraftRunner()
    verifier = FakeVerifierClient()
    core = EdgeSessionCore(draft_runner, verifier)

    core.create_session(request)
    await core.abort_sessions([request.request_id])

    assert not core.registry.has(request.request_id)
    assert draft_runner.closed_sessions == [request.request_id]
    assert verifier.calls == [("close_session", request.request_id)]


@pytest.mark.asyncio
async def test_edge_session_core_emits_logprobs_for_verified_tokens():
    request = make_request(request_id="request-logprobs", max_tokens=5, logprobs=1)
    draft_runner = FakeDraftRunner()
    verifier = FakeVerifierClient()
    core = EdgeSessionCore(draft_runner, verifier)

    session = core.create_session(request)
    steps = [step async for step in core.run_request(request, session)]

    assert [step.new_token_ids for step in steps] == [[11, 12, 99], [77], [88]]
    assert all(step.new_logprobs is not None for step in steps)
    assert steps[0].new_logprobs.logprob_token_ids.shape == (3, 2)
    assert steps[1].new_logprobs.logprob_token_ids.tolist() == [[77, 77]]
    assert steps[2].new_logprobs.logprob_token_ids.tolist() == [[88, 88]]


@pytest.mark.asyncio
async def test_edge_session_core_emits_prompt_logprobs_before_decode_steps():
    request = make_request(
        request_id="request-prompt-logprobs",
        max_tokens=5,
        prompt_logprobs=1,
    )
    draft_runner = FakeDraftRunner()
    verifier = FakeVerifierClient()
    core = EdgeSessionCore(draft_runner, verifier)

    session = core.create_session(request)
    steps = [step async for step in core.run_request(request, session)]

    assert steps[0].new_token_ids == []
    assert steps[0].new_prompt_logprobs_tensors is not None
    assert steps[0].new_prompt_logprobs_tensors.logprob_token_ids.tolist() == [
        [2, 2],
        [3, 3],
    ]
    assert [step.new_token_ids for step in steps[1:]] == [[11, 12, 99], [77], [88]]
