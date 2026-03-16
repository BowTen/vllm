# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    OpenSessionRequest,
    OpenSessionResponse,
    ResyncSessionRequest,
    ResyncSessionResponse,
    SamplingMetadata,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.verification_core import VerificationCore


class FakeVerificationRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | str]] = []

    async def open_session(
        self, request: OpenSessionRequest
    ) -> OpenSessionResponse:
        self.calls.append(("open", request.session_id))
        return OpenSessionResponse(
            session_id=request.session_id,
            session_version=request.initial_version,
            vocab_size=32,
        )

    async def close_session(self, request: CloseSessionRequest) -> None:
        self.calls.append(("close", request.session_id))

    async def resync_session(
        self, request: ResyncSessionRequest
    ) -> ResyncSessionResponse:
        self.calls.append(("resync", request.edge_version))
        return ResyncSessionResponse(
            session_id=request.session_id,
            session_version=request.edge_version,
        )

    async def verify_proposal(
        self, proposal: DraftProposal
    ) -> VerificationResult:
        self.calls.append(("verify", proposal.proposal_id))
        await asyncio.sleep(0.01)
        return VerificationResult(
            session_id=proposal.session_id,
            proposal_id=proposal.proposal_id,
            base_version=proposal.base_version,
            accepted_len=len(proposal.draft_token_ids),
            accepted_token_ids=list(proposal.draft_token_ids),
            verifier_version=proposal.base_version + len(proposal.draft_token_ids),
        )


class BatchingFakeVerificationRunner(FakeVerificationRunner):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[list[int]] = []

    async def verify_proposal(
        self, proposal: DraftProposal
    ) -> VerificationResult:
        raise AssertionError(
            f"verify_proposal should not be used for batched proposal {proposal.proposal_id}."
        )

    async def verify_proposals_batch(
        self, proposals: list[DraftProposal]
    ) -> list[VerificationResult]:
        self.batch_calls.append([proposal.proposal_id for proposal in proposals])
        await asyncio.sleep(0.01)
        return [
            VerificationResult(
                session_id=proposal.session_id,
                proposal_id=proposal.proposal_id,
                base_version=proposal.base_version,
                accepted_len=len(proposal.draft_token_ids),
                accepted_token_ids=list(proposal.draft_token_ids),
                verifier_version=proposal.base_version
                + len(proposal.draft_token_ids),
            )
            for proposal in proposals
        ]


class SlowVerificationRunner(FakeVerificationRunner):
    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    async def verify_proposal(
        self, proposal: DraftProposal
    ) -> VerificationResult:
        self.calls.append(("verify", proposal.proposal_id))
        await asyncio.sleep(self.delay_s)
        return await super().verify_proposal(proposal)


@pytest.mark.asyncio
async def test_verification_core_tracks_sessions_and_serializes_proposals():
    runner = FakeVerificationRunner()
    core = VerificationCore(runner)

    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    session_id = "session-1"
    response = await core.open_session(
        OpenSessionRequest(
            session_id=session_id,
            prompt_token_ids=[1, 2, 3],
            sampling_metadata=sampling,
            initial_version=0,
        )
    )
    assert response.session_version == 0
    assert core.registry.has(session_id)

    proposal0 = DraftProposal(
        session_id=session_id,
        proposal_id=0,
        base_version=0,
        accepted_prefix_len=3,
        draft_token_ids=[10],
        draft_token_probs=[1.0],
    )
    proposal1 = DraftProposal(
        session_id=session_id,
        proposal_id=1,
        base_version=1,
        accepted_prefix_len=4,
        draft_token_ids=[11],
        draft_token_probs=[1.0],
    )
    result0, result1 = await asyncio.gather(
        core.verify_proposal(proposal0),
        core.verify_proposal(proposal1),
    )
    assert result0.accepted_token_ids == [10]
    assert result1.accepted_token_ids == [11]
    assert [call for call in runner.calls if call[0] == "verify"] == [
        ("verify", 0),
        ("verify", 1),
    ]
    assert core.registry.get(session_id).version == 2
    assert core.registry.get(session_id).prefix_len == 5

    resync_response = await core.resync_session(
        ResyncSessionRequest(
            session_id=session_id,
            accepted_prefix_token_ids=[1, 2, 3, 10, 11],
            edge_version=5,
        )
    )
    assert resync_response.session_version == 5
    assert core.registry.get(session_id).version == 5

    await core.close_session(CloseSessionRequest(session_id=session_id))
    assert not core.registry.has(session_id)
    await core.shutdown()


@pytest.mark.asyncio
async def test_verification_core_batches_ready_proposals_across_sessions():
    runner = BatchingFakeVerificationRunner()
    core = VerificationCore(
        runner,
        scheduler_max_batch_size=4,
        scheduler_batch_wait_ms=5.0,
    )

    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    for session_id in ("session-a", "session-b"):
        await core.open_session(
            OpenSessionRequest(
                session_id=session_id,
                prompt_token_ids=[1, 2, 3],
                sampling_metadata=sampling,
                initial_version=0,
            )
        )

    proposal_a = DraftProposal(
        session_id="session-a",
        proposal_id=0,
        base_version=0,
        accepted_prefix_len=3,
        draft_token_ids=[10],
        draft_token_probs=[1.0],
    )
    proposal_b = DraftProposal(
        session_id="session-b",
        proposal_id=1,
        base_version=0,
        accepted_prefix_len=3,
        draft_token_ids=[11],
        draft_token_probs=[1.0],
    )
    result_a, result_b = await asyncio.gather(
        core.verify_proposal(proposal_a),
        core.verify_proposal(proposal_b),
    )

    assert result_a.accepted_token_ids == [10]
    assert result_b.accepted_token_ids == [11]
    assert runner.batch_calls == [[0, 1]]
    assert core.registry.get("session-a").version == 1
    assert core.registry.get("session-b").version == 1

    await core.close_session(CloseSessionRequest(session_id="session-a"))
    await core.close_session(CloseSessionRequest(session_id="session-b"))
    await core.shutdown()


@pytest.mark.asyncio
async def test_verification_core_keeps_one_proposal_per_session_per_batch():
    runner = BatchingFakeVerificationRunner()
    core = VerificationCore(
        runner,
        scheduler_max_batch_size=4,
        scheduler_batch_wait_ms=5.0,
    )

    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    for session_id in ("session-a", "session-b"):
        await core.open_session(
            OpenSessionRequest(
                session_id=session_id,
                prompt_token_ids=[1, 2, 3],
                sampling_metadata=sampling,
                initial_version=0,
            )
        )

    proposals = [
        DraftProposal(
            session_id="session-a",
            proposal_id=0,
            base_version=0,
            accepted_prefix_len=3,
            draft_token_ids=[10],
            draft_token_probs=[1.0],
        ),
        DraftProposal(
            session_id="session-a",
            proposal_id=1,
            base_version=1,
            accepted_prefix_len=4,
            draft_token_ids=[11],
            draft_token_probs=[1.0],
        ),
        DraftProposal(
            session_id="session-b",
            proposal_id=2,
            base_version=0,
            accepted_prefix_len=3,
            draft_token_ids=[12],
            draft_token_probs=[1.0],
        ),
    ]
    results = await asyncio.gather(*(core.verify_proposal(proposal) for proposal in proposals))

    assert [result.accepted_token_ids for result in results] == [[10], [11], [12]]
    assert runner.batch_calls == [[0, 2], [1]]
    await core.shutdown()


@pytest.mark.asyncio
async def test_verification_core_splits_batches_by_token_budget():
    runner = BatchingFakeVerificationRunner()
    core = VerificationCore(
        runner,
        scheduler_max_batch_size=4,
        scheduler_batch_wait_ms=5.0,
        scheduler_max_batch_tokens=2,
    )

    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    for session_id in ("session-a", "session-b"):
        await core.open_session(
            OpenSessionRequest(
                session_id=session_id,
                prompt_token_ids=[1, 2, 3],
                sampling_metadata=sampling,
                initial_version=0,
            )
        )

    proposal_a = DraftProposal(
        session_id="session-a",
        proposal_id=0,
        base_version=0,
        accepted_prefix_len=3,
        draft_token_ids=[10, 11],
        draft_token_probs=[1.0, 1.0],
    )
    proposal_b = DraftProposal(
        session_id="session-b",
        proposal_id=1,
        base_version=0,
        accepted_prefix_len=3,
        draft_token_ids=[12],
        draft_token_probs=[1.0],
    )
    result_a, result_b = await asyncio.gather(
        core.verify_proposal(proposal_a),
        core.verify_proposal(proposal_b),
    )

    assert result_a.accepted_token_ids == [10, 11]
    assert result_b.accepted_token_ids == [12]
    assert runner.batch_calls == [[0], [1]]
    await core.shutdown()


@pytest.mark.asyncio
async def test_verification_core_times_out_stale_queued_proposals():
    runner = SlowVerificationRunner(delay_s=0.05)
    core = VerificationCore(
        runner,
        scheduler_max_batch_size=1,
        scheduler_batch_wait_ms=0.0,
        scheduler_queue_timeout_ms=5.0,
    )

    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    await core.open_session(
        OpenSessionRequest(
            session_id="session-a",
            prompt_token_ids=[1, 2, 3],
            sampling_metadata=sampling,
            initial_version=0,
        )
    )

    proposal_0 = DraftProposal(
        session_id="session-a",
        proposal_id=0,
        base_version=0,
        accepted_prefix_len=3,
        draft_token_ids=[10],
        draft_token_probs=[1.0],
    )
    proposal_1 = DraftProposal(
        session_id="session-a",
        proposal_id=1,
        base_version=1,
        accepted_prefix_len=4,
        draft_token_ids=[11],
        draft_token_probs=[1.0],
    )
    task_0 = asyncio.create_task(core.verify_proposal(proposal_0))
    await asyncio.sleep(0.001)
    task_1 = asyncio.create_task(core.verify_proposal(proposal_1))

    result_0 = await task_0
    assert result_0.accepted_token_ids == [10]
    with pytest.raises(TimeoutError, match="verifier queue"):
        await task_1

    assert core.registry.get("session-a").active_proposals == 0
    await core.shutdown()


@pytest.mark.asyncio
async def test_verification_core_reaps_idle_sessions():
    runner = FakeVerificationRunner()
    core = VerificationCore(
        runner,
        session_idle_timeout_s=0.05,
    )

    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    await core.open_session(
        OpenSessionRequest(
            session_id="session-reap",
            prompt_token_ids=[1, 2, 3],
            sampling_metadata=sampling,
            initial_version=0,
        )
    )

    deadline = asyncio.get_running_loop().time() + 1.0
    while core.registry.has("session-reap") and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)

    assert not core.registry.has("session-reap")
    assert ("close", "session-reap") in runner.calls
    await core.shutdown()
