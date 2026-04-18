from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.dssd.verifier.state_bridge_v1 import VerifierStateBridgeV1
from vllm.dssd.verifier.types import (
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierSession,
)
from vllm.sampling_params import SamplingParams


def _build_session() -> VerifierSession:
    return VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        block_ids=([0],),
        prompt_len=3,
        token_ids=[1, 2, 3],
        num_computed_tokens=3,
        total_len=3,
    )


def _build_runner() -> SimpleNamespace:
    token_ids = np.zeros((1, 16), dtype=np.int32)
    token_ids[0, :3] = [1, 2, 3]
    is_token_ids = np.zeros((1, 16), dtype=bool)
    is_token_ids[0, :3] = True
    req_output_token_ids = [[]]
    return SimpleNamespace(
        requests={
            "req-1": SimpleNamespace(
                req_id="req-1",
                output_token_ids=req_output_token_ids[0],
                num_computed_tokens=3,
                block_ids=([0],),
            )
        },
        input_batch=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            token_ids_cpu=token_ids,
            is_token_ids=is_token_ids,
            num_tokens_no_spec=np.array([3], dtype=np.int32),
            num_computed_tokens_cpu=np.array([3], dtype=np.int32),
            req_output_token_ids=req_output_token_ids,
            spec_token_ids=[[]],
        ),
    )


def test_finish_prefill_without_commit_keeps_prompt_only() -> None:
    session = _build_session()
    runner = _build_runner()
    bridge = VerifierStateBridgeV1()

    bridge.finish_prefill_without_commit(session, runner)

    assert session.token_ids == [1, 2, 3]
    assert session.total_len == 3
    assert session.num_computed_tokens == 3
    assert runner.requests["req-1"].output_token_ids == []
    assert runner.input_batch.req_output_token_ids[0] is runner.requests[
        "req-1"
    ].output_token_ids
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 3
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 3


def test_begin_round_adds_committed_token_without_advancing_computed_len() -> None:
    session = _build_session()
    runner = _build_runner()
    bridge = VerifierStateBridgeV1()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )

    bridge.begin_round(session, request, runner, gamma=4)

    assert session.token_ids == [1, 2, 3, 9]
    assert session.total_len == 4
    assert session.num_computed_tokens == 3
    assert runner.requests["req-1"].output_token_ids == [9]
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 4
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 3
    assert runner.input_batch.token_ids_cpu[0, 3] == 9


def test_finish_round_only_appends_accepted_draft_prefix() -> None:
    session = _build_session()
    runner = _build_runner()
    bridge = VerifierStateBridgeV1()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )
    bridge.begin_round(session, request, runner, gamma=4)

    result = VerifierRoundResult(
        req_id="req-1",
        accepted_len=2,
        rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
    )
    bridge.finish_round(session, request, result, runner)

    assert session.token_ids == [1, 2, 3, 9, 11, 12]
    assert session.total_len == 6
    assert session.num_computed_tokens == 6
    assert runner.requests["req-1"].output_token_ids == [9, 11, 12]
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 6
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 6
    assert runner.input_batch.token_ids_cpu[0, 4:6].tolist() == [11, 12]


def test_round_updates_mark_new_tokens_as_token_ids() -> None:
    session = _build_session()
    runner = _build_runner()
    bridge = VerifierStateBridgeV1()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )

    bridge.begin_round(session, request, runner, gamma=4)

    assert runner.input_batch.is_token_ids[0, :4].tolist() == [
        True,
        True,
        True,
        True,
    ]

    result = VerifierRoundResult(
        req_id="req-1",
        accepted_len=2,
        rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
    )
    bridge.finish_round(session, request, result, runner)

    assert runner.input_batch.is_token_ids[0, :6].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_inject_local_token_appends_and_advances_computed_len() -> None:
    session = _build_session()
    runner = _build_runner()
    bridge = VerifierStateBridgeV1()

    bridge.inject_local_token(
        session,
        42,
        runner,
        computed_delta=1,
    )

    assert session.token_ids == [1, 2, 3, 42]
    assert session.total_len == 4
    assert session.num_computed_tokens == 4
    assert runner.requests["req-1"].output_token_ids == [42]
    assert runner.requests["req-1"].num_computed_tokens == 4
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 4
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 4
    assert runner.input_batch.token_ids_cpu[0, :4].tolist() == [1, 2, 3, 42]
    assert runner.input_batch.is_token_ids[0, :4].tolist() == [
        True,
        True,
        True,
        True,
    ]


def test_commit_local_token_appends_after_decode() -> None:
    session = _build_session()
    runner = _build_runner()
    bridge = VerifierStateBridgeV1()
    bridge.inject_local_token(session, 42, runner, computed_delta=1)

    bridge.commit_local_token(session, 43, runner)

    assert session.token_ids == [1, 2, 3, 42, 43]
    assert session.total_len == 5
    assert session.num_computed_tokens == 5
    assert runner.requests["req-1"].output_token_ids == [42, 43]
    assert runner.requests["req-1"].num_computed_tokens == 5
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 5
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 5
    assert runner.input_batch.token_ids_cpu[0, :5].tolist() == [
        1,
        2,
        3,
        42,
        43,
    ]


@pytest.mark.parametrize(
    "result",
    [
        VerifierRoundResult(
            req_id="req-1",
            accepted_len=3,
            rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
        ),
        VerifierRoundResult(
            req_id="req-1",
            accepted_len=2,
            bonus_token_id=99,
        ),
        VerifierRoundResult(
            req_id="req-1",
            accepted_len=4,
            rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
        ),
    ],
)
def test_finish_round_rejects_invalid_result_invariants(
    result: VerifierRoundResult,
) -> None:
    session = _build_session()
    runner = _build_runner()
    bridge = VerifierStateBridgeV1()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )
    bridge.begin_round(session, request, runner, gamma=4)

    with pytest.raises(ValueError):
        bridge.finish_round(session, request, result, runner)

    assert session.token_ids == [1, 2, 3, 9]
    assert session.total_len == 4
    assert session.num_computed_tokens == 3
    assert runner.requests["req-1"].output_token_ids == [9]
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 4
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 3
