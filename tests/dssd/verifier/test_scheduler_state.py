# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.dssd.verifier.scheduler import VerifierSchedulerAdapter
from vllm.dssd.verifier.state_bridge import VerifierStateBridge
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


def _build_model_runner() -> SimpleNamespace:
    return SimpleNamespace(
        req_states=SimpleNamespace(
            req_id_to_index={"req-1": 0},
            last_sampled_tokens=torch.zeros((1, 1), dtype=torch.int64),
            draft_tokens=torch.zeros((1, 4), dtype=torch.int64),
        ),
        commit_input_token=mock.Mock(),
    )


def test_allocate_blocks_uses_kv_cache_manager() -> None:
    kv_cache_manager = mock.MagicMock()
    blocks = mock.MagicMock()
    blocks.get_block_ids.return_value = ([7],)
    kv_cache_manager.allocate_slots.return_value = blocks
    adapter = VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager)

    block_ids = adapter.allocate_blocks(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )

    assert block_ids == ([7],)
    kv_cache_manager.allocate_slots.assert_called_once()


def test_scheduler_tracks_new_blocks_for_open_and_verify_steps() -> None:
    prompt_blocks = mock.MagicMock()
    prompt_blocks.get_block_ids.return_value = ([7],)
    prompt_blocks.get_unhashed_block_ids_all_groups.return_value = [[7]]
    round_blocks = mock.MagicMock()
    round_blocks.get_block_ids.return_value = ([8],)
    round_blocks.get_unhashed_block_ids_all_groups.return_value = [[8]]
    kv_cache_manager = mock.MagicMock()
    kv_cache_manager.allocate_slots.side_effect = [prompt_blocks, round_blocks]
    kv_cache_manager.take_new_block_ids.side_effect = [[7], [8]]
    adapter = VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager)

    session = _build_session()
    session.block_ids = adapter.allocate_blocks(
        req_id=session.req_id,
        prompt_token_ids=session.prompt_token_ids,
        sampling_params=session.sampling_params,
    )

    open_step = adapter.build_open_session_step(session)

    assert open_step.new_block_ids_to_zero == [7]

    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )
    verify_step = adapter.build_verify_step(session, request)

    assert verify_step.scheduled_cached_reqs.new_block_ids == [([8],)]
    assert verify_step.new_block_ids_to_zero == [8]
    verify_allocate = kv_cache_manager.allocate_slots.call_args_list[1]
    assert verify_allocate.kwargs["num_new_tokens"] == 3
    assert kv_cache_manager.take_new_block_ids.call_count == 2


def test_scheduler_build_steps_and_free_blocks_follow_session_shape() -> None:
    session = _build_session()
    adapter = VerifierSchedulerAdapter()

    open_step = adapter.build_open_session_step(session)

    assert open_step.total_num_scheduled_tokens == 3
    assert open_step.num_scheduled_tokens == {"req-1": 3}
    assert open_step.scheduled_new_reqs[0].prompt_token_ids == [1, 2, 3]
    assert open_step.scheduled_new_reqs[0].prefill_token_ids == [1, 2, 3]

    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )
    verify_step = adapter.build_verify_step(session, request)

    assert verify_step.total_num_scheduled_tokens == 3
    assert verify_step.num_scheduled_tokens == {"req-1": 3}
    assert verify_step.scheduled_spec_decode_tokens == {"req-1": [11, 12]}
    assert verify_step.scheduled_cached_reqs.req_ids == ["req-1"]
    assert verify_step.scheduled_cached_reqs.num_computed_tokens == [3]
    assert verify_step.scheduled_cached_reqs.num_output_tokens == [0]

    empty_draft_request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[],
        draft_q_values=[],
    )
    empty_draft_step = adapter.build_verify_step(session, empty_draft_request)
    assert empty_draft_step.total_num_scheduled_tokens == 1
    assert empty_draft_step.scheduled_spec_decode_tokens == {}

    mismatched_request = VerifierRoundRequest(
        req_id="req-2",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )
    with pytest.raises(ValueError, match="session and request req_id"):
        adapter.build_verify_step(session, mismatched_request)

    close_step = adapter.build_close_step("req-1")
    assert close_step.finished_req_ids == {"req-1"}

    session.token_ids = [1, 2, 3, 9, 11]
    session.num_computed_tokens = 5
    kv_cache_manager = mock.MagicMock()
    adapter = VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager)

    adapter.free_blocks(session)

    free_request = kv_cache_manager.free.call_args.args[0]
    assert free_request.request_id == "req-1"
    assert list(free_request.output_token_ids) == [9, 11]
    assert free_request.num_computed_tokens == 5


def test_finish_prefill_without_commit_keeps_prompt_only() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()

    bridge.finish_prefill_without_commit(session)

    assert session.num_computed_tokens == session.prompt_len
    assert session.total_len == session.prompt_len
    assert session.token_ids == [1, 2, 3]


def test_prepare_round_and_commit_committed_token_update_state() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    model_runner = _build_model_runner()
    mismatched_request = VerifierRoundRequest(
        req_id="req-2",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )
    with pytest.raises(ValueError, match="session and round req_id"):
        bridge.prepare_round(session, mismatched_request, model_runner, gamma=4)

    invalid_request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )

    with pytest.raises(ValueError, match="超过固定 gamma"):
        bridge.prepare_round(session, invalid_request, model_runner, gamma=2)

    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )

    bridge.prepare_round(session, request, model_runner, gamma=4)

    with pytest.raises(ValueError, match="already in progress"):
        bridge.prepare_round(session, request, model_runner, gamma=4)

    assert int(model_runner.req_states.last_sampled_tokens[0, 0].item()) == 9
    assert model_runner.req_states.draft_tokens[0].tolist() == [11, 12, 0, 0]
    round_state = bridge._round_state(session)
    assert round_state.committed_token_id == 9
    assert round_state.draft_token_ids == [11, 12]
    assert round_state.draft_q_values == [0.2, 0.3]

    bridge.set_round_q_values(session, [0.7, 0.8])
    assert round_state.draft_q_values == [0.7, 0.8]

    with pytest.raises(ValueError, match="prepared committed token"):
        bridge.commit_committed_token_before_postprocess(session, 10, model_runner)

    bridge.commit_committed_token_before_postprocess(session, 9, model_runner)

    model_runner.commit_input_token.assert_called_once_with(0, 9)
    assert session.token_ids == [1, 2, 3, 9]
    assert session.total_len == 4

    with pytest.raises(ValueError, match="already committed"):
        bridge.commit_committed_token_before_postprocess(session, 9, model_runner)


def test_set_round_result_only_commits_accepted_prefix() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    model_runner = _build_model_runner()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )
    bridge.prepare_round(session, request, model_runner, gamma=4)
    bridge.commit_committed_token_before_postprocess(session, 9, model_runner)
    result = VerifierRoundResult(
        req_id="req-1",
        accepted_len=2,
        rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
    )

    bridge.set_round_result(session, result)

    assert session.token_ids == [1, 2, 3, 9, 11, 12]
    assert session.total_len == 6
    assert session.num_computed_tokens == 6
    round_state = bridge._round_state(session)
    assert round_state.committed_token_id is None
    assert round_state.committed_token_committed is False
    assert round_state.draft_token_ids == []
    assert round_state.draft_q_values == []
    assert not hasattr(round_state, "last_result")


def test_set_round_result_requires_committed_token_to_be_committed_first() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    model_runner = _build_model_runner()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )
    bridge.prepare_round(session, request, model_runner, gamma=4)

    with pytest.raises(ValueError, match="committed token must be committed"):
        bridge.set_round_result(
            session,
            VerifierRoundResult(
                req_id="req-1",
                accepted_len=1,
                rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
            ),
        )


def test_round_state_helpers_reset_and_remove_tracking() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    round_state = bridge._round_state(session)
    round_state.committed_token_id = 9
    round_state.committed_token_committed = True
    round_state.draft_token_ids = [11, 12]
    round_state.draft_q_values = [0.2, 0.3]

    bridge.clear_round_state(session)

    assert round_state.committed_token_id is None
    assert round_state.committed_token_committed is False
    assert round_state.draft_token_ids == []
    assert round_state.draft_q_values == []
    assert not hasattr(round_state, "last_result")

    bridge.remove_round_state(session)

    assert session.req_id not in bridge._round_states


def test_set_round_result_rejects_invalid_round_metadata() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    round_state = bridge._round_state(session)
    round_state.draft_token_ids = [11, 12, 13]
    round_state.committed_token_committed = True

    with pytest.raises(ValueError, match="session and round req_id"):
        bridge.set_round_result(
            session,
            VerifierRoundResult(
                req_id="req-2",
                accepted_len=1,
                rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
            ),
        )

    with pytest.raises(ValueError, match="accepted_len"):
        bridge.set_round_result(
            session,
            VerifierRoundResult(
                req_id="req-1",
                accepted_len=4,
                rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
            ),
        )

    with pytest.raises(ValueError, match="accepted_len"):
        bridge.set_round_result(
            session,
            VerifierRoundResult(
                req_id="req-1",
                accepted_len=-1,
                rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
            ),
        )


def test_set_round_result_rejects_all_accept_payload_for_partial_accept() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    model_runner = _build_model_runner()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )
    bridge.prepare_round(session, request, model_runner, gamma=4)
    bridge.commit_committed_token_before_postprocess(session, 9, model_runner)

    with pytest.raises(ValueError, match="rejected payload"):
        bridge.set_round_result(
            session,
            VerifierRoundResult(
                req_id="req-1",
                accepted_len=1,
                bonus_token_id=17,
            ),
        )


def test_set_round_result_rejects_reject_payload_for_full_accept() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    model_runner = _build_model_runner()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )
    bridge.prepare_round(session, request, model_runner, gamma=4)
    bridge.commit_committed_token_before_postprocess(session, 9, model_runner)

    with pytest.raises(ValueError, match="bonus payload"):
        bridge.set_round_result(
            session,
            VerifierRoundResult(
                req_id="req-1",
                accepted_len=2,
                rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
            ),
        )
