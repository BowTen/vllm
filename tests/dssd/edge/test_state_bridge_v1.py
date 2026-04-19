from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.edge.state_bridge_v1 import EdgeStateBridgeV1
from vllm.dssd.edge.types import EdgeSession
from vllm.sampling_params import SamplingParams


@dataclass
class FakeReqState:
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0


class FakeInputBatch:
    def __init__(self) -> None:
        self.req_id_to_index = {"req-1": 0}
        self.req_output_token_ids = [[]]
        self.token_ids_cpu = torch.zeros((1, 8), dtype=torch.int32).numpy()
        self.is_token_ids = torch.zeros((1, 8), dtype=torch.bool).numpy()
        self.num_tokens_no_spec = torch.zeros((1,), dtype=torch.int32).numpy()
        self.num_computed_tokens_cpu = torch.zeros((1,), dtype=torch.int32).numpy()
        self.prev_sampled_token_ids = object()
        self.prev_req_id_to_index = {"req-1": 0}


def make_runner():
    return SimpleNamespace(
        requests={"req-1": FakeReqState()},
        input_batch=FakeInputBatch(),
    )


def make_session() -> EdgeSession:
    return EdgeSession(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
        block_ids=([1],),
        prompt_len=2,
        total_len=2,
        token_ids=[10, 11],
    )


def test_inject_external_token_updates_v1_cached_request_and_input_batch() -> None:
    bridge = EdgeStateBridgeV1()
    runner = make_runner()
    session = make_session()

    bridge.inject_external_token(session, 20, runner)

    req_state = runner.requests["req-1"]
    assert req_state.output_token_ids == [20]
    assert req_state.num_computed_tokens == 2
    assert runner.input_batch.req_output_token_ids[0] is req_state.output_token_ids
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 3
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 2
    assert runner.input_batch.prev_sampled_token_ids is None
    assert runner.input_batch.prev_req_id_to_index is None


def test_rollback_rewinds_v1_cached_request_and_input_batch() -> None:
    bridge = EdgeStateBridgeV1()
    runner = make_runner()
    session = make_session()
    bridge.inject_external_token(session, 20, runner)
    bridge.commit_token(session, 21, runner)

    bridge.rollback(session, 1, runner)

    req_state = runner.requests["req-1"]
    assert req_state.output_token_ids == [20]
    assert req_state.num_computed_tokens == 3
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 3
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 3


def test_rollback_zero_is_a_true_no_op() -> None:
    bridge = EdgeStateBridgeV1()
    runner = make_runner()
    session = make_session()

    bridge.rollback(session, 0, runner)

    req_state = runner.requests["req-1"]
    assert session.token_ids == [10, 11]
    assert session.total_len == 2
    assert session.num_computed_tokens == 0
    assert req_state.output_token_ids == []
    assert req_state.num_computed_tokens == 0
    assert runner.input_batch.prev_sampled_token_ids is not None
    assert runner.input_batch.prev_req_id_to_index == {"req-1": 0}


def test_sync_clears_cached_sampled_tokens_without_req_batch_index() -> None:
    bridge = EdgeStateBridgeV1()
    runner = make_runner()
    runner.input_batch.req_id_to_index = {}
    session = make_session()

    bridge.inject_external_token(session, 20, runner)

    req_state = runner.requests["req-1"]
    assert req_state.output_token_ids == [20]
    assert req_state.num_computed_tokens == 2
    assert runner.input_batch.prev_sampled_token_ids is None
    assert runner.input_batch.prev_req_id_to_index is None


def test_rollback_keeps_num_computed_tokens_for_external_tail_token() -> None:
    bridge = EdgeStateBridgeV1()
    runner = make_runner()
    session = make_session()

    bridge.inject_external_token(session, 20, runner)
    bridge.commit_token(session, 21, runner)
    bridge.inject_external_token(session, 22, runner)

    bridge.rollback(session, 1, runner)

    req_state = runner.requests["req-1"]
    assert session.token_ids == [10, 11, 20, 21]
    assert session.total_len == 4
    assert session.num_computed_tokens == 3
    assert req_state.output_token_ids == [20, 21]
    assert req_state.num_computed_tokens == 3
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 4
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 3


def test_prepare_next_decode_rejects_multiple_pending_tokens() -> None:
    bridge = EdgeStateBridgeV1()
    runner = make_runner()
    session = make_session()

    bridge.inject_external_token(session, 20, runner)
    bridge.commit_token(session, 21, runner)
    bridge.inject_external_token(session, 22, runner)

    with pytest.raises(
        RuntimeError,
        match="exactly one trailing pending token",
    ):
        bridge.prepare_next_decode(session, 22, runner)

    req_state = runner.requests["req-1"]
    assert session.token_ids == [10, 11, 20, 21, 22]
    assert session.total_len == 5
    assert session.num_computed_tokens == 3
    assert req_state.output_token_ids == [20, 21, 22]
    assert req_state.num_computed_tokens == 3
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 5
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 3


def test_sampled_token_after_external_input_stays_pending_until_next_decode() -> None:
    bridge = EdgeStateBridgeV1()
    runner = make_runner()
    session = make_session()

    bridge.inject_external_token(session, 20, runner)
    bridge.commit_token(session, 21, runner)
    bridge.commit_token(session, 22, runner)
    bridge.mark_pending_token_computed(session, runner)
    bridge.inject_external_token(session, 23, runner)
    bridge.prepare_next_decode(session, 23, runner)
    bridge.commit_token(session, 24, runner)
    bridge.prepare_next_decode(session, 24, runner)

    bridge.rollback(session, 1, runner)

    req_state = runner.requests["req-1"]
    assert session.token_ids == [10, 11, 20, 21, 22, 23]
    assert session.total_len == 6
    assert session.num_computed_tokens == 6
    assert req_state.output_token_ids == [20, 21, 22, 23]
    assert req_state.num_computed_tokens == 6
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 6
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 6
