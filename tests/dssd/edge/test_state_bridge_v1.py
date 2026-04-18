from dataclasses import dataclass, field
from types import SimpleNamespace

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
    assert req_state.num_computed_tokens == 2
    assert int(runner.input_batch.num_tokens_no_spec[0]) == 3
    assert int(runner.input_batch.num_computed_tokens_cpu[0]) == 2
