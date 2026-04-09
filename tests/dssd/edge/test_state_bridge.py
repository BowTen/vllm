from types import SimpleNamespace

import torch
from vllm.sampling_params import SamplingParams

from vllm.dssd.edge.state_bridge import EdgeStateBridge
from vllm.dssd.edge.types import EdgeSession


class FakeTensorField:
    def __init__(self, values):
        self.values = list(values)

    def stage_write_elem(self, index, value):
        self.values[index] = value


class FakeTokenStore:
    def __init__(self):
        self.rows = {0: [10, 11, 0, 0, 0, 0]}

    def stage_write(self, index, offset, values):
        row = self.rows[index]
        row[offset:offset + len(values)] = list(values)


class FakeReqStates:
    def __init__(self):
        self.req_id_to_index = {"req-1": 0}
        self.last_sampled_tokens = torch.zeros((1, 1), dtype=torch.int64)
        self.all_token_ids = FakeTokenStore()
        self.total_len = FakeTensorField([2])
        self.num_computed_tokens = FakeTensorField([2])
        self.apply_calls = 0

    def apply_staged_writes(self):
        self.apply_calls += 1


def make_session() -> EdgeSession:
    return EdgeSession(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
        block_ids=([1],),
        prompt_len=2,
        num_computed_tokens=2,
        total_len=2,
        token_ids=[10, 11],
    )


def test_bootstrap_and_external_commit_write_req_state() -> None:
    bridge = EdgeStateBridge()
    session = make_session()
    model_runner = SimpleNamespace(req_states=FakeReqStates())

    bridge.bootstrap_first_token(session, 20, model_runner)
    bridge.inject_external_token(session, 21, model_runner)

    assert session.token_ids == [10, 11, 20, 21]
    assert session.total_len == 4
    assert session.num_computed_tokens == 2
    assert int(model_runner.req_states.last_sampled_tokens[0, 0]) == 21
    assert model_runner.req_states.total_len.values == [4]


def test_rollback_rewinds_session_and_req_state() -> None:
    bridge = EdgeStateBridge()
    session = make_session()
    session.token_ids = [10, 11, 20, 30, 31]
    session.total_len = 5
    session.num_computed_tokens = 4
    model_runner = SimpleNamespace(req_states=FakeReqStates())

    bridge.rollback(session, 2, model_runner)

    assert session.token_ids == [10, 11, 20]
    assert session.total_len == 3
    assert session.num_computed_tokens == 3
    assert int(model_runner.req_states.last_sampled_tokens[0, 0]) == 20
    assert model_runner.req_states.total_len.values == [3]
