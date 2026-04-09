from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm.dssd.edge.state_bridge import EdgeStateBridge
from vllm.dssd.edge.types import EdgeSession


class FakeTensorField:
    def __init__(self, values):
        self.values = list(values)
        self.staged = {}

    def stage_write_elem(self, index, value):
        self.staged[index] = value

    def apply_staged_writes(self):
        for index, value in self.staged.items():
            self.values[index] = value
        self.staged.clear()


class FakeTokenStore:
    def __init__(self):
        self.rows = {0: [10, 11, 0, 0, 0, 0]}
        self.staged = []

    def stage_write(self, index, offset, values):
        self.staged.append((index, offset, list(values)))

    def apply_staged_writes(self):
        for index, offset, values in self.staged:
            row = self.rows[index]
            row[offset:offset + len(values)] = values
        self.staged.clear()


class FakeReqStates:
    def __init__(self):
        self.req_id_to_index = {"req-1": 0}
        self.last_sampled_tokens = torch.zeros((1, 1), dtype=torch.int64)
        self.all_token_ids = FakeTokenStore()
        self.total_len = FakeTensorField([2])
        self.num_computed_tokens = FakeTensorField([2])
        self.apply_calls = 0

    def apply_staged_writes(self):
        self.all_token_ids.apply_staged_writes()
        self.total_len.apply_staged_writes()
        self.num_computed_tokens.apply_staged_writes()
        self.apply_calls += 1


def make_session(
    sampling_params: SamplingParams | None = None,
) -> EdgeSession:
    return EdgeSession(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=sampling_params or SamplingParams(max_tokens=8),
        block_ids=([1],),
        prompt_len=2,
        num_computed_tokens=2,
        total_len=2,
        token_ids=[10, 11],
    )


@pytest.mark.parametrize(
    ("sampling_params", "expected_message"),
    [
        (SamplingParams(max_tokens=8, presence_penalty=0.5),
         "presence_penalty"),
        (SamplingParams(max_tokens=8, frequency_penalty=0.5),
         "frequency_penalty"),
        (SamplingParams(max_tokens=8, repetition_penalty=1.1),
         "repetition_penalty"),
        (SamplingParams(max_tokens=8, bad_words=["nope"]), "bad_words"),
    ],
)
def test_bootstrap_rejects_unsupported_sampling_params(
    sampling_params: SamplingParams,
    expected_message: str,
) -> None:
    bridge = EdgeStateBridge()
    session = make_session(sampling_params)
    model_runner = SimpleNamespace(req_states=FakeReqStates())

    with pytest.raises(ValueError, match=expected_message):
        bridge.bootstrap_first_token(session, 20, model_runner)


@pytest.mark.parametrize(
    ("sampling_params", "expected_message"),
    [
        (SamplingParams(max_tokens=8, presence_penalty=0.5),
         "presence_penalty"),
        (SamplingParams(max_tokens=8, frequency_penalty=0.5),
         "frequency_penalty"),
        (SamplingParams(max_tokens=8, repetition_penalty=1.1),
         "repetition_penalty"),
        (SamplingParams(max_tokens=8, bad_words=["nope"]), "bad_words"),
    ],
)
def test_rollback_rejects_unsupported_sampling_params(
    sampling_params: SamplingParams,
    expected_message: str,
) -> None:
    bridge = EdgeStateBridge()
    session = make_session(sampling_params)
    session.token_ids = [10, 11, 20, 30]
    session.total_len = 4
    session.num_computed_tokens = 3
    req_states = FakeReqStates()
    model_runner = SimpleNamespace(req_states=req_states)

    with pytest.raises(ValueError, match=expected_message):
        bridge.rollback(session, 1, model_runner)

    assert session.token_ids == [10, 11, 20, 30]
    assert session.total_len == 4
    assert session.num_computed_tokens == 3
    assert int(req_states.last_sampled_tokens[0, 0]) == 0
    assert req_states.total_len.values == [2]
    assert req_states.num_computed_tokens.values == [2]
    assert req_states.apply_calls == 0


def test_bootstrap_and_external_commit_write_req_state() -> None:
    bridge = EdgeStateBridge()
    session = make_session()
    req_states = FakeReqStates()
    model_runner = SimpleNamespace(req_states=req_states)

    bridge.bootstrap_first_token(session, 20, model_runner)
    assert req_states.apply_calls == 1
    assert req_states.all_token_ids.rows[0][:3] == [10, 11, 20]
    assert req_states.total_len.values == [3]
    assert req_states.num_computed_tokens.values == [2]

    bridge.inject_external_token(session, 21, model_runner)

    assert session.token_ids == [10, 11, 20, 21]
    assert session.total_len == 4
    assert session.num_computed_tokens == 2
    assert int(req_states.last_sampled_tokens[0, 0]) == 21
    assert req_states.all_token_ids.rows[0][:4] == [10, 11, 20, 21]
    assert req_states.total_len.values == [4]
    assert req_states.num_computed_tokens.values == [2]
    assert req_states.apply_calls == 2


def test_prepare_next_decode_updates_last_sampled_without_staged_apply() -> None:
    bridge = EdgeStateBridge()
    session = make_session()
    req_states = FakeReqStates()
    model_runner = SimpleNamespace(req_states=req_states)

    bridge.prepare_next_decode(session, 20, model_runner)

    assert session.token_ids == [10, 11]
    assert session.total_len == 2
    assert session.num_computed_tokens == 2
    assert session.round_state.committed_token_id == 20
    assert int(req_states.last_sampled_tokens[0, 0]) == 20
    assert req_states.all_token_ids.rows[0] == [10, 11, 0, 0, 0, 0]
    assert req_states.total_len.values == [2]
    assert req_states.num_computed_tokens.values == [2]
    assert req_states.apply_calls == 0


def test_commit_token_appends_computed_token() -> None:
    bridge = EdgeStateBridge()
    session = make_session()

    bridge.commit_token(session, 20)

    assert session.token_ids == [10, 11, 20]
    assert session.total_len == 3
    assert session.num_computed_tokens == 3


def test_rollback_rewinds_session_and_req_state() -> None:
    bridge = EdgeStateBridge()
    session = make_session()
    session.token_ids = [10, 11, 20, 30, 31]
    session.total_len = 5
    session.num_computed_tokens = 4
    req_states = FakeReqStates()
    model_runner = SimpleNamespace(req_states=req_states)

    bridge.rollback(session, 2, model_runner)

    assert session.token_ids == [10, 11, 20]
    assert session.total_len == 3
    assert session.num_computed_tokens == 3
    assert int(req_states.last_sampled_tokens[0, 0]) == 20
    assert req_states.total_len.values == [3]
    assert req_states.num_computed_tokens.values == [3]
    assert req_states.apply_calls == 1
