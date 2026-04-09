from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.edge import EdgeDecodeEngine
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


UNSUPPORTED_SAMPLING_CASES = [
    (SamplingParams(max_tokens=8, presence_penalty=0.5), "presence_penalty"),
    (SamplingParams(max_tokens=8, frequency_penalty=0.5), "frequency_penalty"),
    (SamplingParams(max_tokens=8, repetition_penalty=1.1),
     "repetition_penalty"),
    (SamplingParams(max_tokens=8, bad_words=["nope"]), "bad_words"),
    (
        SamplingParams(
            max_tokens=8,
            structured_outputs=StructuredOutputsParams(
                grammar="root ::= 'hi'",
            ),
        ),
        "structured_outputs",
    ),
]


class FakeAsyncOutput(AsyncModelRunnerOutput):

    def __init__(self, output: ModelRunnerOutput) -> None:
        self.output = output
        self.calls = 0

    def get_output(self) -> ModelRunnerOutput:
        self.calls += 1
        return self.output


class FakeModel:

    def __init__(self) -> None:
        self.compute_logits_calls = []

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_logits_calls.append(hidden_states.clone())
        return hidden_states.to(dtype=torch.float32) + 100.0


class FakeModelRunner:

    def __init__(self, vocab_size: int = 4) -> None:
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.model = FakeModel()
        self.execute_model_state = None
        self.postprocess_calls = []

    def postprocess(
        self,
        input_batch,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        self.postprocess_calls.append(
            (input_batch, sampled_tokens.clone(), num_sampled.clone(),
             num_rejected.clone())
        )


class FakeWorker:

    def __init__(self, model_runner: FakeModelRunner) -> None:
        self.model_runner = model_runner
        self.execute_calls = []
        self.outputs = []
        self.sample_tokens_calls = 0

    def execute_model(self, scheduler_output):
        self.execute_calls.append(scheduler_output)
        if self.outputs:
            return self.outputs.pop(0)
        return None

    def sample_tokens(self, *_args, **_kwargs):
        self.sample_tokens_calls += 1
        raise AssertionError("EdgeDecodeEngine should not call sample_tokens")


@dataclass
class FakeScheduler:
    allocate_result: tuple[list[int], ...] = ([7, 8],)

    def __post_init__(self) -> None:
        self.allocate_calls = []
        self.prefill_calls = []
        self.decode_calls = []
        self.close_calls = []
        self.free_calls = []

    def allocate_blocks(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request=None,
    ) -> tuple[list[int], ...]:
        self.allocate_calls.append(
            (req_id, list(prompt_token_ids), sampling_params, lora_request)
        )
        return self.allocate_result

    def build_prefill_step(self, session):
        self.prefill_calls.append(session)
        return ("prefill", session.req_id)

    def build_decode_step(self, session):
        self.decode_calls.append(session)
        return ("decode", session.req_id, len(self.decode_calls))

    def build_close_step(self, req_id: str):
        self.close_calls.append(req_id)
        return ("close", req_id)

    def free_blocks(self, session) -> None:
        self.free_calls.append(session)


class FakeStateBridge:

    def __init__(self) -> None:
        self.bootstrap_calls = []
        self.prepare_calls = []
        self.inject_calls = []
        self.rollback_calls = []
        self.clear_calls = []
        self.commit_calls = []

    def bootstrap_first_token(
        self,
        session,
        bootstrap_token_id: int,
        model_runner: FakeModelRunner,
    ) -> int:
        self.bootstrap_calls.append((session, bootstrap_token_id, model_runner))
        self.inject_external_token(session, bootstrap_token_id, model_runner)
        self.clear_round_state(session)
        return int(bootstrap_token_id)

    def prepare_next_decode(
        self,
        session,
        input_token_id: int,
        model_runner: FakeModelRunner,
    ) -> None:
        self.prepare_calls.append((session, input_token_id, model_runner))
        session.round_state.committed_token_id = int(input_token_id)

    def commit_token(self, session, token_id: int) -> None:
        self.commit_calls.append((session, token_id))
        session.append_token(int(token_id), computed_delta=1)

    def inject_external_token(
        self,
        session,
        token_id: int,
        model_runner: FakeModelRunner,
    ) -> None:
        self.inject_calls.append((session, token_id, model_runner))
        session.append_token(int(token_id), computed_delta=0)
        session.round_state.committed_token_id = int(token_id)

    def rollback(self, session, rejected_count: int, model_runner) -> None:
        self.rollback_calls.append((session, rejected_count, model_runner))
        session.rollback(rejected_count)

    def clear_round_state(self, session) -> None:
        self.clear_calls.append(session)
        session.round_state.reset()


class FakeDraftSampler:

    def __init__(self) -> None:
        self.sample_step_calls = []
        self.sampled_tokens = [13, 14, 15, 16]

    def sample_step(
        self,
        logits: torch.Tensor,
        input_batch,
        processed_logits_dst: torch.Tensor,
    ) -> tuple[int, float]:
        self.sample_step_calls.append(
            (logits.clone(), input_batch, processed_logits_dst.clone())
        )
        processed_logits_dst.copy_(logits)
        token_id = self.sampled_tokens[len(self.sample_step_calls) - 1]
        q_value = round(0.1 * len(self.sample_step_calls), 3)
        return token_id, q_value

    def build_sampled_tokens(
        self,
        token_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor([[token_id]], dtype=torch.int64, device=device)

    def build_num_sampled(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([1], dtype=torch.int32, device=device)

    def build_num_rejected(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.int32, device=device)


def make_engine():
    model_runner = FakeModelRunner()
    worker = FakeWorker(model_runner)
    scheduler = FakeScheduler()
    state_bridge = FakeStateBridge()
    draft_sampler = FakeDraftSampler()
    engine = EdgeDecodeEngine(
        vllm_config=SimpleNamespace(),
        worker=worker,
        scheduler=scheduler,
        state_bridge=state_bridge,
        draft_sampler=draft_sampler,
    )
    return engine, worker, scheduler, state_bridge, draft_sampler, model_runner


def make_execute_model_state(
    hidden_states: torch.Tensor | None = None,
):
    if hidden_states is None:
        hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    return SimpleNamespace(hidden_states=hidden_states, input_batch=input_batch)


def test_open_prefill_commit_external_rollback_and_close_session() -> None:
    engine, worker, scheduler, state_bridge, _draft_sampler, _model_runner = (
        make_engine())
    worker.outputs = [
        ModelRunnerOutput(req_ids=["req-1"], req_id_to_index={"req-1": 0}),
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    ]

    session = engine.open_session(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
    )
    token_id = engine.prefill(session, bootstrap_token_id=21)
    committed = engine.commit_external_token(session, 22)
    engine.rollback(session, rejected_count=1)
    engine.close_session(session)

    assert engine.model_runner is worker.model_runner
    assert scheduler.allocate_calls[0][0] == "req-1"
    assert session.block_ids == ([7, 8],)
    assert worker.execute_calls[0] == ("prefill", "req-1")
    assert token_id == 21
    assert committed == 22
    assert worker.sample_tokens_calls == 0
    assert session.num_computed_tokens == session.prompt_len
    assert session.token_ids == [10, 11, 21]
    assert state_bridge.bootstrap_calls[0][1] == 21
    assert [call[1] for call in state_bridge.inject_calls] == [21, 22]
    assert state_bridge.rollback_calls[0][1] == 1
    assert scheduler.free_calls == [session]
    assert scheduler.close_calls == ["req-1"]
    assert worker.execute_calls[1] == ("close", "req-1")
    assert "req-1" not in engine.sessions
    assert state_bridge.clear_calls[-1] is session
    assert session.round_state.draft_token_ids == []


@pytest.mark.parametrize(("sampling_params", "expected_message"),
                         UNSUPPORTED_SAMPLING_CASES)
def test_open_session_rejects_unsupported_sampling_params_before_side_effects(
    sampling_params: SamplingParams,
    expected_message: str,
) -> None:
    engine, worker, scheduler, state_bridge, _draft_sampler, _model_runner = (
        make_engine())

    with pytest.raises(ValueError, match=expected_message):
        engine.open_session(
            req_id="req-1",
            prompt_token_ids=[10, 11],
            sampling_params=sampling_params,
        )

    assert scheduler.allocate_calls == []
    assert worker.execute_calls == []
    assert worker.sample_tokens_calls == 0
    assert engine.sessions == {}
    assert state_bridge.bootstrap_calls == []
    assert state_bridge.prepare_calls == []
    assert state_bridge.inject_calls == []
    assert state_bridge.rollback_calls == []
    assert state_bridge.clear_calls == []
    assert state_bridge.commit_calls == []


def test_execute_normalizes_async_and_sync_outputs() -> None:
    engine, worker, _scheduler, _state_bridge, _draft_sampler, _model_runner = (
        make_engine())
    sync_output = ModelRunnerOutput(req_ids=["req-1"], req_id_to_index={"req-1": 0})
    async_output = FakeAsyncOutput(sync_output)
    worker.outputs = [sync_output, async_output, None]

    first = engine._execute("sync-step")
    second = engine._execute("async-step")
    third = engine._execute("empty-step")

    assert first is sync_output
    assert second is sync_output
    assert third is None
    assert async_output.calls == 1
    assert worker.execute_calls == ["sync-step", "async-step", "empty-step"]


def test_decode_one_executes_step_samples_and_commits_token() -> None:
    engine, worker, scheduler, state_bridge, draft_sampler, model_runner = (
        make_engine())
    session = engine.open_session(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
    )
    session.num_computed_tokens = session.prompt_len
    model_runner.execute_model_state = make_execute_model_state()
    worker.outputs = [
        ModelRunnerOutput(req_ids=["req-1"], req_id_to_index={"req-1": 0})
    ]
    logits_dst = torch.empty((1, model_runner.vocab_size), dtype=torch.float32)

    token_id, q_value = engine.decode_one(session, 21, logits_dst)

    assert state_bridge.prepare_calls[0][1] == 21
    assert worker.execute_calls[-1] == ("decode", "req-1", 1)
    assert scheduler.decode_calls == [session]
    assert token_id == 13
    assert q_value == pytest.approx(0.1)
    assert session.token_ids == [10, 11, 13]
    assert session.num_computed_tokens == 3
    assert state_bridge.commit_calls[0][1] == 13
    assert model_runner.execute_model_state is None
    assert torch.equal(
        draft_sampler.sample_step_calls[0][0],
        torch.tensor([[101.0, 102.0, 103.0, 104.0]], dtype=torch.float32),
    )
    assert torch.equal(
        model_runner.postprocess_calls[0][1],
        torch.tensor([[13]], dtype=torch.int64),
    )
    assert torch.equal(model_runner.postprocess_calls[0][2],
                       torch.tensor([1], dtype=torch.int32))
    assert torch.equal(model_runner.postprocess_calls[0][3],
                       torch.tensor([0], dtype=torch.int32))


def test_draft_repeats_decode_gamma_times_and_collects_round_state() -> None:
    engine, worker, scheduler, state_bridge, _draft_sampler, model_runner = (
        make_engine())
    session = engine.open_session(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
    )
    session.round_state.draft_token_ids = [999]
    session.round_state.draft_q_values = [0.999]
    session.round_state.committed_token_id = -1
    session.num_computed_tokens = session.prompt_len
    worker.outputs = [
        ModelRunnerOutput(req_ids=["req-1"], req_id_to_index={"req-1": 0}),
        ModelRunnerOutput(req_ids=["req-1"], req_id_to_index={"req-1": 0}),
    ]

    decode_states = [
        make_execute_model_state(
            torch.tensor([[5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        ),
        make_execute_model_state(
            torch.tensor([[9.0, 10.0, 11.0, 12.0]], dtype=torch.float32)
        ),
    ]
    original_execute = worker.execute_model

    def execute_model_with_state(step):
        model_runner.execute_model_state = decode_states[len(worker.execute_calls)]
        return original_execute(step)

    worker.execute_model = execute_model_with_state

    round_state = engine.draft(session, first_token_id=21, gamma=2)

    assert state_bridge.clear_calls[0] is session
    assert round_state is session.round_state
    assert round_state.committed_token_id == 13
    assert round_state.draft_token_ids == [13, 14]
    assert round_state.draft_q_values == [pytest.approx(0.1), pytest.approx(0.2)]
    assert round_state.draft_logits_buffer is not None
    assert round_state.draft_logits_buffer.shape == (2, model_runner.vocab_size)
    assert state_bridge.prepare_calls[0][1] == 21
    assert state_bridge.prepare_calls[1][1] == 13
    assert [call[1] for call in state_bridge.commit_calls] == [13, 14]
    assert worker.execute_calls == [("decode", "req-1", 1), ("decode", "req-1", 2)]
    assert scheduler.decode_calls == [session, session]


def test_sample_with_draft_sampler_requires_execute_model_state() -> None:
    engine, _worker, _scheduler, _state_bridge, _draft_sampler, _model_runner = (
        make_engine())
    session = engine.open_session(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
    )

    with pytest.raises(RuntimeError, match="execute_model"):
        engine._sample_with_draft_sampler(session, torch.empty((1, 4)))
