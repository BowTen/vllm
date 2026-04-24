from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.dssd.verifier.engine_v1 import VerifierDecodeEngineV1
from vllm.dssd.verifier.state_bridge_v1 import VerifierStateBridgeV1
from vllm.dssd.verifier.types import (
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierSession,
)
from vllm.sampling_params import SamplingParams


class FakeWorker:
    def __init__(self, model_runner) -> None:
        self.model_runner = model_runner
        self.execute_calls = []

    def execute_model(self, scheduler_output):
        self.execute_calls.append(scheduler_output)
        return None


class FakeScheduler:
    def __init__(self) -> None:
        self.allocate_calls = []
        self.open_calls = []
        self.decode_calls = []
        self.verify_calls = []
        self.close_calls = []
        self.free_calls = []

    def allocate_blocks(self, **kwargs):
        self.allocate_calls.append(kwargs)
        return ([7, 8],)

    def build_open_session_step(self, session):
        self.open_calls.append(session.req_id)
        return ("open", session.req_id)

    def build_verify_step(self, session, request):
        self.verify_calls.append((session.req_id, list(request.draft_token_ids)))
        return ("verify", session.req_id, len(request.draft_token_ids) + 1)

    def build_decode_step(self, session):
        self.decode_calls.append(session.req_id)
        return ("decode", session.req_id)

    def build_close_step(self, req_id):
        self.close_calls.append(req_id)
        return ("close", req_id)

    def free_blocks(self, session):
        self.free_calls.append(session.req_id)


def _runner() -> SimpleNamespace:
    runner = SimpleNamespace(
        num_spec_tokens=2,
        requests={},
        input_batch=SimpleNamespace(
            req_id_to_index={},
            sampling_metadata=SimpleNamespace(generators={}),
        ),
    )
    runner.take_execute_model_state_calls = 0

    def take_execute_model_state():
        if runner.execute_model_state is None:
            raise RuntimeError("execute_model_state is empty")
        runner.take_execute_model_state_calls += 1
        state = runner.execute_model_state
        runner.execute_model_state = None
        return state

    runner.take_execute_model_state = take_execute_model_state
    return runner


def _runner_with_request_state() -> SimpleNamespace:
    runner = _runner()
    token_ids_cpu = np.zeros((1, 16), dtype=np.int32)
    token_ids_cpu[0, :3] = [1, 2, 3]
    is_token_ids = np.zeros((1, 16), dtype=bool)
    is_token_ids[0, :3] = True
    req_output_token_ids = [[]]
    runner.requests = {
        "req-1": SimpleNamespace(
            output_token_ids=req_output_token_ids[0],
            num_computed_tokens=3,
        ),
    }
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"req-1": 0},
        sampling_metadata=SimpleNamespace(generators={}),
        token_ids_cpu=token_ids_cpu,
        is_token_ids=is_token_ids,
        num_tokens_no_spec=np.array([3], dtype=np.int32),
        num_computed_tokens_cpu=np.array([3], dtype=np.int32),
        req_output_token_ids=req_output_token_ids,
    )
    return runner


def _session() -> VerifierSession:
    return VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        block_ids=([7, 8],),
        prompt_len=3,
        token_ids=[1, 2, 3],
        num_computed_tokens=3,
        total_len=3,
    )


def test_open_session_samples_bootstrap_without_committing_it() -> None:
    model_runner = _runner()
    model_runner.execute_model_state = SimpleNamespace(
        logits=None,
        spec_decode_metadata=None,
    )
    worker = FakeWorker(model_runner)
    scheduler = FakeScheduler()
    sampler = SimpleNamespace(sample_bootstrap=lambda **_kwargs: 17)
    bridge = SimpleNamespace(
        finish_prefill_without_commit=lambda session, runner: (
            setattr(session, "token_ids", list(session.prompt_token_ids)),
            setattr(session, "num_computed_tokens", session.prompt_len),
            setattr(session, "total_len", session.prompt_len),
        )
    )
    engine = VerifierDecodeEngineV1(
        vllm_config=SimpleNamespace(),
        worker=worker,
        scheduler=scheduler,
        state_bridge=bridge,
        verifier_sampler=sampler,
    )

    result = engine.open_session(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )

    assert result.bootstrap_token_id == 17
    assert engine.sessions["req-1"].token_ids == [1, 2, 3]
    assert worker.execute_calls == [("open", "req-1")]
    assert model_runner.take_execute_model_state_calls == 1
    assert model_runner.execute_model_state is None


def test_verify_round_runs_single_forward_and_only_commits_accepted_prefix(
) -> None:
    model_runner = _runner_with_request_state()
    model_runner.execute_model_state = SimpleNamespace(
        logits="logits",
        spec_decode_metadata="metadata",
    )
    worker = FakeWorker(model_runner)
    scheduler = FakeScheduler()
    round_result = VerifierRoundResult(
        req_id="req-1",
        accepted_len=1,
        bonus_token_id=None,
        rejected_target_logits=torch.tensor([0.1, 0.2, 0.3]),
    )
    sampler = SimpleNamespace(
        verify_round=lambda **_kwargs: round_result,
    )
    begin_calls = []
    bridge = VerifierStateBridgeV1()
    original_begin_round = bridge.begin_round

    def begin_round(session, request, runner, *, gamma):
        begin_calls.append((session.req_id, request.committed_token_id, gamma))
        return original_begin_round(session, request, runner, gamma=gamma)

    bridge.begin_round = begin_round
    engine = VerifierDecodeEngineV1(
        vllm_config=SimpleNamespace(),
        worker=worker,
        scheduler=scheduler,
        state_bridge=bridge,
        verifier_sampler=sampler,
    )
    session = _session()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )

    result = engine.verify_round(
        session,
        request,
    )

    assert result is round_result
    assert begin_calls == [("req-1", 9, 2)]
    assert worker.execute_calls == [("verify", "req-1", 3)]
    assert session.token_ids == [1, 2, 3, 9, 11]
    assert session.total_len == 5
    assert session.num_computed_tokens == 5
    assert model_runner.requests["req-1"].output_token_ids == [9, 11]
    assert int(model_runner.input_batch.num_tokens_no_spec[0]) == 5
    assert int(model_runner.input_batch.num_computed_tokens_cpu[0]) == 5
    assert model_runner.input_batch.token_ids_cpu[0, :5].tolist() == [
        1,
        2,
        3,
        9,
        11,
    ]
    assert model_runner.take_execute_model_state_calls == 1
    assert model_runner.execute_model_state is None


def test_verify_round_rolls_back_round_state_when_sampler_raises() -> None:
    model_runner = _runner_with_request_state()
    model_runner.execute_model_state = SimpleNamespace(
        logits="logits",
        spec_decode_metadata="metadata",
    )
    worker = FakeWorker(model_runner)
    scheduler = FakeScheduler()
    bridge = VerifierStateBridgeV1()
    engine = VerifierDecodeEngineV1(
        vllm_config=SimpleNamespace(),
        worker=worker,
        scheduler=scheduler,
        state_bridge=bridge,
        verifier_sampler=SimpleNamespace(
            verify_round=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("sampler boom"))
        ),
    )
    session = _session()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )

    with pytest.raises(RuntimeError, match="sampler boom"):
        engine.verify_round(session, request)

    assert worker.execute_calls == [("verify", "req-1", 3)]
    assert session.token_ids == [1, 2, 3]
    assert session.total_len == 3
    assert session.num_computed_tokens == 3
    assert model_runner.requests["req-1"].output_token_ids == []
    assert model_runner.requests["req-1"].num_computed_tokens == 3
    assert int(model_runner.input_batch.num_tokens_no_spec[0]) == 3
    assert int(model_runner.input_batch.num_computed_tokens_cpu[0]) == 3
    assert model_runner.input_batch.token_ids_cpu[0, :5].tolist() == [
        1,
        2,
        3,
        0,
        0,
    ]
    assert model_runner.input_batch.is_token_ids[0, :5].tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert model_runner.take_execute_model_state_calls == 1
    assert model_runner.execute_model_state is None
    assert "req-1" not in bridge._round_states


def test_decode_one_local_executes_step_samples_and_commits_token() -> None:
    model_runner = _runner_with_request_state()
    model_runner.requests["req-1"].output_token_ids.append(12)
    model_runner.requests["req-1"].num_computed_tokens = 4
    model_runner.input_batch.token_ids_cpu[0, 3] = 12
    model_runner.input_batch.is_token_ids[0, 3] = True
    model_runner.input_batch.num_tokens_no_spec[0] = 4
    model_runner.input_batch.num_computed_tokens_cpu[0] = 4
    model_runner.execute_model_state = SimpleNamespace(
        logits="decode-logits",
        spec_decode_metadata=None,
    )
    worker = FakeWorker(model_runner)
    scheduler = FakeScheduler()
    sampler_calls = []

    def sample_bootstrap(**kwargs):
        sampler_calls.append(kwargs)
        return 13

    engine = VerifierDecodeEngineV1(
        vllm_config=SimpleNamespace(),
        worker=worker,
        scheduler=scheduler,
        state_bridge=VerifierStateBridgeV1(),
        verifier_sampler=SimpleNamespace(sample_bootstrap=sample_bootstrap),
    )
    session = _session()
    session.token_ids.append(12)
    session.total_len = 4
    session.num_computed_tokens = 4

    token_id = engine.decode_one_local(session, 12)

    assert token_id == 13
    assert worker.execute_calls == [("decode", "req-1")]
    assert scheduler.decode_calls == ["req-1"]
    assert sampler_calls[0]["logits"] == "decode-logits"
    assert session.token_ids == [1, 2, 3, 12, 13]
    assert session.total_len == 5
    assert session.num_computed_tokens == 5
    assert model_runner.requests["req-1"].output_token_ids == [12, 13]
    assert int(model_runner.input_batch.num_tokens_no_spec[0]) == 5
    assert int(model_runner.input_batch.num_computed_tokens_cpu[0]) == 5
    assert model_runner.input_batch.token_ids_cpu[0, :5].tolist() == [
        1,
        2,
        3,
        12,
        13,
    ]


def test_generate_local_uses_empty_verify_rounds_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_runner = _runner()
    worker = FakeWorker(model_runner)
    scheduler = FakeScheduler()
    close_calls = []
    verify_calls = []
    session = VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=3),
        block_ids=([7, 8],),
        prompt_len=3,
        token_ids=[1, 2, 3],
        num_computed_tokens=3,
        total_len=3,
    )

    engine = VerifierDecodeEngineV1(
        vllm_config=SimpleNamespace(),
        worker=worker,
        scheduler=scheduler,
        state_bridge=SimpleNamespace(),
        verifier_sampler=SimpleNamespace(),
    )

    def fake_open_session(req_id, prompt_token_ids, sampling_params, lora_request=None):
        del prompt_token_ids, sampling_params, lora_request
        engine.sessions[req_id] = session
        return SimpleNamespace(req_id=req_id, bootstrap_token_id=21)

    def fake_verify_round(local_session, request):
        verify_calls.append((
            local_session,
            request.committed_token_id,
            list(request.draft_token_ids),
            list(request.draft_q_values),
        ))
        token_id = 22 if len(verify_calls) == 1 else 23
        return VerifierRoundResult(
            req_id=request.req_id,
            accepted_len=0,
            bonus_token_id=token_id,
        )

    monkeypatch.setattr(engine, "open_session", fake_open_session)
    monkeypatch.setattr(engine, "verify_round", fake_verify_round)
    monkeypatch.setattr(
        engine,
        "close_session",
        lambda local_session: close_calls.append(local_session),
    )

    output_token_ids = engine.generate_local(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=3),
    )

    assert output_token_ids == [21, 22, 23]
    assert verify_calls == [
        (session, 21, [], []),
        (session, 22, [], []),
    ]
    assert close_calls == [session]
