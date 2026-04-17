# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.verifier.engine import VerifierDecodeEngine
from vllm.dssd.verifier.types import VerifierSession
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class FakeModelRunner:
    def __init__(self) -> None:
        self.execute_model_state = None
        self.postprocess_calls = []
        self.sample_calls = []
        self.num_speculative_steps = 0

    def take_execute_model_state(self):
        state = self.execute_model_state
        self.execute_model_state = None
        return state

    def sample(self, hidden_states, input_batch, grammar_output):
        del grammar_output
        self.sample_calls.append((hidden_states.clone(), input_batch))
        sampler_output = SamplerOutput(
            sampled_token_ids=torch.tensor([[13]], dtype=torch.int64),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.tensor([1], dtype=torch.int32),
        )
        return (
            sampler_output,
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        )

    def postprocess(self, input_batch, sampled_tokens, num_sampled, num_rejected):
        self.postprocess_calls.append(
            (
                input_batch,
                sampled_tokens.clone(),
                num_sampled.clone(),
                num_rejected.clone(),
            )
        )


class FakeWorker:
    def __init__(self, model_runner: FakeModelRunner) -> None:
        self.model_runner = model_runner
        self.execute_calls = []

    def execute_model(self, scheduler_output):
        self.execute_calls.append(scheduler_output)
        return None


@dataclass
class FakeScheduler:
    allocate_result: tuple[list[int], ...] = ([7, 8],)

    def __post_init__(self) -> None:
        self.allocate_calls = []
        self.open_calls = []
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

    def build_open_session_step(self, session):
        self.open_calls.append(session)
        return ("prefill", session.req_id)

    def build_decode_step(self, session):
        self.decode_calls.append(session)
        return ("decode", session.req_id)

    def build_close_step(self, req_id: str):
        self.close_calls.append(req_id)
        return ("close", req_id)

    def free_blocks(self, session) -> None:
        self.free_calls.append(session)


class FakeStateBridge:
    def __init__(self) -> None:
        self.finish_prefill_calls = []
        self.inject_calls = []
        self.prepare_local_decode_calls = []
        self.commit_local_token_calls = []
        self.remove_round_state_calls = []

    def finish_prefill_without_commit(self, session) -> None:
        self.finish_prefill_calls.append(session)
        session.token_ids = list(session.prompt_token_ids)
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len

    def inject_local_token(
        self,
        session,
        token_id: int,
        model_runner,
        *,
        computed_delta: int,
    ) -> None:
        del model_runner
        self.inject_calls.append((session, token_id, computed_delta))
        session.token_ids.append(int(token_id))
        session.total_len += 1
        session.num_computed_tokens += computed_delta

    def prepare_local_decode(self, session, input_token_id: int, model_runner) -> None:
        del model_runner
        self.prepare_local_decode_calls.append((session, input_token_id))

    def commit_local_token(self, session, token_id: int) -> None:
        self.commit_local_token_calls.append((session, token_id))
        session.token_ids.append(int(token_id))
        session.total_len += 1
        session.num_computed_tokens += 1

    def remove_round_state(self, session) -> None:
        self.remove_round_state_calls.append(session)


class UnusedVerifierSampler:
    num_speculative_steps = 0

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("generate_local should not call verifier sampler")


def _make_engine():
    model_runner = FakeModelRunner()
    worker = FakeWorker(model_runner)
    scheduler = FakeScheduler()
    state_bridge = FakeStateBridge()
    engine = VerifierDecodeEngine(
        vllm_config=SimpleNamespace(),
        worker=worker,
        scheduler=scheduler,
        state_bridge=state_bridge,
        verifier_sampler=UnusedVerifierSampler(),
    )
    return engine, worker, scheduler, state_bridge, model_runner


def _make_execute_model_state():
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    hidden_states = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    return SimpleNamespace(hidden_states=hidden_states, input_batch=input_batch)


def test_decode_one_local_executes_step_samples_and_commits_token() -> None:
    engine, worker, scheduler, state_bridge, model_runner = _make_engine()
    session = VerifierSession(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=4),
        block_ids=([7, 8],),
        prompt_len=2,
        token_ids=[10, 11, 12],
        total_len=3,
        num_computed_tokens=3,
    )
    model_runner.execute_model_state = _make_execute_model_state()

    token_id = engine.decode_one_local(session, 12)

    assert token_id == 13
    assert state_bridge.prepare_local_decode_calls == [(session, 12)]
    assert worker.execute_calls == [("decode", "req-1")]
    assert scheduler.decode_calls == [session]
    assert state_bridge.commit_local_token_calls == [(session, 13)]
    assert session.token_ids == [10, 11, 12, 13]
    assert session.num_computed_tokens == 4
    assert torch.equal(
        model_runner.postprocess_calls[0][1],
        torch.tensor([[13]], dtype=torch.int64),
    )


def test_generate_local_bootstraps_runs_decode_loop_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _worker, _scheduler, state_bridge, _model_runner = _make_engine()
    session = VerifierSession(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=3),
        block_ids=([7, 8],),
        prompt_len=2,
        token_ids=[10, 11],
    )
    close_calls = []
    decode_calls = []

    def fake_open_session(req_id, prompt_token_ids, sampling_params, lora_request=None):
        del prompt_token_ids, sampling_params, lora_request
        engine.sessions[req_id] = session
        return SimpleNamespace(req_id=req_id, bootstrap_token_id=21)

    def fake_decode_one_local(local_session, input_token_id):
        decode_calls.append((local_session, input_token_id))
        return 22 if len(decode_calls) == 1 else 23

    monkeypatch.setattr(engine, "open_session", fake_open_session)
    monkeypatch.setattr(engine, "decode_one_local", fake_decode_one_local)
    monkeypatch.setattr(
        engine,
        "close_session",
        lambda local_session: close_calls.append(local_session),
    )

    output_token_ids = engine.generate_local(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=3),
    )

    assert output_token_ids == [21, 22, 23]
    assert state_bridge.inject_calls == [(session, 21, 1)]
    assert decode_calls == [(session, 21), (session, 22)]
    assert close_calls == [session]
