# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.protocol import OpenSessionResponse, VerifyRoundResponse
from vllm.sampling_params import SamplingParams


@dataclass
class FakeRoundState:
    draft_token_ids: list[int] = field(default_factory=list)
    draft_q_values: list[float] = field(default_factory=list)
    draft_logits_rows: list[torch.Tensor] = field(default_factory=list)

    def q_dist_at(self, index: int) -> torch.Tensor:
        return self.draft_logits_rows[index]


@dataclass
class FakeSession:
    req_id: str
    prompt_len: int
    token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    round_state: FakeRoundState = field(default_factory=FakeRoundState)

    def committed_output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]


class FakeEdgeDecodeEngine:
    def __init__(self) -> None:
        self.open_calls = []
        self.prefill_calls = []
        self.draft_calls = []
        self.rollback_calls = []
        self.commit_external_calls = []
        self.close_calls = []
        self.sessions: dict[str, FakeSession] = {}

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> FakeSession:
        self.open_calls.append((req_id, list(prompt_token_ids), sampling_params))
        session = FakeSession(
            req_id=req_id,
            prompt_len=len(prompt_token_ids),
            token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
        )
        self.sessions[req_id] = session
        return session

    def prefill(self, session: FakeSession, bootstrap_token_id: int) -> int:
        self.prefill_calls.append((session, bootstrap_token_id))
        session.token_ids.append(bootstrap_token_id)
        return bootstrap_token_id

    def draft(
        self,
        session: FakeSession,
        first_token_id: int,
        gamma: int,
    ) -> FakeRoundState:
        self.draft_calls.append((session, first_token_id, gamma))
        session.round_state = FakeRoundState(
            draft_token_ids=[19, 20],
            draft_q_values=[0.6, 0.4],
            draft_logits_rows=[
                torch.zeros(8, dtype=torch.float32),
                torch.zeros(8, dtype=torch.float32),
            ],
        )
        session.token_ids.extend(session.round_state.draft_token_ids)
        return session.round_state

    def rollback(self, session: FakeSession, rejected_count: int) -> None:
        self.rollback_calls.append(rejected_count)
        if rejected_count > 0:
            session.token_ids = session.token_ids[:-rejected_count]

    def commit_external_token(self, session: FakeSession, token_id: int) -> int:
        self.commit_external_calls.append(token_id)
        session.token_ids.append(token_id)
        session.round_state = FakeRoundState()
        return token_id

    def close_session(self, session: FakeSession) -> None:
        self.close_calls.append(session.req_id)
        self.sessions.pop(session.req_id, None)


class FailingEdgeDecodeEngine(FakeEdgeDecodeEngine):
    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> FakeSession:
        raise RuntimeError("edge open failed")


class FakeVerifierTransport:
    def __init__(self, verify_responses: list[VerifyRoundResponse]) -> None:
        self.verify_responses = list(verify_responses)
        self.open_calls = []
        self.verify_calls = []
        self.close_calls = []
        self.raise_on_verify = False

    def open_session(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> OpenSessionResponse:
        self.open_calls.append((req_id, list(prompt_token_ids), sampling_params))
        return OpenSessionResponse(req_id=req_id, bootstrap_token_id=17)

    def verify_round(self, request) -> VerifyRoundResponse:
        if self.raise_on_verify:
            raise RuntimeError("verify failed")
        self.verify_calls.append(request)
        return self.verify_responses.pop(0)

    def close_session(self, req_id: str):
        self.close_calls.append(req_id)
        return SimpleNamespace(req_id=req_id, ack=True)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, prompt: str):
        self.calls.append(("encode", prompt))
        return SimpleNamespace(input_ids=[11, 12])

    def decode(self, token_ids: list[int], **kwargs) -> str:
        self.calls.append(("decode", list(token_ids), kwargs))
        return " completed text"


def test_edge_service_open_session_prefills_with_verifier_bootstrap() -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    edge_engine = FakeEdgeDecodeEngine()
    verifier = FakeVerifierTransport(verify_responses=[])
    sampling_params = SamplingParams(max_tokens=8)
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        eos_token_id=2,
        gamma=2,
    )

    opened = service.open_session(
        req_id="req-1",
        prompt_token_ids=[1, 3, 5],
        sampling_params=sampling_params,
    )

    assert opened.req_id == "req-1"
    assert opened.bootstrap_token_id == 17
    assert len(edge_engine.prefill_calls) == 1
    assert verifier.open_calls == [("req-1", [1, 3, 5], sampling_params)]


def test_edge_service_complete_encodes_prompt_and_decodes_generated_ids() -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    tokenizer = FakeTokenizer()
    edge_engine = FakeEdgeDecodeEngine()
    verifier = FakeVerifierTransport(
        verify_responses=[
            VerifyRoundResponse(
                req_id="req-1",
                accepted_len=2,
                bonus_token_id=21,
            )
        ]
    )
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        tokenizer=tokenizer,
        eos_token_id=2,
        gamma=2,
    )
    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)

    completion = service.complete(
        req_id="req-1",
        prompt="hello",
        sampling_params=sampling_params,
    )

    assert completion == " completed text"
    assert edge_engine.open_calls == [("req-1", [11, 12], sampling_params)]
    assert tokenizer.calls == [
        ("encode", "hello"),
        ("decode", [17, 19, 20, 21], {"skip_special_tokens": True}),
    ]


def test_edge_service_generate_reject_path_rolls_back_and_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples: torch.tensor([2], dtype=torch.int64),
    )
    edge_engine = FakeEdgeDecodeEngine()
    verifier = FakeVerifierTransport(
        verify_responses=[
            VerifyRoundResponse(
                req_id="req-1",
                accepted_len=1,
                rejected_target_logits=torch.tensor(
                    [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=torch.float32,
                ),
            )
        ]
    )
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        eos_token_id=2,
        gamma=2,
    )

    output_ids = service.generate(
        req_id="req-1",
        prompt_token_ids=[1, 3],
        sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
    )

    assert edge_engine.rollback_calls == [1]
    assert edge_engine.commit_external_calls == [2]
    assert output_ids == [17, 19, 2]
    assert edge_engine.close_calls == ["req-1"]
    assert verifier.close_calls == ["req-1"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_edge_service_reject_path_accepts_remote_cpu_logits_with_local_cuda_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples: torch.tensor([2], device=probs.device),
    )
    service = DSSDEdgeService(
        decode_engine=FakeEdgeDecodeEngine(),
        verifier=FakeVerifierTransport(verify_responses=[]),
        eos_token_id=2,
        gamma=2,
    )
    session = FakeSession(
        req_id="req-1",
        prompt_len=2,
        token_ids=[1, 3, 17, 19, 20],
        round_state=FakeRoundState(
            draft_token_ids=[19, 20],
            draft_q_values=[0.6, 0.4],
            draft_logits_rows=[
                torch.zeros(8, dtype=torch.float16, device="cuda"),
                torch.zeros(8, dtype=torch.float16, device="cuda"),
            ],
        ),
    )
    response = VerifyRoundResponse(
        req_id="req-1",
        accepted_len=1,
        rejected_target_logits=torch.tensor(
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cpu",
        ),
    )

    token_id = service._resample_rejected_token(session, response)  # noqa: SLF001

    assert token_id == 2


def test_edge_service_greedy_reject_path_uses_target_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    def fail_multinomial(*_args, **_kwargs):
        raise AssertionError("greedy rejection must not resample")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    service = DSSDEdgeService(
        decode_engine=FakeEdgeDecodeEngine(),
        verifier=FakeVerifierTransport(verify_responses=[]),
        eos_token_id=2,
        gamma=2,
    )
    session = FakeSession(
        req_id="req-1",
        prompt_len=2,
        token_ids=[1, 3, 17, 19, 20],
        sampling_params=SamplingParams(temperature=0.0),
        round_state=FakeRoundState(
            draft_token_ids=[19, 20],
            draft_q_values=[0.6, 0.4],
            draft_logits_rows=[
                torch.zeros(8, dtype=torch.float32),
                torch.zeros(8, dtype=torch.float32),
            ],
        ),
    )
    response = VerifyRoundResponse(
        req_id="req-1",
        accepted_len=1,
        rejected_target_logits=torch.tensor(
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        ),
    )

    token_id = service._resample_rejected_token(session, response)  # noqa: SLF001

    assert token_id == 2


def test_edge_service_close_session_closes_both_sides() -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    edge_engine = FakeEdgeDecodeEngine()
    verifier = FakeVerifierTransport(verify_responses=[])
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        eos_token_id=2,
        gamma=2,
    )
    session = edge_engine.open_session(
        req_id="req-1",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=8),
    )
    edge_engine.sessions["req-1"] = session

    service.close_session("req-1")

    assert edge_engine.close_calls == ["req-1"]
    assert verifier.close_calls == ["req-1"]


def test_edge_service_generate_stops_at_max_tokens_without_requiring_eos() -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    edge_engine = FakeEdgeDecodeEngine()
    verifier = FakeVerifierTransport(
        verify_responses=[
            VerifyRoundResponse(
                req_id="req-1",
                accepted_len=2,
                bonus_token_id=21,
            )
        ]
    )
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        eos_token_id=-1,
        gamma=2,
    )

    output_ids = service.generate(
        req_id="req-1",
        prompt_token_ids=[1, 3],
        sampling_params=SamplingParams(max_tokens=2, temperature=0.0),
    )

    assert output_ids == [17, 19]
    assert edge_engine.rollback_calls == [2]
    assert edge_engine.commit_external_calls == [21]
    assert edge_engine.close_calls == ["req-1"]
    assert verifier.close_calls == ["req-1"]


def test_edge_service_generate_with_stats_tracks_acceptance() -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    edge_engine = FakeEdgeDecodeEngine()
    verifier = FakeVerifierTransport(
        verify_responses=[
            VerifyRoundResponse(
                req_id="req-1",
                accepted_len=2,
                bonus_token_id=21,
            )
        ]
    )
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        eos_token_id=-1,
        gamma=2,
    )

    output_ids, stats = service.generate_with_stats(
        req_id="req-1",
        prompt_token_ids=[1, 3],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )

    assert output_ids == [17, 19, 20, 21]
    assert stats.total_rounds == 1
    assert stats.total_draft_tokens == 2
    assert stats.total_accepted_tokens == 2
    assert stats.all_accept_rounds == 1
    assert stats.draft_acceptance_rate == 1.0
    assert stats.all_accept_round_rate == 1.0
    assert stats.avg_accepted_len_per_round == 2.0


def test_edge_service_generate_closes_sessions_when_verify_round_raises() -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    edge_engine = FakeEdgeDecodeEngine()
    verifier = FakeVerifierTransport(verify_responses=[])
    verifier.raise_on_verify = True
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        eos_token_id=2,
        gamma=2,
    )

    with pytest.raises(RuntimeError, match="verify failed"):
        service.generate(
            req_id="req-1",
            prompt_token_ids=[1, 3],
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
        )

    assert edge_engine.close_calls == ["req-1"]
    assert verifier.close_calls == ["req-1"]


def test_edge_service_open_session_closes_remote_if_edge_open_fails() -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    edge_engine = FailingEdgeDecodeEngine()
    verifier = FakeVerifierTransport(verify_responses=[])
    service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=verifier,
        eos_token_id=2,
        gamma=2,
    )
    sampling_params = SamplingParams(max_tokens=8)

    with pytest.raises(RuntimeError, match="edge open failed"):
        service.open_session(
            req_id="req-1",
            prompt_token_ids=[1, 3, 5],
            sampling_params=sampling_params,
        )

    assert verifier.open_calls == [("req-1", [1, 3, 5], sampling_params)]
    assert verifier.close_calls == ["req-1"]
