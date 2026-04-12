# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from vllm.sampling_params import SamplingParams


@dataclass
class FakeRoundState:
    draft_token_ids: list[int] = field(default_factory=list)
    draft_q_values: list[float] = field(default_factory=list)
    draft_logits_rows: list[torch.Tensor] = field(default_factory=list)

    def q_dist_at(self, index: int) -> torch.Tensor:
        return self.draft_logits_rows[index]


@dataclass
class FakeEdgeSession:
    req_id: str
    prompt_len: int
    token_ids: list[int]
    round_state: FakeRoundState = field(default_factory=FakeRoundState)

    def committed_output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]


class FakeEdgeDecodeEngine:
    def __init__(self) -> None:
        self.sessions: dict[str, FakeEdgeSession] = {}
        self.close_calls: list[str] = []

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> FakeEdgeSession:
        session = FakeEdgeSession(
            req_id=req_id,
            prompt_len=len(prompt_token_ids),
            token_ids=list(prompt_token_ids),
        )
        self.sessions[req_id] = session
        return session

    def prefill(self, session: FakeEdgeSession, bootstrap_token_id: int) -> int:
        session.token_ids.append(bootstrap_token_id)
        return bootstrap_token_id

    def draft(self, session: FakeEdgeSession, first_token_id: int, gamma: int):
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

    def rollback(self, session: FakeEdgeSession, rejected_count: int) -> None:
        if rejected_count > 0:
            session.token_ids = session.token_ids[:-rejected_count]

    def commit_external_token(self, session: FakeEdgeSession, token_id: int) -> int:
        session.token_ids.append(token_id)
        session.round_state = FakeRoundState()
        return token_id

    def close_session(self, session: FakeEdgeSession) -> None:
        self.close_calls.append(session.req_id)
        self.sessions.pop(session.req_id, None)


@dataclass
class FakeVerifierOpenResult:
    req_id: str
    bootstrap_token_id: int


@dataclass
class FakeVerifierRoundResult:
    req_id: str
    accepted_len: int
    bonus_token_id: int | None = None
    rejected_target_logits: torch.Tensor | None = None


class FakeVerifierDecodeEngine:
    def __init__(self) -> None:
        self.open_calls: list[str] = []
        self.verify_calls: list[object] = []
        self.close_calls: list[str] = []
        self.sessions: dict[str, object] = {}

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> FakeVerifierOpenResult:
        self.open_calls.append(req_id)
        self.sessions[req_id] = object()
        return FakeVerifierOpenResult(req_id=req_id, bootstrap_token_id=17)

    def verify_round(self, session, request) -> FakeVerifierRoundResult:
        self.verify_calls.append(request)
        return FakeVerifierRoundResult(
            req_id=request.req_id,
            accepted_len=1,
            rejected_target_logits=torch.tensor(
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=torch.float32,
            ),
        )

    def close_session(self, session) -> None:
        req_id = next(key for key, value in self.sessions.items() if value is session)
        self.close_calls.append(req_id)
        self.sessions.pop(req_id, None)


def test_in_process_service_round_trip_runs_full_dssd_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.service import DSSDEdgeService, DSSDVerifierService
    from vllm.dssd.transport import FakeNetwork, InProcessVerifierTransport

    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples: torch.tensor([2], dtype=torch.int64),
    )

    verifier_engine = FakeVerifierDecodeEngine()
    verifier_service = DSSDVerifierService(decode_engine=verifier_engine)
    transport = InProcessVerifierTransport(
        verifier_service=verifier_service,
        request_network=FakeNetwork(
            fixed_latency_ms=0.0,
            bandwidth_bytes_per_s=1e9,
        ),
        response_network=FakeNetwork(
            fixed_latency_ms=0.0,
            bandwidth_bytes_per_s=1e9,
        ),
    )
    edge_engine = FakeEdgeDecodeEngine()
    edge_service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=transport,
        eos_token_id=2,
        gamma=2,
    )

    output_ids = edge_service.generate(
        req_id="req-1",
        prompt_token_ids=[1, 3],
        sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
    )

    assert output_ids == [17, 19, 2]
    assert verifier_engine.open_calls == ["req-1"]
    assert len(verifier_engine.verify_calls) == 1
    assert verifier_engine.close_calls == ["req-1"]
    assert edge_engine.close_calls == ["req-1"]
