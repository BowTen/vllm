# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.dssd.protocol import (
    CloseSessionAck,
    OpenSessionRequest,
    VerifyRoundRequest,
)
from vllm.sampling_params import SamplingParams


@dataclass
class FakeOpenResult:
    req_id: str
    bootstrap_token_id: int


@dataclass
class FakeVerifyResult:
    req_id: str
    accepted_len: int
    bonus_token_id: int | None = None
    rejected_target_logits: object | None = None


class FakeVerifierEngine:
    def __init__(self) -> None:
        self.open_calls: list[tuple[str, list[int], object, object | None]] = []
        self.verify_calls: list[tuple[object, object]] = []
        self.close_calls: list[object] = []
        self.sessions = {"req-1": object()}

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> FakeOpenResult:
        self.open_calls.append(
            (req_id, list(prompt_token_ids), sampling_params, lora_request)
        )
        session = object()
        self.sessions[req_id] = session
        return FakeOpenResult(req_id=req_id, bootstrap_token_id=11)

    def verify_round(self, session, request) -> FakeVerifyResult:
        self.verify_calls.append((session, request))
        return FakeVerifyResult(
            req_id=request.req_id,
            accepted_len=1,
            bonus_token_id=13,
        )

    def close_session(self, session) -> None:
        self.close_calls.append(session)


def test_verifier_service_open_session_delegates_to_decode_engine() -> None:
    from vllm.dssd.service.verifier_service import DSSDVerifierService

    engine = FakeVerifierEngine()
    service = DSSDVerifierService(decode_engine=engine)
    sampling_params = SamplingParams(max_tokens=8)

    result = service.open_session(
        OpenSessionRequest(
            req_id="req-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sampling_params,
        ))

    assert result.req_id == "req-1"
    assert result.bootstrap_token_id == 11
    assert engine.open_calls == [("req-1", [1, 2, 3], sampling_params, None)]


def test_verifier_service_verify_round_uses_session_by_req_id() -> None:
    from vllm.dssd.service.verifier_service import DSSDVerifierService

    engine = FakeVerifierEngine()
    service = DSSDVerifierService(decode_engine=engine)
    request = VerifyRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[10],
        draft_q_values=[0.7],
    )

    response = service.verify_round(request)

    assert response.req_id == "req-1"
    assert response.accepted_len == 1
    assert response.bonus_token_id == 13
    assert engine.verify_calls == [(engine.sessions["req-1"], request)]


def test_verifier_service_close_session_closes_engine_session() -> None:
    from vllm.dssd.service.verifier_service import DSSDVerifierService

    engine = FakeVerifierEngine()
    service = DSSDVerifierService(decode_engine=engine)

    ack = service.close_session("req-1")

    assert ack == CloseSessionAck(req_id="req-1")
    assert engine.close_calls == [engine.sessions["req-1"]]


def test_verifier_service_verify_round_requires_existing_session() -> None:
    from vllm.dssd.service.verifier_service import DSSDVerifierService

    engine = FakeVerifierEngine()
    service = DSSDVerifierService(decode_engine=engine)
    request = VerifyRoundRequest(
        req_id="missing",
        committed_token_id=9,
        draft_token_ids=[],
        draft_q_values=[],
    )

    with pytest.raises(KeyError):
        service.verify_round(request)
