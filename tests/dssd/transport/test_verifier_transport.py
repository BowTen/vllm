# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

from vllm.dssd.protocol import (
    CloseSessionAck,
    OpenSessionResponse,
    VerifyRoundRequest,
    VerifyRoundResponse,
)
from vllm.sampling_params import SamplingParams


class FakeNetwork:
    def __init__(self) -> None:
        self.payloads: list[object] = []

    def simulate_transfer(self, payload: object) -> None:
        self.payloads.append(payload)


@dataclass
class FakeVerifierService:
    open_calls: list[object]
    verify_calls: list[object]
    close_calls: list[str]

    def open_session(self, request):
        self.open_calls.append(request)
        return OpenSessionResponse(req_id=request.req_id, bootstrap_token_id=23)

    def verify_round(self, request):
        self.verify_calls.append(request)
        return VerifyRoundResponse(
            req_id=request.req_id,
            accepted_len=1,
            bonus_token_id=13,
        )

    def close_session(self, req_id: str):
        self.close_calls.append(req_id)
        return CloseSessionAck(req_id=req_id)


def make_transport():
    from vllm.dssd.transport.verifier_transport import InProcessVerifierTransport

    service = FakeVerifierService(open_calls=[], verify_calls=[], close_calls=[])
    request_network = FakeNetwork()
    response_network = FakeNetwork()
    transport = InProcessVerifierTransport(
        verifier_service=service,
        request_network=request_network,
        response_network=response_network,
    )
    return transport, service, request_network, response_network


def test_verifier_transport_routes_open_session_through_network_and_service() -> None:
    transport, service, request_network, response_network = make_transport()
    sampling_params = SamplingParams(max_tokens=8)

    result = transport.open_session(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
    )

    assert result.bootstrap_token_id == 23
    assert len(service.open_calls) == 1
    assert len(request_network.payloads) == 1
    assert len(response_network.payloads) == 1


def test_verifier_transport_forwards_verify_round_request_object() -> None:
    transport, service, request_network, response_network = make_transport()
    request = VerifyRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[10],
        draft_q_values=[0.7],
    )

    response = transport.verify_round(request)

    assert response.accepted_len == 1
    assert service.verify_calls == [request]
    assert request_network.payloads == [request]
    assert response_network.payloads == [response]


def test_verifier_transport_close_session_returns_ack() -> None:
    transport, service, request_network, response_network = make_transport()

    ack = transport.close_session("req-1")

    assert ack == CloseSessionAck(req_id="req-1")
    assert service.close_calls == ["req-1"]
    assert len(request_network.payloads) == 1
    assert len(response_network.payloads) == 1


def test_verifier_transport_can_remap_remote_req_ids() -> None:
    from vllm.dssd.transport.verifier_transport import InProcessVerifierTransport

    service = FakeVerifierService(open_calls=[], verify_calls=[], close_calls=[])
    transport = InProcessVerifierTransport(
        verifier_service=service,
        request_network=FakeNetwork(),
        response_network=FakeNetwork(),
        remote_req_id_factory=lambda req_id: f"{req_id}::verifier",
    )

    opened = transport.open_session(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
    )
    response = transport.verify_round(
        VerifyRoundRequest(
            req_id="req-1",
            committed_token_id=9,
            draft_token_ids=[10],
            draft_q_values=[0.7],
        )
    )
    ack = transport.close_session("req-1")

    assert service.open_calls[0].req_id == "req-1::verifier"
    assert opened.req_id == "req-1::verifier"
    assert service.verify_calls[0].req_id == "req-1::verifier"
    assert response.req_id == "req-1::verifier"
    assert service.close_calls == ["req-1::verifier"]
    assert ack == CloseSessionAck(req_id="req-1::verifier")
