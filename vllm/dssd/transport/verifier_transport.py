from __future__ import annotations

from collections.abc import Callable

from vllm.dssd.protocol import OpenSessionRequest


class InProcessVerifierTransport:
    def __init__(
        self,
        *,
        verifier_service,
        request_network,
        response_network,
        remote_req_id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self.verifier_service = verifier_service
        self.request_network = request_network
        self.response_network = response_network
        self.remote_req_id_factory = remote_req_id_factory or (lambda req_id: req_id)
        self._remote_req_ids: dict[str, str] = {}

    def open_session(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ):
        remote_req_id = self.remote_req_id_factory(req_id)
        self._remote_req_ids[req_id] = remote_req_id
        request = OpenSessionRequest(
            req_id=remote_req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        self.request_network.simulate_transfer(request)
        response = self.verifier_service.open_session(request)
        self.response_network.simulate_transfer(response)
        return response

    def verify_round(self, request):
        remote_req_id = self._remote_req_ids.get(request.req_id, request.req_id)
        if remote_req_id != request.req_id:
            request = type(request)(
                req_id=remote_req_id,
                committed_token_id=request.committed_token_id,
                draft_token_ids=list(request.draft_token_ids),
                draft_q_values=list(request.draft_q_values),
            )
        self.request_network.simulate_transfer(request)
        response = self.verifier_service.verify_round(request)
        self.response_network.simulate_transfer(response)
        return response

    def close_session(self, req_id: str):
        remote_req_id = self._remote_req_ids.pop(req_id, req_id)
        self.request_network.simulate_transfer(remote_req_id)
        response = self.verifier_service.close_session(remote_req_id)
        self.response_network.simulate_transfer(response)
        return response
