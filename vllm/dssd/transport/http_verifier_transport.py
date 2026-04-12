from __future__ import annotations

from collections.abc import Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from vllm.dssd.protocol import (
    CloseSessionRequest,
    OpenSessionRequest,
)

from .http_utils import (
    close_session_ack_from_payload,
    close_session_request_to_payload,
    dump_json,
    load_json,
    open_session_request_to_payload,
    open_session_response_from_payload,
    verify_round_request_to_payload,
    verify_round_response_from_payload,
)


class HTTPVerifierTransport:
    def __init__(
        self,
        *,
        server_url: str,
        request_network=None,
        response_network=None,
        remote_req_id_factory: Callable[[str], str] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.request_network = request_network
        self.response_network = response_network
        self.remote_req_id_factory = remote_req_id_factory or (lambda req_id: req_id)
        self.timeout_s = float(timeout_s)
        self._remote_req_ids: dict[str, str] = {}
        self._opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))

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
        if self.request_network is not None:
            self.request_network.simulate_transfer(request)
        payload = self._post(
            path="/open_session",
            payload=open_session_request_to_payload(request),
        )
        response = open_session_response_from_payload(payload)
        if self.response_network is not None:
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
        if self.request_network is not None:
            self.request_network.simulate_transfer(request)
        payload = self._post(
            path="/verify_round",
            payload=verify_round_request_to_payload(request),
        )
        response = verify_round_response_from_payload(payload)
        if self.response_network is not None:
            self.response_network.simulate_transfer(response)
        return response

    def close_session(self, req_id: str):
        remote_req_id = self._remote_req_ids.pop(req_id, req_id)
        request = CloseSessionRequest(req_id=remote_req_id)
        if self.request_network is not None:
            self.request_network.simulate_transfer(request)
        payload = self._post(
            path="/close_session",
            payload=close_session_request_to_payload(request),
        )
        response = close_session_ack_from_payload(payload)
        if self.response_network is not None:
            self.response_network.simulate_transfer(response)
        return response

    def _post(self, *, path: str, payload: dict) -> dict:
        http_request = urllib_request.Request(
            url=f"{self.server_url}{path}",
            data=dump_json(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(http_request,
                                   timeout=self.timeout_s) as response:
                return load_json(response.read())
        except urllib_error.HTTPError as exc:
            error_payload = load_json(exc.read())
            detail = error_payload.get("error", str(exc))
            raise RuntimeError(
                f"verifier request failed on {path}: {detail}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                f"verifier request failed on {path}: {exc.reason}"
            ) from exc
