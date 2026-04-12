from __future__ import annotations

import json
from typing import Any

import msgspec
import torch

from vllm.dssd.protocol import (
    CloseSessionAck,
    CloseSessionRequest,
    OpenSessionRequest,
    OpenSessionResponse,
    VerifyRoundRequest,
    VerifyRoundResponse,
)
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


def dump_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def load_json(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


def open_session_request_to_payload(request: OpenSessionRequest) -> dict[str, Any]:
    return {
        "req_id": request.req_id,
        "prompt_token_ids": list(request.prompt_token_ids),
        "sampling_params": msgspec.to_builtins(request.sampling_params),
        "lora_request": (
            None if request.lora_request is None else msgspec.to_builtins(request.lora_request)
        ),
    }


def open_session_request_from_payload(payload: dict[str, Any]) -> OpenSessionRequest:
    return OpenSessionRequest(
        req_id=payload["req_id"],
        prompt_token_ids=list(payload["prompt_token_ids"]),
        sampling_params=msgspec.convert(
            payload["sampling_params"],
            type=SamplingParams,
        ),
        lora_request=_decode_lora_request(payload.get("lora_request")),
    )


def open_session_response_to_payload(
    response: OpenSessionResponse,
) -> dict[str, Any]:
    return {
        "req_id": response.req_id,
        "bootstrap_token_id": response.bootstrap_token_id,
    }


def open_session_response_from_payload(
    payload: dict[str, Any],
) -> OpenSessionResponse:
    return OpenSessionResponse(
        req_id=payload["req_id"],
        bootstrap_token_id=payload["bootstrap_token_id"],
    )


def verify_round_request_to_payload(request: VerifyRoundRequest) -> dict[str, Any]:
    return {
        "req_id": request.req_id,
        "committed_token_id": request.committed_token_id,
        "draft_token_ids": list(request.draft_token_ids),
        "draft_q_values": list(request.draft_q_values),
    }


def verify_round_request_from_payload(payload: dict[str, Any]) -> VerifyRoundRequest:
    return VerifyRoundRequest(
        req_id=payload["req_id"],
        committed_token_id=payload["committed_token_id"],
        draft_token_ids=list(payload["draft_token_ids"]),
        draft_q_values=list(payload["draft_q_values"]),
    )


def verify_round_response_to_payload(
    response: VerifyRoundResponse,
) -> dict[str, Any]:
    return {
        "req_id": response.req_id,
        "accepted_len": response.accepted_len,
        "bonus_token_id": response.bonus_token_id,
        "rejected_target_logits": _encode_tensor(response.rejected_target_logits),
    }


def verify_round_response_from_payload(
    payload: dict[str, Any],
) -> VerifyRoundResponse:
    return VerifyRoundResponse(
        req_id=payload["req_id"],
        accepted_len=payload["accepted_len"],
        bonus_token_id=payload.get("bonus_token_id"),
        rejected_target_logits=_decode_tensor(payload.get("rejected_target_logits")),
    )


def close_session_request_to_payload(
    request: CloseSessionRequest,
) -> dict[str, Any]:
    return {"req_id": request.req_id}


def close_session_request_from_payload(
    payload: dict[str, Any],
) -> CloseSessionRequest:
    return CloseSessionRequest(req_id=payload["req_id"])


def close_session_ack_to_payload(ack: CloseSessionAck) -> dict[str, Any]:
    return {"req_id": ack.req_id}


def close_session_ack_from_payload(payload: dict[str, Any]) -> CloseSessionAck:
    return CloseSessionAck(req_id=payload["req_id"])


def _decode_lora_request(payload: Any) -> LoRARequest | None:
    if payload is None:
        return None
    return msgspec.convert(payload, type=LoRARequest)


def _encode_tensor(tensor: torch.Tensor | None) -> Any:
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


def _decode_tensor(payload: Any) -> torch.Tensor | None:
    if payload is None:
        return None
    return torch.tensor(payload, dtype=torch.float32)

