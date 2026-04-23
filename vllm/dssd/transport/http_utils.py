from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    import msgspec

    return {
        "req_id": request.req_id,
        "prompt_token_ids": list(request.prompt_token_ids),
        "sampling_params": msgspec.to_builtins(request.sampling_params),
        "lora_request": (
            None if request.lora_request is None else msgspec.to_builtins(request.lora_request)
        ),
    }


def open_session_request_from_payload(payload: dict[str, Any]) -> OpenSessionRequest:
    import msgspec

    from vllm.dssd.protocol import OpenSessionRequest
    from vllm.sampling_params import SamplingParams

    return OpenSessionRequest(
        req_id=payload["req_id"],
        prompt_token_ids=list(payload["prompt_token_ids"]),
        sampling_params=msgspec.convert(
            payload["sampling_params"],
            type=SamplingParams,
        ),
        lora_request=_decode_lora_request(payload.get("lora_request")),
    )


def edge_generate_request_to_payload(
    *,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> dict[str, Any]:
    import msgspec

    return {
        "req_id": req_id,
        "prompt_token_ids": list(prompt_token_ids),
        "sampling_params": msgspec.to_builtins(sampling_params),
        "lora_request": (
            None if lora_request is None else msgspec.to_builtins(lora_request)
        ),
    }


def edge_generate_request_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import msgspec

    from vllm.sampling_params import SamplingParams

    return {
        "req_id": payload["req_id"],
        "prompt_token_ids": list(payload["prompt_token_ids"]),
        "sampling_params": msgspec.convert(
            payload["sampling_params"],
            type=SamplingParams,
        ),
        "lora_request": _decode_lora_request(payload.get("lora_request")),
    }


def edge_generate_response_to_payload(
    *,
    req_id: str,
    output_ids: list[int],
) -> dict[str, Any]:
    return {
        "req_id": req_id,
        "output_ids": list(output_ids),
    }


def edge_generate_response_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "req_id": payload["req_id"],
        "output_ids": list(payload["output_ids"]),
    }


def edge_complete_request_to_payload(
    *,
    req_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> dict[str, Any]:
    import msgspec

    return {
        "req_id": req_id,
        "prompt": prompt,
        "sampling_params": msgspec.to_builtins(sampling_params),
        "lora_request": (
            None if lora_request is None else msgspec.to_builtins(lora_request)
        ),
    }


def edge_complete_request_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import msgspec

    from vllm.sampling_params import SamplingParams

    return {
        "req_id": payload["req_id"],
        "prompt": payload["prompt"],
        "sampling_params": msgspec.convert(
            payload["sampling_params"],
            type=SamplingParams,
        ),
        "lora_request": _decode_lora_request(payload.get("lora_request")),
    }


def edge_complete_response_to_payload(
    *,
    req_id: str,
    text: str,
) -> dict[str, Any]:
    return {
        "req_id": req_id,
        "text": text,
    }


def edge_complete_response_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "req_id": payload["req_id"],
        "text": payload["text"],
    }


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
    from vllm.dssd.protocol import OpenSessionResponse

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
    from vllm.dssd.protocol import VerifyRoundRequest

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
    from vllm.dssd.protocol import VerifyRoundResponse

    return VerifyRoundResponse(
        req_id=payload["req_id"],
        accepted_len=payload["accepted_len"],
        bonus_token_id=payload.get("bonus_token_id"),
        rejected_target_logits=_decode_tensor(payload.get("rejected_target_logits")),
    )


def verify_round_response_to_http_payload(
    response: VerifyRoundResponse,
) -> tuple[str, bytes]:
    if response.rejected_target_logits is None:
        return (
            "application/json",
            dump_json(verify_round_response_to_payload(response)),
        )

    import torch

    logits = response.rejected_target_logits.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous()
    metadata_bytes = dump_json(
        {
            "req_id": response.req_id,
            "accepted_len": response.accepted_len,
            "bonus_token_id": None,
            "dtype": "float32",
            "shape": list(logits.shape),
        }
    )
    return (
        "application/octet-stream",
        len(metadata_bytes).to_bytes(4, byteorder="little", signed=False)
        + metadata_bytes
        + logits.numpy().tobytes(),
    )


def verify_round_response_from_http_payload(
    *,
    content_type: str,
    payload: bytes,
) -> VerifyRoundResponse:
    normalized = content_type.lower()
    if normalized.startswith("application/json"):
        return verify_round_response_from_payload(load_json(payload))
    if not normalized.startswith("application/octet-stream"):
        raise RuntimeError(
            f"unsupported verify_round content type: {content_type}"
        )
    return _decode_binary_verify_round_response(payload)


def close_session_request_to_payload(
    request: CloseSessionRequest,
) -> dict[str, Any]:
    return {"req_id": request.req_id}


def close_session_request_from_payload(
    payload: dict[str, Any],
) -> CloseSessionRequest:
    from vllm.dssd.protocol import CloseSessionRequest

    return CloseSessionRequest(req_id=payload["req_id"])


def close_session_ack_to_payload(ack: CloseSessionAck) -> dict[str, Any]:
    return {"req_id": ack.req_id}


def close_session_ack_from_payload(payload: dict[str, Any]) -> CloseSessionAck:
    from vllm.dssd.protocol import CloseSessionAck

    return CloseSessionAck(req_id=payload["req_id"])


def _decode_lora_request(payload: Any) -> LoRARequest | None:
    if payload is None:
        return None

    import msgspec

    from vllm.lora.request import LoRARequest

    return msgspec.convert(payload, type=LoRARequest)


def _encode_tensor(tensor: torch.Tensor | None) -> Any:
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


def _decode_tensor(payload: Any) -> torch.Tensor | None:
    if payload is None:
        return None

    import torch

    return torch.tensor(payload, dtype=torch.float32)


def _decode_binary_verify_round_response(payload: bytes) -> VerifyRoundResponse:
    import torch

    from vllm.dssd.protocol import VerifyRoundResponse

    if len(payload) < 4:
        raise RuntimeError(
            "malformed binary verify_round response: missing metadata length"
        )
    metadata_len = int.from_bytes(payload[:4], byteorder="little", signed=False)
    metadata_end = 4 + metadata_len
    if len(payload) < metadata_end:
        raise RuntimeError(
            "malformed binary verify_round response: truncated metadata"
        )
    try:
        metadata = load_json(payload[4:metadata_end])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"malformed binary verify_round response: invalid metadata ({exc})"
        ) from exc

    dtype = metadata.get("dtype")
    if dtype != "float32":
        raise RuntimeError(
            f"malformed binary verify_round response: unsupported dtype {dtype!r}"
        )
    shape_payload = metadata.get("shape")
    if not isinstance(shape_payload, list) or not shape_payload:
        raise RuntimeError(
            "malformed binary verify_round response: invalid shape"
        )
    try:
        shape = tuple(int(dim) for dim in shape_payload)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "malformed binary verify_round response: invalid shape"
        ) from exc
    if any(dim <= 0 for dim in shape):
        raise RuntimeError(
            "malformed binary verify_round response: invalid shape"
        )

    logits_bytes = payload[metadata_end:]
    expected_bytes = math.prod(shape) * torch.tensor([], dtype=torch.float32).element_size()
    if len(logits_bytes) != expected_bytes:
        raise RuntimeError(
            "malformed binary verify_round response: logits size mismatch"
        )

    logits = torch.frombuffer(
        bytearray(logits_bytes),
        dtype=torch.float32,
    ).clone().reshape(shape)
    try:
        return VerifyRoundResponse(
            req_id=metadata["req_id"],
            accepted_len=int(metadata["accepted_len"]),
            bonus_token_id=metadata.get("bonus_token_id"),
            rejected_target_logits=logits,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"malformed binary verify_round response: invalid metadata ({exc})"
        ) from exc
