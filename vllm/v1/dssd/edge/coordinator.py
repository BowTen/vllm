# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vllm.v1.dssd.protocol import (
    BindVerifierRequest,
    BindVerifierResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    VerifyRoundResponse,
)

if TYPE_CHECKING:
    from fastapi import Request

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


class DSSDRoundCoordinator:
    def __init__(self, edge_engine, transport, *, gamma: int = 1) -> None:
        self.edge_engine = edge_engine
        self.transport = transport
        self.gamma = gamma
        self._binding: BindVerifierResponse | None = None

    async def create_chat_completion(
        self,
        *,
        request: "ChatCompletionRequest",
        raw_request: "Request | None",
        serving: "OpenAIServingChat",
    ) -> "ChatCompletionResponse | ErrorResponse":
        if self.transport is None:
            return serving.create_error_response(
                "DSSD edge mode requires a verifier transport",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        if request.stream:
            return serving.create_error_response(
                "DSSD edge streaming is not implemented yet",
                status_code=HTTPStatus.NOT_IMPLEMENTED,
            )

        result = await serving.render_chat_request(request)
        if hasattr(result, "error"):
            return result

        _conversation, engine_prompts = result
        if len(engine_prompts) != 1:
            return serving.create_error_response(
                "DSSD edge mode only supports a single rendered prompt",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        prompt_token_ids = self._extract_prompt_token_ids(engine_prompts[0])
        if prompt_token_ids is None:
            return serving.create_error_response(
                "DSSD edge mode could not extract prompt token ids",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        binding = await self._ensure_binding(request)
        await self.transport.create_session(
            CreateSessionRequest(
                binding_id=binding.binding_id,
                request_id=self._request_id(request, raw_request),
                prompt_token_ids=prompt_token_ids,
                sampling_params_digest=self._sampling_params_digest(request),
                max_new_tokens=self._max_new_tokens(request),
                stop_token_ids=list(getattr(request, "stop_token_ids", []) or []),
            )
        )
        return serving.create_error_response(
            "DSSD edge round loop is not wired yet",
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )

    async def _ensure_binding(
        self, request: "ChatCompletionRequest"
    ) -> BindVerifierResponse:
        if self._binding is not None:
            return self._binding
        assert self.transport is not None
        bind_request = BindVerifierRequest(
            protocol_version="v1alpha1",
            edge_instance_id=str(getattr(request, "user", None) or "edge-local"),
            tokenizer_hash=self._tokenizer_hash(),
            vocab_hash=self._vocab_hash(),
            supported_gamma_max=self.gamma,
        )
        self._binding = await self.transport.bind_verifier(bind_request)
        return self._binding

    def _build_committed_tokens(
        self,
        draft_token_ids: list[int],
        response: VerifyRoundResponse,
        *,
        resampled_token: int | None,
    ) -> list[int]:
        committed = list(draft_token_ids[: response.accepted_count])
        if response.all_accepted:
            if response.bonus_token_id is not None:
                committed.append(response.bonus_token_id)
            return committed
        if resampled_token is not None:
            committed.append(resampled_token)
        return committed

    def _request_id(
        self,
        request: "ChatCompletionRequest",
        raw_request: "Request | None",
    ) -> str:
        user_request_id = getattr(request, "request_id", None)
        if user_request_id:
            return str(user_request_id)
        if raw_request is not None:
            state = getattr(raw_request, "state", None)
            metadata = getattr(state, "request_metadata", None)
            request_id = getattr(metadata, "request_id", None)
            if request_id:
                return request_id
        return f"dssd-{uuid4().hex}"

    def _sampling_params_digest(self, request: "ChatCompletionRequest") -> str:
        relevant = {
            "temperature": getattr(request, "temperature", None),
            "top_p": getattr(request, "top_p", None),
            "top_k": getattr(request, "top_k", None),
            "max_tokens": self._max_new_tokens(request),
            "stop_token_ids": list(getattr(request, "stop_token_ids", []) or []),
        }
        encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _max_new_tokens(self, request: "ChatCompletionRequest") -> int | None:
        return getattr(request, "max_completion_tokens", None) or getattr(
            request, "max_tokens", None
        )

    def _extract_prompt_token_ids(self, engine_prompt: Any) -> list[int] | None:
        if isinstance(engine_prompt, dict):
            prompt_token_ids = engine_prompt.get("prompt_token_ids")
        else:
            prompt_token_ids = getattr(engine_prompt, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return None
        return list(prompt_token_ids)

    def _tokenizer_hash(self) -> str:
        return "unavailable"

    def _vocab_hash(self) -> str:
        return "unavailable"
