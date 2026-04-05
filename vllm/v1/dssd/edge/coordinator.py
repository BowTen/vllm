# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import time
from contextlib import suppress
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch

from vllm.v1.dssd.protocol import (
    BindVerifierRequest,
    BindVerifierResponse,
    CloseSessionRequest,
    CreateSessionRequest,
    DraftRoundRequest,
    VerifyRoundRequest,
    VerifyRoundResponse,
)
from vllm.v1.dssd.tokenizer_utils import compute_tokenizer_fingerprints
from vllm.v1.dssd.worker.resample import select_residual_token

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

        if not hasattr(self.edge_engine, "dssd_draft_round_async"):
            return serving.create_error_response(
                "DSSD edge draft_round utility is not available",
                status_code=HTTPStatus.NOT_IMPLEMENTED,
            )

        request_id = self._request_id(request, raw_request)
        try:
            binding = await self._ensure_binding(request)
        except ValueError as exc:
            return serving.create_error_response(
                f"DSSD verifier binding rejected: {exc}",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        session_response = await self.transport.create_session(
            CreateSessionRequest(
                binding_id=binding.binding_id,
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params_digest=self._sampling_params_digest(request),
                max_new_tokens=self._max_new_tokens(request),
                stop_token_ids=list(getattr(request, "stop_token_ids", []) or []),
            )
        )
        try:
            local_session_id = f"edge-{request_id}"
            draft_result = await self.edge_engine.dssd_draft_round_async(
                DraftRoundRequest(
                    local_session_id=local_session_id,
                    prompt_token_ids=prompt_token_ids,
                    committed_token_ids=[],
                    seq_no=0,
                    gamma=self.gamma,
                )
            )
            verify_response = await self.transport.verify_round(
                VerifyRoundRequest(
                    binding_id=binding.binding_id,
                    verifier_session_id=session_response.verifier_session_id,
                    seq_no=0,
                    prefix_delta_token_ids=[],
                    draft_token_ids=list(draft_result.draft_token_ids),
                    q_values=list(draft_result.q_values),
                )
            )
            resampled_token = None
            if not verify_response.all_accepted:
                try:
                    resampled_token = self._resample_reject_token(
                        verify_response=verify_response,
                        q_distributions=getattr(draft_result, "q_distributions", None),
                    )
                except ValueError as exc:
                    return serving.create_error_response(
                        f"DSSD edge resample failed: {exc}",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
            committed_token_ids = self._build_committed_tokens(
                list(draft_result.draft_token_ids),
                verify_response,
                resampled_token=resampled_token,
            )
            committed_token_ids, finish_reason = self._apply_response_constraints(
                request=request,
                verify_response=verify_response,
                committed_token_ids=committed_token_ids,
            )
            return self._build_chat_response(
                request_id=request_id,
                request=request,
                serving=serving,
                prompt_token_ids=prompt_token_ids,
                committed_token_ids=committed_token_ids,
                finish_reason=finish_reason,
            )
        finally:
            with suppress(Exception):
                await self.transport.close_session(
                    CloseSessionRequest(
                        verifier_session_id=session_response.verifier_session_id,
                        reason="edge_request_finished",
                    )
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
        binding = await self.transport.bind_verifier(bind_request)
        if binding.protocol_version != bind_request.protocol_version:
            raise ValueError("verifier protocol_version mismatch")
        if binding.tokenizer_hash != bind_request.tokenizer_hash:
            raise ValueError("verifier tokenizer_hash mismatch")
        if binding.vocab_hash != bind_request.vocab_hash:
            raise ValueError("verifier vocab_hash mismatch")
        if binding.supported_gamma_max < self.gamma:
            raise ValueError("verifier supported_gamma_max is smaller than edge gamma")
        self._binding = binding
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

    def _resample_reject_token(
        self,
        *,
        verify_response: VerifyRoundResponse,
        q_distributions: list[list[float]] | None,
    ) -> int:
        if verify_response.reject_target_probs is None:
            raise ValueError("reject_target_probs is missing")
        if q_distributions is None:
            raise ValueError("q_distributions is missing")
        reject_index = verify_response.reject_index
        if reject_index is None:
            reject_index = verify_response.accepted_count
        if reject_index < 0 or reject_index >= len(q_distributions):
            raise ValueError("reject_index is out of bounds")
        p = torch.tensor(verify_response.reject_target_probs, dtype=torch.float32)
        q = torch.tensor(q_distributions[reject_index], dtype=torch.float32)
        return select_residual_token(p, q)

    def _apply_response_constraints(
        self,
        *,
        request: "ChatCompletionRequest",
        verify_response: VerifyRoundResponse,
        committed_token_ids: list[int],
    ) -> tuple[list[int], str]:
        output_token_ids = list(committed_token_ids)
        finish_reason = verify_response.finish_reason
        stop_token_ids = set(getattr(request, "stop_token_ids", []) or [])
        if stop_token_ids:
            for index, token_id in enumerate(output_token_ids):
                if token_id in stop_token_ids:
                    output_token_ids = output_token_ids[:index]
                    finish_reason = "stop"
                    break
        max_new_tokens = self._max_new_tokens(request)
        if max_new_tokens is not None and len(output_token_ids) > max_new_tokens:
            output_token_ids = output_token_ids[:max_new_tokens]
            finish_reason = "length"
        if finish_reason is None:
            finish_reason = "stop" if verify_response.finished else "length"
        return output_token_ids, finish_reason

    def _build_chat_response(
        self,
        *,
        request_id: str,
        request: "ChatCompletionRequest",
        serving: "OpenAIServingChat",
        prompt_token_ids: list[int],
        committed_token_ids: list[int],
        finish_reason: str,
    ):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
        )
        from vllm.entrypoints.openai.engine.protocol import UsageInfo

        tokenizer = self._get_tokenizer(serving)
        content = tokenizer.decode(
            committed_token_ids,
            skip_special_tokens=bool(getattr(request, "skip_special_tokens", True)),
        )
        usage = UsageInfo(
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(committed_token_ids),
            total_tokens=len(prompt_token_ids) + len(committed_token_ids),
        )
        model_name = self._model_name(serving)
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    def _get_tokenizer(self, serving: "OpenAIServingChat"):
        renderer = getattr(serving, "renderer", None)
        if renderer is None:
            raise ValueError("DSSD edge serving requires a renderer tokenizer")
        tokenizer_getter = getattr(renderer, "get_tokenizer", None)
        if callable(tokenizer_getter):
            return tokenizer_getter()
        tokenizer = getattr(renderer, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("DSSD edge serving requires a renderer tokenizer")
        return tokenizer

    def _model_name(self, serving: "OpenAIServingChat") -> str:
        models = getattr(serving, "models", None)
        if models is not None and hasattr(models, "model_name"):
            return models.model_name(None)
        return "dssd-edge"

    def _tokenizer_hash(self) -> str:
        tokenizer = self._tokenizer_for_fingerprint()
        tokenizer_hash, _ = compute_tokenizer_fingerprints(tokenizer)
        return tokenizer_hash

    def _vocab_hash(self) -> str:
        tokenizer = self._tokenizer_for_fingerprint()
        _, vocab_hash = compute_tokenizer_fingerprints(tokenizer)
        return vocab_hash

    def _tokenizer_for_fingerprint(self):
        renderer = getattr(self.edge_engine, "renderer", None)
        if renderer is None:
            return None
        tokenizer_getter = getattr(renderer, "get_tokenizer", None)
        if callable(tokenizer_getter):
            return tokenizer_getter()
        return getattr(renderer, "tokenizer", None)
