# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import time
from contextlib import suppress
from http import HTTPStatus
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator
from uuid import uuid4

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.dssd.protocol import (
    BindVerifierRequest,
    BindVerifierResponse,
    CloseSessionRequest,
    CreateSessionRequest,
    DraftRoundRequest,
    VerifyRoundRequest,
    VerifyRoundResponse,
)
from vllm.v1.dssd.edge.session import DSSDEdgeSessionState
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


@dataclass
class _PreparedChatCompletion:
    request_id: str
    prompt_token_ids: list[int]
    resolved_sampling_params: SamplingParams
    binding: BindVerifierResponse


@dataclass
class VerifiedRoundDelta:
    seq_no: int
    delta_token_ids: list[int]
    delta_text: str
    finished: bool
    finish_reason: str | None


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
            return self.create_chat_completion_stream(
                request=request,
                raw_request=raw_request,
                serving=serving,
            )

        prepared = await self._prepare_chat_completion(request, raw_request, serving)
        if self._is_error_response(prepared):
            return prepared

        try:
            committed_token_ids, finish_reason = (
                await self._collect_verified_completion(
                    request=request,
                    raw_request=raw_request,
                    serving=serving,
                    prompt_token_ids=prepared.prompt_token_ids,
                    resolved_sampling_params=prepared.resolved_sampling_params,
                    request_id=prepared.request_id,
                    binding=prepared.binding,
                )
            )
        except ValueError as exc:
            return serving.create_error_response(
                str(exc),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        return self._build_chat_response(
            request_id=prepared.request_id,
            request=request,
            serving=serving,
            prompt_token_ids=prepared.prompt_token_ids,
            committed_token_ids=committed_token_ids,
            finish_reason=finish_reason,
        )

    def create_chat_completion_stream(
        self,
        *,
        request: "ChatCompletionRequest",
        raw_request: "Request | None",
        serving: "OpenAIServingChat",
    ):
        return self._create_chat_completion_stream_generator(
            request=request,
            raw_request=raw_request,
            serving=serving,
        )

    async def _prepare_chat_completion(
        self,
        request: "ChatCompletionRequest",
        raw_request: "Request | None",
        serving: "OpenAIServingChat",
    ) -> _PreparedChatCompletion | "ErrorResponse":
        if self.transport is None:
            return serving.create_error_response(
                "DSSD edge mode requires a verifier transport",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
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

        resolved_sampling_params = self._resolve_sampling_params(
            request=request,
            serving=serving,
            prompt_len=len(prompt_token_ids),
        )
        if resolved_sampling_params is None:
            return serving.create_error_response(
                "DSSD edge mode requires resolvable sampling params",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        request_id = self._request_id(request, raw_request)
        try:
            binding = await self._ensure_binding(request)
        except ValueError as exc:
            return serving.create_error_response(
                f"DSSD verifier binding rejected: {exc}",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        return _PreparedChatCompletion(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            resolved_sampling_params=resolved_sampling_params,
            binding=binding,
        )

    async def _collect_verified_completion(
        self,
        *,
        request: "ChatCompletionRequest",
        raw_request: "Request | None",
        serving: "OpenAIServingChat",
        prompt_token_ids: list[int],
        resolved_sampling_params: SamplingParams,
        request_id: str,
        binding: BindVerifierResponse,
    ) -> tuple[list[int], str]:
        committed_token_ids: list[int] = []
        finish_reason = "length"
        async for delta in self._iter_verified_round_deltas(
            request=request,
            raw_request=raw_request,
            serving=serving,
            prompt_token_ids=prompt_token_ids,
            resolved_sampling_params=resolved_sampling_params,
            request_id=request_id,
            binding=binding,
        ):
            committed_token_ids.extend(delta.delta_token_ids)
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
                break
        return committed_token_ids, finish_reason

    async def _iter_verified_round_deltas(
        self,
        *,
        request: "ChatCompletionRequest",
        raw_request: "Request | None",
        serving: "OpenAIServingChat",
        prompt_token_ids: list[int],
        resolved_sampling_params: SamplingParams,
        request_id: str,
        binding: BindVerifierResponse,
    ) -> AsyncIterator[VerifiedRoundDelta]:
        tokenizer = self._get_tokenizer(serving)
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
        session_state = DSSDEdgeSessionState(
            request_id=request_id,
            local_session_id=f"edge-{request_id}",
            verifier_binding_id=binding.binding_id,
            verifier_session_id=session_response.verifier_session_id,
            prompt_token_ids=prompt_token_ids,
        )
        visible_token_ids: list[int] = []
        try:
            while True:
                draft_result = await self.edge_engine.dssd_draft_round_async(
                    DraftRoundRequest(
                        local_session_id=session_state.local_session_id,
                        prompt_token_ids=session_state.prompt_token_ids,
                        committed_token_ids=list(session_state.committed_token_ids),
                        seq_no=session_state.seq_no,
                        gamma=self.gamma,
                        sampling_params=resolved_sampling_params,
                    )
                )
                verify_response = await self.transport.verify_round(
                    VerifyRoundRequest(
                        binding_id=session_state.verifier_binding_id,
                        verifier_session_id=session_state.verifier_session_id,
                        seq_no=session_state.seq_no,
                        prefix_delta_token_ids=list(
                            session_state.pending_prefix_delta_token_ids
                        ),
                        draft_token_ids=list(draft_result.draft_token_ids),
                        q_values=list(draft_result.q_values),
                    )
                )
                session_state.pending_prefix_delta_token_ids.clear()
                if verify_response.seq_no != session_state.seq_no:
                    raise ValueError(
                        "DSSD edge verify_round returned mismatched seq_no"
                    )

                resampled_token = None
                if not verify_response.all_accepted:
                    try:
                        resampled_token = self._resample_reject_token(
                            verify_response=verify_response,
                            q_distributions=getattr(
                                draft_result, "q_distributions", None
                            ),
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"DSSD edge resample failed: {exc}"
                        ) from exc

                round_committed_token_ids = self._build_committed_tokens(
                    list(draft_result.draft_token_ids),
                    verify_response,
                    resampled_token=resampled_token,
                )
                session_state.committed_token_ids.extend(round_committed_token_ids)
                if not verify_response.all_accepted and resampled_token is not None:
                    session_state.pending_prefix_delta_token_ids.append(
                        resampled_token
                    )
                if not round_committed_token_ids and not verify_response.finished:
                    raise ValueError("DSSD edge encountered zero-progress round")

                should_stop, committed_token_ids, finish_reason = (
                    self._apply_response_constraints(
                        request=request,
                        verify_response=verify_response,
                        committed_token_ids=session_state.committed_token_ids,
                    )
                )
                delta_token_ids = committed_token_ids[len(visible_token_ids) :]
                visible_token_ids = committed_token_ids
                yield VerifiedRoundDelta(
                    seq_no=session_state.seq_no,
                    delta_token_ids=delta_token_ids,
                    delta_text=tokenizer.decode(
                        delta_token_ids,
                        skip_special_tokens=bool(
                            getattr(request, "skip_special_tokens", True)
                        ),
                    ),
                    finished=should_stop,
                    finish_reason=finish_reason if should_stop else None,
                )
                if should_stop:
                    break
                session_state.seq_no += 1
        finally:
            with suppress(Exception):
                await self.transport.close_session(
                    CloseSessionRequest(
                        verifier_session_id=session_response.verifier_session_id,
                        reason="edge_request_finished",
                    )
                )

    async def _create_chat_completion_stream_generator(
        self,
        *,
        request: "ChatCompletionRequest",
        raw_request: "Request | None",
        serving: "OpenAIServingChat",
    ):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponseStreamChoice,
            ChatCompletionStreamResponse,
        )
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage, UsageInfo
        from vllm.entrypoints.utils import should_include_usage

        prepared = await self._prepare_chat_completion(request, raw_request, serving)
        if self._is_error_response(prepared):
            yield (
                "data: "
                f"{self._create_streaming_error_response(serving, prepared)}\n\n"
            )
            yield "data: [DONE]\n\n"
            return

        model_name = self._model_name(serving)
        created = int(time.time())
        role = getattr(serving, "response_role", "assistant")
        include_usage, _ = should_include_usage(
            getattr(request, "stream_options", None),
            getattr(serving, "enable_force_include_usage", False),
        )
        try:
            first_choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role=role, content=""),
                logprobs=None,
                finish_reason=None,
            )
            first_chunk = ChatCompletionStreamResponse(
                id=prepared.request_id,
                created=created,
                model=model_name,
                choices=[first_choice],
                prompt_token_ids=(
                    prepared.prompt_token_ids
                    if getattr(request, "return_token_ids", False)
                    else None
                ),
            )
            yield f"data: {first_chunk.model_dump_json(exclude_unset=True)}\n\n"

            committed_token_ids: list[int] = []
            async for delta in self._iter_verified_round_deltas(
                request=request,
                raw_request=raw_request,
                serving=serving,
                prompt_token_ids=prepared.prompt_token_ids,
                resolved_sampling_params=prepared.resolved_sampling_params,
                request_id=prepared.request_id,
                binding=prepared.binding,
            ):
                committed_token_ids.extend(delta.delta_token_ids)
                if delta.delta_token_ids or delta.delta_text:
                    choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            content=delta.delta_text or None,
                        ),
                        logprobs=None,
                        finish_reason=None,
                        token_ids=(
                            delta.delta_token_ids
                            if getattr(request, "return_token_ids", False)
                            else None
                        ),
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=prepared.request_id,
                        created=created,
                        model=model_name,
                        choices=[choice],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                if delta.finished:
                    finish_choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        logprobs=None,
                        finish_reason=delta.finish_reason or "stop",
                    )
                    finish_chunk = ChatCompletionStreamResponse(
                        id=prepared.request_id,
                        created=created,
                        model=model_name,
                        choices=[finish_choice],
                    )
                    yield f"data: {finish_chunk.model_dump_json(exclude_unset=True)}\n\n"
                    break

            if include_usage:
                usage_chunk = ChatCompletionStreamResponse(
                    id=prepared.request_id,
                    created=created,
                    model=model_name,
                    choices=[],
                    usage=UsageInfo(
                        prompt_tokens=len(prepared.prompt_token_ids),
                        completion_tokens=len(committed_token_ids),
                        total_tokens=(
                            len(prepared.prompt_token_ids) + len(committed_token_ids)
                        ),
                    ),
                )
                yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"
        except ValueError as exc:
            yield (
                "data: "
                f"{self._create_streaming_error_response(serving, exc, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)}\n\n"
            )
        except Exception as exc:
            yield (
                "data: "
                f"{self._create_streaming_error_response(serving, exc, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)}\n\n"
            )
        yield "data: [DONE]\n\n"

    @staticmethod
    def _is_error_response(value: Any) -> bool:
        return hasattr(value, "error") or (
            isinstance(value, dict) and "message" in value
        )

    @staticmethod
    def _error_message(value: Any) -> str:
        if hasattr(value, "error"):
            error = getattr(value, "error")
            return str(getattr(error, "message", error))
        if isinstance(value, dict):
            return str(value.get("message", value))
        return str(value)

    def _create_streaming_error_response(
        self,
        serving: "OpenAIServingChat",
        value: Any,
        *,
        status_code: HTTPStatus | None = None,
    ) -> str:
        resolved_status = status_code
        if resolved_status is None:
            if hasattr(value, "error"):
                resolved_status = HTTPStatus(getattr(value.error, "code"))
            elif isinstance(value, dict) and "status_code" in value:
                resolved_status = HTTPStatus(value["status_code"])
            else:
                resolved_status = HTTPStatus.BAD_REQUEST
        return serving.create_streaming_error_response(
            self._error_message(value),
            status_code=resolved_status,
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

    def _resolve_sampling_params(
        self,
        *,
        request: "ChatCompletionRequest",
        serving: "OpenAIServingChat",
        prompt_len: int,
    ) -> SamplingParams | None:
        to_sampling_params = getattr(request, "to_sampling_params", None)
        default_sampling_params = getattr(serving, "default_sampling_params", None)
        if not callable(to_sampling_params) or default_sampling_params is None:
            return None

        max_tokens = self._resolve_sampling_max_tokens(
            request=request,
            serving=serving,
            prompt_len=prompt_len,
        )
        if max_tokens is None:
            return None
        return to_sampling_params(max_tokens, default_sampling_params)

    def _resolve_sampling_max_tokens(
        self,
        *,
        request: "ChatCompletionRequest",
        serving: "OpenAIServingChat",
        prompt_len: int,
    ) -> int | None:
        model_config = getattr(serving, "model_config", None)
        default_sampling_params = getattr(serving, "default_sampling_params", None)
        if model_config is not None and default_sampling_params is not None:
            max_model_len = getattr(model_config, "max_model_len", None)
            if max_model_len is not None:
                with suppress(
                    AttributeError,
                    ImportError,
                    TypeError,
                    ValueError,
                ):
                    from vllm.entrypoints.utils import get_max_tokens

                    return get_max_tokens(
                        max_model_len,
                        getattr(request, "max_completion_tokens", None)
                        if getattr(request, "max_completion_tokens", None)
                        is not None
                        else getattr(request, "max_tokens", None),
                        prompt_len,
                        default_sampling_params,
                        getattr(serving, "override_max_tokens", None),
                    )
                requested_max_tokens = getattr(request, "max_completion_tokens", None)
                if requested_max_tokens is None:
                    requested_max_tokens = getattr(request, "max_tokens", None)
                fallback_max_tokens = (
                    requested_max_tokens
                    if requested_max_tokens is not None
                    else default_sampling_params.get("max_tokens")
                )
                override_max_tokens = getattr(serving, "override_max_tokens", None)
                return min(
                    value
                    for value in (
                        max_model_len - prompt_len,
                        fallback_max_tokens,
                        override_max_tokens,
                    )
                    if value is not None
                )
        return self._max_new_tokens(request)

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
    ) -> tuple[bool, list[int], str]:
        output_token_ids = list(committed_token_ids)
        stop_token_ids = set(getattr(request, "stop_token_ids", []) or [])
        if stop_token_ids:
            for index, token_id in enumerate(output_token_ids):
                if token_id in stop_token_ids:
                    output_token_ids = output_token_ids[:index]
                    return True, output_token_ids, "stop"
        max_new_tokens = self._max_new_tokens(request)
        if max_new_tokens is not None:
            if len(output_token_ids) > max_new_tokens:
                output_token_ids = output_token_ids[:max_new_tokens]
                return True, output_token_ids, "length"
            if len(output_token_ids) == max_new_tokens:
                finish_reason = (
                    verify_response.finish_reason or "stop"
                    if verify_response.finished
                    else "length"
                )
                return True, output_token_ids, finish_reason
        if verify_response.finished:
            return True, output_token_ids, verify_response.finish_reason or "stop"
        return False, output_token_ids, "length"

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
