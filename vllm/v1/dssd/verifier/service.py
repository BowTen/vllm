# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from uuid import uuid4

import msgspec

from vllm.v1.dssd.protocol import (
    BindVerifierRequest,
    BindVerifierResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    VerifierForwardResult,
    VerifyRoundRequest,
    VerifyRoundResponse,
)
from vllm.v1.dssd.tokenizer_utils import compute_tokenizer_fingerprints
from vllm.v1.dssd.verifier.session import DSSDVerifierSessionManager


class DSSDVerifierService:
    PROTOCOL_VERSION = "v1alpha1"

    def __init__(self, engine_client, vllm_config) -> None:
        self.engine_client = engine_client
        self.vllm_config = vllm_config
        self.session_manager = DSSDVerifierSessionManager()
        self._bindings: set[str] = set()

    async def bind_verifier(
        self, request: BindVerifierRequest
    ) -> BindVerifierResponse:
        dssd_config = getattr(self.vllm_config, "dssd_config", None)
        if dssd_config is None or not dssd_config.enabled or dssd_config.role != "verifier":
            raise ValueError("DSSD verifier service is disabled")

        binding_id = f"bind-{uuid4().hex}"
        self._bindings.add(binding_id)
        supported_gamma_max = min(request.supported_gamma_max, dssd_config.gamma)
        verifier_model_id = getattr(
            getattr(self.engine_client, "model_config", None),
            "model",
            "unknown-model",
        )
        return BindVerifierResponse(
            binding_id=binding_id,
            protocol_version=self.PROTOCOL_VERSION,
            verifier_model_id=verifier_model_id,
            tokenizer_hash=self._tokenizer_hash(),
            vocab_hash=self._vocab_hash(),
            supported_gamma_max=supported_gamma_max,
            capabilities={
                "session_lifecycle": "bind/create/verify/close",
                "tokenizer_validation": "caller-supplied",
            },
        )

    async def create_session(
        self, request: CreateSessionRequest
    ) -> CreateSessionResponse:
        self._ensure_binding(request.binding_id)
        verifier_session_id = f"vs-{uuid4().hex}"
        self.session_manager.create_session(
            verifier_session_id,
            binding_id=request.binding_id,
            sampling_params_fingerprint=request.sampling_params_digest,
            prompt_token_ids=request.prompt_token_ids,
        )
        return CreateSessionResponse(
            verifier_session_id=verifier_session_id,
            accepted_prompt_len=len(request.prompt_token_ids),
            expires_at=None,
        )

    async def verify_round(
        self, request: VerifyRoundRequest
    ) -> VerifyRoundResponse:
        self._ensure_binding(request.binding_id)
        session = self.session_manager.get_session(request.verifier_session_id)
        if session.binding_id != request.binding_id:
            raise ValueError("binding_id does not match verifier session")

        cached = self.session_manager.get_cached_response(
            request.verifier_session_id,
            seq_no=request.seq_no,
        )
        if cached is not None:
            return cached

        self.session_manager.ensure_next_seq_no(
            request.verifier_session_id,
            request.seq_no,
        )

        raw_response = await self.engine_client.dssd_verify_round_async(request)
        response = self._coerce_verify_response(
            request=request,
            raw_response=raw_response,
            session=session,
        )
        self.session_manager.append_prefix_delta(
            request.verifier_session_id,
            request.prefix_delta_token_ids,
        )
        self.session_manager.cache_response(
            request.verifier_session_id,
            seq_no=request.seq_no,
            response=response,
        )
        self.session_manager.update_seq_no(
            request.verifier_session_id,
            request.seq_no,
        )
        return response

    async def close_session(
        self, request: CloseSessionRequest
    ) -> CloseSessionResponse:
        return CloseSessionResponse(
            closed=self.session_manager.delete_session(request.verifier_session_id)
        )

    def _ensure_binding(self, binding_id: str) -> None:
        if binding_id not in self._bindings:
            raise ValueError("unknown verifier binding")

    def _coerce_verify_response(
        self,
        *,
        request: VerifyRoundRequest,
        raw_response,
        session,
    ) -> VerifyRoundResponse:
        if isinstance(raw_response, VerifyRoundResponse):
            return raw_response
        if not isinstance(raw_response, VerifierForwardResult):
            raw_response = msgspec.convert(raw_response, type=VerifierForwardResult)
        return self._build_verify_response_from_forward(
            request=request,
            forward_result=raw_response,
            session=session,
        )

    def _build_verify_response_from_forward(
        self,
        *,
        request: VerifyRoundRequest,
        forward_result: VerifierForwardResult,
        session,
    ) -> VerifyRoundResponse:
        if len(forward_result.seq_probs) != len(request.draft_token_ids):
            raise ValueError("verifier forward result length does not match draft")

        accepted_count = 0
        for index, (draft_token_id, q_value, seq_probs) in enumerate(
            zip(
                request.draft_token_ids,
                request.q_values,
                forward_result.seq_probs,
                strict=True,
            )
        ):
            p_value = self._get_token_probability(seq_probs, draft_token_id)
            accept_prob = self._accept_prob(p_value, q_value)
            if session.rng.random() < accept_prob:
                accepted_count += 1
                continue
            return VerifyRoundResponse(
                verifier_session_id=forward_result.verifier_session_id,
                seq_no=forward_result.seq_no,
                accepted_count=accepted_count,
                all_accepted=False,
                reject_index=index,
                reject_target_probs=list(seq_probs),
                finished=forward_result.finished,
                finish_reason=forward_result.finish_reason,
            )

        bonus_token_id = self._sample_token_id(session.rng, forward_result.bonus_probs)
        return VerifyRoundResponse(
            verifier_session_id=forward_result.verifier_session_id,
            seq_no=forward_result.seq_no,
            accepted_count=accepted_count,
            all_accepted=True,
            bonus_token_id=bonus_token_id,
            finished=forward_result.finished,
            finish_reason=forward_result.finish_reason,
        )

    def _get_token_probability(self, seq_probs: list[float], token_id: int) -> float:
        if token_id < 0 or token_id >= len(seq_probs):
            raise ValueError("draft token id is out of bounds for verifier probs")
        return float(seq_probs[token_id])

    def _accept_prob(self, p_value: float, q_value: float) -> float:
        if q_value <= 0:
            return 1.0
        return max(0.0, min(1.0, p_value / q_value))

    def _sample_token_id(self, rng, probs: list[float]) -> int | None:
        if not probs:
            return None
        total = 0.0
        for prob in probs:
            if prob < 0:
                raise ValueError("bonus probabilities must be non-negative")
            total += prob
        if total <= 0:
            raise ValueError("bonus probabilities must have positive mass")

        threshold = rng.random() * total
        cumulative = 0.0
        for token_id, prob in enumerate(probs):
            cumulative += prob
            if threshold < cumulative:
                return token_id
        return len(probs) - 1

    def _tokenizer_hash(self) -> str:
        tokenizer = self._tokenizer_for_fingerprint()
        tokenizer_hash, _ = compute_tokenizer_fingerprints(tokenizer)
        return tokenizer_hash

    def _vocab_hash(self) -> str:
        tokenizer = self._tokenizer_for_fingerprint()
        _, vocab_hash = compute_tokenizer_fingerprints(tokenizer)
        return vocab_hash

    def _tokenizer_for_fingerprint(self):
        renderer = getattr(self.engine_client, "renderer", None)
        if renderer is None:
            return None
        tokenizer_getter = getattr(renderer, "get_tokenizer", None)
        if callable(tokenizer_getter):
            return tokenizer_getter()
        return getattr(renderer, "tokenizer", None)
