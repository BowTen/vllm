# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import msgspec

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.structured_output.backend_types import StructuredOutputKey


class SamplingMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    seed: int | None = None
    max_tokens: int | None = None
    min_tokens: int = 0
    stop_token_ids: list[int] = msgspec.field(default_factory=list)
    eos_token_id: int | None = None
    ignore_eos: bool = False
    logit_bias: dict[int, float] | None = None
    allowed_token_ids: list[int] | None = None
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    bad_words_token_ids: list[list[int]] | None = None
    structured_output_backend: str | None = None
    structured_output_type: str | None = None
    structured_output_spec: str | None = None
    structured_output_disable_any_whitespace: bool = False
    structured_output_disable_additional_properties: bool = False
    structured_output_max_rollback: int = 0

    @classmethod
    def from_sampling_params(cls, params: "SamplingParams") -> "SamplingMetadata":
        structured_output_backend = None
        structured_output_type = None
        structured_output_spec = None
        structured_output_disable_any_whitespace = False
        structured_output_disable_additional_properties = False
        if params.structured_outputs is not None and not params.structured_outputs.all_constraints_none():
            from vllm.v1.structured_output.request import get_structured_output_key

            if params.structured_outputs._backend is None:
                raise ValueError(
                    "Structured outputs backend was not resolved before "
                    "distributed speculative request creation."
                )
            key: StructuredOutputKey = get_structured_output_key(
                params.structured_outputs
            )
            structured_output_backend = params.structured_outputs._backend
            structured_output_type = key[0].name
            structured_output_spec = key[1]
            structured_output_disable_any_whitespace = (
                params.structured_outputs.disable_any_whitespace
            )
            structured_output_disable_additional_properties = (
                params.structured_outputs.disable_additional_properties
            )
        return cls(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            min_p=params.min_p,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            repetition_penalty=params.repetition_penalty,
            seed=params.seed,
            max_tokens=params.max_tokens,
            min_tokens=params.min_tokens,
            stop_token_ids=list(params.stop_token_ids or ()),
            eos_token_id=params.eos_token_id,
            ignore_eos=params.ignore_eos,
            logit_bias=params.logit_bias,
            allowed_token_ids=params.allowed_token_ids,
            logprobs=params.logprobs,
            prompt_logprobs=params.prompt_logprobs,
            bad_words_token_ids=params.bad_words_token_ids,
            structured_output_backend=structured_output_backend,
            structured_output_type=structured_output_type,
            structured_output_spec=structured_output_spec,
            structured_output_disable_any_whitespace=(
                structured_output_disable_any_whitespace
            ),
            structured_output_disable_additional_properties=(
                structured_output_disable_additional_properties
            ),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "stop_token_ids": self.stop_token_ids,
            "eos_token_id": self.eos_token_id,
            "ignore_eos": self.ignore_eos,
            "logit_bias": self.logit_bias,
            "allowed_token_ids": self.allowed_token_ids,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "bad_words_token_ids": self.bad_words_token_ids,
            "structured_output_backend": self.structured_output_backend,
            "structured_output_type": self.structured_output_type,
            "structured_output_spec": self.structured_output_spec,
            "structured_output_disable_any_whitespace": (
                self.structured_output_disable_any_whitespace
            ),
            "structured_output_disable_additional_properties": (
                self.structured_output_disable_additional_properties
            ),
            "structured_output_max_rollback": self.structured_output_max_rollback,
        }


class PackedLogprobs(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    token_ids: list[int]
    logprobs: list[float]
    sampled_token_rank: int


class OpenSessionRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str
    prompt_token_ids: list[int]
    sampling_metadata: SamplingMetadata
    initial_version: int = 0


class OpenSessionResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str
    session_version: int = 0
    vocab_size: int | None = None
    prompt_logprobs: list[PackedLogprobs] = msgspec.field(default_factory=list)


class DraftProposal(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str
    proposal_id: int
    base_version: int
    accepted_prefix_len: int
    draft_token_ids: list[int]
    draft_token_probs: list[float]
    draft_stopped: bool = False


class VerificationResult(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str
    proposal_id: int
    base_version: int
    accepted_len: int
    accepted_token_ids: list[int]
    verifier_version: int
    bonus_token_id: int | None = None
    reject_pos: int | None = None
    target_probs_at_reject_pos: bytes | None = None
    accepted_logprobs: list[PackedLogprobs] = msgspec.field(default_factory=list)
    bonus_logprobs: PackedLogprobs | None = None


class CloseSessionRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str


class ResyncSessionRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str
    accepted_prefix_token_ids: list[int]
    edge_version: int
    prompt_len: int
    sampling_metadata: SamplingMetadata | None = None


class ResyncSessionResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str
    session_version: int


MSGPACK_ENCODER = msgspec.msgpack.Encoder()
