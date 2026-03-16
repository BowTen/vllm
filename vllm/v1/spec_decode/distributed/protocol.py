# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import msgspec

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


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

    @classmethod
    def from_sampling_params(cls, params: "SamplingParams") -> "SamplingMetadata":
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
        }


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


class ResyncSessionResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    session_id: str
    session_version: int


MSGPACK_ENCODER = msgspec.msgpack.Encoder()
