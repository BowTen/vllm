# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer

from vllm.v1.spec_decode.distributed.protocol import SamplingMetadata
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    LMFormatEnforcerBackend,
)
from vllm.v1.structured_output.backend_outlines import OutlinesBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


@dataclass
class StructuredOutputSession:
    request_id: str
    grammar: StructuredOutputGrammar
    bitmask: torch.Tensor


class StructuredOutputFactory:
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool,
        *,
        num_speculative_tokens: int = 0,
        vocab_size: int | None = None,
        tokenizer: Any | None = None,
        backend_classes: dict[str, type[Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.num_speculative_tokens = num_speculative_tokens
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.vocab_size = int(vocab_size or len(self.tokenizer))
        self._backend_classes = backend_classes or {
            "xgrammar": XgrammarBackend,
            "guidance": GuidanceBackend,
            "outlines": OutlinesBackend,
            "lm-format-enforcer": LMFormatEnforcerBackend,
        }
        self._backends: dict[tuple[object, ...], object] = {}

    def resolve_sampling_params(
        self,
        params: "SamplingParams",
        structured_outputs_config: Any,
    ) -> None:
        if (
            params.structured_outputs is None
            or params.structured_outputs.all_constraints_none()
        ):
            return
        if structured_outputs_config is None:
            raise ValueError(
                "Distributed draft-model speculative decoding requires "
                "structured_outputs_config to validate structured outputs."
            )
        params._validate_structured_outputs(
            structured_outputs_config,
            tokenizer=self.tokenizer,
        )

    def create_session(
        self,
        request_id: str,
        sampling: SamplingMetadata,
    ) -> StructuredOutputSession | None:
        if (
            sampling.structured_output_backend is None
            or sampling.structured_output_type is None
            or sampling.structured_output_spec is None
        ):
            return None
        backend = self._get_backend(sampling)
        grammar = backend.compile_grammar(
            StructuredOutputOptions[sampling.structured_output_type],
            sampling.structured_output_spec,
        )
        bitmask = backend.allocate_token_bitmask(1)
        return StructuredOutputSession(
            request_id=request_id,
            grammar=grammar,
            bitmask=bitmask,
        )

    def close(self) -> None:
        for backend in self._backends.values():
            backend.destroy()
        self._backends.clear()

    def _get_backend(self, sampling: SamplingMetadata):
        backend_name = sampling.structured_output_backend
        assert backend_name is not None
        max_rollback = self._get_backend_max_rollback(
            backend_name,
            sampling.structured_output_max_rollback,
        )
        key = (
            backend_name,
            sampling.structured_output_disable_any_whitespace,
            sampling.structured_output_disable_additional_properties,
            max_rollback,
        )
        backend = self._backends.get(key)
        if backend is not None:
            return backend

        vllm_config = SimpleNamespace(
            structured_outputs_config=SimpleNamespace(
                disable_any_whitespace=(
                    sampling.structured_output_disable_any_whitespace
                ),
                disable_additional_properties=(
                    sampling.structured_output_disable_additional_properties
                ),
            ),
            speculative_config=SimpleNamespace(
                num_speculative_tokens=max_rollback,
            ),
        )
        backend_cls = self._backend_classes[backend_name]
        backend = backend_cls(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
        )
        self._backends[key] = backend
        return backend

    def _get_backend_max_rollback(
        self,
        backend_name: str,
        requested_rollback: int,
    ) -> int:
        if backend_name in ("xgrammar", "outlines"):
            return max(requested_rollback, self.num_speculative_tokens)
        return 0


def apply_structured_output_mask(
    logits: torch.Tensor,
    session: StructuredOutputSession | None,
) -> torch.Tensor:
    if session is None:
        return logits
    session.bitmask.fill_(-1)
    session.grammar.fill_bitmask(session.bitmask, 0)
    packed = session.bitmask[0].to(device=logits.device, non_blocking=True)
    token_indices = torch.arange(logits.shape[0], device=logits.device)
    word_indices = torch.div(token_indices, 32, rounding_mode="floor")
    bit_indices = token_indices % 32
    allowed = (
        (packed[word_indices].to(dtype=torch.int64) >> bit_indices) & 1
    ).bool()
    masked_logits = logits.to(dtype=torch.float32).clone()
    masked_logits[~allowed] = float("-inf")
    return masked_logits


def accept_structured_output_tokens(
    session: StructuredOutputSession | None,
    token_ids: list[int],
) -> None:
    if session is None or not token_ids:
        return
    if not session.grammar.accept_tokens(session.request_id, token_ids):
        raise ValueError(
            f"Structured output grammar rejected committed tokens for session "
            f"{session.request_id}: {token_ids!r}"
        )


def rollback_structured_output_tokens(
    session: StructuredOutputSession | None,
    num_tokens: int,
) -> None:
    if session is None or num_tokens <= 0:
        return
    session.grammar.rollback(num_tokens)


def reset_structured_output_session(
    session: StructuredOutputSession | None,
    output_token_ids: list[int],
) -> None:
    if session is None:
        return
    session.grammar.reset()
    accept_structured_output_tokens(session, output_token_ids)
