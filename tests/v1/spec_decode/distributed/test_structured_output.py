# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.spec_decode.distributed.protocol import SamplingMetadata
from vllm.v1.spec_decode.distributed.structured_output import (
    StructuredOutputFactory,
    StructuredOutputSession,
    accept_structured_output_tokens,
    apply_structured_output_mask,
    reset_structured_output_session,
    rollback_structured_output_tokens,
)


class FakeTokenizer:
    def __len__(self) -> int:
        return 64


class FakeGrammar:
    def __init__(self) -> None:
        self.accepted: list[int] = []
        self.rollback_calls: list[int] = []
        self.reset_calls = 0

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        del request_id
        for token in tokens:
            if token not in self._allowed_tokens():
                return False
            self.accepted.append(token)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        accepted: list[int] = []
        for token in tokens:
            if token not in self._allowed_tokens(prefix=accepted):
                break
            accepted.append(token)
        return accepted

    def rollback(self, num_tokens: int) -> None:
        self.rollback_calls.append(num_tokens)
        if num_tokens > 0:
            del self.accepted[-num_tokens:]

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        bitmask[batch_index].zero_()
        for token in self._allowed_tokens():
            word_index = token // 32
            bit_index = token % 32
            bitmask[batch_index, word_index] |= 1 << bit_index

    def is_terminated(self) -> bool:
        return False

    def reset(self) -> None:
        self.reset_calls += 1
        self.accepted = []

    def _allowed_tokens(self, prefix: list[int] | None = None) -> list[int]:
        current = self.accepted if prefix is None else self.accepted + prefix
        if not current:
            return [1, 3]
        return [2]


class FakeBackend:
    def __init__(self, vllm_config, tokenizer, vocab_size) -> None:
        self.vllm_config = vllm_config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.grammars: list[FakeGrammar] = []

    def compile_grammar(self, request_type, grammar_spec: str) -> FakeGrammar:
        del request_type, grammar_spec
        grammar = FakeGrammar()
        self.grammars.append(grammar)
        return grammar

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.zeros((max_num_seqs, 2), dtype=torch.int32)

    def destroy(self) -> None:
        return None


def make_sampling_metadata() -> SamplingMetadata:
    return SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        structured_output_backend="outlines",
        structured_output_type="CHOICE",
        structured_output_spec='["yes","no"]',
        structured_output_max_rollback=3,
    )


def test_structured_output_helpers_mask_accept_rollback_and_reset() -> None:
    factory = StructuredOutputFactory(
        model_name="fake-model",
        trust_remote_code=False,
        num_speculative_tokens=3,
        tokenizer=FakeTokenizer(),
        backend_classes={"outlines": FakeBackend},
    )
    session = factory.create_session("request-1", make_sampling_metadata())
    assert session is not None
    assert isinstance(session.grammar, FakeGrammar)

    logits = torch.arange(64, dtype=torch.float32)
    masked_logits = apply_structured_output_mask(logits, session)
    assert torch.isfinite(masked_logits[1])
    assert torch.isfinite(masked_logits[3])
    assert masked_logits[2].item() == float("-inf")
    assert masked_logits[4].item() == float("-inf")

    accept_structured_output_tokens(session, [1])
    masked_logits = apply_structured_output_mask(logits, session)
    assert torch.isfinite(masked_logits[2])
    assert masked_logits[1].item() == float("-inf")

    rollback_structured_output_tokens(session, 1)
    assert session.grammar.rollback_calls == [1]
    assert session.grammar.accepted == []

    reset_structured_output_session(session, [3])
    assert session.grammar.reset_calls == 1
    assert session.grammar.accepted == [3]
    factory.close()


def test_structured_output_factory_resolves_sampling_backend() -> None:
    params = SamplingParams(
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(regex="yes|no"),
    )
    factory = StructuredOutputFactory(
        model_name="fake-model",
        trust_remote_code=False,
        tokenizer=FakeTokenizer(),
        backend_classes={"outlines": FakeBackend},
    )

    factory.resolve_sampling_params(
        params,
        SimpleNamespace(
            backend="outlines",
            disable_any_whitespace=False,
            disable_additional_properties=False,
        ),
    )

    assert params.structured_outputs is not None
    assert params.structured_outputs._backend == "outlines"


def test_structured_output_factory_uses_backend_specific_rollback_limits() -> None:
    factory = StructuredOutputFactory(
        model_name="fake-model",
        trust_remote_code=False,
        num_speculative_tokens=4,
        tokenizer=FakeTokenizer(),
        backend_classes={"outlines": FakeBackend},
    )

    assert factory._get_backend_max_rollback("outlines", 2) == 4
    assert factory._get_backend_max_rollback("xgrammar", 1) == 4
    assert factory._get_backend_max_rollback("guidance", 7) == 0
    assert factory._get_backend_max_rollback("lm-format-enforcer", 7) == 0


def test_structured_output_factory_can_use_runtime_vocab_size_override() -> None:
    factory = StructuredOutputFactory(
        model_name="fake-model",
        trust_remote_code=False,
        vocab_size=96,
        tokenizer=FakeTokenizer(),
        backend_classes={"outlines": FakeBackend},
    )

    assert factory.vocab_size == 96


def test_accept_structured_output_tokens_rejects_invalid_commit() -> None:
    session = StructuredOutputSession(
        request_id="request-invalid",
        grammar=FakeGrammar(),
        bitmask=torch.zeros((1, 2), dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="Structured output grammar rejected"):
        accept_structured_output_tokens(session, [7])
