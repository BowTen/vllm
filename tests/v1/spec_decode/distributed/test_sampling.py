# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.distributed.protocol import SamplingMetadata
from vllm.v1.spec_decode.distributed.sampling import (
    is_terminal_token,
    sample_from_logits,
    sample_from_probs,
)


def test_sample_from_logits_respects_stop_suppression():
    logits = torch.tensor([0.0, 5.0, 1.0], dtype=torch.float32)
    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        eos_token_id=1,
    )

    result = sample_from_logits(
        logits,
        sampling,
        prompt_token_ids=[7],
        output_token_ids=[],
        generator=torch.Generator(device="cpu").manual_seed(1),
        suppress_stops=True,
    )

    assert result.token_id == 2


def test_sample_from_logits_applies_presence_penalty():
    logits = torch.tensor([0.0, 4.0, 3.0], dtype=torch.float32)
    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=2.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )

    result = sample_from_logits(
        logits,
        sampling,
        prompt_token_ids=[0],
        output_token_ids=[1],
        generator=torch.Generator(device="cpu").manual_seed(1),
        suppress_stops=False,
    )

    assert result.token_id == 2


def test_sample_from_probs_normalizes():
    probs = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
    token_id = sample_from_probs(
        probs, torch.Generator(device="cpu").manual_seed(123)
    )
    assert token_id == 0


def test_sample_from_probs_supports_greedy_mode():
    probs = torch.tensor([0.2, 0.7, 0.1], dtype=torch.float32)
    token_id = sample_from_probs(
        probs,
        torch.Generator(device="cpu").manual_seed(123),
        greedy=True,
    )
    assert token_id == 1


def test_is_terminal_token_requires_min_tokens():
    sampling = SamplingMetadata(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        eos_token_id=2,
        min_tokens=2,
    )

    assert not is_terminal_token(2, sampling, output_len_after=1)
    assert is_terminal_token(2, sampling, output_len_after=2)


def test_sample_from_logits_respects_single_token_bad_words():
    logits = torch.tensor([1.0, 5.0, 4.0], dtype=torch.float32)
    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        bad_words_token_ids=[[1]],
    )

    result = sample_from_logits(
        logits,
        sampling,
        prompt_token_ids=[7],
        output_token_ids=[],
        generator=torch.Generator(device="cpu").manual_seed(1),
        suppress_stops=False,
    )

    assert result.token_id == 2


def test_sample_from_logits_respects_multi_token_bad_words_prefix():
    logits = torch.tensor([1.0, 4.0, 5.0], dtype=torch.float32)
    sampling = SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        bad_words_token_ids=[[5, 2]],
    )

    result = sample_from_logits(
        logits,
        sampling,
        prompt_token_ids=[7],
        output_token_ids=[5],
        generator=torch.Generator(device="cpu").manual_seed(1),
        suppress_stops=False,
    )

    assert result.token_id == 1
