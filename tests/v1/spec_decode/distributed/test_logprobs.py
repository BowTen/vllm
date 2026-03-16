# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode.distributed.logprobs import (
    build_logprobs_lists,
    build_logprobs_tensors,
    dense_probs_from_packed_logprobs,
    pack_logprobs_lists,
    pack_logprobs_tensors,
    pack_sample_logprobs,
)


def test_pack_sample_logprobs_preserves_sampled_rank_and_top_tokens():
    probs = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float32)

    packed = pack_sample_logprobs(probs, sampled_token_id=2, num_logprobs=2)

    assert packed is not None
    assert packed.token_ids == [2, 1, 2]
    assert packed.sampled_token_rank == 2
    assert packed.logprobs[0] == pytest.approx(np.log(0.2), rel=1e-5)


def test_build_logprobs_lists_converts_packed_entries():
    first = pack_sample_logprobs(
        torch.tensor([0.6, 0.4], dtype=torch.float32),
        sampled_token_id=0,
        num_logprobs=1,
    )
    second = pack_sample_logprobs(
        torch.tensor([0.3, 0.7], dtype=torch.float32),
        sampled_token_id=1,
        num_logprobs=1,
    )
    assert first is not None
    assert second is not None

    logprobs = build_logprobs_lists([first, second])

    assert logprobs is not None
    assert logprobs.logprob_token_ids.tolist() == [[0, 0], [1, 1]]
    assert logprobs.sampled_token_ranks.tolist() == [1, 1]


def test_pack_sample_logprobs_supports_all_vocab_mode():
    probs = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float32)

    packed = pack_sample_logprobs(probs, sampled_token_id=2, num_logprobs=-1)

    assert packed is not None
    assert len(packed.token_ids) == 4
    assert set(packed.token_ids[1:]) == {0, 1, 2}


def test_build_logprobs_tensors_converts_packed_entries():
    first = pack_sample_logprobs(
        torch.tensor([0.6, 0.4], dtype=torch.float32),
        sampled_token_id=0,
        num_logprobs=1,
    )
    second = pack_sample_logprobs(
        torch.tensor([0.3, 0.7], dtype=torch.float32),
        sampled_token_id=1,
        num_logprobs=1,
    )
    assert first is not None
    assert second is not None

    logprobs = build_logprobs_tensors([first, second])

    assert logprobs is not None
    assert logprobs.logprob_token_ids.tolist() == [[0, 0], [1, 1]]
    assert logprobs.selected_token_ranks.tolist() == [1, 1]


def test_pack_logprobs_lists_moves_sampled_token_to_front():
    logprobs = build_logprobs_lists(
        [
            pack_sample_logprobs(
                torch.tensor([0.1, 0.7, 0.2], dtype=torch.float32),
                sampled_token_id=2,
                num_logprobs=2,
            )
        ]
    )
    assert logprobs is not None
    packed = pack_logprobs_lists(
        logprobs,
        [2],
    )

    assert len(packed) == 1
    assert packed[0].token_ids[0] == 2
    assert packed[0].sampled_token_rank == 2


def test_pack_logprobs_tensors_round_trips_dense_distribution():
    entry = pack_sample_logprobs(
        torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32),
        sampled_token_id=1,
        num_logprobs=-1,
    )
    assert entry is not None
    tensors = build_logprobs_tensors([entry])
    assert tensors is not None

    packed = pack_logprobs_tensors(tensors, [1])
    probs = dense_probs_from_packed_logprobs(packed[0], vocab_size=3)

    assert probs.tolist() == pytest.approx([0.2, 0.3, 0.5], rel=1e-5)
