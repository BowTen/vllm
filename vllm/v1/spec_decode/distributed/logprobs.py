# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch

from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.spec_decode.distributed.protocol import PackedLogprobs


def pack_sample_logprobs(
    probs: torch.Tensor,
    sampled_token_id: int,
    num_logprobs: int | None,
) -> PackedLogprobs | None:
    if num_logprobs is None:
        return None

    probs = probs.to(dtype=torch.float32)
    total = float(probs.sum().item())
    if total <= 0:
        raise ValueError("Cannot pack logprobs from an empty probability distribution.")
    if abs(total - 1.0) > 1e-4:
        probs = probs / total

    log_probs = torch.log(probs.clamp_min(torch.finfo(torch.float32).tiny))
    sampled_logprob = float(log_probs[sampled_token_id].item())
    sampled_token_rank = (
        int(torch.count_nonzero(log_probs > sampled_logprob).item()) + 1
    )

    if num_logprobs == -1:
        top_k = probs.numel()
    else:
        top_k = min(num_logprobs, probs.numel())
    top_logprobs, top_token_ids = torch.topk(log_probs, k=top_k)
    return PackedLogprobs(
        token_ids=[sampled_token_id] + top_token_ids.tolist(),
        logprobs=[sampled_logprob] + top_logprobs.tolist(),
        sampled_token_rank=sampled_token_rank,
    )


def build_logprobs_lists(
    entries: list[PackedLogprobs],
) -> LogprobsLists | None:
    if not entries:
        return None
    return LogprobsLists(
        logprob_token_ids=np.asarray(
            [entry.token_ids for entry in entries], dtype=np.int32
        ),
        logprobs=np.asarray([entry.logprobs for entry in entries], dtype=np.float32),
        sampled_token_ranks=np.asarray(
            [entry.sampled_token_rank for entry in entries],
            dtype=np.int32,
        ),
    )


def build_logprobs_tensors(
    entries: list[PackedLogprobs],
) -> LogprobsTensors | None:
    if not entries:
        return None
    return LogprobsTensors(
        logprob_token_ids=torch.tensor(
            [entry.token_ids for entry in entries],
            dtype=torch.int32,
            device="cpu",
        ),
        logprobs=torch.tensor(
            [entry.logprobs for entry in entries],
            dtype=torch.float32,
            device="cpu",
        ),
        selected_token_ranks=torch.tensor(
            [entry.sampled_token_rank for entry in entries],
            dtype=torch.int32,
            device="cpu",
        ),
    )
