from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _accepted_prefix_length_kernel(
    accepted_len_ptr,
    target_probs_ptr,
    draft_q_values_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    NUM_STEPS: tl.constexpr,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr)
    seed = tl.load(seeds_ptr + req_state_idx)

    accepted_len = 0
    accepting = True
    for step in range(NUM_STEPS):
        if accepting:
            target_prob = tl.load(target_probs_ptr + step).to(tl.float32)
            draft_q = tl.load(draft_q_values_ptr + step).to(tl.float32)
            pos = tl.load(pos_ptr + step)
            u = tl.sum(tl.rand(seed, pos + tl.arange(0, 1)))
            accepting = target_prob > u * draft_q
            accepted_len += accepting

    tl.store(accepted_len_ptr, accepted_len)


def find_accepted_len(
    *,
    processed_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_q_values: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    num_steps = draft_token_ids.numel()
    accepted_len = torch.zeros((), device=processed_logits.device, dtype=torch.int32)
    if num_steps == 0:
        return accepted_len

    step_logits = processed_logits[:num_steps]
    target_probs = torch.softmax(step_logits, dim=-1)
    target_probs = target_probs.gather(1, draft_token_ids.view(-1, 1)).squeeze(1)
    draft_q_values = draft_q_values.to(
        device=processed_logits.device,
        dtype=target_probs.dtype,
    ).clamp_min_(torch.finfo(target_probs.dtype).tiny)

    _accepted_prefix_length_kernel[(1,)](
        accepted_len,
        target_probs.contiguous(),
        draft_q_values.contiguous(),
        expanded_idx_mapping[:1].contiguous(),
        seeds,
        positions[:num_steps].contiguous(),
        NUM_STEPS=num_steps,
    )
    return accepted_len
