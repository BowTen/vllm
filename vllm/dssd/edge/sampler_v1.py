from __future__ import annotations

import torch


class DSSDEdgeDraftSamplerV1:
    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def sample_step(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        processed_logits_dst: torch.Tensor,
    ) -> tuple[int, float]:
        processed_logits = processed_logits_dst
        processed_logits.copy_(logits.to(dtype=torch.float32))
        processed_logits = self.sampler.apply_logits_processors(
            processed_logits,
            sampling_metadata,
        )
        if processed_logits.data_ptr() != processed_logits_dst.data_ptr():
            processed_logits_dst.copy_(processed_logits)
            processed_logits = processed_logits_dst
        sampled = self.sampler.sample(processed_logits, sampling_metadata)
        sampled_token_ids = sampled[0] if isinstance(sampled, tuple) else sampled
        sampled_token_id = int(sampled_token_ids.reshape(-1)[0].item())
        q_value = float(
            torch.softmax(processed_logits[0], dim=-1)[sampled_token_id].item()
        )
        return sampled_token_id, q_value
