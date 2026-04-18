from __future__ import annotations

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm.v1.sample.sampler import _SAMPLING_EPS


class DSSDEdgeDraftSamplerV1:
    """Single-step draft sampler for the old v1 GPU model runner path."""

    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def sample_step(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        processed_logits_dst: torch.Tensor,
    ) -> tuple[int, float]:
        processed_logits = processed_logits_dst.view_as(logits)
        processed_logits.copy_(logits)
        processed_logits = self._copy_back_if_replaced(
            processed_logits_dst,
            logits,
            self.sampler.apply_logits_processors(
                processed_logits,
                sampling_metadata,
                False,
            ),
        )

        sampled, processed_logits = self._sample_with_final_logits(
            processed_logits,
            sampling_metadata,
        )
        token_id = int(sampled.reshape(-1)[0].item())
        q_value = self.extract_q_value(processed_logits[0], token_id)
        return token_id, q_value

    def _sample_with_final_logits(
        self,
        processed_logits: torch.Tensor,
        sampling_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)

        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.sampler.greedy_sample(processed_logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled, processed_logits

        assert sampling_metadata.temperature is not None
        processed_logits = self.sampler.apply_temperature(
            processed_logits,
            sampling_metadata.temperature,
            sampling_metadata.all_random,
        )

        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            processed_logits = self._copy_back_if_replaced(
                processed_logits,
                processed_logits,
                processor.apply(processed_logits),
            )

        processed_logits = self._copy_back_if_replaced(
            processed_logits,
            processed_logits,
            apply_top_k_top_p(
                processed_logits,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
            ),
        )

        probs = processed_logits.softmax(dim=-1, dtype=torch.float32)
        random_sampled = random_sample(probs, sampling_metadata.generators)
        if greedy_sampled is None:
            return random_sampled, processed_logits

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        return sampled, processed_logits

    @staticmethod
    def _copy_back_if_replaced(
        processed_logits_dst: torch.Tensor,
        logits_shape_ref: torch.Tensor,
        maybe_replaced_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        processed_logits = processed_logits_dst.view_as(logits_shape_ref)
        if maybe_replaced_logits is None:
            return processed_logits
        if maybe_replaced_logits.data_ptr() != processed_logits.data_ptr():
            processed_logits.copy_(maybe_replaced_logits)
        return processed_logits

    def extract_q_value(
        self,
        processed_logits_row: torch.Tensor,
        sampled_token_id: int,
    ) -> float:
        return float(
            torch.softmax(processed_logits_row, dim=-1)[sampled_token_id].item()
        )
