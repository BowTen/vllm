from __future__ import annotations

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.sampler import Sampler


class DSSDEdgeDraftSampler:
    """edge 端的一步 draft sampler。"""

    def __init__(self, sampler: Sampler) -> None:
        self.sampler = sampler

    def sample_step(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        processed_logits_dst: torch.Tensor,
    ) -> tuple[int, float]:
        processed_logits = self.apply_sampling_params_into(
            logits,
            input_batch,
            processed_logits_dst,
        )
        sampled_token_id = self.sample_token(processed_logits, input_batch)
        q_value = self.extract_q_value(processed_logits[0], sampled_token_id)
        return sampled_token_id, q_value

    def apply_sampling_params_into(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        processed_logits_dst: torch.Tensor,
    ) -> torch.Tensor:
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        processed_logits = processed_logits_dst.view_as(logits)
        processed_logits.copy_(logits)

        self.sampler.logit_bias_state.apply_logit_bias(
            processed_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
        )
        self.sampler.penalties_state.apply_penalties(
            processed_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            input_ids,
            input_batch.expanded_local_pos,
            self.sampler.num_speculative_tokens,
        )
        self.sampler.bad_words_state.apply_bad_words(
            processed_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            input_ids,
            input_batch.expanded_local_pos,
        )
        self.sampler.sampling_states.apply_temperature(
            processed_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
        )
        self.sampler.sampling_states.apply_min_p(
            processed_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
        )

        final_logits = self.sampler.sampling_states.apply_top_k_top_p(
            processed_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
        )
        if final_logits.data_ptr() != processed_logits.data_ptr():
            processed_logits.copy_(final_logits)
            final_logits = processed_logits
        return final_logits

    def sample_token(
        self,
        processed_logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> int:
        pos = input_batch.positions[input_batch.logits_indices][:1]
        req_idx = input_batch.expanded_idx_mapping[:1]
        sampled = gumbel_sample(
            processed_logits[:1],
            req_idx,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )
        return int(sampled.item())

    def extract_q_value(
        self,
        processed_logits_row: torch.Tensor,
        sampled_token_id: int,
    ) -> float:
        log_q = (
            processed_logits_row[sampled_token_id]
            - torch.logsumexp(processed_logits_row, dim=-1)
        )
        return float(torch.exp(log_q).item())

    def build_sampled_tokens(
        self,
        token_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor([[token_id]], dtype=torch.int64, device=device)

    def build_num_sampled(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([1], dtype=torch.int32, device=device)

    def build_num_rejected(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.int32, device=device)
