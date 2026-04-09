from __future__ import annotations

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from .ops import find_accepted_len
from .types import VerifierRoundRequest, VerifierSamplerRoundResult


class DSSDVerifierSampler:
    def __init__(self, sampler: Sampler, num_speculative_steps: int) -> None:
        self.sampler = sampler
        self.num_speculative_steps = num_speculative_steps

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        request: VerifierRoundRequest,
    ) -> tuple[SamplerOutput, VerifierSamplerRoundResult]:
        if input_batch.num_reqs != 1:
            raise NotImplementedError(
                "DSSDVerifierSampler currently only supports a single request"
            )

        request.validate(gamma=self.num_speculative_steps)
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            input_ids,
            input_batch.expanded_local_pos,
        )

        device = processed_logits.device
        draft_token_ids = torch.tensor(
            request.draft_token_ids,
            device=device,
            dtype=torch.int64,
        )
        draft_q_values = torch.tensor(
            request.draft_q_values,
            device=device,
            dtype=processed_logits.dtype,
        )
        accepted_len_gpu = find_accepted_len(
            processed_logits=processed_logits,
            draft_token_ids=draft_token_ids,
            draft_q_values=draft_q_values,
            expanded_idx_mapping=input_batch.expanded_idx_mapping,
            seeds=self.sampler.sampling_states.seeds.gpu,
            positions=pos,
        )
        accepted_len_gpu = accepted_len_gpu.to(dtype=torch.int32)

        sampled = torch.full(
            (1, self.num_speculative_steps + 1),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        draft_len = len(request.draft_token_ids)
        padded_draft = torch.full(
            (self.num_speculative_steps,),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        if draft_len:
            padded_draft[:draft_len] = draft_token_ids
        draft_steps = torch.arange(
            self.num_speculative_steps,
            device=device,
            dtype=accepted_len_gpu.dtype,
        )
        sampled[0, : self.num_speculative_steps] = torch.where(
            draft_steps < accepted_len_gpu,
            padded_draft,
            torch.full_like(padded_draft, -1),
        )

        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=accepted_len_gpu.view(1),
        )

        all_accepted = accepted_len_gpu == torch.tensor(
            draft_len,
            device=device,
            dtype=accepted_len_gpu.dtype,
        )
        bonus_idx = draft_len
        bonus_token = gumbel_sample(
            processed_logits[bonus_idx : bonus_idx + 1],
            input_batch.expanded_idx_mapping[bonus_idx : bonus_idx + 1],
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            pos[bonus_idx : bonus_idx + 1],
            apply_temperature=False,
        )
        rejected_target_logits = processed_logits.index_select(
            0,
            accepted_len_gpu.view(1).to(dtype=torch.int64),
        ).squeeze(0)
        return sampler_output, VerifierSamplerRoundResult(
            req_id=request.req_id,
            accepted_len=accepted_len_gpu,
            all_accepted=all_accepted,
            bonus_token_id=bonus_token[0],
            rejected_target_logits=rejected_target_logits.detach().clone(),
        )
